"""
Ballistic fitting and the impact solve. Pure math -- no camera, no arm.
"""
import numpy as np
import pytest

from perception.trajectory import (BALL_RADIUS, G_BASE, Z_FLOOR_BASE,
                                   ballistic_position, fit_two_points,
                                   solve_impact, solve_impact_time)


def test_gravity_and_geometry_constants():
    assert np.allclose(G_BASE, [0.0, 0.0, -9.81])
    assert Z_FLOOR_BASE == -0.433, "base frame: base at 0, floor at -base_height"
    assert BALL_RADIUS == 0.0327


def test_position_at_a_hand_computed_time():
    p0 = np.array([0.0, 0.0, 1.0])
    v0 = np.array([2.0, 0.0, 0.0])
    p = ballistic_position(p0, v0, 0.45152)
    assert p[0] == pytest.approx(0.90304, abs=1e-5)
    assert p[2] == pytest.approx(0.0, abs=1e-4)


def test_impact_time_matches_the_closed_form():
    """Drop from z=1.0 with no vertical velocity: t = sqrt(2*1.0/9.81)."""
    t = solve_impact_time(np.array([0.0, 0.0, 1.0]), np.zeros(3), 0.0)
    assert t == pytest.approx(np.sqrt(2.0 / 9.81), rel=1e-12)


def test_impact_takes_the_DESCENDING_root_not_the_first_crossing():
    """
    Rising through the target plane then falling back gives two positive roots.
    Taking the smaller one reports a 'landing' while the ball is still going UP.
    """
    p0 = np.array([0.0, 0.0, 1.0])
    v0 = np.array([1.0, 0.0, 3.0])
    t = solve_impact_time(p0, v0, 1.3)
    assert t == pytest.approx(0.48568, abs=1e-4)
    assert ballistic_position(p0, v0, t)[2] == pytest.approx(1.3, abs=1e-9)
    assert (v0[2] + G_BASE[2] * t) < 0, "must be descending at impact"


def test_no_real_crossing_raises_rather_than_returning_nonsense():
    with pytest.raises(RuntimeError, match="never reaches"):
        solve_impact_time(np.array([0.0, 0.0, 1.0]), np.zeros(3), 1.5)


def test_impact_offsets_by_one_ball_radius():
    """
    A resting sphere's CENTRE is one radius above the floor, and for a sphere
    the centre's (x, y) IS the contact (x, y). This is the same convention as
    ray_plane.ball_center_on_plane -- if the two disagree, the cross-check in
    tests/test_landing_pipeline.py measures nothing.
    """
    p0 = np.array([0.0, 0.0, 1.0])
    v0 = np.array([2.0, 0.0, 0.0])
    x, y, t = solve_impact(p0, v0, z_floor=0.0, ball_radius=BALL_RADIUS)
    assert ballistic_position(p0, v0, t)[2] == pytest.approx(BALL_RADIUS, abs=1e-9)
    x_naive, _, _ = solve_impact(p0, v0, z_floor=0.0, ball_radius=0.0)
    assert x < x_naive, "the radius offset must shorten the flight, not lengthen it"


def test_two_point_fit_recovers_the_generating_trajectory():
    p0 = np.array([0.035, 0.0, 1.137])
    v0 = np.array([1.6218, 0.0, 0.1419])
    ta, tb = 0.12, 0.47
    got_p0, got_v0 = fit_two_points(ta, ballistic_position(p0, v0, ta),
                                    tb, ballistic_position(p0, v0, tb))
    assert np.allclose(got_p0, p0, atol=1e-12)
    assert np.allclose(got_v0, v0, atol=1e-12)


def test_two_point_fit_refuses_a_degenerate_sample():
    p = np.array([0.0, 0.0, 1.0])
    with pytest.raises(RuntimeError, match="separated"):
        fit_two_points(0.20, p, 0.2001, p)


from perception.ray_plane import D435I_IR_848x480
from perception.stereo import D435I_IR_BASELINE_M, StereoRig
from perception.trajectory import FitResult, fit_ballistic

RIG = StereoRig(D435I_IR_848x480, D435I_IR_BASELINE_M)

# Overhead mount, camera at base-frame (0.82, 0, 1.767) looking straight down.
# Same R convention as tests/test_ray_plane.py::_overhead.
R_BC = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
T_BC = np.array([0.82, 0.0, 1.767])

TRUE_P0 = np.array([0.035, 0.0, 1.137])
TRUE_V0 = np.array([1.6218, 0.0, 0.1419])


def _synth_obs(times, p0=TRUE_P0, v0=TRUE_V0, noise_px=0.0, seed=0):
    """Project a known base-frame parabola into both IR images."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in times:
        p_b = ballistic_position(p0, v0, t)
        p_c = R_BC.T @ (p_b - T_BC)
        u1, v1, u2, v2 = RIG.project(p_c)
        rows.append([t, u1, v1, u2, v2])
    obs = np.asarray(rows, float)
    if noise_px:
        obs[:, 1:] += rng.normal(0.0, noise_px, size=obs[:, 1:].shape)
    return obs


def test_noiseless_fit_recovers_the_trajectory_exactly():
    obs = _synth_obs(np.linspace(0.13, 0.55, 40))
    fit = fit_ballistic(obs, RIG, R_BC, T_BC)
    assert isinstance(fit, FitResult)
    assert np.allclose(fit.p0, TRUE_P0, atol=1e-6)
    assert np.allclose(fit.v0, TRUE_V0, atol=1e-6)
    assert fit.rms_px < 1e-6
    assert fit.n_obs == 40


def test_landing_error_under_realistic_pixel_noise_beats_the_budget():
    """
    Spec's error budget claims ~4.4 mm total sigma at 0.15 px centroid noise.
    Assert the realised spread over 30 trials is under 1 cm -- comfortably
    inside the budget, but loose enough not to be a flaky test.
    """
    truth = solve_impact(TRUE_P0, TRUE_V0, z_floor=Z_FLOOR_BASE)
    errs = []
    for seed in range(30):
        obs = _synth_obs(np.linspace(0.13, 0.55, 40), noise_px=0.15, seed=seed)
        fit = fit_ballistic(obs, RIG, R_BC, T_BC)
        x, y, _ = solve_impact(fit.p0, fit.v0, z_floor=Z_FLOOR_BASE)
        errs.append(np.hypot(x - truth[0], y - truth[1]))
    assert np.mean(errs) < 0.010, f"mean landing error {np.mean(errs) * 1e3:.1f} mm"


def test_fit_reports_a_usable_covariance():
    obs = _synth_obs(np.linspace(0.13, 0.55, 40), noise_px=0.15, seed=7)
    fit = fit_ballistic(obs, RIG, R_BC, T_BC)
    assert fit.cov.shape == (6, 6)
    assert np.all(np.diag(fit.cov) > 0)
    assert np.allclose(fit.cov, fit.cov.T, atol=1e-12)


def test_fit_refuses_too_few_observations():
    """Spec section 6: fewer than 12 usable frames is a refusal, not a guess."""
    obs = _synth_obs(np.linspace(0.13, 0.55, 8))
    with pytest.raises(RuntimeError, match="12"):
        fit_ballistic(obs, RIG, R_BC, T_BC)


from perception.trajectory import ransac_track


def _arm_like_outliers(n, seed=3):
    """
    Rows that look like detections but do not lie on ANY g=9.81 parabola --
    what the moving arm, a reflection, or the second bounce produce.
    """
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.13, 0.55, size=n)
    u1 = rng.uniform(100, 700, size=n)
    v1 = rng.uniform(60, 420, size=n)
    return np.stack([t, u1, v1, u1 - rng.uniform(8, 20, size=n), v1], axis=-1)


def test_ransac_rejects_arm_like_outliers_and_recovers_the_ball():
    good = _synth_obs(np.linspace(0.13, 0.55, 40), noise_px=0.15, seed=1)
    obs = np.vstack([good, _arm_like_outliers(12)])
    idx, fit = ransac_track(obs, RIG, R_BC, T_BC)
    assert len(idx) >= 36, f"kept only {len(idx)} of 40 true inliers"
    assert set(idx.tolist()).issubset(set(range(40))), "an outlier was kept"
    assert np.allclose(fit.p0, TRUE_P0, atol=0.02)
    assert np.allclose(fit.v0, TRUE_V0, atol=0.05)


def test_ransac_refuses_when_the_inlier_fraction_is_too_low():
    """Spec section 6: below 0.6 inliers is a refusal, not a best effort."""
    good = _synth_obs(np.linspace(0.13, 0.55, 14), noise_px=0.15, seed=2)
    obs = np.vstack([good, _arm_like_outliers(40)])
    with pytest.raises(RuntimeError, match="inlier"):
        ransac_track(obs, RIG, R_BC, T_BC)


def test_ransac_is_deterministic_for_a_fixed_seed():
    obs = np.vstack([_synth_obs(np.linspace(0.13, 0.55, 40), noise_px=0.15, seed=1),
                     _arm_like_outliers(12)])
    a, _ = ransac_track(obs, RIG, R_BC, T_BC, seed=11)
    b, _ = ransac_track(obs, RIG, R_BC, T_BC, seed=11)
    assert np.array_equal(a, b)
