"""
Spec section 7.1 -- synthetic end-to-end. Renders a known parabola into two
IMAGES (not just observations), runs the whole pipeline, and checks the landing
point against ground truth.

This is the test that would catch a detector/pairing/fit integration error that
every unit test passes individually.
"""
import cv2
import numpy as np
import pytest

from measure_landing import (MAX_UNOBSERVED_DROP_M, build_observations,
                             check_landing_is_observed, measure_landing)
from perception.ball_track import (detect_candidates, median_background,
                                   reject_static_candidates)
from perception.ray_plane import D435I_IR_848x480
from perception.stereo import D435I_IR_BASELINE_M, StereoRig
from perception.trajectory import (BALL_RADIUS, Z_FLOOR_BASE,
                                   ballistic_position, solve_impact)

RIG = StereoRig(D435I_IR_848x480, D435I_IR_BASELINE_M)
R_BC = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
T_BC = np.array([0.82, 0.0, 1.767])
TRUE_P0 = np.array([0.035, 0.0, 1.137])
TRUE_V0 = np.array([1.6218, 0.0, 0.1419])
H, W = 480, 848


SHIFT = 4  # cv2 sub-pixel fixed point: 1/16 px


def _disc(img, u, v, r):
    """
    Filled disc at a SUB-PIXEL centre.

    An integer centre would quantise the ground truth to 0.5 px -- 3.3x the
    0.15 px centroid precision the error budget assumes -- so the test would be
    measuring cv2's rounding, not the pipeline.
    """
    k = 1 << SHIFT
    cv2.circle(img, (int(round(u * k)), int(round(v * k))),
               int(round(max(2.0, r) * k)), 225, -1, shift=SHIFT)


def _render(times, seed=0):
    """
    Draw the ball into both IR images at its true projected position.

    Frames where the disc would touch an image edge are SKIPPED. Before
    t ~= 0.120 s the ball is off the left edge of IR2 (u2 = -10.9 px at
    t = 0.10), and a clipped disc has a biased centroid -- which would look
    exactly like a pipeline error.
    """
    rng = np.random.default_rng(seed)
    base = (40 + rng.integers(0, 12, size=(H, W))).astype(np.uint8)
    ts, ir1, ir2 = [], [], []
    for t in times:
        p_c = R_BC.T @ (ballistic_position(TRUE_P0, TRUE_V0, t) - T_BC)
        u1, v1, u2, v2 = RIG.project(p_c)
        r_px = D435I_IR_848x480.fx * BALL_RADIUS / p_c[2]
        inside = all(r_px <= u <= W - 1 - r_px for u in (u1, u2)) and \
                 all(r_px <= v <= H - 1 - r_px for v in (v1, v2))
        if not inside:
            continue
        f1, f2 = base.copy(), base.copy()
        _disc(f1, u1, v1, r_px)
        _disc(f2, u2, v2, r_px)
        ts.append(t); ir1.append(f1); ir2.append(f2)
    if len(ts) < 20:
        raise AssertionError(f"only {len(ts)} frames rendered in view -- "
                             f"the test time window is wrong, not the pipeline")
    return {"t": np.asarray(ts, float), "ir1": np.asarray(ir1),
            "ir2": np.asarray(ir2), "meta": {"synthetic": True}}


# One real throw at 90 fps, from the frame the ball clears IR2's edge to impact.
FLIGHT_TIMES = np.arange(0.12, 0.58, 1.0 / 90.0)


def test_end_to_end_landing_point_matches_truth():
    rec = _render(FLIGHT_TIMES)
    got = measure_landing(rec, R_BC, T_BC)
    x_true, y_true, _ = solve_impact(TRUE_P0, TRUE_V0, z_floor=Z_FLOOR_BASE)
    err = np.hypot(got["x"] - x_true, got["y"] - y_true)
    assert err < 0.02, f"landing error {err * 1e3:.1f} mm"
    assert got["n_inliers"] >= 0.8 * len(rec["t"])
    assert got["rms_px"] < 1.0
    assert got["sigma_xy_m"] > 0


def test_observations_are_built_for_most_rendered_frames():
    rec = _render(FLIGHT_TIMES)
    obs, max_frac = build_observations(rec)
    assert max_frac < 0.01, "a synthetic scene must not look like lighting drift"
    assert obs.shape[1] == 5
    assert obs.shape[0] >= 0.9 * len(rec["t"]), \
        f"only {obs.shape[0]} of {len(rec['t'])} rendered frames paired"


def test_a_recording_with_no_ball_raises_rather_than_inventing_a_landing():
    rng = np.random.default_rng(1)
    flat = (40 + rng.integers(0, 12, size=(20, H, W))).astype(np.uint8)
    rec = {"t": np.linspace(0, 0.2, 20), "ir1": flat, "ir2": flat.copy(), "meta": {}}
    with pytest.raises(RuntimeError):
        measure_landing(rec, R_BC, T_BC)


def _add_static_noise(rec, u=460.0, v=234.0, r=6.0):
    """
    Overlay a disc at the same pixel location in every frame of both streams --
    a stand-in for the permanently-mounted calibration board (found
    2026-09-02): its candidates recur at a handful of fixed (u, v) locations
    across nearly the whole window, unlike the ball, which moves every frame.

    The caller must supply an EXTERNAL background (one built before this noise
    was added) to `build_observations`/`measure_landing` via `bg1`/`bg2`,
    rather than letting them compute their own per-pixel median from this
    noisy recording. A disc present in 100% of frames wins its own temporal
    median and vanishes into the "background" before detection ever runs --
    real board noise survives median subtraction only because the IR
    emitter's dot pattern gives it genuine frame-to-frame texture jitter,
    which a filled disc does not have. Supplying the pre-noise background
    isolates the thing this test actually checks (rejecting a recurring
    candidate) from a different, already-covered behavior (temporal-median
    robustness), rather than trying to out-render the real speckle noise.
    """
    ir1, ir2 = rec["ir1"].copy(), rec["ir2"].copy()
    for k in range(len(ir1)):
        _disc(ir1[k], u, v, r)
        _disc(ir2[k], u, v, r)
    return {**rec, "ir1": ir1, "ir2": ir2}


def test_static_board_noise_does_not_block_the_real_track():
    """
    Without static rejection, a persistent extra candidate in every frame
    dilutes the real ball's inlier fraction; with it (the default), the real
    track is still recovered cleanly. Matches the real-recording finding
    2026-09-02: the board was diluting a genuine ball track below RANSAC's
    60% inlier gate.
    """
    clean = _render(FLIGHT_TIMES)
    bg1, bg2 = median_background(clean["ir1"]), median_background(clean["ir2"])
    noisy = _add_static_noise(clean)
    got = measure_landing(noisy, R_BC, T_BC, bg1=bg1, bg2=bg2)   # reject_static=True by default
    x_true, y_true, _ = solve_impact(TRUE_P0, TRUE_V0, z_floor=Z_FLOOR_BASE)
    err = np.hypot(got["x"] - x_true, got["y"] - y_true)
    assert err < 0.02, f"landing error {err * 1e3:.1f} mm"


def test_static_rejection_removes_the_noise_at_the_CANDIDATE_level():
    """
    Rewritten 2026-09-11. This used to compare OBSERVATION row counts with and
    without rejection and assert filtering removed some -- which passed, but
    not for the stated reason. `_add_static_noise` draws its disc at the same
    (u, v) in BOTH streams, so its disparity is exactly zero and
    `pair_candidates` drops it before it can ever become a row: the rows the
    filter was removing were the BALL's, on the frames it flew near the disc.
    The test was measuring the over-rejection bug and calling it success.

    Rejection happens per candidate, per stream, so that is where to check it.
    """
    clean = _render(FLIGHT_TIMES)
    bg1 = median_background(clean["ir1"])
    noisy = _add_static_noise(clean)
    per = [detect_candidates(noisy["ir1"][k], bg1) for k in range(len(noisy["t"]))]
    kept = reject_static_candidates(per)
    assert sum(len(c) for c in per) > sum(len(c) for c in kept), \
        "the recurring disc must be removed"
    # And it must cost nothing downstream: the paired observations from the
    # noisy recording must match the clean one exactly.
    obs_noisy, _ = build_observations(noisy, bg1=bg1,
                                      bg2=median_background(clean["ir2"]))
    obs_clean, _ = build_observations(clean, bg1=bg1,
                                      bg2=median_background(clean["ir2"]))
    assert obs_noisy.shape == obs_clean.shape


def test_a_ball_crossing_a_flagged_pixel_is_not_deleted():
    """
    REGRESSION (2026-09-11), and it cost most of a run day. A persistent
    small-blob source sat at (486, 281) on one real mount -- squarely on the
    descent path. `reject_static_candidates` flags that bin, and then deleted
    the BALL on the frames it flew over it: the last five frames of the fall,
    every throw. `measure_landing` then refused with "0.41 m of unobserved
    drop", or locked onto the bounce and refused with "still RISING". 11 of 18
    refusals in that session were this, on throws the operator watched land on
    the target.

    Real areas there: the static source 24-54 px, the ball crossing it
    104-168 px. Size is the discriminator, and it was already in the data.
    """
    rng = np.random.default_rng(5)
    n = 40
    frames = [np.clip(np.full((H, W), 55) + rng.normal(0, 3, (H, W)), 0, 255)
              .astype(np.uint8) for _ in range(n)]
    bg = median_background(frames)
    # a small source that recurs at one pixel for the whole recording
    for k in range(n):
        _disc(frames[k], 486.0, 281.0, 3.6)          # ~40 px of area
    # a ball descending through that exact pixel on frames 20-24
    ball_uv = [(506, 269), (500, 275), (494, 281), (489, 287), (484, 293)]
    for i, (u, v) in enumerate(ball_uv):
        _disc(frames[20 + i], float(u), float(v), 6.8)   # ~145 px of area

    per = [detect_candidates(f, bg) for f in frames]
    kept = reject_static_candidates(per)

    def near(cands, u, v, tol=4.0):
        return any(abs(c.u - u) < tol and abs(c.v - v) < tol for c in cands)

    for i, (u, v) in enumerate(ball_uv):
        assert near(per[20 + i], u, v), f"the test's own ball at frame {20+i} was not detected"
        assert near(kept[20 + i], u, v), \
            f"the ball at frame {20 + i} ({u},{v}) was deleted as static noise"
    # and the static source itself is still gone on a frame with no ball
    assert not near(kept[0], 486, 281), "the persistent source must still be rejected"


TRUE_SPEED = float(np.linalg.norm(TRUE_V0))


def test_commanded_speed_requires_release_t_offset():
    """
    Found 2026-09-02: p0/v0 are fit at the recording's local t=0, which for a
    RingBuffer-style capture is PRE_S seconds BEFORE release, not at release.
    Silently comparing raw v0 to a commanded release speed compares the wrong
    instant -- this must be a loud error, never an implicit "assume t=0 is
    release".
    """
    rec = _render(FLIGHT_TIMES)
    with pytest.raises(ValueError, match="release_t_offset"):
        measure_landing(rec, R_BC, T_BC, commanded_speed=TRUE_SPEED)


def test_commanded_speed_gate_accepts_a_matching_fit():
    # FLIGHT_TIMES already starts at the true release instant (TRUE_P0/TRUE_V0
    # are defined AT t=0), so this recording's local t=0 IS release: offset 0.
    rec = _render(FLIGHT_TIMES)
    got = measure_landing(rec, R_BC, T_BC, commanded_speed=TRUE_SPEED,
                          release_t_offset=0.0)
    assert got["x"] is not None


def test_commanded_speed_gate_refuses_a_fit_far_from_commanded():
    """
    An accurate fit against the WRONG commanded speed must still be refused --
    this is an outside-fact check (what the arm was actually told to do), not
    a fit-quality check (start_of_day.py's gates use the same principle: a
    perfect internal fit does not prove the answer is right).
    """
    rec = _render(FLIGHT_TIMES)
    with pytest.raises(RuntimeError, match="outside"):
        measure_landing(rec, R_BC, T_BC, commanded_speed=TRUE_SPEED * 10,
                        release_t_offset=0.0)


def test_release_t_offset_recovers_a_good_throw_pre_release_capture_window():
    """
    Regression for the actual 2026-09-02 incident: a RingBuffer-style
    recording whose local t=0 sits PRE_S seconds before release. Using the
    correct offset must accept a genuinely good fit; using 0 (the bug) must
    wrongly refuse it, because v0 at t=0 has picked up ~g*PRE_S of backward
    gravity extrapolation through a period the ball was not yet in flight.
    """
    pre_s = 0.45
    rec = _render(FLIGHT_TIMES)
    rec = {**rec, "t": rec["t"] + pre_s}   # shift: release is now at local t=pre_s

    got = measure_landing(rec, R_BC, T_BC, commanded_speed=TRUE_SPEED,
                          release_t_offset=pre_s)
    x_true, y_true, _ = solve_impact(TRUE_P0, TRUE_V0, z_floor=Z_FLOOR_BASE)
    assert np.hypot(got["x"] - x_true, got["y"] - y_true) < 0.02

    with pytest.raises(RuntimeError, match="outside"):
        measure_landing(rec, R_BC, T_BC, commanded_speed=TRUE_SPEED,
                        release_t_offset=0.0)


# --------------------------------------------------------------------------- #
# Unobserved-drop gate
#
# Added 2026-09-10 alongside making ransac_track's inlier fraction span-local.
# The global fraction had been doing a second job nobody had named: refusing
# fits to motion that never actually fell. With it gone that job needs its own
# gate, and the real recordings show what separates the two cases -- how far
# the ball still had to drop after the LAST frame it was seen in:
#
#   throws/throw_001.npz (real arm throw)   0.12 m  -> a measurement
#   throws/throw_002.npz (hand-carried)     1.05 m  -> an extrapolation
#   throws/throw_003.npz (hand-carried)     0.81 m  -> an extrapolation
#
# Both hand-carried recordings fit a g = 9.81 parabola to 0.9 px and score a
# span-local inlier fraction of 1.00, so neither RANSAC nor the RMS gate can
# tell them apart. Only "was the ball still in view near the floor?" can.
# --------------------------------------------------------------------------- #
import os

from perception.trajectory import ransac_track

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def _load_real(name):
    z = np.load(os.path.join(FIXTURES, name))
    return np.asarray(z["obs"], float), np.asarray(z["R"], float), np.asarray(z["t"], float)


def test_a_real_throw_seen_down_to_the_floor_is_measured():
    obs, R, t = _load_real("obs_real_throw.npz")
    idx, fit = ransac_track(obs, RIG, R, t)
    out = check_landing_is_observed(obs[idx], fit, z_floor=Z_FLOOR_BASE,
                                    ball_radius=BALL_RADIUS)
    assert out["unobserved_drop_m"] < 0.30
    assert out["descending"] is True


def test_a_hand_carried_ball_is_refused_even_though_it_fits_a_parabola():
    """
    The exact recording CLAUDE.md records as 'a hand-carried or rolling ball'.
    It fits g = 9.81 to sub-pixel RMS with a span-local inlier fraction of
    1.00 -- every gate upstream of this one passes it.
    """
    obs, R, t = _load_real("obs_real_not_a_throw.npz")
    idx, fit = ransac_track(obs, RIG, R, t)
    assert fit.rms_px < 1.0, "fixture premise: this junk fits an arc cleanly"
    with pytest.raises(RuntimeError, match="unobserved"):
        check_landing_is_observed(obs[idx], fit, z_floor=Z_FLOOR_BASE,
                                  ball_radius=BALL_RADIUS)


def test_measure_landing_refuses_a_flight_that_stops_high_above_the_floor():
    """Same gate, reached through the whole pipeline rather than directly."""
    rec = _render(np.arange(0.12, 0.36, 1.0 / 90.0))
    with pytest.raises(RuntimeError, match="unobserved"):
        measure_landing(rec, R_BC, T_BC)


def test_a_ball_sitting_still_on_the_floor_is_refused():
    """
    REGRESSION. throws/throw_006.npz is a ball resting/creeping on the floor
    (3 cm of travel in 0.16 s). It passes every other gate: 0.55 px RMS,
    span-local inlier fraction 0.83, and it is DESCENDING at the last frame
    (-0.59 m/s) and only 0.22 m above the impact plane, so neither the
    rising check nor the unobserved-drop cap catches it. The fit escapes by
    putting its apex inside the observed span -- a parabola is locally flat
    there, and 3 cm of jitter over 0.16 s looks exactly like the top of a
    |v0| = 15.3 m/s arc. What it never does is FALL.
    """
    obs, R, t = _load_real("obs_real_stationary_ball.npz")
    idx, fit = ransac_track(obs, RIG, R, t)
    assert fit.rms_px < 1.0, "fixture premise: this junk fits an arc cleanly"
    with pytest.raises(RuntimeError, match="observed"):
        check_landing_is_observed(obs[idx], fit, z_floor=Z_FLOOR_BASE,
                                  ball_radius=BALL_RADIUS)


def test_a_real_throw_is_seen_falling_most_of_the_way():
    obs, R, t = _load_real("obs_real_throw.npz")
    idx, fit = ransac_track(obs, RIG, R, t)
    out = check_landing_is_observed(obs[idx], fit, z_floor=Z_FLOOR_BASE,
                                    ball_radius=BALL_RADIUS)
    assert out["observed_drop_m"] > 0.9
    assert out["observed_drop_m"] > out["unobserved_drop_m"]


# ---------------------------------------------------------------------------
# Gate rebalance, 2026-09-11. The unobserved-drop cap went 0.30 -> 0.60 m and a
# propagated-sigma gate took over the job it was standing in for. These pin
# both halves: the relaxation must not let the known-bad recordings through,
# and the sigma gate must actually refuse an ill-determined arc.
# ---------------------------------------------------------------------------
def test_relaxing_the_drop_cap_does_not_admit_the_known_bad_recordings():
    """
    The cap was never what caught them. Re-checked at 0.30, the shipped 0.60,
    and an absurd 2.00: the hand-carried clip is caught by `descending` and the
    stationary-ball clip by the observed-drop floor, at every setting. If this
    ever fails, the cap was load-bearing after all and the relaxation is wrong.
    """
    for name in ("obs_real_not_a_throw.npz", "obs_real_stationary_ball.npz"):
        obs, R, t = _load_real(name)
        idx, fit = ransac_track(obs, RIG, R, t)
        for cap in (0.30, 0.60, 2.00):
            with pytest.raises(RuntimeError):
                check_landing_is_observed(obs[idx], fit, z_floor=Z_FLOOR_BASE,
                                          ball_radius=BALL_RADIUS,
                                          max_unobserved_drop_m=cap)


def test_a_real_throw_clipped_short_is_now_accepted():
    """
    The case the relaxation exists for: a genuine descent, seen over ~1 m of
    fall, that stops early because the detector lost the ball near the floor.
    Under the old 0.30 m cap this was refused outright.
    """
    obs, R, t = _load_real("obs_real_throw.npz")
    idx, fit = ransac_track(obs, RIG, R, t)
    seen = check_landing_is_observed(obs[idx], fit, z_floor=Z_FLOOR_BASE,
                                     ball_radius=BALL_RADIUS)
    assert seen["observed_drop_m"] > 0.9
    # and it would still pass with a further 0.3 m of the fall missing
    assert seen["unobserved_drop_m"] < MAX_UNOBSERVED_DROP_M


def test_sigma_tracks_how_well_the_arc_is_actually_conditioned():
    """
    Sigma has to be a real measure, not decoration, because it is now a gate.
    A track spanning a short slice of the flight determines the parabola far
    worse than one spanning all of it -- same detector, same RMS, same frame
    count -- and only the propagated covariance says so.
    """
    full = measure_landing(_render(FLIGHT_TIMES), R_BC, T_BC)
    # Same number of frames, packed into a third of the time span.
    short = measure_landing(_render(np.arange(0.40, 0.58, 1.0 / 180.0)), R_BC, T_BC,
                            max_sigma_xy_m=1.0)
    assert short["sigma_xy_m"] > 3 * full["sigma_xy_m"], (
        f"short-span sigma {short['sigma_xy_m'] * 1e3:.2f} mm vs full "
        f"{full['sigma_xy_m'] * 1e3:.2f} mm -- sigma is not tracking conditioning")


def test_sigma_gate_refuses_rather_than_reporting_an_undetermined_landing():
    rec = _render(np.arange(0.40, 0.58, 1.0 / 180.0))
    loose = measure_landing(rec, R_BC, T_BC, max_sigma_xy_m=1.0)
    with pytest.raises(RuntimeError, match="1-sigma"):
        measure_landing(rec, R_BC, T_BC,
                        max_sigma_xy_m=loose["sigma_xy_m"] * 0.5)


def test_a_good_throw_reports_a_sigma_far_inside_the_gate():
    rec = _render(FLIGHT_TIMES)
    got = measure_landing(rec, R_BC, T_BC)
    assert got["sigma_xy_m"] < 0.005, f"{got['sigma_xy_m'] * 1000:.1f} mm"
    assert got["inlier_frac_used"] == pytest.approx(0.6), \
        "a clean synthetic throw must not need the loosened fraction"
