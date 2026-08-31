"""Tests for the pure decision + learning logic behind the hardware session."""
import numpy as np
import pytest

from hardware_learning import next_allowed_scale, propose_targets, scale_allowed

BAND = ((0.68, 0.74), (-0.25, 0.25))


def test_propose_targets_stays_inside_the_trained_band():
    t = propose_targets(10, BAND, seed=0)
    assert t.shape == (10, 2)
    assert np.all(t[:, 0] >= 0.68) and np.all(t[:, 0] <= 0.74)
    assert np.all(t[:, 1] >= -0.25) and np.all(t[:, 1] <= 0.25)


def test_propose_targets_actually_spreads():
    """10 near-identical throws teach the GP almost nothing -- that is the point."""
    t = propose_targets(10, BAND, seed=0)
    assert t[:, 1].max() - t[:, 1].min() > 0.30   # uses most of the y range
    assert t[:, 0].max() - t[:, 0].min() > 0.03   # and both ends of the narrow x range


def test_propose_targets_is_deterministic_for_a_seed():
    assert np.allclose(propose_targets(10, BAND, seed=7), propose_targets(10, BAND, seed=7))


def test_escalation_starts_at_the_bottom_of_the_ladder():
    assert next_allowed_scale([]) == pytest.approx(0.15)
    ok, why = scale_allowed(1.00, [])
    assert not ok and "0.15" in why


def test_escalation_advances_one_rung_per_clean_run():
    assert next_allowed_scale([0.15]) == pytest.approx(0.30)
    assert next_allowed_scale([0.15, 0.30]) == pytest.approx(0.60)
    assert next_allowed_scale([0.15, 0.30, 0.60]) == pytest.approx(1.00)


def test_escalation_refuses_skipping_a_rung():
    ok, why = scale_allowed(0.60, [0.15])
    assert not ok and "0.30" in why


def test_escalation_allows_repeating_or_dropping_back():
    assert scale_allowed(0.15, [0.15, 0.30])[0]
    assert scale_allowed(0.30, [0.15, 0.30])[0]


def test_track_to_state_samples_has_the_exact_shapes_the_model_expects():
    """(n, 8) = [x,y,z,vx,vy,vz,Px,Py] and (n, 1) with the speed only at t=0 --
    verified against PyBulletThrowingSystem.rollout, not assumed."""
    from hardware_learning import track_to_state_samples
    t = np.arange(0.0, 0.50, 1 / 90.0)
    p0, v0, g = np.array([0.3, 0.0, 0.02]), np.array([1.39, 0.0, 0.37]), np.array([0, 0, -9.81])
    pts = p0 + np.outer(t, v0) + 0.5 * np.outer(t ** 2, g)
    s, u = track_to_state_samples(pts, t, (0.71, 0.02), 1.44, ts=0.02)
    assert s.shape[1] == 8 and u.shape[1] == 1
    assert s.shape[0] == u.shape[0]
    assert u[0, 0] == pytest.approx(1.44)
    assert np.allclose(u[1:, 0], 0.0)
    assert np.allclose(s[:, 6], 0.71) and np.allclose(s[:, 7], 0.02)


def test_track_to_state_samples_recovers_a_known_velocity_profile():
    from hardware_learning import track_to_state_samples
    t = np.arange(0.0, 0.50, 1 / 90.0)
    p0, v0, g = np.array([0.3, 0.0, 0.02]), np.array([1.39, 0.0, 0.37]), np.array([0, 0, -9.81])
    pts = p0 + np.outer(t, v0) + 0.5 * np.outer(t ** 2, g)
    s, _ = track_to_state_samples(pts, t, (0.71, 0.02), 1.44, ts=0.02)
    assert np.allclose(s[0, 0:3], p0, atol=2e-3)
    assert np.allclose(s[0, 3:6], v0, atol=2e-2)
    dt = 0.02
    dv = (s[1:, 3:6] - s[:-1, 3:6]) / dt
    assert np.allclose(dv[:, 2].mean(), -9.81, atol=0.5)


def test_track_to_state_samples_is_sampled_at_ts_not_at_camera_rate():
    """90 fps in, 50 Hz out -- the GP's propagation assumes Ts spacing."""
    from hardware_learning import track_to_state_samples
    t = np.arange(0.0, 0.50, 1 / 90.0)
    pts = np.stack([t * 1.4, t * 0, 0.02 - 4.9 * t ** 2], axis=1)
    s, _ = track_to_state_samples(pts, t, (0.71, 0.0), 1.44, ts=0.02)
    assert 24 <= s.shape[0] <= 26        # 0.50 s / 0.02 s


def test_track_to_state_samples_rejects_a_track_too_short_to_difference():
    from hardware_learning import track_to_state_samples
    t = np.array([0.0, 0.01])
    pts = np.zeros((2, 3))
    with pytest.raises(ValueError, match="too short"):
        track_to_state_samples(pts, t, (0.71, 0.0), 1.44, ts=0.02)
