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
