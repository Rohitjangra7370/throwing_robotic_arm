"""
Stereo IR triangulation. Pure geometry -- no camera.

Every landing measurement flows through this, and a baseline or a disparity
sign error here produces plausible-looking positions rather than an exception,
so it is pinned down before anything else is built on it.
"""
import numpy as np
import pytest

from perception.ray_plane import D435I_IR_848x480
from perception.stereo import D435I_IR_BASELINE_M, StereoRig, pair_candidates

RIG = StereoRig(D435I_IR_848x480, D435I_IR_BASELINE_M)


def test_ir_constants_match_the_device():
    """Read off the lab D435i on 2026-08-25; guards a silent edit."""
    intr = D435I_IR_848x480
    assert (intr.fx, intr.fy) == (426.167, 426.167)
    assert (intr.ppx, intr.ppy) == (420.286, 238.505)
    assert (intr.width, intr.height) == (848, 480)
    assert not any(intr.coeffs), "IR stream is factory-rectified"
    assert D435I_IR_BASELINE_M == 0.0499448
    assert intr.hfov_deg() == pytest.approx(89.7, abs=0.1)
    assert intr.vfov_deg() == pytest.approx(58.8, abs=0.1)


def test_disparity_at_two_metres_matches_hand_calculation():
    """d = fx*B/Z = 426.167*0.0499448/2.0 = 10.642 px. A baseline typo shows here."""
    u1, v1, u2, v2 = RIG.project(np.array([0.0, 0.0, 2.0]))
    assert (u1 - u2) == pytest.approx(10.642, abs=0.01)


def test_triangulate_inverts_project_exactly():
    p = np.array([0.30, -0.20, 2.00])
    got = RIG.triangulate(*RIG.project(p))
    assert np.allclose(got, p, atol=1e-9)


def test_rectified_pair_has_identical_rows():
    """IR1->IR2 rotation is exactly identity, so v2 must equal v1."""
    u1, v1, u2, v2 = RIG.project(np.array([0.4, 0.25, 1.7]))
    assert v2 == pytest.approx(v1, abs=1e-12)


def test_ir2_is_to_the_right_so_disparity_is_positive():
    """IR2 sits at +49.9448 mm along IR1's +X. Getting this backwards flips
    the sign of every depth. A negative disparity must not yield a position."""
    u1, v1, u2, v2 = RIG.project(np.array([0.0, 0.0, 2.0]))
    assert u1 > u2
    with pytest.raises(RuntimeError, match="disparity"):
        RIG.triangulate(u1, v1, u1 + 1.0, v2)


def test_batch_shape_contract():
    pts = np.array([[0.0, 0.0, 2.0], [0.3, -0.2, 1.5], [-0.4, 0.1, 2.4]])
    u1, v1, u2, v2 = RIG.project(pts)
    assert u1.shape == (3,)
    got = RIG.triangulate(u1, v1, u2, v2)
    assert got.shape == (3, 3)
    assert np.allclose(got, pts, atol=1e-9)


def test_pairing_matches_on_row_and_area():
    left = [(500.0, 200.0, 150.0), (300.0, 400.0, 140.0)]
    right = [(292.0, 400.5, 145.0), (489.0, 199.6, 152.0)]
    assert sorted(pair_candidates(left, right)) == [(0, 1), (1, 0)]


def test_pairing_rejects_a_row_mismatch():
    left = [(500.0, 200.0, 150.0)]
    right = [(489.0, 260.0, 152.0)]
    assert pair_candidates(left, right) == []


def test_pairing_rejects_an_area_mismatch():
    """Same row, but one blob is 10x the other -- not the same object."""
    left = [(500.0, 200.0, 150.0)]
    right = [(489.0, 200.0, 1500.0)]
    assert pair_candidates(left, right) == []


def test_pairing_rejects_negative_disparity():
    """A right-image detection to the RIGHT of its left partner is impossible."""
    left = [(400.0, 200.0, 150.0)]
    right = [(430.0, 200.0, 150.0)]
    assert pair_candidates(left, right) == []


def test_pairing_is_one_to_one_and_prefers_the_closer_row():
    left = [(500.0, 200.0, 150.0)]
    right = [(489.0, 202.9, 150.0), (487.0, 200.1, 150.0)]
    assert pair_candidates(left, right) == [(0, 1)]
