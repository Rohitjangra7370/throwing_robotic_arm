"""
Floor-marker localisation: an ArUco tag on the floor -> base-frame position.

Renders markers through the SAME intrinsics and extrinsic the overhead D435i
actually has, at the real 239 px/m floor scale, and checks the position comes
back. The renderer is the mirror image of the detector's job, so a frame or
handedness mistake in either shows up as a large error rather than passing.
"""
import cv2
import numpy as np
import pytest

from perception.floor_marker import (ARUCO_DICT, detect_bin_marker,
                                     detect_floor_markers)
from perception.ray_plane import D435I_IR_848x480 as INTR

W, H = 848, 480
Z_FLOOR = -0.433

# The real overhead mount, near-nadir, measured 2026-08-31. Camera +x maps
# almost exactly onto base +y here, which is why a colour-vs-IR1 frame slip
# would show up as a lateral bias.
R_BC = np.array([[-0.0029, 0.9968, 0.0802],
                 [0.9996, 0.0051, -0.0273],
                 [-0.0276, 0.0800, -0.9964]])
T_BC = np.array([0.8331, -0.0608, 1.3355])

# IR ink levels sampled off the ChArUco board sitting in a real recording, and
# the floor grey around it.
BLACK, WHITE, FLOOR = 28, 123, 57


def _orthonormal(R):
    u, _, vt = np.linalg.svd(R)
    return u @ vt


R_BC = _orthonormal(R_BC)


def _base_to_pixel(P):
    p_c = R_BC.T @ (np.asarray(P, float) - T_BC)
    return np.array([INTR.fx * p_c[0] / p_c[2] + INTR.ppx,
                     INTR.fy * p_c[1] / p_c[2] + INTR.ppy])


def _render(centre_xy, side_m, angle=0.0, marker_id=7, seed=0, img=None,
            z_floor=Z_FLOOR):
    """
    Draw one floor-lying marker into an IR-like frame.

    The corner order is REVERSED relative to the natural base-frame winding.
    An overhead camera flips handedness, and feeding the un-reversed order to
    getPerspectiveTransform renders the marker MIRRORED -- which decodes to no
    valid id at all and reads as "too small to detect" while sizes are swept.
    That cost an hour on 2026-09-11; the reversal is the fix, and this comment
    is why it is not a typo.
    """
    rng = np.random.default_rng(seed)
    if img is None:
        img = np.full((H, W), FLOOR, np.uint8)
        img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    cx, cy = centre_xy
    c, s = np.cos(angle), np.sin(angle)
    L = side_m
    loc = np.array([[-L / 2, -L / 2], [L / 2, -L / 2],
                    [L / 2, L / 2], [-L / 2, L / 2]])[::-1]
    world = np.array([[cx + c * a - s * b, cy + s * a + c * b] for a, b in loc])
    dst = np.array([_base_to_pixel([w[0], w[1], z_floor]) for w in world], np.float32)
    pix = float(np.mean([np.linalg.norm(dst[i] - dst[(i + 1) % 4]) for i in range(4)]))

    # Render at ~2x the destination size: warpPerspective is bilinear, and
    # downsampling further than 2x aliases the bit cells into garbage.
    m_px = max(24, int(round(pix * 2)))
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    tag = cv2.aruco.generateImageMarker(d, marker_id, m_px)
    qz = max(3, int(round(m_px * 0.15)))
    tag = cv2.copyMakeBorder(tag, qz, qz, qz, qz, cv2.BORDER_CONSTANT, value=255)
    ink = (BLACK + (tag / 255.0) * (WHITE - BLACK)).astype(np.uint8)
    n = ink.shape[0]
    src = np.array([[qz, qz], [n - qz, qz], [n - qz, n - qz], [qz, n - qz]], np.float32)
    Hm = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(ink, Hm, (W, H), borderValue=0)
    mask = cv2.warpPerspective(np.full((n, n), 255, np.uint8), Hm, (W, H), borderValue=0)
    out = img.copy()
    out[mask > 128] = warp[mask > 128]
    return out, pix


def test_marker_centre_is_recovered_to_a_few_millimetres():
    want = np.array([1.22, 0.0])
    img, _ = _render(want, 0.150)
    m = detect_bin_marker(img, INTR, R_BC, T_BC, z_floor=Z_FLOOR,
                          marker_size_m=0.150)
    assert np.linalg.norm(m.centre_xy - want) < 0.005, f"got {m.centre_xy}"
    assert m.marker_id == 7


@pytest.mark.parametrize("side_mm", [80, 150, 200, 300])
def test_accuracy_is_independent_of_marker_size(side_mm):
    """
    Measured 2026-09-11 against a real IR background: detection succeeds and
    the centre lands within ~2 mm from 80 mm (18.8 px) upward. An 80 mm tag is
    the smallest the existing bin marker could be, so it must keep working.
    """
    want = np.array([1.18, -0.10])
    img, pix = _render(want, side_mm / 1000.0, angle=0.4, seed=side_mm)
    m = detect_bin_marker(img, INTR, R_BC, T_BC, z_floor=Z_FLOOR,
                          marker_size_m=side_mm / 1000.0)
    assert np.linalg.norm(m.centre_xy - want) < 0.006, \
        f"{side_mm} mm ({pix:.1f} px): {m.centre_xy} vs {want}"


@pytest.mark.parametrize("angle", [0.0, 0.3, 0.9, 1.4])
def test_rotation_does_not_move_the_recovered_centre(angle):
    want = np.array([1.25, 0.20])
    img, _ = _render(want, 0.150, angle=angle, seed=int(angle * 100))
    m = detect_bin_marker(img, INTR, R_BC, T_BC, z_floor=Z_FLOOR)
    assert np.linalg.norm(m.centre_xy - want) < 0.006


def test_marker_is_found_across_the_whole_reachable_landing_band():
    """Every point aim_at.py can solve for must actually be localisable."""
    for x in (1.14, 1.22, 1.30):
        for y in (-0.45, 0.0, 0.45):
            img, _ = _render(np.array([x, y]), 0.150, angle=0.2)
            m = detect_bin_marker(img, INTR, R_BC, T_BC, z_floor=Z_FLOOR)
            assert np.linalg.norm(m.centre_xy - np.array([x, y])) < 0.008, \
                f"({x}, {y}) -> {m.centre_xy}"


def test_no_marker_in_view_refuses_rather_than_returning_a_position():
    rng = np.random.default_rng(2)
    blank = np.clip(np.full((H, W), FLOOR) + rng.normal(0, 3, (H, W)),
                    0, 255).astype(np.uint8)
    assert detect_floor_markers(blank, INTR, R_BC, T_BC) == []
    with pytest.raises(RuntimeError, match="no ArUco marker"):
        detect_bin_marker(blank, INTR, R_BC, T_BC)


def test_two_markers_without_an_id_refuses_to_guess():
    img, _ = _render(np.array([1.20, -0.20]), 0.150, marker_id=7)
    img, _ = _render(np.array([1.20, 0.25]), 0.150, marker_id=9, img=img)
    with pytest.raises(RuntimeError, match="Refusing to guess"):
        detect_bin_marker(img, INTR, R_BC, T_BC)
    m = detect_bin_marker(img, INTR, R_BC, T_BC, marker_id=9)
    assert m.marker_id == 9
    assert np.linalg.norm(m.centre_xy - np.array([1.20, 0.25])) < 0.008


def test_the_glued_down_calibration_board_can_be_excluded():
    """
    The ChArUco board is glued to this floor and shares the dictionary. Its
    tags are ~6 px at this mount and do not detect today, but that is a
    property of the mount, not a guarantee.
    """
    img, _ = _render(np.array([1.20, 0.0]), 0.150, marker_id=7)
    img, _ = _render(np.array([0.55, 0.0]), 0.150, marker_id=3, img=img)
    with pytest.raises(RuntimeError, match="Refusing to guess"):
        detect_bin_marker(img, INTR, R_BC, T_BC)
    m = detect_bin_marker(img, INTR, R_BC, T_BC, exclude_ids=(3,))
    assert m.marker_id == 7


def test_a_rescaled_printout_is_caught_by_the_size_check():
    """
    The failure make_aruco_printable.py's ruler exists for: the tag was printed
    at 150 mm but the caller believes 80 mm. Detection is perfectly happy;
    only a length measured against the known floor plane can catch it.
    """
    img, _ = _render(np.array([1.20, 0.0]), 0.150)
    ok = detect_bin_marker(img, INTR, R_BC, T_BC, marker_size_m=0.150)
    assert abs(ok.side_error_m) < 0.010
    with pytest.raises(RuntimeError, match="rescaled printout|measures"):
        detect_bin_marker(img, INTR, R_BC, T_BC, marker_size_m=0.080)


def test_a_wrong_floor_height_is_caught_by_the_size_check():
    """A stale z_floor scales every distance; the size check is what sees it."""
    img, _ = _render(np.array([1.20, 0.0]), 0.150)
    with pytest.raises(RuntimeError, match="measures"):
        detect_bin_marker(img, INTR, R_BC, T_BC, z_floor=Z_FLOOR + 0.30,
                          marker_size_m=0.150)


def test_size_check_is_skipped_when_no_printed_size_is_given():
    img, _ = _render(np.array([1.20, 0.0]), 0.150)
    m = detect_bin_marker(img, INTR, R_BC, T_BC)
    assert np.isnan(m.side_error_m)


# ---------------------------------------------------------------------------
# End to end: a marker on the floor -> the number to type into the session GUI.
# This is the "move the bin, tell me where to aim" loop, exercised without a
# camera, an arm, or Tk. Slow (it loads the checkpoint and runs the release LP
# per Newton step), so it is one test, not a sweep.
# ---------------------------------------------------------------------------
def test_marker_position_becomes_a_commanded_target_that_lands_on_it():
    import argparse
    import os

    if not os.path.isdir("results_kinetic_chain_gen3_tcp/1"):
        pytest.skip("trained checkpoint not on this machine")

    from hardware_session import aim_at_bin, aimer_args

    want = np.array([1.21, -0.18])          # inside the reachable band
    img, _ = _render(want, 0.150, angle=0.35)

    base = argparse.Namespace(
        robot="kinova_gen3_dyn", log_path="results_kinetic_chain_gen3_tcp/1",
        opt_pose="throw_pose_table_tcp.npy", tool_offset_z=0.12,
        base_height=0.433, ball_radius=0.0327, u_cap=2.00,
        wrist_roll_offset_deg=90.0, aim_model="additive")
    marker, target, pred, refusals = aim_at_bin(
        img, aimer_args(base), (R_BC, T_BC), marker_size_m=0.150)

    assert refusals == [], refusals
    assert np.linalg.norm(marker.centre_xy - want) < 0.006

    # The commanded target is NOT the bin -- that is the entire point. It is
    # ~45 cm shorter, because the arm throws that much further than the
    # checkpoint believes.
    assert target[0] < want[0] - 0.35, \
        f"target {target} should be far short of the bin at {want}"

    # And the prediction must land back on the marker.
    assert np.linalg.norm(np.array(pred["predicted_landing"]) - marker.centre_xy) < 0.005
    assert pred["measurable"] and pred["precheck_ok"] and pred["release_in_box"]
