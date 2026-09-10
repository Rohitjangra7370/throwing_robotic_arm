"""
Regression tests for the wrist-camera calibration chain.

The defect these exist to prevent (found 2026-08-31, on the arm): both
calibration scripts built `base -> wrist camera` by composing the URDF's
FLANGE -> camera offset onto `Base.GetMeasuredCartesianPose()`. That call does
not report the flange -- it reports the TOOL frame, and this arm has
`tool_transform = (0, 0, 0.12) m` configured for the Robotiq 2F-85. Measured
against PyBullet FK on the same URDF the throw planner uses, the reported pose
sat 0.1251 m along flange +z from the flange. Chained into the extrinsic that
put the calibration board 14.0 cm below the floor it was physically lying on;
via FK instead it lands 1.2 cm from it.

That is the same 0.12 m tool offset that already cost this project a retrain
(see CLAUDE.md's gripper-TCP-offset entry) reappearing in a second place, so
it gets a regression test rather than just a fix.
"""
import cv2
import numpy as np
import pytest

from perception.wrist_chain import (
    CAMERA_COLOR_LINK,
    average_transforms,
    FLANGE_LINK,
    board_from_spec,
    base_to_wrist_camera,
    chain_base_to_camera,
    compose,
    detect_board_pose,
    drift_check,
    floor_plane_check,
    invert,
    rt,
)

Q_LAB = np.array([0.00028, 1.35528, -3.09902, -0.66248, -0.03101, -0.78639, 1.55079])
TOOL_Z = 0.12


# --------------------------------------------------------------------------
# geometry primitives
# --------------------------------------------------------------------------
def test_invert_is_a_true_inverse():
    R, t = rt([0.1, -0.2, 0.3], [12.0, -34.0, 56.0])
    Ri, ti = invert(R, t)
    R0, t0 = compose(R, t, Ri, ti)
    assert np.allclose(R0, np.eye(3), atol=1e-12)
    assert np.allclose(t0, 0.0, atol=1e-12)


def test_compose_matches_explicit_point_mapping():
    R1, t1 = rt([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
    R2, t2 = rt([-0.5, 0.25, 0.75], [-5.0, 15.0, 45.0])
    R, t = compose(R1, t1, R2, t2)
    p = np.array([0.3, -0.4, 0.5])
    assert np.allclose(R @ p + t, R1 @ (R2 @ p + t2) + t1, atol=1e-12)


# --------------------------------------------------------------------------
# the actual bug
# --------------------------------------------------------------------------
def test_wrist_camera_offset_matches_the_urdf():
    """FK camera pose must sit at the URDF's documented flange->camera offset."""
    R_bc, t_bc = base_to_wrist_camera(Q_LAB)
    R_bf, t_bf = base_to_wrist_camera(Q_LAB, cam_link=FLANGE_LINK)
    offset_flange = R_bf.T @ (t_bc - t_bf)
    assert np.allclose(offset_flange, [0.0, 0.05639, -0.00305], atol=1e-5)
    # ...and the optical frame is already optical in this URDF: a pure diag(-1,-1,1)
    # flip off the flange. Composing a ROS->optical rotation on top is the error
    # this asserts against (it threw the board 86 cm off in the lab).
    assert np.allclose(R_bf.T @ R_bc, np.diag([-1.0, -1.0, 1.0]), atol=1e-6)


def test_tool_frame_pose_is_not_the_flange_pose():
    """
    THE REGRESSION. Anyone reintroducing GetMeasuredCartesianPose as the base of
    this chain reintroduces a 12 cm error; this pins the magnitude.
    """
    R_bf, t_bf = base_to_wrist_camera(Q_LAB, cam_link=FLANGE_LINK)
    R_tool, t_tool = compose(R_bf, t_bf, *rt([0, 0, TOOL_Z], [0, 0, 0]))

    R_ec, t_ec = rt([0.0, 0.05639, -0.00305], [180.0, 180.0, 0.0])
    cam_via_flange = compose(R_bf, t_bf, R_ec, t_ec)[1]
    cam_via_tool = compose(R_tool, t_tool, R_ec, t_ec)[1]

    err = np.linalg.norm(cam_via_tool - cam_via_flange)
    assert err == pytest.approx(TOOL_Z, abs=1e-9), (
        "composing the camera offset onto the TOOL pose must be exactly one "
        "tool_transform away from the truth -- if this changed, the chain moved")

    # and the correct one agrees with FK to sub-millimetre
    assert np.allclose(cam_via_flange, base_to_wrist_camera(Q_LAB)[1], atol=1e-6)


def test_base_to_wrist_camera_rejects_wrong_joint_count():
    with pytest.raises(ValueError, match="7 joint angles"):
        base_to_wrist_camera(np.zeros(6))


# --------------------------------------------------------------------------
# board detection: duplicate marker ids
# --------------------------------------------------------------------------
def _render_board(px=900):
    board = board_from_spec(5, 7, 35.0, 0.75)
    img = board.generateImage((px, int(px * 7 / 5)))
    canvas = np.full((img.shape[0] + 400, img.shape[1] + 400), 255, np.uint8)
    canvas[200:200 + img.shape[0], 200:200 + img.shape[1]] = img
    return board, canvas


def _fake_K(gray):
    h, w = gray.shape[:2]
    return np.array([[900.0, 0, w / 2], [0, 900.0, h / 2], [0, 0, 1]])


def test_detect_board_pose_on_clean_render():
    board, gray = _render_board()
    det = detect_board_pose(gray, board, _fake_K(gray), np.zeros(5))
    assert det.n_corners >= 20
    assert det.reproj_px < 1.0
    assert det.dropped_ids == []


def test_detect_board_pose_drops_a_duplicated_marker_id():
    """
    The D435i saw a loose id=1 marker taped to a box beside the board. A
    duplicated id makes CharucoDetector.detectBoard return ZERO corners, which
    reads as 'board not visible' rather than 'ambiguous id'. Filtering the
    duplicate recovers the detection.
    """
    board, gray = _render_board()
    dup_id = int(np.asarray(board.getIds()).flatten()[1])
    stamp = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), dup_id, 120)
    gray[20:140, 20:140] = stamp

    naive = cv2.aruco.CharucoDetector(board).detectBoard(gray)[0]
    assert naive is None or len(naive) == 0, (
        "if OpenCV starts tolerating duplicate ids this filter is obsolete")

    det = detect_board_pose(gray, board, _fake_K(gray), np.zeros(5))
    assert det.dropped_ids == [dup_id]
    assert det.n_corners >= 15


def test_detect_board_pose_raises_when_board_absent():
    gray = np.full((480, 640), 255, np.uint8)
    with pytest.raises(RuntimeError, match="no ArUco markers"):
        detect_board_pose(gray, board_from_spec(5, 7, 35.0, 0.75),
                          _fake_K(gray), np.zeros(5))


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------
def test_floor_plane_check_passes_for_a_board_lying_on_the_floor():
    R, t = rt([1.2, 0.08, -0.4453], [0.0, 0.0, 25.0])
    ok, dz, tilt = floor_plane_check(R, t, floor_z=-0.433, tol_m=0.02, tol_deg=5.0)
    assert ok
    assert dz == pytest.approx(-0.0123, abs=1e-4)
    assert tilt == pytest.approx(0.0, abs=1e-9)


def test_floor_plane_check_catches_the_tool_frame_error():
    """The 14 cm the bug produced must not pass the gate."""
    R, t = rt([1.2, 0.08, -0.5728], [0.0, 0.0, 25.0])
    ok, dz, _ = floor_plane_check(R, t, floor_z=-0.433, tol_m=0.02, tol_deg=5.0)
    assert not ok
    assert dz == pytest.approx(-0.14, abs=0.001)


def test_floor_plane_check_catches_a_tilted_board():
    R, t = rt([1.2, 0.08, -0.433], [30.0, 0.0, 0.0])
    ok, _, tilt = floor_plane_check(R, t, floor_z=-0.433, tol_m=0.02, tol_deg=5.0)
    assert not ok
    assert tilt == pytest.approx(30.0, abs=1e-6)


def test_floor_plane_check_accepts_an_upside_down_board_normal():
    """Sign of the board's z axis is a detection convention, not a tilt."""
    R, t = rt([1.2, 0.08, -0.433], [180.0, 0.0, 0.0])
    ok, _, tilt = floor_plane_check(R, t, floor_z=-0.433, tol_m=0.02, tol_deg=5.0)
    assert ok
    assert tilt == pytest.approx(0.0, abs=1e-6)


def test_drift_check_flags_a_knocked_camera():
    R_a, t_a = rt([0.83, -0.03, 1.16], [0.0, 0.0, 90.0])
    R_b, t_b = rt([0.83, -0.03, 1.16], [0.0, 0.0, 90.0])
    ok, dt, dR = drift_check(R_a, t_a, R_b, t_b, tol_m=0.03, tol_deg=5.0)
    assert ok and dt == pytest.approx(0.0) and dR == pytest.approx(0.0)

    R_c, t_c = rt([0.83, 0.05, 1.16], [0.0, 0.0, 90.0])
    ok, dt, _ = drift_check(R_a, t_a, R_c, t_c, tol_m=0.03, tol_deg=5.0)
    assert not ok and dt == pytest.approx(0.08, abs=1e-9)

    R_d, t_d = rt([0.83, -0.03, 1.16], [0.0, 0.0, 100.0])
    ok, _, dR = drift_check(R_a, t_a, R_d, t_d, tol_m=0.03, tol_deg=5.0)
    assert not ok and dR == pytest.approx(10.0, abs=1e-6)


def test_average_transforms_centres_and_reports_worst_deviation():
    base_R, base_t = rt([0.83, -0.03, 1.16], [0.0, 0.0, 90.0])
    shots = [(base_R, base_t + d) for d in
             ([0, 0, 0], [0.01, 0, 0], [-0.01, 0, 0], [0, 0.004, 0])]
    R, t, spread_m, spread_deg = average_transforms(shots)
    assert np.allclose(t, base_t, atol=6e-3)          # median, not dragged by outliers
    assert spread_m == pytest.approx(0.01, abs=1e-3)  # WORST, not an rms
    assert spread_deg == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(R, base_R, atol=1e-9)


def test_average_transforms_reports_rotation_spread():
    shots = [rt([0, 0, 0], [0, 0, a]) for a in (-2.0, 0.0, 2.0)]
    _, _, _, spread_deg = average_transforms(shots)
    assert spread_deg == pytest.approx(2.0, abs=1e-6)


def test_chain_base_to_camera_recovers_a_known_extrinsic():
    """Synthetic round trip: invent T_B_C, project a board, chain it back."""
    R_BC, t_BC = rt([0.83, -0.03, 1.16], [178.0, 2.0, 88.0])
    R_bw, t_bw = rt([0.80, -0.01, 0.10], [160.0, 0.0, 87.0])
    R_bm, t_bm = rt([1.17, 0.09, -0.445], [0.0, 0.0, 25.0])

    R_wm, t_wm = compose(*invert(R_bw, t_bw), R_bm, t_bm)   # wrist cam sees board
    R_dm, t_dm = compose(*invert(R_BC, t_BC), R_bm, t_bm)   # D435i sees board

    R_out, t_out = chain_base_to_camera(R_bw, t_bw, R_wm, t_wm, R_dm, t_dm)
    assert np.allclose(R_out, R_BC, atol=1e-12)
    assert np.allclose(t_out, t_BC, atol=1e-12)
