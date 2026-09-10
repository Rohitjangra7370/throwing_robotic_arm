"""
The one implementation of the wrist-camera calibration chain.

WHAT THIS IS FOR
-----------------
`T_B_C` (D435i pose in the arm's base frame) is calibrated by looking at one
fiducial from two places at once: the arm's own wrist camera, whose pose in the
base frame the robot knows, and the D435i, whose pose is what we are solving
for. The board drops out:

    T_base_board = T_base_wristcam @ T_wristcam_board     (arm knows where it is)
    T_base_D435i = T_base_board    @ inv(T_D435i_board)

No tape measure enters anywhere, which is the entire point -- see
`feedback_extrinsic_calibration_method` in project memory.

WHY `base_to_wrist_camera` USES FK AND NOT `GetMeasuredCartesianPose`
----------------------------------------------------------------------
Found on the arm 2026-08-31. `Base.GetMeasuredCartesianPose()` reports the
**TOOL** frame, not the flange: this arm has `tool_transform = (0, 0, 0.12) m`
configured for the Robotiq 2F-85, and the reported pose measured 0.1251 m along
flange +z from PyBullet FK's flange. Both calibration scripts were composing the
URDF's *flange*->camera offset onto that tool pose, i.e. building the camera
pose 12 cm out of place, and feeding it straight into `T_B_C`.

Measured, with the board physically lying on the floor (base-frame z = -0.433):

    Kortex TOOL pose @ flange->cam    board came out at z = -0.5728  (-14.0 cm)
    tool transform removed            board came out at z = -0.4588  ( -2.6 cm)
    FK to camera_color_frame          board came out at z = -0.4453  ( -1.2 cm)

So this module goes from measured JOINT ANGLES through PyBullet FK on the same
URDF the throw planner already uses. That removes three assumptions at once: the
tool transform, `GetMeasuredCartesianPose`'s Euler convention (assumed intrinsic
XYZ degrees, never verified), and the hardcoded flange->camera offset. Any URDF
error that remains is common-mode with the planner it is calibrating for.

The 1.2 cm residual is itself the cross-check: nothing told this chain where the
floor was, and it put a board lying on the floor 1.2 cm from it -- which also
independently confirms the 0.433 m base height the repo had only ever asserted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation

__all__ = [
    "ARUCO_DICT", "CAMERA_COLOR_LINK", "FLANGE_LINK", "BoardDetection",
    "rt", "compose", "invert", "board_from_spec", "base_to_wrist_camera",
    "detect_board_pose", "chain_base_to_camera", "average_transforms",
    "floor_plane_check", "drift_check", "save_extrinsic",
]

ARUCO_DICT = cv2.aruco.DICT_4X4_50      # matches make_aruco_targets.py
FLANGE_LINK = 7                          # end_effector_link, gen3.urdf
CAMERA_COLOR_LINK = 10                   # camera_color_frame, gen3.urdf
_GEN3_URDF = "kinova_gen3/gen3.urdf"
_MIN_CORNERS = 6


# ---------------------------------------------------------------------------
# rigid transform primitives.  Convention throughout: (R, t) maps a point in the
# child frame to the parent frame, p_parent = R @ p_child + t -- the same
# convention perception/base_frame.py loads and measure_landing.py consumes.
# ---------------------------------------------------------------------------
def rt(xyz, rpy_deg, order="xyz"):
    return Rotation.from_euler(order, rpy_deg, degrees=True).as_matrix(), np.asarray(xyz, float)


def compose(R1, t1, R2, t2):
    """(R1,t1) applied after (R2,t2): p -> R1 @ (R2 @ p + t2) + t1."""
    return R1 @ R2, R1 @ t2 + t1


def invert(R, t):
    Ri = R.T
    return Ri, -Ri @ t


# ---------------------------------------------------------------------------
# arm side
# ---------------------------------------------------------------------------
def base_to_wrist_camera(q, cam_link: int = CAMERA_COLOR_LINK, urdf_rel_path: str = _GEN3_URDF):
    """
    Measured joint angles -> (R, t) of the wrist colour camera in the base frame.

    `q` is the 7 joint angles in radians as `read_joint_state()` returns them
    (wrapped to (-pi, pi]; the continuous joints make that harmless for FK).

    Pass `cam_link=FLANGE_LINK` to get the flange instead -- used by the tests
    to pin the offset, not by the calibration itself.
    """
    q = np.asarray(q, float).reshape(-1)
    if q.size != 7:
        raise ValueError(f"expected 7 joint angles for the Gen3, got {q.size}")

    cid = p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
        rid = p.loadURDF(os.path.join(pybullet_data.getDataPath(), urdf_rel_path),
                         basePosition=[0, 0, 0], useFixedBase=True, physicsClientId=cid)
        for j, ang in enumerate(q):
            p.resetJointState(rid, j, float(ang), physicsClientId=cid)
        st = p.getLinkState(rid, cam_link, computeForwardKinematics=True, physicsClientId=cid)
        R = np.array(p.getMatrixFromQuaternion(st[5])).reshape(3, 3)
        t = np.array(st[4], float)
    finally:
        p.disconnect(cid)
    return R, t


# ---------------------------------------------------------------------------
# board side
# ---------------------------------------------------------------------------
@dataclass
class BoardDetection:
    R: np.ndarray                 # board pose in the camera frame
    t: np.ndarray
    n_corners: int
    reproj_px: float
    distance_m: float
    dropped_ids: List[int] = field(default_factory=list)
    seen_ids: List[int] = field(default_factory=list)


def board_from_spec(squares_x: int, squares_y: int, square_mm: float, marker_ratio: float):
    return cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_mm / 1000.0, square_mm * marker_ratio / 1000.0,
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT))


def detect_board_pose(gray, board, K, dist, min_corners: int = _MIN_CORNERS) -> BoardDetection:
    """
    Grayscale image -> board pose in the camera frame.

    Detects markers first and filters before interpolating, for one reason:
    a marker id that appears TWICE in frame makes
    `CharucoDetector.detectBoard` return zero chessboard corners, which surfaces
    as "board not visible" rather than "ambiguous id". That is not hypothetical
    -- the overhead D435i sees the board *and* a loose id=1 ArUco taped to a box
    beside it (the check-point markers make_aruco_targets.py tells you to lay
    out), and it reported 15 markers / 0 corners until this filter existed.

    Both copies of a duplicated id are dropped: which one belongs to the board
    is exactly what cannot be known from the id alone.
    """
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT), cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        raise RuntimeError("no ArUco markers detected at all -- is the board in view, "
                           "flat, lit, and not motion-blurred?")

    ids = np.asarray(ids).flatten()
    board_ids = set(np.asarray(board.getIds()).flatten().tolist())
    counts = {int(i): int((ids == i).sum()) for i in set(ids.tolist())}
    dropped = sorted(i for i, c in counts.items() if c > 1)
    keep = [k for k, i in enumerate(ids) if int(i) in board_ids and counts[int(i)] == 1]
    if not keep:
        raise RuntimeError(
            f"no unambiguous board markers left (saw {sorted(counts)}, "
            f"duplicated {dropped}) -- move the loose check-point markers out of view")

    kept_corners = tuple(corners[k] for k in keep)
    kept_ids = ids[keep].reshape(-1, 1).astype(np.int32)
    ch_corners, ch_ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(
        gray, None, None, kept_corners, kept_ids)
    n = 0 if ch_corners is None else len(ch_corners)
    if n < min_corners:
        raise RuntimeError(f"only {n} chessboard corners (need >= {min_corners}) "
                           f"from {len(keep)} usable markers")

    obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed on the board corners")
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    err = float(np.linalg.norm(proj.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1).mean())
    R, _ = cv2.Rodrigues(rvec)
    t = np.asarray(tvec, float).flatten()
    return BoardDetection(R=R, t=t, n_corners=int(n), reproj_px=err,
                          distance_m=float(np.linalg.norm(t)),
                          dropped_ids=dropped, seen_ids=sorted(counts))


# ---------------------------------------------------------------------------
# the chain, and the gates on it
# ---------------------------------------------------------------------------
def chain_base_to_camera(R_base_wrist, t_base_wrist, R_wrist_board, t_wrist_board,
                         R_d435i_board, t_d435i_board):
    """(base->wristcam, wristcam->board, D435i->board) -> base->D435i."""
    R_bm, t_bm = compose(R_base_wrist, t_base_wrist, R_wrist_board, t_wrist_board)
    return compose(R_bm, t_bm, *invert(R_d435i_board, t_d435i_board))


def average_transforms(transforms):
    """
    [(R, t), ...] from repeated shots of a STATIONARY scene -> (R, t, spread_m, spread_deg).

    Single-shot PnP off one board is noisier than it looks: two back-to-back
    calibrations of an untouched rig disagreed by 1.6 cm / 0.5 deg, which is the
    same order as the drift tolerance meant to detect a knocked camera mount. A
    drift gate cannot see a movement smaller than its own measurement noise, so
    the noise gets averaged down and, more importantly, reported -- an unmeasured
    error term is the thing this project keeps getting caught by.

    Position is a median (robust to a single bad frame); rotation is the proper
    Karcher mean via scipy. Spread is the worst deviation from that centre, not a
    standard deviation, so one bad frame in ten cannot hide.
    """
    Rs = np.stack([np.asarray(R, float) for R, _ in transforms])
    ts = np.stack([np.asarray(t, float) for _, t in transforms])
    t_med = np.median(ts, axis=0)
    R_mean = Rotation.from_matrix(Rs).mean().as_matrix()
    spread_m = float(np.max(np.linalg.norm(ts - t_med, axis=1)))
    cos = np.clip((np.trace(Rs @ R_mean.T, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    spread_deg = float(np.degrees(np.arccos(cos)).max())
    return R_mean, t_med, spread_m, spread_deg


def floor_plane_check(R_base_board, t_base_board, floor_z: float,
                      tol_m: float = 0.02, tol_deg: float = 5.0):
    """
    The board is lying flat on the floor, and we know where the floor is. So its
    calibrated pose has to agree, in height and in tilt.

    This is the gate that catches what a reprojection error cannot: a planar PnP
    absorbs a wrong principal point, a wrong Euler convention or a mis-scaled
    printout into the pose and still reports a fraction of a pixel. Reprojection
    error says the model is self-consistent; only an outside fact says it is
    right. The 12 cm tool-frame bug reprojected at 0.18 px.

    Returns (ok, dz_m, tilt_deg). `dz` is signed: negative = calibrated below the
    real floor.
    """
    normal = np.asarray(R_base_board, float)[:, 2]
    cos = abs(float(normal @ np.array([0.0, 0.0, 1.0])))   # abs: board z sign is a convention
    tilt = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    dz = float(np.asarray(t_base_board, float)[2] - floor_z)
    return (abs(dz) <= tol_m and tilt <= tol_deg), dz, tilt


def drift_check(R_new, t_new, R_old, t_old, tol_m: float = 0.03, tol_deg: float = 5.0):
    """Today's extrinsic vs the stored one. A knocked mount shows up here."""
    dt = float(np.linalg.norm(np.asarray(t_new, float) - np.asarray(t_old, float)))
    cos = (np.trace(np.asarray(R_new, float) @ np.asarray(R_old, float).T) - 1.0) / 2.0
    dR = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    return (dt <= tol_m and dR <= tol_deg), dt, dR


def save_extrinsic(R, t, npz_path):
    """Write (R, t) where perception.base_frame.load_extrinsic expects them."""
    os.makedirs(os.path.dirname(os.path.abspath(npz_path)), exist_ok=True)
    np.savez(npz_path, R=np.asarray(R, float), t=np.asarray(t, float))
    return npz_path
