"""
Solve the D435i-to-arm-base extrinsic T_B_C from one live ChArUco frame.

WHAT THIS SOLVES
-----------------
`perception/ray_plane.py` needs R, t such that p_B = R @ p_C + t (camera pose
in the base frame). Everything else -- intrinsics, the ray-plane math, the
ball-radius correction -- already exists; only T_B_C was ever unknown. This
script measures it: detect the ChArUco board `make_aruco_targets.py` /
`make_aruco_printable.py` prints, solvePnP for the board's pose in the CAMERA
frame, then compose with the board's known pose in the BASE frame (where you
physically put it) to get the camera's pose in the base frame.

THE BOARD-TO-BASE POSE IS A MEASUREMENT, NOT A CONVENTION
-----------------------------------------------------------
The board's local origin is the bottom-left chessboard corner (per
`make_aruco_printable.py`'s printed note), local +X runs along columns
(the `squares_x` / short-by-default direction), local +Y along rows, local
+Z out of the printed face. Lying flat with the pattern face up, only a
yaw rotation about base +Z is physically possible -- so orientation is given
as two direction vectors (`--board_x_dir`, `--board_y_dir`) rather than a
single implicit convention, since this project has repeatedly been burned by
implicit frame conventions (see CLAUDE.md). Get `--corner_xyz` and the two
direction vectors from an actual tape measurement of where the origin corner
sits and which way the board points -- do not guess them here.

QUALITY GATE
------------
Below ~35 px marker edge, ArUco/ChArUco corners degrade (see
`cam_snapshot.py`). This script reports detected corner count and mean
marker edge size and REFUSES to write a result file below a minimum corner
count, but does not refuse on edge size alone -- read the printed verdict
before trusting the output for anything beyond a rough first pass.

    python3 calibrate_camera_extrinsics.py \\
        --corner_xyz 1.10 0.0875 -0.433 \\
        --board_x_dir 0 1 0 --board_y_dir 1 0 0
"""

import argparse
import json
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from perception.ray_plane import D435I_COLOR_1280x720, D435I_COLOR_1920x1080

MIN_CORNERS = 6


def _capture(width, height, settle, exposure):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    pipe.start(cfg)
    try:
        prof = pipe.get_active_profile()
        if exposure is not None:
            for s in prof.get_device().query_sensors():
                if s.supports(rs.option.exposure) and "RGB" in s.get_info(rs.camera_info.name):
                    s.set_option(rs.option.enable_auto_exposure, 0)
                    s.set_option(rs.option.exposure, exposure)
        for _ in range(settle):
            pipe.wait_for_frames(2000)
        img = np.asanyarray(pipe.wait_for_frames(2000).get_color_frame().get_data())
    finally:
        pipe.stop()
    return img


def _board_to_base(corner_xyz, x_dir, y_dir):
    """R_bb, t_bb such that p_base = R_bb @ p_board + t_bb."""
    ex = np.asarray(x_dir, float)
    ex /= np.linalg.norm(ex)
    ey = np.asarray(y_dir, float)
    ey = ey - ex * (ex @ ey)          # orthogonalize against x, don't trust exact input
    n = np.linalg.norm(ey)
    if n < 1e-6:
        raise ValueError("--board_x_dir and --board_y_dir are parallel")
    ey /= n
    ez = np.cross(ex, ey)
    R_bb = np.stack([ex, ey, ez], axis=1)   # columns = board axes expressed in base frame
    return R_bb, np.asarray(corner_xyz, float)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corner_xyz", type=float, nargs=3, required=True,
                    help="board's local origin (bottom-left chessboard corner), "
                         "metres, in the arm base frame")
    ap.add_argument("--board_x_dir", type=float, nargs=3, required=True,
                    help="unit-ish direction board local +X (short/column axis) "
                         "points in the base frame, e.g. 0 1 0")
    ap.add_argument("--board_y_dir", type=float, nargs=3, required=True,
                    help="unit-ish direction board local +Y (long/row axis) "
                         "points in the base frame, e.g. 1 0 0")
    ap.add_argument("--squares_x", type=int, default=5)
    ap.add_argument("--squares_y", type=int, default=7)
    ap.add_argument("--square_mm", type=float, default=35.0)
    ap.add_argument("--marker_ratio", type=float, default=0.75)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--settle", type=int, default=40)
    ap.add_argument("--exposure", type=float, default=None)
    ap.add_argument("--out", default="camera_extrinsics.json")
    ap.add_argument("--annotate_out", default="/tmp/calib_annotated.jpg")
    args = ap.parse_args()

    if (args.width, args.height) == (1920, 1080):
        intr = D435I_COLOR_1920x1080
    elif (args.width, args.height) == (1280, 720):
        intr = D435I_COLOR_1280x720
    else:
        raise SystemExit(f"no vetted intrinsics for {args.width}x{args.height}; "
                          f"use 1920x1080 or 1280x720 (perception/ray_plane.py)")

    img = _capture(args.width, args.height, args.settle, args.exposure)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        args.square_mm / 1000.0, args.square_mm * args.marker_ratio / 1000.0, d)
    detector = cv2.aruco.CharucoDetector(board)
    ch_corners, ch_ids, mk_corners, mk_ids = detector.detectBoard(gray)

    n_corners = 0 if ch_corners is None else len(ch_corners)
    n_markers = 0 if mk_ids is None else len(mk_ids)
    print(f"detected: {n_markers} ArUco markers, {n_corners} ChArUco corners")
    if mk_corners:
        edges = [float(np.mean([np.linalg.norm(c[0][k] - c[0][(k + 1) % 4])
                                for k in range(4)])) for c in mk_corners]
        print(f"marker edge size: mean {np.mean(edges):.1f} px "
              f"({'robust' if np.mean(edges) >= 60 else 'marginal' if np.mean(edges) >= 35 else 'TOO SMALL -- move closer'})")

    annotated = img.copy()
    if mk_ids is not None:
        cv2.aruco.drawDetectedMarkers(annotated, mk_corners, mk_ids)
    if ch_corners is not None and n_corners:
        cv2.aruco.drawDetectedCornersCharuco(annotated, ch_corners, ch_ids)
    cv2.imwrite(args.annotate_out, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"annotated frame -> {args.annotate_out}")

    if n_corners < MIN_CORNERS:
        raise SystemExit(f"only {n_corners} ChArUco corners detected (need >= {MIN_CORNERS}); "
                          f"move the camera closer / improve lighting and retry. "
                          f"No result written.")

    obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
    K = intr.K
    dist = np.zeros(5)   # factory-rectified colour stream, see perception/ray_plane.py
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise SystemExit("solvePnP failed to converge")

    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    reproj_err = float(np.linalg.norm(proj.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1).mean())
    print(f"solvePnP reprojection error: {reproj_err:.2f} px "
          f"({'good' if reproj_err < 1.0 else 'high -- treat the result with suspicion'})")

    R_cv, _ = cv2.Rodrigues(rvec)          # p_cam = R_cv @ p_board + tvec
    R_bb, t_bb = _board_to_base(args.corner_xyz, args.board_x_dir, args.board_y_dir)

    R_B_C = R_bb @ R_cv.T                  # camera->base rotation
    t_B_C = t_bb - R_bb @ R_cv.T @ tvec.flatten()   # camera origin in base frame

    print("\n--- T_B_C : camera pose in the arm base frame ---")
    print(f"camera position (m): x={t_B_C[0]:.4f}  y={t_B_C[1]:.4f}  z={t_B_C[2]:.4f}")
    print(f"R =\n{R_B_C}")
    floor_z = args.corner_xyz[2]
    print(f"camera height above the board's plane: {t_B_C[2] - floor_z:.3f} m "
          f"({'plausible for a laptop-held camera' if 0.15 < t_B_C[2] - floor_z < 2.5 else 'CHECK SIGNS -- this looks wrong'})")

    result = {
        "R_B_C": R_B_C.tolist(),
        "t_B_C": t_B_C.tolist(),
        "reprojection_error_px": reproj_err,
        "n_charuco_corners": int(n_corners),
        "n_aruco_markers": int(n_markers),
        "intrinsics_wh": [args.width, args.height],
        "board": {"squares_x": args.squares_x, "squares_y": args.squares_y,
                  "square_mm": args.square_mm, "marker_ratio": args.marker_ratio},
        "corner_xyz_input": list(args.corner_xyz),
        "board_x_dir_input": list(args.board_x_dir),
        "board_y_dir_input": list(args.board_y_dir),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "D435i attached to a laptop, not the final fixed mount -- re-run "
                "this once the camera is on its permanent rig.",
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
