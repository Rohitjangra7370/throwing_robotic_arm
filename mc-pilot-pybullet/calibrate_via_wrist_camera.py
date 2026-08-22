"""
Chain the D435i's base-frame extrinsic through the arm's OWN wrist camera + FK,
bypassing the tape-measured board placement entirely.

WHY THIS EXISTS
----------------
`calibrate_camera_extrinsics.py` needs a human-measured board pose
(`--corner_xyz`, `--board_x_dir`, `--board_y_dir`) in the base frame -- exactly
the kind of manually-eyeballed geometry this project keeps getting bitten by
(see CLAUDE.md's frame-convention warnings). If the SAME board is also visible
to the arm's own wrist camera, we get the board's base-frame pose a second,
independent way that needs no tape measure at all:

    T_base_marker = T_base_EE (Kortex forward kinematics, trusted)
                  @ T_EE_wristcam (fixed URDF offset, factory spec)
                  @ T_wristcam_marker (solvePnP off the wrist camera image)

...and then the D435i's pose drops out with no placement guess in the chain:

    T_base_D435i = T_base_marker @ inv(T_D435i_marker)

WHAT THIS DOES **NOT** MAKE PERFECT
-------------------------------------
- `T_EE_wristcam` is Kinova's factory-nominal offset from the URDF
  (`camera_color_frame`: xyz=(0, 0.05639, -0.00305), rpy=(pi,pi,0) from
  `end_effector_link`) -- never independently verified against this physical
  unit.
- The wrist camera's intrinsics come from `VisionConfig.GetIntrinsicParameters`
  live off the device, which is trustworthy, but this is the first time this
  codebase has ever pulled an RGB frame from the arm's onboard camera or used
  its RTSP stream -- treat the capture path itself as unverified.
- `GetMeasuredCartesianPose`'s theta_x/y/z are assumed to be the Kinova/
  ros_kortex convention: intrinsic XYZ Euler in degrees, i.e.
  `R = Rx(tx) @ Ry(ty) @ Rz(tz)` (scipy `Rotation.from_euler('xyz', ..., degrees=True)`).
  This has never been used anywhere else in this codebase (everything else
  plans in joint space), so it is unverified here too. This script prints
  the independent tape-measured result (if given) side by side specifically
  so a large disagreement is visible rather than silently trusted.

Reads only: `GetMeasuredCartesianPose`, `VisionConfig.GetIntrinsicParameters`,
`DeviceManager.ReadAllDevices`, and an RTSP frame pull. No command is sent to
the arm; it must already be holding the pose you want measured.

    python3 calibrate_via_wrist_camera.py --ip 192.168.1.101 \\
        --compare_json /path/to/camera_extrinsics2.json
"""

import argparse
import json
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation

sys.path.append("..")
from robot_arm.kinova_hardware import _KortexBackend, _patch_collections_abc
from perception.ray_plane import D435I_COLOR_1280x720, D435I_COLOR_1920x1080

# URDF fixed offset, end_effector_link -> camera_color_frame (robot_arm/_urdf_cache/gen3*.urdf)
T_EE_CAM_XYZ = np.array([0.0, 0.05639, -0.00305])
T_EE_CAM_RPY_DEG = np.array([180.0, 180.0, 0.0])

DICT = cv2.aruco.DICT_4X4_50
MIN_CORNERS = 6


def _rt(xyz, rpy_deg, order="xyz"):
    R = Rotation.from_euler(order, rpy_deg, degrees=True).as_matrix()
    return R, np.asarray(xyz, float)


def _compose(R1, t1, R2, t2):
    """(R1,t1) applied after (R2,t2): p -> R1 @ (R2 @ p + t2) + t1."""
    return R1 @ R2, R1 @ t2 + t1


def _invert(R, t):
    Ri = R.T
    return Ri, -Ri @ t


def _detect_board(gray, board, K, dist, label):
    detector = cv2.aruco.CharucoDetector(board)
    ch_corners, ch_ids, mk_corners, mk_ids = detector.detectBoard(gray)
    n = 0 if ch_corners is None else len(ch_corners)
    print(f"[{label}] {0 if mk_ids is None else len(mk_ids)} markers, {n} charuco corners")
    if n < MIN_CORNERS:
        raise SystemExit(f"[{label}] only {n} corners (need >= {MIN_CORNERS}); aborting")
    obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise SystemExit(f"[{label}] solvePnP failed")
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    err = float(np.linalg.norm(proj.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1).mean())
    print(f"[{label}] reprojection error: {err:.2f} px")
    R_cam_marker, _ = cv2.Rodrigues(rvec)
    return R_cam_marker, tvec.flatten(), err


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default="192.168.1.101")
    ap.add_argument("--username", default="admin")
    ap.add_argument("--password", default="admin")
    ap.add_argument("--squares_x", type=int, default=5)
    ap.add_argument("--squares_y", type=int, default=7)
    ap.add_argument("--square_mm", type=float, default=35.0)
    ap.add_argument("--marker_ratio", type=float, default=0.75)
    ap.add_argument("--d435i_width", type=int, default=1920)
    ap.add_argument("--d435i_height", type=int, default=1080)
    ap.add_argument("--rtsp_url", default=None,
                    help="override; default rtsp://<ip>/color (Kinova convention, unverified here)")
    ap.add_argument("--compare_json", default=None,
                    help="an earlier camera_extrinsics*.json to sanity-check against")
    ap.add_argument("--out", default="camera_extrinsics_via_arm.json")
    args = ap.parse_args()

    _patch_collections_abc()
    from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
    from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient
    from kortex_api.autogen.messages import VisionConfig_pb2, DeviceConfig_pb2

    backend = _KortexBackend(n_dofs=7, ip=args.ip, username=args.username, password=args.password)
    backend.connect()   # zero writes: TCP session open only
    base = backend._base
    router = backend._router

    # --- 1. EE pose, base frame, from Kortex forward kinematics -------------
    pose = base.GetMeasuredCartesianPose()
    R_base_ee, t_base_ee = _rt([pose.x, pose.y, pose.z],
                               [pose.theta_x, pose.theta_y, pose.theta_z])
    print(f"EE pose (base frame): x={pose.x:.4f} y={pose.y:.4f} z={pose.z:.4f}  "
          f"theta=({pose.theta_x:.1f},{pose.theta_y:.1f},{pose.theta_z:.1f}) deg")

    # --- 2. Wrist camera intrinsics, live off the device --------------------
    dm = DeviceManagerClient(router)
    devices = dm.ReadAllDevices()
    vision_ids = [d.device_identifier for d in devices.device_handle
                  if d.device_type == DeviceConfig_pb2.DeviceTypes.Value("VISION")]
    if not vision_ids:
        raise SystemExit("no VISION device found via DeviceManager -- is the module present/enabled?")
    vision_device_id = vision_ids[0]
    print(f"vision device id: {vision_device_id}")

    vc = VisionConfigClient(router)
    sid = VisionConfig_pb2.SensorIdentifier()
    sid.sensor = VisionConfig_pb2.SENSOR_COLOR
    intr_msg = vc.GetIntrinsicParameters(sid, vision_device_id)
    K_wrist = np.array([[intr_msg.focal_length_x, 0, intr_msg.principal_point_x],
                        [0, intr_msg.focal_length_y, intr_msg.principal_point_y],
                        [0, 0, 1]])
    dist_wrist = np.array(list(intr_msg.distortion_coeffs.k)) if hasattr(intr_msg.distortion_coeffs, "k") \
        else np.zeros(5)
    print(f"wrist cam intrinsics (live): fx={intr_msg.focal_length_x:.2f} fy={intr_msg.focal_length_y:.2f} "
          f"cx={intr_msg.principal_point_x:.2f} cy={intr_msg.principal_point_y:.2f} "
          f"res={intr_msg.resolution}")

    # --- 3. Capture wrist camera frame over RTSP -----------------------------
    rtsp = args.rtsp_url or f"rtsp://{args.ip}/color"
    print(f"opening {rtsp} ...")
    cap = cv2.VideoCapture(rtsp)
    if not cap.isOpened():
        raise SystemExit(f"could not open {rtsp} -- RTSP path unverified on this codebase, "
                         f"check the URL / that the color stream is enabled")
    frame_wrist = None
    for _ in range(30):   # let the stream settle
        ok, frame_wrist = cap.read()
        if not ok:
            time.sleep(0.05)
    cap.release()
    if frame_wrist is None:
        raise SystemExit("no frame received from wrist camera RTSP stream")
    gray_wrist = cv2.cvtColor(frame_wrist, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("/tmp/wrist_cam_frame.jpg", frame_wrist)
    print(f"wrist frame {frame_wrist.shape[1]}x{frame_wrist.shape[0]} -> /tmp/wrist_cam_frame.jpg")

    # --- 4. Capture D435i frame (laptop) at the same time --------------------
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.d435i_width, args.d435i_height, rs.format.bgr8, 30)
    pipe.start(cfg)
    try:
        for _ in range(40):
            pipe.wait_for_frames(2000)
        frame_d435i = np.asanyarray(pipe.wait_for_frames(2000).get_color_frame().get_data())
    finally:
        pipe.stop()
    gray_d435i = cv2.cvtColor(frame_d435i, cv2.COLOR_BGR2GRAY)
    intr_d435i = D435I_COLOR_1920x1080 if (args.d435i_width, args.d435i_height) == (1920, 1080) \
        else D435I_COLOR_1280x720

    # --- 5. Detect board in both frames, solvePnP -----------------------------
    d = cv2.aruco.getPredefinedDictionary(DICT)
    board = cv2.aruco.CharucoBoard((args.squares_x, args.squares_y),
                                   args.square_mm / 1000.0,
                                   args.square_mm * args.marker_ratio / 1000.0, d)

    R_wrist_marker, t_wrist_marker, err_wrist = _detect_board(
        gray_wrist, board, K_wrist, dist_wrist, "wrist")
    R_d435i_marker, t_d435i_marker, err_d435i = _detect_board(
        gray_d435i, board, intr_d435i.K, np.zeros(5), "D435i")

    # --- 6. Chain: base -> EE -> wrist_cam -> marker --------------------------
    R_ee_cam, t_ee_cam = _rt(T_EE_CAM_XYZ, T_EE_CAM_RPY_DEG)
    R_base_cam, t_base_cam = _compose(R_base_ee, t_base_ee, R_ee_cam, t_ee_cam)
    R_base_marker, t_base_marker = _compose(R_base_cam, t_base_cam, R_wrist_marker, t_wrist_marker)

    # base -> D435i = base->marker  @  inv(D435i->marker)
    R_marker_d435i, t_marker_d435i = _invert(R_d435i_marker, t_d435i_marker)
    R_base_d435i, t_base_d435i = _compose(R_base_marker, t_base_marker, R_marker_d435i, t_marker_d435i)

    print("\n--- T_base_marker (via arm FK + wrist cam, no tape measure) ---")
    print(f"position (m): {t_base_marker}")
    print("\n--- T_base_D435i (chained through the arm) ---")
    print(f"position (m): {t_base_d435i}")
    print(f"R =\n{R_base_d435i}")

    if args.compare_json:
        with open(args.compare_json) as f:
            prior = json.load(f)
        t_prior = np.array(prior["t_B_C"])
        R_prior = np.array(prior["R_B_C"])
        dt = np.linalg.norm(t_base_d435i - t_prior)
        dR_deg = np.degrees(np.arccos(np.clip((np.trace(R_base_d435i @ R_prior.T) - 1) / 2, -1, 1)))
        print(f"\n--- comparison against {args.compare_json} (tape-measured chain) ---")
        print(f"position disagreement: {dt*100:.1f} cm")
        print(f"rotation disagreement: {dR_deg:.1f} deg")
        print("=> " + ("AGREES -- both chains support this result" if dt < 0.03 and dR_deg < 5
                        else "DISAGREES -- do not trust either number until this is reconciled"))

    result = {
        "R_B_C": R_base_d435i.tolist(), "t_B_C": t_base_d435i.tolist(),
        "method": "via_wrist_camera_and_FK",
        "T_base_marker": {"R": R_base_marker.tolist(), "t": t_base_marker.tolist()},
        "reprojection_error_px": {"wrist": err_wrist, "d435i": err_d435i},
        "ee_pose_raw": {"x": pose.x, "y": pose.y, "z": pose.z,
                        "theta_x": pose.theta_x, "theta_y": pose.theta_y, "theta_z": pose.theta_z},
        "assumed_euler_convention": "intrinsic xyz degrees, R = Rx@Ry@Rz -- UNVERIFIED, see docstring",
        "T_EE_wristcam_source": "URDF camera_module joint, factory nominal, UNVERIFIED on this unit",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
