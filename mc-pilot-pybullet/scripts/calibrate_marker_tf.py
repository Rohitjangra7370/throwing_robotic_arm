"""
Calibrate the D435i-to-arm-base extrinsic from ONE static ArUco marker both
the depth camera and the arm's own wrist camera can see at the same time, and
save it where perception.base_frame's "auto tf" picks it up automatically.

SAME METHOD AS calibrate_via_wrist_camera.py, ONE MARKER INSTEAD OF A BOARD
-----------------------------------------------------------------------------
That script chains base -> EE (Kortex forward kinematics) -> wrist camera
(fixed URDF offset) -> marker (solvePnP), which needs no tape-measured board
placement -- see feedback_extrinsic_calibration_method in project memory for
why this is the preferred method over `calibrate_camera_extrinsics.py`'s
tape-measured chain. This script reuses that exact composition math
(`_rt`/`_compose`/`_invert`, imported, not copied) and only swaps the target:
one already-printed ArUco tag (make_aruco_targets.py's loose `markers/`
output) in place of the ChArUco board, since that is what is actually sitting
in the lab as a static shared fiducial. A single marker's pose is noisier
than a ChArUco board's (4 corners vs. dozens), so prefer the board via the
other script when accuracy matters more than setup simplicity.

    T_base_marker = T_base_EE  @  T_EE_wristcam  @  T_wristcam_marker
    T_base_D435i  = T_base_marker  @  inv(T_D435i_marker)

WHAT THIS DOES NOT MAKE PERFECT
--------------------------------
Same caveats as calibrate_via_wrist_camera.py: T_EE_wristcam is Kinova's
factory-nominal URDF offset (never independently verified on this unit), and
GetMeasuredCartesianPose's Euler convention is assumed intrinsic-XYZ degrees
(confirmed to 1.23 deg once, see project memory, but not re-verified here).

Reads only -- GetMeasuredCartesianPose, VisionConfig.GetIntrinsicParameters,
an RTSP frame, a D435i colour frame. No command is sent to the arm; it must
already be holding a pose where its wrist camera sees the marker.

    python3 scripts/calibrate_marker_tf.py --ip 192.168.1.101 --marker_id 0 \\
        --marker_length_m 0.080
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../mc-pilot-pybullet
sys.path.insert(0, ROOT)

import calibrate_via_wrist_camera as wristcal  # reuse the proven composition math
from perception.ray_plane import D435I_COLOR_1280x720, D435I_COLOR_1920x1080

DICT = cv2.aruco.DICT_4X4_50  # matches make_aruco_targets.py
CALIB_DIR = os.path.join(ROOT, "calib")


def _detect_marker(gray, marker_id, marker_length_m, K, dist, label):
    """One image -> (R_cam_marker, t_cam_marker, reprojection_error_px)."""
    d = cv2.aruco.getPredefinedDictionary(DICT)
    detector = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    seen = [] if ids is None else ids.flatten().tolist()
    print(f"[{label}] saw marker ids: {seen}")
    if ids is None or marker_id not in seen:
        raise SystemExit(f"[{label}] marker id {marker_id} not detected (saw {seen}) -- "
                         f"make sure it is in view, flat, well lit, and not motion-blurred")
    idx = seen.index(marker_id)
    img_pts = corners[idx].reshape(4, 2).astype(np.float64)

    # Standard OpenCV convention: marker-frame origin at its centre, corners
    # ordered top-left/top-right/bottom-right/bottom-left in that plane.
    h = marker_length_m / 2.0
    obj_pts = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        raise SystemExit(f"[{label}] solvePnP failed")
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    err = float(np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1).mean())
    print(f"[{label}] reprojection error: {err:.2f} px")
    R_cam_marker, _ = cv2.Rodrigues(rvec)
    return R_cam_marker, tvec.flatten(), err


def main():
    import pyrealsense2 as rs

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default="192.168.1.101")
    ap.add_argument("--username", default="admin")
    ap.add_argument("--password", default="admin")
    ap.add_argument("--marker_id", type=int, required=True)
    ap.add_argument("--marker_length_m", type=float, required=True,
                    help="measured (not nominal) printed marker edge, metres -- "
                         "see make_aruco_targets.py's print-scale warning")
    ap.add_argument("--d435i_width", type=int, default=1920)
    ap.add_argument("--d435i_height", type=int, default=1080)
    ap.add_argument("--rtsp_url", default=None)
    ap.add_argument("--out_name", default=None,
                    help="basename (no ext) for the .npz/.json pair under calib/; "
                         "default is a timestamp")
    args = ap.parse_args()

    wristcal._patch_collections_abc()
    from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
    from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient
    from kortex_api.autogen.messages import DeviceConfig_pb2, VisionConfig_pb2

    backend = wristcal._KortexBackend(n_dofs=7, ip=args.ip, username=args.username,
                                      password=args.password)
    backend.connect()  # zero writes: TCP session open only
    base = backend._base
    router = backend._router

    # --- 1. EE pose, base frame, from Kortex forward kinematics -------------
    pose = base.GetMeasuredCartesianPose()
    R_base_ee, t_base_ee = wristcal._rt([pose.x, pose.y, pose.z],
                                        [pose.theta_x, pose.theta_y, pose.theta_z])
    print(f"EE pose (base frame): x={pose.x:.4f} y={pose.y:.4f} z={pose.z:.4f}  "
          f"theta=({pose.theta_x:.1f},{pose.theta_y:.1f},{pose.theta_z:.1f}) deg")

    # --- 2. Wrist camera intrinsics, live off the device ---------------------
    dm = DeviceManagerClient(router)
    devices = dm.ReadAllDevices()
    vision_ids = [d.device_identifier for d in devices.device_handle
                 if d.device_type == DeviceConfig_pb2.DeviceTypes.Value("VISION")]
    if not vision_ids:
        raise SystemExit("no VISION device found via DeviceManager")
    vc = VisionConfigClient(router)
    sid = VisionConfig_pb2.SensorIdentifier()
    sid.sensor = VisionConfig_pb2.SENSOR_COLOR
    intr_msg = vc.GetIntrinsicParameters(sid, vision_ids[0])
    K_wrist = np.array([[intr_msg.focal_length_x, 0, intr_msg.principal_point_x],
                        [0, intr_msg.focal_length_y, intr_msg.principal_point_y],
                        [0, 0, 1]])
    dist_wrist = (np.array(list(intr_msg.distortion_coeffs.k))
                 if hasattr(intr_msg.distortion_coeffs, "k") else np.zeros(5))
    print(f"wrist cam intrinsics (live): fx={intr_msg.focal_length_x:.2f} "
          f"fy={intr_msg.focal_length_y:.2f} res={intr_msg.resolution}")

    # --- 3. Capture wrist camera frame over RTSP ------------------------------
    rtsp = args.rtsp_url or f"rtsp://{args.ip}/color"
    print(f"opening {rtsp} ...")
    cap = cv2.VideoCapture(rtsp)
    if not cap.isOpened():
        raise SystemExit(f"could not open {rtsp}")
    frame_wrist = None
    for _ in range(30):
        ok, frame_wrist = cap.read()
        if not ok:
            time.sleep(0.05)
    cap.release()
    if frame_wrist is None:
        raise SystemExit("no frame received from wrist camera RTSP stream")
    gray_wrist = cv2.cvtColor(frame_wrist, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("/tmp/wrist_cam_frame.jpg", frame_wrist)

    # --- 4. Capture D435i frame at the same time ------------------------------
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
    cv2.imwrite("/tmp/d435i_marker_frame.jpg", frame_d435i)
    intr_d435i = (D435I_COLOR_1920x1080 if (args.d435i_width, args.d435i_height) == (1920, 1080)
                 else D435I_COLOR_1280x720)

    # --- 5. Detect the SAME marker in both frames -----------------------------
    R_wrist_marker, t_wrist_marker, err_wrist = _detect_marker(
        gray_wrist, args.marker_id, args.marker_length_m, K_wrist, dist_wrist, "wrist")
    R_d435i_marker, t_d435i_marker, err_d435i = _detect_marker(
        gray_d435i, args.marker_id, args.marker_length_m, intr_d435i.K, np.zeros(5), "D435i")

    # --- 6. Chain: base -> EE -> wrist_cam -> marker, then invert to D435i ----
    R_ee_cam, t_ee_cam = wristcal._rt(wristcal.T_EE_CAM_XYZ, wristcal.T_EE_CAM_RPY_DEG)
    R_base_cam, t_base_cam = wristcal._compose(R_base_ee, t_base_ee, R_ee_cam, t_ee_cam)
    R_base_marker, t_base_marker = wristcal._compose(
        R_base_cam, t_base_cam, R_wrist_marker, t_wrist_marker)

    R_marker_d435i, t_marker_d435i = wristcal._invert(R_d435i_marker, t_d435i_marker)
    R_base_d435i, t_base_d435i = wristcal._compose(
        R_base_marker, t_base_marker, R_marker_d435i, t_marker_d435i)

    print("\n--- T_base_D435i (via one shared ArUco marker + arm FK) ---")
    print(f"position (m): {t_base_d435i}")
    print(f"R =\n{R_base_d435i}")

    os.makedirs(CALIB_DIR, exist_ok=True)
    name = args.out_name or time.strftime("%Y%m%d_%H%M%S")
    npz_path = os.path.join(CALIB_DIR, f"{name}.npz")
    json_path = os.path.join(CALIB_DIR, f"{name}.json")
    canonical_path = os.path.join(CALIB_DIR, "T_B_C.npz")

    np.savez(npz_path, R=R_base_d435i, t=t_base_d435i)
    np.savez(canonical_path, R=R_base_d435i, t=t_base_d435i)
    with open(json_path, "w") as f:
        json.dump({
            "R_B_C": R_base_d435i.tolist(), "t_B_C": t_base_d435i.tolist(),
            "method": "single_marker_via_wrist_camera_and_FK",
            "marker_id": args.marker_id, "marker_length_m": args.marker_length_m,
            "reprojection_error_px": {"wrist": err_wrist, "d435i": err_d435i},
            "ee_pose_raw": {"x": pose.x, "y": pose.y, "z": pose.z,
                            "theta_x": pose.theta_x, "theta_y": pose.theta_y,
                            "theta_z": pose.theta_z},
            "assumed_euler_convention": "intrinsic xyz degrees, R = Rx@Ry@Rz -- see docstring",
            "T_EE_wristcam_source": "URDF camera_module joint, factory nominal, UNVERIFIED",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)

    print(f"\nwrote {npz_path}")
    print(f"wrote {json_path}  (audit trail: reprojection error, raw EE pose, method)")
    print(f"wrote {canonical_path}  <- perception.base_frame.auto_to_base reads THIS one")
    print(f"\nsingle-marker PnP (4 points) is noisier than the ChArUco board's dozens of "
          f"corners -- cross-check with calibrate_via_wrist_camera.py --compare_json "
          f"{json_path} if you have a board reading to compare against.")


if __name__ == "__main__":
    main()
