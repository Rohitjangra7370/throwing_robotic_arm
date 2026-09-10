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

    T_base_marker = T_base_wristcam (PyBullet FK on the measured joint angles,
                                     straight to gen3.urdf's camera_color_frame)
                  @ T_wristcam_marker (solvePnP off the wrist camera image)

...and then the D435i's pose drops out with no placement guess in the chain:

    T_base_D435i = T_base_marker @ inv(T_D435i_marker)

FIXED 2026-08-31: THIS USED TO BUILD THE CHAIN OFF THE TOOL FRAME
-------------------------------------------------------------------
This script previously started from `Base.GetMeasuredCartesianPose()` and
composed the URDF's FLANGE -> camera offset onto it. That call reports the
**TOOL** frame, and this arm has `tool_transform = (0, 0, 0.12) m` configured
for the Robotiq 2F-85, so the camera was placed 12 cm out and the error went
straight into `T_B_C`: the board came out 14.0 cm below the floor it was
physically lying on. It now goes from measured JOINT ANGLES through PyBullet FK
to `camera_color_frame`, which also retires the unverified Euler-convention
assumption that used to live here. See `perception/wrist_chain.py` -- the chain
lives there now, is shared with `scripts/calibrate_marker_tf.py` and
`start_of_day.py`, and is covered by `tests/test_wrist_chain.py`.

`start_of_day.py` is the fuller tool: same chain, plus the floor/scale/drift
gates and a written `calib/T_B_C.npz`. Use this one for a board reading to
cross-check a single-marker calibration against (`--compare_json`).

WHAT THIS STILL DOES NOT MAKE PERFECT
---------------------------------------
- The URDF's camera offset is Kinova's factory nominal, never verified against
  this physical unit. It is now at least self-consistent with the kinematics the
  throw planner uses, since both come from the same URDF.
- Single-shot board PnP repeats to only ~1.8 cm / 0.6 deg (measured, 5 shots,
  stationary rig). `start_of_day.py` averages frames and reports that spread;
  this script does not.

Reads only: `GetMeasuredJointAngles` (via `read_joint_state`),
`VisionConfig.GetIntrinsicParameters`, `DeviceManager.ReadAllDevices`, and an
RTSP frame pull. No command is sent to the arm; it must already be holding the
pose you want measured.

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
from perception.wrist_chain import (ARUCO_DICT as DICT, base_to_wrist_camera,
                                    board_from_spec, chain_base_to_camera, compose,
                                    detect_board_pose, invert, rt, save_extrinsic)

MIN_CORNERS = 6


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
    ap.add_argument("--no_canonical", action="store_true",
                    help="do not also write perception.base_frame's canonical "
                         "calib/T_B_C.npz (which everything downstream reads)")
    args = ap.parse_args()

    _patch_collections_abc()
    from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
    from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient
    from kortex_api.autogen.messages import VisionConfig_pb2, DeviceConfig_pb2

    backend = _KortexBackend(n_dofs=7, ip=args.ip, username=args.username, password=args.password)
    backend.connect()   # zero writes: TCP session open only
    base = backend._base
    router = backend._router

    # --- 1. wrist camera pose, base frame, from measured joints + FK ---------
    # NOT GetMeasuredCartesianPose: that reports the TOOL frame (this arm has a
    # 0.12 m tool_transform for the 2F-85) and composing a flange->camera offset
    # onto it put the board 14 cm underground. See the module docstring.
    q, _ = backend.read_joint_state()
    R_base_cam, t_base_cam = base_to_wrist_camera(q)
    print(f"joint angles (rad): {np.round(q, 5)}")
    print(f"wrist camera in base frame: {np.round(t_base_cam, 4)}")

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
    board = board_from_spec(args.squares_x, args.squares_y,
                            args.square_mm, args.marker_ratio)

    det_wrist = detect_board_pose(gray_wrist, board, K_wrist, dist_wrist, MIN_CORNERS)
    det_d435i = detect_board_pose(gray_d435i, board, intr_d435i.K, np.zeros(5), MIN_CORNERS)
    for label, det in (("wrist", det_wrist), ("D435i", det_d435i)):
        print(f"[{label}] {det.n_corners} corners, {det.reproj_px:.3f} px, "
              f"board at {det.distance_m:.3f} m"
              + (f", dropped duplicated ids {det.dropped_ids}" if det.dropped_ids else ""))
    R_wrist_marker, t_wrist_marker, err_wrist = det_wrist.R, det_wrist.t, det_wrist.reproj_px
    R_d435i_marker, t_d435i_marker, err_d435i = det_d435i.R, det_d435i.t, det_d435i.reproj_px

    # --- 6. Chain: base -> wrist_cam -> marker, then invert to the D435i ------
    R_base_marker, t_base_marker = compose(R_base_cam, t_base_cam,
                                           R_wrist_marker, t_wrist_marker)
    R_base_d435i, t_base_d435i = chain_base_to_camera(
        R_base_cam, t_base_cam, R_wrist_marker, t_wrist_marker,
        R_d435i_marker, t_d435i_marker)

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
        "method": "charuco_board_via_wrist_camera_and_pybullet_FK",
        "q_measured_rad": np.asarray(q, float).tolist(),
        "T_base_marker": {"R": R_base_marker.tolist(), "t": t_base_marker.tolist()},
        "reprojection_error_px": {"wrist": err_wrist, "d435i": err_d435i},
        "T_base_wristcam": {"R": R_base_cam.tolist(), "t": t_base_cam.tolist()},
        "T_EE_wristcam_source": "gen3.urdf camera_color_frame via PyBullet FK "
                                "(factory nominal geometry, but self-consistent with "
                                "the kinematics the throw planner uses)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {args.out}")

    if not args.no_canonical:
        from perception import base_frame
        save_extrinsic(R_base_d435i, t_base_d435i, base_frame.CANONICAL_PATH)
        print(f"wrote {base_frame.CANONICAL_PATH}  <- perception.base_frame reads THIS one")
    print("no gates were applied to this result -- start_of_day.py checks it against "
          "the floor plane, the depth sensor and the stored extrinsic before trusting it")


if __name__ == "__main__":
    main()
