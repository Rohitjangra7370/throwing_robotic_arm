"""
One command, run once at the start of a run day, that answers one question:
are we good to throw?

    python3 start_of_day.py --ip 192.168.1.101

Point the overhead D435i and the arm's own wrist camera at the same ChArUco
board lying flat on the floor, leave the arm holding that pose, and run this.
It calibrates `T_B_C` and then tries hard to prove the calibration wrong before
letting you throw against it.

WHY THIS EXISTS AS ONE SCRIPT
------------------------------
Every ingredient already existed -- `hw_readonly_check.py`, the wrist-camera
extrinsic chain, `run_hardware_throw.py plan` -- and none of them was ever run
as a set, in order, with one verdict at the end. The pieces that fell through
the gaps between them were exactly the ones that bit us: no calibration ever
reached `perception/base_frame.py`'s canonical path, so `measure_landing.py`
could not run; and `plan` prints `PRECHECK: PASS` on a line ABOVE a separate
`release pos in safe box: False`, which is a trap this project has already
documented twice and still has to read two lines to avoid.

THE GATES, AND WHY REPROJECTION ERROR IS NOT ONE OF THEM
---------------------------------------------------------
A planar target's PnP will absorb a wrong principal point, a wrong Euler
convention, or a printout scaled by "fit to page" into the pose it returns and
still report a fraction of a pixel of reprojection error. The 12 cm tool-frame
bug this script's chain was written to fix reprojected at 0.18 px. Reprojection
error only says the model is self-consistent. Being *right* needs facts from
outside the model, so these gates are outside facts:

  FLOOR   the board is lying on the floor and we know where the floor is, so
          its calibrated height and tilt have to agree (catches the whole
          arm-side chain: FK, tool frame, hand-eye offset, Euler convention)
  SCALE   the D435i's own depth sensor measures the board distance a second
          way, independent of intrinsics and of the printed square size
          (catches a mis-scaled printout, which nothing else here would)
  DRIFT   against the stored extrinsic, so a knocked mount is loud
  THROW   the checkpoint still plans a feasible throw, reading BOTH lines

Read-only on the arm throughout: joint angles, tool config, camera intrinsics,
one RTSP frame. `run_hardware_throw.py plan` is likewise a planner, not a
motion. Nothing here commands the arm to move.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.append("..")

ROOT = os.path.dirname(os.path.abspath(__file__))

PASS, FAIL, WARN, INFO = "PASS", "FAIL", "WARN", "INFO"
_MARK = {PASS: "  OK  ", FAIL: " FAIL ", WARN: " WARN ", INFO: "      "}


class Report:
    """Stage results, so the verdict is computed rather than remembered."""

    def __init__(self):
        self.rows = []

    def add(self, stage, level, msg):
        self.rows.append((stage, level, msg))
        print(f"[{_MARK[level]}] {msg}")
        return level != FAIL

    def failed(self):
        return [r for r in self.rows if r[1] == FAIL]

    def warned(self):
        return [r for r in self.rows if r[1] == WARN]


def section(title):
    print(f"\n--- {title} " + "-" * max(0, 62 - len(title)))


# ---------------------------------------------------------------------------
# 1. environment
# ---------------------------------------------------------------------------
def stage_env(rep):
    section("1. environment")
    ok = True
    if not sys.executable.startswith("/usr/bin/python3"):
        ok &= rep.add("env", WARN,
                      f"interpreter is {sys.executable} -- this project's deps "
                      f"live in /usr/bin/python3 (3.10). If imports fail, that is why.")
    else:
        rep.add("env", PASS, f"interpreter {sys.executable}")

    for mod in ("cv2", "numpy", "pybullet", "pyrealsense2", "torch", "kortex_api"):
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", "?")
            rep.add("env", PASS, f"import {mod} ({v})")
        except Exception as e:
            ok &= rep.add("env", FAIL, f"import {mod} failed: {e}")
    return ok


# ---------------------------------------------------------------------------
# 2. arm, read-only
# ---------------------------------------------------------------------------
def stage_arm(rep, args):
    section("2. arm (read-only -- hw_readonly_check.py)")
    cmd = [sys.executable, os.path.join(ROOT, "hw_readonly_check.py"),
           "--ip", args.ip, "--robot", args.robot]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return rep.add("arm", FAIL, "hw_readonly_check.py timed out -- arm reachable?")

    summary = [l for l in r.stdout.splitlines() if "checks:" in l and "FAIL" in l]
    detail = summary[-1].strip() if summary else "(no summary line)"
    if r.returncode == 0:
        return rep.add("arm", PASS, f"hw_readonly_check: {detail}")
    for line in r.stdout.splitlines():
        if "FAIL" in line and "checks:" not in line:
            rep.add("arm", INFO, f"  {line.strip()}")
    return rep.add("arm", FAIL, f"hw_readonly_check: {detail} (exit {r.returncode})")


# ---------------------------------------------------------------------------
# 3 + 4. cameras and the calibration itself
# ---------------------------------------------------------------------------
def _capture_wrist(ip, rtsp_url=None, n_frames=1, settle=30):
    import cv2
    url = rtsp_url or f"rtsp://{ip}/color"
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {url}")
    frames = []
    try:
        for _ in range(settle):
            ok, f = cap.read()
            if not ok:
                time.sleep(0.05)
        while len(frames) < n_frames:
            ok, f = cap.read()
            if ok:
                frames.append(f)
            else:
                time.sleep(0.05)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"no frame from {url}")
    return frames


def _capture_d435i(width, height, n_frames=1, settle=40):
    """N colour frames, depth aligned to the last one, and live colour intrinsics."""
    import pyrealsense2 as rs
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    profile = pipe.start(cfg)
    colors = []
    try:
        align = rs.align(rs.stream.color)
        scale = profile.get_device().first_depth_sensor().get_depth_scale()
        for _ in range(settle):
            pipe.wait_for_frames(2000)
        for _ in range(n_frames):
            frames = align.process(pipe.wait_for_frames(2000))
            colors.append(np.asanyarray(frames.get_color_frame().get_data()).copy())
        depth = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32) * scale
        i = frames.get_color_frame().profile.as_video_stream_profile().intrinsics
        K = np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]], float)
        dist = np.array(i.coeffs, float)
    finally:
        pipe.stop()
    return colors, depth, K, dist


def stage_calibrate(rep, args):
    import cv2
    from perception import base_frame
    from perception.ray_plane import D435I_COLOR_1280x720, D435I_COLOR_1920x1080
    from perception.wrist_chain import (average_transforms, base_to_wrist_camera,
                                        board_from_spec, chain_base_to_camera,
                                        compose, detect_board_pose, drift_check,
                                        floor_plane_check, save_extrinsic)
    from robot_arm.kinova_hardware import _KortexBackend, _patch_collections_abc

    section("3. cameras")
    try:
        colors, depth, K_d, dist_d = _capture_d435i(args.d435i_width, args.d435i_height,
                                                    args.n_frames)
        color = colors[-1]
    except RuntimeError as e:
        msg = str(e)
        if "busy" in msg.lower():
            msg += "  <- something else holds the camera (realsense-viewer?): " \
                   "close it, or `fuser -v /dev/video*` to find it"
        return rep.add("camera", FAIL, f"D435i capture failed: {msg}"), None
    rep.add("camera", PASS, f"D435i colour {color.shape[1]}x{color.shape[0]} x{len(colors)} "
                            f"+ aligned depth")

    ref = D435I_COLOR_1920x1080 if (args.d435i_width, args.d435i_height) == (1920, 1080) \
        else D435I_COLOR_1280x720
    dfx = abs(K_d[0, 0] - ref.fx)
    if dfx > 5.0:
        rep.add("camera", WARN, f"live fx {K_d[0,0]:.1f} vs ray_plane.py's {ref.fx:.1f} "
                                f"(delta {dfx:.1f}) -- using the live values")
    else:
        rep.add("camera", PASS, f"live intrinsics agree with ray_plane.py (fx {K_d[0,0]:.1f})")

    _patch_collections_abc()
    be = _KortexBackend(7, ip=args.ip, username=args.username, password=args.password)
    be.connect()
    try:
        q, _ = be.read_joint_state()
    finally:
        be.disconnect()
    rep.add("arm", PASS, f"joint angles read: {np.round(q, 4)}")

    try:
        wrists = _capture_wrist(args.ip, args.rtsp_url, args.n_frames)
    except RuntimeError as e:
        return rep.add("camera", FAIL, f"wrist camera: {e}"), None
    rep.add("camera", PASS, f"wrist RTSP {wrists[0].shape[1]}x{wrists[0].shape[0]} "
                            f"x{len(wrists)}")

    section("4. calibration")
    board = board_from_spec(args.squares_x, args.squares_y, args.square_mm, args.marker_ratio)
    K_w, dist_w = _wrist_intrinsics(args, wrists[0].shape)
    R_bw, t_bw = base_to_wrist_camera(q)      # arm is stationary: one FK for all frames

    def _detect_all(frames, K, dist, label):
        dets, errs = [], []
        for f in frames:
            try:
                dets.append(detect_board_pose(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY),
                                              board, K, dist))
            except RuntimeError as exc:
                errs.append(str(exc))
        if not dets:
            raise RuntimeError(errs[-1] if errs else "no frames")
        if errs:
            rep.add("calib", WARN, f"{label}: {len(errs)}/{len(frames)} frames "
                                   f"undetectable ({errs[-1]})")
        return dets

    try:
        dets_w = _detect_all(wrists, K_w, dist_w, "wrist")
    except RuntimeError as e:
        return rep.add("calib", FAIL, f"wrist camera cannot see the board: {e}"), None
    try:
        dets_d = _detect_all(colors, K_d, dist_d, "D435i")
    except RuntimeError as e:
        return rep.add("calib", FAIL, f"D435i cannot see the board: {e}"), None

    det_w, det_d = dets_w[-1], dets_d[-1]
    rep.add("calib", PASS, f"wrist: {det_w.n_corners} corners, {det_w.reproj_px:.3f} px, "
                           f"board at {det_w.distance_m:.3f} m ({len(dets_w)} frames)")
    rep.add("calib", PASS, f"D435i: {det_d.n_corners} corners, {det_d.reproj_px:.3f} px, "
                           f"board at {det_d.distance_m:.3f} m ({len(dets_d)} frames)")
    if det_d.dropped_ids or det_w.dropped_ids:
        rep.add("calib", INFO, f"duplicated marker ids dropped -- wrist {det_w.dropped_ids}, "
                               f"D435i {det_d.dropped_ids} (loose check-point markers in view)")

    # one extrinsic per frame pair, then average -- the spread IS the measurement
    # noise, which the drift gate below has to be able to see past.
    pairs = [chain_base_to_camera(R_bw, t_bw, w_.R, w_.t, d_.R, d_.t)
             for w_, d_ in zip(dets_w, dets_d)]
    R_BC, t_BC, spread_m, spread_deg = average_transforms(pairs)
    R_bm, t_bm, _, _ = average_transforms(
        [compose(R_bw, t_bw, w_.R, w_.t) for w_ in dets_w])

    if len(pairs) > 1:
        level = PASS if spread_m <= args.repeat_tol_m else WARN
        rep.add("gate", level, f"REPEAT: {len(pairs)} shots spread {spread_m*100:.1f} cm, "
                               f"{spread_deg:.2f} deg"
                               + ("" if level == PASS else
                                  "  -- noisy; drift smaller than this is invisible"))
    rep.add("calib", INFO, f"board in base frame: {np.round(t_bm, 4)}")
    rep.add("calib", INFO, f"T_B_C position:      {np.round(t_BC, 4)} "
                           f"({t_BC[2] - args.floor_z:.3f} m above the floor)")

    section("5. gates")
    ok = True

    for label, det in (("wrist", det_w), ("D435i", det_d)):
        if det.reproj_px <= args.max_reproj_px:
            rep.add("gate", PASS, f"reproj {label}: {det.reproj_px:.3f} px "
                                  f"<= {args.max_reproj_px} px")
        else:
            ok &= rep.add("gate", FAIL, f"reproj {label}: {det.reproj_px:.3f} px "
                                        f"> {args.max_reproj_px} px")
        if det.n_corners < args.min_corners:
            ok &= rep.add("gate", FAIL, f"corners {label}: {det.n_corners} "
                                        f"< {args.min_corners}")

    good, dz, tilt = floor_plane_check(R_bm, t_bm, args.floor_z,
                                       args.floor_tol_m, args.tilt_tol_deg)
    msg = (f"FLOOR: board at z={t_bm[2]:+.4f} m vs floor {args.floor_z:+.3f} "
           f"-> {dz*100:+.1f} cm, tilt {tilt:.1f} deg")
    if good:
        rep.add("gate", PASS, msg)
    else:
        ok &= rep.add("gate", FAIL, msg + f"  (tol {args.floor_tol_m*100:.0f} cm / "
                                          f"{args.tilt_tol_deg:.0f} deg)")
        rep.add("gate", INFO, "  a systematic height offset here means the ARM-side chain, "
                              "not the fitter -- 0.12 m means the tool frame is back")

    scale_ok, d_pnp, d_depth = _scale_gate(depth, det_d, K_d, args.scale_tol_m)
    if d_depth is None:
        rep.add("gate", WARN, "SCALE: no valid depth at the board centre -- gate skipped")
    elif scale_ok:
        rep.add("gate", PASS, f"SCALE: PnP {d_pnp:.3f} m vs depth {d_depth:.3f} m "
                              f"({abs(d_pnp-d_depth)*100:+.1f} cm)")
    else:
        ok &= rep.add("gate", FAIL, f"SCALE: PnP {d_pnp:.3f} m vs depth {d_depth:.3f} m "
                                    f"({(d_pnp-d_depth)*100:+.1f} cm > "
                                    f"{args.scale_tol_m*100:.0f} cm) -- suspect the printed "
                                    f"board is not {args.square_mm:.0f} mm per square")

    if base_frame.has_calibration():
        R_old, t_old = base_frame.load_extrinsic()
        same, dt, dR = drift_check(R_BC, t_BC, R_old, t_old, args.drift_tol_m, args.drift_tol_deg)
        msg = f"DRIFT vs stored: {dt*100:.1f} cm, {dR:.1f} deg"
        rep.add("gate", PASS if same else WARN,
                msg + ("" if same else "  -- camera mount moved since last calibration"))
    else:
        rep.add("gate", INFO, "DRIFT: no stored calibration yet (first run)")

    if ok and not args.no_write:
        save_extrinsic(R_BC, t_BC, base_frame.CANONICAL_PATH)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        save_extrinsic(R_BC, t_BC, os.path.join(ROOT, "calib", f"T_B_C_{stamp}.npz"))
        audit = os.path.join(ROOT, "calib", f"T_B_C_{stamp}.json")
        with open(audit, "w") as f:
            json.dump({
                "R_B_C": R_BC.tolist(), "t_B_C": t_BC.tolist(),
                "method": "charuco_board_via_wrist_camera_and_pybullet_FK",
                "q_measured_rad": np.asarray(q, float).tolist(),
                "board": {"squares_x": args.squares_x, "squares_y": args.squares_y,
                          "square_mm": args.square_mm, "marker_ratio": args.marker_ratio},
                "T_base_board": {"R": R_bm.tolist(), "t": t_bm.tolist()},
                "gates": {"reproj_px": {"wrist": det_w.reproj_px, "d435i": det_d.reproj_px},
                          "corners": {"wrist": det_w.n_corners, "d435i": det_d.n_corners},
                          "floor_dz_m": dz, "board_tilt_deg": tilt,
                          "repeatability_m": spread_m,
                          "repeatability_deg": spread_deg,
                          "n_frames": len(pairs),
                          "pnp_vs_depth_m": None if d_depth is None else d_pnp - d_depth},
                "dropped_marker_ids": {"wrist": det_w.dropped_ids, "d435i": det_d.dropped_ids},
                "floor_z_base_frame": args.floor_z,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)
        rep.add("calib", PASS, f"wrote {base_frame.CANONICAL_PATH} (+ {stamp} archive/audit)")
    elif not ok:
        rep.add("calib", WARN, "gates failed -- extrinsic NOT written; nothing downstream "
                               "will silently pick up a bad calibration")
    return ok, (R_BC, t_BC)


def _wrist_intrinsics(args, shape):
    """Live off the arm; the RTSP frame size must match what the arm reports."""
    from robot_arm.kinova_hardware import _KortexBackend, _patch_collections_abc
    _patch_collections_abc()
    from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
    from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient
    from kortex_api.autogen.messages import DeviceConfig_pb2, VisionConfig_pb2

    be = _KortexBackend(7, ip=args.ip, username=args.username, password=args.password)
    be.connect()
    try:
        dm = DeviceManagerClient(be._router)
        vids = [d.device_identifier for d in dm.ReadAllDevices().device_handle
                if d.device_type == DeviceConfig_pb2.DeviceTypes.Value("VISION")]
        if not vids:
            raise RuntimeError("no VISION device on the arm")
        vc = VisionConfigClient(be._router)
        sid = VisionConfig_pb2.SensorIdentifier()
        sid.sensor = VisionConfig_pb2.SENSOR_COLOR
        im = vc.GetIntrinsicParameters(sid, vids[0])
    finally:
        be.disconnect()
    K = np.array([[im.focal_length_x, 0, im.principal_point_x],
                  [0, im.focal_length_y, im.principal_point_y], [0, 0, 1]], float)
    dist = np.array(list(im.distortion_coeffs.k), float) \
        if hasattr(im.distortion_coeffs, "k") else np.zeros(5)
    return K, dist


def _scale_gate(depth_m, det, K, tol_m, win=7):
    """
    PnP distance vs the depth sensor's own measurement, at the board centre.

    Independent of both the colour intrinsics' scale and the printed square size,
    which is exactly what makes it able to catch a "fit to page" printout -- the
    failure mode make_aruco_targets.py warns about and that nothing else in this
    pipeline would notice.
    """
    import cv2
    centre_cam = det.t.reshape(1, 3)
    px, _ = cv2.projectPoints(np.zeros((1, 3)), cv2.Rodrigues(det.R)[0],
                              det.t.reshape(3, 1), K, np.zeros(5))
    u, v = np.round(px.reshape(2)).astype(int)
    h, w = depth_m.shape[:2]
    if not (win <= u < w - win and win <= v < h - win):
        return True, float(centre_cam[0, 2]), None
    patch = depth_m[v - win:v + win + 1, u - win:u + win + 1]
    valid = patch[patch > 0.05]
    if valid.size < 10:
        return True, float(centre_cam[0, 2]), None
    d_depth = float(np.median(valid))
    d_pnp = float(det.t[2])            # depth is z, not radial distance
    return abs(d_pnp - d_depth) <= tol_m, d_pnp, d_depth


# ---------------------------------------------------------------------------
# 6. can the checkpoint still plan a throw?
# ---------------------------------------------------------------------------
def stage_throw(rep, args):
    section("6. throw readiness (planner only -- no motion)")
    # no --base_height here: `plan` reads it from the checkpoint's config_log.pkl,
    # which self-describes it (unlike tool_offset_z, which the trainer never recorded
    # and which therefore must always be passed explicitly).
    cmd = [sys.executable, os.path.join(ROOT, "run_hardware_throw.py"), "plan",
           "--robot", args.robot, "--log_path", args.log_path,
           "--opt_pose", args.opt_pose, "--tool_offset_z", str(args.tool_offset_z),
           "--u_cap", str(args.u_cap),
           "--target", str(args.target[0]), str(args.target[1]),
           "--speed_scale", str(args.plan_speed_scale)]
    if args.wrist_roll_offset_deg is not None:
        cmd += ["--wrist_roll_offset_deg", str(args.wrist_roll_offset_deg)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return rep.add("throw", FAIL, "run_hardware_throw.py plan timed out")

    out = r.stdout.replace("\r", "\n")
    precheck = next((l for l in out.splitlines() if l.startswith("PRECHECK:")), None)
    safebox = next((l for l in out.splitlines() if "release pos in safe box:" in l), None)
    speed = next((l for l in out.splitlines() if "release speed" in l.lower()), None)

    ok = True
    # deliberately two separate assertions: `plan` prints PASS on the first even
    # when the second says False, which is the documented R2 trap.
    if precheck and "PASS" in precheck:
        rep.add("throw", PASS, precheck.strip())
    else:
        ok &= rep.add("throw", FAIL, precheck.strip() if precheck
                      else f"no PRECHECK line (exit {r.returncode})")
    if safebox and safebox.strip().endswith("True"):
        rep.add("throw", PASS, safebox.strip())
    else:
        ok &= rep.add("throw", FAIL, safebox.strip() if safebox
                      else "no 'release pos in safe box' line")
    if speed:
        rep.add("throw", INFO, speed.strip())
    if not ok and r.stderr.strip():
        rep.add("throw", INFO, f"  stderr: {r.stderr.strip().splitlines()[-1]}")
    return ok


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default="192.168.1.101")
    ap.add_argument("--username", default="admin")
    ap.add_argument("--password", default="admin")
    ap.add_argument("--robot", default="kinova_gen3_dyn")

    g = ap.add_argument_group("board (must match what is physically on the floor)")
    g.add_argument("--squares_x", type=int, default=5)
    g.add_argument("--squares_y", type=int, default=7)
    g.add_argument("--square_mm", type=float, default=35.0,
                   help="MEASURED printed square, not nominal -- a 'fit to page' "
                        "printout silently rescales every distance downstream")
    g.add_argument("--marker_ratio", type=float, default=0.75)

    g = ap.add_argument_group("cameras")
    g.add_argument("--d435i_width", type=int, default=1920)
    g.add_argument("--d435i_height", type=int, default=1080)
    g.add_argument("--rtsp_url", default=None)
    g.add_argument("--n_frames", type=int, default=5,
                   help="frames per camera; one extrinsic is solved per frame pair "
                        "and averaged, so single-shot PnP noise is measured rather "
                        "than inherited silently")

    g = ap.add_argument_group("gates")
    g.add_argument("--floor_z", type=float, default=None,
                   help="floor height in the BASE frame; defaults to -base_height, "
                        "which is the frame relation itself (base frame: base at 0, "
                        "floor at -base_height). Override only for a board that is "
                        "not lying on the floor.")
    g.add_argument("--floor_tol_m", type=float, default=0.02)
    g.add_argument("--tilt_tol_deg", type=float, default=5.0)
    g.add_argument("--scale_tol_m", type=float, default=0.05)
    g.add_argument("--repeat_tol_m", type=float, default=0.02)
    g.add_argument("--drift_tol_m", type=float, default=0.03)
    g.add_argument("--drift_tol_deg", type=float, default=5.0)
    g.add_argument("--max_reproj_px", type=float, default=1.0)
    g.add_argument("--min_corners", type=int, default=8)

    g = ap.add_argument_group("throw readiness")
    g.add_argument("--log_path", default="results_kinetic_chain_gen3_tcp/1")
    g.add_argument("--opt_pose", default="throw_pose_table_tcp.npy")
    g.add_argument("--tool_offset_z", type=float, default=0.12)
    g.add_argument("--base_height", type=float, default=0.433)
    g.add_argument("--u_cap", type=float, default=2.00)
    g.add_argument("--target", type=float, nargs=2, default=[0.71, 0.0])
    g.add_argument("--plan_speed_scale", type=float, default=1.0,
                   help="plan at 1.0, not the 0.15 default -- 0.15 shows a 7x "
                        "smaller qd and hides the headroom")
    g.add_argument("--wrist_roll_offset_deg", type=float, default=None,
                   help="passed through to `plan`. Plan with the value you will "
                        "actually throw with: it is still an OPEN item for this "
                        "checkpoint (the 15 deg release posture differs from the 5 deg "
                        "one the current 90 deg default was tuned against) and passing "
                        "the precheck is feasibility, not finger clearance -- that has "
                        "to be seen on the arm.")

    ap.add_argument("--no_write", action="store_true",
                    help="run every check but do not save the extrinsic")
    ap.add_argument("--skip_throw", action="store_true")
    args = ap.parse_args()
    if args.floor_z is None:
        args.floor_z = -args.base_height

    print(f"start_of_day  {time.strftime('%Y-%m-%d %H:%M:%S')}  arm {args.ip}")
    rep = Report()

    ok = stage_env(rep)
    ok &= stage_arm(rep, args)
    calib_ok, _ = stage_calibrate(rep, args) if ok else (False, None)
    ok &= calib_ok
    if not args.skip_throw:
        ok &= stage_throw(rep, args)

    section("VERDICT")
    for stage, level, msg in rep.rows:
        if level in (FAIL, WARN):
            print(f"  {level}  [{stage}] {msg}")
    n_fail, n_warn = len(rep.failed()), len(rep.warned())
    print(f"\n  {len(rep.rows)} checks: {n_fail} FAIL, {n_warn} WARN")
    if n_fail:
        print("\n  NO-GO -- do not throw. Fix the FAILs above and re-run.")
        return 1
    print("\n  GO -- calibrated, planned, and gated. Bring-up staging still applies:")
    print("        escalate speed_scale 0.15 -> 0.30 -> 0.60 -> 1.00, e-stop in hand,")
    print("        and re-verify --wrist_roll_offset_deg visually on the first swing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
