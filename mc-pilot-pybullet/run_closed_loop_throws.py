"""
Closed-loop hardware throw session -- one throw per invocation, live status
dashboard, structured per-throw log. Wraps run_hardware_throw.py's already
safety-gated plan/precheck/throw path; does not reimplement or bypass any of
it (see release_solver.py's docstring on why a second copy always drifts).

WHAT THIS SCRIPT DOES NOT DO: manage the camera. `throw_capture.py` runs as
its own long-lived process for the whole session (auto- or manually-triggered
ring-buffer recorder) and is not started/stopped per throw here -- the two
are deliberately decoupled, matching measure_landing.py's own design ("the
offline half of the vision pipeline... a changed fitter can be re-run against
a real throw from weeks ago"). This script's job is the ARM side: plan,
show the operator exactly what HARDWARE_RUNBOOK.md says to check, gate on
--confirm, throw, and log a structured record an operator (or a later batch
job) can cross-reference against the camera's own recordings by index/time
and fill in `landing_xy` once measure_landing.py has been run offline.

Usage (per throw, escalate speed_scale exactly as HARDWARE_RUNBOOK.md says):
  python run_closed_loop_throws.py --log_path results_kinetic_chain_gen3_tcp/1 \\
      --opt_pose throw_pose_table_tcp.npy --tool_offset_z 0.12 \\
      --target 0.75 0.05 --throw_index 0 --ball_id tennis-01 \\
      --arm --speed_scale 0.15 --confirm
"""
import argparse
import dataclasses
import datetime
import glob
import json
import os
import sys
import time

import numpy as np

import run_hardware_throw as H
from robot_arm.kinova_hardware import HardwareThrowExecutor


DEFAULT_LOG_PATH = "closed_loop_throw_log.jsonl"


def _newest_recording_after(throws_dir, after_ts):
    """Newest .npz under throws_dir with mtime >= after_ts, or None.

    Pure filesystem polling for the OPT-IN --measure path -- throw_capture.py
    is a separate, independently armed process; this never starts, stops, or
    communicates with it directly, matching this file's decoupled-by-default
    design for everything else.
    """
    if not os.path.isdir(throws_dir):
        return None
    candidates = [
        f for f in glob.glob(os.path.join(throws_dir, "*.npz"))
        if os.path.getmtime(f) >= after_ts
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def load_extrinsic_any(path):
    """
    Load T_B_C from either format this project produces:
      - .npz with R (3,3) / t (3,) -- measure_landing.py's own convention
      - .json with R_B_C / t_B_C   -- calibrate_camera_extrinsics.py's output

    No converter between the two exists anywhere in the repo, so
    measure_landing.py's --extrinsic would refuse calibrate_camera_extrinsics.py's
    own output file directly (wrong extension, wrong key names). Bridging it
    here rather than in either of those files keeps this fix local instead of
    touching camera-side code someone else is actively editing.
    """
    if path.endswith(".json"):
        with open(path) as f:
            d = json.load(f)
        R, t = np.asarray(d["R_B_C"], float), np.asarray(d["t_B_C"], float)
    else:
        from measure_landing import load_extrinsic
        return load_extrinsic(path)

    if R.shape != (3, 3) or t.shape != (3,):
        raise ValueError(f"expected R (3,3) and t (3,), got {R.shape} and {t.shape}")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
        raise ValueError("R is not orthonormal -- this is not a rotation")
    return R, t


def _measure(recording, extrinsic, z_floor, ball_radius, seed, target_xy):
    from measure_landing import measure_landing
    from perception.ir_capture import load_recording
    from perception.trajectory import Z_FLOOR_BASE

    if z_floor is None:
        z_floor = Z_FLOOR_BASE
    rec = load_recording(recording)
    R_bc, t_bc = load_extrinsic_any(extrinsic)
    out = measure_landing(rec, R_bc, t_bc, z_floor=z_floor,
                          ball_radius=ball_radius, seed=seed)
    err = float(np.hypot(out["x"] - target_xy[0], out["y"] - target_xy[1]))
    print(f"\n=== LANDING (measured) ===")
    print(f"x={out['x']:+.4f}  y={out['y']:+.4f}   sigma={out['sigma_xy_m'] * 1e3:.1f} mm")
    print(f"error vs target: {err * 100:.2f} cm")
    return out, err


def format_status_dashboard(target, speed, speed_scale, precheck_ok,
                            precheck_report, release_box_ok, release_pos):
    """
    Mirrors exactly what a human is supposed to read off `plan` per
    HARDWARE_RUNBOOK.md: PRECHECK and 'release pos in safe box' are reported
    as TWO separate lines on purpose -- `plan` prints PRECHECK: PASS even
    when the release-box line says False, and collapsing them into one
    combined verdict is the specific mistake that note exists to prevent.
    """
    lines = [
        "=== CLOSED-LOOP THROW STATUS ===",
        f"target: {tuple(round(float(x), 3) for x in target)}   "
        f"commanded release speed: {float(speed):.3f} m/s   "
        f"speed_scale: {float(speed_scale)} "
        f"({'REAL THROW' if float(speed_scale) >= 0.99 else 'SLOW REHEARSAL'})",
        f"release pos: {tuple(round(float(x), 3) for x in release_pos)}",
        f"release pos in safe box: {bool(release_box_ok)}",
        "--- trajectory precheck ---",
        str(precheck_report),
        f"PRECHECK: {'PASS' if precheck_ok else 'FAIL'}",
    ]
    if not precheck_ok or not release_box_ok:
        lines.append(
            "REFUSE: do NOT throw -- "
            + ("precheck failed" if not precheck_ok else "")
            + (" AND " if not precheck_ok and not release_box_ok else "")
            + ("release position outside the safe box" if not release_box_ok else "")
        )
    return "\n".join(lines)


class _NumpyJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return super().default(o)


def build_throw_record(throw_index, target, commanded_speed, speed_scale,
                       q_release, qd_release, precheck_ok, exec_stats,
                       ball_id, capture_file, landing_xy,
                       measurement=None, release_in_box=None):
    """
    One line of the dataset HARDWARE_RUNBOOK.md Sec 4 describes: "Record per
    throw (this is the dataset, not a debug log)".

    `measurement` is measure_landing()'s dict when a track was accepted, or
    {"refusal_reason": str} when it was refused. A refused throw is still
    logged: it is evidence about the rig, and dropping it would quietly bias
    the dataset toward the throws that happened to track well.

    `measured_v0` is the ball's ACTUAL release velocity from vision. Commanded
    speed vs measured_v0 is where the sim-to-real gap actually lives at these
    speeds (25 ms quantisation ~ 3 cm, drag ~ 5 mm), so it is a first-class
    field, not a diagnostic.
    """
    m = measurement or {}

    def _vec(key):
        v = m.get(key)
        return None if v is None else [float(x) for x in np.asarray(v).reshape(-1)]

    return {
        "throw_index": int(throw_index),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "target": [float(x) for x in target],
        "commanded_speed": float(commanded_speed),
        "speed_scale": float(speed_scale),
        "q_release": [float(x) for x in q_release],
        "qd_release": [float(x) for x in qd_release],
        "precheck_ok": bool(precheck_ok),
        "release_in_box": None if release_in_box is None else bool(release_in_box),
        "exec_stats": exec_stats,
        "ball_id": ball_id,
        "capture_file": capture_file,
        "landing_xy": landing_xy,
        "sigma_xy_m": m.get("sigma_xy_m"),
        "n_frames": m.get("n_frames"),
        "n_inliers": m.get("n_inliers"),
        "rms_px": m.get("rms_px"),
        "measured_p0": _vec("p0"),
        "measured_v0": _vec("v0"),
        "refusal_reason": m.get("refusal_reason"),
    }


def append_log(record, log_path):
    with open(log_path, "a") as f:
        f.write(json.dumps(record, cls=_NumpyJSONEncoder) + "\n")


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--robot", default="kinova_gen3_dyn")
    ap.add_argument("--arm", action="store_true", help="talk to the REAL arm (default: dry-run)")
    ap.add_argument("--ip", default="192.168.1.101")
    ap.add_argument("--log_path", required=True, help="trained checkpoint results/<seed>/")
    ap.add_argument("--opt_pose", default=None)
    ap.add_argument("--tool_offset_z", type=float, default=0.0)
    ap.add_argument("--u_cap", type=float, default=None)
    ap.add_argument("--wrist_roll_offset_deg", type=float, default=0.0,
                    help="degrees added to joint 6 at release for finger/release-path "
                         "clearance -- see run_hardware_throw.py; provably free of the "
                         "release itself (frozen at qd=0, last in chain), still goes "
                         "through the normal feasibility check.")
    ap.add_argument("--target", type=float, nargs=2, required=True)
    ap.add_argument("--throw_index", type=int, required=True,
                    help="position in today's session -- cross-referenced against "
                         "throw_capture.py's own event numbering by the operator")
    ap.add_argument("--ball_id", default="unassigned")
    ap.add_argument("--speed_scale", type=float, default=0.15)
    ap.add_argument("--positioning_scale", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--confirm", action="store_true",
                    help="assert workspace clear + e-stop in hand (required with --arm)")
    ap.add_argument("--out_log", default=DEFAULT_LOG_PATH)

    ap.add_argument("--measure", action="store_true",
                    help="OPT-IN: after the throw, poll --throws_dir for a new "
                         "recording and fill in capture_file/landing_xy immediately, "
                         "instead of the default decoupled offline pass.")
    ap.add_argument("--throws_dir", default="throws/")
    ap.add_argument("--extrinsic", default=None,
                    help="T_B_C from calibrate_camera_extrinsics.py (.json) or "
                         "measure_landing.py's own format (.npz); required with --measure")
    ap.add_argument("--z_floor", type=float, default=None)
    ap.add_argument("--ball_radius", type=float, default=0.0327)
    ap.add_argument("--measure_seed", type=int, default=0)
    ap.add_argument("--measure_timeout", type=float, default=30.0,
                    help="seconds to wait for throw_capture.py's recording to appear")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.arm and not args.confirm:
        print("REFUSED: real motion requires --confirm (workspace clear, ball "
              "secured, e-stop in hand). Aborting.")
        return 2
    if args.measure and not args.extrinsic:
        print("REFUSED: --measure requires --extrinsic <T_B_C file>.")
        return 2

    arm, profile, cid = H.build_arm(args.robot)
    pol, cfg = H.load_policy(args.log_path, None)
    coeffs, q_rel, qd_rel, v_ach, speed, v_cmd, rel = H.plan_throw_for_target(
        arm, profile, cfg, pol, args.target,
        opt_pose=args.opt_pose, u_cap=args.u_cap, tool_offset_z=args.tool_offset_z,
        wrist_roll_offset=np.deg2rad(args.wrist_roll_offset_deg),
    )
    table = H.load_pose_table(cfg, args.opt_pose)
    box = H.release_box_from_table(
        arm, table, tool_offset=[0.0, 0.0, args.tool_offset_z]
    ) if table else None
    limits = H.make_limits(profile, args.speed_scale, release_box=box, arm=arm,
                           positioning_scale=args.positioning_scale)
    ex = HardwareThrowExecutor(limits, dry_run=not args.arm, ip=args.ip)

    release_box_ok = ex.check_release_pos(rel)
    precheck_ok, precheck_report = ex.precheck(
        coeffs, arm, release_speed=float(np.linalg.norm(v_ach))
    )
    print(format_status_dashboard(
        target=args.target, speed=speed, speed_scale=args.speed_scale,
        precheck_ok=precheck_ok, precheck_report=precheck_report,
        release_box_ok=release_box_ok, release_pos=rel,
    ))

    if not precheck_ok or not release_box_ok:
        import pybullet as p
        p.disconnect(cid)
        return 2

    exec_stats = {}
    start_ts = time.time()
    if args.arm:
        with ex:
            ex.set_gripper(closed=True)
            ex.home(arm, np.array(profile.q_neutral, float), duration=args.duration)
            ex.backend.open_realtime_feedback()
            try:
                ex.rehearse_or_throw(coeffs, arm, track=None)
            finally:
                ex.backend.close_realtime_feedback()
        exec_stats = dict(getattr(ex, "last_exec_stats", {}) or {})
    else:
        print("[dry-run] not sending any command to the arm (pass --arm for real motion)")

    import pybullet as p
    p.disconnect(cid)

    capture_file, landing_xy = None, None
    if args.measure and args.arm:
        print(f"\n[measure] polling {args.throws_dir} for a new recording "
              f"(throw_capture.py must already be running and armed there)...")
        deadline = time.time() + args.measure_timeout
        rec_path = None
        while time.time() < deadline:
            rec_path = _newest_recording_after(args.throws_dir, start_ts)
            if rec_path:
                break
            time.sleep(1.0)
        if rec_path is None:
            print(f"[measure] no new recording within {args.measure_timeout}s -- "
                  "is throw_capture.py running and armed? landing_xy stays null; "
                  "the default offline pass still applies.")
        else:
            try:
                out, err = _measure(rec_path, args.extrinsic, args.z_floor,
                                    args.ball_radius, args.measure_seed, args.target)
                capture_file, landing_xy = rec_path, [float(out["x"]), float(out["y"])]
            except Exception as e:
                print(f"[measure] REFUSED this track: {e}. Per HARDWARE_RUNBOOK.md: "
                      "re-throw, do not hand-tune a threshold to force a number out "
                      "of a bad recording. landing_xy stays null.")
                capture_file = rec_path

    record = build_throw_record(
        throw_index=args.throw_index, target=args.target, commanded_speed=speed,
        speed_scale=args.speed_scale, q_release=q_rel, qd_release=qd_rel,
        precheck_ok=precheck_ok, exec_stats=exec_stats, ball_id=args.ball_id,
        capture_file=capture_file, landing_xy=landing_xy,
    )
    append_log(record, args.out_log)
    print(f"[log] appended throw_index={args.throw_index} -> {args.out_log}")
    if landing_xy is None:
        print("[log] landing_xy is null -- run measure_landing.py once calibrated and "
              "fill it in against this throw_index before feeding the closed-loop update.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
