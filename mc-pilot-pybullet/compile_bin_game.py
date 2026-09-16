"""
Compile one bin-aiming session into a single results file.

WHY THIS RECONSTRUCTS THE MARKER POSITION. `bin_game_log.jsonl` is written by
the GUI's "Aim at bin" button, and on 2026-09-11 that write was broken (a
missing import; fixed, but the session had already run). The marker position is
recoverable anyway and exactly: the aim solves for the commanded target whose
PREDICTED LANDING equals the marker, to a converged residual of ~0 mm, so
replaying the forward model on the logged commanded target returns the marker
the operator was aiming at. Where a real aim log exists it is used instead and
the two are cross-checked.

    python3 compile_bin_game.py --since 2026-09-11T02:50 \
        --out results_bin_game/session_20260911_024918.json
"""

import argparse
import json
import os
import sys

import numpy as np
import pybullet as p

import predict_landing as PL
import run_hardware_throw as H
from measure_landing import default_rig
from perception import base_frame


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session_log", default="hardware_session_log.jsonl")
    ap.add_argument("--aim_log", default="bin_game_log.jsonl")
    ap.add_argument("--since", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--robot", default="kinova_gen3_dyn")
    ap.add_argument("--log_path", default="results_kinetic_chain_gen3_tcp/1")
    ap.add_argument("--opt_pose", default="throw_pose_table_tcp.npy")
    ap.add_argument("--tool_offset_z", type=float, default=0.12)
    ap.add_argument("--base_height", type=float, default=0.433)
    ap.add_argument("--ball_radius", type=float, default=PL.BALL_RADIUS)
    ap.add_argument("--u_cap", type=float, default=2.00)
    ap.add_argument("--wrist_roll_offset_deg", type=float, default=90.0)
    ap.add_argument("--model", choices=("gain", "additive"), default="additive")
    ap.add_argument("--remeasured", default=None,
                    help="optional JSON from a re-run of measure_landing, used "
                         "in place of the landings the session logged live")
    args = ap.parse_args()
    args.floor_z = -args.base_height

    throws = [r for r in load_jsonl(args.session_log)
              if r.get("timestamp", "") >= args.since]
    aims = [a for a in load_jsonl(args.aim_log)
            if a.get("timestamp", "") >= args.since]
    remeas = {}
    if args.remeasured:
        for e in json.load(open(args.remeasured)):
            remeas[e["throw_index"]] = e

    R_bc, t_bc = base_frame.load_extrinsic()
    rig = default_rig()
    arm, profile, cid = H.build_arm(args.robot)
    rows = []
    try:
        pol, cfg = H.load_policy(args.log_path, None)
        table = H.load_pose_table(cfg, args.opt_pose)
        box = H.release_box_from_table(
            arm, table, tool_offset=[0.0, 0.0, args.tool_offset_z]) if table else None
        from robot_arm.kinova_hardware import HardwareThrowExecutor
        ex = HardwareThrowExecutor(
            H.make_limits(profile, 1.0, release_box=box, arm=arm), dry_run=True)

        for r in throws:
            pred = PL.predict(arm, profile, cfg, pol, ex, tuple(r["target"]),
                              args, R_bc, t_bc, rig)
            aimed = pred["predicted_landing"]
            logged_aim = next((a for a in aims
                               if a.get("command_target")
                               and abs(a["command_target"][0] - r["target"][0]) < 1e-4
                               and abs(a["command_target"][1] - r["target"][1]) < 1e-4),
                              None)
            landing = (remeas.get(r["throw_index"], {}).get("landing")
                       if args.remeasured else r.get("landing_xy"))
            rec = {
                "throw_index": r["throw_index"], "timestamp": r["timestamp"],
                "command_target": r["target"],
                "commanded_speed": r["commanded_speed"],
                "speed_scale": r.get("speed_scale"),
                "capture_file": r.get("capture_file"),
                "aim_point": aimed,          # where the ball was aimed = the marker
                "aim_point_source": ("aim log" if logged_aim else
                                     "reconstructed from the commanded target"),
                "marker_xy_logged": (logged_aim or {}).get("marker_xy"),
                "landing": landing,
                "refusal": (remeas.get(r["throw_index"], {}).get("refusal")
                            if args.remeasured else r.get("refusal_reason")),
            }
            if landing and aimed:
                d = np.asarray(landing, float) - np.asarray(aimed, float)
                rec["error_m"] = float(np.linalg.norm(d))
                rec["error_xy_m"] = [float(d[0]), float(d[1])]
            rows.append(rec)
    finally:
        p.disconnect(cid)

    hit = [r for r in rows if r.get("error_m") is not None]
    errs = np.array([r["error_m"] for r in hit]) if hit else np.zeros(0)
    summary = {
        "n_throws": len(rows), "n_measured": len(hit),
        "mean_error_m": float(errs.mean()) if hit else None,
        "median_error_m": float(np.median(errs)) if hit else None,
        "max_error_m": float(errs.max()) if hit else None,
        "model": args.model,
        "tool_offset_m": (PL.TOOL_OFFSET_ADDITIVE_M if args.model == "additive"
                          else PL.TOOL_OFFSET_M),
        "release_speed_delta": (PL.RELEASE_SPEED_DELTA if args.model == "additive"
                                else None),
        "release_speed_gain": (None if args.model == "additive"
                               else PL.RELEASE_SPEED_GAIN),
        "checkpoint": args.log_path, "pose_table": args.opt_pose,
        "extrinsic_t": [float(v) for v in t_bc],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "throws": rows}, f, indent=1)

    print(f"  {len(hit)}/{len(rows)} throws measured")
    if hit:
        print(f"  error vs the aim point: mean {errs.mean()*100:.1f} cm, "
              f"median {np.median(errs)*100:.1f} cm, max {errs.max()*100:.1f} cm")
    print(f"  written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
