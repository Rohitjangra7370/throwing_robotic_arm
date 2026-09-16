"""
Predict where a commanded target will actually land, BEFORE throwing it.

WHAT THIS IS FOR
----------------
2026-09-11: 13 measured landings showed the arm throws ~45 cm long, and the
whole systematic is two numbers (see CLAUDE.md) --

    tool_offset 0.12 -> 0.27 m      38.9 cm -> 14.2 cm of model error
    release-speed gain    x1.11     14.2 cm ->  2.1 cm

Both were fitted IN-SAMPLE, on one session, over a commanded release-speed band
only 9% wide (1.40-1.54 m/s). A two-parameter fit that explains 13 points to
2.1 cm is encouraging and is not evidence the parameters are right: if the
x1.11 is really a speed- or pose-dependent effect, it will not survive a
re-searched pose table, and a retrain would silently inherit a wrong
assumption.

This script exists to test that BEFORE spending the re-search + retrain. It
plans a throw through the real planner, applies the two corrections, and
commits a predicted landing point to a file. Then you put a bin there and
throw. Agreement at a few cm over a WIDE speed range (the policy reaches
0.76-1.65 m/s, 2.1x the band the fit saw) is real out-of-sample evidence.
Disagreement that grows with speed is the x1.11 being speed-dependent.

    python3 predict_landing.py --sweep                     # a ready-made test set
    python3 predict_landing.py --target 0.71 0.0 --target 0.62 -0.18
    python3 predict_landing.py --sweep --out my_run.json

THIS PREDICTS THE PHYSICS, NOT THE POLICY
------------------------------------------
The prediction is made from the release state the planner produces for that
target, NOT from the target itself. So a commanded target outside the trained
band (0.68-0.74 x, +-0.25 y) is still a valid test of the forward model -- the
policy is merely the thing that picks a release speed, and any speed it picks
is fair game. It is NOT a test of policy accuracy, and the landing is NOT
expected to be near the commanded target. It is expected to be near the
PREDICTION.

WHY IT ALSO CHECKS VISIBILITY
-----------------------------
A throw whose descent falls outside the stereo frustum cannot be measured, and
`measure_landing` will refuse it -- correctly, but after you have already spent
the throw. The frustum is much narrower high up than it is at floor level, so a
short throw can land well inside the camera's floor footprint and still never
be seen falling. This projects the predicted trajectory into both IR imagers
and reports the same two quantities the gates use (fall seen, fall unseen), so
an untestable target can be dropped before the ball is loaded.
"""

import argparse
import datetime
import json
import os

import numpy as np
import pybullet as p

import run_hardware_throw as H
from measure_landing import (MAX_UNOBSERVED_DROP_M, MIN_OBSERVED_DROP_M,
                             default_rig, load_extrinsic)
from perception.trajectory import BALL_RADIUS, G_BASE, Z_FLOOR_BASE

# --------------------------------------------------------------------------- #
# The two fitted corrections. CHANGE THESE ONLY WITH NEW EVIDENCE, and say what
# the evidence was.
#
# TOOL_OFFSET_M -- the distance from `profile.ee_link` (the bare wrist flange)
#   to the point the ball actually leaves from. The arm's own firmware reports
#   0.12 m (`ControlConfig.GetToolConfiguration()`, read-only) and every table
#   and checkpoint on disk was built for 0.12. 0.27 is the operator's measured
#   physical value for the tool actually fitted on 2026-09-11, and it is
#   independently corroborated by the vision: the closest approach of each
#   measured arc to the FK-predicted release point is 11.8 cm at 0.12, 5.8 cm
#   at 0.27, 12.3 cm at 0.35. Release POSITION is the right discriminator here
#   because a speed gain cannot move it -- landing points alone cannot separate
#   the two ((0.12, x1.41), (0.27, x1.11) and (0.35, x1.00) all fit the 13
#   landings to the same 2.3 cm RMS.
#
# RELEASE_SPEED_GAIN -- what is left over after the geometry is right. Fitted,
#   NOT understood. Candidates: computed-torque tracking overshoot at release,
#   the 25 ms command quantisation, gripper drag, an under-modelled wrist
#   inertia. Any of those could be speed- or pose-dependent, which is the whole
#   reason this script exists.
# --------------------------------------------------------------------------- #
# 2026-09-11, second session: the MULTIPLICATIVE form above is REFUTED. Scored
# out-of-sample on new targets at the top of the reachable speed range, its
# range residual correlates with commanded speed at r = -0.72 over all 17
# measured landings (slope -17.8 cm per m/s, sd 1.71 cm). A constant gain
# cannot do that.
#
# Re-parameterised as a CONSTANT ADDITIVE excess on the release speed, the
# trend disappears: offset 0.22 m + 0.335 m/s gives r = -0.06, slope
# -1.0 cm/(m/s), residual sd 1.19 cm -- against a 0.85 cm measurement sd, i.e.
# ~0.8 cm of unmodelled physics left. An additive excess is also the more
# physical story: every throw uses the same T_R = 1.6 s ramp, so a fixed
# timing error in the release instant buys a roughly fixed velocity bonus,
# whereas a mis-sized tool offset would scale with wrist rate and show up as a
# gain.
#
# STILL NOT DECISIVE. All 17 landings sit in 1.40-1.67 m/s, a 19% span, where
# the additive and multiplicative forms are nearly collinear -- the r = -0.72
# -> -0.06 improvement is suggestive, not proof. They separate at LOW speed:
# at commanded target 0.50 (0.76 m/s) the two predict landings 12.9 cm apart.
# Throw that one and the question is settled. Until then `--model` selects.
TOOL_OFFSET_M = 0.27
RELEASE_SPEED_GAIN = 1.11

TOOL_OFFSET_ADDITIVE_M = 0.22
RELEASE_SPEED_DELTA = 0.335        # m/s, added to |v|, direction unchanged

# Fit residual over the 13 throws it was built from: 2.1 cm mean magnitude,
# range +0.2 +- 1.5 cm, lateral +1.5 +- 0.9 cm. Of that, 0.85 cm sd is the
# measurement itself (from repeat-pair spread), so ~1.45 cm sd is unmodelled
# physics. Quote 2 sigma as the bin-placement tolerance.
PREDICTION_SIGMA_M = 0.017

CAMERA_FPS = 90.0


def _tcp_state(arm, q_release, qd_release, tool_offset_z):
    """FK + Jacobian at the TCP. `localPosition` folds in omega x r itself."""
    q_full = arm._ik_q_neutral.copy()
    for li, dof in enumerate(arm._dof_ids):
        q_full[dof] = q_release[li]
        p.resetJointState(arm._arm_id, dof, q_release[li], physicsClientId=arm._cid)
    ls = p.getLinkState(arm._arm_id, arm._ee_link, computeForwardKinematics=True,
                        physicsClientId=arm._cid)
    pos, _ = p.multiplyTransforms(ls[4], ls[5], [0.0, 0.0, tool_offset_z],
                                  [0.0, 0.0, 0.0, 1.0], physicsClientId=arm._cid)
    jl, _ = p.calculateJacobian(arm._arm_id, arm._ee_link, [0.0, 0.0, tool_offset_z],
                                q_full.tolist(), [0.0] * arm._n_dofs,
                                [0.0] * arm._n_dofs, physicsClientId=arm._cid)
    J = np.array(jl)[:, arm._dof_ids]
    return np.array(pos), J @ np.asarray(qd_release, float)


def _impact(p0, v0, z_floor, ball_radius):
    a, b, c = 0.5 * G_BASE[2], v0[2], p0[2] - (z_floor + ball_radius)
    disc = b * b - 4 * a * c
    if disc < 0:
        return None, None
    t = (-b - np.sqrt(disc)) / (2 * a)
    return np.array([p0[0] + v0[0] * t, p0[1] + v0[1] * t]), float(t)


def visibility(p0, v0, t_impact, R_bc, t_bc, rig, fps=CAMERA_FPS):
    """
    How much of this trajectory's DESCENT both IR imagers can see.

    Returns the same two quantities `check_landing_is_observed` gates on, so a
    target that would be refused after the throw can be dropped before it.
    """
    W = rig.intr.width or 848
    Hh = rig.intr.height or 480
    ts = np.arange(0.0, t_impact, 1.0 / fps)
    seen = []
    for t in ts:
        p_b = p0 + v0 * t + 0.5 * G_BASE * t * t
        p_c = R_bc.T @ (p_b - t_bc)
        if p_c[2] <= 0:
            continue
        try:
            u1, v1, u2, v2 = rig.project(p_c)
        except RuntimeError:
            continue
        if 0 <= u1 < W and 0 <= u2 < W and 0 <= v1 < Hh and 0 <= v2 < Hh:
            seen.append((t, p_b[2]))
    if not seen:
        return {"n_frames": 0, "observed_drop_m": 0.0,
                "unobserved_drop_m": float(p0[2] - (Z_FLOOR_BASE + BALL_RADIUS)),
                "t_first": None, "t_last": None}
    zs = [z for _, z in seen]
    return {"n_frames": len(seen),
            "observed_drop_m": float(max(zs) - zs[-1]),
            "unobserved_drop_m": float(zs[-1] - (Z_FLOOR_BASE + BALL_RADIUS)),
            "t_first": float(seen[0][0]), "t_last": float(seen[-1][0])}


def predict(arm, profile, cfg, pol, ex, target_xy, args, R_bc, t_bc, rig):
    coeffs, q_rel, qd_rel, v_ach, speed, v_cmd, rel = H.plan_throw_for_target(
        arm, profile, cfg, pol, target_xy, opt_pose=args.opt_pose,
        u_cap=args.u_cap, tool_offset_z=args.tool_offset_z,
        wrist_roll_offset=np.deg2rad(args.wrist_roll_offset_deg))

    # As the checkpoint believes it (the flange-offset physics it was trained
    # under) and as corrected. Reporting both makes the correction auditable
    # rather than a number that appears from nowhere.
    p_nom, v_nom = _tcp_state(arm, q_rel, qd_rel, args.tool_offset_z)
    xy_nom, _ = _impact(p_nom, v_nom, args.floor_z, args.ball_radius)

    if args.model == "additive":
        p_cor, v_cor = _tcp_state(arm, q_rel, qd_rel, TOOL_OFFSET_ADDITIVE_M)
        n = np.linalg.norm(v_cor)
        v_cor = v_cor * ((n + RELEASE_SPEED_DELTA) / n)
    else:
        p_cor, v_cor = _tcp_state(arm, q_rel, qd_rel, TOOL_OFFSET_M)
        v_cor = v_cor * RELEASE_SPEED_GAIN
    xy_cor, t_imp = _impact(p_cor, v_cor, args.floor_z, args.ball_radius)

    box_ok = ex.check_release_pos(rel)
    precheck_ok, report = ex.precheck(coeffs, arm,
                                      release_speed=float(np.linalg.norm(v_ach)))
    vis = (visibility(p_cor, v_cor, t_imp, R_bc, t_bc, rig)
           if xy_cor is not None else None)
    measurable = bool(
        vis and vis["n_frames"] >= 12
        and vis["observed_drop_m"] >= MIN_OBSERVED_DROP_M
        and vis["unobserved_drop_m"] <= MAX_UNOBSERVED_DROP_M)
    return {
        "target": [float(target_xy[0]), float(target_xy[1])],
        "commanded_speed": float(speed),
        "release_pos_corrected": p_cor.tolist(),
        "release_speed_corrected": float(np.linalg.norm(v_cor)),
        "predicted_landing": None if xy_cor is None else xy_cor.tolist(),
        "landing_if_model_were_uncorrected": None if xy_nom is None else xy_nom.tolist(),
        "flight_time_s": t_imp,
        "precheck_ok": bool(precheck_ok),
        "release_in_box": bool(box_ok),
        "visibility": vis,
        "measurable": measurable,
        "q_release": list(map(float, q_rel)),
        "qd_release": list(map(float, qd_rel)),
    }


def default_test_set():
    """
    Targets chosen to make the test DISCRIMINATING, not comfortable.

    The x1.11 gain was fitted over commanded speeds 1.40-1.54 m/s -- a 9% band.
    Repeating that band proves nothing. These span the policy's full reachable
    0.76-1.65 m/s, bracketing the fitted band on both sides, so a gain that is
    really speed-dependent shows up as a residual that grows with speed rather
    than as scatter. Two entries are deliberate repeats of targets thrown on
    2026-09-11 -- they are the control: if those two miss their old landings,
    something drifted between sessions and the rest of the set means nothing.
    """
    return [
        (0.724, -0.009),   # CONTROL: thrown 2026-09-11, landed (+1.173, -0.022)
        (0.50,   0.00),    # 0.76 m/s -- half the fitted band, and the LOWEST
                           # target that is still measurable: 0.48 gives only
                           # 7 visible frames and would be refused
        (0.58,   0.00),    # 1.27 m/s -- between the low end and the band
        (0.62,  -0.18),    # below band, off-axis
        (0.70,   0.30),    # in band, azimuth outside the trained +-0.25
        (0.78,   0.00),    # above band
        (0.86,   0.00),    # top of the reachable speed range
        (0.82,  -0.30),    # above band, off-axis the other way
        (0.731, -0.148),   # CONTROL: thrown 2026-09-11, landed (+1.166, -0.248)
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", default="kinova_gen3_dyn")
    ap.add_argument("--log_path", default="results_kinetic_chain_gen3_tcp/1")
    ap.add_argument("--opt_pose", default="throw_pose_table_tcp.npy")
    ap.add_argument("--tool_offset_z", type=float, default=0.12,
                    help="what the POSE TABLE and checkpoint were built for. "
                         "This is not the physical offset -- that is "
                         "TOOL_OFFSET_M in this file, and changing this flag "
                         "will just make the planner refuse against the "
                         "table's stamp.")
    ap.add_argument("--base_height", type=float, default=0.433)
    ap.add_argument("--floor_z", type=float, default=None)
    ap.add_argument("--ball_radius", type=float, default=BALL_RADIUS)
    ap.add_argument("--u_cap", type=float, default=2.00)
    ap.add_argument("--wrist_roll_offset_deg", type=float, default=90.0)
    ap.add_argument("--extrinsic", default="calib/T_B_C.npz")
    ap.add_argument("--target", type=float, nargs=2, action="append", metavar=("X", "Y"))
    ap.add_argument("--sweep", action="store_true",
                    help="use the built-in discriminating test set")
    ap.add_argument("--model", choices=("gain", "additive"), default="additive",
                    help="'additive' (default): tool offset 0.22 m + a constant "
                         "+0.335 m/s on the release speed -- flat in speed over "
                         "all 17 landings. 'gain': the original 0.27 m + x1.11, "
                         "kept so the two can be thrown against each other.")
    ap.add_argument("--out", default=None,
                    help="where to commit the predictions (default: "
                         "predictions_<timestamp>.json)")
    args = ap.parse_args()
    if args.floor_z is None:
        args.floor_z = -args.base_height

    targets = list(args.target or [])
    if args.sweep or not targets:
        targets = default_test_set() + targets

    R_bc, t_bc = load_extrinsic(args.extrinsic)
    rig = default_rig()
    arm, profile, cid = H.build_arm(args.robot)
    try:
        pol, cfg = H.load_policy(args.log_path, None)
        table = H.load_pose_table(cfg, args.opt_pose)
        box = H.release_box_from_table(
            arm, table, tool_offset=[0.0, 0.0, args.tool_offset_z]) if table else None
        limits = H.make_limits(profile, 1.0, release_box=box, arm=arm)
        from robot_arm.kinova_hardware import HardwareThrowExecutor
        ex = HardwareThrowExecutor(limits, dry_run=True)

        rows = [predict(arm, profile, cfg, pol, ex, t, args, R_bc, t_bc, rig)
                for t in targets]
    finally:
        p.disconnect(cid)

    tol = 2 * PREDICTION_SIGMA_M
    if args.model == "additive":
        print(f"\nmodel 'additive': tool offset {TOOL_OFFSET_ADDITIVE_M:.2f} m, "
              f"release speed +{RELEASE_SPEED_DELTA:.3f} m/s")
    else:
        print(f"\nmodel 'gain': tool offset {TOOL_OFFSET_M:.2f} m, "
              f"gain x{RELEASE_SPEED_GAIN:.2f}  (REFUTED at r = -0.72 vs speed)")
    print(f"place the bin at the PREDICTED point. 2-sigma tolerance +-{tol * 100:.1f} cm.\n")
    print("  commanded target    cmd v    PREDICTED LANDING     flight  frames  seen/unseen   verdict")
    print("  " + "-" * 96)
    for r in rows:
        tx, ty = r["target"]
        if r["predicted_landing"] is None:
            print(f"  ({tx:+.3f},{ty:+.3f})       {r['commanded_speed']:.3f}    "
                  f"never reaches the floor")
            continue
        lx, ly = r["predicted_landing"]
        v = r["visibility"]
        bad = []
        if not r["precheck_ok"]:
            bad.append("PRECHECK FAIL")
        if not r["release_in_box"]:
            bad.append("OUTSIDE SAFE BOX")
        if not r["measurable"]:
            bad.append("NOT MEASURABLE")
        verdict = "ok" if not bad else " + ".join(bad)
        print(f"  ({tx:+.3f},{ty:+.3f})       {r['commanded_speed']:.3f}    "
              f"({lx:+.3f}, {ly:+.3f})      {r['flight_time_s']:.3f}s  "
              f"{v['n_frames']:>4}   {v['observed_drop_m']:.2f}/{v['unobserved_drop_m']:.2f} m   "
              f"{verdict}")

    out = args.out or f"predictions_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out, "w") as f:
        json.dump({
            "written": datetime.datetime.now().isoformat(timespec="seconds"),
            "model": args.model,
            "tool_offset_m": (TOOL_OFFSET_ADDITIVE_M if args.model == "additive"
                              else TOOL_OFFSET_M),
            "release_speed_gain": (1.0 if args.model == "additive"
                                   else RELEASE_SPEED_GAIN),
            "release_speed_delta": (RELEASE_SPEED_DELTA if args.model == "additive"
                                    else 0.0),
            "prediction_sigma_m": PREDICTION_SIGMA_M,
            "log_path": args.log_path, "opt_pose": args.opt_pose,
            "wrist_roll_offset_deg": args.wrist_roll_offset_deg,
            "u_cap": args.u_cap, "floor_z": args.floor_z,
            "predictions": rows,
        }, f, indent=1)
    print(f"\ncommitted to {out} -- written BEFORE the throws, which is what makes "
          f"this out-of-sample.\nCompare afterwards with:  python3 score_predictions.py "
          f"{out}")


if __name__ == "__main__":
    main()
