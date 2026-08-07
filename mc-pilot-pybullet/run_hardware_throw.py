"""
Staged bring-up CLI for running the MC-PILOT throw on the real Kinova Gen3.

SAFETY MODEL: nothing moves unless you ASK for motion AND clear the interlocks.
Escalate one stage at a time. Do NOT skip to `throw`.

  # 0. Dry-run plan only (no arm, no kortex_api needed) -- ALWAYS do this first:
  python run_hardware_throw.py plan  --log_path results_mc_pilot_pb_A_kinova_gen3/1 \
                                     --robot kinova_gen3 --target 0.75 0.05

  # 1. Connect + read joint state (no motion):
  python run_hardware_throw.py connect --arm --ip 192.168.1.101

  # 2. Gentle homing move to neutral (slow, capped):
  python run_hardware_throw.py home --arm --ip 192.168.1.101 --robot kinova_gen3

  # 3. Gripper open/close test (no arm motion):
  python run_hardware_throw.py gripper --arm --ip 192.168.1.101 --open
  python run_hardware_throw.py gripper --arm --ip 192.168.1.101 --close

  # 4. SLOW rehearsal throw at 15% speed (ball dribbles; validates motion+release):
  python run_hardware_throw.py throw --arm --ip 192.168.1.101 --robot kinova_gen3 \
        --log_path results_mc_pilot_pb_A_kinova_gen3/1 --target 0.75 0.05 \
        --speed_scale 0.15 --confirm

  # 5. Escalate speed_scale (0.15 -> 0.3 -> 0.6 -> 1.0) ONLY after each is clean:
  #    ... --speed_scale 0.30 --confirm     (repeat, watching every run)

Interlocks for `throw`: requires --arm AND --confirm (you assert the workspace is
clear, ball secured, e-stop in hand). Default speed_scale is a slow rehearsal.
"""

import argparse
import os
import sys
import pickle as pkl

import numpy as np
import torch

sys.path.append("..")

import pybullet as p
import pybullet_data

import policy_learning.Policy as Policy
from robot_arm.arm_controller import ArmController
from robot_arm.robot_profiles import get_robot_profile
from robot_arm.kinova_hardware import (HIGH_LEVEL_MAX_HZ, HardwareThrowExecutor,
                                       SafetyLimits)
from simulation_class.release_solver import OptimizedReleaseSolver


# --------------------------------------------------------------------------- #
# Planning (shared with sim -- single source of truth)
# --------------------------------------------------------------------------- #
def build_arm(robot):
    profile = get_robot_profile(robot)
    cid = p.connect(p.DIRECT)
    # GRAVITY IS NOT OPTIONAL HERE, AND ITS ABSENCE IS SILENT.
    #
    # PyBullet defaults a fresh client to zero gravity, and
    # `calculateInverseDynamics` then returns INERTIAL TORQUE ONLY -- no error,
    # no warning, just numbers that look plausible. Every other client in this
    # repo sets it (model_pybullet.py:192, find_throw_pose.py, the tests); this
    # one did not, and it is the only one whose numbers gate motion on a real
    # arm.
    #
    # Measured cost of the omission on the shipped throw: the precheck reported
    # peak 8.6 Nm (22% of limit) when the same trajectory with gravity needs
    # 36.7 Nm (94.2%) -- a 4.3x under-report on the worst joint, and the
    # difference between "PASS" and "refuse to move".
    p.setGravity(0.0, 0.0, -9.81, physicsClientId=cid)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    urdf = pybullet_data.getDataPath() + "/" + profile.urdf_rel_path
    arm = ArmController(cid, urdf, robot_name=robot)
    return arm, profile, cid


def load_policy(log_path, uM):
    with open(os.path.join(log_path, "log.pkl"), "rb") as f:
        log = pkl.load(f)
    with open(os.path.join(log_path, "config_log.pkl"), "rb") as f:
        cfg = pkl.load(f)
    st = log["parameters_trial_list"][-1]
    common = dict(
        full_state_dim=8, target_dim=2, num_basis=st["centers"].shape[0], u_max=cfg["uM"],
        lengthscales_init=st["log_lengthscales"].exp().numpy()[0],
        centers_init=st["centers"].numpy(), weight_init=st["f_linear.weight"].numpy(),
        flg_drop=False, dtype=torch.float64, device=torch.device("cpu"),
    )
    if cfg.get("residual_physics"):
        pol = Policy.Residual_Throwing_Policy(
            release_pos=np.asarray(cfg["release_pos"], float),
            launch_angle_deg=cfg.get("launch_angle_deg", 35.0),
            target_height=cfg.get("target_height", 0.0),
            delta_max_frac=cfg.get("delta_max_frac", 0.5), **common,
        )
    else:
        pol = Policy.Throwing_Policy(**common)
    pol.load_state_dict(st)
    pol.eval()
    return pol, cfg


def load_pose_table(cfg, override=None):
    """
    The azimuth->posture table this policy was TRAINED through, or None for a
    legacy IK+pinv checkpoint.

    Checkpoints written before the trainer recorded 'opt_pose' in config_log.pkl
    cannot report their own release geometry, so `override` (--opt_pose) exists
    to supply it. Getting this wrong is not a subtle error: without the table the
    planner falls back to IK+pinv and plans an entirely different, far weaker
    near-horizontal throw than the one that was trained and validated.
    """
    path = override or cfg.get("opt_pose")
    if path is None:
        return None
    return list(np.load(path, allow_pickle=True))


def plan_throw_for_target(arm, profile, cfg, pol, target_xy, opt_pose=None,
                          u_cap=None):
    """
    policy(target) -> release speed -> release state -> joint trajectory.

    This MUST mirror `PyBulletThrowingSystem._simulate_pybullet` exactly, because
    the whole hardware argument is that the arm executes the motion the simulator
    was trained on. Three things were previously taken from the wrong place and
    are now read from the trained config:

      * the RELEASE STATE comes from the shared `OptimizedReleaseSolver` and the
        pose table -- not from IK + pinv, which plans a different throw;
      * PHASE TIMINGS come from cfg's T_W/T_R -- `profile.timing` is a generic
        per-arm default and disagreed with training (0.4/0.8 vs 0.5/1.6), which
        would have roughly doubled the commanded peak joint velocity;
      * `monotonic_windup` is enabled in optimized-posture mode, as in training.
    """
    rel = np.asarray(cfg["release_pos"], float)
    tgt = np.asarray(target_xy, float)
    with torch.no_grad():
        s = torch.tensor(np.concatenate([rel, np.zeros(3), tgt]), dtype=torch.float64).unsqueeze(0)
        speed = float(pol(s, t=0).item())
    if u_cap is not None:
        speed = min(speed, float(u_cap))

    # Generic ballistic velocity vector. In optimized-posture mode only its
    # MAGNITUDE survives (the solver substitutes the table's own exact v_dir),
    # so cfg's launch_angle_deg is irrelevant there -- |v_cmd| == speed for any
    # launch angle by construction.
    a = np.deg2rad(float(cfg.get("launch_angle_deg", 35.0)))
    azimuth = np.arctan2(tgt[1] - rel[1], tgt[0] - rel[0])
    v_cmd = np.array([speed * np.cos(a) * np.cos(azimuth),
                      speed * np.cos(a) * np.sin(azimuth),
                      speed * np.sin(a)])

    table = load_pose_table(cfg, opt_pose)
    solver = OptimizedReleaseSolver(
        opt_posture_table=table,
        opt_launch_deg=float(cfg.get("opt_launch_deg",
                                     table[0]["elev_deg"] if table else 43.0)),
    )
    q_ovr = qd_ovr = None
    if solver.active:
        rel, q_ovr, qd_ovr, v_cmd = solver.solve(arm, v_cmd, target_xy=tgt)

    # Same timing source and t_arm formula as the sim rollout.
    t_w = float(cfg["T_W"])
    t_r = float(cfg["T_R"])
    t_arm = max(1.20, float(profile.timing[2]), float(cfg["T"]) + t_r)

    coeffs, q_release, qd_release, v_ach = arm.plan_throw(
        v_cmd, rel, t_w=t_w, t_r=t_r, T=t_arm,
        q_release_override=q_ovr, qd_release_override=qd_ovr,
        monotonic_windup=solver.active,
    )
    return coeffs, q_release, qd_release, v_ach, speed, v_cmd, rel


def release_box_from_table(arm, table, margin=0.10):
    """
    Safe release box derived from the pose table's OWN release locus.

    The previous hardcoded box ([0.2,-0.5,0.1]..[0.9,0.5,0.9]) was sized for the
    old low IK+pinv throw and does NOT contain the overhead release, which sits
    almost directly above the base at r~0.035 m, z~1.137 m -- so `throw` would
    have refused every valid overhead plan. Deriving the box from the table makes
    the check meaningful instead of arbitrary: it asserts the planned release is
    where THIS calibrated table says it should be, and still catches a release
    that has wandered somewhere unexpected.
    """
    pts = []
    for e in table:
        q = np.asarray(e["q"], dtype=float)
        for li, jid in enumerate(arm._joint_ids):
            p.resetJointState(arm._arm_id, jid, float(q[li]), physicsClientId=arm._cid)
        pts.append(np.array(p.getLinkState(
            arm._arm_id, arm._ee_link, computeForwardKinematics=True,
            physicsClientId=arm._cid)[4]))
    for li, jid in enumerate(arm._joint_ids):
        p.resetJointState(arm._arm_id, jid, float(arm._q_neutral[li]), 0.0,
                          physicsClientId=arm._cid)
    pts = np.array(pts)
    return pts.min(axis=0) - margin, pts.max(axis=0) + margin


def make_limits(profile, speed_scale, release_box=None,
                control_hz=HIGH_LEVEL_MAX_HZ, arm=None, q_margin=0.10):
    """
    Safety envelope for one run. Pass `arm` whenever one exists.

    The soft joint envelope used to be a flat +-6.10 on every joint, which is
    right for the four CONTINUOUS joints (+-6.28) and badly wrong for the three
    LIMITED ones: joint 5's real range is +-2.09, so the guard was 2.9x too
    loose and would have passed a trajectory driving it nearly three times past
    its stop. The shipped throw happens to peak at 57% of the real ranges, so
    this was a latent hole rather than an active bug -- but it is exactly the
    check that is supposed to catch a bad new plan.

    With `arm`, the envelope comes from the URDF per joint, inset by `q_margin`.
    Without one (connect / gripper, where no trajectory is checked) it falls
    back to the flat value.
    """
    qd_max = np.array(profile.qd_max, float)
    if arm is not None:
        q_soft_lo = np.asarray(arm._q_lo, float) + q_margin
        q_soft_hi = np.asarray(arm._q_hi, float) - q_margin
    else:
        q_soft_lo = -6.10 * np.ones(len(qd_max))
        q_soft_hi = 6.10 * np.ones(len(qd_max))
    kw = {}
    if release_box is not None:
        kw["release_box_lo"], kw["release_box_hi"] = release_box
    if profile.tau_max is not None:
        kw["tau_max"] = np.array(profile.tau_max, float)
    return SafetyLimits(
        qd_max=qd_max, q_soft_lo=q_soft_lo, q_soft_hi=q_soft_hi,
        speed_scale=speed_scale, control_hz=control_hz,
        # The trained overhead trajectory is already torque-stretched to ~8.5 s
        # at speed_scale=1.0, and a 0.15 rehearsal stretches it to ~57 s. The old
        # 8 s cap refused BOTH. The cap still exists to catch a runaway plan, but
        # has to be sized for the real trajectory.
        max_traj_seconds=180.0,
        **kw,
    )


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def cmd_plan(args):
    arm, profile, cid = build_arm(args.robot)
    pol, cfg = load_policy(args.log_path, None)
    coeffs, q_rel, qd_rel, v_ach, speed, v_cmd, rel = plan_throw_for_target(
        arm, profile, cfg, pol, args.target,
        opt_pose=args.opt_pose, u_cap=args.u_cap)
    table = load_pose_table(cfg, args.opt_pose)
    box = release_box_from_table(arm, table) if table else None
    limits = make_limits(profile, args.speed_scale, release_box=box, arm=arm)
    ex = HardwareThrowExecutor(limits, dry_run=True)
    print(f"\n=== PLAN (dry-run) robot={args.robot} target={args.target} ===")
    print(f"policy release speed: {speed:.3f} m/s   v_cmd EE: {np.round(v_cmd,3)}")
    print(f"v achievable (after qd clip): {np.round(v_ach,3)}  |v|={np.linalg.norm(v_ach):.3f}")
    print(f"q_release (rad): {np.round(q_rel,3)}")
    print(f"qd_release (rad/s): {np.round(qd_rel,3)}   qd_max: {np.round(profile.qd_max,3)}")
    print(f"release pos in safe box: {ex.check_release_pos(rel)}")
    ok, report = ex.precheck(coeffs, arm, release_speed=np.linalg.norm(v_ach))
    print("--- trajectory precheck ---")
    print(report)
    print(f"PRECHECK: {'PASS' if ok else 'FAIL -- do NOT run on hardware'}")
    p.disconnect(cid)
    return 0 if ok else 2


def cmd_connect(args):
    profile = get_robot_profile(args.robot)
    limits = make_limits(profile, args.speed_scale)
    with HardwareThrowExecutor(limits, dry_run=not args.arm, ip=args.ip) as ex:
        q, qd = ex.backend.read_joint_state()
        print("joint positions (rad):", np.round(q, 4))
        print("joint velocities (rad/s):", np.round(qd, 4))
    return 0


def cmd_home(args):
    arm, profile, cid = build_arm(args.robot)
    limits = make_limits(profile, args.speed_scale, arm=arm)
    with HardwareThrowExecutor(limits, dry_run=not args.arm, ip=args.ip) as ex:
        ex.home(arm, np.array(profile.q_neutral, float), duration=args.duration)
    p.disconnect(cid)
    return 0


def cmd_gripper(args):
    profile = get_robot_profile(args.robot)
    limits = make_limits(profile, args.speed_scale)
    with HardwareThrowExecutor(limits, dry_run=not args.arm, ip=args.ip) as ex:
        ex.set_gripper(closed=args.close)
        print(f"gripper -> {'CLOSE' if args.close else 'OPEN'} commanded")
    return 0


def cmd_throw(args):
    if args.arm and not args.confirm:
        print("REFUSED: real motion requires --confirm (workspace clear, ball "
              "secured, e-stop in hand). Aborting.")
        return 2
    arm, profile, cid = build_arm(args.robot)
    pol, cfg = load_policy(args.log_path, None)
    coeffs, q_rel, qd_rel, v_ach, speed, v_cmd, rel = plan_throw_for_target(
        arm, profile, cfg, pol, args.target,
        opt_pose=args.opt_pose, u_cap=args.u_cap)
    table = load_pose_table(cfg, args.opt_pose)
    box = release_box_from_table(arm, table) if table else None
    limits = make_limits(profile, args.speed_scale, release_box=box, arm=arm)
    with HardwareThrowExecutor(limits, dry_run=not args.arm, ip=args.ip) as ex:
        if not ex.check_release_pos(rel):
            raise RuntimeError(f"release pos {rel} outside safe box; abort.")
        ok, report = ex.precheck(coeffs, arm, release_speed=np.linalg.norm(v_ach))
        print(report)
        if not ok:
            raise RuntimeError("precheck FAILED; refuse to move.")
        print(f"\n>>> speed_scale={args.speed_scale} "
              f"({'REAL THROW' if args.speed_scale >= 0.99 else 'SLOW REHEARSAL'})")
        ex.set_gripper(closed=True)          # grasp
        ex.home(arm, np.array(profile.q_neutral, float), duration=args.duration)
        ex.rehearse_or_throw(coeffs, arm)
    p.disconnect(cid)
    return 0


# --------------------------------------------------------------------------- #
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--robot", default="kinova_gen3")
        sp.add_argument("--arm", action="store_true", help="talk to the REAL arm (default: dry-run)")
        sp.add_argument("--ip", default="192.168.1.101")
        sp.add_argument("--speed_scale", type=float, default=0.15)

    def throw_planning(sp):
        sp.add_argument("--log_path", required=True)
        sp.add_argument("--target", type=float, nargs=2, required=True)
        sp.add_argument(
            "--opt_pose", default=None,
            help="pose-table path override for checkpoints whose config_log.pkl "
                 "predates the trainer recording 'opt_pose' (e.g. "
                 "throw_pose_table.npy). Without the right table the planner "
                 "silently falls back to a DIFFERENT, weaker IK+pinv throw.",
        )
        sp.add_argument(
            "--u_cap", type=float, default=None,
            help="hard ceiling on commanded release speed (m/s). The shipped "
                 "Gen3 table's kinematic max 1.628 is NOT follow-through "
                 "recoverable; 1.60 is the measured safe cap.",
        )

    sp = sub.add_parser("plan"); common(sp); throw_planning(sp)
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("connect"); common(sp); sp.set_defaults(func=cmd_connect)

    sp = sub.add_parser("home"); common(sp)
    sp.add_argument("--duration", type=float, default=4.0); sp.set_defaults(func=cmd_home)

    sp = sub.add_parser("gripper"); common(sp)
    g = sp.add_mutually_exclusive_group(required=True)
    g.add_argument("--open", dest="close", action="store_false")
    g.add_argument("--close", dest="close", action="store_true")
    sp.set_defaults(func=cmd_gripper)

    sp = sub.add_parser("throw"); common(sp); throw_planning(sp)
    sp.add_argument("--duration", type=float, default=4.0)
    sp.add_argument("--confirm", action="store_true", help="assert workspace clear + e-stop in hand")
    sp.set_defaults(func=cmd_throw)
    return ap


if __name__ == "__main__":
    args = build_parser().parse_args()
    sys.exit(args.func(args))
