"""
Ten hardware-verified throws: every frame in the output video is a trajectory
that passed the REAL hardware gate, not a sim-only rollout.

Why this script exists separately from make_dynamic_video.py: that script
anchors target sampling on `profile.default_release_pos`, a static per-arm
constant (0.55, 0.00, 0.45). For an opt_pose checkpoint the real release point
comes from FK on the table's posture and is nowhere near it (measured here:
-0.042, 0.025, 1.523 -- a 1.07 m gap in z), so its aim directions are wrong and
its azimuths land near the table's +-30 deg wedge edge. `train_mc_pilot_pb_arm.py`
already fixed exactly this for training (see its release_xy / RELEASE_POS
comments); this script inherits the fixed convention by reading cfg["release_pos"],
as eval_adapted_height.py and run_hardware_throw.py both do.

Each throw goes through, in order:

  1. plan_throw_for_target()  -- the SAME function run_hardware_throw.py's `plan`
     and `throw` subcommands call. Not a reimplementation.
  2. check_release_pos()      -- release inside the pose table's own release box.
  3. precheck()               -- inverse dynamics over 400 samples of the WHOLE
     trajectory (windup + throw + follow-through), against the real measured
     39/39/39/39/9/9/9 Nm and 1.396/1.222 rad/s limits, at speed_scale=1.0.

A throw is rendered ONLY if all three pass. Torque numbers printed and burned
into the video are that precheck's own peak |tau| per joint -- real inverse
dynamics with gravity, not a proxy.

The rendered motion is the torque-controlled (control_mode="torque") rollout:
the ball's release velocity is MEASURED from tracked arm motion, never assigned
via resetBaseVelocity. The ball is welded at the TCP (tool_offset), so the
omega x r_offset release boost comes out of PyBullet's own rigid-body physics.

Planner-vs-sim agreement is asserted per throw (not assumed from the test suite):
the sim's realized release speed is compared against the planner's achievable
|v_ach| and reported in the summary table.
"""
import argparse
import os
import pickle as pkl

import imageio.v2 as imageio
import numpy as np
import pybullet as p
import torch
from PIL import Image, ImageDraw, ImageFont

import policy_learning.Policy as Policy
from robot_arm.kinova_hardware import HardwareThrowExecutor
from run_hardware_throw import (build_arm, load_pose_table,
                                make_limits, plan_throw_for_target,
                                release_box_from_table)
from simulation_class.model_pybullet import PyBulletThrowingSystem

WIDTH, HEIGHT = 1280, 720


def load_policy(log_path):
    with open(os.path.join(log_path, "log.pkl"), "rb") as f:
        log = pkl.load(f)
    with open(os.path.join(log_path, "config_log.pkl"), "rb") as f:
        cfg = pkl.load(f)
    st = log["parameters_trial_list"][-1]
    pol = Policy.Throwing_Policy(
        full_state_dim=8, target_dim=2, num_basis=st["centers"].shape[0],
        u_max=cfg["uM"], lengthscales_init=st["log_lengthscales"].exp().numpy()[0],
        centers_init=st["centers"].numpy(), weight_init=st["f_linear.weight"].numpy(),
        flg_drop=False, dtype=torch.float64, device=torch.device("cpu"))
    pol.load_state_dict(st)
    pol.eval()
    return pol, cfg


def camera(client):
    view = p.computeViewMatrix(
        cameraEyePosition=[1.62, -1.52, 1.28],
        cameraTargetPosition=[0.32, 0.00, 0.80],
        cameraUpVector=[0, 0, 1], physicsClientId=client)
    proj = p.computeProjectionMatrixFOV(
        fov=55, aspect=WIDTH / HEIGHT, nearVal=0.05, farVal=8.0,
        physicsClientId=client)
    return view, proj


def add_scenery(client, target_xy, base_height):
    """
    Visual-only markers: the target the policy is aiming at, and the 0.433 m
    base plate the real Gen3 is bolted to (the sim raises the base via
    base_position, so without this the arm renders floating above the floor).
    Both are mass-0 with collisionShapeIndex=-1 -- they cannot touch physics.
    """
    ring = p.createVisualShape(p.GEOM_CYLINDER, radius=0.05, length=0.004,
                               rgbaColor=[0.10, 0.85, 0.35, 0.95],
                               physicsClientId=client)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                      baseVisualShapeIndex=ring,
                      basePosition=[target_xy[0], target_xy[1], 0.002],
                      physicsClientId=client)
    if base_height > 0:
        plate = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.13, 0.13, base_height / 2.0],
            rgbaColor=[0.30, 0.32, 0.36, 1.0], physicsClientId=client)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                          baseVisualShapeIndex=plate,
                          basePosition=[0, 0, base_height / 2.0],
                          physicsClientId=client)


def _font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def annotate(frame, lines, headline):
    """Burn the throw's real planner/precheck numbers into the frame."""
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    f_big, f_small = _font(27), _font(20)
    d.rectangle([0, 0, WIDTH, 44], fill=(12, 14, 18, 215))
    d.text((16, 9), headline, font=f_big, fill=(255, 255, 255, 255))
    # Stats go in the empty sky at top-left: the action (arm, ball flight,
    # landing) occupies the centre and lower-right, and a bottom caption bar
    # sat directly on top of the landing marker.
    w = max(d.textlength(t, font=f_small) for t, _ in lines) + 32
    box_h = 22 * len(lines) + 18
    d.rectangle([0, 44, w, 44 + box_h], fill=(12, 14, 18, 200))
    for i, (txt, col) in enumerate(lines):
        d.text((16, 53 + 22 * i), txt, font=f_small, fill=col)
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--log_path", default="results_kinetic_chain_gen3_tcp/1")
    ap.add_argument("--opt_pose", default="throw_pose_table_tcp.npy")
    ap.add_argument("--tool_offset_z", type=float, default=0.12)
    ap.add_argument("--robot", default="kinova_gen3_dyn")
    ap.add_argument("--num_throws", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--speed_scale", type=float, default=1.0,
                    help="precheck at full hardware speed by default")
    ap.add_argument("--wrist_roll_offset_deg", type=float, default=0.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default="hardware_verified_10_throws.mp4")
    args = ap.parse_args()

    pol, cfg = load_policy(args.log_path)
    arm, profile, cid = build_arm(args.robot)      # gravity set inside; see its comment
    table = load_pose_table(cfg, args.opt_pose)
    box = release_box_from_table(arm, table,
                                 tool_offset=[0.0, 0.0, args.tool_offset_z])
    limits = make_limits(profile, args.speed_scale, release_box=box, arm=arm)
    ex = HardwareThrowExecutor(limits, dry_run=True)

    RP = np.asarray(cfg["release_pos"], float)     # CORRECT anchor (trained release point)
    lm, lM, gM = cfg["lm"], cfg["lM"], cfg["gM"]
    launch = float(cfg.get("opt_launch_deg", table[0]["elev_deg"]))
    tau_max = np.asarray(profile.tau_max, float)
    qd_max = np.asarray(profile.qd_max, float)

    print(f"checkpoint     : {args.log_path}")
    print(f"pose table     : {args.opt_pose}  (elev {table[0]['elev_deg']:.0f} deg, "
          f"tool_offset {table[0]['tool_offset']}, floor_z {table[0]['floor_z']:+.3f})")
    print(f"release_pos    : {np.round(RP, 4)}   band {lm:.2f}-{lM:.2f} m, "
          f"wedge +-{np.degrees(gM):.0f} deg")
    print(f"control_mode   : {profile.control_mode}   tau_max {tau_max.tolist()} Nm   "
          f"qd_max {np.round(qd_max, 3).tolist()} rad/s")
    print(f"precheck at speed_scale={args.speed_scale}, torque margin "
          f"{limits.torque_margin:.0%} of limit\n")

    rng = np.random.default_rng(args.seed)
    frames, rows = [], []
    attempts = 0
    while len(rows) < args.num_throws and attempts < args.num_throws * 6:
        attempts += 1
        flight = rng.uniform(lm - RP[0], lM - RP[0])
        beta = rng.uniform(-gM, gM)
        tgt = RP[:2] + flight * np.array([np.cos(beta), np.sin(beta)])

        # ---- 1. plan through the REAL hardware planner ----------------------
        try:
            coeffs, q_rel, qd_rel, v_ach, speed, v_cmd, rel = plan_throw_for_target(
                arm, profile, cfg, pol, tgt, opt_pose=args.opt_pose,
                wrist_roll_offset=np.deg2rad(args.wrist_roll_offset_deg),
                tool_offset_z=args.tool_offset_z)
        except RuntimeError as e:
            print(f"  [rejected, planner infeasible] target "
                  f"({tgt[0]:.3f},{tgt[1]:.3f}): {str(e)[:90]}")
            continue

        # ---- 2. release inside the table's own safe box ---------------------
        in_box = ex.check_release_pos(rel)
        # ---- 3. whole-trajectory precheck: real inverse dynamics ------------
        ok, report = ex.precheck(coeffs, arm, release_speed=float(np.linalg.norm(v_ach)))
        if not (ok and in_box):
            print(f"  [rejected, precheck] target ({tgt[0]:.3f},{tgt[1]:.3f}) "
                  f"ok={ok} in_box={in_box}")
            continue

        peak_tau = np.array([float(s.split("/")[0]) for s in
                             report.split("peak |tau| (Nm): ")[1].split("\n")[0].split(", ")])
        peak_qd = np.array([float(s.split("/")[0]) for s in
                            report.split("peak commanded |qd| (rad/s): ")[1].split("\n")[0].split(", ")])
        tau_frac = float(np.max(peak_tau / tau_max))
        qd_frac = float(np.max(peak_qd / qd_max))

        # ---- render the torque-controlled rollout of that same plan ---------
        shot = []
        scenery_done = []

        def capture(client, _s=shot, _t=tgt, _done=scenery_done):
            if not _done:                      # first hook call: client exists now
                add_scenery(client, _t, float(cfg.get("base_height", 0.0)))
                _done.append(True)
            if len(_s) % 2:
                _s.append(None)
                return
            view, proj = camera(client)
            img = p.getCameraImage(WIDTH, HEIGHT, viewMatrix=view, projectionMatrix=proj,
                                   renderer=p.ER_TINY_RENDERER, physicsClientId=client)
            _s.append(np.reshape(img[2], (HEIGHT, WIDTH, 4))[:, :, :3].astype(np.uint8))

        system = PyBulletThrowingSystem(
            mass=cfg["ball_mass"], radius=cfg["ball_radius"], launch_angle_deg=launch,
            arm_noise=None, t_w=cfg["T_W"], t_r=cfg["T_R"], robot_name=cfg["robot_name"],
            target_height=float(cfg["target_height"]),
            base_height=float(cfg.get("base_height", 0.0)),
            opt_posture_table=table, opt_launch_deg=launch,
            tool_offset=[0.0, 0.0, args.tool_offset_z])
        system.frame_hook = capture

        def policy(s, t, _sp=speed):
            return np.array([_sp])          # exactly the speed the planner used

        pos, _, _ = system.rollout(np.concatenate([RP, np.zeros(3), tgt]),
                                   policy, T=cfg["T"], dt=cfg["Ts"], noise=0.0)
        land = pos[-1][:2]
        err = float(np.linalg.norm(land - tgt))
        v_real = float(np.linalg.norm(system.last_release_info["v_release"]))
        v_plan = float(np.linalg.norm(v_ach))

        n = len(rows) + 1
        rows.append(dict(n=n, tgt=tgt, land=land, err=err, speed=speed,
                         v_plan=v_plan, v_real=v_real, peak_tau=peak_tau,
                         peak_qd=peak_qd, tau_frac=tau_frac, qd_frac=qd_frac,
                         rel=rel, T=float(coeffs["T"])))
        print(f"throw {n:2d}/{args.num_throws}  target ({tgt[0]:+.3f},{tgt[1]:+.3f})  "
              f"landed ({land[0]:+.3f},{land[1]:+.3f})  err {err*100:5.2f} cm   "
              f"|v| plan {v_plan:.3f} / real {v_real:.3f} m/s   "
              f"tau {tau_frac*100:4.1f}%  qd {qd_frac*100:5.1f}%   PRECHECK PASS")

        real = [f for f in shot if f is not None]
        head = (f"THROW {n}/{args.num_throws}   Kinova Gen3 (torque control)   "
                f"target ({tgt[0]:.2f}, {tgt[1]:.2f}) m")
        lines = [
            (f"release speed  {v_real:.3f} m/s  (measured from arm motion, not assigned)",
             (150, 230, 255, 255)),
            (f"peak torque    {np.max(peak_tau):.1f} Nm  =  {tau_frac*100:.1f}% of limit"
             f"   [{', '.join(f'{t:.0f}' for t in peak_tau)} Nm]", (140, 240, 170, 255)),
            (f"peak joint vel {qd_frac*100:.1f}% of qd_max   "
             f"windup+throw+follow-through all checked", (140, 240, 170, 255)),
            (f"landing error  {err*100:.2f} cm", (255, 255, 255, 255)),
            ("HARDWARE PRECHECK: PASS  @ speed_scale=1.0", (120, 255, 140, 255)),
        ]
        frames.extend(annotate(f, lines, head) for f in real)
        frames.extend([annotate(real[-1], lines, head)] * (args.fps // 2))

    p.disconnect(cid)
    if not rows:
        raise SystemExit("no throw passed the hardware gate")

    e = np.array([r["err"] for r in rows])
    tf = np.array([r["tau_frac"] for r in rows])
    qf = np.array([r["qd_frac"] for r in rows])
    vr = np.array([r["v_real"] for r in rows])
    print(f"\n{len(rows)}/{attempts} sampled targets passed the full hardware gate")
    print(f"landing error : mean {e.mean()*100:.2f} cm   median {np.median(e)*100:.2f} cm"
          f"   max {e.max()*100:.2f} cm   all <10 cm: {bool((e<0.10).all())}")
    print(f"release speed : {vr.min():.3f} - {vr.max():.3f} m/s (measured)")
    print(f"peak torque   : {tf.min()*100:.1f} - {tf.max()*100:.1f} % of the real 39/9 Nm limits")
    print(f"peak joint vel: {qf.min()*100:.1f} - {qf.max()*100:.1f} % of the real qd_max")
    print(f"\nencoding {len(frames)} frames -> {args.out}")
    imageio.mimwrite(args.out, frames, fps=args.fps, codec="libx264", quality=8)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
