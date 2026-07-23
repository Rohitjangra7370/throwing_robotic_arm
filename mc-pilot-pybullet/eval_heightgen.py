"""
Real-physics evaluation for a height-generalized (9-D state) kinova policy.

Unlike the earlier (flawed) naive-transfer height sweep in
eval_generalization.py -- which fed a HEIGHT-BLIND policy a ground-XY target
and just truncated the arc early -- this evaluator gives the policy the true
3-D target (Px, Py, h) it was trained to consume, and measures landing error
in the (X, Y) plane AT that target's height. This is the honest test of
whether height-conditioning actually works.
"""

import argparse
import os
import pickle as pkl

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import policy_learning.Policy as Policy
from robot_arm.robot_profiles import get_robot_profile
from simulation_class.model_pybullet import PyBulletThrowingSystem


def load_policy_9d(log_path):
    with open(os.path.join(log_path, "log.pkl"), "rb") as f:
        log = pkl.load(f)
    with open(os.path.join(log_path, "config_log.pkl"), "rb") as f:
        cfg = pkl.load(f)
    state = log["parameters_trial_list"][-1]
    policy_obj = Policy.Throwing_Policy(
        full_state_dim=9,
        target_dim=3,
        num_basis=state["centers"].shape[0],
        u_max=cfg["uM"],
        lengthscales_init=state["log_lengthscales"].exp().numpy()[0],
        centers_init=state["centers"].numpy(),
        weight_init=state["f_linear.weight"].numpy(),
        flg_drop=False,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    policy_obj.load_state_dict(state)
    policy_obj.eval()
    return policy_obj, cfg


def make_targets_3d(cfg, release_xy, n, rng):
    lm, lM, gM = cfg["lm"], cfg["lM"], cfg.get("gM", np.pi / 6)
    H_MAX = cfg.get("H_MAX", 0.45)
    height_slope = cfg.get("height_slope", None)
    flight_mode = bool(cfg.get("flight_targets", False))
    f_lo = lm - release_xy[0]
    f_hi0 = lM - release_xy[0]
    slope = height_slope if height_slope is not None else 0.0

    targets = []
    for _ in range(n):
        h = rng.uniform(0.0, H_MAX)
        if flight_mode:
            f_hi_h = max(f_lo + 0.02, f_hi0 - slope * h)
            flight = rng.uniform(f_lo, f_hi_h)
            beta = rng.uniform(-gM, gM)
            xy = release_xy + flight * np.array([np.cos(beta), np.sin(beta)])
        else:
            lM_h = max(lm + 0.1, lM - 1.1 * h)
            dist = rng.uniform(lm, lM_h)
            angle = rng.uniform(-gM, gM)
            xy = np.array([dist * np.cos(angle), dist * np.sin(angle)])
        targets.append(np.array([xy[0], xy[1], h]))
    return np.array(targets)


def run_throws_3d(policy_obj, targets_3d, release_pos, t_w, t_r, robot_name,
                  mass=0.0577, radius=0.0327, opt_posture_table=None,
                  opt_launch_deg=35.0, u_cap=None):
    """targets_3d: (N,3) array of (Px, Py, h). System's target_height is set
    per-throw so the trajectory is cut at the target's own plane, and the
    9-D policy input carries h so it can actually condition on it.

    opt_posture_table must be passed for policies trained with --opt_pose:
    without it the system falls back to the IK+pinv release and evaluates a
    completely different throw from the one that was trained."""
    errs = []
    for tgt in targets_3d:
        px, py, h = tgt
        system = PyBulletThrowingSystem(
            mass=mass, radius=radius, robot_name=robot_name,
            t_w=t_w, t_r=t_r, target_height=float(h),
            opt_posture_table=opt_posture_table,
            opt_launch_deg=opt_launch_deg,
            launch_angle_deg=(opt_launch_deg if opt_posture_table is not None else 35.0),
        )

        def policy(s, t):
            with torch.no_grad():
                inp = torch.tensor(np.asarray(s, dtype=float),
                                   dtype=torch.float64).unsqueeze(0)
                u = float(policy_obj(inp, t=0, p_dropout=0.0).item())
            # Safety limiter, as a real deployment would have: the posture
            # table's kinematic max speed is NOT necessarily follow-through
            # feasible (measured 1.611 vs the table's 1.628 for the overhead
            # release), and plan_throw correctly refuses the infeasible ones.
            if u_cap is not None:
                u = min(u, u_cap)
            return np.array([u])

        s0 = np.concatenate([release_pos, np.zeros(3), np.array([px, py, h])])
        _, _, clean = system.rollout(s0, policy, T=2.0, dt=0.02, noise=0.0)
        land_xy = clean[-1, 0:2]
        errs.append(float(np.linalg.norm(land_xy - np.array([px, py]))))
    return np.array(errs)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--log_path", type=str, default="results_mc_pilot_pb_A_kinova_gen3_hgen/1")
    ap.add_argument("--robot", type=str, default="kinova_gen3")
    ap.add_argument("--out", type=str, default="results_generalization")
    ap.add_argument("--num_throws", type=int, default=30)
    ap.add_argument("--seed", type=int, default=555)
    ap.add_argument("--u_cap", type=float, default=None,
                    help="clamp commanded release speed (follow-through-safe max)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    policy_obj, cfg = load_policy_9d(args.log_path)
    profile = get_robot_profile(args.robot)
    release_pos = np.array(profile.default_release_pos, dtype=float)
    t_w, t_r, _ = profile.timing

    # opt_pose policies release from a searched posture, not the profile
    # default -- both the table and the real-FK release point must come from
    # the training config, or the eval measures a different throw.
    opt_table, opt_launch = None, 35.0
    if cfg.get("opt_pose"):
        opt_table = list(np.load(cfg["opt_pose"], allow_pickle=True))
        opt_launch = float(cfg.get("opt_launch_deg", opt_table[0]["elev_deg"]))
        release_pos = np.array(cfg["release_pos"], dtype=float)
        t_w, t_r = cfg.get("T_W", t_w), cfg.get("T_R", t_r)
        print(f"opt_pose eval: table={cfg['opt_pose']}  release={np.round(release_pos,4)}  "
              f"launch={opt_launch:.0f}deg")

    rng = np.random.default_rng(args.seed)
    targets = make_targets_3d(cfg, release_pos[:2], args.num_throws, rng)

    errs = run_throws_3d(policy_obj, targets, release_pos, t_w, t_r, args.robot,
                         opt_posture_table=opt_table, opt_launch_deg=opt_launch,
                         u_cap=args.u_cap)

    print(f"=== Height-generalized eval: {args.log_path} on {args.robot} (n={args.num_throws}) ===")
    print(f"  overall: mean {errs.mean()*100:.2f}cm  max {errs.max()*100:.2f}cm  std {errs.std()*100:.2f}cm")
    H_MAX = cfg.get("H_MAX", 0.45)
    print(f"\n  by height band (H_MAX={H_MAX:.2f}m):")
    for lo in np.arange(0.0, H_MAX, H_MAX / 4):
        band = (targets[:, 2] >= lo) & (targets[:, 2] < lo + H_MAX / 4)
        if band.any():
            print(f"    h in [{lo:.3f},{lo+H_MAX/4:.3f}): mean {errs[band].mean()*100:.2f}cm "
                  f"(n={band.sum()})")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(targets[:, 2] * 100, errs * 100, s=20)
    ax.set_xlabel("target height (cm)")
    ax.set_ylabel("landing error (cm)")
    ax.set_title(f"Height-generalized policy: error vs height ({args.robot})")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"heightgen_{args.robot}.png"), dpi=150)
    np.savez(os.path.join(args.out, f"heightgen_{args.robot}.npz"),
             targets=targets, errs=errs)
    print(f"\nSaved {args.out}/heightgen_{args.robot}.{{npz,png}}")


if __name__ == "__main__":
    main()
