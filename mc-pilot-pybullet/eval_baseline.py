"""
Analytical ballistic baseline (paper Eq. 13) vs the trained MC-PILOT policy.

The paper explicitly benchmarks MC-PILOT against this closed-form baseline
(no drag, no learning) and a model-free NN -- we never ran that comparison
for kinova. This closes that gap.

pi(P) = sqrt( g*d^2 / (2*cos^2(alpha)*(d*tan(alpha) - z_P + z_rel)) )

where d is the horizontal distance from the release point to the target,
alpha is the (fixed) launch elevation, z_rel/z_P are release/target heights.
Real drag is NOT modeled by this formula -- it's expected to systematically
undershoot, which is exactly the "analytical methods are imprecise" point
the paper makes (Section 3.3).
"""

import argparse
import os

import numpy as np
import torch

import policy_learning.Policy as Policy
from robot_arm.robot_profiles import get_robot_profile
from simulation_class.model_pybullet import PyBulletThrowingSystem


def load_policy(log_path):
    import pickle as pkl
    with open(os.path.join(log_path, "log.pkl"), "rb") as f:
        log = pkl.load(f)
    with open(os.path.join(log_path, "config_log.pkl"), "rb") as f:
        cfg = pkl.load(f)
    state = log["parameters_trial_list"][-1]
    common = dict(
        full_state_dim=8, target_dim=2, num_basis=state["centers"].shape[0],
        u_max=cfg["uM"], lengthscales_init=state["log_lengthscales"].exp().numpy()[0],
        centers_init=state["centers"].numpy(), weight_init=state["f_linear.weight"].numpy(),
        flg_drop=False, dtype=torch.float64, device=torch.device("cpu"),
    )
    if cfg.get("residual_physics"):
        policy_obj = Policy.Residual_Throwing_Policy(
            release_pos=np.asarray(cfg["release_pos"], dtype=float),
            launch_angle_deg=cfg.get("launch_angle_deg", 35.0),
            target_height=cfg.get("target_height", 0.0),
            delta_max_frac=cfg.get("delta_max_frac", 0.5),
            **common,
        )
    else:
        policy_obj = Policy.Throwing_Policy(**common)
    policy_obj.load_state_dict(state)
    policy_obj.eval()
    return policy_obj, cfg


def baseline_speed(release_pos, target_xy, launch_angle_deg, target_height=0.0):
    """Paper Eq. 13: closed-form no-drag ballistic release speed."""
    g = 9.81
    alpha = np.deg2rad(launch_angle_deg)
    d = np.linalg.norm(np.asarray(target_xy) - np.asarray(release_pos[:2]))
    z_rel = release_pos[2]
    denom = 2.0 * np.cos(alpha) ** 2 * (d * np.tan(alpha) - target_height + z_rel)
    if denom <= 0:
        return None  # target geometrically unreachable at this angle (too high/close)
    return float(np.sqrt(g * d ** 2 / denom))


def make_targets(cfg, release_xy, n, rng):
    lm, lM, gM = cfg["lm"], cfg["lM"], cfg.get("gM", np.pi / 6)
    flight_mode = bool(cfg.get("flight_targets", False))
    targets = []
    for _ in range(n):
        if flight_mode:
            flight = rng.uniform(lm - release_xy[0], lM - release_xy[0])
            beta = rng.uniform(-gM, gM)
            targets.append(release_xy + flight * np.array([np.cos(beta), np.sin(beta)]))
        else:
            dist = rng.uniform(lm, lM)
            angle = rng.uniform(-gM, gM)
            targets.append(np.array([dist * np.cos(angle), dist * np.sin(angle)]))
    return np.array(targets)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--log_path", type=str, default="results_mc_pilot_pb_A_kinova_gen3/1")
    ap.add_argument("--robot", type=str, default="kinova_gen3")
    ap.add_argument("--num_throws", type=int, default=30)
    ap.add_argument("--seed", type=int, default=246810)
    args = ap.parse_args()

    policy_obj, cfg = load_policy(args.log_path)
    profile = get_robot_profile(args.robot)
    release = np.array(profile.default_release_pos, dtype=float)
    t_w, t_r, _ = profile.timing
    uM, uMin = cfg["uM"], cfg["uMin"]

    rng = np.random.default_rng(args.seed)
    targets = make_targets(cfg, release[:2], args.num_throws, rng)

    # Geometry comes from the checkpoint, never from the evaluator's defaults:
    # evaluating a plate-trained policy with the arm back at floor level throws
    # to a different world than the one it was trained in, and the only symptom
    # is a worse number.
    base_height = float(cfg.get("base_height", 0.0))
    if base_height:
        release = release + np.array([0.0, 0.0, base_height])
        print(f"checkpoint base_height={base_height:.3f} m (arm on a plate)")
    system = PyBulletThrowingSystem(
        robot_name=args.robot, t_w=t_w, t_r=t_r,
        mass=cfg.get("ball_mass", 0.0577),
        radius=cfg.get("ball_radius", 0.0327),
        base_height=base_height,
    )

    def mcpilot_policy(s, t):
        with torch.no_grad():
            inp = torch.tensor(np.asarray(s, dtype=float), dtype=torch.float64).unsqueeze(0)
            return np.array([float(policy_obj(inp, t=0, p_dropout=0.0).item())])

    mcpilot_errs, baseline_errs, baseline_unreachable = [], [], 0
    for tgt in targets:
        s0 = np.concatenate([release, np.zeros(3), tgt])
        _, _, clean = system.rollout(s0, mcpilot_policy, T=2.0, dt=0.02, noise=0.0)
        mcpilot_errs.append(np.linalg.norm(clean[-1, 0:2] - tgt))

        u_base = baseline_speed(release, tgt, launch_angle_deg=35.0)
        if u_base is None:
            baseline_unreachable += 1
            baseline_errs.append(np.nan)
            continue
        u_base_clipped = float(np.clip(u_base, uMin, uM))
        policy_b = lambda s, t, _u=u_base_clipped: np.array([_u])
        _, _, clean_b = system.rollout(s0, policy_b, T=2.0, dt=0.02, noise=0.0)
        baseline_errs.append(np.linalg.norm(clean_b[-1, 0:2] - tgt))

    mcpilot_errs = np.array(mcpilot_errs)
    baseline_errs = np.array(baseline_errs)
    valid = ~np.isnan(baseline_errs)

    print(f"=== Analytical baseline (Eq. 13) vs MC-PILOT, {args.robot} (n={args.num_throws}) ===")
    print(f"MC-PILOT:  mean {mcpilot_errs.mean()*100:.2f}cm  max {mcpilot_errs.max()*100:.2f}cm")
    print(f"Baseline:  mean {baseline_errs[valid].mean()*100:.2f}cm  "
          f"max {baseline_errs[valid].max()*100:.2f}cm  "
          f"({baseline_unreachable} targets geometrically unreachable by Eq.13 at this angle)")
    print(f"Baseline speed vs uM: mean {np.mean([baseline_speed(release, t, 35.0) for t in targets]):.3f} "
          f"m/s (uM={uM:.2f}, so clipping may also degrade the baseline)")


if __name__ == "__main__":
    main()
