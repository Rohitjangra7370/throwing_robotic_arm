"""
Sim2sim gap: replay the trained Kinova policy under (a) the kinematic release
used in training and (b) torque-tracked dynamic release, same 10 targets.

Usage: python3 eval_sim2sim_gap.py --log_path results_mc_pilot_pb_A_kinova_gen3/1
"""

import argparse
import os
import pickle as pkl

import numpy as np
import torch

import policy_learning.Policy as Policy
from robot_arm.robot_profiles import get_robot_profile
from simulation_class.model_pybullet import PyBulletThrowingSystem


def load_policy(log_path):
    with open(os.path.join(log_path, "log.pkl"), "rb") as f:
        log = pkl.load(f)
    with open(os.path.join(log_path, "config_log.pkl"), "rb") as f:
        cfg = pkl.load(f)
    state = log["parameters_trial_list"][-1]
    policy_obj = Policy.Throwing_Policy(
        full_state_dim=8,
        target_dim=2,
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


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--log_path", type=str,
                    default="results_mc_pilot_pb_A_kinova_gen3/1")
    ap.add_argument("--num_throws", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", type=str, default="results_tracking_error")
    args = ap.parse_args()

    policy_obj, cfg = load_policy(args.log_path)
    lm, lM, gM = cfg["lm"], cfg["lM"], cfg.get("gM", np.pi / 6)
    profile = get_robot_profile("kinova_gen3")
    release_pos = np.array(profile.default_release_pos, dtype=float)
    t_w, t_r, _ = profile.timing

    def policy(s, t):
        with torch.no_grad():
            inp = torch.tensor(np.asarray(s, dtype=float),
                               dtype=torch.float64).unsqueeze(0)
            return np.array([float(policy_obj(inp, t=0, p_dropout=0.0).item())])

    rng = np.random.default_rng(args.seed)
    targets = []
    for _ in range(args.num_throws):
        dist = rng.uniform(lm, lM)
        ang = rng.uniform(-gM, gM)
        targets.append([dist * np.cos(ang), dist * np.sin(ang)])
    targets = np.array(targets)

    results = {}
    for label, robot in (("kinematic", "kinova_gen3"), ("dynamic", "kinova_gen3_dyn")):
        system = PyBulletThrowingSystem(robot_name=robot, t_w=t_w, t_r=t_r)
        errs = []
        for tgt in targets:
            s0 = np.concatenate([release_pos, np.zeros(3), tgt])
            _, _, clean = system.rollout(s0, policy, T=2.0, dt=0.02, noise=0.0)
            errs.append(float(np.linalg.norm(clean[-1, 0:2] - tgt)))
        results[label] = np.array(errs)

    print(f"\n=== Sim2sim gap ({args.num_throws} throws, policy {args.log_path}) ===")
    print(f"{'target':>16s} {'kinematic':>10s} {'dynamic':>10s}")
    for i, tgt in enumerate(targets):
        print(f"({tgt[0]:+.2f},{tgt[1]:+.2f})  "
              f"{results['kinematic'][i] * 100:8.2f}cm "
              f"{results['dynamic'][i] * 100:8.2f}cm")
    for label in ("kinematic", "dynamic"):
        e = results[label]
        print(f"{label}: mean {e.mean() * 100:.2f} cm, max {e.max() * 100:.2f} cm")

    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "sim2sim_gap.npz"),
             targets=targets, **results)


if __name__ == "__main__":
    main()
