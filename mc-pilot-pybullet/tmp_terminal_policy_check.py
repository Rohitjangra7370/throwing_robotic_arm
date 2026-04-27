"""
Temporary terminal-only checker for a trained MC-PILOT throwing policy.

This intentionally avoids PyBullet and uses the analytic free-flight simulator
from simulation_class/model.py so we can tell whether a run is failing because
of the policy itself or because of PyBullet execution.
"""

import argparse
import os
import pickle as pkl

import numpy as np
import torch

import policy_learning.Policy as Policy
from simulation_class.model import ThrowingSystem


def build_parser():
    parser = argparse.ArgumentParser("Temporary terminal-only throw checker")
    parser.add_argument("--log_path", type=str, required=True, help="path to results/<seed>/")
    parser.add_argument("--seed", type=int, default=42, help="target sampling seed")
    parser.add_argument("--target_x", type=float, default=None, help="optional fixed target x")
    parser.add_argument("--target_y", type=float, default=None, help="optional fixed target y")
    parser.add_argument("--hit_radius", type=float, default=0.10, help="hit threshold in meters")
    return parser


def sample_target(cfg, rng):
    if cfg["target_x"] is not None and cfg["target_y"] is not None:
        return np.array([cfg["target_x"], cfg["target_y"]], dtype=float)

    dist = rng.uniform(cfg["lm"], cfg["lM"])
    angle = rng.uniform(-cfg["gM"], cfg["gM"])
    return np.array([dist * np.cos(angle), dist * np.sin(angle)], dtype=float)


def main():
    args = build_parser().parse_args()

    log_file = os.path.join(args.log_path, "log.pkl")
    cfg_file = os.path.join(args.log_path, "config_log.pkl")
    with open(log_file, "rb") as f:
        log = pkl.load(f)
    with open(cfg_file, "rb") as f:
        cfg = pkl.load(f)

    policy_state = log["parameters_trial_list"][-1]
    release_pos = np.array(cfg["release_pos"], dtype=float)
    uM = float(cfg["uM"])
    Nb = int(cfg.get("Nb", 250))

    dtype = torch.float64
    device = torch.device("cpu")
    policy_obj = Policy.Throwing_Policy(
        full_state_dim=8,
        target_dim=2,
        num_basis=Nb,
        u_max=uM,
        lengthscales_init=policy_state["log_lengthscales"].exp().numpy()[0],
        centers_init=policy_state["centers"].numpy(),
        weight_init=policy_state["f_linear.weight"].numpy(),
        flg_drop=False,
        dtype=dtype,
        device=device,
    )
    policy_obj.load_state_dict(policy_state)
    policy_obj.eval()

    rng = np.random.default_rng(args.seed)
    target = sample_target(
        {
            "lm": float(cfg["lm"]),
            "lM": float(cfg["lM"]),
            "gM": float(cfg["gM"]),
            "target_x": args.target_x,
            "target_y": args.target_y,
        },
        rng,
    )

    s0 = np.concatenate([release_pos, np.zeros(3), target])

    with torch.no_grad():
        speed = float(
            policy_obj(torch.tensor(s0, dtype=dtype, device=device).unsqueeze(0), t=0, p_dropout=0.0).item()
        )
    speed = min(max(speed, 0.0), uM)

    throwing_system = ThrowingSystem(
        mass=0.0577,
        radius=0.0327,
        launch_angle_deg=35.0,
    )

    _, inputs, clean_states = throwing_system.rollout(
        s0=s0,
        policy=lambda _state, _t: np.array([speed]),
        T=float(cfg.get("T", 0.60)),
        dt=float(cfg.get("Ts", 0.02)),
        noise=0.0,
    )

    landing = clean_states[-1, 0:3]
    error = float(np.linalg.norm(landing[0:2] - target))
    hit = error <= args.hit_radius

    print(f"robot={cfg.get('robot_name', 'unknown')}")
    print(f"target=({target[0]:.3f}, {target[1]:.3f})")
    print(f"predicted_speed={speed:.6f}")
    print(f"landing=({landing[0]:.3f}, {landing[1]:.3f}, {landing[2]:.3f})")
    print(f"error_cm={error * 100.0:.2f}")
    print("HIT" if hit else "MISS")
    print(f"input_samples={inputs.shape[0]}")


if __name__ == "__main__":
    main()
