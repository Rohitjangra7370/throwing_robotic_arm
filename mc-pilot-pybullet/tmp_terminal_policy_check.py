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
from robot_arm.robot_profiles import get_robot_profile
from simulation_class.model import ThrowingSystem


def build_parser():
    parser = argparse.ArgumentParser("Temporary terminal-only throw checker")
    parser.add_argument("--log_path", type=str, required=True, help="path to results/<seed>/")
    parser.add_argument("--seed", type=int, default=42, help="target sampling seed")
    parser.add_argument("--num_trials", type=int, default=5, help="number of terminal-only throws to run")
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


def infer_robot_name(log_path, cfg):
    robot_name = cfg.get("robot_name")
    if robot_name:
        return robot_name
    log_path_lower = os.path.normpath(log_path).lower()
    if "franka_panda" in log_path_lower:
        return "franka_panda"
    if "xarm6" in log_path_lower:
        return "xarm6"
    return "kuka_iiwa"


def speed_to_velocity(speed, release_pos, target_xy, launch_angle_deg=35.0):
    dx = target_xy[0] - release_pos[0]
    dy = target_xy[1] - release_pos[1]
    azimuth = np.arctan2(dy, dx)
    alpha = np.deg2rad(launch_angle_deg)
    return np.array(
        [
            speed * np.cos(alpha) * np.cos(azimuth),
            speed * np.cos(alpha) * np.sin(azimuth),
            speed * np.sin(alpha),
        ],
        dtype=float,
    )


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
    robot_name = infer_robot_name(args.log_path, cfg)
    profile = get_robot_profile(robot_name)
    release_time = float(profile.timing[1])

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
    target_cfg = {
        "lm": float(cfg["lm"]),
        "lM": float(cfg["lM"]),
        "gM": float(cfg["gM"]),
        "target_x": args.target_x,
        "target_y": args.target_y,
    }

    throwing_system = ThrowingSystem(
        mass=0.0577,
        radius=0.0327,
        launch_angle_deg=35.0,
    )
    speeds = []
    errors = []
    hits = 0

    print(f"robot={robot_name}")
    print(f"log_path={args.log_path}")
    print(f"hit_radius={args.hit_radius * 100.0:.1f} cm")

    for throw_idx in range(args.num_trials):
        target = sample_target(target_cfg, rng)
        s0 = np.concatenate([release_pos, np.zeros(3), target])

        with torch.no_grad():
            speed = float(
                policy_obj(
                    torch.tensor(s0, dtype=dtype, device=device).unsqueeze(0),
                    t=0,
                    p_dropout=0.0,
                ).item()
            )
        speed = min(max(speed, 0.0), uM)
        v_cmd = speed_to_velocity(speed, release_pos, target)

        _, _, clean_states = throwing_system.rollout(
            s0=s0,
            policy=lambda _state, _t, speed=speed: np.array([speed]),
            T=float(cfg.get("T", 0.60)),
            dt=float(cfg.get("Ts", 0.02)),
            noise=0.0,
        )

        landing = clean_states[-1, 0:3]
        error = float(np.linalg.norm(landing[0:2] - target))
        hit = error <= args.hit_radius

        speeds.append(speed)
        errors.append(error)
        hits += int(hit)

        print(f"\nThrow {throw_idx + 1}/{args.num_trials}: target=({target[0]:.3f}, {target[1]:.3f})")
        print(f"Policy speed: {speed:.3f} m/s, v_cmd: {np.round(v_cmd, 3)}")
        print(f"  Released at t={release_time:.3f}s with |v|={np.linalg.norm(v_cmd):.3f} m/s")
        print(
            f"  Landing: ({landing[0]:.3f}, {landing[1]:.3f}), "
            f"error={error * 100.0:.1f}cm {'HIT' if hit else 'MISS'}"
        )

    miss_rate = 100.0 * (1.0 - hits / max(args.num_trials, 1))
    hit_rate = 100.0 * hits / max(args.num_trials, 1)
    errors_np = np.asarray(errors, dtype=float)
    speeds_np = np.asarray(speeds, dtype=float)

    print("\nSummary")
    print(f"  Throws: {args.num_trials}")
    print(f"  Avg speed: {speeds_np.mean():.3f} m/s")
    print(f"  Speed range: [{speeds_np.min():.3f}, {speeds_np.max():.3f}] m/s")
    print(f"  Hit rate: {hit_rate:.1f}% ({hits}/{args.num_trials})")
    print(f"  Miss rate: {miss_rate:.1f}%")
    print(f"  Mean error: {errors_np.mean() * 100.0:.2f} cm")
    print(f"  Median error: {np.median(errors_np) * 100.0:.2f} cm")
    print(f"  Max error: {errors_np.max() * 100.0:.2f} cm")


if __name__ == "__main__":
    main()
