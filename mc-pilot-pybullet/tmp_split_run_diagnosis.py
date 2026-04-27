"""
Temporary run diagnosis helper.

Splits a saved MC-PILOT run into a few terminal-only checks:
  1. What speeds were actually executed during logged trials?
  2. Did the logged rollouts ever produce meaningful landings?
  3. What does the final saved policy output on a small target grid?
  4. If we ignore PyBullet and use pure ballistic flight, does the final policy hit?
"""

import argparse
import os
import pickle as pkl

import numpy as np
import torch

import policy_learning.Policy as Policy
from simulation_class.model import ThrowingSystem


def build_parser():
    parser = argparse.ArgumentParser("Temporary split run diagnosis")
    parser.add_argument("--log_path", type=str, required=True, help="path to results/<seed>/")
    parser.add_argument("--grid_n", type=int, default=5, help="number of x-samples across target range")
    parser.add_argument("--hit_radius", type=float, default=0.10, help="hit threshold in meters")
    return parser


def load_policy(log, cfg):
    policy_state = log["parameters_trial_list"][-1]
    Nb = int(cfg.get("Nb", 250))
    policy_obj = Policy.Throwing_Policy(
        full_state_dim=8,
        target_dim=2,
        num_basis=Nb,
        u_max=float(cfg["uM"]),
        lengthscales_init=policy_state["log_lengthscales"].exp().numpy()[0],
        centers_init=policy_state["centers"].numpy(),
        weight_init=policy_state["f_linear.weight"].numpy(),
        flg_drop=False,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    policy_obj.load_state_dict(policy_state)
    policy_obj.eval()
    return policy_obj


def final_policy_speed(policy_obj, release_pos, target_xy, u_max):
    s0 = np.concatenate([release_pos, np.zeros(3), target_xy])
    with torch.no_grad():
        speed = float(
            policy_obj(
                torch.tensor(s0, dtype=torch.float64).unsqueeze(0),
                t=0,
                p_dropout=0.0,
            ).item()
        )
    return min(max(speed, 0.0), u_max)


def run_ballistic_check(policy_obj, cfg, target_xy):
    release_pos = np.array(cfg["release_pos"], dtype=float)
    u_max = float(cfg["uM"])
    speed = final_policy_speed(policy_obj, release_pos, target_xy, u_max)
    throwing_system = ThrowingSystem(
        mass=0.0577,
        radius=0.0327,
        launch_angle_deg=35.0,
    )
    s0 = np.concatenate([release_pos, np.zeros(3), target_xy])
    _, _, clean_states = throwing_system.rollout(
        s0=s0,
        policy=lambda _state, _t: np.array([speed]),
        T=float(cfg.get("T", 0.60)),
        dt=float(cfg.get("Ts", 0.02)),
        noise=0.0,
    )
    landing = clean_states[-1, 0:3]
    error = float(np.linalg.norm(landing[0:2] - target_xy))
    return speed, landing, error


def main():
    args = build_parser().parse_args()
    log_file = os.path.join(args.log_path, "log.pkl")
    cfg_file = os.path.join(args.log_path, "config_log.pkl")

    with open(log_file, "rb") as f:
        log = pkl.load(f)
    with open(cfg_file, "rb") as f:
        cfg = pkl.load(f)

    policy_obj = load_policy(log, cfg)
    release_pos = np.array(cfg["release_pos"], dtype=float)
    u_max = float(cfg["uM"])

    print(f"robot={cfg.get('robot_name', 'unknown')}")
    print(f"log_path={args.log_path}")

    input_hist = log["input_samples_history"]
    trial_first_speeds = [float(np.asarray(arr).reshape(-1)[0]) for arr in input_hist]
    print("logged_first_speeds=" + ",".join(f"{v:.4f}" for v in trial_first_speeds))
    print(
        "logged_speed_summary="
        f"min={min(trial_first_speeds):.4f}, "
        f"max={max(trial_first_speeds):.4f}, "
        f"last={trial_first_speeds[-1]:.4f}"
    )

    landings = []
    for arr in log["noiseless_states_history"]:
        traj = np.asarray(arr)
        landings.append(traj[-1, 0:2])
    landing_norms = [float(np.linalg.norm(xy - release_pos[:2])) for xy in landings]
    print(
        "logged_landing_summary="
        f"min_range={min(landing_norms):.4f}, "
        f"max_range={max(landing_norms):.4f}, "
        f"last_range={landing_norms[-1]:.4f}"
    )

    x_values = np.linspace(float(cfg["lm"]), float(cfg["lM"]), args.grid_n)
    grid_targets = [np.array([x, 0.0], dtype=float) for x in x_values]
    grid_speeds = [final_policy_speed(policy_obj, release_pos, tgt, u_max) for tgt in grid_targets]
    print("final_policy_grid_speeds=" + ",".join(f"{v:.6f}" for v in grid_speeds))

    center_target = np.array([(float(cfg["lm"]) + float(cfg["lM"])) / 2.0, 0.0], dtype=float)
    speed, landing, error = run_ballistic_check(policy_obj, cfg, center_target)
    hit = error <= args.hit_radius
    print(f"ballistic_center_target=({center_target[0]:.3f},{center_target[1]:.3f})")
    print(f"ballistic_speed={speed:.6f}")
    print(f"ballistic_landing=({landing[0]:.3f},{landing[1]:.3f},{landing[2]:.3f})")
    print(f"ballistic_error_cm={error * 100.0:.2f}")
    print("ballistic_result=" + ("HIT" if hit else "MISS"))

    if max(trial_first_speeds) > 0.25 and max(grid_speeds) < 0.05:
        print("diagnosis=final_policy_collapsed_after_nonzero_logged_trials")
    elif max(trial_first_speeds) < 0.05:
        print("diagnosis=run_never_generated_meaningful_release_speeds")
    else:
        print("diagnosis=policy_outputs_nontrivial_speeds_check_other_components")


if __name__ == "__main__":
    main()
