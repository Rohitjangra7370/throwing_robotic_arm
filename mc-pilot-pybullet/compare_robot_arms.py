"""
Compare supported robot arms on throwing performance.

By default this matches the original mc-pilot-pybullet release logic: the ball
is launched with the commanded velocity, so results are comparable with the
existing KUKA baseline. A second "achieved" mode is also available to study
kinematic-feasibility limits when the arm can only deliver the clipped velocity.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

try:
    import pybullet as p
    import pybullet_data
except Exception as exc:  # pragma: no cover - this is the expected path on the current machine
    raise SystemExit(
        "PyBullet could not be imported on this machine. "
        "This comparison script is ready, but live runs need the DLL policy fixed first.\n"
        f"Import error: {exc}"
    )

from robot_arm.arm_controller import ArmController
from robot_arm.robot_profiles import get_robot_profile, iter_profiles
from simulation_class.model import _ball_accel


BALL_MASS = 0.0577
BALL_RADIUS = 0.0327
LAUNCH_ANGLE = np.deg2rad(35.0)
DT = 0.01


def _speed_to_velocity(speed: float, release_pos: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    dx = target_xy[0] - release_pos[0]
    dy = target_xy[1] - release_pos[1]
    azimuth = np.arctan2(dy, dx)
    return np.array(
        [
            speed * np.cos(LAUNCH_ANGLE) * np.cos(azimuth),
            speed * np.cos(LAUNCH_ANGLE) * np.sin(azimuth),
            speed * np.sin(LAUNCH_ANGLE),
        ]
    )


def _integrate_ballistics(release_pos: np.ndarray, release_vel: np.ndarray, horizon: float = 3.0) -> np.ndarray:
    pos = release_pos.copy().astype(float)
    vel = release_vel.copy().astype(float)
    prev_pos = pos.copy()
    prev_vel = vel.copy()
    for _ in range(int(horizon / DT)):
        prev_pos = pos.copy()
        prev_vel = vel.copy()
        acc = _ball_accel(pos, vel, mass=BALL_MASS, radius=BALL_RADIUS, wind=np.zeros(3))
        vel = vel + acc * DT
        pos = pos + vel * DT
        if pos[2] <= 0.0:
            frac = prev_pos[2] / max(prev_pos[2] - pos[2], 1e-8)
            land_pos = prev_pos + frac * (pos - prev_pos)
            land_pos[2] = 0.0
            return land_pos
    return pos


def _solve_command_speed(
    profile,
    release_pos: np.ndarray,
    target_xy: np.ndarray,
    solver_grid: int,
) -> tuple[float, float]:
    speeds = np.linspace(profile.speed_bounds[0], profile.speed_bounds[1], solver_grid)
    best_speed = float(speeds[0])
    best_err = float("inf")
    target = np.array([target_xy[0], target_xy[1], 0.0], dtype=float)
    for speed in speeds:
        v_cmd = _speed_to_velocity(float(speed), release_pos, target_xy)
        landing = _integrate_ballistics(release_pos, v_cmd)
        err = float(np.linalg.norm(landing[:2] - target[:2]))
        if err < best_err:
            best_err = err
            best_speed = float(speed)
    return best_speed, best_err


def main():
    parser = argparse.ArgumentParser("Compare robot-arm throwing performance")
    parser.add_argument("--out_dir", type=str, default="results_robot_arm_compare")
    parser.add_argument("--robots", nargs="*", default=None)
    parser.add_argument("--target_dists", nargs="*", type=float, default=[0.7, 0.9, 1.1])
    parser.add_argument("--solver_grid", type=int, default=121)
    parser.add_argument("--hit_threshold", type=float, default=0.10)
    parser.add_argument(
        "--release_mode",
        type=str,
        default="commanded",
        choices=["commanded", "achieved"],
        help="commanded matches the original baseline; achieved uses the clipped arm velocity",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    raw_rows = []
    summary_rows = []

    profiles = iter_profiles(args.robots)
    for profile in profiles:
        client = p.connect(p.DIRECT)
        urdf_path = pybullet_data.getDataPath() + "/" + profile.urdf_rel_path
        arm = ArmController(client, urdf_path, robot_name=profile.name)
        robot_errors = []
        robot_ratios = []
        robot_utils = []
        robot_hits = 0

        for target_dist in args.target_dists:
            release_pos = np.array(profile.default_release_pos, dtype=float)
            target_xy = np.array([target_dist, 0.0], dtype=float)
            speed, ideal_command_error = _solve_command_speed(
                profile, release_pos, target_xy, args.solver_grid
            )
            v_cmd = _speed_to_velocity(speed, release_pos, target_xy)
            coeffs, _, qd_release, v_achieved = arm.plan_throw(
                v_cmd, release_pos, *profile.timing
            )
            release_vel = v_cmd if args.release_mode == "commanded" else v_achieved
            landing = _integrate_ballistics(release_pos, release_vel)
            target = np.array([target_dist, 0.0, 0.0], dtype=float)
            landing_error = float(np.linalg.norm(landing[:2] - target[:2]))
            speed_ratio = float(np.linalg.norm(v_achieved) / max(np.linalg.norm(v_cmd), 1e-8))
            max_joint_util = float(
                np.max(np.abs(qd_release) / np.array(profile.qd_max, dtype=float))
            )
            clipped = coeffs["clip_scale"] > 1.0 + 1e-9
            hit = landing_error <= args.hit_threshold

            robot_errors.append(landing_error)
            robot_ratios.append(speed_ratio)
            robot_utils.append(max_joint_util)
            robot_hits += int(hit)
            raw_rows.append(
                {
                    "robot": profile.name,
                    "release_mode": args.release_mode,
                    "target_dist_m": target_dist,
                    "command_speed_mps": float(speed),
                    "ideal_command_error_m": ideal_command_error,
                    "command_norm_mps": float(np.linalg.norm(v_cmd)),
                    "achieved_speed_mps": float(np.linalg.norm(v_achieved)),
                    "used_release_speed_mps": float(np.linalg.norm(release_vel)),
                    "speed_ratio": speed_ratio,
                    "max_joint_util": max_joint_util,
                    "clipped": clipped,
                    "landing_x_m": float(landing[0]),
                    "landing_z_m": float(landing[2]),
                    "landing_error_m": landing_error,
                    "hit": hit,
                }
            )

        summary_rows.append(
            {
                "robot": profile.name,
                "release_mode": args.release_mode,
                "trials": len(robot_errors),
                "hits": robot_hits,
                "hit_rate_pct": float(100.0 * robot_hits / len(robot_errors)) if robot_errors else np.nan,
                "mean_landing_error_m": float(np.mean(robot_errors)),
                "std_landing_error_m": float(np.std(robot_errors)),
                "mean_speed_ratio": float(np.mean(robot_ratios)),
                "min_speed_ratio": float(np.min(robot_ratios)),
                "mean_joint_util": float(np.mean(robot_utils)),
                "max_joint_util": float(np.max(robot_utils)),
            }
        )
        p.disconnect(client)

    raw_csv = os.path.join(args.out_dir, "robot_compare_raw.csv")
    summary_csv = os.path.join(args.out_dir, "robot_compare_summary.csv")
    summary_md = os.path.join(args.out_dir, "robot_compare_summary.md")
    plot_path = os.path.join(args.out_dir, "robot_compare_metrics.png")

    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("| Robot | Mode | Hits | Hit Rate % | Mean Error m | Std Error m | Mean Speed Ratio | Min Speed Ratio | Mean Joint Util | Max Joint Util |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            f.write(
                f"| {row['robot']} | {row['release_mode']} | {row['hits']} / {row['trials']} | {row['hit_rate_pct']:.1f} | {row['mean_landing_error_m']:.4f} | "
                f"{row['std_landing_error_m']:.4f} | {row['mean_speed_ratio']:.4f} | "
                f"{row['min_speed_ratio']:.4f} | {row['mean_joint_util']:.4f} | {row['max_joint_util']:.4f} |\n"
            )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        robots = [row["robot"] for row in summary_rows]
        mean_errors = [row["mean_landing_error_m"] for row in summary_rows]
        speed_ratios = [row["mean_speed_ratio"] for row in summary_rows]
        hit_rates = [row["hit_rate_pct"] for row in summary_rows]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].bar(robots, hit_rates, color=["#3b82f6", "#ef4444", "#10b981"])
        axes[0].set_title("Hit Rate")
        axes[0].set_ylabel("percent")
        axes[0].grid(axis="y", alpha=0.25)

        axes[1].bar(robots, mean_errors, color=["#3b82f6", "#ef4444", "#10b981"])
        axes[1].set_title("Mean Landing Error")
        axes[1].set_ylabel("meters")
        axes[1].grid(axis="y", alpha=0.25)

        axes[2].bar(robots, speed_ratios, color=["#3b82f6", "#ef4444", "#10b981"])
        axes[2].set_title("Mean Achieved / Commanded Speed")
        axes[2].set_ylabel("ratio")
        axes[2].grid(axis="y", alpha=0.25)

        fig.tight_layout()
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {plot_path}")
    except Exception as exc:
        print(f"[WARN] Could not generate plot: {exc}")

    print(f"Wrote {raw_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_md}")
    print("\nSummary:")
    for row in summary_rows:
        print(
            f"  {row['robot']}: hit_rate={row['hit_rate_pct']:.1f}%, "
            f"mean_error={row['mean_landing_error_m']:.4f} m, "
            f"mean_speed_ratio={row['mean_speed_ratio']:.4f}, "
            f"max_joint_util={row['max_joint_util']:.4f}"
        )


if __name__ == "__main__":
    main()
