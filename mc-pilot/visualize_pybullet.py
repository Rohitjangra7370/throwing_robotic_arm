"""
Visualize saved MC-PILOT throws in PyBullet.

- Loads trajectories from results_mc_pilot/1/log.pkl
- Uses noiseless_states_history entries as throw trajectories
- Extracts ball position (x, y, z) and target (Px, Py)
- Animates one throw per trial (last num_trials trajectories)
- Prints hit/miss per throw based on landing error threshold

This script does not retrain anything.
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pybullet as p


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def load_log_trajectories(log_path: Path):
    log = load_pickle(log_path)
    keys = sorted(log.keys())
    print("Log keys:", keys)

    if "noiseless_states_history" not in log:
        raise KeyError("Expected key 'noiseless_states_history' in log.pkl")

    raw_hist = log["noiseless_states_history"]
    print("noiseless_states_history length:", len(raw_hist))

    if len(raw_hist) == 0:
        raise ValueError("noiseless_states_history is empty")

    trajectories = []
    for idx, raw in enumerate(raw_hist):
        arr = np.asarray(raw, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 8:
            raise ValueError(
                f"Trajectory {idx} has invalid shape {arr.shape}; expected [T, >=8]"
            )
        trajectories.append(arr)

    print("first trajectory shape:", trajectories[0].shape)
    return log, trajectories


def select_trial_trajectories(trajectories, config_path: Path):
    num_trials = None
    ts = 0.02

    if config_path.exists():
        config = load_pickle(config_path)
        num_trials = int(config.get("num_trials", 0))
        ts = float(config.get("Ts", ts))
        print("config num_trials:", num_trials)
        print("config Ts:", ts)

    if num_trials is not None and num_trials > 0 and num_trials <= len(trajectories):
        selected = trajectories[-num_trials:]
        skipped = len(trajectories) - len(selected)
        if skipped > 0:
            print(f"Skipping {skipped} exploration throw(s); visualizing last {len(selected)} trial throw(s)")
    else:
        selected = trajectories
        print("Could not infer trial-only subset; visualizing all stored throws")

    return selected, ts


def build_scene(ball_radius: float, bin_half_extents: np.ndarray):
    if p.connect(p.GUI) < 0:
        raise RuntimeError("Failed to connect to PyBullet GUI")

    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(
        cameraDistance=3.0,
        cameraYaw=35.0,
        cameraPitch=-35.0,
        cameraTargetPosition=[1.1, 0.0, 0.25],
    )

    floor_half_extents = [4.0, 4.0, 0.01]
    floor_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=floor_half_extents)
    floor_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=floor_half_extents,
        rgbaColor=[0.92, 0.92, 0.92, 1.0],
    )
    p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=floor_collision,
        baseVisualShapeIndex=floor_visual,
        basePosition=[0.0, 0.0, -0.01],
    )

    ball_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
    ball_visual = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=ball_radius,
        rgbaColor=[0.10, 0.35, 1.0, 1.0],
    )
    ball_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=ball_collision,
        baseVisualShapeIndex=ball_visual,
        basePosition=[0.0, 0.0, ball_radius],
    )

    bin_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=bin_half_extents.tolist(),
    )
    bin_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=bin_half_extents.tolist(),
        rgbaColor=[1.0, 0.15, 0.15, 0.95],
    )
    bin_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=bin_collision,
        baseVisualShapeIndex=bin_visual,
        basePosition=[0.0, 0.0, float(bin_half_extents[2])],
    )

    return ball_id, bin_id


def animate_throws(trajectories, playback_dt: float, pause_between_throws: float, hit_radius: float):
    ball_radius = 0.0327
    bin_half_extents = np.array([0.10, 0.10, 0.06], dtype=float)

    ball_id, bin_id = build_scene(ball_radius=ball_radius, bin_half_extents=bin_half_extents)

    try:
        for idx, traj in enumerate(trajectories, start=1):
            xyz = traj[:, 0:3]
            target_xy = traj[0, 6:8]
            landing_xy = xyz[-1, 0:2]

            error = float(np.linalg.norm(landing_xy - target_xy))
            hit = error <= hit_radius

            bin_pos = [float(target_xy[0]), float(target_xy[1]), float(bin_half_extents[2])]
            p.resetBasePositionAndOrientation(bin_id, bin_pos, [0.0, 0.0, 0.0, 1.0])

            status = "HIT" if hit else "MISS"
            print(
                f"Throw {idx}/{len(trajectories)} | "
                f"target=({target_xy[0]:.3f}, {target_xy[1]:.3f}) | "
                f"landing=({landing_xy[0]:.3f}, {landing_xy[1]:.3f}) | "
                f"error={error:.3f} m | {status}"
            )

            for point in xyz:
                z = max(float(point[2]), ball_radius)
                p.resetBasePositionAndOrientation(
                    ball_id,
                    [float(point[0]), float(point[1]), z],
                    [0.0, 0.0, 0.0, 1.0],
                )
                p.stepSimulation()
                time.sleep(playback_dt)

            time.sleep(pause_between_throws)

        print("Done. Closing PyBullet in 2 seconds.")
        time.sleep(2.0)
    finally:
        if p.isConnected():
            p.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Visualize saved MC-PILOT trajectories in PyBullet")
    parser.add_argument(
        "--log-path",
        type=str,
        default="results_mc_pilot/1/log.pkl",
        help="Path to saved log.pkl",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="results_mc_pilot/1/config_log.pkl",
        help="Path to saved config_log.pkl (used to infer number of trials and Ts)",
    )
    parser.add_argument(
        "--playback-dt",
        type=float,
        default=None,
        help="Playback timestep in seconds; default uses Ts from config_log.pkl if available",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=1.0,
        help="Pause duration in seconds between throws",
    )
    parser.add_argument(
        "--hit-radius",
        type=float,
        default=0.10,
        help="Hit threshold radius in meters",
    )
    args = parser.parse_args()

    log_path = Path(args.log_path)
    config_path = Path(args.config_path)

    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    _, trajectories = load_log_trajectories(log_path)
    trial_trajectories, ts = select_trial_trajectories(trajectories, config_path)

    playback_dt = args.playback_dt if args.playback_dt is not None else ts
    print("Using playback dt:", playback_dt)

    animate_throws(
        trajectories=trial_trajectories,
        playback_dt=playback_dt,
        pause_between_throws=args.pause,
        hit_radius=args.hit_radius,
    )


if __name__ == "__main__":
    main()
