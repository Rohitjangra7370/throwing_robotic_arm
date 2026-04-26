"""
Real-time PyBullet throw environment (Gazebo-like workflow).

What this script does:
- Lets you choose an arm model (Panda or KUKA iiwa)
- Spawns the arm and a red target bin
- Accepts a joint trajectory (.npy or .json)
- Replays the trajectory in real-time
- Releases a ball at a chosen step and simulates throw physics
- Prints HIT/MISS based on landing error to target

This script does NOT retrain anything.
Dependencies: pybullet, numpy
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data


ARM_SPECS = {
    "panda": {
        "name": "Franka Panda",
        "urdf": "franka_panda/panda.urdf",
        "controlled_joints": [0, 1, 2, 3, 4, 5, 6],
        "ee_link": 11,
        "home": [0.0, -0.4, 0.0, -2.3, 0.0, 2.0, 0.8],
        "windup": [-0.2, -1.0, 0.2, -2.0, 0.0, 1.4, 0.4],
        "release": [0.35, -0.2, -0.25, -1.5, 0.0, 2.2, 1.1],
        "follow": [0.1, 0.3, -0.1, -1.1, 0.0, 2.4, 1.2],
    },
    "kuka": {
        "name": "KUKA iiwa",
        "urdf": "kuka_iiwa/model.urdf",
        "controlled_joints": [0, 1, 2, 3, 4, 5, 6],
        "ee_link": 6,
        "home": [0.0, 0.4, 0.0, -1.2, 0.0, 1.0, 0.0],
        "windup": [-0.4, 0.7, 0.2, -1.6, 0.2, 0.7, -0.2],
        "release": [0.5, 0.1, -0.3, -1.0, -0.1, 1.6, 0.5],
        "follow": [0.2, -0.1, -0.1, -0.7, 0.0, 1.2, 0.2],
    },
}


def choose_arm_interactive(default_arm="panda"):
    print("Available arms:")
    for key, spec in ARM_SPECS.items():
        print(f"  - {key}: {spec['name']}")

    choice = input(f"Choose arm [{default_arm}]: ").strip().lower()
    if not choice:
        choice = default_arm

    if choice not in ARM_SPECS:
        raise ValueError(f"Unknown arm '{choice}'. Choose one of: {list(ARM_SPECS.keys())}")
    return choice


def lerp_segment(q0, q1, steps, endpoint=False):
    alpha = np.linspace(0.0, 1.0, steps, endpoint=endpoint)
    return (1.0 - alpha)[:, None] * q0[None, :] + alpha[:, None] * q1[None, :]


def generate_demo_trajectory(arm_key):
    spec = ARM_SPECS[arm_key]
    home = np.array(spec["home"], dtype=float)
    windup = np.array(spec["windup"], dtype=float)
    release = np.array(spec["release"], dtype=float)
    follow = np.array(spec["follow"], dtype=float)

    seg1 = lerp_segment(home, windup, 120, endpoint=False)
    seg2 = lerp_segment(windup, release, 60, endpoint=False)
    seg3 = lerp_segment(release, follow, 120, endpoint=True)
    traj = np.vstack([seg1, seg2, seg3])

    release_step = len(seg1) + len(seg2) - 1
    dt = 1.0 / 240.0
    return traj, dt, release_step


def load_trajectory_file(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".npy":
        payload = np.load(path, allow_pickle=True)
        if isinstance(payload, np.ndarray) and payload.dtype == object and payload.shape == ():
            payload = payload.item()
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        raise ValueError("Trajectory must be .npy or .json")

    file_dt = None
    file_release = None

    if isinstance(payload, dict):
        if "joint_positions" not in payload:
            raise KeyError("Trajectory dict must contain key 'joint_positions'")
        traj = np.asarray(payload["joint_positions"], dtype=float)
        if "dt" in payload:
            file_dt = float(payload["dt"])
        if "release_step" in payload:
            file_release = int(payload["release_step"])
    else:
        traj = np.asarray(payload, dtype=float)

    if traj.ndim != 2:
        raise ValueError(f"Expected trajectory shape [T, DoF], got {traj.shape}")

    return traj, file_dt, file_release


def adapt_trajectory_dof(trajectory, dof, home):
    steps, cols = trajectory.shape
    if cols == dof:
        return trajectory

    if cols < dof:
        padded = np.repeat(home.reshape(1, -1), steps, axis=0)
        padded[:, :cols] = trajectory
        print(f"Trajectory has {cols} columns; padded to {dof} using home pose")
        return padded

    print(f"Trajectory has {cols} columns; truncating to first {dof}")
    return trajectory[:, :dof]


def set_joint_targets(robot_id, joint_indices, targets):
    p.setJointMotorControlArray(
        bodyUniqueId=robot_id,
        jointIndices=joint_indices,
        controlMode=p.POSITION_CONTROL,
        targetPositions=targets.tolist(),
        forces=[200.0] * len(joint_indices),
        positionGains=[0.25] * len(joint_indices),
        velocityGains=[1.0] * len(joint_indices),
    )


def reset_robot_pose(robot_id, joint_indices, q):
    for j, val in zip(joint_indices, q):
        p.resetJointState(robot_id, j, float(val))
    set_joint_targets(robot_id, joint_indices, q)


def get_ee_position(robot_id, ee_link):
    state = p.getLinkState(robot_id, ee_link, computeForwardKinematics=True)
    return np.array(state[4], dtype=float)


def aimed_release_velocity(release_pos, target_xyz, speed, throw_angle_deg):
    direction_xy = np.array([target_xyz[0] - release_pos[0], target_xyz[1] - release_pos[1]], dtype=float)
    norm_xy = np.linalg.norm(direction_xy)
    if norm_xy < 1e-9:
        direction_xy = np.array([1.0, 0.0], dtype=float)
    else:
        direction_xy = direction_xy / norm_xy

    angle = np.deg2rad(throw_angle_deg)
    v_xy = speed * np.cos(angle) * direction_xy
    v_z = speed * np.sin(angle)
    return np.array([v_xy[0], v_xy[1], v_z], dtype=float)


def spawn_target_bin(target_xyz, half_extents):
    bin_pos = [float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2] + half_extents[2])]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=[1.0, 0.15, 0.15, 0.95],
    )
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=bin_pos,
    )


def spawn_ball(position_xyz, radius, mass):
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    vis = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=[0.1, 0.35, 1.0, 1.0],
    )
    return p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=position_xyz.tolist(),
    )


def maybe_read_target_from_log(log_path: Path, throw_index: int):
    import pickle

    with log_path.open("rb") as f:
        log = pickle.load(f)

    hist = log.get("noiseless_states_history", None)
    if not hist:
        raise ValueError("No noiseless_states_history found in target log")

    idx = throw_index
    if idx < 0:
        idx = len(hist) + idx
    if idx < 0 or idx >= len(hist):
        raise IndexError(f"throw_index {throw_index} out of range for {len(hist)} throws")

    arr = np.asarray(hist[idx], dtype=float)
    target_xy = arr[-1, 6:8]
    return np.array([float(target_xy[0]), float(target_xy[1]), 0.0], dtype=float)


def run_throw(
    arm_key,
    trajectory,
    dt,
    release_step,
    target_xyz,
    hit_radius,
    settle_seconds,
    ball_speed,
    throw_angle_deg,
    connection_mode,
    keep_open,
):
    conn = p.GUI if connection_mode == "gui" else p.DIRECT
    cid = p.connect(conn)
    if cid < 0:
        raise RuntimeError("Could not connect to PyBullet")

    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        if connection_mode == "gui":
            p.resetDebugVisualizerCamera(
                cameraDistance=2.4,
                cameraYaw=35,
                cameraPitch=-35,
                cameraTargetPosition=[target_xyz[0] * 0.6, target_xyz[1], 0.35],
            )

        plane_id = p.loadURDF("plane.urdf")

        spec = ARM_SPECS[arm_key]
        robot_id = p.loadURDF(spec["urdf"], basePosition=[0.0, 0.0, 0.0], useFixedBase=True)
        joint_indices = spec["controlled_joints"]
        ee_link = spec["ee_link"]

        bin_half = [0.12, 0.12, 0.08]
        bin_id = spawn_target_bin(target_xyz=target_xyz, half_extents=bin_half)

        reset_robot_pose(robot_id, joint_indices, trajectory[0])
        for _ in range(10):
            p.stepSimulation()

        prev_ee = get_ee_position(robot_id, ee_link)
        ball_id = None
        released = False
        touched_bin = False
        landing_error = None

        release_step = int(np.clip(release_step, 0, len(trajectory) - 1))
        print(f"Using release_step={release_step} / {len(trajectory)-1}")

        for step_idx, q in enumerate(trajectory):
            set_joint_targets(robot_id, joint_indices, q)
            p.stepSimulation()
            ee = get_ee_position(robot_id, ee_link)

            if step_idx == release_step and not released:
                if ball_speed is None:
                    vel = (ee - prev_ee) / max(dt, 1e-6)
                    speed = np.linalg.norm(vel)
                    if speed < 0.5:
                        vel = aimed_release_velocity(ee, target_xyz, speed=3.5, throw_angle_deg=throw_angle_deg)
                else:
                    vel = aimed_release_velocity(ee, target_xyz, speed=ball_speed, throw_angle_deg=throw_angle_deg)

                ball_id = spawn_ball(position_xyz=ee, radius=0.0327, mass=0.0577)
                p.resetBaseVelocity(ball_id, linearVelocity=vel.tolist())
                released = True
                print(
                    "Released ball at step "
                    f"{step_idx}: pos=({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}), "
                    f"vel=({vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f})"
                )

            if released and ball_id is not None:
                if not touched_bin and p.getContactPoints(ball_id, bin_id):
                    touched_bin = True
                    print("Ball touched target bin")

                ground_contacts = p.getContactPoints(ball_id, plane_id)
                if ground_contacts and landing_error is None:
                    ball_pos = np.array(p.getBasePositionAndOrientation(ball_id)[0], dtype=float)
                    landing_error = float(np.linalg.norm(ball_pos[:2] - target_xyz[:2]))

            prev_ee = ee
            if connection_mode == "gui":
                time.sleep(dt)

        settle_steps = int(max(settle_seconds, 0.0) / max(dt, 1e-6))
        for _ in range(settle_steps):
            p.stepSimulation()
            if released and ball_id is not None:
                if not touched_bin and p.getContactPoints(ball_id, bin_id):
                    touched_bin = True
                ground_contacts = p.getContactPoints(ball_id, plane_id)
                if ground_contacts and landing_error is None:
                    ball_pos = np.array(p.getBasePositionAndOrientation(ball_id)[0], dtype=float)
                    landing_error = float(np.linalg.norm(ball_pos[:2] - target_xyz[:2]))
            if connection_mode == "gui":
                time.sleep(dt)

        if not released:
            print("No release occurred. Check release_step and trajectory length.")
            result = "MISS"
        else:
            if landing_error is None and ball_id is not None:
                ball_pos = np.array(p.getBasePositionAndOrientation(ball_id)[0], dtype=float)
                landing_error = float(np.linalg.norm(ball_pos[:2] - target_xyz[:2]))

            hit = touched_bin or (landing_error is not None and landing_error <= hit_radius)
            result = "HIT" if hit else "MISS"
            err_txt = "n/a" if landing_error is None else f"{landing_error:.3f} m"
            print(f"Throw result: {result} | landing error: {err_txt} | hit radius: {hit_radius:.3f} m")

        if connection_mode == "gui" and keep_open:
            print("Simulation complete. Close the PyBullet window or press Ctrl+C to exit.")
            try:
                while p.isConnected():
                    p.stepSimulation()
                    time.sleep(1.0 / 240.0)
            except KeyboardInterrupt:
                pass

    finally:
        if p.isConnected():
            p.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Real-time PyBullet throwing environment")
    parser.add_argument("--arm", choices=list(ARM_SPECS.keys()), default=None, help="Arm to spawn")
    parser.add_argument("--interactive", action="store_true", help="Prompt for arm/trajectory in terminal")
    parser.add_argument(
        "--trajectory",
        type=str,
        default=None,
        help="Path to joint trajectory (.npy or .json). If omitted, uses built-in demo trajectory.",
    )
    parser.add_argument(
        "--target",
        type=float,
        nargs=3,
        default=[1.10, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="Target coordinates in world frame",
    )
    parser.add_argument(
        "--target-log",
        type=str,
        default=None,
        help="Optional log.pkl path to read target from noiseless_states_history",
    )
    parser.add_argument(
        "--target-throw-index",
        type=int,
        default=-1,
        help="Throw index when reading target from --target-log (default: last throw)",
    )
    parser.add_argument("--release-step", type=int, default=None, help="Step index to release the ball")
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Simulation/control step in seconds. If omitted, uses trajectory-provided dt or default.",
    )
    parser.add_argument(
        "--ball-speed",
        type=float,
        default=None,
        help="If provided, ball speed at release (m/s) aimed at target; overrides EE-derived speed",
    )
    parser.add_argument(
        "--throw-angle-deg",
        type=float,
        default=35.0,
        help="Elevation angle used with --ball-speed",
    )
    parser.add_argument("--hit-radius", type=float, default=0.12, help="Hit threshold in meters")
    parser.add_argument("--settle-seconds", type=float, default=2.0, help="Post-trajectory settle time")
    parser.add_argument(
        "--connection",
        choices=["gui", "direct"],
        default="gui",
        help="PyBullet connection mode",
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="Keep GUI open after throw (ignored in direct mode)",
    )

    args = parser.parse_args()

    if args.interactive:
        arm_key = choose_arm_interactive()
        traj_prompt = input("Trajectory file (.npy/.json), or Enter for built-in demo: ").strip()
        traj_path = Path(traj_prompt) if traj_prompt else None
    else:
        arm_key = args.arm if args.arm is not None else "panda"
        traj_path = Path(args.trajectory) if args.trajectory else None

    if arm_key not in ARM_SPECS:
        raise ValueError(f"Unknown arm '{arm_key}'")

    spec = ARM_SPECS[arm_key]
    dof = len(spec["controlled_joints"])
    home = np.array(spec["home"], dtype=float)

    if traj_path is None:
        trajectory, file_dt, file_release = generate_demo_trajectory(arm_key)
        print("Using built-in demo trajectory")
    else:
        if not traj_path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {traj_path}")
        trajectory, file_dt, file_release = load_trajectory_file(traj_path)
        trajectory = adapt_trajectory_dof(trajectory, dof=dof, home=home)
        print(f"Loaded trajectory from {traj_path} with shape {trajectory.shape}")

    dt = args.dt if args.dt is not None else (file_dt if file_dt is not None else 1.0 / 240.0)

    if args.release_step is not None:
        release_step = args.release_step
    elif file_release is not None:
        release_step = file_release
    else:
        release_step = int(0.65 * (len(trajectory) - 1))

    if args.target_log:
        target_xyz = maybe_read_target_from_log(Path(args.target_log), args.target_throw_index)
        print(
            "Using target from log: "
            f"({target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f})"
        )
    else:
        target_xyz = np.array(args.target, dtype=float)

    print(f"Arm: {arm_key} ({spec['name']})")
    print(f"Trajectory steps: {len(trajectory)} | dof: {trajectory.shape[1]} | dt: {dt}")
    print(f"Target: ({target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f})")

    run_throw(
        arm_key=arm_key,
        trajectory=trajectory,
        dt=dt,
        release_step=release_step,
        target_xyz=target_xyz,
        hit_radius=args.hit_radius,
        settle_seconds=args.settle_seconds,
        ball_speed=args.ball_speed,
        throw_angle_deg=args.throw_angle_deg,
        connection_mode=args.connection,
        keep_open=args.keep_open,
    )


if __name__ == "__main__":
    main()
