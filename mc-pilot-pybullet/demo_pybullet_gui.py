"""
PyBullet GUI demo - replay a trained MC-PILOT policy visually.

Loads a trained policy from a results directory, spawns a supported robot arm
in PyBullet GUI mode, and executes N throws toward random targets.

Usage:
  python demo_pybullet_gui.py --log_path results_mc_pilot_pb_A/1
  python demo_pybullet_gui.py --log_path results_mc_pilot_pb_A/1 --robot franka_panda
  python demo_pybullet_gui.py --log_path results_mc_pilot_pb_A/1 --robot xarm6 --num_throws 10 --slow 3 --record demo.mp4
"""

import argparse
import os
import pickle as pkl
import time

import numpy as np
import pybullet as p
import pybullet_data
import torch

import sys

sys.path.insert(0, ".")
import policy_learning.Policy as Policy
from robot_arm.arm_controller import ArmController
from robot_arm.robot_profiles import available_robot_names, get_robot_profile


parser = argparse.ArgumentParser("PyBullet GUI demo for mc-pilot-pybullet")
parser.add_argument("--log_path", type=str, required=True, help="path to results/<seed>/")
parser.add_argument("--robot", type=str, default=None, choices=available_robot_names(),
                    help="robot arm profile to visualize; defaults to config value or kuka_iiwa")
parser.add_argument("--use_profile_defaults", action="store_true",
                    help="when overriding the robot, use that robot's safer release pose and speed cap")
parser.add_argument("--num_throws", type=int, default=5, help="number of demo throws")
parser.add_argument("--slow", type=float, default=1.0, help="playback slowdown factor")
parser.add_argument("--record", type=str, default=None, help="output .mp4 path")
parser.add_argument("--seed", type=int, default=42, help="seed for target sampling")
parser.add_argument("--speed_scale", type=float, default=1.0,
                    help="extra multiplier on the policy speed for GUI stability experiments")
args = parser.parse_args()

np.random.seed(args.seed)

log_file = os.path.join(args.log_path, "log.pkl")
cfg_file = os.path.join(args.log_path, "config_log.pkl")
if not os.path.exists(log_file):
    raise FileNotFoundError(f"No log.pkl found at {log_file}")

with open(log_file, "rb") as f:
    log = pkl.load(f)
with open(cfg_file, "rb") as f:
    cfg = pkl.load(f)

robot_name = args.robot or cfg.get("robot_name", "kuka_iiwa")
profile = get_robot_profile(robot_name)
cfg_robot_name = cfg.get("robot_name", "kuka_iiwa")
robot_overridden = robot_name != cfg_robot_name

print(f"Loaded log from {log_file}")
print(f"Robot profile: {profile.name}")
if robot_overridden:
    print(f"Config was trained/logged for robot={cfg_robot_name}; visualizing with robot={robot_name}")

lm = cfg.get("lm", 0.5)
lM = cfg.get("lM", 1.0)
gM = cfg.get("gM", np.pi / 6)
uM = cfg.get("uM", profile.speed_bounds[1])
Ts = cfg.get("Ts", 0.02)
T = cfg.get("T", 0.60)
release_pos_cfg = cfg.get("release_pos")
use_profile_defaults = args.use_profile_defaults or robot_overridden
if use_profile_defaults:
    RELEASE_POS = np.array(profile.default_release_pos, dtype=float)
    uM = min(float(uM), float(profile.speed_bounds[1]))
    print(
        f"Using profile defaults for {profile.name}: "
        f"release_pos={RELEASE_POS.tolist()}, uM={uM:.3f}"
    )
else:
    RELEASE_POS = np.array(
        release_pos_cfg if release_pos_cfg is not None else profile.default_release_pos,
        dtype=float,
    )
LAUNCH_ANGLE = np.deg2rad(35.0)


def sample_target():
    dist = np.random.uniform(lm, lM)
    angle = np.random.uniform(-gM, gM)
    return np.array([dist * np.cos(angle), dist * np.sin(angle)])


BALL_MASS = 0.0577
BALL_RADIUS = 0.0327


def _cd_sphere(v_norm, radius):
    rho = 1.225
    mu = 1.81e-5
    re = rho * v_norm * 2 * radius / mu if v_norm > 0 else 0.0
    return 0.2 if re > 2e5 else 0.47


def _ball_accel(pos, vel, wind=None):
    if wind is None:
        wind = np.zeros(3)
    area = np.pi * BALL_RADIUS**2
    v_rel = vel - wind
    v_norm = np.linalg.norm(v_rel)
    cd = _cd_sphere(v_norm, BALL_RADIUS)
    f_drag = -0.5 * 1.225 * cd * area * v_norm * v_rel
    f_grav = np.array([0.0, 0.0, -BALL_MASS * 9.81])
    return (f_drag + f_grav) / BALL_MASS


def speed_to_velocity(speed, release_pos, target_xy):
    dx = target_xy[0] - release_pos[0]
    dy = target_xy[1] - release_pos[1]
    phi = np.arctan2(dy, dx)
    return np.array(
        [
            speed * np.cos(LAUNCH_ANGLE) * np.cos(phi),
            speed * np.cos(LAUNCH_ANGLE) * np.sin(phi),
            speed * np.sin(LAUNCH_ANGLE),
        ]
    )


Nb = cfg.get("Nb", 250)
dtype = torch.float64
device = torch.device("cpu")
STATE_DIM = 8
TARGET_DIM = 2
ls_init = cfg.get("lengthscales_init", [0.08, 0.08])

if "parameters_trial_list" in log and log["parameters_trial_list"]:
    policy_state = log["parameters_trial_list"][-1]
    print(f"Loaded policy weights from trial {len(log['parameters_trial_list'])}")
else:
    print("WARNING: no policy weights in log - demo will use a zero-init policy")
    policy_state = None

if policy_state is not None:
    centers_init = policy_state["centers"].numpy()
    weight_init = policy_state["f_linear.weight"].numpy()
    ls_log = policy_state["log_lengthscales"].exp().numpy()[0]
else:
    centers_init = np.zeros((Nb, 2))
    weight_init = np.zeros((1, Nb))
    ls_log = np.array(ls_init)

policy_obj = Policy.Throwing_Policy(
    full_state_dim=STATE_DIM,
    target_dim=TARGET_DIM,
    num_basis=Nb,
    u_max=uM,
    lengthscales_init=ls_log,
    centers_init=centers_init,
    weight_init=weight_init,
    flg_drop=False,
    dtype=dtype,
    device=device,
)
if policy_state is not None:
    policy_obj.load_state_dict(policy_state)
policy_obj.eval()


def get_speed(target_xy):
    s0 = np.concatenate([RELEASE_POS, np.zeros(3), target_xy])
    with torch.no_grad():
        inp = torch.tensor(s0, dtype=dtype, device=device).unsqueeze(0)
        spd = policy_obj(inp, t=0, p_dropout=0.0)
    return float(spd.item())


T_W, T_R, T_ARM = profile.timing
DT = Ts

client = p.connect(p.GUI)
p.setGravity(0, 0, -9.81, physicsClientId=client)
p.setTimeStep(DT, physicsClientId=client)
p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)

p.resetDebugVisualizerCamera(
    cameraDistance=2.0,
    cameraYaw=45,
    cameraPitch=-30,
    cameraTargetPosition=[float(RELEASE_POS[0]), float(RELEASE_POS[1]), max(0.25, float(RELEASE_POS[2]) * 0.7)],
    physicsClientId=client,
)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=client)

p.loadURDF("plane.urdf", physicsClientId=client)

urdf_path = pybullet_data.getDataPath() + "/" + profile.urdf_rel_path
arm = ArmController(client, urdf_path, robot_name=profile.name)
if profile.name == "franka_panda":
    # Panda GUI playback is more stable when we replay the planned joint path
    # kinematically instead of relying on PyBullet motor dynamics.
    arm._control_mode = "kinematic"
    print("Using kinematic GUI playback for franka_panda stability.")

log_id = None
if args.record:
    log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, args.record, physicsClientId=client)
    print(f"Recording to {args.record}")

for throw_idx in range(args.num_throws):
    target = sample_target()
    print(f"\nThrow {throw_idx + 1}/{args.num_throws}: target=({target[0]:.3f}, {target[1]:.3f})")

    speed = get_speed(target)
    speed = speed * args.speed_scale
    speed = min(speed, uM)
    v_cmd = speed_to_velocity(speed, RELEASE_POS, target)
    print(f"  Policy speed: {speed:.3f} m/s, v_cmd: {np.round(v_cmd, 3)}")

    tgt_vis = p.createVisualShape(
        p.GEOM_SPHERE, radius=0.06, rgbaColor=[1, 0, 0, 0.8], physicsClientId=client
    )
    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=tgt_vis,
        basePosition=[target[0], target[1], 0.03],
        physicsClientId=client,
    )

    arm.reset()
    p.stepSimulation(physicsClientId=client)
    time.sleep(0.3 * args.slow)

    ee_pos, _, _, _ = arm.ee_state()
    bcol = p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_RADIUS, physicsClientId=client)
    bvis = p.createVisualShape(
        p.GEOM_SPHERE, radius=BALL_RADIUS, rgbaColor=[1, 1, 0, 1], physicsClientId=client
    )
    ball_id = p.createMultiBody(
        baseMass=BALL_MASS,
        baseCollisionShapeIndex=bcol,
        baseVisualShapeIndex=bvis,
        basePosition=ee_pos.tolist(),
        physicsClientId=client,
    )
    p.changeDynamics(ball_id, -1, linearDamping=0, angularDamping=0, physicsClientId=client)

    arm.attach_ball(ball_id)
    coeffs, _, _, _ = arm.plan_throw(v_cmd, RELEASE_POS, T_W, T_R, T_ARM)

    released = False
    ball_positions = []
    n_steps = int(T_ARM / DT) + int(T / DT) + 20

    for step in range(n_steps):
        t = step * DT
        if not released:
            q_t, qd_t = arm.get_setpoint(coeffs, t)
            arm.step(q_t, qd_t)
            if t >= T_R:
                ee_pos_now, _, _, _ = arm.ee_state()
                speed_norm = np.linalg.norm(v_cmd)
                release_dir = v_cmd / speed_norm if speed_norm > 1e-9 else np.zeros(3)
                use_safe_release = profile.use_safe_release
                safe_release_pos = ee_pos_now + release_dir * (1.25 * BALL_RADIUS)
                release_vel = arm.release_ball(
                    ball_id,
                    set_vel=v_cmd,
                    release_pos=safe_release_pos if use_safe_release else None,
                    keep_collision_disabled=use_safe_release,
                )
                print(f"  Released at t={t:.3f}s with |v|={np.linalg.norm(release_vel):.3f} m/s")
                released = True
        else:
            ball_pos_raw, _ = p.getBasePositionAndOrientation(ball_id, physicsClientId=client)
            ball_vel_raw, _ = p.getBaseVelocity(ball_id, physicsClientId=client)
            pos = np.array(ball_pos_raw)
            vel = np.array(ball_vel_raw)

            a_total = _ball_accel(pos, vel)
            f_drag = BALL_MASS * (a_total - np.array([0, 0, -9.81]))
            p.applyExternalForce(
                ball_id, -1, f_drag.tolist(), [0, 0, 0], p.WORLD_FRAME, physicsClientId=client
            )

            ball_positions.append(pos.copy())
            if len(ball_positions) > 1:
                p.addUserDebugLine(
                    ball_positions[-2].tolist(),
                    ball_positions[-1].tolist(),
                    lineColorRGB=[0, 0.4, 1],
                    lineWidth=2,
                    physicsClientId=client,
                )

            if pos[2] <= BALL_RADIUS + 0.005 and len(ball_positions) > 3:
                break

        p.stepSimulation(physicsClientId=client)
        time.sleep(DT * args.slow)

    if ball_positions:
        landing = ball_positions[-1]
        error = np.linalg.norm(landing[:2] - target)
        hit = error < 0.1
        print(
            f"  Landing: ({landing[0]:.3f}, {landing[1]:.3f}), "
            f"error={error * 100:.1f}cm {'HIT' if hit else 'MISS'}"
        )

        color = [0, 1, 0, 0.9] if hit else [1, 0.5, 0, 0.9]
        land_vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.04, rgbaColor=color, physicsClientId=client
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=land_vis,
            basePosition=landing.tolist(),
            physicsClientId=client,
        )
        p.addUserDebugText(
            f"Err: {error * 100:.1f}cm",
            (landing + np.array([0, 0, 0.15])).tolist(),
            textColorRGB=color[:3],
            textSize=1.5,
            physicsClientId=client,
        )

    time.sleep(1.5 * args.slow)

print("Demo complete. Close the PyBullet window to exit.")
try:
    while p.isConnected(client):
        p.stepSimulation(physicsClientId=client)
        time.sleep(1.0 / 240.0)
except KeyboardInterrupt:
    pass

if log_id is not None:
    p.stopStateLogging(log_id, physicsClientId=client)
p.disconnect(client)
