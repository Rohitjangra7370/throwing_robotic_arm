"""
Standalone arm throw test - no MC-PILOT, no GP.

Loads one of the supported robot arms in PyBullet, plans a single throw toward
a target, executes the cubic trajectory, releases the ball, and lets it fly.

Run modes:
  python standalone_throw.py --robot kuka_iiwa
  python standalone_throw.py --robot franka_panda --direct
  python standalone_throw.py --robot xarm6 --direct --speed 1.6 --target_dist 0.8
"""

import argparse
import sys
import time

import numpy as np
import pybullet as p
import pybullet_data

sys.path.insert(0, ".")
from robot_arm.arm_controller import ArmController
from robot_arm.robot_profiles import available_robot_names, get_robot_profile
from simulation_class.model import _ball_accel


parser = argparse.ArgumentParser()
parser.add_argument("--direct", action="store_true", help="headless mode")
parser.add_argument("--speed", type=float, default=2.0, help="release speed (m/s)")
parser.add_argument("--target_dist", type=float, default=0.9, help="target distance (m)")
parser.add_argument(
    "--robot",
    type=str,
    default="kuka_iiwa",
    choices=available_robot_names(),
    help="robot arm profile to use",
)
args = parser.parse_args()
profile = get_robot_profile(args.robot)

BALL_MASS = 0.0577
BALL_RADIUS = 0.0327
LAUNCH_ANGLE = np.deg2rad(35.0)
DT = 0.02
T_W, T_R, T_TOTAL = profile.timing
RELEASE_POS = np.array(profile.default_release_pos, dtype=float)
TARGET = np.array([args.target_dist, 0.0, 0.0])

mode = p.DIRECT if args.direct else p.GUI
client = p.connect(mode)
p.setGravity(0, 0, -9.81, physicsClientId=client)
p.setTimeStep(DT, physicsClientId=client)
p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
p.loadURDF("plane.urdf", physicsClientId=client)

urdf_path = pybullet_data.getDataPath() + "/" + profile.urdf_rel_path
arm = ArmController(client, urdf_path, robot_name=profile.name)
arm.reset()

ee_pos, _, _, _ = arm.ee_state()
ball_col = p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_RADIUS, physicsClientId=client)
ball_vis = p.createVisualShape(
    p.GEOM_SPHERE, radius=BALL_RADIUS, rgbaColor=[1, 1, 0, 1], physicsClientId=client
)
ball_id = p.createMultiBody(
    baseMass=BALL_MASS,
    baseCollisionShapeIndex=ball_col,
    baseVisualShapeIndex=ball_vis,
    basePosition=ee_pos.tolist(),
    physicsClientId=client,
)
p.changeDynamics(ball_id, -1, linearDamping=0.0, angularDamping=0.0, physicsClientId=client)

tgt_vis = p.createVisualShape(
    p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 0.7], physicsClientId=client
)
p.createMultiBody(baseMass=0, baseVisualShapeIndex=tgt_vis, basePosition=TARGET.tolist(), physicsClientId=client)

speed = args.speed
phi = np.arctan2(TARGET[1] - RELEASE_POS[1], TARGET[0] - RELEASE_POS[0])
v_cmd = np.array(
    [
        speed * np.cos(LAUNCH_ANGLE) * np.cos(phi),
        speed * np.cos(LAUNCH_ANGLE) * np.sin(phi),
        speed * np.sin(LAUNCH_ANGLE),
    ]
)
print(f"robot={profile.name} v_cmd={v_cmd} |v|={np.linalg.norm(v_cmd):.3f} m/s")

arm.attach_ball(ball_id)
coeffs, _, qd_release, v_achieved = arm.plan_throw(v_cmd, RELEASE_POS, t_w=T_W, t_r=T_R, T=T_TOTAL)
print(
    f"planned speed={np.linalg.norm(v_cmd):.3f} m/s, "
    f"achieved speed={np.linalg.norm(v_achieved):.3f} m/s, "
    f"max joint util={np.max(np.abs(qd_release) / np.array(profile.qd_max)):.3f}"
)

released = False
ball_positions = []
n_steps = int(T_TOTAL / DT) + 50

for step in range(n_steps):
    t = step * DT
    if not released:
        q_t, qd_t = arm.get_setpoint(coeffs, t)
        arm.step(q_t, qd_t)
        if t >= T_R:
            release_vel = arm.release_ball(ball_id, set_vel=v_cmd)
            print(f"Released at t={t:.3f}s with |v|={np.linalg.norm(release_vel):.3f} m/s")
            released = True
    else:
        ball_pos, _ = p.getBasePositionAndOrientation(ball_id, physicsClientId=client)
        ball_vel, _ = p.getBaseVelocity(ball_id, physicsClientId=client)
        pos = np.array(ball_pos)
        vel = np.array(ball_vel)
        a_total = _ball_accel(pos, vel, mass=BALL_MASS, radius=BALL_RADIUS, wind=np.zeros(3))
        a_drag = a_total - np.array([0.0, 0.0, -9.81])
        f_drag = BALL_MASS * a_drag
        p.applyExternalForce(
            ball_id, -1, f_drag.tolist(), [0, 0, 0], p.WORLD_FRAME, physicsClientId=client
        )
        ball_positions.append(pos.copy())

        if len(ball_positions) > 1 and not args.direct:
            p.addUserDebugLine(
                ball_positions[-2].tolist(),
                ball_positions[-1].tolist(),
                lineColorRGB=[0, 0.5, 1],
                lineWidth=2,
                physicsClientId=client,
            )

        if pos[2] <= BALL_RADIUS + 0.005:
            break

    p.stepSimulation(physicsClientId=client)
    if not args.direct:
        time.sleep(DT)

if ball_positions:
    landing = ball_positions[-1]
    error = np.linalg.norm(landing[:2] - TARGET[:2])
    print(f"Landing pos: ({landing[0]:.3f}, {landing[1]:.3f}, {landing[2]:.3f})")
    print(f"Target:      ({TARGET[0]:.3f}, {TARGET[1]:.3f}, {TARGET[2]:.3f})")
    print(f"Landing error: {error:.3f} m")
else:
    print("Ball never landed; check release_pos, target, or timing.")

p.disconnect(client)
