"""
Standalone arm throw test — no MC-PILOT, no GP.

Loads KUKA iiwa7 in PyBullet, plans a single throw toward a hardcoded target,
executes the cubic trajectory, releases the ball, and lets it fly.

Run modes:
  python standalone_throw.py          # p.GUI — visualise the throw
  python standalone_throw.py --direct # p.DIRECT — headless, prints landing pos only

Success criteria:
  - Arm visibly moves from neutral → windup → release → follow-through
  - Ball attaches to EE, moves with the arm, detaches at release
  - Ball lands somewhere plausible (within ~0.5m of expected for the command speed)
  - No NaN/inf in positions
"""

import argparse
import sys
import time

import numpy as np
import pybullet as p
import pybullet_data

sys.path.insert(0, ".")
from robot_arm.arm_controller import ArmController
from simulation_class.model import _ball_accel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--direct",      action="store_true", help="headless mode")
parser.add_argument("--speed",       type=float, default=3.5, help="release speed (m/s)")
parser.add_argument("--target_dist", type=float, default=1.5, help="target distance (m)")
parser.add_argument("--z_release",   type=float, default=1.0,
                    help="release height (m); arm base placed at z_release-0.5. "
                         "Use 1.0/1.5/2.0 to match configs B/C/D.")
parser.add_argument("--vel_mult",    type=float, default=1.5,
                    help="joint velocity limit multiplier (default 1.5 for uM=3.5 m/s)")
args = parser.parse_args()

BALL_MASS    = 0.0577
BALL_RADIUS  = 0.0327
LAUNCH_ANGLE = np.deg2rad(35.0)
DT           = 0.02
T_W          = 0.3
T_R          = 0.6
T_TOTAL      = max(1.5, args.z_release * 0.6 + 1.0)   # longer horizon for higher releases

# Release position straight forward at the requested height
RELEASE_POS  = np.array([0.0, 0.0, args.z_release])
ARM_BASE_POS = (0.0, 0.0, args.z_release - 0.5)   # pedestal

# Target: on the ground, straight ahead
target_dist = args.target_dist
TARGET = np.array([target_dist, 0.0, 0.0])

# ---------------------------------------------------------------------------
# Launch PyBullet
# ---------------------------------------------------------------------------
mode = p.DIRECT if args.direct else p.GUI
client = p.connect(mode)
p.setGravity(0, 0, -9.81, physicsClientId=client)
p.setTimeStep(DT, physicsClientId=client)
p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)

# Ground plane
p.loadURDF("plane.urdf", physicsClientId=client)

# Arm — mounted on pedestal so EE reaches RELEASE_POS
urdf_path = pybullet_data.getDataPath() + "/kuka_iiwa/model.urdf"
arm = ArmController(client, urdf_path, base_position=ARM_BASE_POS,
                    vel_limit_multiplier=args.vel_mult)
arm.reset()

# Ball — placed at EE position
ee_pos, _, _, _ = arm.ee_state()
ball_col = p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_RADIUS, physicsClientId=client)
ball_vis = p.createVisualShape(p.GEOM_SPHERE, radius=BALL_RADIUS,
                               rgbaColor=[1, 1, 0, 1], physicsClientId=client)
ball_id = p.createMultiBody(
    baseMass=BALL_MASS,
    baseCollisionShapeIndex=ball_col,
    baseVisualShapeIndex=ball_vis,
    basePosition=ee_pos.tolist(),
    physicsClientId=client,
)
p.changeDynamics(ball_id, -1, linearDamping=0.0, angularDamping=0.0,
                 physicsClientId=client)

# Target marker (red sphere)
tgt_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.05,
                               rgbaColor=[1, 0, 0, 0.7], physicsClientId=client)
p.createMultiBody(baseMass=0, baseVisualShapeIndex=tgt_vis,
                  basePosition=TARGET.tolist(), physicsClientId=client)

# ---------------------------------------------------------------------------
# Plan throw
# ---------------------------------------------------------------------------
speed = args.speed
phi = np.arctan2(TARGET[1] - RELEASE_POS[1], TARGET[0] - RELEASE_POS[0])
v_cmd = np.array([
    speed * np.cos(LAUNCH_ANGLE) * np.cos(phi),
    speed * np.cos(LAUNCH_ANGLE) * np.sin(phi),
    speed * np.sin(LAUNCH_ANGLE),
])
print(f"v_cmd = {v_cmd}, |v| = {np.linalg.norm(v_cmd):.3f} m/s")

arm.attach_ball(ball_id)
coeffs, q_release, qd_release, v_achieved = arm.plan_throw(
    v_cmd, RELEASE_POS, t_w=T_W, t_r=T_R, T=T_TOTAL
)
print(f"v_cmd = {np.linalg.norm(v_cmd):.3f} m/s, v_achieved = {np.linalg.norm(v_achieved):.3f} m/s")

# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------
released = False
ball_positions = []
n_steps = int(T_TOTAL / DT) + 50   # extra steps after follow-through

for step in range(n_steps):
    t = step * DT

    if not released:
        q_t, qd_t = arm.get_setpoint(coeffs, t)
        arm.step(q_t, qd_t)

        if t >= T_R:
            # Override ball velocity to the planned v_cmd (decouples arm tracking from ball physics)
            release_vel = arm.release_ball(ball_id, set_vel=v_cmd)
            print(f"Released at t={t:.3f}s, v_release={release_vel}, |v|={np.linalg.norm(release_vel):.3f}")
            released = True
    else:
        # Apply paper's Eq. 35 drag force manually
        ball_pos, _ = p.getBasePositionAndOrientation(ball_id, physicsClientId=client)
        ball_vel, _ = p.getBaseVelocity(ball_id, physicsClientId=client)
        pos = np.array(ball_pos)
        vel = np.array(ball_vel)

        a_total = _ball_accel(pos, vel, mass=BALL_MASS, radius=BALL_RADIUS,
                              wind=np.zeros(3))
        a_drag = a_total - np.array([0.0, 0.0, -9.81])
        F_drag = BALL_MASS * a_drag
        p.applyExternalForce(ball_id, -1, F_drag.tolist(), [0, 0, 0],
                             p.WORLD_FRAME, physicsClientId=client)

        ball_positions.append(pos.copy())

        # Draw trajectory trace
        if len(ball_positions) > 1 and not args.direct:
            p.addUserDebugLine(
                ball_positions[-2].tolist(), ball_positions[-1].tolist(),
                lineColorRGB=[0, 0.5, 1], lineWidth=2,
                physicsClientId=client,
            )

        # Stop when ball center reaches ground (z = ball radius = touching ground)
        if pos[2] <= BALL_RADIUS + 0.005:
            break

    p.stepSimulation(physicsClientId=client)
    if not args.direct:
        time.sleep(DT)

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
if ball_positions:
    landing = ball_positions[-1]
    error = np.linalg.norm(landing[:2] - TARGET[:2])
    print(f"\nLanding pos:  ({landing[0]:.3f}, {landing[1]:.3f}, {landing[2]:.3f})")
    print(f"Target:       ({TARGET[0]:.3f}, {TARGET[1]:.3f}, {TARGET[2]:.3f})")
    print(f"Landing error: {error:.3f} m")

    if not args.direct:
        # Pause so user can see the result
        p.addUserDebugText(
            f"Error: {error*100:.0f} cm",
            landing.tolist(),
            textColorRGB=[1, 1, 1],
            textSize=2,
            physicsClientId=client,
        )
        time.sleep(3.0)
else:
    print("Ball never landed (check release_pos and T_TOTAL).")

p.disconnect(client)
