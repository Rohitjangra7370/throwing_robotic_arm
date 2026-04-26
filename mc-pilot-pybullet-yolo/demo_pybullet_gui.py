"""
PyBullet GUI demo — replay a trained MC-PILOT policy visually.

Loads a trained policy from a results directory, spawns the KUKA iiwa7 arm
in PyBullet GUI mode, and executes N throws. In vision mode, each throw:
  - Samples a hidden ground-truth target
  - Places a visible target bin at that location
  - Uses a front-of-base depth camera + OpenCV to estimate the bin coordinates
  - Feeds the detected target into the existing policy / throw planner
  - Renders the arm motion, ball trajectory, target bin, and landing result

Usage:
  python demo_pybullet_gui.py --log_path results_mc_pilot_pb_A/1
  python demo_pybullet_gui.py --log_path results_mc_pilot_pb_A_noisy/sigma_0p15_aware/1 \\
                               --num_throws 10 --slow 3 --record demo.mp4

Controls:
  The PyBullet GUI window stays open between throws.  Close it (or Ctrl-C) to quit.
"""

import argparse
import os
import pickle as pkl
import time

import numpy as np
import pybullet as p
import pybullet_data
import torch

import cv2

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser("PyBullet GUI demo for mc-pilot-pybullet")
parser.add_argument("--log_path",   type=str,   required=True, help="path to results/<seed>/")
parser.add_argument("--num_throws", type=int,   default=5,     help="number of demo throws")
parser.add_argument("--slow",       type=float, default=1.0,   help="playback slowdown factor")
parser.add_argument("--record",     type=str,   default=None,  help="output .mp4 path")
parser.add_argument("--seed",       type=int,   default=42,    help="seed for target sampling")
parser.add_argument(
    "--target-source",
    type=str,
    default="vision",
    choices=["vision", "random"],
    help="use wrist-camera bin detection or the sampled target directly",
)
parser.add_argument(
    "--save-detections",
    type=str,
    default=None,
    help="optional directory for saving front-base camera detection debug images",
)
parser.add_argument(
    "--show-detections",
    type=int,
    default=1,
    help="1=show OpenCV detection window with bounding box, 0=disable it",
)
args = parser.parse_args()

np.random.seed(args.seed)

if args.save_detections is not None:
    os.makedirs(args.save_detections, exist_ok=True)

# ---------------------------------------------------------------------------
# Load trained policy
# ---------------------------------------------------------------------------
log_file = os.path.join(args.log_path, "log.pkl")
cfg_file = os.path.join(args.log_path, "config_log.pkl")

if not os.path.exists(log_file):
    raise FileNotFoundError(f"No log.pkl found at {log_file}")

with open(log_file, "rb") as f:
    log = pkl.load(f)
with open(cfg_file, "rb") as f:
    cfg = pkl.load(f)

# Extract final trained policy (last control policy checkpoint)
# log contains the mc_pilot_obj serialised state
# We'll reload the config params and reconstruct the policy from weights
print(f"Loaded log from {log_file}")
print(f"Config: {cfg}")

# Physical params
lm     = cfg.get("lm",   0.5)
lM     = cfg.get("lM",   1.0)
gM     = cfg.get("gM",   np.pi/6)
uM     = cfg.get("uM",   2.5)
Ts     = cfg.get("Ts",   0.02)
T      = cfg.get("T",    0.60)
RELEASE_POS = np.array(cfg.get("release_pos", [0.0, 0.0, 0.5]))
LAUNCH_ANGLE = np.deg2rad(35.0)

# Target sampler
def sample_target():
    dist  = np.random.uniform(lm, lM)
    angle = np.random.uniform(-gM, gM)
    return np.array([dist * np.cos(angle), dist * np.sin(angle)])

# ---------------------------------------------------------------------------
# Physics constants
# ---------------------------------------------------------------------------
BALL_MASS   = 0.0577
BALL_RADIUS = 0.0327

def _cd_sphere(v_norm, radius):
    rho = 1.225; mu = 1.81e-5
    Re = rho * v_norm * 2 * radius / mu if v_norm > 0 else 0.0
    return 0.2 if Re > 2e5 else 0.47

def _ball_accel(pos, vel, wind=None):
    if wind is None: wind = np.zeros(3)
    A = np.pi * BALL_RADIUS**2
    v_rel = vel - wind
    v_norm = np.linalg.norm(v_rel)
    cd = _cd_sphere(v_norm, BALL_RADIUS)
    F_drag = -0.5 * 1.225 * cd * A * v_norm * v_rel
    F_grav = np.array([0.0, 0.0, -BALL_MASS * 9.81])
    return (F_drag + F_grav) / BALL_MASS

def speed_to_velocity(speed, release_pos, target_xy):
    dx = target_xy[0] - release_pos[0]
    dy = target_xy[1] - release_pos[1]
    phi = np.arctan2(dy, dx)
    return np.array([
        speed * np.cos(LAUNCH_ANGLE) * np.cos(phi),
        speed * np.cos(LAUNCH_ANGLE) * np.sin(phi),
        speed * np.sin(LAUNCH_ANGLE),
    ])

# ---------------------------------------------------------------------------
# Reconstruct policy from log
# ---------------------------------------------------------------------------
import sys; sys.path.insert(0, ".")
import policy_learning.Policy as Policy

Nb    = cfg.get("Nb", 250)
dtype = torch.float64
device = torch.device("cpu")

# Rebuild policy and load saved weights
STATE_DIM  = 8
TARGET_DIM = 2
ls_init = cfg.get("lengthscales_init", [0.08, 0.08])

# Final policy weights are saved in parameters_trial_list[-1] (OrderedDict)
if "parameters_trial_list" in log and log["parameters_trial_list"]:
    policy_state = log["parameters_trial_list"][-1]
    print(f"Loaded policy weights from trial {len(log['parameters_trial_list'])}")
else:
    print("WARNING: no policy weights in log — demo will use random policy")
    policy_state = None

# Build policy object using saved centers/weights as initial values
if policy_state is not None:
    centers_init = policy_state["centers"].numpy()
    weight_init  = policy_state["f_linear.weight"].numpy()
    ls_log       = policy_state["log_lengthscales"].exp().numpy()[0]
else:
    centers_init = np.zeros((Nb, 2))
    weight_init  = np.zeros((1, Nb))
    ls_log       = np.array(ls_init)

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

# ---------------------------------------------------------------------------
# PyBullet GUI setup
# ---------------------------------------------------------------------------
from robot_arm.arm_controller import ArmController
from robot_arm.depth_camera import ArmMountedDepthCamera, BIN_HEIGHT, spawn_target_bin

T_W = 0.3; T_R = 0.6; T_ARM = 1.2
DT  = Ts

client = p.connect(p.GUI)
p.setGravity(0, 0, -9.81, physicsClientId=client)
p.setTimeStep(DT, physicsClientId=client)
p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)

# Camera
p.resetDebugVisualizerCamera(
    cameraDistance=2.0, cameraYaw=45, cameraPitch=-30,
    cameraTargetPosition=[0.5, 0, 0.3], physicsClientId=client,
)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=client)

# Ground
p.loadURDF("plane.urdf", physicsClientId=client)

# Arm
urdf_path = pybullet_data.getDataPath() + "/kuka_iiwa/model.urdf"
arm = ArmController(client, urdf_path)
depth_camera = ArmMountedDepthCamera(client, arm)
depth_camera.draw_debug_pose()
print(
    "Front-base camera configured to look at workspace:",
    {
        "offset": depth_camera.local_offset.tolist(),
        "focus": depth_camera.workspace_focus.tolist(),
        "fov_deg": depth_camera.fov_deg,
    },
)

show_detection_window = bool(args.show_detections)

# Video recording
log_id = None
if args.record:
    log_id = p.startStateLogging(
        p.STATE_LOGGING_VIDEO_MP4, args.record, physicsClientId=client
    )
    print(f"Recording to {args.record}")

# ---------------------------------------------------------------------------
# Demo loop
# ---------------------------------------------------------------------------
landing_markers = []
trace_ids = []

for throw_idx in range(args.num_throws):
    target_true = sample_target()
    target_bin_id = spawn_target_bin(client, target_true)

    # Reset arm before front-base-camera perception.
    arm.reset()
    p.stepSimulation(physicsClientId=client)
    depth_camera.draw_debug_pose()
    time.sleep(0.3 * args.slow)

    target = target_true.copy()
    detection_marker_id = None
    if args.target_source == "vision":
        observation = depth_camera.capture()
        projected_true_bin = depth_camera.project_world_to_pixel(
            [target_true[0], target_true[1], BIN_HEIGHT / 2.0],
            observation,
        )
        try:
            detection = depth_camera.detect_bin(
                observation,
                debug=True,
                projected_target_px=projected_true_bin["pixel"],
                projected_target_visible=projected_true_bin["visible"],
            )
            target = detection["target_xy"]
            surface_target = detection["surface_target_xy"]
            projected_policy_target = depth_camera.project_world_to_pixel(
                [target[0], target[1], 0.0],
                observation,
            )
            detection_error = np.linalg.norm(target - target_true)
            surface_error = np.linalg.norm(surface_target - target_true)

            if detection.get("debug_bgr") is not None and projected_policy_target["pixel"] is not None:
                target_px = tuple(np.round(projected_policy_target["pixel"]).astype(int).tolist())
                cv2.circle(detection["debug_bgr"], target_px, 6, (255, 0, 255), -1)
                cv2.putText(
                    detection["debug_bgr"],
                    "policy target",
                    (target_px[0] + 8, max(20, target_px[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            print(
                f"\nThrow {throw_idx+1}/{args.num_throws}: "
                f"true_target=({target_true[0]:.3f}, {target_true[1]:.3f}), "
                f"surface=({surface_target[0]:.3f}, {surface_target[1]:.3f}), "
                f"detected=({target[0]:.3f}, {target[1]:.3f}), "
                f"surface_err={surface_error*100:.1f} cm, "
                f"vision_err={detection_error*100:.1f} cm, "
                f"in_view={projected_true_bin['visible']}"
            )

            det_vis = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.025,
                rgbaColor=[1.0, 0.0, 1.0, 0.95],
                physicsClientId=client,
            )
            detection_marker_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=det_vis,
                basePosition=[target[0], target[1], 0.015],
                physicsClientId=client,
            )

            if args.save_detections is not None:
                debug_path = os.path.join(
                    args.save_detections,
                    f"throw_{throw_idx + 1:03d}.png",
                )
                cv2.imwrite(debug_path, detection["debug_bgr"])
            if show_detection_window:
                try:
                    cv2.imshow("Front Base Camera Detection", detection["debug_bgr"])
                    cv2.waitKey(1)
                except cv2.error:
                    show_detection_window = False
        except RuntimeError as err:
            raw_debug = depth_camera.make_debug_image(
                observation=observation,
                projected_target_px=projected_true_bin["pixel"],
                projected_target_visible=projected_true_bin["visible"],
                status_text=str(err),
            )
            print(
                f"\nThrow {throw_idx+1}/{args.num_throws}: "
                f"vision detection failed ({err}); using sampled target "
                f"({target_true[0]:.3f}, {target_true[1]:.3f}), "
                f"in_view={projected_true_bin['visible']}"
            )
            if args.save_detections is not None:
                debug_path = os.path.join(
                    args.save_detections,
                    f"throw_{throw_idx + 1:03d}_failed.png",
                )
                cv2.imwrite(debug_path, raw_debug)
            if show_detection_window:
                try:
                    cv2.imshow("Front Base Camera Detection", raw_debug)
                    cv2.waitKey(1)
                except cv2.error:
                    show_detection_window = False
    else:
        print(
            f"\nThrow {throw_idx+1}/{args.num_throws}: "
            f"target=({target[0]:.3f}, {target[1]:.3f})"
        )

    # Query policy with the detected target.
    speed  = get_speed(target)
    v_cmd  = speed_to_velocity(speed, RELEASE_POS, target)
    print(f"  Policy speed: {speed:.3f} m/s, v_cmd: {np.round(v_cmd, 3)}")

    # Ball
    ee_pos, _, _, _ = arm.ee_state()
    bcol = p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_RADIUS, physicsClientId=client)
    bvis = p.createVisualShape(
        p.GEOM_SPHERE, radius=BALL_RADIUS,
        rgbaColor=[1, 1, 0, 1], physicsClientId=client,
    )
    ball_id = p.createMultiBody(
        baseMass=BALL_MASS, baseCollisionShapeIndex=bcol,
        baseVisualShapeIndex=bvis, basePosition=ee_pos.tolist(),
        physicsClientId=client,
    )
    p.changeDynamics(ball_id, -1, linearDamping=0, angularDamping=0,
                     physicsClientId=client)

    # Plan and execute throw
    arm.attach_ball(ball_id)
    T_total = T_ARM
    coeffs, _, _, v_ach = arm.plan_throw(v_cmd, RELEASE_POS, T_W, T_R, T_total)

    released = False
    ball_positions = []
    n_steps = int(T_total / DT) + int(T / DT) + 20

    for step in range(n_steps):
        t = step * DT
        if not released:
            q_t, qd_t = arm.get_setpoint(coeffs, t)
            arm.step(q_t, qd_t)
            if t >= T_R:
                arm.release_ball(ball_id, set_vel=v_cmd)
                released = True
        else:
            ball_pos_raw, _ = p.getBasePositionAndOrientation(ball_id, physicsClientId=client)
            ball_vel_raw, _ = p.getBaseVelocity(ball_id, physicsClientId=client)
            pos = np.array(ball_pos_raw)
            vel = np.array(ball_vel_raw)

            # Eq. 35 drag
            a_total = _ball_accel(pos, vel)
            F_drag  = BALL_MASS * (a_total - np.array([0, 0, -9.81]))
            p.applyExternalForce(ball_id, -1, F_drag.tolist(), [0, 0, 0],
                                 p.WORLD_FRAME, physicsClientId=client)

            ball_positions.append(pos.copy())

            # Trace
            if len(ball_positions) > 1:
                tid = p.addUserDebugLine(
                    ball_positions[-2].tolist(), ball_positions[-1].tolist(),
                    lineColorRGB=[0, 0.4, 1], lineWidth=2, physicsClientId=client,
                )
                trace_ids.append(tid)

            if pos[2] <= BALL_RADIUS + 0.005 and len(ball_positions) > 3:
                break

        p.stepSimulation(physicsClientId=client)
        time.sleep(DT * args.slow)

    # Landing result
    if ball_positions:
        landing = ball_positions[-1]
        error   = np.linalg.norm(landing[:2] - target)
        hit     = error < 0.1
        print(f"  Landing: ({landing[0]:.3f}, {landing[1]:.3f}), error={error*100:.0f}mm {'HIT' if hit else 'MISS'}")

        # Landing marker (green=hit, orange=miss)
        colour = [0, 1, 0, 0.9] if hit else [1, 0.5, 0, 0.9]
        land_vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.04,
            rgbaColor=colour, physicsClientId=client,
        )
        p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=land_vis,
            basePosition=landing.tolist(), physicsClientId=client,
        )
        p.addUserDebugText(
            f"Err: {error*100:.0f}cm",
            (landing + np.array([0, 0, 0.15])).tolist(),
            textColorRGB=colour[:3], textSize=1.5, physicsClientId=client,
        )

    # Pause between throws
    time.sleep(1.5 * args.slow)

    if detection_marker_id is not None:
        p.removeBody(detection_marker_id, physicsClientId=client)
    p.removeBody(target_bin_id, physicsClientId=client)

# ---------------------------------------------------------------------------
# Hold GUI open
# ---------------------------------------------------------------------------
print(f"\nAll {args.num_throws} throws complete. GUI stays open — close window to exit.")
if log_id is not None:
    p.stopStateLogging(log_id, physicsClientId=client)

try:
    while True:
        p.stepSimulation(physicsClientId=client)
        time.sleep(0.05)
except (KeyboardInterrupt, p.error):
    pass

if show_detection_window:
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

p.disconnect(client)
