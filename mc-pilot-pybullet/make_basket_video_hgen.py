"""
Height-generalized throw video: ONE KUKA policy (target = Px, Py, h) throwing to a
WIDE spread of targets at DIFFERENT heights, each with a raised basket at its true
3-D location. This is the honest "is it real?" demo -- widely separated targets and
varied heights, all from a single 9-D-state policy (results_mc_pilot_pb_A_hgen,
the email-2 "100/100 fresh throws" checkpoint).

Method identical to make_basket_video.py (DIRECT + TinyRenderer + frame_hook), with:
  - 9-D state [release, 0,0,0, Px, Py, h]; a fresh system per throw at target_height=h.
  - a visual-only raised bin: base plate + 4 translucent walls at z=h, on a thin support
    pole down to the floor (so it reads as a stand at that height).
"""

import argparse
import os
import pickle as pkl

import imageio.v2 as imageio
import numpy as np
import pybullet as p
import torch

import policy_learning.Policy as Policy
from robot_arm.robot_profiles import get_robot_profile
from simulation_class.model_pybullet import PyBulletThrowingSystem

_VIDS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "status_update", "vids")

WIDTH, HEIGHT = 1024, 720


def load_policy_9d(log_path):
    with open(os.path.join(log_path, "log.pkl"), "rb") as f:
        log = pkl.load(f)
    with open(os.path.join(log_path, "config_log.pkl"), "rb") as f:
        cfg = pkl.load(f)
    st = log["parameters_trial_list"][-1]
    pol = Policy.Throwing_Policy(
        full_state_dim=9, target_dim=3, num_basis=st["centers"].shape[0], u_max=cfg["uM"],
        lengthscales_init=st["log_lengthscales"].exp().numpy()[0],
        centers_init=st["centers"].numpy(), weight_init=st["f_linear.weight"].numpy(),
        flg_drop=False, dtype=torch.float64, device=torch.device("cpu"),
    )
    pol.load_state_dict(st)
    pol.eval()
    return pol, cfg


def make_camera_matrices(client):
    view = p.computeViewMatrix(
        cameraEyePosition=[2.0, -2.0, 1.45], cameraTargetPosition=[0.78, 0.0, 0.22],
        cameraUpVector=[0, 0, 1], physicsClientId=client)
    proj = p.computeProjectionMatrixFOV(
        fov=55, aspect=WIDTH / HEIGHT, nearVal=0.05, farVal=6.0, physicsClientId=client)
    return view, proj


def add_basket(client, center_xy, h, half=0.075, wall_h=0.09, wall_t=0.006):
    """Visual-only bin at (x, y) whose FLOOR is at height h, on a thin support pole."""
    cx, cy = float(center_xy[0]), float(center_xy[1])
    base_rgba = [0.45, 0.28, 0.12, 1.0]
    wall_rgba = [0.90, 0.45, 0.12, 0.55]
    pole_rgba = [0.30, 0.30, 0.33, 1.0]

    def vis_box(he, rgba):
        return p.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=rgba,
                                   physicsClientId=client)

    def body(vis, pos):
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis,
                          basePosition=pos, physicsClientId=client)

    if h > 0.02:  # support pole from floor up to the basket floor
        body(vis_box([0.014, 0.014, h / 2], pole_rgba), [cx, cy, h / 2])
    body(vis_box([half, half, 0.004], base_rgba), [cx, cy, h + 0.004])
    body(vis_box([half, wall_t, wall_h / 2], wall_rgba), [cx, cy + half, h + wall_h / 2])
    body(vis_box([half, wall_t, wall_h / 2], wall_rgba), [cx, cy - half, h + wall_h / 2])
    body(vis_box([wall_t, half, wall_h / 2], wall_rgba), [cx + half, cy, h + wall_h / 2])
    body(vis_box([wall_t, half, wall_h / 2], wall_rgba), [cx - half, cy, h + wall_h / 2])


def demo_targets(cfg, release_xy):
    """Domain-ADAPTIVE spread: fills the arm's real reachable envelope (azimuth,
    distance, height) so the demo shows the widest variety THAT ARM can actually do.
    Fast arms (KUKA) get a wide distance span; the Kinova gets a narrow band but a
    full +-gM azimuth fan and its full height range -- honest to the hardware."""
    lm, lM, gM = cfg["lm"], cfg["lM"], cfg.get("gM", np.pi / 6)
    H_MAX = cfg.get("H_MAX", 0.45)
    flight_mode = bool(cfg.get("flight_targets", False))
    slope = cfg.get("height_slope", None) or 0.0
    # (angle_frac in [-1,1], dist_frac in [0,1], height_frac in [0,1]) chosen for spread
    fracs = [
        (-1.00, 1.00, 0.00),
        ( 0.85, 0.00, 1.00),
        (-0.55, 0.75, 0.35),
        ( 0.95, 0.30, 0.65),
        ( 0.15, 1.00, 0.10),
        (-0.90, 0.10, 0.90),
    ]
    out = []
    for af, df, hf in fracs:
        h = hf * H_MAX
        beta = af * gM
        if flight_mode:
            # azimuth + distance are measured FROM THE RELEASE POINT (trained geometry)
            f_lo = lm - release_xy[0]
            f_hi = max(f_lo + 0.02, (lM - release_xy[0]) - slope * h)
            flight = f_lo + df * (f_hi - f_lo)
            xy = np.asarray(release_xy) + flight * np.array([np.cos(beta), np.sin(beta)])
        else:
            lM_h = max(lm + 0.02, lM - 1.1 * h)   # polar-from-origin (fast arms)
            dist = lm + df * (lM_h - lm)
            xy = np.array([dist * np.cos(beta), dist * np.sin(beta)])
        out.append(np.array([xy[0], xy[1], h]))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--log_path", type=str, default="results_mc_pilot_pb_A_hgen/1")
    ap.add_argument("--robot", type=str, default="kuka_iiwa")
    ap.add_argument("--out", type=str,
                    default=os.path.join(_VIDS, "mc_pilot_kuka_wide_heights_baskets.mp4"))
    ap.add_argument("--fps", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    pol, cfg = load_policy_9d(args.log_path)
    profile = get_robot_profile(args.robot)
    release_pos = np.array(profile.default_release_pos, dtype=float)
    t_w, t_r, _ = profile.timing
    targets = demo_targets(cfg, release_pos[:2])

    def policy(s, t):
        with torch.no_grad():
            inp = torch.tensor(np.asarray(s, dtype=float), dtype=torch.float64).unsqueeze(0)
            return np.array([float(pol(inp, t=0, p_dropout=0.0).item())])

    all_frames, hit = [], 0
    for i, (px, py, h) in enumerate(targets):
        frames = []
        st = {"drawn": False}

        def capture(client, _f=frames, _s=st, _t=(px, py, h)):
            if not _s["drawn"]:
                add_basket(client, (_t[0], _t[1]), _t[2])
                _s["drawn"] = True
            if len(_f) % 2 != 0:
                _f.append(None); return
            view, proj = make_camera_matrices(client)
            img = p.getCameraImage(WIDTH, HEIGHT, viewMatrix=view, projectionMatrix=proj,
                                   renderer=p.ER_TINY_RENDERER, physicsClientId=client)
            _f.append(np.reshape(img[2], (HEIGHT, WIDTH, 4))[:, :, :3].astype(np.uint8))

        system = PyBulletThrowingSystem(robot_name=args.robot, t_w=t_w, t_r=t_r, target_height=float(h))
        system.frame_hook = capture
        s0 = np.concatenate([release_pos, np.zeros(3), np.array([px, py, h])])
        _, _, clean = system.rollout(s0, policy, T=2.0, dt=0.02, noise=0.0)
        land = clean[-1, 0:2]
        err = np.linalg.norm(land - np.array([px, py]))
        hit += int(err < 0.02)
        dist = np.hypot(px, py)
        print(f"throw {i+1}: dist={dist:.2f}m h={h*100:.0f}cm target({px:.2f},{py:.2f}) "
              f"landed({land[0]:.2f},{land[1]:.2f}) err={err*100:.2f}cm", flush=True)
        real = [f for f in frames if f is not None]
        if real:
            all_frames.extend(real)
            all_frames.extend([real[-1]] * (args.fps // 2))

    print(f"\n{hit}/{len(targets)} within 2cm; encoding {len(all_frames)} frames")
    imageio.mimwrite(args.out, all_frames, fps=args.fps, codec="libx264", quality=8)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
