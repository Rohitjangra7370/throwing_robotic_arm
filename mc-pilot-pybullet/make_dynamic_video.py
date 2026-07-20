"""
Video of the dynamic (torque-controlled, physically real) Kinova Gen3 release --
the "hardware config" checkpoint (1.54cm mean accuracy, results_mc_pilot_pb_A_kinova_gen3_dyn).

Captures frames via a frame_hook into PyBulletThrowingSystem.rollout (DIRECT mode,
TinyRenderer -- fast and reliable, per this project's established video method),
draws a visual-only target marker per throw, and encodes to mp4 via imageio/libx264.
"""

import argparse
import os
import pickle as pkl

import imageio.v2 as imageio
import numpy as np
import pybullet as p
import pybullet_data
import torch

import policy_learning.Policy as Policy
from eval_generalization import make_targets
from robot_arm.robot_profiles import get_robot_profile
from simulation_class.model_pybullet import PyBulletThrowingSystem

WIDTH, HEIGHT = 960, 720


def load_policy(log_path):
    with open(os.path.join(log_path, "log.pkl"), "rb") as f:
        log = pkl.load(f)
    with open(os.path.join(log_path, "config_log.pkl"), "rb") as f:
        cfg = pkl.load(f)
    state = log["parameters_trial_list"][-1]
    policy_obj = Policy.Throwing_Policy(
        full_state_dim=8, target_dim=2, num_basis=state["centers"].shape[0],
        u_max=cfg["uM"], lengthscales_init=state["log_lengthscales"].exp().numpy()[0],
        centers_init=state["centers"].numpy(), weight_init=state["f_linear.weight"].numpy(),
        flg_drop=False, dtype=torch.float64, device=torch.device("cpu"),
    )
    policy_obj.load_state_dict(state)
    policy_obj.eval()
    return policy_obj, cfg


def make_camera_matrices(client):
    view = p.computeViewMatrix(
        cameraEyePosition=[1.35, -1.35, 0.95],
        cameraTargetPosition=[0.60, 0.0, 0.30],
        cameraUpVector=[0, 0, 1],
        physicsClientId=client,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=48, aspect=WIDTH / HEIGHT, nearVal=0.05, farVal=5.0,
        physicsClientId=client,
    )
    return view, proj


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--log_path", type=str, default="results_mc_pilot_pb_A_kinova_gen3_dyn/1")
    ap.add_argument("--out", type=str,
                    default="/home/olympusforge/trade/throwing_robotic_arm/status_update/vids/mc_pilot_kinova_dynamic_throws.mp4")
    ap.add_argument("--num_throws", type=int, default=6)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--fps", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    policy_obj, cfg = load_policy(args.log_path)
    profile = get_robot_profile("kinova_gen3_dyn")
    release_pos = np.array(profile.default_release_pos, dtype=float)
    t_w, t_r, _ = profile.timing

    rng = np.random.default_rng(args.seed)
    targets = make_targets(cfg, release_pos[:2], args.num_throws, rng)

    def policy(s, t):
        with torch.no_grad():
            inp = torch.tensor(np.asarray(s, dtype=float), dtype=torch.float64).unsqueeze(0)
            return np.array([float(policy_obj(inp, t=0, p_dropout=0.0).item())])

    all_frames = []
    hit_count = 0
    for i, tgt in enumerate(targets):
        frames = []

        def capture(client, _frames=frames):
            if len(_frames) % 2 != 0:  # capture every other physics step (~25fps of sim motion, upsampled to 50fps output via hold)
                _frames.append(None)
                return
            view, proj = make_camera_matrices(client)
            img = p.getCameraImage(
                WIDTH, HEIGHT, viewMatrix=view, projectionMatrix=proj,
                renderer=p.ER_TINY_RENDERER, physicsClientId=client,
            )
            rgb = np.reshape(img[2], (HEIGHT, WIDTH, 4))[:, :, :3].astype(np.uint8)
            _frames.append(rgb)

        system = PyBulletThrowingSystem(robot_name="kinova_gen3_dyn", t_w=t_w, t_r=t_r)
        system.frame_hook = capture
        s0 = np.concatenate([release_pos, np.zeros(3), tgt])
        _, _, clean = system.rollout(s0, policy, T=2.0, dt=0.02, noise=0.0)
        land = clean[-1, 0:2]
        err = np.linalg.norm(land - tgt)
        hit = err < 0.02
        hit_count += int(hit)
        print(f"throw {i+1}/{args.num_throws}: target ({tgt[0]:.3f},{tgt[1]:.3f}) "
              f"landed ({land[0]:.3f},{land[1]:.3f}) err={err*100:.2f}cm "
              f"{'HIT' if hit else 'miss'} ({len(frames)} frames)", flush=True)

        real_frames = [f for f in frames if f is not None]
        if real_frames:
            # hold the last frame briefly between throws
            all_frames.extend(real_frames)
            all_frames.extend([real_frames[-1]] * (args.fps // 2))

    print(f"\n{hit_count}/{args.num_throws} throws within 2cm")
    print(f"Encoding {len(all_frames)} frames -> {args.out}")
    imageio.mimwrite(args.out, all_frames, fps=args.fps, codec="libx264", quality=8)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
