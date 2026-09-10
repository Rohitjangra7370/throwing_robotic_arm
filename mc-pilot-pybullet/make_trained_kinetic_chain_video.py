"""
Video of the TRAINED opt_pose (kinetic-chain) policy, using the real training
rollout class (PyBulletThrowingSystem with opt_posture_table) and the real
trained weights from log.pkl -- no bespoke throw mechanics, no oracle speed.
"""
import argparse
import pickle as pkl
import os

import numpy as np
import torch
import pybullet as p
import imageio.v2 as imageio

import policy_learning.Policy as Policy
from robot_arm.robot_profiles import get_robot_profile
from simulation_class.model_pybullet import PyBulletThrowingSystem

W, H = 1024, 720


def load_policy(log_path, cfg):
    with open(os.path.join(log_path, "log.pkl"), "rb") as f:
        log = pkl.load(f)
    st = log["parameters_trial_list"][-1]
    pol = Policy.Throwing_Policy(
        full_state_dim=8, target_dim=2, num_basis=st["centers"].shape[0], u_max=cfg["uM"],
        lengthscales_init=st["log_lengthscales"].exp().numpy()[0],
        centers_init=st["centers"].numpy(), weight_init=st["f_linear.weight"].numpy(),
        flg_drop=False, dtype=torch.float64, device=torch.device("cpu"))
    pol.load_state_dict(st)
    pol.eval()
    return pol, len(log["parameters_trial_list"])


def cam(cid):
    v = p.computeViewMatrix([2.1, -2.1, 1.5], [0.9, 0.0, 0.15], [0, 0, 1], physicsClientId=cid)
    return v, p.computeProjectionMatrixFOV(52, W / H, 0.05, 7.0, physicsClientId=cid)


def draw_bin(cid, cx, cy, half=0.09, wall=0.10, t=0.006):
    def vb(he, rgba):
        return p.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=rgba, physicsClientId=cid)

    def bd(vis, pos):
        p.createMultiBody(0, -1, vis, pos, physicsClientId=cid)

    bd(vb([half, half, 0.004], [0.45, 0.28, 0.12, 1]), [cx, cy, 0.004])
    for dx, dy, hx, hy in [(0, half, half, t), (0, -half, half, t),
                           (half, 0, t, half), (-half, 0, t, half)]:
        bd(vb([hx, hy, wall / 2], [0.9, 0.45, 0.12, 0.6]), [cx + dx, cy + dy, wall / 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_path", default="results_kinetic_chain_gen3/1")
    ap.add_argument("--opt_pose", default="throw_pose_table.npy")
    ap.add_argument("--n_throws", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="status_update/vids/gen3_trained_kinetic_chain.mp4")
    ap.add_argument("--tool_offset_z", type=float, default=0.0,
                    help="TCP offset the --opt_pose table was searched at; must "
                         "match its stamp (see run_hardware_throw.py).")
    args = ap.parse_args()

    with open(os.path.join(args.log_path, "config_log.pkl"), "rb") as f:
        cfg = pkl.load(f)
    pol, n_trials = load_policy(args.log_path, cfg)
    print(f"Loaded trained policy: {n_trials} trials logged, robot={cfg['robot_name']}")

    profile = get_robot_profile(cfg["robot_name"])
    RELEASE_POS = np.array(cfg["release_pos"], dtype=float)

    table = list(np.load(args.opt_pose, allow_pickle=True))
    launch_deg = float(table[0]["elev_deg"])
    from run_hardware_throw import _check_tool_offset_matches_table
    _check_tool_offset_matches_table(table, args.tool_offset_z)

    def policy(s, t):
        with torch.no_grad():
            inp = torch.tensor(np.asarray(s), dtype=torch.float64).unsqueeze(0)
            return np.array([float(pol(inp, t=0, p_dropout=0.0).item())])

    # same flight-annulus sampling the training run used (--flight_targets)
    _az0 = min(table, key=lambda e: abs(e["azimuth_deg"]))
    _cid0 = p.connect(p.DIRECT)
    _arm0 = p.loadURDF(
        __import__("pybullet_data").getDataPath() + "/" + profile.urdf_rel_path,
        useFixedBase=True, physicsClientId=_cid0)
    for j in range(7):
        p.resetJointState(_arm0, j, _az0["q"][j], physicsClientId=_cid0)
    release_xy = np.array(
        p.getLinkState(_arm0, profile.ee_link, computeForwardKinematics=True,
                       physicsClientId=_cid0)[4][:2])
    p.disconnect(_cid0)

    lm, lM, gM = cfg["lm"], cfg["lM"], cfg["gM"]
    f_lo, f_hi = lm - release_xy[0], lM - release_xy[0]
    rng = np.random.default_rng(args.seed)

    def sample_target():
        flight = rng.uniform(f_lo, f_hi)
        beta = rng.uniform(-gM, gM)
        return release_xy + flight * np.array([np.cos(beta), np.sin(beta)])

    targets = [sample_target() for _ in range(args.n_throws)]

    frames = []
    errs = []
    for i, tgt in enumerate(targets):
        sysm = PyBulletThrowingSystem(
            mass=cfg["ball_mass"], radius=cfg["ball_radius"],
            launch_angle_deg=launch_deg, arm_noise=None,
            t_w=cfg["T_W"], t_r=cfg["T_R"], robot_name=cfg["robot_name"],
            target_height=cfg["target_height"], opt_posture_table=table,
            opt_launch_deg=launch_deg,
            base_height=float(cfg.get("base_height", 0.0)),
            tool_offset=[0.0, 0.0, args.tool_offset_z])

        fr, drawn = [], {"d": False}

        def cap(cid, _fr=fr, _d=drawn, _t=tgt):
            if not _d["d"]:
                draw_bin(cid, _t[0], _t[1])
                _d["d"] = True
            if len(_fr) % 2:
                _fr.append(None)
                return
            v, pr = cam(cid)
            img = p.getCameraImage(W, H, viewMatrix=v, projectionMatrix=pr,
                                   renderer=p.ER_TINY_RENDERER, physicsClientId=cid)
            _fr.append(np.reshape(img[2], (H, W, 4))[:, :, :3].astype(np.uint8))

        sysm.frame_hook = cap
        s0 = np.concatenate([RELEASE_POS, np.zeros(3), tgt])
        pos, vel, wind = sysm.rollout(s0, policy, T=cfg["T"], dt=cfg["Ts"], noise=0.0)
        land = pos[-1][:2]
        err = float(np.linalg.norm(land - tgt))
        errs.append(err)
        speed = float(np.linalg.norm(sysm.last_release_info["v_release"])) \
            if sysm.last_release_info is not None else float("nan")
        print(f"throw {i+1}/{args.n_throws}: target=({tgt[0]:.3f},{tgt[1]:.3f}) "
             f"landed=({land[0]:.3f},{land[1]:.3f}) err={err*100:.1f}cm "
             f"release_speed={speed:.3f}m/s")

        real = [f for f in fr if f is not None]
        frames += real + [real[-1]] * 25

    imageio.mimwrite(args.out, frames, fps=50, codec="libx264", quality=8)
    errs = np.array(errs)
    print(f"\nmean err {errs.mean()*100:.1f}cm  median {np.median(errs)*100:.1f}cm  "
         f"max {errs.max()*100:.1f}cm  hit<10cm {100*np.mean(errs<0.10):.0f}%")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
