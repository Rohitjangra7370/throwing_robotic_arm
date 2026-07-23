"""
Video across multiple basket heights: the h=0 (10-real-trial) policy plus the
three zero-new-trial adapted policies (h=0.10/0.20/0.30), each rendered from
its own real trained/adapted weights, real PyBullet rollout (no oracle speed,
no assigned velocity). One clip, back to back, with each bin drawn on a
pedestal at its real height so the visual reads directly.
"""
import argparse
import os
import pickle as pkl

import imageio.v2 as imageio
import numpy as np
import pybullet as p
import torch

import policy_learning.Policy as Policy
from simulation_class.model_pybullet import PyBulletThrowingSystem

W, H = 1024, 720
SAFE_U_CAP = 1.60


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
    return pol


def cam(cid):
    v = p.computeViewMatrix([2.1, -2.1, 1.5], [0.9, 0.0, 0.15], [0, 0, 1], physicsClientId=cid)
    return v, p.computeProjectionMatrixFOV(52, W / H, 0.05, 7.0, physicsClientId=cid)


def draw_bin(cid, cx, cy, z0, half=0.09, wall=0.10, t=0.006):
    def vb(he, rgba):
        return p.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=rgba, physicsClientId=cid)

    def bd(vis, pos):
        p.createMultiBody(0, -1, vis, pos, physicsClientId=cid)

    if z0 > 0.02:
        bd(vb([half * 0.5, half * 0.5, z0 / 2], [0.55, 0.55, 0.55, 1]), [cx, cy, z0 / 2])
    bd(vb([half, half, 0.004], [0.45, 0.28, 0.12, 1]), [cx, cy, z0 + 0.004])
    for dx, dy, hx, hy in [(0, half, half, t), (0, -half, half, t),
                           (half, 0, t, half), (-half, 0, t, half)]:
        bd(vb([hx, hy, wall / 2], [0.9, 0.45, 0.12, 0.6]), [cx + dx, cy + dy, wall / 2 + z0])


def render_height(log_path, n_throws, seed, u_cap):
    cfg = pkl.load(open(os.path.join(log_path, "config_log.pkl"), "rb"))
    pol = load_policy(log_path, cfg)
    # the ground checkpoint predates opt_pose being recorded in config_log
    table_path = cfg.get("opt_pose") or "throw_pose_table.npy"
    table = list(np.load(table_path, allow_pickle=True))
    launch = float(cfg.get("opt_launch_deg", table[0]["elev_deg"]))
    RP = np.array(cfg["release_pos"], dtype=float)
    h = float(cfg["target_height"])
    lm, lM, gM = cfg["lm"], cfg["lM"], cfg["gM"]
    f_lo, f_hi = lm - RP[0], lM - RP[0]

    def policy(s, t):
        with torch.no_grad():
            u = float(pol(torch.tensor(np.asarray(s), dtype=torch.float64)
                         .unsqueeze(0), t=0, p_dropout=0.0).item())
        return np.array([min(u, u_cap)])

    rng = np.random.default_rng(seed)
    frames, errs = [], []
    for i in range(n_throws):
        flight = rng.uniform(f_lo, f_hi)
        beta = rng.uniform(-gM, gM)
        tgt = RP[:2] + flight * np.array([np.cos(beta), np.sin(beta)])

        sysm = PyBulletThrowingSystem(
            mass=cfg["ball_mass"], radius=cfg["ball_radius"],
            launch_angle_deg=launch, arm_noise=None,
            t_w=cfg["T_W"], t_r=cfg["T_R"], robot_name=cfg["robot_name"],
            target_height=h, opt_posture_table=table, opt_launch_deg=launch)

        fr, drawn = [], {"d": False}

        def cap(cid, _fr=fr, _d=drawn, _t=tgt, _h=h):
            if not _d["d"]:
                draw_bin(cid, _t[0], _t[1], _h)
                _d["d"] = True
            if len(_fr) % 2:
                _fr.append(None)
                return
            v, pr = cam(cid)
            img = p.getCameraImage(W, H, viewMatrix=v, projectionMatrix=pr,
                                   renderer=p.ER_TINY_RENDERER, physicsClientId=cid)
            _fr.append(np.reshape(img[2], (H, W, 4))[:, :, :3].astype(np.uint8))

        sysm.frame_hook = cap
        s0 = np.concatenate([RP, np.zeros(3), tgt])
        pos, vel, wind = sysm.rollout(s0, policy, T=cfg["T"], dt=cfg["Ts"], noise=0.0)
        land = pos[-1][:2]
        err = float(np.linalg.norm(land - tgt))
        errs.append(err)
        speed = float(np.linalg.norm(sysm.last_release_info["v_release"])) \
            if sysm.last_release_info is not None else float("nan")
        print(f"  h={h:.2f} throw {i+1}/{n_throws}: target=({tgt[0]:.3f},{tgt[1]:.3f}) "
             f"landed=({land[0]:.3f},{land[1]:.3f}) err={err*100:.1f}cm speed={speed:.3f}m/s")

        real = [f for f in fr if f is not None]
        frames += real + [real[-1]] * 25
    errs = np.array(errs)
    print(f"  h={h:.2f} summary: mean {errs.mean()*100:.1f}cm max {errs.max()*100:.1f}cm "
         f"hit<10cm {100*np.mean(errs<0.10):.0f}%")
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_paths", nargs="+",
                    default=["results_kinetic_chain_gen3/1", "results_kinetic_chain_gen3_h10/1",
                            "results_kinetic_chain_gen3_h20/1", "results_kinetic_chain_gen3_h30/1"])
    ap.add_argument("--n_throws_each", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--u_cap", type=float, default=SAFE_U_CAP)
    ap.add_argument("--out", default="status_update/vids/gen3_heightgen_throws.mp4")
    args = ap.parse_args()

    all_frames = []
    for lp in args.log_paths:
        print(f"=== {lp} ===")
        all_frames += render_height(lp, args.n_throws_each, args.seed, args.u_cap)

    imageio.mimwrite(args.out, all_frames, fps=50, codec="libx264", quality=8)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
