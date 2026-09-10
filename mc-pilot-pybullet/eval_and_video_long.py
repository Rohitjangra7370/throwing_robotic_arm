"""
Evaluate + visualize the long-range Gen3 policy across a VARIETY of targets.
Prints accuracy by distance and azimuth band, and renders a basket video.
"""
import argparse, os, pickle as pkl
import numpy as np, torch
import pybullet as p
import imageio.v2 as imageio
import policy_learning.Policy as Policy
from robot_arm.robot_profiles import get_robot_profile
from simulation_class.model_pybullet import PyBulletThrowingSystem

_VIDS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "status_update", "vids")

W, H = 1024, 720


def load_policy(log_path):
    log = pkl.load(open(os.path.join(log_path, "log.pkl"), "rb"))
    cfg = pkl.load(open(os.path.join(log_path, "config_log.pkl"), "rb"))
    st = log["parameters_trial_list"][-1]
    pol = Policy.Throwing_Policy(
        full_state_dim=8, target_dim=2, num_basis=st["centers"].shape[0], u_max=cfg["uM"],
        lengthscales_init=st["log_lengthscales"].exp().numpy()[0],
        centers_init=st["centers"].numpy(), weight_init=st["f_linear.weight"].numpy(),
        flg_drop=False, dtype=torch.float64, device=torch.device("cpu"))
    pol.load_state_dict(st); pol.eval()
    return pol, cfg


def varied_targets(cfg, rel_xy, n, rng):
    lm, lM, gM = cfg["lm"], cfg["lM"], cfg.get("gM", np.pi/6)
    f_lo, f_hi = lm - rel_xy[0], lM - rel_xy[0]
    T = []
    for _ in range(n):
        fl = rng.uniform(f_lo, f_hi); b = rng.uniform(-gM, gM)
        T.append(rel_xy + fl*np.array([np.cos(b), np.sin(b)]))
    return np.array(T)


def cam(cid):
    v = p.computeViewMatrix([2.2, -2.2, 1.5], [0.9, 0.0, 0.15], [0, 0, 1], physicsClientId=cid)
    return v, p.computeProjectionMatrixFOV(52, W/H, 0.05, 7.0, physicsClientId=cid)


def basket(cid, c, half=0.08, wall=0.09, t=0.006):
    cx, cy = float(c[0]), float(c[1])
    def vb(he, rgba): return p.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=rgba, physicsClientId=cid)
    def bd(v, pos): p.createMultiBody(0, -1, v, pos, physicsClientId=cid)
    bd(vb([half, half, .004], [.45, .28, .12, 1]), [cx, cy, .004])
    for dx, dy, hx, hy in [(0, half, half, t), (0, -half, half, t), (half, 0, t, half), (-half, 0, t, half)]:
        bd(vb([hx, hy, wall/2], [.9, .45, .12, .55]), [cx+dx, cy+dy, wall/2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_path", default="results_mc_pilot_pb_A_kinova_gen3_long/1")
    ap.add_argument("--robot", default="kinova_gen3")
    ap.add_argument("--n_eval", type=int, default=40)
    ap.add_argument("--n_video", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(_VIDS, "mc_pilot_kinova_long_variety.mp4"))
    args = ap.parse_args()

    pol, cfg = load_policy(args.log_path)
    prof = get_robot_profile(args.robot); rel = np.array(prof.default_release_pos)
    t_w, t_r, _ = prof.timing
    def policy(s, t):
        with torch.no_grad():
            return np.array([float(pol(torch.tensor(np.asarray(s), dtype=torch.float64).unsqueeze(0), t=0).item())])

    # ---- eval on variety ----
    rng = np.random.default_rng(20240721)
    tg = varied_targets(cfg, rel[:2], args.n_eval, rng)
    errs, dists, azs = [], [], []
    for t in tg:
        sysm = PyBulletThrowingSystem(robot_name=args.robot, t_w=t_w, t_r=t_r)
        s0 = np.concatenate([rel, np.zeros(3), t])
        _, _, cl = sysm.rollout(s0, policy, T=2.5, dt=0.02, noise=0.0)
        errs.append(np.linalg.norm(cl[-1, :2] - t))
        dists.append(np.linalg.norm(t)); azs.append(np.degrees(np.arctan2(t[1], t[0])))
    errs, dists, azs = np.array(errs), np.array(dists), np.array(azs)
    print(f"\n=== LONG-RANGE Gen3 (kinematic) eval, n={args.n_eval} ===")
    print(f"overall: mean {errs.mean()*100:.2f}cm  median {np.median(errs)*100:.2f}cm  max {errs.max()*100:.2f}cm")
    print(f"hit<5cm: {100*np.mean(errs<0.05):.0f}%   hit<10cm: {100*np.mean(errs<0.10):.0f}%")
    print(f"throw distance span: {dists.min():.2f} - {dists.max():.2f} m")
    print("by distance band:")
    for lo in [0.7, 1.0, 1.3]:
        m = (dists >= lo) & (dists < lo+0.3)
        if m.any(): print(f"  {lo:.1f}-{lo+0.3:.1f}m: mean {errs[m].mean()*100:.2f}cm (n={m.sum()})")
    print("by azimuth band:")
    for lo in [-30, -10, 10]:
        m = (azs >= lo) & (azs < lo+20)
        if m.any(): print(f"  {lo:+d}..{lo+20:+d}deg: mean {errs[m].mean()*100:.2f}cm (n={m.sum()})")

    # ---- variety video ----
    tgv = varied_targets(cfg, rel[:2], args.n_video, np.random.default_rng(7))
    frames = []
    for t in tgv:
        fr = []; drawn = {"d": False}
        def cap(cid, _f=fr, _s=drawn, _t=t):
            if not _s["d"]: basket(cid, _t); _s["d"] = True
            if len(_f) % 2: _f.append(None); return
            v, pr = cam(cid)
            img = p.getCameraImage(W, H, viewMatrix=v, projectionMatrix=pr, renderer=p.ER_TINY_RENDERER, physicsClientId=cid)
            _f.append(np.reshape(img[2], (H, W, 4))[:, :, :3].astype(np.uint8))
        sysm = PyBulletThrowingSystem(robot_name=args.robot, t_w=t_w, t_r=t_r); sysm.frame_hook = cap
        sysm.rollout(np.concatenate([rel, np.zeros(3), t]), policy, T=2.5, dt=0.02, noise=0.0)
        real = [f for f in fr if f is not None]
        frames += real + [real[-1]] * 25
    imageio.mimwrite(args.out, frames, fps=50, codec="libx264", quality=8)
    print(f"\nsaved variety video -> {args.out}")


if __name__ == "__main__":
    main()
