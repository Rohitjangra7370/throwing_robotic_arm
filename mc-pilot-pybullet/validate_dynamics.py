"""
Full validation sweep for the velocity-from-dynamics (torque control) work.

Runs three things and saves plots + npz under --out:
  1. Torque-margin / time-scaling map — reuses results_tracking_error/tracking_error.npz
     if present (else runs a fresh sweep). Shows how close to torque saturation the
     arm runs, and how much the throw phase gets stretched, across the full
     speed x angle grid.
  2. Joint-tracking-error grid — a coarser grid, stepping the controller directly
     (bypassing rollout) to record max |q_des - q_meas| per throw. Answers "does the
     controller actually track" at every commanded speed/angle, not just one slow throw.
  3. Multi-seed sim2sim gap — all 5 trained kinova_gen3 checkpoints, replayed under
     both the kinematic release used in training and torque-tracked dynamic release,
     same random targets per seed. Boxplot + per-seed stats.

Usage:
  python3 validate_dynamics.py --out results_dynamics_validation
"""

import argparse
import os
import pickle as pkl

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
import torch

import policy_learning.Policy as Policy
from robot_arm.arm_controller import ArmController
from robot_arm.robot_profiles import get_robot_profile
from simulation_class.model_pybullet import PyBulletThrowingSystem
from measure_tracking_error import run_sweep, POS_ERR_FLAG


# ---------------------------------------------------------------------------
# 1. Torque-margin / time-scaling map
# ---------------------------------------------------------------------------

def torque_margin_map(out_dir, sweep_npz=None):
    if sweep_npz is not None and os.path.exists(sweep_npz):
        d = np.load(sweep_npz)
        print(f"Reusing existing sweep: {sweep_npz}")
    else:
        u_grid = np.linspace(0.3, 1.0, 25)
        angle_grid = np.deg2rad(np.linspace(-30.0, 30.0, 9))
        rec = run_sweep(u_grid, angle_grid, robot_name="kinova_gen3_dyn")
        d = rec

    u = d["u_cmd"]
    time_scale = d["time_scale"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(u, time_scale, s=14)
    ax.axhline(1.0, color="gray", lw=0.5, label="no stretch needed")
    ax.set_xlabel("commanded speed u (m/s)")
    ax.set_ylabel("throw-phase time_scale")
    ax.set_title("Torque-feasibility time scaling vs commanded speed")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "torque_margin.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"\n=== Torque margin ===")
    print(f"  time_scale range: [{time_scale.min():.2f}, {time_scale.max():.2f}]")
    for lo in np.arange(0.3, 1.0, 0.1):
        band = (u >= lo) & (u < lo + 0.1)
        if band.any():
            print(f"  u in [{lo:.1f},{lo+0.1:.1f}): mean time_scale {time_scale[band].mean():.2f}")
    print(f"  saved {path}")
    return d


# ---------------------------------------------------------------------------
# 2. Joint-tracking-error grid (direct controller stepping, not rollout)
# ---------------------------------------------------------------------------

def joint_tracking_grid(out_dir, u_grid, angle_grid):
    prof = get_robot_profile("kinova_gen3_dyn")
    t_w, t_r, T = prof.timing
    dt = 0.02

    client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(dt, physicsClientId=client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.loadURDF("plane.urdf", physicsClientId=client)
    urdf = pybullet_data.getDataPath() + "/" + prof.urdf_rel_path
    arm = ArmController(client, urdf, robot_name="kinova_gen3_dyn")

    alpha = np.deg2rad(35.0)
    release_pos = np.array(prof.default_release_pos, dtype=float)

    rec_u, rec_ang, rec_err, rec_ts = [], [], [], []
    n_total = len(u_grid) * len(angle_grid)
    i = 0
    for u in u_grid:
        for ang in angle_grid:
            i += 1
            arm.reset()
            v_cmd = np.array([
                u * np.cos(alpha) * np.cos(ang),
                u * np.cos(alpha) * np.sin(ang),
                u * np.sin(alpha),
            ])
            coeffs, _, _, _ = arm.plan_throw(v_cmd, release_pos, t_w, t_r, T)
            max_err = 0.0
            n_steps = int(coeffs["t_r"] / dt)
            for step in range(n_steps):
                t = step * dt
                q_t, qd_t, qdd_t = arm.get_setpoint(coeffs, t, with_accel=True)
                arm.step(q_t, qd_t, qdd_t)
                p.stepSimulation(physicsClientId=client)
                if t > coeffs["t_w"]:
                    states = p.getJointStates(arm.arm_id, arm.joint_ids, physicsClientId=client)
                    q_meas = np.array([s[0] for s in states])
                    max_err = max(max_err, float(np.max(np.abs(q_t - q_meas))))
            rec_u.append(u)
            rec_ang.append(ang)
            rec_err.append(max_err)
            rec_ts.append(coeffs["time_scale"])
            print(f"[{i}/{n_total}] u={u:.2f} ang={np.rad2deg(ang):+.0f}deg "
                  f"max_joint_err={max_err:.4f} rad", flush=True)
    p.disconnect(client)

    rec = {
        "u": np.array(rec_u), "angle": np.array(rec_ang),
        "max_joint_err": np.array(rec_err), "time_scale": np.array(rec_ts),
    }
    np.savez(os.path.join(out_dir, "joint_tracking_grid.npz"), **rec)

    n_u, n_a = len(u_grid), len(angle_grid)
    err_grid = rec["max_joint_err"].reshape(n_u, n_a)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(err_grid, aspect="auto", origin="lower",
                   extent=[np.rad2deg(angle_grid[0]), np.rad2deg(angle_grid[-1]),
                          u_grid[0], u_grid[-1]], cmap="viridis")
    fig.colorbar(im, ax=ax, label="max joint tracking error (rad)")
    ax.set_xlabel("target azimuth (deg)")
    ax.set_ylabel("commanded speed u (m/s)")
    ax.set_title("Joint tracking error across the throw grid")
    fig.tight_layout()
    path = os.path.join(out_dir, "joint_tracking_error.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"\n=== Joint tracking ===")
    print(f"  max over grid: {rec['max_joint_err'].max():.4f} rad "
          f"(gate threshold 0.02 rad from the slow-throw test)")
    print(f"  mean over grid: {rec['max_joint_err'].mean():.4f} rad")
    print(f"  saved {path}")
    return rec


# ---------------------------------------------------------------------------
# 3. Multi-seed sim2sim gap
# ---------------------------------------------------------------------------

def load_policy(log_path):
    with open(os.path.join(log_path, "log.pkl"), "rb") as f:
        log = pkl.load(f)
    with open(os.path.join(log_path, "config_log.pkl"), "rb") as f:
        cfg = pkl.load(f)
    state = log["parameters_trial_list"][-1]
    policy_obj = Policy.Throwing_Policy(
        full_state_dim=8,
        target_dim=2,
        num_basis=state["centers"].shape[0],
        u_max=cfg["uM"],
        lengthscales_init=state["log_lengthscales"].exp().numpy()[0],
        centers_init=state["centers"].numpy(),
        weight_init=state["f_linear.weight"].numpy(),
        flg_drop=False,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    policy_obj.load_state_dict(state)
    policy_obj.eval()
    return policy_obj, cfg


def multiseed_sim2sim(out_dir, seeds, num_throws, rng_seed=123):
    profile = get_robot_profile("kinova_gen3")
    release_pos = np.array(profile.default_release_pos, dtype=float)
    t_w, t_r, _ = profile.timing

    all_kin, all_dyn = {}, {}
    for seed in seeds:
        log_path = f"results_mc_pilot_pb_A_kinova_gen3/{seed}"
        policy_obj, cfg = load_policy(log_path)
        lm, lM, gM = cfg["lm"], cfg["lM"], cfg.get("gM", np.pi / 6)

        def policy(s, t, _p=policy_obj):
            with torch.no_grad():
                inp = torch.tensor(np.asarray(s, dtype=float), dtype=torch.float64).unsqueeze(0)
                return np.array([float(_p(inp, t=0, p_dropout=0.0).item())])

        rng = np.random.default_rng(rng_seed + seed)
        targets = []
        for _ in range(num_throws):
            dist = rng.uniform(lm, lM)
            ang = rng.uniform(-gM, gM)
            targets.append([dist * np.cos(ang), dist * np.sin(ang)])
        targets = np.array(targets)

        for label, robot, store in (("kinematic", "kinova_gen3", all_kin),
                                    ("dynamic", "kinova_gen3_dyn", all_dyn)):
            system = PyBulletThrowingSystem(robot_name=robot, t_w=t_w, t_r=t_r)
            errs = []
            for tgt in targets:
                s0 = np.concatenate([release_pos, np.zeros(3), tgt])
                _, _, clean = system.rollout(s0, policy, T=2.0, dt=0.02, noise=0.0)
                errs.append(float(np.linalg.norm(clean[-1, 0:2] - tgt)))
            store[seed] = np.array(errs)
            print(f"seed {seed} {label}: mean {np.mean(errs)*100:.2f}cm "
                  f"max {np.max(errs)*100:.2f}cm", flush=True)

    np.savez(os.path.join(out_dir, "sim2sim_multiseed.npz"),
             seeds=np.array(seeds),
             **{f"kin_s{s}": all_kin[s] for s in seeds},
             **{f"dyn_s{s}": all_dyn[s] for s in seeds})

    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = []
    data = []
    labels = []
    for i, s in enumerate(seeds):
        positions += [i * 3, i * 3 + 1]
        data += [all_kin[s] * 100, all_dyn[s] * 100]
        labels += [f"s{s} kin", f"s{s} dyn"]
    bp = ax.boxplot(data, positions=positions, widths=0.8, patch_artist=True)
    for j, box in enumerate(bp["boxes"]):
        box.set_facecolor("#4C72B0" if j % 2 == 0 else "#DD8452")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("landing error (cm)")
    ax.set_title("Sim2sim gap: kinematic (train) vs dynamic (torque) release, per seed")
    fig.tight_layout()
    path = os.path.join(out_dir, "sim2sim_multiseed_boxplot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    kin_all = np.concatenate([all_kin[s] for s in seeds]) * 100
    dyn_all = np.concatenate([all_dyn[s] for s in seeds]) * 100
    print(f"\n=== Multi-seed sim2sim gap ({len(seeds)} seeds x {num_throws} throws) ===")
    print(f"  kinematic: mean {kin_all.mean():.2f}cm, max {kin_all.max():.2f}cm, std {kin_all.std():.2f}cm")
    print(f"  dynamic:   mean {dyn_all.mean():.2f}cm, max {dyn_all.max():.2f}cm, std {dyn_all.std():.2f}cm")
    print(f"  saved {path}")
    return kin_all, dyn_all


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--out", type=str, default="results_dynamics_validation")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--throws_per_seed", type=int, default=15)
    ap.add_argument("--joint_grid_u", type=int, default=5)
    ap.add_argument("--joint_grid_angle", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("### 1/3 Torque-margin map ###")
    torque_margin_map(args.out, sweep_npz="results_tracking_error/tracking_error.npz")

    print("\n### 2/3 Joint-tracking-error grid ###")
    u_grid = np.linspace(0.3, 1.0, args.joint_grid_u)
    angle_grid = np.deg2rad(np.linspace(-30.0, 30.0, args.joint_grid_angle))
    joint_tracking_grid(args.out, u_grid, angle_grid)

    print("\n### 3/3 Multi-seed sim2sim gap ###")
    multiseed_sim2sim(args.out, args.seeds, args.throws_per_seed)

    print(f"\nAll validation artifacts saved under {args.out}/")


if __name__ == "__main__":
    main()
