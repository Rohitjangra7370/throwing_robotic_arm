"""
Generalization / robustness sweeps for trained kinova flight-space checkpoints.

Three sweeps, all through the true PyBulletThrowingSystem.rollout pipeline:

  1. object   — ball mass x radius grid, dynamic (torque) release. The policy
                and GP were trained on the 57.7 g / 3.27 cm ball; this measures
                how landing error degrades for unseen objects (drag + payload
                dynamics both change). Payload compensation reads the actual
                ball mass, so the controller adapts; the POLICY does not.
  2. height   — NOT a generalization test. A ground-trained (height-blind)
                policy has no way to retarget for an elevated plane; cutting
                the same ground-aimed trajectory short at height h and
                comparing against the *original ground-XY target* measures
                naive-transfer sensitivity (how much a shorter, un-adapted
                flight misses by), not whether the arm/policy can hit a true
                3-D target. That's expected to grow with h for any policy.
                A real height-generalized kinova policy (analogous to the
                existing kuka h25/h45/hgen work) is a separate, unbuilt
                capability — flight budget check (2026-07-19): ~0.15-0.19m
                flight survives a 0.05-0.20m height range at u~0.57, so it's
                buildable, just not what this sweep measures.
  3. noise    — release-velocity noise stress on the kinematic profile
                (torque profiles reject arm_noise by design): fitted
                TrackingErrorNoise and VelocitySlipNoise levels.

Usage:
  python3 eval_generalization.py --log_path results_mc_pilot_pb_A_kinova_gen3/1 \
      --out results_generalization
"""

import argparse
import os
import pickle as pkl

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import policy_learning.Policy as Policy
from robot_arm.noise_models import TrackingErrorNoise, VelocitySlipNoise
from robot_arm.robot_profiles import get_robot_profile
from simulation_class.model_pybullet import PyBulletThrowingSystem


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


def make_targets(cfg, release_xy, n, rng):
    lm, lM, gM = cfg["lm"], cfg["lM"], cfg.get("gM", np.pi / 6)
    flight_mode = bool(cfg.get("flight_targets", False))
    targets = []
    for _ in range(n):
        if flight_mode:
            flight = rng.uniform(lm - release_xy[0], lM - release_xy[0])
            beta = rng.uniform(-gM, gM)
            targets.append(release_xy + flight * np.array([np.cos(beta), np.sin(beta)]))
        else:
            dist = rng.uniform(lm, lM)
            ang = rng.uniform(-gM, gM)
            targets.append(np.array([dist * np.cos(ang), dist * np.sin(ang)]))
    return np.array(targets)


def run_throws(policy_obj, targets, release_pos, t_w, t_r, robot_name,
               mass=0.0577, radius=0.0327, target_height=0.0, arm_noise=None):
    system = PyBulletThrowingSystem(
        mass=mass, radius=radius, robot_name=robot_name,
        t_w=t_w, t_r=t_r, target_height=target_height, arm_noise=arm_noise,
    )

    def policy(s, t):
        with torch.no_grad():
            inp = torch.tensor(np.asarray(s, dtype=float),
                               dtype=torch.float64).unsqueeze(0)
            return np.array([float(policy_obj(inp, t=0, p_dropout=0.0).item())])

    errs = []
    for tgt in targets:
        s0 = np.concatenate([release_pos, np.zeros(3), tgt])
        _, _, clean = system.rollout(s0, policy, T=2.0, dt=0.02, noise=0.0)
        errs.append(float(np.linalg.norm(clean[-1, 0:2] - tgt)))
    return np.array(errs)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--log_path", type=str,
                    default="results_mc_pilot_pb_A_kinova_gen3/1")
    ap.add_argument("--out", type=str, default="results_generalization")
    ap.add_argument("--num_throws", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    policy_obj, cfg = load_policy(args.log_path)
    profile = get_robot_profile("kinova_gen3")
    release_pos = np.array(profile.default_release_pos, dtype=float)
    t_w, t_r, _ = profile.timing
    rng = np.random.default_rng(args.seed)
    targets = make_targets(cfg, release_pos[:2], args.num_throws, rng)

    results = {}

    # ---- 1. object sweep (dynamic release) --------------------------------
    masses = [0.030, 0.0577, 0.100, 0.150]
    radii = [0.020, 0.0327, 0.045]
    print("=== 1. Object sweep (mass x radius), dynamic release ===")
    obj_grid = np.zeros((len(masses), len(radii)))
    for i, m in enumerate(masses):
        for j, r in enumerate(radii):
            errs = run_throws(policy_obj, targets, release_pos, t_w, t_r,
                              "kinova_gen3_dyn", mass=m, radius=r)
            obj_grid[i, j] = errs.mean()
            trained = " (trained object)" if (m == 0.0577 and r == 0.0327) else ""
            print(f"  mass={m*1000:5.1f}g radius={r*100:.2f}cm: "
                  f"mean {errs.mean()*100:5.2f}cm max {errs.max()*100:5.2f}cm{trained}",
                  flush=True)
    results["object_grid"] = obj_grid
    results["masses"] = np.array(masses)
    results["radii"] = np.array(radii)

    # ---- 2. height sweep (ground-trained policy) --------------------------
    print("\n=== 2. Naive height-transfer sensitivity (NOT a generalization test -- see docstring) ===")
    heights = [0.0, 0.1, 0.2, 0.3]
    h_rows = []
    for h in heights:
        row = {}
        for label, robot in (("kinematic", "kinova_gen3"), ("dynamic", "kinova_gen3_dyn")):
            errs = run_throws(policy_obj, targets, release_pos, t_w, t_r,
                              robot, target_height=h)
            row[label] = errs.mean()
        h_rows.append(row)
        print(f"  h={h:.1f}m: kinematic {row['kinematic']*100:5.2f}cm  "
              f"dynamic {row['dynamic']*100:5.2f}cm", flush=True)
    results["heights"] = np.array(heights)
    results["height_kin"] = np.array([r["kinematic"] for r in h_rows])
    results["height_dyn"] = np.array([r["dynamic"] for r in h_rows])

    # ---- 3. noise stress (kinematic profile only) -------------------------
    print("\n=== 3. Release-noise stress (kinematic profile) ===")
    noise_cases = [("none", None)]
    fit_path = "results_tracking_error/tracking_error.npz"
    if os.path.exists(fit_path):
        noise_cases.append(
            ("TrackingErrorNoise(fitted)",
             TrackingErrorNoise.from_measurements(fit_path, seed=0))
        )
    noise_cases += [
        ("slip a=0.10", VelocitySlipNoise(alpha=0.10, sigma=0.04, seed=0)),
        ("slip a=0.20", VelocitySlipNoise(alpha=0.20, sigma=0.04, seed=0)),
    ]
    noise_names, noise_means = [], []
    for name, noise in noise_cases:
        errs = run_throws(policy_obj, targets, release_pos, t_w, t_r,
                          "kinova_gen3", arm_noise=noise)
        noise_names.append(name)
        noise_means.append(errs.mean())
        print(f"  {name:28s}: mean {errs.mean()*100:5.2f}cm max {errs.max()*100:5.2f}cm",
              flush=True)
    results["noise_names"] = np.array(noise_names)
    results["noise_means"] = np.array(noise_means)

    np.savez(os.path.join(args.out, "generalization.npz"), **results)

    # ---- figure -----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im = axes[0].imshow(obj_grid * 100, cmap="viridis", origin="lower", aspect="auto")
    axes[0].set_xticks(range(len(radii)), [f"{r*100:.1f}" for r in radii])
    axes[0].set_yticks(range(len(masses)), [f"{m*1000:.0f}" for m in masses])
    axes[0].set_xlabel("radius (cm)")
    axes[0].set_ylabel("mass (g)")
    axes[0].set_title("Object sweep: mean landing error (cm)\n(trained: 58g / 3.3cm)")
    fig.colorbar(im, ax=axes[0])

    axes[1].plot(heights, results["height_kin"] * 100, "o-", label="kinematic")
    axes[1].plot(heights, results["height_dyn"] * 100, "s-", label="dynamic")
    axes[1].set_xlabel("landing-plane height (m)")
    axes[1].set_ylabel("mean landing error (cm)")
    axes[1].set_title("Elevated targets, ground-trained policy")
    axes[1].legend()

    axes[2].bar(range(len(noise_names)), np.array(noise_means) * 100)
    axes[2].set_xticks(range(len(noise_names)), noise_names,
                       rotation=20, ha="right", fontsize=8)
    axes[2].set_ylabel("mean landing error (cm)")
    axes[2].set_title("Release-noise stress (kinematic)")

    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "generalization.png"), dpi=150)
    print(f"\nSaved {args.out}/generalization.npz and generalization.png")


if __name__ == "__main__":
    main()
