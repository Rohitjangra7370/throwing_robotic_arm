"""
Proper noise-robustness dose-response sweep (fixes the n=10-throws-too-small
issue in eval_generalization.py's noise stress test).

n=50 throws per condition, same targets across conditions, and a real
dose-response range (not just one or two arbitrary levels).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval_generalization import load_policy, make_targets, run_throws
from robot_arm.noise_models import TrackingErrorNoise, VelocitySlipNoise, VelocityBiasNoise
from robot_arm.robot_profiles import get_robot_profile


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--log_path", type=str, default="results_mc_pilot_pb_A_kinova_gen3/1")
    ap.add_argument("--out", type=str, default="results_generalization")
    ap.add_argument("--num_throws", type=int, default=50)
    ap.add_argument("--seed", type=int, default=321)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    policy_obj, cfg = load_policy(args.log_path)
    profile = get_robot_profile("kinova_gen3")
    release_pos = np.array(profile.default_release_pos, dtype=float)
    t_w, t_r, _ = profile.timing
    rng = np.random.default_rng(args.seed)
    targets = make_targets(cfg, release_pos[:2], args.num_throws, rng)

    fit_path = "results_tracking_error/tracking_error.npz"
    cases = [("none (baseline)", None)]
    cases += [(f"bias sigma={s:.2f}", VelocityBiasNoise(sigma=s, seed=1)) for s in [0.02, 0.05, 0.10]]
    cases += [(f"slip a={a:.2f}", VelocitySlipNoise(alpha=a, sigma=0.04, seed=1)) for a in [0.05, 0.10, 0.20, 0.30]]
    if os.path.exists(fit_path):
        cases.append(("TrackingErrorNoise(fitted)", TrackingErrorNoise.from_measurements(fit_path, seed=1)))

    names, means, maxs, stds = [], [], [], []
    print(f"=== Noise dose-response, n={args.num_throws} throws, same targets across conditions ===")
    for name, noise in cases:
        errs = run_throws(policy_obj, targets, release_pos, t_w, t_r, "kinova_gen3", arm_noise=noise)
        names.append(name); means.append(errs.mean()); maxs.append(errs.max()); stds.append(errs.std())
        print(f"  {name:28s}: mean {errs.mean()*100:5.2f}cm  std {errs.std()*100:5.2f}cm  "
              f"max {errs.max()*100:5.2f}cm", flush=True)

    np.savez(os.path.join(args.out, "noise_dose_response.npz"),
             names=np.array(names), means=np.array(means), maxs=np.array(maxs), stds=np.array(stds))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    ax.bar(x, np.array(means) * 100, yerr=np.array(stds) * 100 / np.sqrt(args.num_throws),
          capsize=3)
    ax.axhline(means[0] * 100, color="red", ls="--", lw=1, label="baseline mean")
    ax.set_xticks(x, names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("mean landing error (cm)")
    ax.set_title(f"Noise dose-response (n={args.num_throws} throws/condition, error bars = SEM)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "noise_dose_response.png"), dpi=150)
    print(f"\nSaved {args.out}/noise_dose_response.{{npz,png}}")


if __name__ == "__main__":
    main()
