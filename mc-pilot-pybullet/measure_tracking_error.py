"""
Measure command-vs-actual release-velocity error for a torque-mode profile.

Sweeps commanded release speed x target azimuth through the TRUE
PyBulletThrowingSystem.rollout pipeline (hand-rolled replays showed ~10 cm
systematic discrepancy in earlier work) and records the actual ball velocity
at dynamic release. Sim is deterministic: the distribution comes from command
diversity, not repeats.

Sim control rate is 50 Hz (Ts = 0.02 s); real Gen3 low-level Kortex control is
1 kHz — that rate gap is a documented sim-vs-real delta, not corrected here.

Usage:
  python3 measure_tracking_error.py                # full 25x9 grid, 225 throws
  python3 measure_tracking_error.py --quick        # 5x3 grid for smoke tests
  python3 measure_tracking_error.py --out results_tracking_error
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robot_arm.robot_profiles import get_robot_profile
from simulation_class.model_pybullet import PyBulletThrowingSystem

POS_ERR_FLAG = 0.05  # m; throws worse than this are flagged + excluded from fits


def run_sweep(u_grid, angle_grid, robot_name="kinova_gen3_dyn"):
    profile = get_robot_profile(robot_name)
    t_w, t_r, _ = profile.timing
    release_pos = np.array(profile.default_release_pos, dtype=float)
    # Target azimuth is set through the target position; distance only fixes
    # the aim direction (speed is commanded directly), so mid-band is fine.
    dist = 0.77

    system = PyBulletThrowingSystem(robot_name=robot_name, t_w=t_w, t_r=t_r)

    rec = {k: [] for k in ("u_cmd", "angle", "v_cmd", "v_planned", "v_release",
                           "release_pos_err", "time_scale", "land_xy", "flag")}
    n_total = len(u_grid) * len(angle_grid)
    i = 0
    for u in u_grid:
        for ang in angle_grid:
            i += 1
            target_xy = np.array([dist * np.cos(ang), dist * np.sin(ang)])
            s0 = np.concatenate([release_pos, np.zeros(3), target_xy])
            policy = lambda s, t, _u=u: np.array([_u])
            _, _, clean = system.rollout(s0, policy, T=2.0, dt=0.02, noise=0.0)
            info = system.last_release_info

            rec["u_cmd"].append(u)
            rec["angle"].append(ang)
            rec["v_cmd"].append(info["v_cmd"])
            rec["v_planned"].append(info["v_planned"])
            rec["v_release"].append(info["v_release"])
            rec["release_pos_err"].append(info["release_pos_err"])
            rec["time_scale"].append(info["time_scale"])
            rec["land_xy"].append(clean[-1, 0:2])
            rec["flag"].append(1.0 if info["release_pos_err"] > POS_ERR_FLAG else 0.0)
            print(f"[{i}/{n_total}] u={u:.3f} ang={np.rad2deg(ang):+.0f}deg "
                  f"|dv|={np.linalg.norm(info['v_release'] - info['v_cmd']):.4f} "
                  f"pos_err={info['release_pos_err'] * 100:.2f}cm "
                  f"ts={info['time_scale']:.2f}", flush=True)

    return {k: np.array(v) for k, v in rec.items()}


def print_stats(rec):
    ok = rec["flag"] < 0.5
    n_flag = int(rec["flag"].sum())
    print(f"\n=== Tracking-error stats ({ok.sum()} throws, {n_flag} flagged/excluded) ===")
    dv = rec["v_release"][ok] - rec["v_cmd"][ok]
    for j, name in enumerate("xyz"):
        print(f"  dv_{name}: mean {dv[:, j].mean():+.4f}  std {dv[:, j].std():.4f} m/s")
    u = rec["u_cmd"][ok]
    print("  per u-band (|dv| mean):")
    for lo in np.arange(0.3, 1.0, 0.1):
        band = (u >= lo) & (u < lo + 0.1)
        if band.any():
            mags = np.linalg.norm(dv[band], axis=1)
            print(f"    u in [{lo:.1f},{lo + 0.1:.1f}): {mags.mean():.4f} m/s (n={band.sum()})")


def save_figure(rec, path):
    ok = rec["flag"] < 0.5
    dv = rec["v_release"] - rec["v_cmd"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    for j, (ax, name) in enumerate(zip(axes, "xyz")):
        ax.scatter(rec["u_cmd"][ok], dv[ok, j], s=12, label="ok")
        if (~ok).any():
            ax.scatter(rec["u_cmd"][~ok], dv[~ok, j], s=12, c="red", label="flagged")
        ax.axhline(0.0, color="gray", lw=0.5)
        ax.set_xlabel("commanded speed u (m/s)")
        ax.set_ylabel(f"dv_{name} (m/s)")
        ax.legend(fontsize=7)
    fig.suptitle("Release-velocity tracking error vs commanded speed (Gen3 torque mode)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--robot", type=str, default="kinova_gen3_dyn")
    ap.add_argument("--out", type=str, default="results_tracking_error")
    ap.add_argument("--quick", action="store_true", help="5x3 grid instead of 25x9")
    args = ap.parse_args()

    if args.quick:
        u_grid = np.linspace(0.3, 1.0, 5)
        angle_grid = np.deg2rad(np.linspace(-30.0, 30.0, 3))
    else:
        u_grid = np.linspace(0.3, 1.0, 25)
        angle_grid = np.deg2rad(np.linspace(-30.0, 30.0, 9))

    rec = run_sweep(u_grid, angle_grid, robot_name=args.robot)
    os.makedirs(args.out, exist_ok=True)
    npz_path = os.path.join(args.out, "tracking_error.npz")
    np.savez(npz_path, **rec)
    save_figure(rec, os.path.join(args.out, "tracking_error.png"))
    print_stats(rec)
    print(f"\nSaved {npz_path} and tracking_error.png")


if __name__ == "__main__":
    main()
