#!/usr/bin/env python3
"""
Analyze wind experiment results and generate comparison report.

Loads log.pkl and config_log.pkl from each experiment directory,
computes landing error statistics, and prints a summary table.

Usage:
  python analyze_wind_results.py
  python analyze_wind_results.py --seed 1 --hit_threshold 0.1
"""

import argparse
import os
import pickle as pkl
import glob

import numpy as np

parser = argparse.ArgumentParser("Analyze MC-PILOT wind experiment results")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--hit_threshold", type=float, default=0.1,
                    help="hit threshold in meters")
args = parser.parse_args()

# Experiment result directories to scan
result_dirs = {
    # W1: Constant wind
    "W1-calm":     f"results_wind_W1/w0p0_blind/{args.seed}",
    "W1-light":    f"results_wind_W1/w2p5_blind/{args.seed}",
    "W1-moderate": f"results_wind_W1/w5p0_blind/{args.seed}",
    "W1-strong":   f"results_wind_W1/w8p0_blind/{args.seed}",
    "W1-aware":    f"results_wind_W1/w5p0_aware/{args.seed}",
    # W2: Gusts
    "W2-blind":    f"results_wind_W2/wmax4.0_blind/{args.seed}",
    "W2-aware":    f"results_wind_W2/wmax4.0_aware/{args.seed}",
    # W3: Turbulence
    "W3-blind":    f"results_wind_W3/turb_s4.0_blind/{args.seed}",
    "W3-aware":    f"results_wind_W3/turb_s4.0_aware/{args.seed}",
}

HIT_THRESH = args.hit_threshold

print("=" * 80)
print("MC-PILOT Wind Experiment Results Analysis")
print(f"Hit threshold: {HIT_THRESH*100:.0f} cm")
print("=" * 80)

results_table = []

for name, dirpath in result_dirs.items():
    log_file = os.path.join(dirpath, "log.pkl")
    cfg_file = os.path.join(dirpath, "config_log.pkl")

    if not os.path.exists(log_file):
        results_table.append({
            "name": name, "status": "NOT RUN",
            "trials": 0, "hits": 0, "hit_rate": 0,
            "mean_err": float("nan"), "std_err": float("nan"),
            "final_cost": float("nan"),
        })
        continue

    with open(log_file, "rb") as f:
        log = pkl.load(f)
    with open(cfg_file, "rb") as f:
        cfg = pkl.load(f)

    nexp = cfg.get("Nexp", 5)
    num_trials = cfg.get("num_trials", 15)

    # Extract landing errors from noiseless state history
    # Skip exploration throws (first Nexp entries)
    errors = []
    noiseless = log.get("noiseless_states_history", [])

    for trial_idx in range(nexp, len(noiseless)):
        traj = noiseless[trial_idx]
        # Final state: landing position (x, y) vs target (Px, Py)
        landing_xy = traj[-1, 0:2]
        target_xy  = traj[-1, 6:8]
        err = np.linalg.norm(landing_xy - target_xy)
        errors.append(err)

    errors = np.array(errors)
    hits = np.sum(errors < HIT_THRESH)
    hit_rate = hits / len(errors) * 100 if len(errors) > 0 else 0

    # Final cost from last trial
    cost_list = log.get("cost_trial_list", [])
    final_cost = cost_list[-1][-1] if cost_list else float("nan")

    results_table.append({
        "name": name,
        "status": "OK",
        "trials": len(errors),
        "hits": int(hits),
        "hit_rate": hit_rate,
        "mean_err": np.mean(errors) * 100,  # cm
        "std_err": np.std(errors) * 100,     # cm
        "final_cost": float(final_cost),
    })

# Print results table
print(f"\n{'Config':<15} {'Status':<8} {'Trials':>6} {'Hits':>5} "
      f"{'Rate':>6} {'Mean Err':>10} {'Std Err':>9} {'Final Cost':>11}")
print("-" * 80)

for r in results_table:
    if r["status"] == "NOT RUN":
        print(f"{r['name']:<15} {'—':<8} {'—':>6} {'—':>5} "
              f"{'—':>6} {'—':>10} {'—':>9} {'—':>11}")
    else:
        print(f"{r['name']:<15} {r['status']:<8} {r['trials']:>6} "
              f"{r['hits']:>5} {r['hit_rate']:>5.0f}% "
              f"{r['mean_err']:>8.1f}cm {r['std_err']:>7.1f}cm "
              f"{r['final_cost']:>10.4f}")

# Key comparisons
print(f"\n{'=' * 80}")
print("KEY COMPARISONS")
print(f"{'=' * 80}")

def get_result(name):
    for r in results_table:
        if r["name"] == name:
            return r
    return None

# W1: wind speed vs accuracy
print("\n1. Constant Wind — Accuracy Degradation")
for name in ["W1-calm", "W1-light", "W1-moderate", "W1-strong"]:
    r = get_result(name)
    if r and r["status"] == "OK":
        print(f"   {name:<15}: {r['hit_rate']:.0f}% hits, "
              f"mean err = {r['mean_err']:.1f}cm")

# W1: blind vs aware at moderate wind
print("\n2. Constant Wind — Blind vs Aware (0.7 m/s)")
for name in ["W1-moderate", "W1-aware"]:
    r = get_result(name)
    if r and r["status"] == "OK":
        print(f"   {name:<15}: {r['hit_rate']:.0f}% hits, "
              f"mean err = {r['mean_err']:.1f}cm")

# W2: gusts blind vs aware
print("\n3. Random Gusts — Blind vs Aware")
for name in ["W2-blind", "W2-aware"]:
    r = get_result(name)
    if r and r["status"] == "OK":
        print(f"   {name:<15}: {r['hit_rate']:.0f}% hits, "
              f"mean err = {r['mean_err']:.1f}cm")

# W3: turbulence blind vs aware
print("\n4. Turbulence — Blind vs Aware")
for name in ["W3-blind", "W3-aware"]:
    r = get_result(name)
    if r and r["status"] == "OK":
        print(f"   {name:<15}: {r['hit_rate']:.0f}% hits, "
              f"mean err = {r['mean_err']:.1f}cm")

print(f"\n{'=' * 80}")
print("CONCLUSION")
print(f"{'=' * 80}")
print("Check if wind-aware GP+policy consistently outperforms blind baseline.")
print("If W1-moderate (blind) matches W1-calm, GP absorbs constant wind as bias.")
print("If W2-blind degrades but W2-aware recovers, explicit wind input is required")
print("for time-varying wind.")
