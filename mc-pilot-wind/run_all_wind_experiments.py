#!/usr/bin/env python3
"""
Run all wind experiments sequentially.

Usage:
  python run_all_wind_experiments.py
  python run_all_wind_experiments.py --seed 1 --num_trials 15
  python run_all_wind_experiments.py --quick   # 5 trials for fast testing
"""
import argparse
import subprocess
import sys
import time

parser = argparse.ArgumentParser("Run all MC-PILOT wind experiments")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--num_trials", type=int, default=15)
parser.add_argument("--quick", action="store_true", help="5 trials for quick test")
args = parser.parse_args()

if args.quick:
    args.num_trials = 5

seed = args.seed
nt = args.num_trials

# All experiment configurations
experiments = [
    # W1: Constant wind
    ("W1-calm",     "test_wind_W1.py", f"-seed {seed} -num_trials {nt} -wind_speed 0.0"),
    ("W1-light",    "test_wind_W1.py", f"-seed {seed} -num_trials {nt} -wind_speed 2.5"),
    ("W1-moderate", "test_wind_W1.py", f"-seed {seed} -num_trials {nt} -wind_speed 5.0"),
    ("W1-strong",   "test_wind_W1.py", f"-seed {seed} -num_trials {nt} -wind_speed 8.0"),
    ("W1-aware",    "test_wind_W1.py", f"-seed {seed} -num_trials {nt} -wind_speed 5.0 -wind_aware 1"),
    # W2: Gusts
    ("W2-blind",    "test_wind_W2.py", f"-seed {seed} -num_trials {nt} -w_max 4.0"),
    ("W2-aware",    "test_wind_W2.py", f"-seed {seed} -num_trials {nt} -w_max 4.0 -wind_aware 1"),
    # W3: Turbulence
    ("W3-blind",    "test_wind_W3.py", f"-seed {seed} -num_trials {nt} -w_mean_x 2.5 -sigma 4.0"),
    ("W3-aware",    "test_wind_W3.py", f"-seed {seed} -num_trials {nt} -w_mean_x 2.5 -sigma 4.0 -wind_aware 1"),
]

print(f"=" * 70)
print(f"MC-PILOT Wind Experiments: {len(experiments)} configs, "
      f"seed={seed}, trials={nt}")
print(f"=" * 70)

results = []
total_start = time.time()

for name, script, cli_args in experiments:
    print(f"\n{'─' * 60}")
    print(f"Starting: {name}")
    print(f"Command:  python {script} {cli_args}")
    print(f"{'─' * 60}")

    t0 = time.time()
    cmd = [sys.executable, script] + cli_args.split() + ["--resume"]
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    results.append((name, status, elapsed))
    print(f"\n{name}: {status} ({elapsed:.0f}s)")

total_elapsed = time.time() - total_start

print(f"\n{'=' * 70}")
print(f"ALL EXPERIMENTS COMPLETE — Total time: {total_elapsed/60:.1f} min")
print(f"{'=' * 70}")
print(f"\n{'Name':<15} {'Status':<20} {'Time':>8}")
print(f"{'-'*15} {'-'*20} {'-'*8}")
for name, status, elapsed in results:
    print(f"{name:<15} {status:<20} {elapsed:>7.0f}s")

print(f"\nRun 'python analyze_wind_results.py' to generate comparison report.")
