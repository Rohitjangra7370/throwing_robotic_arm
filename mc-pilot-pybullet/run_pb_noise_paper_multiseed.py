"""
Run a paper-style multi-seed noise study without modifying the original scripts.

This wrapper keeps the same underlying training/evaluation code but provides:
  - fixed default configs for the current paper study
  - resumable execution via --skip_existing
  - optional filtering to a subset of conditions
  - fresh output roots so previous results stay untouched

Examples:
  python run_pb_noise_paper_multiseed.py --run --seeds 1,2
  python run_pb_noise_paper_multiseed.py --run --seeds 1,2,3 --only slip:0.10,saltpepper:0.05
  python run_pb_noise_paper_multiseed.py --summarize_only
"""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
from pathlib import Path


DEFAULT_STUDY_ROOT = "results_mc_pilot_pb_noise_multiseed_study"
DEFAULT_SWEEP_ROOT = "results_mc_pilot_pb_noise_multiseed_sweep"
DEFAULT_REPORT_ROOT = "results_mc_pilot_pb_noise_multiseed_report"


def parse_csv(raw: str, cast=str):
    values = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.append(cast(token))
    return values


def parse_only_specs(raw: str):
    if not raw.strip():
        return set()
    specs = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        noise_type, value = token.split(":")
        specs.add((noise_type.strip(), round(float(value), 6)))
    return specs


def build_configs():
    configs = []
    for alpha in [0.05, 0.10, 0.15, 0.20]:
        configs.append(
            {
                "noise_type": "slip",
                "alpha": alpha,
                "sigma": 0.04,
                "p_spike": None,
                "spike_scale": None,
                "sp_sigma": None,
                "noise_level": alpha,
            }
        )
    for p_spike in [0.05, 0.10, 0.15]:
        configs.append(
            {
                "noise_type": "saltpepper",
                "alpha": None,
                "sigma": None,
                "p_spike": p_spike,
                "spike_scale": 0.30,
                "sp_sigma": 0.00,
                "noise_level": p_spike,
            }
        )
    return configs


def result_dir_for_cfg(study_root: Path, cfg: dict, aware: bool, seed: int) -> Path:
    aware_tag = "aware" if aware else "naive"
    if cfg["noise_type"] == "slip":
        tag = f"slip_alpha_{cfg['alpha']:.3f}_sigma_{cfg['sigma']:.3f}".replace(".", "p")
    else:
        tag = (
            f"saltpepper_p_{cfg['p_spike']:.3f}_scale_{cfg['spike_scale']:.3f}_sigma_{cfg['sp_sigma']:.3f}"
        ).replace(".", "p")
    return study_root / f"{tag}_{aware_tag}" / str(seed)


def run_job(project_dir: Path, study_root: Path, cfg: dict, aware: bool, seed: int, skip_existing: bool):
    run_dir = result_dir_for_cfg(study_root, cfg, aware, seed)
    if skip_existing and (run_dir / "log.pkl").exists() and (run_dir / "config_log.pkl").exists():
        print(f"[SKIP] {cfg['noise_type']} level={cfg['noise_level']} aware={aware} seed={seed}")
        return True

    cmd = [
        sys.executable,
        "test_mc_pilot_pb_noise_study.py",
        "-seed",
        str(seed),
        "-num_trials",
        "4",
        "-noise_type",
        cfg["noise_type"],
        "-noise_aware",
        "1" if aware else "0",
        "-backend",
        "numpy",
        "-Nexp",
        "6",
        "-Nopt",
        "500",
        "-M",
        "400",
        "-Nb",
        "250",
        "-uM",
        "3.0",
        "-uMin",
        "1.4",
        "-Ts",
        "0.02",
        "-T",
        "0.60",
        "-lc",
        "0.5",
        "-lm",
        "0.6",
        "-lM",
        "1.1",
        "-results_root",
        str(study_root),
    ]
    if cfg["noise_type"] == "slip":
        cmd += ["-alpha", str(cfg["alpha"]), "-sigma", str(cfg["sigma"])]
    else:
        cmd += [
            "-p_spike",
            str(cfg["p_spike"]),
            "-spike_scale",
            str(cfg["spike_scale"]),
            "-sp_sigma",
            str(cfg["sp_sigma"]),
        ]

    print("\n[RUN]", " ".join(cmd))
    result = subprocess.run(cmd, cwd=project_dir, capture_output=True, text=True)
    if result.stdout:
        tail = "\n".join(result.stdout.splitlines()[-8:])
        if tail:
            print(tail)
    if result.returncode != 0:
        print("[ERROR] job failed")
        if result.stderr:
            print(result.stderr)
        return False
    return True


def summarize(project_dir: Path, study_root: Path, report_root: Path):
    cmd = [
        sys.executable,
        "summarize_pb_noise_results.py",
        "--study_results_root",
        str(study_root),
        "--output_dir",
        str(report_root),
    ]
    print("\n[SUMMARIZE]", " ".join(cmd))
    result = subprocess.run(cmd, cwd=project_dir, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.returncode != 0 and result.stderr:
        print(result.stderr)
    return result.returncode == 0


def inspect(report_root: Path):
    summary_path = report_root / "noise_summary.csv"
    if not summary_path.exists():
        print(f"[WARN] Missing summary at {summary_path}")
        return
    print(f"\nSummary written to: {summary_path}")


def main():
    parser = argparse.ArgumentParser("Paper-style multi-seed runner for mc-pilot-pybullet noise study")
    parser.add_argument("--project_dir", type=str, default=".")
    parser.add_argument("--study_root", type=str, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--sweep_root", type=str, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--report_root", type=str, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--seeds", type=str, default="1,2,3")
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="optional subset filter like slip:0.10,saltpepper:0.05",
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summarize_only", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    study_root = (project_dir / args.study_root).resolve()
    report_root = (project_dir / args.report_root).resolve()
    study_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    seeds = parse_csv(args.seeds, int)
    only_specs = parse_only_specs(args.only)
    configs = build_configs()
    if only_specs:
        configs = [
            cfg
            for cfg in configs
            if (cfg["noise_type"], round(float(cfg["noise_level"]), 6)) in only_specs
        ]

    if args.run and not args.summarize_only:
        failures = []
        for cfg in configs:
            for aware in [True, False]:
                for seed in seeds:
                    ok = run_job(project_dir, study_root, cfg, aware, seed, args.skip_existing)
                    if not ok:
                        failures.append((cfg["noise_type"], cfg["noise_level"], aware, seed))
        if failures:
            print("\nFailures:")
            for failure in failures:
                print(" ", failure)

    summarize(project_dir, study_root, report_root)
    inspect(report_root)


if __name__ == "__main__":
    main()
