"""
Collect a paper-friendly salt-and-pepper noise sweep for mc-pilot-pybullet.

Examples:
  python collect_pb_saltpepper_sweep.py --run
  python collect_pb_saltpepper_sweep.py --run --seeds 1,2 --sp_probs 0.05,0.10,0.15,0.20
  python collect_pb_saltpepper_sweep.py --analyze_only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser("Collect salt-and-pepper sweep results")
    parser.add_argument("--project_dir", type=str, default=".")
    parser.add_argument("--python_exe", type=str, default=sys.executable)
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "pybullet", "numpy"])
    parser.add_argument("--run", action="store_true", help="execute experiments before analysis")
    parser.add_argument("--analyze_only", action="store_true", help="skip execution and summarize existing runs")
    parser.add_argument("--aware_modes", type=str, default="aware,naive")
    parser.add_argument("--seeds", type=str, default="1,2,3")
    parser.add_argument("--sp_probs", type=str, default="0.05,0.10,0.15,0.20")
    parser.add_argument("--sp_scales", type=str, default="0.30")
    parser.add_argument("--sp_sigmas", type=str, default="0.00")
    parser.add_argument("--study_results_root", type=str, default="results_mc_pilot_pb_noise_study")
    parser.add_argument("--output_dir", type=str, default="results_mc_pilot_pb_saltpepper_sweep")
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--Nexp", type=int, default=8)
    parser.add_argument("--Nopt", type=int, default=500)
    parser.add_argument("--M", type=int, default=400)
    parser.add_argument("--Nb", type=int, default=250)
    parser.add_argument("--uM", type=float, default=3.0)
    parser.add_argument("--uMin", type=float, default=1.4)
    parser.add_argument("--Ts", type=float, default=0.02)
    parser.add_argument("--T", type=float, default=0.60)
    parser.add_argument("--lc", type=float, default=0.5)
    parser.add_argument("--lm", type=float, default=0.6)
    parser.add_argument("--lM", type=float, default=1.1)
    parser.add_argument("--hit_threshold", type=float, default=0.10)
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    cmd = [
        args.python_exe,
        str((project_dir / "run_pb_noise_sweep.py").resolve()),
        "--project_dir",
        str(project_dir),
        "--backend",
        args.backend,
        "--noise_types",
        "saltpepper",
        "--aware_modes",
        args.aware_modes,
        "--seeds",
        args.seeds,
        "--sp_probs",
        args.sp_probs,
        "--sp_scales",
        args.sp_scales,
        "--sp_sigmas",
        args.sp_sigmas,
        "--study_results_root",
        args.study_results_root,
        "--output_dir",
        args.output_dir,
        "--num_trials",
        str(args.num_trials),
        "--Nexp",
        str(args.Nexp),
        "--Nopt",
        str(args.Nopt),
        "--M",
        str(args.M),
        "--Nb",
        str(args.Nb),
        "--uM",
        str(args.uM),
        "--uMin",
        str(args.uMin),
        "--Ts",
        str(args.Ts),
        "--T",
        str(args.T),
        "--lc",
        str(args.lc),
        "--lm",
        str(args.lm),
        "--lM",
        str(args.lM),
        "--hit_threshold",
        str(args.hit_threshold),
    ]

    if args.run:
        cmd.append("--run")
    if args.analyze_only:
        cmd.append("--analyze_only")

    subprocess.run(cmd, check=True, cwd=project_dir)


if __name__ == "__main__":
    main()
