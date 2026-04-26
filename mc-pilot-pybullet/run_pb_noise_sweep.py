"""
Run noise sweeps for mc-pilot-pybullet and generate paper-ready tables/plots.

This script can:
1) Execute multiple runs of test_mc_pilot_pb_noise_study.py
2) Parse each run's log.pkl/config_log.pkl
3) Compute hit-rate and error statistics
4) Save CSV + Markdown tables + PNG plots

Examples:
  python run_pb_noise_sweep.py --run --noise_types slip,saltpepper --seeds 1
  python run_pb_noise_sweep.py --run --noise_types saltpepper --sp_probs 0.05,0.1,0.2 --seeds 1,2
  python run_pb_noise_sweep.py --analyze_only
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import statistics
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_num_list(raw: str, cast=float):
    values = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.append(cast(token))
    return values


def build_run_configs(args):
    configs = []
    noise_types = [n.strip() for n in args.noise_types.split(",") if n.strip()]

    if "slip" in noise_types:
        alphas = parse_num_list(args.slip_alphas, float)
        sigmas = parse_num_list(args.slip_sigmas, float)
        for alpha in alphas:
            for sigma in sigmas:
                configs.append(
                    {
                        "noise_type": "slip",
                        "alpha": alpha,
                        "sigma": sigma,
                        "p_spike": None,
                        "spike_scale": None,
                        "sp_sigma": None,
                        "noise_level": alpha,
                        "noise_level_name": "alpha",
                    }
                )

    if "saltpepper" in noise_types:
        probs = parse_num_list(args.sp_probs, float)
        scales = parse_num_list(args.sp_scales, float)
        sigmas = parse_num_list(args.sp_sigmas, float)
        for p_spike in probs:
            for spike_scale in scales:
                for sp_sigma in sigmas:
                    configs.append(
                        {
                            "noise_type": "saltpepper",
                            "alpha": None,
                            "sigma": None,
                            "p_spike": p_spike,
                            "spike_scale": spike_scale,
                            "sp_sigma": sp_sigma,
                            "noise_level": p_spike,
                            "noise_level_name": "p_spike",
                        }
                    )

    return configs


def run_one_job(args, cfg, seed, aware):
    cmd = [
        args.python_exe,
        args.study_script,
        "-seed",
        str(seed),
        "-num_trials",
        str(args.num_trials),
        "-noise_type",
        cfg["noise_type"],
        "-noise_aware",
        "1" if aware else "0",
        "-backend",
        str(args.backend),
        "-Nexp",
        str(args.Nexp),
        "-Nopt",
        str(args.Nopt),
        "-M",
        str(args.M),
        "-Nb",
        str(args.Nb),
        "-uM",
        str(args.uM),
        "-uMin",
        str(args.uMin),
        "-Ts",
        str(args.Ts),
        "-T",
        str(args.T),
        "-lc",
        str(args.lc),
        "-lm",
        str(args.lm),
        "-lM",
        str(args.lM),
        "-results_root",
        str(args.study_results_root),
    ]

    if cfg["noise_type"] == "slip":
        cmd += ["-alpha", str(cfg["alpha"]), "-sigma", str(cfg["sigma"])]
    elif cfg["noise_type"] == "saltpepper":
        cmd += [
            "-p_spike",
            str(cfg["p_spike"]),
            "-spike_scale",
            str(cfg["spike_scale"]),
            "-sp_sigma",
            str(cfg["sp_sigma"]),
        ]

    print("\n[RUN]", " ".join(cmd))
    result = subprocess.run(cmd, cwd=args.project_dir, capture_output=True, text=True)
    tail = "\n".join(result.stdout.splitlines()[-8:])
    if tail:
        print(tail)
    if result.returncode != 0:
        print("[ERROR] job failed")
        print(result.stderr)
    return result.returncode == 0


def find_run_dir(study_results_root: Path, cfg: dict, aware: bool, seed: int) -> Path:
    aware_tag = "aware" if aware else "naive"
    for d in study_results_root.glob(f"*_{aware_tag}"):
        cfg_path = d / str(seed) / "config_log.pkl"
        if not cfg_path.exists():
            continue
        try:
            with open(cfg_path, "rb") as f:
                c = pickle.load(f)
        except Exception:
            continue

        if c.get("noise_type") != cfg["noise_type"]:
            continue

        if cfg["noise_type"] == "slip":
            if abs(float(c.get("alpha", -999)) - float(cfg["alpha"])) > 1e-12:
                continue
            if abs(float(c.get("sigma", -999)) - float(cfg["sigma"])) > 1e-12:
                continue
        else:
            if abs(float(c.get("p_spike", -999)) - float(cfg["p_spike"])) > 1e-12:
                continue
            if abs(float(c.get("spike_scale", -999)) - float(cfg["spike_scale"])) > 1e-12:
                continue
            if abs(float(c.get("sp_sigma", -999)) - float(cfg["sp_sigma"])) > 1e-12:
                continue

        return d / str(seed)

    raise FileNotFoundError("Could not locate run directory for config")


def compute_metrics(run_dir: Path, hit_threshold: float):
    with open(run_dir / "config_log.pkl", "rb") as f:
        config = pickle.load(f)
    with open(run_dir / "log.pkl", "rb") as f:
        log = pickle.load(f)

    trajectories = log.get("noiseless_states_history", [])
    nexp = int(config.get("Nexp", 0))
    policy_trajectories = trajectories[nexp:]

    def errors_from_trajectories(trajs):
        errs = []
        for traj in trajs:
            if traj is None or len(traj) == 0:
                continue
            landing = np.array(traj[-1][0:2], dtype=float)
            target = np.array(traj[-1][6:8], dtype=float)
            errs.append(float(np.linalg.norm(landing - target)))
        return errs

    all_errors = errors_from_trajectories(trajectories)
    policy_errors = errors_from_trajectories(policy_trajectories)

    cost_trial_list = log.get("cost_trial_list", [])
    final_cost = np.nan
    if cost_trial_list is not None and len(cost_trial_list) > 0:
        last_cost_series = np.asarray(cost_trial_list[-1]).reshape(-1)
        if last_cost_series.size > 0:
            final_cost = float(last_cost_series[-1])

    def summarize(errors):
        if not errors:
            return {
                "n": 0,
                "hits": 0,
                "hit_rate": np.nan,
                "mean": np.nan,
                "median": np.nan,
                "p90": np.nan,
            }
        errs = np.array(errors, dtype=float)
        hits = int((errs <= hit_threshold).sum())
        return {
            "n": int(len(errs)),
            "hits": hits,
            "hit_rate": float(100.0 * hits / len(errs)),
            "mean": float(np.mean(errs)),
            "median": float(np.median(errs)),
            "p90": float(np.quantile(errs, 0.90)),
        }

    s_all = summarize(all_errors)
    s_pol = summarize(policy_errors)

    row = {
        "run_dir": str(run_dir),
        "backend": config.get("backend", config.get("simulator", "unknown")),
        "noise_type": config.get("noise_type"),
        "noise_aware": bool(config.get("noise_aware", False)),
        "seed": int(config.get("seed", -1)),
        "noise_level": np.nan,
        "noise_level_name": "",
        "alpha": config.get("alpha", np.nan),
        "sigma": config.get("sigma", np.nan),
        "p_spike": config.get("p_spike", np.nan),
        "spike_scale": config.get("spike_scale", np.nan),
        "sp_sigma": config.get("sp_sigma", np.nan),
        "final_cost": final_cost,
        "all_throws": s_all["n"],
        "all_hits": s_all["hits"],
        "all_hit_rate_pct": s_all["hit_rate"],
        "all_mean_error_m": s_all["mean"],
        "policy_throws": s_pol["n"],
        "policy_hits": s_pol["hits"],
        "policy_hit_rate_pct": s_pol["hit_rate"],
        "policy_mean_error_m": s_pol["mean"],
        "policy_median_error_m": s_pol["median"],
        "policy_p90_error_m": s_pol["p90"],
    }

    if row["noise_type"] == "slip":
        row["noise_level"] = float(row["alpha"])
        row["noise_level_name"] = "alpha"
    elif row["noise_type"] == "saltpepper":
        row["noise_level"] = float(row["p_spike"])
        row["noise_level_name"] = "p_spike"

    return row


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict]):
    grouped = {}
    for r in rows:
        key = (
            r["noise_type"],
            r["noise_level_name"],
            float(r["noise_level"]),
            bool(r["noise_aware"]),
            float(r.get("sigma", np.nan)) if r["noise_type"] == "slip" else float(r.get("spike_scale", np.nan)),
        )
        grouped.setdefault(key, []).append(r)

    out = []
    for key, group in grouped.items():
        noise_type, level_name, level, aware, sec_param = key
        hit_rates = [g["policy_hit_rate_pct"] for g in group if not np.isnan(g["policy_hit_rate_pct"])]
        mean_errs = [g["policy_mean_error_m"] for g in group if not np.isnan(g["policy_mean_error_m"])]
        final_costs = [g["final_cost"] for g in group if not np.isnan(g["final_cost"])]

        row = {
            "noise_type": noise_type,
            "noise_level_name": level_name,
            "noise_level": level,
            "noise_aware": aware,
            "num_seeds": len(group),
            "secondary_param": sec_param,
            "policy_hit_rate_mean_pct": float(np.mean(hit_rates)) if hit_rates else np.nan,
            "policy_hit_rate_std_pct": float(np.std(hit_rates)) if len(hit_rates) > 1 else 0.0,
            "policy_mean_error_mean_m": float(np.mean(mean_errs)) if mean_errs else np.nan,
            "policy_mean_error_std_m": float(np.std(mean_errs)) if len(mean_errs) > 1 else 0.0,
            "final_cost_mean": float(np.mean(final_costs)) if final_costs else np.nan,
            "final_cost_std": float(np.std(final_costs)) if len(final_costs) > 1 else 0.0,
        }
        out.append(row)

    out.sort(key=lambda x: (x["noise_type"], x["secondary_param"], x["noise_level"], x["noise_aware"]))
    return out


def write_markdown_table(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("No data available.\n", encoding="utf-8")
        return

    header = (
        "| Noise Type | Level Name | Level | Aware | Seeds | Secondary Param | "
        "Hit Rate % (mean±std) | Mean Error m (mean±std) | Final Cost (mean±std) |\n"
    )
    sep = "|---|---|---:|:---:|---:|---:|---:|---:|---:|\n"
    lines = [header, sep]

    for r in rows:
        lines.append(
            "| {noise_type} | {noise_level_name} | {noise_level:.3f} | {aware} | {num_seeds} | {secondary_param:.3f} | "
            "{hrm:.2f} +- {hrs:.2f} | {emm:.4f} +- {ems:.4f} | {fcm:.4f} +- {fcs:.4f} |\n".format(
                noise_type=r["noise_type"],
                noise_level_name=r["noise_level_name"],
                noise_level=r["noise_level"],
                aware="Y" if r["noise_aware"] else "N",
                num_seeds=r["num_seeds"],
                secondary_param=r["secondary_param"],
                hrm=r["policy_hit_rate_mean_pct"],
                hrs=r["policy_hit_rate_std_pct"],
                emm=r["policy_mean_error_mean_m"],
                ems=r["policy_mean_error_std_m"],
                fcm=r["final_cost_mean"],
                fcs=r["final_cost_std"],
            )
        )

    path.write_text("".join(lines), encoding="utf-8")


def make_plots(agg_rows: list[dict], output_dir: Path):
    if not agg_rows:
        return

    for noise_type in sorted({r["noise_type"] for r in agg_rows}):
        rows = [r for r in agg_rows if r["noise_type"] == noise_type]
        levels = sorted({r["noise_level"] for r in rows})

        aware_map = {True: {}, False: {}}
        for r in rows:
            aware_map[bool(r["noise_aware"])][r["noise_level"]] = r

        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        for aware in [True, False]:
            label = "aware" if aware else "naive"
            color = "tab:blue" if aware else "tab:orange"
            x = []
            y_hit = []
            y_err = []
            y_cost = []
            for lv in levels:
                row = aware_map[aware].get(lv)
                if row is None:
                    continue
                x.append(lv)
                y_hit.append(row["policy_hit_rate_mean_pct"])
                y_err.append(row["policy_mean_error_mean_m"])
                y_cost.append(row["final_cost_mean"])

            if x:
                axs[0].plot(x, y_hit, marker="o", label=label, color=color)
                axs[1].plot(x, y_err, marker="o", label=label, color=color)
                axs[2].plot(x, y_cost, marker="o", label=label, color=color)

        axs[0].set_title(f"{noise_type}: policy hit rate")
        axs[0].set_xlabel("noise level")
        axs[0].set_ylabel("hit rate (%)")
        axs[0].grid(True, alpha=0.3)

        axs[1].set_title(f"{noise_type}: policy mean error")
        axs[1].set_xlabel("noise level")
        axs[1].set_ylabel("error (m)")
        axs[1].grid(True, alpha=0.3)

        axs[2].set_title(f"{noise_type}: final cost")
        axs[2].set_xlabel("noise level")
        axs[2].set_ylabel("cost")
        axs[2].grid(True, alpha=0.3)

        for ax in axs:
            ax.legend()

        fig.tight_layout()
        out_path = output_dir / f"noise_sweep_{noise_type}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser("Run and analyze PyBullet noise sweeps")
    parser.add_argument("--project_dir", type=str, default=".")
    parser.add_argument("--python_exe", type=str, default=sys.executable)
    parser.add_argument("--study_script", type=str, default="test_mc_pilot_pb_noise_study.py")
    parser.add_argument("--study_results_root", type=str, default="results_mc_pilot_pb_noise_study")
    parser.add_argument("--output_dir", type=str, default="results_mc_pilot_pb_noise_sweep")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "pybullet", "numpy"])

    parser.add_argument("--run", action="store_true", help="execute experiments before analysis")
    parser.add_argument("--analyze_only", action="store_true", help="skip execution and analyze existing runs")

    parser.add_argument("--noise_types", type=str, default="slip,saltpepper")
    parser.add_argument("--aware_modes", type=str, default="aware,naive")
    parser.add_argument("--seeds", type=str, default="1")

    parser.add_argument("--slip_alphas", type=str, default="0.10,0.20,0.30")
    parser.add_argument("--slip_sigmas", type=str, default="0.04")

    parser.add_argument("--sp_probs", type=str, default="0.05,0.10,0.20")
    parser.add_argument("--sp_scales", type=str, default="0.30")
    parser.add_argument("--sp_sigmas", type=str, default="0.00")

    parser.add_argument("--num_trials", type=int, default=6)
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

    args.project_dir = str(Path(args.project_dir).resolve())
    args.study_results_root = Path(args.project_dir) / args.study_results_root
    args.output_dir = Path(args.project_dir) / args.output_dir
    args.study_script = str((Path(args.project_dir) / args.study_script).resolve())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.study_results_root.mkdir(parents=True, exist_ok=True)

    seeds = parse_num_list(args.seeds, int)
    aware_modes = [m.strip().lower() for m in args.aware_modes.split(",") if m.strip()]
    run_aware = [m == "aware" for m in aware_modes]

    configs = build_run_configs(args)

    if args.run and not args.analyze_only:
        for cfg in configs:
            for aware in run_aware:
                for seed in seeds:
                    run_one_job(args, cfg, seed, aware)

    raw_rows = []
    for cfg in configs:
        for aware in run_aware:
            for seed in seeds:
                try:
                    run_dir = find_run_dir(args.study_results_root, cfg, aware, seed)
                    raw_rows.append(compute_metrics(run_dir, args.hit_threshold))
                except FileNotFoundError:
                    print(f"[WARN] Missing run for cfg={cfg} aware={aware} seed={seed}")

    raw_csv = args.output_dir / "noise_sweep_raw_runs.csv"
    write_csv(raw_csv, raw_rows)

    agg_rows = aggregate_rows(raw_rows)
    agg_csv = args.output_dir / "noise_sweep_summary.csv"
    write_csv(agg_csv, agg_rows)

    md_table = args.output_dir / "noise_sweep_summary.md"
    write_markdown_table(md_table, agg_rows)

    make_plots(agg_rows, args.output_dir)

    print("\nNoise sweep complete")
    print(f"  Raw runs CSV     : {raw_csv}")
    print(f"  Summary CSV      : {agg_csv}")
    print(f"  Summary table MD : {md_table}")
    print(f"  Plots            : {args.output_dir}")


if __name__ == "__main__":
    main()
