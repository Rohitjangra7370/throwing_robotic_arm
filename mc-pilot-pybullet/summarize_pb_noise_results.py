"""
Build paper-oriented tables and plots from mc-pilot-pybullet noise-study runs.

Outputs:
  - CSV tables for raw runs and aggregated summaries
  - Markdown and LaTeX tables for the paper
  - Error-bar plots for quantitative trends
  - Landing scatter plots for qualitative comparison

Examples:
  python summarize_pb_noise_results.py
  python summarize_pb_noise_results.py --study_results_root results_mc_pilot_pb_noise_study
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def format_num(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    return f"{value:.{digits}f}"


def infer_secondary_label(noise_type: str) -> str:
    return "sigma" if noise_type == "slip" else "spike_scale"


def load_run(run_dir: Path):
    with open(run_dir / "config_log.pkl", "rb") as f:
        config = pickle.load(f)
    with open(run_dir / "log.pkl", "rb") as f:
        log = pickle.load(f)
    return config, log


def collect_run_dirs(study_results_root: Path):
    run_dirs = []
    for config_path in sorted(study_results_root.glob("*/*/config_log.pkl")):
        run_dir = config_path.parent
        if (run_dir / "log.pkl").exists():
            run_dirs.append(run_dir)
    return run_dirs


def get_policy_landings(config: dict, log: dict):
    trajectories = log.get("noiseless_states_history", [])
    nexp = int(config.get("Nexp", 0))
    policy_trajectories = trajectories[nexp:]
    landings = []
    targets = []
    for traj in policy_trajectories:
        if traj is None or len(traj) == 0:
            continue
        last = np.asarray(traj[-1], dtype=float)
        landings.append(last[0:2])
        targets.append(last[6:8])
    if not landings:
        return np.empty((0, 2)), np.empty((0, 2))
    return np.asarray(landings), np.asarray(targets)


def compute_run_row(run_dir: Path, hit_threshold: float):
    config, log = load_run(run_dir)
    trajectories = log.get("noiseless_states_history", [])
    nexp = int(config.get("Nexp", 0))
    policy_trajectories = trajectories[nexp:]

    def errors_from_trajectories(trajs):
        errors = []
        for traj in trajs:
            if traj is None or len(traj) == 0:
                continue
            last = np.asarray(traj[-1], dtype=float)
            landing = last[0:2]
            target = last[6:8]
            errors.append(float(np.linalg.norm(landing - target)))
        return errors

    all_errors = np.asarray(errors_from_trajectories(trajectories), dtype=float)
    policy_errors = np.asarray(errors_from_trajectories(policy_trajectories), dtype=float)

    cost_trial_list = log.get("cost_trial_list", [])
    final_cost = np.nan
    if cost_trial_list is not None and len(cost_trial_list) > 0:
        last_cost_series = np.asarray(cost_trial_list[-1]).reshape(-1)
        if last_cost_series.size > 0:
            final_cost = float(last_cost_series[-1])

    def summarize(errors: np.ndarray):
        if errors.size == 0:
            return 0, 0, np.nan, np.nan, np.nan, np.nan
        hits = int((errors <= hit_threshold).sum())
        return (
            int(errors.size),
            hits,
            float(100.0 * hits / errors.size),
            float(errors.mean()),
            float(np.median(errors)),
            float(np.quantile(errors, 0.90)),
        )

    all_n, all_hits, all_hit_rate, all_mean, _, _ = summarize(all_errors)
    pol_n, pol_hits, pol_hit_rate, pol_mean, pol_median, pol_p90 = summarize(policy_errors)

    noise_type = config.get("noise_type", "unknown")
    secondary_param = config.get("sigma", np.nan) if noise_type == "slip" else config.get("spike_scale", np.nan)
    noise_level = config.get("alpha", np.nan) if noise_type == "slip" else config.get("p_spike", np.nan)

    return {
        "run_dir": str(run_dir),
        "study_root": str(run_dir.parents[1]),
        "backend": config.get("backend", config.get("simulator", "unknown")),
        "noise_type": noise_type,
        "seed": int(config.get("seed", -1)),
        "noise_aware": bool(config.get("noise_aware", False)),
        "noise_level_name": "alpha" if noise_type == "slip" else "p_spike",
        "noise_level": float(noise_level),
        "secondary_param_name": infer_secondary_label(noise_type),
        "secondary_param": float(secondary_param),
        "sigma": float(config.get("sigma", np.nan)),
        "alpha": float(config.get("alpha", np.nan)),
        "p_spike": float(config.get("p_spike", np.nan)),
        "spike_scale": float(config.get("spike_scale", np.nan)),
        "sp_sigma": float(config.get("sp_sigma", np.nan)),
        "Nexp": int(config.get("Nexp", 0)),
        "num_trials": int(config.get("num_trials", 0)),
        "all_throws": all_n,
        "all_hits": all_hits,
        "all_hit_rate_pct": all_hit_rate,
        "all_mean_error_m": all_mean,
        "policy_throws": pol_n,
        "policy_hits": pol_hits,
        "policy_hit_rate_pct": pol_hit_rate,
        "policy_mean_error_m": pol_mean,
        "policy_median_error_m": pol_median,
        "policy_p90_error_m": pol_p90,
        "final_cost": final_cost,
    }


def dedupe_run_rows(rows: list[dict]):
    deduped = {}
    for row in rows:
        key = (
            row["noise_type"],
            row["noise_aware"],
            row["seed"],
            row["noise_level"],
            row["secondary_param"],
        )
        deduped[key] = row
    return list(deduped.values())


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict]):
    grouped = {}
    for row in rows:
        key = (
            row["noise_type"],
            row["noise_level_name"],
            row["noise_level"],
            row["secondary_param_name"],
            row["secondary_param"],
            row["noise_aware"],
        )
        grouped.setdefault(key, []).append(row)

    out = []
    for key, group in grouped.items():
        noise_type, level_name, noise_level, sec_name, sec_value, aware = key
        hit_rates = np.asarray([g["policy_hit_rate_pct"] for g in group], dtype=float)
        mean_errors = np.asarray([g["policy_mean_error_m"] for g in group], dtype=float)
        median_errors = np.asarray([g["policy_median_error_m"] for g in group], dtype=float)
        p90_errors = np.asarray([g["policy_p90_error_m"] for g in group], dtype=float)
        final_costs = np.asarray([g["final_cost"] for g in group], dtype=float)

        def mean_std(values: np.ndarray):
            valid = values[~np.isnan(values)]
            if valid.size == 0:
                return np.nan, np.nan
            return float(valid.mean()), float(valid.std())

        hr_mean, hr_std = mean_std(hit_rates)
        err_mean, err_std = mean_std(mean_errors)
        median_mean, median_std = mean_std(median_errors)
        p90_mean, p90_std = mean_std(p90_errors)
        cost_mean, cost_std = mean_std(final_costs)

        out.append(
            {
                "noise_type": noise_type,
                "noise_level_name": level_name,
                "noise_level": noise_level,
                "secondary_param_name": sec_name,
                "secondary_param": sec_value,
                "noise_aware": aware,
                "num_seeds": len(group),
                "policy_hit_rate_mean_pct": hr_mean,
                "policy_hit_rate_std_pct": hr_std,
                "policy_mean_error_mean_m": err_mean,
                "policy_mean_error_std_m": err_std,
                "policy_median_error_mean_m": median_mean,
                "policy_median_error_std_m": median_std,
                "policy_p90_error_mean_m": p90_mean,
                "policy_p90_error_std_m": p90_std,
                "final_cost_mean": cost_mean,
                "final_cost_std": cost_std,
            }
        )

    out.sort(key=lambda row: (row["noise_type"], row["secondary_param"], row["noise_level"], row["noise_aware"]))
    return out


def write_markdown_table(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("No data available.\n", encoding="utf-8")
        return

    lines = [
        "| Noise Type | Level | Secondary | Aware | Seeds | Hit Rate % | Mean Error (m) | Median Error (m) | P90 Error (m) | Final Cost |\n",
        "|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in rows:
        lines.append(
            "| {noise_type} | {level} | {secondary} | {aware} | {seeds} | {hr_mean} +- {hr_std} | "
            "{err_mean} +- {err_std} | {med_mean} +- {med_std} | {p90_mean} +- {p90_std} | "
            "{cost_mean} +- {cost_std} |\n".format(
                noise_type=row["noise_type"],
                level=format_num(row["noise_level"], 3),
                secondary=format_num(row["secondary_param"], 3),
                aware="Y" if row["noise_aware"] else "N",
                seeds=row["num_seeds"],
                hr_mean=format_num(row["policy_hit_rate_mean_pct"], 2),
                hr_std=format_num(row["policy_hit_rate_std_pct"], 2),
                err_mean=format_num(row["policy_mean_error_mean_m"], 4),
                err_std=format_num(row["policy_mean_error_std_m"], 4),
                med_mean=format_num(row["policy_median_error_mean_m"], 4),
                med_std=format_num(row["policy_median_error_std_m"], 4),
                p90_mean=format_num(row["policy_p90_error_mean_m"], 4),
                p90_std=format_num(row["policy_p90_error_std_m"], 4),
                cost_mean=format_num(row["final_cost_mean"], 4),
                cost_std=format_num(row["final_cost_std"], 4),
            )
        )

    path.write_text("".join(lines), encoding="utf-8")


def write_latex_table(path: Path, rows: list[dict]):
    lines = [
        "\\begin{tabular}{llcccc}\n",
        "\\hline\n",
        "Noise & Aware & Level & Secondary & Hit rate (\\%) & Mean error (m) \\\\\n",
        "\\hline\n",
    ]
    for row in rows:
        lines.append(
            "{noise_type} & {aware} & {level} & {secondary} & {hr_mean} $\\pm$ {hr_std} & {err_mean} $\\pm$ {err_std} \\\\\n".format(
                noise_type=row["noise_type"],
                aware="Y" if row["noise_aware"] else "N",
                level=format_num(row["noise_level"], 3),
                secondary=format_num(row["secondary_param"], 3),
                hr_mean=format_num(row["policy_hit_rate_mean_pct"], 2),
                hr_std=format_num(row["policy_hit_rate_std_pct"], 2),
                err_mean=format_num(row["policy_mean_error_mean_m"], 4),
                err_std=format_num(row["policy_mean_error_std_m"], 4),
            )
        )
    lines.extend(["\\hline\n", "\\end{tabular}\n"])
    path.write_text("".join(lines), encoding="utf-8")


def filter_rows(rows: list[dict], noise_type: str):
    return [row for row in rows if row["noise_type"] == noise_type]


def make_trend_plot(rows: list[dict], output_path: Path, title_prefix: str):
    if not rows:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = [
        ("policy_hit_rate_mean_pct", "policy_hit_rate_std_pct", "Hit Rate (%)"),
        ("policy_mean_error_mean_m", "policy_mean_error_std_m", "Mean Error (m)"),
        ("final_cost_mean", "final_cost_std", "Final Cost"),
    ]

    aware_styles = {
        True: ("aware", "tab:blue"),
        False: ("naive", "tab:orange"),
    }

    for aware, (label, color) in aware_styles.items():
        aware_rows = [row for row in rows if row["noise_aware"] == aware]
        levels = sorted({row["noise_level"] for row in aware_rows})
        for ax, (mean_key, std_key, ylabel) in zip(axes, metrics):
            x = []
            y = []
            yerr = []
            for level in levels:
                matches = [row for row in aware_rows if row["noise_level"] == level]
                if not matches:
                    continue
                row = matches[0]
                x.append(level)
                y.append(row[mean_key])
                yerr.append(row[std_key])
            if x:
                ax.errorbar(x, y, yerr=yerr, marker="o", capsize=4, linewidth=1.8, label=label, color=color)
            ax.set_xlabel("noise level")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

    axes[0].set_title(f"{title_prefix}: hit rate")
    axes[1].set_title(f"{title_prefix}: mean error")
    axes[2].set_title(f"{title_prefix}: final cost")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_qualitative_plot(run_rows: list[dict], noise_type: str, output_path: Path):
    if not run_rows:
        return

    levels = sorted({row["noise_level"] for row in run_rows})
    if not levels:
        return

    representative_levels = levels if len(levels) <= 4 else [levels[0], levels[len(levels) // 2], levels[-1]]
    fig, axes = plt.subplots(len(representative_levels), 2, figsize=(9, 3.4 * len(representative_levels)), squeeze=False)

    for row_idx, level in enumerate(representative_levels):
        for col_idx, aware in enumerate([True, False]):
            ax = axes[row_idx][col_idx]
            matching_runs = [
                row for row in run_rows
                if row["noise_level"] == level and row["noise_aware"] == aware
            ]
            for run_row in matching_runs:
                run_dir = Path(run_row["run_dir"])
                config, log = load_run(run_dir)
                landings, targets = get_policy_landings(config, log)
                if landings.size == 0:
                    continue
                ax.scatter(targets[:, 0], targets[:, 1], s=28, marker="x", alpha=0.85, color="tab:green")
                ax.scatter(landings[:, 0], landings[:, 1], s=22, alpha=0.65, color="tab:red")

            ax.set_title(f"{noise_type} level={format_num(level, 3)} {'aware' if aware else 'naive'}")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser("Summarize PyBullet noise-study results")
    parser.add_argument(
        "--study_results_root",
        type=str,
        default="results_mc_pilot_pb_noise_study",
        help="Comma-separated study roots to merge before summarizing",
    )
    parser.add_argument("--output_dir", type=str, default="results_mc_pilot_pb_noise_report")
    parser.add_argument("--hit_threshold", type=float, default=0.10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    study_roots = [Path(part.strip()).resolve() for part in args.study_results_root.split(",") if part.strip()]
    run_dirs = []
    for root in study_roots:
        run_dirs.extend(collect_run_dirs(root))

    run_rows = [compute_run_row(run_dir, args.hit_threshold) for run_dir in run_dirs]
    run_rows = dedupe_run_rows(run_rows)
    run_rows.sort(key=lambda row: (row["noise_type"], row["noise_level"], row["noise_aware"], row["seed"]))

    agg_rows = aggregate_rows(run_rows)

    write_csv(output_dir / "noise_raw_runs.csv", run_rows)
    write_csv(output_dir / "noise_summary.csv", agg_rows)
    write_markdown_table(output_dir / "noise_summary.md", agg_rows)
    write_latex_table(output_dir / "noise_summary.tex", agg_rows)

    for noise_type in ["slip", "saltpepper"]:
        noise_rows = filter_rows(agg_rows, noise_type)
        run_noise_rows = filter_rows(run_rows, noise_type)
        write_csv(output_dir / f"{noise_type}_summary.csv", noise_rows)
        write_markdown_table(output_dir / f"{noise_type}_summary.md", noise_rows)
        write_latex_table(output_dir / f"{noise_type}_summary.tex", noise_rows)
        make_trend_plot(noise_rows, output_dir / f"{noise_type}_trends.png", noise_type)
        make_qualitative_plot(run_noise_rows, noise_type, output_dir / f"{noise_type}_landings.png")

    print("Saved paper-oriented noise summaries to", output_dir)


if __name__ == "__main__":
    main()
