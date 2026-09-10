#!/usr/bin/env python3
"""Regenerate the ICRA paper figures at final print size.

Every figure here is authored at the exact width it is placed at in the
two-column IEEE layout (COL for single-column, FULL for figure*), with fonts
set so that in-figure text lands at roughly caption size in the printed PDF.
The previous figure set was authored ~6-7.5in wide and scaled into a 3.11in
column, which rendered all internal text at 41-52% of its intended size.

Do not "make it bigger later" by changing \\includegraphics width -- that is
what broke the old set. Change FIGSIZE here and re-run.

Usage:  python3 make_figures.py [--outdir ../overleaf/figs]

Data provenance
  fig_seed_reliability   PROGRESS_REPORT.md S1 (seeds converged, 5-seed sweep)
  fig_torque_sweep       ../results/ablation_kinova_gen3_dyn_tau*.json  (real files)
  fig_object_sweep       mc-pilot-pybullet/results_generalization/generalization.npz
  fig_height_adaptation  PROGRESS_REPORT.md S6.4 / paper Sec. V-C
  fig_error_budget       paper Sec. V-G (independently measured bring-up terms)
  fig_height_gen         ../results/heightgen_kuka_iiwa.npz -- real eval_heightgen.py
                         re-run (2026-08-28) on results_mc_pilot_pb_A_hgen/1 (KUKA
                         iiwa, the checkpoint actually behind the paper's 2.0/5.0cm
                         claim -- confirmed via matched setup + reproduced numbers
                         across 3 fresh seeds; NOT the Gen3 overhead checkpoint)
  fig_range_ceiling      ../results/range_speed_sweep.json -- real
                         mc-pilot-pybullet/paper_range_speed_sweep.py re-run
                         2026-09-03 (fixed a stale hardcoded t_throw=1.1s that
                         didn't match the checkpoint's real T_R=1.6s; verified
                         the fix doesn't move safe_speed/safe_range here --
                         follow_through_feasible, which doesn't take t_throw,
                         is what actually binds at the cutoff -- but it was
                         wrong and drifting from cfg regardless). Replaces a
                         hand-made, never-committed PNG whose in-image numbers
                         (0.83/0.86m) matched neither this script's output
                         (0.70/0.94m) nor the paper prose (0.82/0.87m, from
                         the separate full-grid Table I/II analysis) -- see
                         PROGRESS_REPORT.md or ask; not resolved, flagged only.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE.parent / "results"
NPZ = REPO / "mc-pilot-pybullet" / "results_generalization" / "generalization.npz"

# --- print geometry -------------------------------------------------------
COL = 3.11   # single IEEE column at 0.9\linewidth
FULL = 7.16  # \textwidth for figure*
DPI = 400

# Okabe-Ito subset; validated CVD-safe (worst adjacent deutan dE 11.0,
# normal-vision dE 25.8) via the dataviz validate_palette.js six checks.
BLUE = "#0072B2"
VERM = "#D55E00"
GREEN = "#009E73"
GREY = "#595959"  # darkened from #6b6b6b -- the lighter grey read as washed-out
                   # once used as tick-label text, not just as a bar fill
INK = "#1a1a1a"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.labelweight": "bold",
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.edgecolor": "#444444",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "grid.color": "#d0d0d0",
    "grid.linewidth": 0.5,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def _bold_ticks(ax):
    """Axis tick numerals/categories are the smallest text in every figure --
    bold them everywhere so they hold up at actual print (3.11in column) size,
    not just at the full-size render this script's preview shows."""
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")


def _finish(ax, ygrid=True):
    """Recessive axes: no top/right spine, grid behind the marks."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ygrid:
        ax.yaxis.grid(True, linestyle="-", alpha=0.7)
        ax.set_axisbelow(True)
    _bold_ticks(ax)


def save(fig, outdir, name):
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# --- 1. seed reliability --------------------------------------------------
def fig_seed_reliability(outdir):
    labels = ["Random\nexploration", "+ Stratified\nexploration", "+ Scaled\nlengthscale"]
    converged = [1, 3, 5]

    bar_colors = [GREY, BLUE, GREEN]
    fig, ax = plt.subplots(figsize=(COL, 1.52))
    bars = ax.bar(labels, converged, width=0.62, color=bar_colors,
                  edgecolor="white", linewidth=0.8, zorder=3)
    for b, v in zip(bars, converged):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.12, f"{v}/5",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylabel("Seeds converged")
    ax.set_ylim(0, 5.9)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_xlabel("Exploration and lengthscale initialization")
    _finish(ax)
    # Category labels stay black. They were previously tinted to match their
    # own bar, which is the one thing a formal figure must not do: text and
    # data then share a colour, and the label stops being readable as text.
    save(fig, outdir, "fig_seed_reliability")


# --- 1b. height-generalization scatter (KUKA) -----------------------------
def _rolling_mean_std(x_sorted, y_sorted, win):
    """Centered rolling mean/std over y (already sorted by x). Windowed by
    index, not by x-distance -- fine here since heights are near-uniformly
    sampled. n is small (~120), so a plain O(n*win) loop is cheap and keeps
    this dependency-free (no scipy/pandas import for one figure)."""
    n = len(y_sorted)
    half = win // 2
    means = np.empty(n)
    stds = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        window = y_sorted[lo:hi]
        means[i] = window.mean()
        stds[i] = window.std()
    return means, stds


def fig_height_gen(outdir):
    npz = RESULTS / "heightgen_kuka_iiwa.npz"
    if not npz.exists():
        sys.exit(f"missing {npz} -- run eval_heightgen.py first")
    d = np.load(npz, allow_pickle=True)
    h_cm = d["targets"][:, 2] * 100.0
    err_cm = d["errs"] * 100.0

    order = np.argsort(h_cm)
    hs, es = h_cm[order], err_cm[order]
    trend, trend_std = _rolling_mean_std(hs, es, win=15)

    fig, ax = plt.subplots(figsize=(COL, 1.75))
    ax.axhline(10.0, color=VERM, linestyle="--", linewidth=0.8, zorder=2)
    ax.axhline(5.0, color=GREY, linestyle=":", linewidth=0.8, zorder=2)
    # rolling band under the scatter: shows the claimed "no systematic
    # degradation with height" directly, instead of asking the reader to
    # eyeball an unstructured cloud of 120+ points
    ax.fill_between(hs, trend - trend_std, trend + trend_std,
                    color=BLUE, alpha=0.15, zorder=1, linewidth=0)
    ax.plot(hs, trend, color=INK, linewidth=1.2, zorder=2,
           solid_capstyle="round")
    ax.scatter(h_cm, err_cm, s=10, color=BLUE, alpha=0.75,
              edgecolor="none", zorder=3)
    # threshold labels: bold black, not color-coded to the line -- legible
    # on its own and doesn't rely on a reader distinguishing dashed vs.
    # dotted at print size
    ax.text(h_cm.max(), 10.0, "threshold 10 cm", color=INK,
           fontsize=7.5, fontweight="bold", va="bottom", ha="right")
    ax.text(h_cm.max(), 5.0, "threshold 5 cm", color=INK,
           fontsize=7.5, fontweight="bold", va="bottom", ha="right")
    ax.set_xlabel("target height (cm)")
    ax.set_ylabel("Landing error (cm)")
    ax.set_ylim(0, 11.5)
    _finish(ax)
    save(fig, outdir, "fig_height_gen")


# --- 2. torque-headroom dose-response ------------------------------------
# Which ablation dataset the paper reports. The original runs loaded the URDF
# raw, giving PyBullet's default 1 kg to each of the Gen3's bodyless links --
# +3.00 kg / +46% hung off the wrist, which inflated every torque check on the
# Gen3 side only (the Panda has no bodyless links, and its repaired run
# reproduces the published numbers exactly, which is the control). "_repaired"
# is the same search with those links given an explicit zero mass.
# Set to "" to reproduce the pre-2026-09-10 (phantom-mass) numbers.
ABLATION_VARIANT = "_repaired"


def _ablation_path(stem):
    """Preferred ablation file for `stem`, newest-provenance first.

    `_t16` is the repaired search re-run at the ramp duration the deployed
    planner actually uses (T_R = 1.6 s, forced by train_mc_pilot_pb_arm.py
    whenever --opt_pose is given). The older files used the script's 1.1 s
    default, which is shorter than deployed and so over-rejects on torque.
    """
    for suffix in (ABLATION_VARIANT + "_t16", ABLATION_VARIANT, ""):
        cand = RESULTS / f"ablation_{stem}{suffix}.json"
        if cand.exists():
            return cand
    return None


def _load_torque_sweep():
    """Read the real ablation JSONs; express each failure stage as a share of
    the release-instant-feasible candidates so the three stack to the total
    rejection rate on a single axis (never a dual axis)."""
    ks = ["0.50", "0.75", "1.00", "1.50", "2.25", "3.33"]
    rows = []
    for k in ks:
        p = _ablation_path(f"kinova_gen3_dyn_tau{k}")
        if p is None and k == "1.00":
            p = RESULTS / "ablation_kinova_gen3_dyn_tau1.00_reproduction_check.json"
            if not p.exists():
                p = None
        if p is None:
            sys.exit(f"missing ablation file for k={k}")
        s = json.loads(p.read_text())["stats"]
        inst = s["n_release_instant_ok"]
        rows.append(dict(
            k=float(k),
            windup=100.0 * s["n_windup_path_fail"] / inst,
            ramp=100.0 * s["n_ramp_fail"] / inst,
            follow=100.0 * s["n_follow_fail"] / inst,
        ))
    return rows


def fig_torque_sweep(outdir):
    rows = _load_torque_sweep()
    x = np.arange(len(rows))
    windup = np.array([r["windup"] for r in rows])
    ramp = np.array([r["ramp"] for r in rows])
    follow = np.array([r["follow"] for r in rows])
    total = windup + ramp + follow

    fig, ax = plt.subplots(figsize=(COL, 1.80))
    # 2px-equivalent white gap between stacked segments (linewidth in pt).
    # Hatch density rises with stack order so the three stages stay
    # distinguishable under grayscale/B&W printing, not just by hue.
    common = dict(width=0.66, edgecolor="white", linewidth=0.7, zorder=3)
    ax.bar(x, windup, label="Windup path", color=BLUE, hatch="", **common)
    ax.bar(x, ramp, bottom=windup, label="Throw ramp", color=VERM,
           hatch="///", **common)
    ax.bar(x, follow, bottom=windup + ramp, label="Follow-through",
           color=GREEN, hatch="...", **common)

    for xi, t in zip(x, total):
        ax.text(xi, t + 0.018 * float(total.max()) * 1.42,
                f"{t:.0f}", ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['k']:.2f}" for r in rows])
    ax.set_xlabel(r"torque scale $k$   ($\tau_{\max}$ relative to real Gen3)")
    ax.set_ylabel("Candidates rejected (%)")
    # Headroom scales with the data: under repaired inertials the tallest bar
    # is ~78%, not 100%, so the fixed 132 ceiling left the plot mostly empty.
    top = float(total.max()) * 1.42
    ax.set_ylim(0, top)
    ax.set_yticks([t for t in (0, 25, 50, 75, 100) if t <= top])

    # Marker for the real arm, offset to the right of k=1.00: placed directly
    # above the bar it crowded both neighbouring value labels.
    ax.annotate("real Gen3", xy=(2.0, total[2] + 0.055 * top),
                xytext=(2.0, 0.42 * top),
                ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                color=INK,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=GREY,
                                shrinkA=2, shrinkB=2))
    # legend above the axes: inside the plot it collided with the k=0.50/0.75 bars
    # Same overflow as fig_height_adaptation: three entries on one row exceed
    # the axes width by ~26%, so the legend printed past the end of the x axis.
    leg = ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0),
              ncol=2, handlelength=1.0, borderpad=0.2, labelspacing=0.2,
              columnspacing=0.9, handletextpad=0.4, fontsize=7.0)
    for t in leg.get_texts():
        t.set_fontweight("bold")
    _finish(ax)
    save(fig, outdir, "fig_torque_sweep")


# --- 3. object mass/radius sweep -----------------------------------------
def fig_object_sweep(outdir):
    if not NPZ.exists():
        sys.exit(f"missing {NPZ}")
    d = np.load(NPZ, allow_pickle=True)
    grid = np.asarray(d["object_grid"]) * 100.0  # m -> cm
    masses = np.asarray(d["masses"]) * 1000.0    # kg -> g
    radii = np.asarray(d["radii"]) * 100.0       # m -> cm

    fig, ax = plt.subplots(figsize=(COL, 1.66))
    # Sequential: single hue, light -> dark (never a rainbow). Anchored at 0
    # deliberately: the finding is that error is ~constant (1.38-1.46 cm), and
    # auto-scaling to that 0.08 cm spread renders near-invariance as a dramatic
    # gradient, visually contradicting the claim the figure supports.
    # Pale end of the single-hue ramp only (Blues 0.05-0.42): the cell
    # annotations have to be black like every other label in the paper, and
    # black is unreadable on the saturated end of full-range "Blues".
    pale = matplotlib.colors.LinearSegmentedColormap.from_list(
        "pale_blues", plt.get_cmap("Blues")(np.linspace(0.05, 0.42, 256)))
    im = ax.imshow(grid, cmap=pale, aspect="auto", origin="lower",
                   vmin=0.0, vmax=2.0)
    ax.set_xticks(range(len(radii)))
    ax.set_xticklabels([f"{r:.1f}" for r in radii])
    ax.set_yticks(range(len(masses)))
    ax.set_yticklabels([f"{m:.0f}" for m in masses])
    ax.set_xlabel("ball radius (cm)")
    ax.set_ylabel("ball mass (g)")

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, fontweight="bold", color=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("mean landing error (cm)", fontsize=7.5,
                 fontweight="bold")
    cb.ax.tick_params(labelsize=7)
    for lbl in cb.ax.get_yticklabels():
        lbl.set_fontweight("bold")
    cb.outline.set_linewidth(0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _bold_ticks(ax)
    save(fig, outdir, "fig_object_sweep")


# --- 4. zero-new-trial height adaptation ---------------------------------
def fig_height_adaptation(outdir):
    """Zero-new-trial height adaptation, all conditions on ONE shared target set.

    Earlier numbers (3.15 / 2.95 / 3.41 / 3.80 cm) were each measured on the
    condition's own re-optimized target band, and those bands are not
    comparable -- the ground checkpoint's is 7 cm wide, the h=0.10 band 87 cm.
    These are re-evaluated on a single fixed 30-target set drawn from the
    intersection of all four bands (flight 0.712-0.782 m, |beta| <= 30 deg),
    so the only thing that varies across bars is the target height. Source:
    ../results/height_adapt_shared/*.json.

    The complete 9-D retraining bar is deliberately absent: the only such
    checkpoint on disk was trained through the pre-TCP-offset pose table from a
    release point 0.39 m lower, so putting it beside these would compare the
    TCP-offset fix, not retraining against adaptation."""
    shared = HERE.parent / "results" / "height_adapt_shared"
    labels = ["ground", "0.10", "0.20", "0.30"]
    files = ["ground", "h10", "h20", "h30"]
    vals = []
    for f in files:
        j = shared / f"{f}.json"
        if not j.exists():
            sys.exit(f"missing {j} -- run eval_adapted_height.py --targets first")
        rows = json.loads(j.read_text())
        vals.append(100.0 * float(np.mean([r["err"] for r in rows])))
    colors = [GREY] + [GREEN] * 3

    fig, ax = plt.subplots(figsize=(COL, 1.56))
    bars = ax.bar(labels, vals, width=0.62, color=colors, edgecolor="white",
                  linewidth=0.8, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                color=INK)
    ax.set_ylabel("Mean landing error (cm)")
    ax.set_xlabel("Target height $h$ (m)")
    ax.set_ylim(0, max(vals) * 1.35)

    handles = [plt.Rectangle((0, 0), 1, 1, color=GREY),
               plt.Rectangle((0, 0), 1, 1, color=GREEN)]
    leg = ax.legend(handles, ["ground baseline", "zero new trials"],
                    frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0),
                    ncol=2, handlelength=1.0, borderpad=0.2, labelspacing=0.2,
                    columnspacing=0.9, handletextpad=0.4, fontsize=7.0)
    for t in leg.get_texts():
        t.set_fontweight("bold")
    _finish(ax)
    save(fig, outdir, "fig_height_adaptation")


# --- 5. sim-to-real release-timing budget --------------------------------
def fig_error_budget(outdir):
    """Net landing error against the instant the object actually leaves the hand.

    Replaces the earlier stacked bar chart, which put four independently
    measured terms on a common magnitude axis and so implied they add. They do
    not: the arm's command lag makes the throw SHORT, a late release makes it
    LONG, and the two cross. Plotting the net error as a function of the
    departure instant is the honest version of the same measurement, and it is
    recomputed from the recorded traces rather than transcribed.
    """
    j = RESULTS / "hw_release_timing.json"
    if not j.exists():
        sys.exit(f"missing {j}")
    d = json.loads(j.read_text())
    x = np.array(d["delays_s"]) * 1e3
    mu = np.array([v[0] for v in d["landing_vs_delay_cm"]])
    sd = np.array([v[1] for v in d["landing_vs_delay_cm"]])
    sim = 1.90  # checkpoint's own fresh-seed sim accuracy, cm

    fig, ax = plt.subplots(figsize=(COL, 1.85))
    ax.axhspan(-sim, sim, color=GREY, alpha=0.18, zorder=1, linewidth=0)
    ax.axhline(0.0, color=GREY, linewidth=0.7, zorder=2)
    ax.fill_between(x, mu - sd, mu + sd, color=BLUE, alpha=0.22, zorder=3,
                    linewidth=0)
    ax.plot(x, mu, color=BLUE, linewidth=1.6, zorder=4, solid_capstyle="round")

    # zero crossing by linear interpolation between the bracketing samples
    k = int(np.where(mu > 0)[0][0])
    xz = x[k - 1] + (x[k] - x[k - 1]) * (-mu[k - 1]) / (mu[k] - mu[k - 1])
    ax.scatter([xz], [0.0], s=26, color=VERM, zorder=5, edgecolor="white",
               linewidth=0.6)

    # window in which the fingers were observed to first move during a throw
    lo = min(g["first_motion_after_tr_s"] for g in d["gripper_in_throw"]) * 1e3
    hi = max(g["first_motion_after_tr_s"] for g in d["gripper_in_throw"]) * 1e3
    ax.axvspan(lo, hi, color=VERM, alpha=0.14, zorder=1, linewidth=0)

    ax.annotate(f"net error crosses\nzero at {xz:.0f} ms", xy=(xz, 0.0),
                xytext=(0.30 * x[-1], -7.4), fontsize=7, fontweight="bold",
                color=INK, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.7, color=GREY,
                                connectionstyle="arc3,rad=0.0"))
    ax.annotate("fingers\nfirst move", xy=(0.5 * (lo + hi), 0.99 * mu.max()),
                xytext=(0.5 * (lo + hi), 0.99 * mu.max()), fontsize=7,
                fontweight="bold", color=VERM, ha="center", va="top",
                linespacing=1.1)
    ax.text(0.985 * x[-1], -sim - 0.7, "checkpoint sim accuracy", fontsize=6.5,
            fontweight="bold", color=GREY, ha="right", va="top")

    ax.set_xlabel("object departure, ms past the planned release instant")
    ax.set_ylabel("landing error\nvs. plan (cm)")
    ax.set_xlim(0, x[-1])
    ax.set_ylim(mu.min() - 2.6, mu.max() + 3.4)
    _finish(ax)
    save(fig, outdir, "fig_error_budget")


# --- 5b. release-speed / range envelope -----------------------------------
def fig_range_ceiling(outdir):
    """Landing distance against commanded release speed in the deployed
    release direction, with the trained operating band marked.

    Reads the TCP-corrected sweep (deployed pose table, deployed ramp duration,
    repaired inertias). The earlier version of this figure split the curve into
    a feasible and an infeasible segment; under corrected inertias the whole
    sweep passes the cascade, so what the figure now shows is headroom: where
    the policy operates versus where the arm's joint-velocity limit puts the
    ceiling.
    """
    j = RESULTS / "range_speed_sweep_tcp.json"
    if not j.exists():
        sys.exit(f"missing {j} -- run paper_range_speed_sweep_tcp.py first")
    d = json.loads(j.read_text())
    speeds = np.array(d["speeds"])
    ranges = np.array(d["ranges"])
    ok = np.array(d["cascade_ok"], dtype=bool)
    u_lo, u_hi = 1.345, 1.470       # speeds the trained policy actually commands
    r_lo, r_hi = 0.67, 0.74         # trained target band

    fig, ax = plt.subplots(figsize=(COL, 1.78))
    ax.axhspan(r_lo, r_hi, color=BLUE, alpha=0.15, zorder=1, linewidth=0)
    ax.axvspan(u_lo, u_hi, color=BLUE, alpha=0.15, zorder=1, linewidth=0)
    ax.plot(speeds, ranges, color=BLUE, linewidth=1.6, zorder=3,
            solid_capstyle="round")
    if not ok.all():
        i = int(np.where(ok)[0][-1])
        ax.plot(speeds[i:], ranges[i:], color=VERM, linewidth=1.6,
                linestyle=(0, (4, 1.5)), zorder=4)
    ax.scatter([speeds[-1]], [ranges[-1]], s=26, color=VERM, zorder=5,
               edgecolor="white", linewidth=0.6)

    xmax, ymax = speeds[-1] * 1.10, ranges[-1] * 1.34
    ax.annotate(f"joint-velocity ceiling\n{speeds[-1]:.2f} m/s, {ranges[-1]:.2f} m",
                xy=(speeds[-1], ranges[-1]),
                xytext=(0.985 * xmax, 0.30 * ymax),
                fontsize=7, fontweight="bold", color=INK, ha="right",
                va="bottom",
                arrowprops=dict(arrowstyle="-", lw=0.7, color=GREY,
                                connectionstyle="arc3,rad=0.18"))
    ax.text(0.04 * xmax, 0.97 * ymax, "trained band", fontsize=7,
            fontweight="bold", color=BLUE, ha="left", va="top")

    ax.set_xlabel("commanded release speed (m/s)")
    ax.set_ylabel("landing distance\nfrom base (m)")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    _finish(ax)
    save(fig, outdir, "fig_range_ceiling")


# --- 5c. per-seed evaluation spread ---------------------------------------
def fig_eval_spread(outdir):
    """Final-policy landing error per seed, 50 unseen targets each.

    Replaces status_update/fig6_eval_error_distribution.png, which was authored
    in a different style (sans-serif, chatty title, no x label, colour-filled
    boxes) and read as a figure from another paper. Source is
    status_update/eval_matrix.csv -- the same run that produces the 1.86 cm /
    3.17 cm numbers quoted in the text. That file records per-seed summary
    statistics, not the 250 individual throws, so this draws the recorded
    median / p95 / max explicitly rather than a box plot whose hinges would
    have to be invented."""
    csv = REPO / "status_update" / "eval_matrix.csv"
    if not csv.exists():
        sys.exit(f"missing {csv}")
    rows = [l.split(",") for l in csv.read_text().strip().splitlines()[1:]]
    seeds = [int(r[0]) for r in rows]
    mean = np.array([float(r[4]) for r in rows])
    median = np.array([float(r[5]) for r in rows])
    p95 = np.array([float(r[6]) for r in rows])
    mx = np.array([float(r[7]) for r in rows])
    x = np.arange(len(seeds))

    fig, ax = plt.subplots(figsize=(COL, 1.72))
    ax.axhline(10.0, color=VERM, linestyle="--", linewidth=0.8, zorder=2)
    ax.axhline(5.0, color=GREY, linestyle=":", linewidth=0.8, zorder=2)
    ax.vlines(x, median, mx, color=BLUE, linewidth=4.5, alpha=0.30, zorder=3)
    ax.plot(x, p95, "_", color=BLUE, markersize=9, markeredgewidth=1.4, zorder=4)
    ax.plot(x, mean, "o", color=BLUE, markersize=4.0, zorder=5,
            markeredgecolor="white", markeredgewidth=0.6)
    for xi, m in zip(x, mean):
        ax.text(xi, m - 0.42, f"{m:.2f}", ha="center", va="top", fontsize=6.6,
                fontweight="bold", color=INK)

    ax.text(x[-1] + 0.42, 10.0, "threshold 10 cm", color=INK, fontsize=7,
            fontweight="bold", va="bottom", ha="right")
    ax.text(x[-1] + 0.42, 5.0, "threshold 5 cm", color=INK, fontsize=7,
            fontweight="bold", va="bottom", ha="right")
    # Three marks on one column need naming: without a key the reader cannot
    # tell the p95 rule from the top of the median-max band.
    handles = [
        plt.Line2D([], [], color=BLUE, marker="o", linestyle="none",
                   markersize=4.0, markeredgecolor="white"),
        plt.Line2D([], [], color=BLUE, marker="_", linestyle="none",
                   markersize=9, markeredgewidth=1.4),
        plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.30),
    ]
    leg = ax.legend(handles, ["mean", "95th pct", "median\u2013max"], frameon=False,
                    loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3,
                    handlelength=1.0, borderpad=0.2, labelspacing=0.25,
                    columnspacing=1.0, handletextpad=0.4)
    for t in leg.get_texts():
        t.set_fontweight("bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s_) for s_ in seeds])
    ax.set_xlabel("Random seed (50 unseen targets each)")
    ax.set_ylabel("Landing error (cm)")
    ax.set_xlim(-0.55, len(seeds) - 0.45)
    ax.set_ylim(0, 11.8)
    _finish(ax)
    save(fig, outdir, "fig_eval_spread")


# --- 6. method/pipeline overview (teaser, page 1, figure*) ---------------
def _box(ax, cx, cy, w, h, text, fc, ec=INK, fontsize=7.8, textcolor="white"):
    from matplotlib.patches import FancyBboxPatch
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                        boxstyle="round,pad=0.0,rounding_size=0.10",
                        linewidth=1.1, facecolor=fc, edgecolor=ec, zorder=3)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=textcolor, zorder=4, linespacing=1.3)
    return (cx, cy, w, h)


def _arrow(ax, b_from, b_to, color=INK, lw=1.3, label=None, label_dy=0.18,
           connectionstyle="arc3,rad=0.0", side="auto", label_ha="center"):
    from matplotlib.patches import FancyArrowPatch
    x0, y0, w0, h0 = b_from
    x1, y1, w1, h1 = b_to
    if side == "auto":
        if abs(x1 - x0) >= abs(y1 - y0):
            p0 = (x0 + (w0 / 2 if x1 > x0 else -w0 / 2), y0)
            p1 = (x1 - (w1 / 2 if x1 > x0 else -w1 / 2), y1)
        else:
            p0 = (x0, y0 + (h0 / 2 if y1 > y0 else -h0 / 2))
            p1 = (x1, y1 - (h1 / 2 if y1 > y0 else -h1 / 2))
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11,
                        linewidth=lw, color=color, zorder=2,
                        connectionstyle=connectionstyle, shrinkA=1, shrinkB=1)
    ax.add_patch(a)
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        # Edge labels are black, never the arrow's colour, and are given a
        # white backing so a short arrow span cannot push the text under the
        # box it points into (which is what "speed u" did).
        ax.text(mx, my + label_dy, label, ha=label_ha,
                va="bottom" if label_dy >= 0 else "top",
                fontsize=7, fontweight="bold", color=INK, zorder=5,
                bbox=dict(boxstyle="square,pad=0.10", facecolor="white",
                          edgecolor="none"))


def fig_pipeline(outdir):
    """Graphical abstract: what is searched once offline, what runs the online
    learning loop, and where they meet at throw execution. Figure 1, placed in
    the introduction column.

    Laid out for COL width, not FULL. The earlier version put all five online
    stages in one row, which only works across the full text width; rendered
    into a single column it printed its own box labels at roughly 45% of the
    intended size -- the exact failure this module's header warns about. The
    online chain is therefore folded into two rows of three, which is also what
    makes the model-update loop read as a loop.

    Color coding is semantic and reused from the palette above: GREY = data
    in/out, VERM = offline/classical search, BLUE = the learned components
    (policy + GP model), GREEN = the one step that touches the real world. All
    edge and header text is black regardless of what it labels.
    """
    from matplotlib.patches import FancyArrowPatch
    fig, ax = plt.subplots(figsize=(COL, 2.24))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    box = dict(fontsize=6.0)

    # --- offline row: searched once, before any training trial ---
    ax.add_patch(plt.Rectangle((0.15, 5.02), 9.7, 1.62, facecolor="#f2f2f2",
                                edgecolor="none", zorder=1))
    ax.text(0.30, 6.92, "OFFLINE — arm-specific", fontsize=6.4,
            fontweight="bold", color=INK, ha="left", va="center")
    limits = _box(ax, 2.05, 5.83, 3.5, 1.28,
                  "Arm torque +\nvelocity limits", GREY, **box)
    search = _box(ax, 6.90, 5.83, 5.6, 1.28,
                  "Release-state search\n+ feasibility cascade", VERM, **box)
    _arrow(ax, limits, search, lw=1.0)

    # --- online block: two rows of three, so the loop closes visibly ---
    ax.text(0.30, 4.36, "ONLINE — per trial", fontsize=6.4,
            fontweight="bold", color=INK, ha="left", va="center")
    target = _box(ax, 1.35, 3.06, 2.3, 1.22, "Target\n$(P_x,P_y,h)$", GREY, **box)
    policy = _box(ax, 4.60, 3.06, 3.0, 1.22, "RBF policy", BLUE, **box)
    execute = _box(ax, 8.50, 3.06, 2.8, 1.22, "Execute\nthrow", GREEN, **box)
    dv = _box(ax, 8.50, 0.95, 2.8, 1.22, "Observed\n$\\Delta v$", GREY, **box)
    gp = _box(ax, 4.60, 0.95, 3.0, 1.22, "GP dynamics\nmodel", BLUE, **box)

    _arrow(ax, target, policy, lw=1.0)
    # Deliberately unlabelled. The inter-box gap is narrower than "speed u",
    # and the only band with room above it already carries the ONLINE header
    # and the release-state-table label. That the policy emits the release
    # speed is stated in Sec. II-A instead.
    _arrow(ax, policy, execute, lw=1.0)
    _arrow(ax, execute, dv, lw=1.0)
    _arrow(ax, dv, gp, lw=1.0)
    _arrow(ax, gp, policy, lw=1.0)

    # search -> execute: straight drop onto the column that executes the throw
    drop = FancyArrowPatch((8.50, 5.02), (8.50, 3.72),
                           arrowstyle="-|>", mutation_scale=8, linewidth=1.0,
                           color=VERM, zorder=2, shrinkA=1, shrinkB=1)
    ax.add_patch(drop)
    # Right-aligned against the drop arrow. The headers on this band were
    # shortened to make room for it on a single line.
    ax.text(8.20, 4.36, "release-state table", fontsize=5.6,
            fontweight="bold", color=INK, ha="right", va="center", zorder=4)
    ax.text(4.28, 2.00, "re-optimize\npolicy", fontsize=5.6, fontweight="bold",
            color=INK, ha="right", va="center", zorder=4, linespacing=1.2)

    save(fig, outdir, "fig_pipeline")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(HERE.parent / "overleaf" / "figs"))
    args = ap.parse_args()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"writing to {outdir}")
    fig_pipeline(outdir)
    fig_seed_reliability(outdir)
    fig_height_gen(outdir)
    fig_torque_sweep(outdir)
    fig_object_sweep(outdir)
    fig_height_adaptation(outdir)
    fig_error_budget(outdir)
    fig_eval_spread(outdir)
    fig_range_ceiling(outdir)
    print("done")


if __name__ == "__main__":
    main()
