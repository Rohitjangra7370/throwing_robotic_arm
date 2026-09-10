# Figure/table placement instructions — paper_icra2027/overleaf/main.tex

Companion to `2026-08-27-paper-figures-checklist.md` (the shopping list). This
doc is the next step: exact anchor point in `main.tex` for every item, plus a
ready-to-paste LaTeX block. Line numbers below are current as of this
session's `main.tex` — re-check with `grep -n "^\\\\subsection"` before
pasting if the file has been edited since.

One asset was built and is ready to drop in now (§1). Everything else is
either a ready LaTeX table (data already exists, §2–3) or a placement stub
with a generation command for later (§4).

---

## 1. READY NOW — throw-motion figure

**SUPERSEDED THIS SESSION.** The original 4-panel static strip
(`trajectory_phase_strip.png`, still on disk, not deleted) read as four
disconnected screenshots with no sense of motion or path — flagged as not
good enough. Replaced with a single chronophotography-style composite:

**Asset**: `paper_icra2027/overleaf/figs/throw_motion_ghost.png` (945×945).
Built from the same source video
(`status_update/vids/gen3_optimized_throw_p0.mp4`) but properly derived, not
hand-picked frames:
- **Background plate** = per-pixel median over all 129 frames (removes
  anything that moves, leaves the static checkerboard/box).
- **Ghosted arm trail** = 8 frames across the windup->release window (n=0,5,
  10,15,20,24,27,30), each frame's foreground (pixels differing from the
  background plate beyond a threshold) alpha-blended onto the plate with
  opacity increasing from 0.30 (earliest/windup) to 1.0 (release) — motion
  reads as a fading trail culminating in a solid arm at the release pose,
  standard chronophotography convention.
- **Ball flight path** = actual detected ball centroid (olive-color
  threshold in RGB) in every one of the 95/129 frames where a real blob was
  found, not interpolated or guessed. Traced frame 34 (first detection,
  release) through frame 104 (position stops changing, landed in the box),
  drawn as a time-graded (plasma colormap) connecting line with markers
  every 8 frames and an arrowhead at the landing point. Release and landing
  are explicitly circled and labeled; all text has a white halo for
  legibility against the checkerboard.

Same honesty caveat as before: this render predates the 2026-08-27
TCP-offset fix and `results_kinetic_chain_gen3_tcp/1` — illustrates the
trajectory *and flight-path* structure, not a specific reported accuracy
number. Caption should say so explicitly.

**Anchor**: end of `\subsection{Dynamics-Aware Throw Execution}`
(`main.tex` line 317, the paragraph ending "...zero-amplitude windup
configuration."), immediately before `\subsection{Height-Generalized Policy}`.

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.85\linewidth]{figs/throw_motion_ghost.png}
\caption{Throw motion and resulting ball flight, simulated. The arm is
shown as a fading trail from windup (dim) to release (solid); the ball's
flight path is traced from its actual detected position in each video
frame, not interpolated. Shown for illustration of the trajectory and
release-to-landing structure described above; not tied to a specific
reported accuracy figure.}
\label{fig:phase-strip}
\end{figure}
```

---

## 2. READY NOW — tables from existing data (no new figure needed)

### 2a. Parameters table

No equivalent to MC-PILOT's Table 1 currently exists in the draft (checked:
only `tab:hw-status` and the feasibility-check table are present). Values
below are sourced from `train_mc_pilot_pb_arm.py` argparse defaults and
`robot_arm/robot_profiles.py`'s `kinova_gen3_dyn` entry — both confirmed by
direct read this session. **Two cells are marked TBD** — don't fill them from
memory, pull from the actual table file named:

**Anchor**: end of `\subsection{Training and Implementation Details}`
(`main.tex` line ~392, after "...physical scale check before use."),
immediately before `\section{Experimental Results}`.

```latex
\begin{table}[t]
\centering
\small
\caption{MC-PILOT training parameters, Gen3 hardware track.}
\label{tab:params}
\begin{tabular}{@{}lc@{}}
\toprule
Parameter & Value \\
\midrule
$N_{\mathrm{exp}}$ (exploration trials) & 5 \\
$N_a$ (rotation augmentations) & 0 \\
$N_{\mathrm{opt}}$ (policy-optimization steps) & 1500 \\
$M$ (sparse-GP inducing points) & 400 \\
$N_b$ (mini-batch size) & 250 \\
$T_s$ (control timestep, s) & 0.02 \\
$T$ (trial horizon, s) & 0.60 \\
$\ell_c$ (cost lengthscale, m) & 0.5 \\
$\gamma_M$ (azimuth half-wedge, deg) & 30.0 \\
$u_{\min}, u_M$ (release speed bounds, m/s) & \textbf{TBD} -- pull from \texttt{throw\_pose\_table\_tcp.npy}'s recorded speed range, not the profile default (0.3, 0.6), which applies only to non-\texttt{opt\_pose} training \\
$\dot q^{\max}$ (joint velocity, rad/s) & 1.3963 (J1--4), 1.2218 (J5--7) \\
$\tau_{\max}$ (joint torque, N\textperiodcentered m) & 39 (J1--4), 9 (J5--7) \\
base plate height (m) & 0.433 \\
TCP offset $r_{\mathrm{offset}}$ (m) & 0.12 along ee\_link $z$ \\
lengthscale init & $0.15\times(l_M-l_m)$, trained thereafter \\
\bottomrule
\end{tabular}
\end{table}
```

### 2b. Per-seed evaluation matrix table

Data already computed at `status_update/eval_matrix.md` — not yet in the
LaTeX draft. This formalizes the "1.86 cm mean, 3.17 cm P95" already stated
in prose at line ~428.

**Anchor**: immediately after the `fig6_eval_error_distribution.png` figure
block closes (`main.tex`, search `\label{fig:eval-dist}` or the line right
after `fig6`'s `\end{figure}`), before `\subsection{Cross-Manipulator
Evaluation}`.

```latex
\begin{table}[t]
\centering
\small
\caption{Per-seed evaluation, 50 fresh targets each (250 throws total).}
\label{tab:eval-matrix}
\begin{tabular}{@{}cccccccc@{}}
\toprule
Seed & Targets & Hit\textless10cm & Hit\textless5cm & Mean & Median & P95 & Max \\
\midrule
1 & 50 & 100\% & 100\% & 1.8 & 1.7 & 3.2 & 3.8 \\
2 & 50 & 100\% & 100\% & 1.9 & 1.9 & 2.9 & 3.3 \\
3 & 50 & 100\% & 100\% & 1.8 & 1.7 & 3.2 & 3.7 \\
4 & 50 & 100\% & 100\% & 1.8 & 1.7 & 3.1 & 3.6 \\
5 & 50 & 100\% & 100\% & 2.0 & 2.0 & 3.2 & 3.7 \\
\midrule
\textbf{All} & \textbf{250} & \textbf{100.0\%} & \textbf{100.0\%} & \textbf{1.86} & \textbf{1.77} & \textbf{3.17} & \textbf{3.81} \\
\bottomrule
\end{tabular}
\\[2pt]
\footnotesize All errors in cm.
\end{table}
```

### 2c. Drag-crossover results table

Currently only prose ("0.6--1.7 cm, 3--7$\times$ improvement" at line ~460).
Numbers need pulling from the underlying run logs (drag-sweep results dir) —
**do not fill in placeholder regime/mass values below without checking**,
this table skeleton just fixes the structure MC-PILOT's paper lacks and ours
should have.

**Anchor**: immediately after that paragraph (line ~461), before the
`generalization.png` figure block.

```latex
\begin{table}[t]
\centering
\small
\caption{Landing error by drag regime: analytical baseline (Eq.~13) vs.\ learned model.}
\label{tab:drag-crossover}
\begin{tabular}{@{}lccc@{}}
\toprule
Regime & Analytical (cm) & Learned (cm) & Ratio \\
\midrule
Low drag (tennis ball) & \textbf{TBD} & \textbf{TBD} & baseline wins \\
High drag (whiffle ball) & \textbf{TBD} & 0.6--1.7 & 3--7$\times$ \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 3. BUILT THIS SESSION — ready to drop in

All four assets below are real, generated from actual repo data/logs (not
placeholders). PDFs are the vector originals for LaTeX; PNGs are preview
copies only.

### 3a. Height-adaptation comparison chart — REBUILT
`figs/height_adaptation.pdf` replaces `results/final_figures/fig_height_adaptation.png`
at its existing anchor (line ~507). Same numbers already in the draft (2.95/
3.41/3.80 cm vs. 3.15 cm ground vs. 3.63 cm retrain, stated in prose at line
~484) — this version has no baked-in title and matches the rest of the
regenerated figures' style.

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.9\linewidth]{figs/height_adaptation.pdf}
\caption{Zero-new-trial adaptation to new target heights by reusing the
trained GP and re-optimizing only the policy, compared against a full 9-D
retrain.}
\label{fig:height-adapt}
\end{figure}
```

### 3b. Torque-vs-time trace figure — BUILT
`figs/torque_trace.pdf`. Generated by loading the actual
`kinova_gen3_dyn` profile and `throw_pose_table_tcp.npy` (the current
preferred, TCP-corrected pose table), solving one release via
`OptimizedReleaseSolver.solve()`, planning through `ArmController.plan_throw`,
then sampling `get_setpoint`+`inverse_dynamics` (both public methods used
elsewhere in the codebase — no reimplementation of release/feasibility
logic) across the full windup/throw/follow-through trajectory. Peak torque
ratio observed: J2 at ~0.82 during the throw phase, everything else well
under 1.0 — consistent with an accepted candidate.

**Anchor**: `\subsection{Impact of Whole-Trajectory Feasibility}`, right
after the sweep table closes (line ~552), before "The rejection rate falls
monotonically...".

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figs/torque_trace.pdf}
\caption{Torque ratio $|\tau_j|/\tau_{\max,j}$ for all seven joints across
the full windup/throw/follow-through trajectory, for the accepted
TCP-corrected release state (\texttt{results\_kinetic\_chain\_gen3\_tcp/1}).
Peak occurs at J2 during the throw phase ($\approx$0.82).}
\label{fig:torque-trace}
\end{figure}
```

### 3c. Model-belief-trap figure — BUILT
`figs/cost_belief_trap.pdf`. Real data from
`results_kinetic_chain_gen3_tcp/1/log.pkl`'s `cost_trial_list` — the
model-simulated particle cost collapses to ~1e-3 by trial 2 and stays there
for all 10 trials, annotated against the checkpoint's real fresh-seed
physical re-evaluation (mean 1.90 cm, max 4.20 cm). **Deliberately not** a
literal per-point (cost, real-error) scatter — real evaluation was only ever
run on the final converged policy, not per-trial, so a true paired scatter
doesn't exist in the data. This shows the same "model believes it's done,
physical error doesn't move" point honestly, without fabricating a pairing
finer than what was actually measured.

**Anchor**: `\section{Discussion}` (line 725), after the three-conclusions
paragraph, before `\textbf{Limitations.}`. Directly supports the
"training-time particle cost is not used as the primary measure" claim
already stated at `\subsection{Evaluation Protocol}` (line ~343).

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.9\linewidth]{figs/cost_belief_trap.pdf}
\caption{Model-simulated training cost collapses toward zero within two
trials and stays there, while real physical re-evaluation of the same
checkpoint on fresh targets remains at 1.90~cm mean error -- the two are
different metrics on different scales, shown together only to make the
divergence visible.}
\label{fig:cost-belief-trap}
\end{figure}
```

### 3d. Real annotated IR perception frame + triangulated 3D track — BUILT
`figs/ir_perception_frame.png`. Ran the existing
`perception/visualize.py::render_annotated` on the real verified capture
`throws/throw_003.npz` (119/130 frames paired, matches the number already
recorded in `status_update/HANDOFF.md`). Left panel: one annotated stereo-IR
frame (green = detected centroid per camera, red = tracked trail, ChArUco
board visible in scene). Right panel: the resulting triangulated 3D track in
camera frame (no base extrinsic applied yet — that's `T_B_C`, still not on
disk per HANDOFF).

**Anchor**: no perception/vision subsection currently exists in `main.tex` —
add one (Setup or a new Method subsection) before placing this, or use as a
forward-pointer figure in `\subsection{Hardware Validation}`.

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figs/ir_perception_frame.png}
\caption{Real dual-IR stereo frame from the D435i (left) with detected ball
centroids and tracked trail, and the resulting triangulated 3D track in
camera frame (right). From \texttt{throw\_003.npz}, 119/130 frames paired.}
\label{fig:ir-frame}
\end{figure}
```

---

## 4. BLOCKED / SHOOT — placement stub only, asset doesn't exist yet

For each, the LaTeX is a ready stub with the exact figure path it will need
— drop in once the asset exists, no restructuring required.

### 4a. Hero photo (paper opener)
**Status: SHOOT.** No posed real-hardware photo exists anywhere in the repo
— checked video dirs and top-level tree this session. Anchor: top of
`\section{Introduction}` (line 37) or wrapped in `figure*` at the top of
column 1, matching MC-PILOT's Fig. 1 placement.

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figs/hero_gen3_throw.jpg}
\caption{The Kinova Gen3 executing a throw toward the target bin.}
\label{fig:hero}
\end{figure}
```
Anonymization check before committing the photo: no name tags, no lab
signage, no faces.

### 4b. Sim-vs-real release-pose pair
**Status: unblocked on the checkpoint side (TCP-offset fix landed
2026-08-27, `results_kinetic_chain_gen3_tcp/1` ready) but still blocked on
an actual executed loaded throw** — `tab:hw-status` (line ~631) still shows
"Loaded throw: pending." Anchor: `\subsection{Hardware Validation}`, right
after the `error_budget_waterfall.png` figure (line ~649).

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figs/sim_vs_real_release.png}
\caption{Release configuration: simulated (left) vs.\ real Gen3 (right),
same policy checkpoint.}
\label{fig:sim-vs-real}
\end{figure}
```

### 4c. Real IR perception frame (annotated) + triangulated 3D arc
**Status: BUILD, blocked only on running `render_annotated`.** The code and
a verified real frame both already exist (`perception/visualize.py`,
`throws/throw_003.npz`, per `[[project_ball_tracking_real_frame_verification]]`).
This is executable now, unlike the other §4 items. Anchor: wherever the
perception/vision pipeline gets a Method or Setup subsection written up (not
yet in `main.tex` — the vision track isn't described in the current draft at
all; add a subsection first, or place in Hardware Validation as a forward
pointer).

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figs/ir_frame_annotated.png}
\caption{Real dual-IR frame from the D435i with detected ball centroids
(left, right camera) and the resulting triangulated 3D track.}
\label{fig:ir-frame}
\end{figure}
```

### 4d. Real landing-scatter figure (headline hardware result)
**Status: BLOCKED.** Needs, in order: (1) `T_B_C` extrinsic on disk at
`calib/T_B_C.npz` (currently stale/laptop-rig only, per HANDOFF), (2) a
completed loaded ball throw on the TCP-corrected checkpoint, (3)
`perception/trajectory.py`'s ballistic fit validated on real data (currently
synthetic-only). Anchor: `\subsection{Hardware Validation}`, would become
the section's centerpiece figure, likely replacing the current closing
sentence "no real landing result is reported" (line ~663) — **note that
sentence and the 0.500 m/s Cartesian-ceiling caveat immediately above it
(line ~660) are now stale text**, per updated `CLAUDE.md`: the ceiling
question was resolved 2026-08-22 (confirmed not applicable to joint-speed
streaming mode). Not fixing prose here per this session's scope, flagging
for whoever next touches that paragraph.

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figs/real_landing_scatter.png}
\caption{Real Gen3 landing positions vs.\ target locations, N throws,
measured via dual-IR triangulation.}
\label{fig:real-landing}
\end{figure}
```

### 4e. Setup / rig photo
**Status: SHOOT.** Anchor: `\subsection{Hardware Platform}` (line ~369).

### 4f. Release-state geometry schematic
**Status: BUILD, not started.** A diagram (not a photo/render) analogous to
MC-PILOT's Fig. 2 bottom panel: azimuth wedge, $\ell_m$/$\ell_M$ radii,
target domain, base frame axes. Anchor: `\subsection{Release-State
Optimization}` (line ~161), which currently has no accompanying figure at
all despite being the paper's core method contribution.

---

## Summary table

| # | Item | Status | Anchor line |
|---|---|---|---|
| 1 | Trajectory-phase strip | **BUILT** | 317 |
| 2a | Params table | **READY** (2 cells TBD) | 392 |
| 2b | Eval-matrix table | **READY** | ~428 |
| 2c | Drag-crossover table | skeleton only, numbers TBD | ~461 |
| 3a | Height-adapt comparison chart | **BUILT** (rebuilt, title stripped) | 507 |
| 3b | Torque-vs-time trace | **BUILT** | 552 |
| 3c | Model-belief-trap figure | **BUILT** | 725 |
| 3d | IR frame annotated + 3D track | **BUILT** | (new subsection) |
| 4a | Hero photo | SHOOT | 37 |
| 4b | Sim-vs-real pose pair | blocked on loaded throw | 649 |
| 4c | Real landing-scatter | blocked on extrinsic + throw | 663 |
| 4d | Setup/rig photo | SHOOT | 369 |
| 4e | Release-geometry schematic | BUILD, not started | 161 |

## Assets on disk (`paper_icra2027/overleaf/figs/`)

| File | Built | Vector? |
|---|---|---|
| `throw_motion_ghost.png` | this session (v2) | raster (photo/render, correct) |
| `trajectory_phase_strip.png` | this session (v1, superseded, kept on disk) | raster |
| `height_adaptation.pdf` / `.png` | this session | PDF is the vector original |
| `torque_trace.pdf` / `.png` | this session | PDF is the vector original |
| `cost_belief_trap.pdf` / `.png` | this session | PDF is the vector original |
| `ir_perception_frame.png` | this session | raster (photo, correct) |
| `error_budget_waterfall.png` | prior session | still has baked-in title, not yet stripped |
