# Paper visual assets checklist — figures, photos, plots, tables

Companion to `paper_icra2027/overleaf/main.tex`. Scope: every non-text visual
asset the paper needs, mapped to the section it belongs in, with source and
status. Layout/design pass comes later — this is the shopping list.

Status legend: **HAVE** (raw asset exists, needs extraction/regen only),
**BUILD** (code exists to produce it, hasn't been run/rendered), **SHOOT**
(needs an actual camera at the lab, nothing on disk), **BLOCKED** (needs
something upstream first — named explicitly).

---

## 1. Hero photo (paper opener, analogous to MC-PILOT's Fig. 1)

- **Gen3 mid-throw, real hardware.** Single wide photo, arm in release
  configuration or mid-swing. Status: **SHOOT**. Nothing posed exists — only
  video (`status_update/vids/gen3_optimized_throw_p0.mp4`,
  `gen3_REAL_dynamics_p2.mp4`). A single clean frame pulled from one of these
  via ffmpeg is an acceptable fallback if a fresh photo session isn't
  possible before a deadline, but a posed shot reads better.
  Anonymization: no visible name tags, lab door signage, whiteboards, or
  faces in frame.

## 2. Method section (`Constraint-Aware Model-Based Pick-and-Throw`)

- **Trajectory-phase strip**, 3–4 panels: `(a) neutral (b) windup (c) release
  (d) follow-through`, same camera angle across panels, PyBullet render.
  Status: **BUILD** — `demo_pybullet_gui.py` / `make_dynamic_video.py`
  already render this motion; need single-frame grabs at each phase
  boundary, not the full video.
- **Release-state geometry diagram** (analogous to MC-PILOT Fig. 2 bottom):
  azimuth wedge, `ℓ_m`/`ℓ_M` radii, target domain `D_P`, base frame axes —
  this is a schematic, not a render. Status: **BUILD** — can be drawn from
  the geometry already implemented in `release_solver.py`
  (`OptimizedReleaseSolver`), no existing figure covers it. Consider
  `artifact-diagramming` conventions or a simple matplotlib schematic.
- **Sim-vs-real release pose pair**: one PyBullet frame at release next to
  one real Gen3 frame at release, same pose. Status: **BLOCKED** — real half
  needs the TCP-offset fix landed first (`[[project_gripper_tcp_offset_blocker]]`)
  so the compared pose is meaningful, not just visually similar.

## 3. Experimental Setup section

- **Full rig photo**: Gen3 + workspace + ArUco/ChArUco boards visible
  (`mc-pilot-pybullet/aruco_targets/` has print-ready boards — confirm
  they're actually mounted before shooting). Status: **SHOOT**.
- **Object set photo** (only if the paper ends up testing multiple ball
  types/masses — check against `generalization.png`'s object-sweep panel,
  which is currently simulated only): tennis ball / whiffle ball / whatever
  is physically thrown, on a ruler for scale, MC-PILOT Fig. 7 style. Status:
  **SHOOT**, contingent on whether a physical object sweep is in scope.
- **Camera mount photo**: D435i mounted per `plan_camera_mount.py`'s output,
  showing the actual chosen mount point. Status: **SHOOT** — no camera
  extrinsic finalized yet either (`[[project_camera_extrinsic_d435i]]`),
  photograph once mount is final, not the laptop-rig interim one.
- **Params table** (MC-PILOT Table 1 equivalent): every hyperparameter,
  sim vs. real columns — `N_exp`, `N_a`, `u_M`, `T_s`, `ℓ_c`, `ℓ_m`, `ℓ_M`,
  `qd_max`, `tau_max`, etc. Status: **BUILD** — all values live in
  `train_mc_pilot_pb_arm.py` defaults / `robot_profiles.py`, just needs
  collecting into one LaTeX table. Not currently in the draft (checked:
  only 2 tables exist — feasibility check, hardware status).

## 4. Results section

### 4.1 Optimization Reliability (existing subsection)
- `fig2_seeds_converged.png` — **HAVE**, needs title stripped from image,
  regenerate as vector PDF.
- `fig6_eval_error_distribution.png` — **HAVE**, same treatment.
- **Eval-matrix table** (per-seed metrics: hit rate <10cm/<5cm, mean/median/
  P95/max error) — status: **HAVE**, exists at `status_update/eval_matrix.md`
  as raw data, not yet in the LaTeX draft as a formatted table.

### 4.2 Cross-Manipulator Evaluation
- `generalization.png` 3-panel small-multiples — **HAVE**, keep the panel
  structure (closest thing to MC-PILOT's small-multiples convention),
  strip titles, unify style/palette with the rest.
- **Missing: spatial landing-scatter figure** — target rings + landing
  points overlaid on the x-y ground plane, per manipulator, MC-PILOT
  Fig. 6/11/13/14/15 style. Status: **BLOCKED** on nothing for *simulated*
  manipulators (eval scripts already produce landing positions per throw —
  `eval_baseline.py` et al. — just needs a plotting pass instead of only
  aggregate error). For the *real* Gen3 this is additionally blocked on the
  TCP-offset fix.

### 4.3 Learned Dynamics under Aerodynamic Drag
- Currently text-only in the draft per the section list — needs at minimum
  a **results table**: method (analytical Eq.-13-equivalent / ours) × drag
  regime (low/high) × mean error, to carry the "drag crossover" finding
  (`PROGRESS_REPORT.md`'s headline sim result). Status: **BUILD** — numbers
  exist in prior run logs, needs collecting.

### 4.4 Target-Height Generalization
- `fig11_hgen_error_vs_height.png` — **HAVE**, strip title.
- `fig_height_adaptation.png` — **HAVE**, strip title.
- **Missing: side-by-side method comparison panel** — ground policy vs.
  zero-new-trial adapted vs. full retrain, same axes, MC-PILOT-style 3-panel
  layout instead of one bar chart. Status: **BUILD** — same underlying data
  as the existing bar chart, needs re-plotting as comparable panels.

### 4.5 Low-Torque Throwing and Feasible Range
- `fig_range_ceiling.png` — **HAVE**, strip title.

### 4.6 Impact of Whole-Trajectory Feasibility
- Existing feasibility-check table (table at line ~549) — **HAVE**, keep.
- Consider adding a **torque-vs-time trace figure**: sampled torque across
  all three phases (windup/throw/follow-through) against `tau_max`, showing
  the mid-ramp/follow-through violations this subsection's whole point
  depends on. Status: **BUILD** — `arm_controller.py::plan_throw`'s
  feasibility sampling already computes this per-phase; needs a plot, not
  new data collection.

### 4.7 Computational Efficiency
- No figure currently — likely a **timing table** is enough (search vs.
  train wall-clock, from `timing_search_gen3.log` / `timing_train_gen3.log`,
  already in `paper_icra2027/results/`). Status: **HAVE** raw logs, **BUILD**
  table.

### 4.8 Hardware Validation
- Existing hardware-status table (line ~626) — **HAVE**, keep, update as
  bring-up progresses.
- `error_budget_waterfall.png` — **HAVE**, strip title, keep concept (no
  MC-PILOT equivalent — this is a stronger honesty move than their paper
  makes). Update once a real ball-throw total error exists to compare
  against the summed budget.
- **Real IR perception frame, annotated** (detected ball centroid marked in
  both IR streams) + the triangulated 3D arc plot beside it. Status:
  **BUILD** — `perception/visualize.py::render_annotated` already produces
  this from `throws/throw_003.npz`, a real verified frame
  (`[[project_ball_tracking_real_frame_verification]]`); needs a still frame
  pulled from its output, not the full video.
- **Real landing-scatter figure** (the headline hardware result once it
  exists): targets vs. real landing points on the floor, MC-PILOT Fig. 11/12
  style. Status: **BLOCKED** — needs a completed ball throw past the
  TCP-offset fix, plus `T_B_C` extrinsic on disk (currently stale/laptop-rig
  only), plus `perception/trajectory.py`'s ballistic fit validated on real
  data (currently synthetic-only per HANDOFF).

## 5. Discussion / Limitations section

- No figure required by convention, but if the "model-belief trap" (training
  cost vs. real accuracy divergence) is discussed, a **cost-vs-real-error
  scatter** (one point per trial: GP-predicted trial cost on x, real
  `eval_baseline.py` error on y) would make that claim visually checkable
  instead of asserted. Status: **BUILD** — both quantities are already
  logged separately (`log.pkl` training cost, eval script outputs); needs
  joining and plotting.

---

## Cross-cutting fixes (apply to every item marked HAVE above)

1. Strip in-image titles from every existing PNG — takeaway belongs in the
   `\caption{}`, not baked into the plot.
2. Regenerate matplotlib figures as vector PDF, not PNG — avoids raster
   blur at IEEE two-column print width.
3. One consistent palette/font/box-style across every regenerated figure —
   currently mixed (flat bars in some, seaborn-ish boxplots in others).
4. Every multi-panel figure gets `(a)/(b)/(c)` corner labels.
5. Every photo: anonymization check before it goes in the repo — no faces,
   no name tags, no identifying signage, consistent with double-anon review.
