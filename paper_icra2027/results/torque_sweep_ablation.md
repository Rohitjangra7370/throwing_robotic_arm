# Torque-headroom sweep: is the feasibility-cascade ablation a Gen3 coincidence?

**Date:** 2026-08-27
**Motivation:** The paper's feasibility-cascade ablation (`ablation_kinova_gen3_dyn.json`,
`ablation_franka_panda_dyn.json`, Table `tab:ablation` in `draft.tex`) shows whole-trajectory
feasibility checking rejecting 83.4% of candidates on the Gen3 (39/9 Nm) and 0% on the Panda
(87/12 Nm). That is two arms, one effect, and the two arms differ in kinematics/DOF layout as
well as torque budget — a reviewer can reasonably ask whether "torque headroom" is the real
causal variable or whether it's a Gen3-specific coincidence. This sweep answers that directly
by holding kinematics fixed and varying only torque.

## Method

`mc-pilot-pybullet/paper_ablation_feasibility.py <robot_name> [out.json]` re-runs the exact
candidate enumeration and feasibility cascade from `find_throw_pose.py::search_release_state`
(read-only analysis, not the production pipeline) and reports, per robot profile, how many
release-instant-feasible candidates additionally survive the full whole-trajectory check.

Six synthetic robot profiles were added to `mc-pilot-pybullet/robot_arm/robot_profiles.py`
(`kinova_gen3_dyn_tau{0.50,0.75,1.00,1.50,2.25,3.33}`): **bit-for-bit copies of
`kinova_gen3_dyn`** (same URDF, `q_neutral`, `qd_max`, `windup_delta`, `kp`/`kd`) with only
`tau_max` scaled by the named factor from the real Gen3's 39/9 Nm. This isolates torque
headroom as the sole independent variable — kinematics, DOF layout, and control gains are
identical across every row, which a real second arm (even a real low-torque one) could not
guarantee.

- k=1.00 reproduces the real Gen3 (39/9 Nm) and is a **reproduction check**, not new evidence:
  its stats are bit-identical to the production `ablation_kinova_gen3_dyn.json`
  (`ablation_kinova_gen3_dyn_tau1.00_reproduction_check.json`, diffed and confirmed).
- k=3.33 (129.87/29.97 Nm) approximates the Panda's real torque *ratio* (87/12 Nm) for an
  external cross-check, not as a physically meaningful arm — Panda's own kinematics are not
  reproduced here, only its torque scale.

Each run: `python3 paper_ablation_feasibility.py kinova_gen3_dyn_tau<k> results/ablation_kinova_gen3_dyn_tau<k>.json`,
from `mc-pilot-pybullet/`, single CPU core, PyBullet DIRECT mode, no GPU. Raw stdout for all
six runs (including the reproduction check) is in `ablation_torque_sweep.log`; per-run JSON
in `ablation_kinova_gen3_dyn_tau*.json`.

## Results

| k (τ scale) | wrist τ_max (Nm) | release-instant-ok | full-traj-ok | rejected | windup-fail | ramp-fail | follow-fail | range, instant-only | range, full |
|---|---|---|---|---|---|---|---|---|---|
| 0.50 | 4.5 | 118,954 | 0 | **100%** | 118,954 | 0 | 0 | 1.031 m | *(no feasible throw)* |
| 0.75 | 6.75 | 344,477 | 0 | **100%** | 344,477 | 0 | 0 | 1.031 m | *(no feasible throw)* |
| 1.00 (real Gen3) | 9.0 | 446,071 | 74,230 | **83.4%** | 242,725 | 66,706 | 62,410 | 1.031 m | 0.824 m |
| 1.50 | 13.5 | 475,123 | 261,945 | **44.9%** | 0 | 159,185 | 53,993 | 1.031 m | 1.031 m |
| 2.25 | 20.25 | 475,123 | 391,238 | **17.7%** | 0 | 0 | 83,885 | 1.031 m | 1.031 m |
| 3.33 (~Panda ratio) | 29.97 | 475,123 | 391,238 | **17.7%** | 0 | 0 | 83,885 | 1.031 m | 1.031 m |

(For reference, the real Panda run in `ablation_franka_panda_dyn.json`: 152,844 / 152,844,
0% rejected, best range 1.649 m identical under either criterion — different kinematics from
this sweep, included for context only, not part of the controlled comparison above.)

## Interpretation

1. **Torque headroom is causally load-bearing, holding kinematics fixed.** Rejection rate
   falls monotonically as τ_max scales up: 100% → 100% → 83.4% → 44.9% → 17.7% → 17.7%. Every
   row shares identical kinematics, so this is a controlled dose-response curve, not an
   artifact of comparing two different arms. This directly answers the "one-arm coincidence"
   objection: the mechanism is measured, not asserted from n=1.
2. **Below k≈0.75 (wrist τ_max ≲ 6.75 Nm), zero throws are feasible at all** under the full
   cascade — not merely reduced range, total infeasibility. Worth stating explicitly rather
   than folding into the range-reduction framing.
3. **The curve saturates at 17.7%, not 0%, from k=2.25 onward.** `follow-fail` plateaus at
   exactly 83,885 for both k=2.25 and k=3.33 — the residual rejection at high torque is driven
   entirely by follow-through infeasibility specific to the Gen3's own kinematics/geometry, not
   by torque anymore (`windup-fail` and `ramp-fail` both hit zero by k=1.50 and k=2.25
   respectively). This means whole-trajectory checking catches two distinct failure
   mechanisms — torque-driven (dominant at low headroom) and kinematics-driven (the residual
   floor) — which is a sharper, more defensible claim than "torque headroom matters," and
   explains why this sweep's floor (17.7%) doesn't reach the real Panda's 0%: the Panda's own
   kinematics, not reproduced here, are the reason its floor is lower still.

## Status

Raw results only — not yet folded into `draft.tex`. Candidate next step: a rejection-%-vs-
headroom figure (log-x) either replacing or supplementing Table `tab:ablation`, and a rewrite
of the Discussion sentence ("whole-trajectory checking becomes most consequential when
actuator torque headroom is limited") to cite this sweep instead of asserting the mechanism
from the two-arm table alone.
