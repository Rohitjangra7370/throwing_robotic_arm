# Project Progress & Next-Steps Plan — MC-PILOT Throwing Arm (AR525)

Snapshot as of 2026-07-24. All numbers below are pulled directly from `git log`,
file counts, and test/result runs in this repo — not estimates.

## 1. Where we started

- **2026-03-25**: first commit. Repo forked the upstream MC-PILOT reference
  implementation (Turcato et al., arXiv:2502.05595, vendored as `MC-PILCO/`)
  and began the baseline reproduction (`mc-pilot/`).
- `MC-PILCO/` (6,980 lines of Python) has never been edited since vendoring —
  verified via `git log -- MC-PILCO/`, which shows only the two commits that
  originally brought it in, none after. Every study is additive, built in
  sibling directories, per the project's own rule.

## 2. Where we are now (2026-07-24, ~4 months in)

- **69 commits, 9 contributors** across the team.
- **5 independent studies**, each a full fork of the baseline extended for one
  research question:

  | Study | Directory | Question | Python LOC |
  |---|---|---|---|
  | Baseline | `mc-pilot/` | Reproduce ground-target throwing | 8,666 |
  | 1 | `mc-pilot-elevated/` | Elevated release heights | 12,162 |
  | 2 | `mc-pilot-pybullet/` | Real arm physics (torque control, noise, multi-arm) | 22,768 |
  | 3 | `mc-pilot-pb-elevated/` | Elevated + real arm physics combined | 10,416 |
  | 4 | `mc-pilot-wind/` | Wind-aware vs blind policy | 3,943 |
  | 5 | `mc-pilot-pybullet-yolo/` | Vision-driven target estimation | 11,116 |

  (Reference point: the vendored upstream itself is 6,980 lines — the project
  has added roughly 10x that in new simulation, control, and test code.)

- **This session's focus, `mc-pilot-pybullet/` (Study 2), is now the most
  hardware-relevant branch of the project**: it replaced the original
  KUKA/generic-arm target with the lab's actual **Kinova Gen3 7-DOF**, under
  real torque-mode dynamics (gravity + payload compensated control, not
  kinematic playback).
- **51 automated tests** in `mc-pilot-pybullet/tests/`, all passing — covering
  torque/velocity feasibility across all three trajectory phases (windup,
  throw, follow-through), not just release-instant checks.
- **52 real experiment output directories** (`results_*`) in that variant
  alone — actual training/eval runs, not placeholders.

## 3. What changed in this session specifically (Gen3 overhead throw)

Chronologically, the real engineering arc, each step forced by evidence:

1. Started from a low toss/lob motion — physically wrong for this arm (Gen3
   wrist torque is only 9 Nm), confirmed by a direct sweep showing range falls
   off monotonically as launch angle increases from flat, i.e. any
   distance-scoring search collapses to a near-horizontal "push."
2. Rebuilt the pose search to optimize release state directly under real
   torque/velocity limits (`search_release_state`), found a genuine overhead
   throw — release ~1.1 m, whip motion, not a push.
3. Found and fixed 6 real, distinct bugs in the pipeline via rendering +
   direct computation (not assumption): a corrupted table entry, a
   release-only torque check that missed mid-ramp inertial torque, an arm
   that free-fell after ball release instead of completing its follow-through,
   a follow-through check that validated torque but not joint velocity, a
   landing-distance formula that inflated every range claim ~1.6x, and a
   duration-scan bug that let the follow-through window silently shrink below
   what was requested.
4. Result: **3.15 cm mean landing error, 100% hit-rate <10 cm, 30 unseen
   targets, 10 training trials**, whole-trajectory torque/velocity validated,
   velocity **measured** at physical separation (`getBaseVelocity` readback),
   never assigned.
5. Reproduced the paper's Sec. 6.4 claim that new basket heights need only
   policy re-optimization through the already-trained dynamics model, zero
   new robot trials: adapted to h=0.10/0.20/0.30 m and got 2.95/3.41/3.80 cm
   mean error with **zero additional throws**, matching or beating a full
   retrain done for comparison.
6. Verified, not assumed: independently re-derived the arm's floor-level
   reach via direct forward-kinematics search over the real joint-limit
   envelope (0.864–0.870 m across three independent search methods) against
   the 0.86 m figure used in the "safe throw vs. reach" framing, and traced
   the code path confirming release velocity is read from physics for every
   number reported, with no silent assignment anywhere on that path.

## 4. Current honest status (no overclaiming)

- Simulation stack for the Gen3 overhead throw is **done and verified**:
  correct dynamics, whole-trajectory safety, accurate, generalizes across
  heights with zero new data, and every claim above has been independently
  re-checked against the code rather than taken on faith.
- **Not yet done**: multi-seed confirmation of the sim result (only seed 1
  exists for the ground-height run — `results_kinetic_chain_gen3/1`, no
  seeds 2/3 yet). Any headline number quoted should say "seed 1" until that
  lands.
- **Not started**: anything on the real Gen3 hardware. Everything above is
  simulation only.

## 5. Next steps — hardware execution plan

This is staged, safety-gated, and already scoped (originally written
2026-07-22, still current):

**Phase 0 — Freeze & multi-seed (in progress).** Commit the sim work
(mostly done), run seeds 2/3 of the ground-height training overnight so the
headline number is "X ± y cm across N seeds," not one lucky seed.

**Phase 1 — Hardware-integration check, no arm motion (~0.5 day).** Confirm
`kinova_hardware.py` / `run_hardware_throw.py` consume the new `plan_throw`
trajectory format (windup/throw/follow cubics) at the Kortex controller's real
1 kHz rate, not the old path. Dry-run with the arm e-stopped. Survey the lab
bench for clearance (≥1.25 m overhead, ±33° floor wedge, 0.5–0.9 m from base).

**Phase 2 — Staged bring-up, no ball (~0.5 day).** Full trajectory at 0.25x
speed first, then 0.5x, then 1.0x — comparing real Kortex torque readings
against the sim's predictions at every stage, e-stop within reach, nobody in
the throw wedge. If real torque exceeds 90% of limit at full speed: stop,
re-derive the ceiling, and report the sim-vs-real gap rather than hiding it.

**Phase 3 — Calibration throws (~0.5 day, the paper's ~10-trial protocol).**
Estimate gripper release latency first (high-speed video, ~5 throws). Then 10
calibration throws at fixed speeds, landing measured directly — this real
(speed, landing) data is fed to the GP exactly as the sim trials are.

**Phase 4 — Real MC-PILOT loop (~0.5–1 day).** Train GP + policy on the real
trials, evaluate on 10-15 held-out targets, measure real hit rate. This
number is the actual headline result of the project.

**Phase 5 — Results package (~1 day).** Sim numbers (multi-seed), real
numbers, the sim-vs-real gap, the safety-ceiling finding, and the 6-bug
whole-trajectory-validation story as the methods contribution over
endpoint-only throw checking.

Total estimate: **~3 lab days** once bench access is available, following the
safety gates above at every stage.

## 6. What justifies the work, in one line

Every number in Sections 3-4 is either a direct measurement off real PyBullet
physics or a re-derived/re-verified quantity, and the one previously-unverified
figure in the whole pipeline (the 0.86 m reach estimate) was checked this
session and confirmed accurate to within 1 cm. Nothing in the current
deliverables is asserted without a corresponding computation or test behind it.
