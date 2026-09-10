---
title: "All Results & Metrics — Single Reference"
subtitle: "Every number recorded, generated, or logged in this repo, compiled in one place"
compiled: "2026-08-26"
scope: "Post-fork work (commit 22d0f03, 2026-07-02) through current head (2bbb09e, 2026-08-25), branch kinetic-chain-throw-pose. Pre-fork Group 3 numbers (baseline/elevated/wind/vision studies) are cited separately where they set context — not claimed as this project's own."
---

# How to read this doc

Every section cites its source (file, log, memory record, or commit) so a number can be
traced back. **Anything under §1–§8 is a simulation result unless the section says
"real arm."** No real ball throw has a confirmed clean release + measured landing yet
(see §12, §17) — do not read any §9–§12 hardware number as validated throw accuracy.

Full narrative for §1–§9 is `PROGRESS_REPORT.md`. Paper-vs-project framing is
`COMPARISON_VS_ORIGINAL_PAPER.md`. Hardware/vision updates after 2026-08-10 are compiled
here from `CLAUDE.md` and session memory (not yet in a narrative doc).

---

# Quick-reference table — headline numbers

| # | Result | Value | Config |
|---|---|---|---|
| 1 | NumPy baseline, reliability-fixed | **100% hit, 1.86 cm mean** | 5 seeds × 50 targets = 250 throws |
| 3 | Continuous height generalization | **100% hit, 2.0 cm mean, 5.0 cm worst** | 100 targets, h ∈ [0, 0.45 m] |
| 5 | Gen3 kinematic (idealized) release | **0.34 ± 0.07 cm mean, 1.00 cm worst** | 5 seeds × 30 targets |
| 5 | Gen3 dynamic (torque) release — real hardware config | **1.54 ± 0.09 cm mean, 3.23 cm worst** | 5 seeds × 30 targets |
| 6 | Multi-arm (KUKA/Franka/xArm6/Gen3) | **1.4–2.7 cm mean** | flight-space targets, per-arm |
| 6 | Drag crossover, high-drag regime | **MC-PILOT 0.6–1.7 cm vs. analytical 4–6 cm (3–7×)** | whiffle-ball-class object |
| 7 | Overhead throw, final Gen3 config | **3.15 cm mean, 100% <10 cm** | 30 unseen targets, 10 training trials |
| 7 | Safe range ceiling (whole-trajectory feasible) | **0.83 m** (inside 0.87 m kinematic reach) | Gen3, overhead pose family |
| 8 | Zero-new-trial height adaptation | **2.95 / 3.41 / 3.80 cm** at h=0.10/0.20/0.30 m | 0 new robot trials |
| 9 | Regression test suite | **54 → 65 → 92 → 103 tests**, all passing | growth over the project |
| 10 | Gripper release latency (static/unloaded) | **67.9 ± 6.4 ms** → 10.2 cm undershoot | 1 kHz UDP measurement |
| 10 | Open-loop drift at release | **0.0171 rad → ≈1.0 cm** (later: 0.0318 rad, unexplained 1.9×) | real Gen3, joint-speed streaming |
| 11 | **Gripper TCP offset (found 2026-08-22)** | **0.39 m/s (26%) / 12.0 cm error** — unmodeled until this date | real Gen3 firmware value, exact |
| — | **Real ball throws with confirmed clean release** | **0 (zero)** | as of 2026-08-25 |

---

# 1. NumPy baseline reliability (pre-fork inherited claim, re-audited by this project)

`mc-pilot/`, `PROGRESS_REPORT.md` §1.

**Before fix** (unstratified exploration + default lengthscale 1.0), 5 seeds:

| Seed | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Hit rate / 5 trials | 60% | 20% | 80% | 10% | 10% |

Reported "5/5" in the pre-fork writeup was single-seed luck.

**After fix** (stratified exploration bands + lengthscale scaled to target range): 5/5 seeds
converge. Final evaluation protocol — 5 seeds × 50 fresh, previously-unused targets = 250
throws (`status_update/eval_matrix.md`, exact):

| Seed | Targets | Hit <10cm | Hit <5cm | Mean err | Median | P95 | Max |
|---|---|---|---|---|---|---|---|
| 1 | 50 | 100% | 100% | 1.8 cm | 1.7 cm | 3.2 cm | 3.8 cm |
| 2 | 50 | 100% | 100% | 1.9 cm | 1.9 cm | 2.9 cm | 3.3 cm |
| 3 | 50 | 100% | 100% | 1.8 cm | 1.7 cm | 3.2 cm | 3.7 cm |
| 4 | 50 | 100% | 100% | 1.8 cm | 1.7 cm | 3.1 cm | 3.6 cm |
| 5 | 50 | 100% | 100% | 2.0 cm | 2.0 cm | 3.2 cm | 3.7 cm |
| **All** | **250** | **100.0%** | **100.0%** | **1.86 cm** | **1.77 cm** | **3.17 cm** | **3.81 cm** |

This is the reliability-corrected baseline everything else is measured against.

PyBullet-arm side of the same audit: training destabilized mid-run on both seeds tried
(cost blew up 10–30×) from a missing arm–ball collision guard after release; GP
lengthscale collapsed 250 → 8.4, noise 0.10 → 0.006 as it absorbed the bad contact
points. Fixed with a 2-line collision-disable; cost/hit-rate curves clean afterward
(figs 4–5, no re-run numeric table beyond the plots).

---

# 2. Lengthscale rule — sensitivity, not a constant

`PROGRESS_REPORT.md` §2. No new numeric table — qualitative/plot-only finding. The
inherited "ℓs ≈ 0.15 × target_range" rule does **not** hold as stated: the pre-fork
group's own configs use ratios 0.43–1.00, and their quoted sensitivity S ≈ 0.066
corresponds to 0.43×range, not 0.15×. Do not cite 0.15 as derived.

---

# 3. Variable and generalized basket heights

`PROGRESS_REPORT.md` §3.

| Config | Hits | Mean error |
|---|---|---|
| h = 0.25 m (separate policy) | 10/10 | 2–4 cm |
| h = 0.45 m (separate policy) | 9/10 | 2–4 cm |
| **Single generalized policy**, target = (Px, Py, h), 100 fresh targets, h ∈ [0, 0.45 m] continuous | **100/100** | **2.0 cm mean, 5.0 cm worst**, no error trend vs. height |

---

# 4. Kinova Gen3 integration (kinematic mode, assigned velocity)

`PROGRESS_REPORT.md` §4. 9/10 hits on both seeds tried, 2–3 cm error. Measured
reachable envelope: **~1.0 m/s** end-effector speed on-axis, **0.67–0.87 m** reachable
band. This is a *kinematic*-mode number (ball velocity assigned, arm cosmetic) —
superseded by §5's real torque-controlled release.

---

# 5. Real torque-controlled release (first physically-real release in the project)

`PROGRESS_REPORT.md` §5, `mc-pilot-pybullet/results_dynamics_validation/`.

5-seed × 30-fresh-target real-physics evaluation:

| Release mode | Mean error | Worst |
|---|---|---|
| Kinematic (idealized) | **0.34 ± 0.07 cm** | 1.00 cm |
| **Dynamic (torque) — real hardware configuration** | **1.54 ± 0.09 cm** | 3.23 cm |

Per-seed dynamic-mode range: 1.42–1.67 cm — all five seeds land inside a
theoretically-derived floor (tracking scatter 0.06 m/s × flight-per-speed slope
0.29 m/(m/s) = predicted 1.4–1.8 cm), i.e. this is the irreducible scatter of a 50 Hz
controller, not a modelling residual.

Three systematic biases found and fixed getting here:
1. **Off-axis unreachability** — honest off-axis speed ceiling is **0.61 m/s** (not the
   ~1.0 m/s on-axis figure from §4); nothing beyond ~15° azimuth was reachable before
   the flight-space target-sampling fix.
2. **Training-cost belief trap** — trained policy showed **17–28% excess speed** vs.
   true-physics-optimal on every target before the particle-model start-point fix
   (empirical mean release position vs. nominal), unmoved by 2.5× more training.
3. Zero-amplitude windup bug (found via frame extraction, not numbers).

---

# 6. Multi-arm generalization, drag crossover, noise dose-response

`PROGRESS_REPORT.md` §6, `mc-pilot-pybullet/results_generalization/`.

| Arm | Mean error |
|---|---|
| KUKA iiwa7 / Franka Panda / xArm6 / Kinova Gen3 | **1.4–2.7 cm**, all four |

**Object generalization** (payload mass/size, no retraining): flat ~1.4 cm across
30–150 g and 2–4.5 cm ball diameter (drag negligible at these speeds — controller
compensates measured payload mass directly; flagged as not expected to hold at higher
release speeds).

**Drag-regime comparison** (MC-PILOT vs. paper's analytical Eq. 13 baseline):

| Regime | Analytical baseline | MC-PILOT |
|---|---|---|
| Low drag (<1% of gravity, tennis-ball-class) | near-optimal | learning adds little |
| High drag (~19% of gravity, whiffle-class) | 4–6 cm | **0.6–1.7 cm (3–7× better)** |

**Noise dose-response** (n = 50/condition): symmetric zero-mean noise degrades
accuracy monotonically and is fundamentally uncompensatable by the GP; only biased
noise (velocity slip, timing jitter) produces a learnable aware-vs-naive gap.

---

# 7. Overhead throw redesign — honest range ceiling and final accuracy

`PROGRESS_REPORT.md` §7, `status_update/HANDOFF.md` (2026-07-23/27 entries).

- **Safe range ceiling with whole-trajectory feasibility enforced: 0.83 m** — inside
  the arm's own 0.87 m kinematic reach. (Earlier, pre-follow-through-check number was
  a bug-inflated ~1.07 m that actually landed at 0.67 m — landing-distance formula bug,
  fixed.)
- **Final real-physics accuracy: 3.15 cm mean, 100% hit <10 cm**, 30 unseen targets,
  10 training trials, release speed 1.16–1.49 m/s (scales with target distance, not
  saturated).
- Pose table geometry (23 azimuth entries, FK'd, `status_update/HANDOFF.md` 2026-07-27):
  release at r = 0.035 m, z = 1.137 m, speed **1.628 m/s**, elevation **5.0°**, range
  **0.800 m**, azimuth −33°…+33°, elbow (−1.396) and wrist (−1.222) both saturated at
  their velocity limits, roll joints exactly 0.
  - Re-searched later against the real floor (`--floor_z -0.433`, base-plate-relative):
    release state came back **bit-identical** (max|Δq| = max|Δq̇| = 0, same 1.6281 m/s),
    only the range figure moved: **0.7999 m → 0.9355 m** (`CLAUDE.md`, corner-solution
    explanation — elbow/wrist already saturated).
- Hardware precheck torque margin on the real plan: **8.1 / 39.0 Nm (21%)**.
- Follow-through safety (found unchecked, then fixed): was silently commanding **3.2×
  torque / 1.9× velocity** limits after release before whole-trajectory checking existed.

---

# 8. Zero-new-trials height adaptation (paper's Sec. 6.4 claim, reproduced)

`PROGRESS_REPORT.md` §8, `status_update/HANDOFF.md` (2026-07-23).

| Height | Mean error | Hit <10cm | New robot trials |
|---|---|---|---|
| h = 0.10 m | 2.95 cm | 100% | 0 |
| h = 0.20 m | 3.41 cm | 100% | 0 |
| h = 0.30 m | 3.80 cm | 100% | 0 |
| Ground baseline (§7, for comparison) | 3.15 cm | 100% | — (original 10) |
| Full 9-D height-conditioned retrain (comparison run) | 3.63 cm | — | full new training run |

Adapted policies match or beat both the ground baseline and the full retrain, at zero
marginal robot trials.

---

# 9. Codebase hardening / regression verification

`PROGRESS_REPORT.md` §9, `status_update/HANDOFF.md`, `CLAUDE.md`.

- Test suite growth: **54 → 65** (2026-07-27 hardening pass) **→ 92** (post-refactor,
  `PROGRESS_REPORT.md` §9) **→ 103** (current, `CLAUDE.md`, ~20 s runtime, no GPU).
- Release-logic refactor (`OptimizedReleaseSolver` extraction) reproduced every §5–§8
  sim result to **max abs diff 0.000e+00** on every landing/error/speed field — a
  reliability pass, not a results change.
- Sim ↔ hardware planner agreement: **1e-12** (`tests/test_hardware_planner.py`, 5
  targets across the wedge — release pos, q, qd, all three cubic segments).
- Dry-run 1 kHz pacing test (never run against real Kortex backend at the time):
  **1000 Hz held over 56,682 ticks, worst tick 0.16 ms late**.
- Five real defects found and fixed in the hardware planning path before it matched
  the sim-validated §7 throw (hardcoded 35° fallback angle silently printing
  `PRECHECK: PASS`; phase timings ~2× too fast; release-height box excluding the real
  overhead release; a too-tight trajectory-duration cap; silent velocity clamping
  instead of failing).

---

# 10. Real hardware bring-up (2026-08-07 → 2026-08-10 initial pass)

`PROGRESS_REPORT.md` §10, `CLAUDE.md`. Real Gen3 at `192.168.1.101`.

| Measurement | Value | Consequence |
|---|---|---|
| Gripper release latency (static, unloaded, 1 kHz UDP) | **67.9 ± 6.4 ms** | 10.2 cm undershoot at 1.498 m/s release — 3.5× the entire §5–§8 sim accuracy budget |
| Control loop ceiling, `SendJointSpeedsCommand` (`SINGLE_LEVEL_SERVOING`) | **40 Hz documented ceiling**, not 1 kHz | any earlier "1 kHz" claim was measuring the dry-run loop, not the arm |
| Open-loop drift at release (joint-speed streaming) | **0.0171 rad** | ≈**1.0 cm** landing error — inside the 2.89 cm sim accuracy budget |
| Joint-angle reporting | Joint 3 read **247.37°** against its own ±2.57 rad limit | Kortex reports all joints on [0°,360°); homing guard fixed to wrap continuous joints (0,2,4,6) shortest-path, limited joints (1,3,5) directly |
| Confirmed hard limits (arm firmware, float32) | `qd_max` = 1.3963 / 1.2218 rad/s; `tau_max` = 39/39/39/39/9/9/9 Nm; accel limit 5.20 rad/s² | planned throw peaks at 34% of the accel limit |
| Two-master hazard | observed live — arm flipped to `ARMSTATE_SERVOING_MANUALLY_CONTROLLED` on web-UI touch during streaming | hard check added: refuse to stream unless `SERVOING_READY` |
| Precheck bug found before risk | was running **without gravity compensation** | fixed before first real stream |
| URDF mass audit | found and fixed a phantom **3 kg** mass | — |

Every non-destructive stage (read-only, gripper, full-speed joint-stream) validated
live at this point; no ball in the gripper yet.

---

# 11. Gripper TCP offset — the accuracy blocker (found 2026-08-22)

Memory: `project_gripper_tcp_offset_blocker.md`; `CLAUDE.md`.

**Finding**: the ball's release point/velocity model had always assumed a
zero-length end effector (`profile.ee_link`, the bare wrist flange) in both sim and
hardware. The real Robotiq 2F-85's reach was never modeled.

**Exact values, read from arm firmware** (`ControlConfig.GetToolConfiguration()`,
read-only, authoritative — not measured/estimated):

| Quantity | Value |
|---|---|
| `tool_transform` | (0, 0, 0.12) m, zero rotation |
| `tool_mass` | 0.831 kg (matches a real 2F-85 — deliberately configured) |
| Angular velocity at trained release state | 3.25 rad/s |
| **Velocity delta** (`v_true = v_flange + ω × r_offset`) | **0.39 m/s — 26.0% of the 1.5 m/s release speed** |
| **Position delta** | **12.0 cm** |
| Independently measured flange-vs-Cartesian gap (cross-check) | 12.25 cm — matches |
| `GetMeasuredCartesianPose` Euler convention (`theta_x/y/z`) | confirmed intrinsic XYZ degrees, FK-vs-Kortex agreement **1.23°** |

This dwarfs every other known error term (gripper latency ~1 cm compensated, 25 ms
quantization ~3.7 cm, open-loop drift ~1 cm) by an order of magnitude. **Fix not yet
coded into `release_solver.py` as of this compilation (2026-08-26)** — this is why no
real ball throw has a validated landing measurement.

---

# 12. Gripper release-during-streaming bug + first real ball throws (2026-08-22)

Memory: `project_gripper_release_streaming_bug.md`, `project_first_ball_throws.md`.

**Bug**: `SendGripperCommand` is silently ignored by the arm's embedded controller for
as long as `SendJointSpeedsCommand` streams continuously — 0% motion, no exception, no
elevated tick latency. A dedicated second TCP session for gripper writes was tried and
explicitly rejected by the arm (`KServerException ERROR_DEVICE/SESSION_NOT_IN_CONTROL`).

**Fix**: `GRIPPER_RELEASE_PAUSE_S = 0.20 s` — a gap in the *same* session's joint-speed
stream at the release instant. Static travel-vs-pause sweep (no throw inertia):

| Pause | Gripper travel (from 99% closed) |
|---|---|
| 100 ms | 83.8% closed |
| 150 ms | 76.0% closed |
| 200 ms (shipped) | 69.0% closed |
| 300 ms | 51.5% closed |

Safety check: worst-case joint-limit margin **76.4°** at a 300 ms pause (200 ms shipped
is conservative). Verified **2/2** on real arm (empty gripper): both fully open (0.87%)
after, no fault.

**First real ball throws, same day, `speed_scale` 0.15 → 1.00** — executed, but **not
valid accuracy data**: TCP offset (§11) uncorrected, and these throws predate the fix
above, so the gripper release itself was never confirmed clean (whatever left the hand
did so via arm momentum, not a confirmed OPEN command).

Other measurements from this session:
- Gripper stall on a real object (tennis ball): **58.08% closed**, position and
  velocity both dead stable — legitimate motor stall against the object, not a bug.
- Gripper latency re-measured **with a loaded ball**: onset **73.2 ± 10.3 ms**
  (statistically indistinguishable from the 67.9 ± 6.4 ms static/unloaded figure);
  separately, `t_clear` (command → ball-free at 50% finger travel) = **391.6 ± 9.9 ms**
  — not a release-delay figure, don't reuse it as one.
- `ROBOT_IN_FAULT` event, root-caused to a power-supply issue (reproduced
  independently via the arm's own manual controller — confirms not application-level).
- `--wrist_roll_offset_deg` flag verified free of throw side-effects: +90° test gave
  identical release speed/position/velocity, only joint 6's own windup travel changed
  (0.00 → 0.10 rad/s peak, well inside its 1.22 rad/s limit).
- Open-loop drift re-measured after the gripper-loop fix: **0.0318 rad** — ~1.9× the
  2026-08-10 figure (0.0171 rad), still within budget but unexplained, noted not
  resolved.

---

# 13. Camera extrinsic calibration (2026-08-22, laptop-mounted D435i — not final mount)

Memory: `project_camera_extrinsic_d435i.md`.

| Quantity | Value |
|---|---|
| Position T_B_C (m) | x=1.161, y=0.101, z=1.166 |
| Quaternion (x,y,z,w) | [0.7015, 0.7118, −0.0234, 0.0238] |
| RPY (deg) | roll=180.0, pitch=3.8, yaw=90.8 |
| Reprojection error | 0.15 px (D435i), 0.17 px (wrist camera) |
| Board origin corner, base frame (m) | (1.358, 0.142, −0.575) |

Method: `calibrate_via_wrist_camera.py` (arm FK + wrist camera + shared ChArUco board)
— the adopted result, preferred over the tape-measured `calibrate_camera_extrinsics.py`
path (see memory `feedback_extrinsic_calibration_method.md`). Caveat: board detection
was marginal quality (16–19 px marker edges, below the 35 px "robust" bar) — treat
absolute position as rough even after the FK cross-check. Belongs to the temporary rig,
not the final camera mount.

---

# 14. Release-pose search feasibility statistics (ablation run, 2026-08-22)

`paper_icra2027/results/ablation_kinova_gen3_dyn.json`,
`ablation_franka_panda_dyn.json`, `timing_search_gen3.log`, `timing_train_gen3.log`.

| | Gen3 (`kinova_gen3_dyn`) | Franka Panda (`franka_panda_dyn`) |
|---|---|---|
| Candidate postures searched | 69,505 | 72,917 |
| Static-feasible | 57,351 | 72,917 (100%) |
| Windup-path failures | 242,725 | 0 |
| Ramp failures | 66,706 | 0 |
| Follow-through failures | 62,410 | 0 |
| **Full-trajectory-feasible** | 74,230 | 152,844 |
| Best full-trajectory release: land | **0.824 m** | **1.649 m** |
| Best full-trajectory release: speed | **1.628 m/s** | **3.272 m/s** |
| Best full-trajectory release: elevation | 5° | 10° |
| Search wall time | 1997.5 s (~33 min) | 1744.9 s (~29 min) |

Panda's stronger actuators (87 Nm vs. Gen3's 9 Nm wrist) hit **zero** windup/ramp/
follow-through failures — the whole-trajectory check never binds for Panda, unlike
Gen3 where it eliminates the large majority of otherwise-static-feasible candidates.

Separately timed runs:
- `find_throw_pose.py --mode overhead` (Gen3): **28 min 27 s** wall (1697.85 s user).
- `train_mc_pilot_pb_arm.py`, 10 trials, `--opt_pose`: **17 min 44 s** wall
  (1064.94 s user), final trial cost **0.0001**.

---

# 15. Settled negative results (kept deliberately, not discarded)

`PROGRESS_REPORT.md` §11, `CLAUDE.md`.

| Ablation | Result |
|---|---|
| `--residual_physics` (analytical Eq. 13 speed as policy residual) | **worse** than plain MC-PILOT |
| `--residual_dynamics` (gravity subtracted as GP mean function) | within noise of plain MC-PILOT |
| GPU vs. CPU | **GPU slower** — confirmed twice; PyBullet is CPU-only regardless, GP tensors too small to benefit |
| Inherited wind study (Study 4, pre-fork, underpowered) | blind 2-D GP beats wind-aware 4-D GP at 15 trials — margin 1–2/15 throws at one seed; "~40+ trial crossover" is asserted from an Nb^(1/d) argument, never run |

---

# 16. Paper (Turcato et al., arXiv:2502.05595) vs. this project — key deltas

Condensed from `COMPARISON_VS_ORIGINAL_PAPER.md` (full framing there).

| Axis | Paper (real Panda) | This project |
|---|---|---|
| Release pose | fixed analytically (Eq. 31), only aim+speed learned | **searched** — direction-constrained LP over full release state, whole-trajectory feasibility |
| Multi-seed reliability | single-run real results | 5-seed audit found "5/5" was single-seed luck (true rate 1–2/5); root-caused and fixed |
| Real accuracy | ~5–15 cm median landing error, real Panda | 1.86 cm (baseline sim), 3.15 cm (Gen3 overhead sim) — **not comparable, no real Gen3 throw exists yet** |
| Arms | 1 (Panda) | 4 in sim (KUKA/Franka/xArm6/Gen3), all 1.4–2.7 cm |
| Height adaptation | 1 qualitative real demo | quantified 3-height sim reproduction (2.95/3.41/3.80 cm) + a continuous-height single policy (100%, 2.0 cm mean) |
| Drag/object generalization | low-drag objects only, no crossover analysis | explicit low↔high drag sweep, 3–7× crossover finding |
| Delay handling | learned distribution `t_d ~ U(â,â+b̂)`, Bayesian-optimized | direct 1 kHz measurement, 67.9±6.4 ms, wall-clock-compensated |
| Real ball thrown | yes, headline result | **no — zero confirmed clean-release real throws as of 2026-08-25** |

---

# 17. In-progress work with no results yet (2026-08-25, uncommitted/latest commits)

Not yet run against sim or hardware — implementation only, listed for completeness
since the user asked for everything recorded:

- `mc-pilot-pybullet/perception/stereo.py`, `ball_track.py`, `trajectory.py` — rectified
  IR stereo rig, median-background candidate detection, left/right pairing, ballistic
  kinematics + descending-root impact solve for the D435i pair. Unit-tested
  (`test_stereo.py`, `test_ball_track.py`, `test_trajectory.py`) but no detection-rate,
  calibration-error, or landing-measurement numbers logged yet.
- Plan: `docs/superpowers/plans/2026-08-25-ball-tracking.md`.

---

# 18. Where this stands (as of 2026-08-26)

- **Top blocker**: gripper TCP offset fix (§11) not yet folded into
  `release_solver.py`. Until it is, no landing measurement from a real throw is
  meaningful.
- **Confirmed-clean real throws**: zero. The 2026-08-22 throws (§12) are momentum-
  released, not confirmed-OPEN-command releases, and predate the streaming-bug fix.
- Every accuracy number in §1–§9 is a **simulation** result on the exact configuration
  that was carried to hardware (bit-identical re-verified after the §9 refactor).
- Vision/ball-tracking track (§17) is mid-implementation, no results yet.

---

*Sources: `PROGRESS_REPORT.md`, `COMPARISON_VS_ORIGINAL_PAPER.md`, `CLAUDE.md`,
`status_update/HANDOFF.md`, `status_update/eval_matrix.md`,
`paper_icra2027/results/*.json`, `paper_icra2027/results/*.log`, session memory
(`project_camera_extrinsic_d435i.md`, `project_gripper_tcp_offset_blocker.md`,
`project_first_ball_throws.md`, `project_gripper_release_streaming_bug.md`), and
`git log` on branch `kinetic-chain-throw-pose` as of commit `2bbb09e` (2026-08-25).*
