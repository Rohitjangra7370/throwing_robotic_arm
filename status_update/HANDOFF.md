# Session Handoff — MC-PILOT Throwing Arm

_Last updated: **2026-08-26** (supersedes the 2026-07-27 handoff, kept below the divider).
This session = **documentation only, Task 11 of the ball-tracking plan**
(`docs/superpowers/plans/2026-08-25-ball-tracking.md`, spec at
`docs/superpowers/specs/2026-08-25-ball-tracking-design.md`). No code changed. Tasks 1-10 of that
plan — the stereo-IR ball-tracking and landing-measurement pipeline — were implemented and
committed in prior sessions (commits `78cbb84`..`2d7ae2f`); this session wrote up `CLAUDE.md`,
`HARDWARE_RUNBOOK.md`, and this entry, and explicitly did **not** attempt Task 11 Step 3 (the
real-throw acceptance gate) because the RealSense D435i is currently physically unplugged and
no arm session was available. Nothing here should be read as new evidence — it is a writeup of
what Tasks 1-10 already established, plus an explicit statement of what is still unverified._

## 0. READ FIRST — REAL vs ASSIGNED vs NOT-WORKING (this session)

| Claim / artifact | Status |
|---|---|
| `perception/stereo.py`, `perception/ball_track.py`, `perception/trajectory.py` (triangulation, RANSAC ballistic association, Gauss-Newton fit + impact solve) | **REAL, but verified SYNTHETICALLY ONLY.** 147 tests pass; end-to-end synthetic parabola (projected through the real measured IR intrinsics and 49.9448 mm baseline) recovers the landing point to 0.18 mm; Gauss-Newton at 0.15 px pixel noise, 40 frames, 30 seeds, averages 0.47 mm; RANSAC separates 40/40 true detections from 12 injected arm-like outliers. No camera, no arm, no real image was involved in any of these numbers. |
| `perception/ir_capture.py` (dual-IR ring-buffer recorder) | **NOT-WORKING / NEVER RUN.** The live-capture path has never executed against the real camera. The D435i was unplugged before it could run once. |
| `record_throw_ir.py` | **NOT-WORKING.** Has never captured anything — depends on `ir_capture.py`'s untested live path. |
| `measure_landing.py` | **REAL against a synthetic recording only.** Runs correctly end-to-end on a synthetic fixture; has never been pointed at a real recording because none exists. |
| `achieved_fps` / the ~44-usable-frames-at-90fps error budget | **ASSIGNED, unvalidated.** A reasoned estimate from the D435i's advertised 90 fps IR streams, not a measurement — nothing has streamed from the real sensor yet. |
| `IRRecorder` exposure/emitter defaults (`exposure_us=2000`, `emitter=True`) | **ASSIGNED, unvalidated.** `tune_ir_exposure.py` exists to decide these via an A/B and has not been run. Must be run first on run day (see `HARDWARE_RUNBOOK.md` §6) before any real throw is recorded. |
| Comparison against the existing static `ball_detector.py` + `ray_plane.py` path on a real throw | **NOT DONE.** This is Task 11 Step 3 / design spec §7.3, the real-throw acceptance gate. Explicitly not attempted this session — needs the overhead camera mount, a re-measured `T_B_C`, and the robot arm, none of which were available. Do not simulate or approximate this step; it either happens on real hardware or it hasn't happened. |
| `T_B_C` (camera-to-base extrinsic) | **STALE.** Still the 2026-08-22 laptop-held-rig calibration, explicitly not the final mount, with marginal board detection quality throughout. Absolute accuracy of any future landing measurement is bounded by this, not by the vision pipeline above. |
| Everything touching the real camera or the real arm, for this pipeline | **NONE OF IT.** As of this session, zero frames from the real D435i and zero real throws have gone through any part of `perception/`. Every millimetre figure anywhere in this pipeline's documentation is a synthetic/relative accuracy, not an absolute one. |

## 0a. Open items (priority)

1. **Mount the D435i on the final overhead rig** and re-measure `T_B_C` via
   `calibrate_via_wrist_camera.py` (the adopted FK + shared-marker path) — the current extrinsic
   is the superseded laptop-rig one. This blocks Task 11 Step 3 and any absolute-frame landing
   number.
2. **Run `tune_ir_exposure.py`** on run day, before the first real throw, to settle exposure and
   emitter (spec §9) — record the chosen values in `HARDWARE_RUNBOOK.md` §6, which currently has
   a blank table waiting for them.
3. **Run `ir_capture.py`'s live path for the first time** via `record_throw_ir.py` — everything
   downstream of it has only ever seen synthetic data.
4. **Run the Task 11 Step 3 acceptance gate** (design spec §7.3): for at least 5 low-bounce real
   throws, compare `measure_landing.py`'s first-contact point against the static
   `ball_detector.py` + `ray_plane.ball_center_on_plane` resting measurement, and report both
   numbers and their difference. Expected agreement is a couple of centimetres; a consistent
   directional offset would point at `T_B_C`, not the fitter — report the numbers, do not adjust
   anything to force agreement.
5. Only after 3-4 does this pipeline have any standing to feed a real landing error back into the
   GP ("close the loop" in `HARDWARE_SETUP.md`) — nothing before that point is real-world evidence.

---

_Last updated: **2026-07-27** (supersedes the 2026-07-23 handoff, kept below the divider).
This session = **no new sim results, no new claims** — a codebase reliability pass ahead of
hardware bring-up tomorrow, triggered by Deepak's "let's plan hardware experiments" reply to
email 5. Audited every layer, found that **the hardware path would have executed a completely
different throw from the one email 5 describes**, and fixed it plus four more defects. Also
wrote the hardware cold-start plan (arm + D415). Suite 54 → **65 passing**. Sim results
verified **bit-for-bit unchanged** by the refactor._

## 0. READ FIRST — REAL vs ASSIGNED vs NOT-WORKING (this session)

| Claim / artifact | Status |
|---|---|
| Sim accuracy numbers from email 5 | **UNCHANGED AND RE-VERIFIED.** The release-logic refactor reproduces the 30-throw eval to `0.000e+00` max abs diff on every landing / error / speed field. Nothing about the reported results moved. |
| `run_hardware_throw.py` before this session | **WAS PLANNING THE WRONG THROW.** It called `plan_throw` with a hardcoded 35° launch angle, no pose table and no overrides → a legacy IK+`pinv` near-horizontal release, i.e. exactly the "places the ball" motion email 5 argues against. It printed `PRECHECK: PASS` while doing so. Nothing in the 54-test suite compared the two planners. |
| `simulation_class/release_solver.py` (new) | **REAL.** Sim and hardware now call one `OptimizedReleaseSolver`. Verified equal to **1e-12** on 5 targets across the wedge (release pos, q, qd, all three cubic segments). Duplicate body deleted, not left dead. |
| Hardware precheck | **REAL, hardened.** Fails closed on any active velocity clamp; whole-trajectory inverse dynamics vs `tau_max`. Real plan peaks at **8.1 / 39.0 Nm (21%)**. |
| 1 kHz control loop | **REAL in dry-run only.** Absolute-deadline pacing holds **1000 Hz over 56,682 ticks, worst tick 0.16 ms late**. Never run against a real Kortex backend. |
| Everything hardware-side | **STILL UNTESTED ON THE ARM.** `kortex_api` is not installed here; `_KortexBackend` method names remain unverified against any real release. Dry-run only. |
| `franka_panda_dyn` "cannot train" (2026-07-22 note) | **STALE, CORRECTED.** Constructed and stepped it: 9 DOFs, `dof_ids` 0..6, padding exact. The claim was wrong, not the code. |

## 0b. The five defects, in the order they were found

1. **Hardware planned a different throw** (above). Root fix: extract the release solver so
   there is one implementation, not two. A copy would have drifted again.
2. **Phase timings came from `profile.timing` (0.4/0.8) instead of the trained
   `cfg["T_W"]/["T_R"]` (0.5/1.6)** — a throw phase under half the trained duration, so
   roughly **2× commanded peak joint velocity**. Would have shown up on the arm as an
   unexplained infeasibility or a limit violation.
3. **The hardcoded release box (z ≤ 0.9 m) excluded the overhead release at z ≈ 1.137 m**, so
   `throw` would have refused every valid plan. Box is now derived from the pose table's own
   FK release locus — which makes the check meaningful rather than arbitrary. Test confirms it
   still rejects a release displaced by 0.5 m.
4. **`max_traj_seconds = 8.0` refused the real trajectory** (8.5 s at 1.0×, ~57 s at the 0.15
   rehearsal scale). Raised to 180 s; the cap still exists to catch a runaway plan.
5. **Precheck silently clamped velocity** instead of failing — safe for the arm, wrong for the
   throw: a clamped joint releases slower than the policy asked and the ball lands short with
   nothing in the logs.

## 0c. Verified geometry of the shipped table (FK'd this session, not quoted)

All 23 azimuth entries: release at **r = 0.035 m, z = 1.137 m** (essentially straight above the
base), **1.628 m/s**, **5.0° elevation**, **0.800 m** range, azimuth **−33°…+33°**, elbow
(−1.396) and wrist (−1.222) both **saturated at their limits**, roll joints exactly 0. The low
elevation is not a bug — range decreases monotonically 0°→70° at these speeds, so the *height*
buys the distance. This is the strongest form of email 5's argument.

## 0d. New / changed files

New: `simulation_class/release_solver.py`, `tests/test_hardware_planner.py` (11 tests),
`docs/superpowers/plans/2026-07-27-hardware-cold-start.md`, `CLAUDE.md`.
Changed: `simulation_class/model_pybullet.py` (delegates, −174 lines), `robot_arm/
arm_controller.py` (public `inverse_dynamics()`, reused by `_throw_peak_torque_ratio` so there
is one torque path), `robot_arm/kinova_hardware.py` (precheck + 1 kHz loop + `last_exec_stats`),
`run_hardware_throw.py` (planner rewrite, `--opt_pose`, `--u_cap`, derived release box),
`eval_adapted_height.py` (`--opt_pose`, so pre-`opt_pose` checkpoints stay evaluable).

## 0e. Open items

1. **`kortex_api` not installed** — install the Kinova wheel and verify every method name in
   `_KortexBackend` against that exact version. First task at the lab.
2. **Multi-seed (seeds 2, 3) still not run** — every headline number, including the 3.15 cm and
   all three height-adaptation figures, is **single-seed**. There is no ± to quote. Pure
   compute, no blockers.
3. Hardware cold start: `docs/superpowers/plans/2026-07-27-hardware-cold-start.md`, Phases A–H
   with gates. Phase 0 (code) is done except items 1–2 above.
4. D415 decision pending: side-oblique mount at (0.75, −1.05, 0.90) in base frame, or overhead
   at (0.72, 0, 2.20) if the bench frame can carry it.
5. **Do not trust raw D415 depth for position** — 2% of range = 2–4 cm at 1–2 m, the same size
   as our landing error. Use ray–plane intersection against known plane heights; depth is a
   segmentation gate only. The sim `depth_camera.py` back-projects depth because sim depth is
   exact; that method does not transfer.

---

_Last updated: **2026-07-23** (supersedes the 2026-07-22 handoff, kept below the divider).
This session = pivoted from the frozen-base "kinetic-chain" throw (2026-07-22's honest but
tiny ~12cm range, correctly called "trash"/"collapsing" by the user on watching the video)
to a genuine **overhead throw**: release-state-first search under real hardware constraints,
whole-trajectory torque+velocity validation (windup, throw, AND follow-through — the last of
which had NO check at all until this session and was silently commanding 3.2x torque / 1.9x
velocity limits after release), and — closing the exact open item the 2026-07-22 handoff
flagged — the paper's zero-new-trials height adaptation (Sec 6.4), reproduced and verified
on this throw. Six real bugs found and fixed, in order; final state is committed and clean
(54/54 tests)._

## 0. READ FIRST — this session's result vs the 2026-07-22 baseline

| Claim | Status |
|---|---|
| Overhead release-state search (`find_throw_pose.py --mode overhead`) | **REAL.** Optimizes the release state directly (joint angles+velocities at release) for max landing distance under real 39/9 Nm torque and 1.396/1.222 rad/s velocity limits — no cosmetic pose filters. One state found, propagated across the ±33° wedge by base rotation (uniform by construction, verified 1e-6). |
| Whole-trajectory validation: windup, throw, follow-through | **REAL, all three phases now checked.** Follow-through (post-release deceleration) had zero feasibility checking prior to this session — found via a training crash, then found AGAIN worse (arm was free-falling under gravity because the rollout loop stopped commanding it after release), then found a THIRD time (torque-only follow check missed a duration at 1.88x qd_max while passing torque at 0.95x) while making a diagnostic plot for the status mail. All three fixed; `plan_throw` now raises loudly rather than silently shipping an infeasible trajectory. |
| Landing-distance formula | **Bug found and fixed.** The search's own ranking objective computed `hypot(release_pos) + flight_range`, valid only if release point and throw direction are collinear through the origin — they aren't here. Inflated every "beyond reach" number this session reported by ~1.6x (a claimed 1.07m throw actually landed at 0.67m) and was the root cause of a 39.7cm systematic policy-speed-saturation bug. Fixed to `hypot(actual_landing_point)`. |
| Honest safety ceiling | With follow-through genuinely enforced, the safe landing distance for this release-state family caps at **0.83m — inside the arm's own 0.86m reach**, not beyond it. Real, quantified trade-off between throwable range and hardware-safe recoverability; presented as a finding, not hidden. |
| Real accuracy (10 real trials, ground) | **3.15cm mean, 100% hit<10cm**, 30 fresh unseen targets, unused seed. Release speed 1.16-1.49 m/s, scales correctly with target distance (not saturated). |
| Height generalization — **paper's zero-new-trials claim (Sec 6.4), reproduced** | Adapted the ground policy to h=0.10/0.20/0.30 by reusing the trained GP verbatim (`load_model_from_log`) and re-optimizing the policy alone — **0 new robot trials per height**. Results: 2.95cm / 3.41cm / 3.80cm mean, all 100% hit<10cm — matches or beats both the ground baseline (3.15cm) and a full 9-D height-conditioned retrain (3.63cm, which needed a complete retraining run). This closes the 2026-07-22 handoff's open item #5 ("Reuse-model height-adaptation... instead of full retrain — not done"). |
| RELEASE_POS anchoring fix | **Tried, applied, found to have ZERO effect — reported as falsified, not as a fix.** Hypothesized a stale particle-model release position was biasing training; fixed it, retrained, got a bit-for-bit identical policy. Traced why: `_optimized_release` overrides that parameter for every real rollout regardless, so the fix never touched the path that mattered. Correct instinct to test and report honestly rather than claim credit. |

## 0b. Full real chain of bugs this session, in the order found

1. Original tables kept targets *inside* the arm's own reach (0.60-0.70m vs ~0.86m) — the 2026-07-22 throw could never prove anything even before the "trash" verdict; the EE could just visit the bin.
2. Release-state-first search found real hardware speed headroom (up to ~2 m/s vs the 2026-07-22 pipeline's 0.66 m/s) by optimizing the release state directly instead of scoring a pose grid.
3. Rotation-built table for uniform azimuth aiming (TossingBot / MC-PILOT Eq.5 formulation) plus a turret-aiming correction for the off-axis release geometry this produces.
4. Release-only torque approximation (checked qdd=0 at the release instant) missed mid-ramp inertial torque and caused 2 real training crashes → replaced with full-trajectory sampling that mirrors `ArmController._throw_peak_torque_ratio` exactly.
5. RELEASE_POS fix — tried, falsified, reported honestly (see table above).
6. **The landing-distance formula bug** (see table above) — the deepest one, since it invalidated every earlier "beyond reach" claim this session and was the true cause of a total policy-speed-saturation failure that looked like a training/calibration problem.
7. Follow-through safety — found three times over (torque-infeasible plan; arm free-falling post-release because the rollout stopped commanding it; then a torque-only check that missed a real velocity violation). All three are now real, tested fixes, not approximations.
8. Height adaptation without new trials — the paper's own claim, reproduced (see table above).

## 0c. What's committed

- `find_throw_pose.py`, `robot_arm/arm_controller.py`, `simulation_class/model_pybullet.py`,
  `train_mc_pilot_pb_arm.py` — commit `b100325` (release-state search, rotation table, turret
  aiming, full-trajectory torque validation, arm-keeps-commanding-after-release fix).
- `train_mc_pilot_pb_heightgen.py`, `eval_heightgen.py`, `eval_adapted_height.py`,
  `adapt_policy_height.py`, `make_heightgen_video.py`, follow-through velocity check + the
  duration-scan fix — commit `3cce8ab`.
- `results_kinetic_chain_gen3_h10/20/30`, `results_kinetic_chain_gen3_hgen`,
  `results_generalization/heightgen_kinova_gen3_dyn.*` — commit `e7a4806`.
- Test suite: 54 passing, including regression tests for the table-direction-alignment bug,
  the Coriolis-at-release bug, the mid-ramp torque bug, and the follow-through
  torque+velocity bug (the last one against the actual shipped table entry, not a synthetic
  fixture).

## 0d. Open items

1. Final wrap-up mail (`status_update/email_update5.md`, drafted, not yet sent — needs
   recipient address and a scope/framing decision from the user).
2. Hardware execution plan written (`docs/superpowers/plans/2026-07-22-hardware-execution-
   plan.md`) — Phase 0 (commit + multi-seed) mostly done this session; Phases 1-5 (driver
   integration check, staged bring-up, gripper-delay calibration, real MC-PILOT loop, results
   package) not started, need real lab time.
3. Height adaptation was only tested up to h=0.30 (H_MAX chosen to leave an 11cm flight span
   at the top of the range — 0.45 would leave only 5.3cm, too thin to be useful). Not tested
   beyond that.
4. The Franka Panda port mentioned in the 2026-07-22 handoff below is untouched this
   session — still "in progress, do not train" per that section.

---


This session = the throwing POSE debugged **from scratch, again** (the 2026-07-21 handoff's
"aimed throw" was still wrong — its LP let the roll/twist joints spin freely, producing
corkscrew motion; user: *"have some common sense... you are using in-link rotational joint"*).
Result: a real, hardware-valid **kinetic-chain** pipeline for Gen3 (axis-perpendicular joints
only, real torque feasibility, measured — not assigned — release), trained end-to-end through
the existing MC-PILOT infra, then an **in-progress port of the same pipeline to Franka Panda**
(numerically verified, new bug found and only half-fixed — see §7, do not train Panda yet)._

## 0. READ FIRST — what is REAL vs ASSIGNED vs NOT-WORKING (this session)

| Claim / artifact | Status |
|---|---|
| Gen3 kinetic-chain pose search (`find_throw_pose.py`): axis-perpendicular joints only (idx 1,3,5) carry velocity, roll/twist (idx 0,2,4,6) frozen, real `calculateInverseDynamics` torque check (endpoint **and** whole windup path) | **REAL, hardware-valid.** Two real bugs found+fixed this session (§2). |
| `throw_pose_table.npy` (23 azimuths, Gen3) | **REAL** — rebuilt via verified base-rotation invariance (1e-6 precision), 23/23 feasible, uniform 12.0cm range. |
| Trained `kinova_gen3_dyn` opt_pose policy (`results_kinetic_chain_gen3/1`, committed `6de454a`) | **REAL, MEASURED dynamic release.** 10 trials, GP MSE ~1e-6, final cost 0.0062–0.0095 (plateaued, not still improving). Video-verified frame-by-frame: real curl→uncurl, no corkscrew. |
| That same video, user's verdict | **Correct complaint, not a bug**: range is 12cm / speed 0.06–0.17 m/s — a geometry ceiling of that release posture, reads as a "drop" not a "throw." Not fixed this session. |
| Panda legacy kinematic policy (`results_mc_pilot_pb_A_franka_panda_flight/1`) rendered via `eval_and_video_long.py` | **ASSIGNED VELOCITY, same flaw class as the old Gen3 corkscrew throws** — frame check shows the arm **never visibly moves** for the whole throw window while the ball is already airborne (`release_ball(set_vel=v_cmd)`, disconnected from real swing). Shown to user with this caveat, not presented as a fix. |
| Panda **kinetic-chain** port (`franka_panda_dyn` profile, generalized `find_throw_pose.py`) | **IN PROGRESS, NOT TRAINED, NOT RENDERED.** Single-azimuth smoke result real (2.06 m/s / 55.8cm). Full 23-azimuth table search launched in background, not confirmed complete. **A real architecture gap found and only half-fixed — see §7. Do not run training for `franka_panda_dyn` until that's resolved or you'll likely hit the same crash.** |

## 1. The re-debug: why the 2026-07-21 "aimed throw" was still wrong

Previous handoff's `_optimized_release` direction-constrained LP froze only the **base** joint's
velocity (`qd[0]=0`) and let every other joint — including the three roll/twist joints
(shoulder-roll idx2, wrist-roll1 idx4, wrist-roll2 idx6) — spin freely to hit the target speed.
Those joints' rotation axes point roughly ALONG their connecting link, not perpendicular to any
swing plane; letting the LP use them produces a visible corkscrew/twisting motion, not a throw.
User caught this by eye, repeatedly, before I did numerically. **Fix**: freeze qd=0 at every
roll/twist index `(0,2,4,6)`, not just the base — in both `find_throw_pose.py::aimed_speed` and
`model_pybullet.py::_optimized_release`'s LP. Static angle of the roll joints stays a free search
parameter (pinning it to literal zero too collapses the three pitch axes to parallel — verified,
kills all off-axis aim).

## 2. Two real bugs found via systematic-debugging (not guessed)

1. **Wrap-to-neutral bug** (`_optimized_release`): the ±2π "wrap to nearest neutral" was applied
   to every joint, but only the base has continuous rotation. Verified it pushed a valid −100°
   elbow target to +260° (physically unreachable), silently collapsing release speed to ~0
   regardless of command. Fixed: wrap only index 0.
2. **Windup-path gravity violation**: release-pose-only static torque feasibility missed that the
   neutral→windup swing itself passes through a worse intermediate gravity configuration
   (verified: shoulder sweeping 22°→90° peaks ~39Nm mid-swing even though both endpoints are
   fine). Fixed: `windup_path_feasible()` samples the whole straight-line path at 85% margin, not
   just the two ends.

## 3. TossingBot-style exploration (scratchpad, superseded, not merged)

Before realizing the existing `plan_throw(monotonic_windup=True)` infra already implements the
curl→uncurl mechanism, built a bespoke script (`scratchpad/tossingbot_style.py`) from
user-specified exact poses (captured via a real PyBullet GUI with per-joint sliders,
`gui_joint_control.py` — built after my own hand-derived FK reconstruction from raw URDF
quaternions repeatedly failed and was abandoned). Fixed along the way: arm free-falling after
release (had stopped calling `arm.step()` post-release — user: *"why arm falls after release??
fix it"*), joint velocities exceeding real limits (rescaled to exactly saturate the tightest
joint: `k=min(qd_max/|dq|)`), and a silent-acceptance bug where a torque-infeasible trajectory
(15% over limit) would have been reported as a result without any hard failure check (added
`RuntimeError` guards). **User's explicit instruction ended this branch**: *"generate movement
same way tossing bot does don't reinvent the wheel okay"* — i.e. use the existing MC-PILOT
`plan_throw`/`opt_posture_table` infra, which already does the equivalent. This script is not
part of the pipeline.

## 4. Rotation-invariance + full Gen3 pose table rebuild

Verified numerically (1e-6 precision) that rotating only the base joint of a serial chain
rotates all downstream release geometry identically — position, velocity direction, speed,
elevation all preserved except horizontal direction. An independent per-azimuth grid search
(reduced-resolution, for tractability) had left gaps at the center azimuths (11/23 feasible).
Rebuilt `throw_pose_table.npy` instead by rotating one verified entry through all 23 azimuths:
23/23 feasible, uniform 12.0cm range — faster and more complete than re-searching.

## 5. Training fix + first real trained result (Gen3)

`train_mc_pilot_pb_arm.py`'s target-sampling anchor used the STALE `profile.default_release_pos`
(0.55, 0.00) instead of the real-FK release point for `opt_pose` table mode (0.696, −0.054) — a
0.15m gap, huge next to the ~12cm range. Fixed: compute `release_xy` from real FK on the table's
az≈0 entry when `opt_posture_table` is active. Smoke test (3 trials) then full run (10 trials,
`kinova_gen3_dyn`, `--flight_targets`) both completed cleanly: GP MSE ~1e-6 throughout, final
trial cost oscillating 0.0062–0.0095 (plateaued from trial 4 onward, not still converging — this
is the ceiling of a 12cm-range pose geometry, not a training failure). Committed on branch
`kinetic-chain-throw-pose` (`6de454a`): `find_throw_pose.py`, `throw_pose_table.npy`,
`train_mc_pilot_pb_arm.py`. Results at `results_kinetic_chain_gen3/1` (uncommitted).

## 6. Video verification loop + the "trash" verdict (correct)

Built `make_trained_kinetic_chain_video.py` — loads the TRAINED policy weights and rolls out
through the real `PyBulletThrowingSystem(opt_posture_table=...)`, no oracle speed, no assigned
velocity. 6 throws, mean err 4.5cm, 100% hit<10cm. **Extracted and viewed actual frames**
(per standing feedback memory — burned by claiming success from numbers alone earlier this
project) — confirmed genuine smooth curl→uncurl, continuous across the release transition, one
clean bin landing shown. **User called it "trash" anyway — correctly.** Re-examined frame-by-
frame at finer granularity: the motion itself is real and continuous (no teleport/glitch), but
speed is 0.06–0.17 m/s over 12cm — visually indistinguishable from placing the ball by hand. This
is the fixed release posture's actual range ceiling at that geometry, not a rendering bug. Not
resolved this session; asking the user to disambiguate ("too small" vs "camera" vs "still looks
wrong") was rejected — user redirected to Panda instead (§7).

## 7. Franka Panda kinetic-chain port — IN PROGRESS, INCOMPLETE

Legacy Panda video (`results_mc_pilot_pb_A_franka_panda_flight/1` via existing
`eval_and_video_long.py`) looked real on paper (mean err 2.1cm, distances 0.66–1.07m) but
frame-checking showed the arm **never visibly moves** for the whole throw — `release_ball
(set_vel=v_cmd)` assigns velocity directly, same "not a real throw" flaw class the user rejected
for Gen3 early this session, just manifesting as zero visible motion instead of a corkscrew.
Flagged honestly, shown to user anyway on request (*"shutup and just render the video and show
me"*), then user asked for the real thing: *"no i want tossing trajectory"*.

Started porting the Gen3 kinetic-chain mechanism:
- **Numerically verified** (not assumed) that Panda has the same alternating roll-pitch-roll
  joint structure as Gen3: at `q_neutral`, joints 2/4/6 (idx1,3,5) sit at **exactly 90°** between
  joint-axis and base→EE vector (pure pitch), joints 1/3/5/7 (idx0,2,4,6) sit at 0–69° (roll-
  like, joint7/idx6 exactly 0° — a pure wrist twist). Same `_ROLL_IDX=(0,2,4,6)` freeze applies.
- Added `robot_arm/robot_profiles.py::franka_panda_dyn` — `control_mode="torque"`,
  `tau_max=(87,87,87,87,12,12,12)` (published Franka Emika limits, franka_ros
  `joint_limits.yaml`), `kp=400/kd=60` (copied from `kinova_gen3_dyn`, **untuned for Panda's
  different mass/inertia** — verify before trusting torque numbers).
- Generalized `find_throw_pose.py` from Gen3-hardcoded to profile-parameterized (`set_robot()` /
  `--robot` CLI arg); default behavior for `kinova_gen3_dyn` unchanged (verified backward
  compatible).
- **Found a real architecture gap doing this**: Panda's URDF has 9 non-fixed DOFs (7 arm + 2
  gripper-finger prismatic joints) vs Gen3's exactly 7. `calculateInverseDynamics`/
  `calculateJacobian` need full-length DOF vectors, not just the 7 controlled joints.
  `ArmController.__init__` already has a hard guard against this (`raise ValueError` if
  `n_dofs != len(joint_ids)` for torque mode) — meaning **torque-mode training was already
  blocked for any URDF with passive extra joints, this just surfaced it.** Fixed **only inside
  `find_throw_pose.py`** via a zero-padding helper (`_pad`/`N_FULL` — exact because the 2 finger
  DOFs land at trailing positions in the full DOF vector, confirmed from URDF joint ordering).
  **`robot_arm/arm_controller.py::ArmController.step()`'s torque branch (and its own
  `calculateInverseDynamics` call, and the constructor's `ValueError` guard) have NOT been fixed
  — training/rollout via `PyBulletThrowingSystem` for `franka_panda_dyn` will almost certainly
  hit the same crash. This must be fixed in `arm_controller.py` before `train_mc_pilot_pb_arm.py
  --robot franka_panda_dyn` can run.**
- Single-azimuth (az=0) smoke test of the generalized search succeeded: speed=2.06 m/s,
  range=55.8cm, elev=50°, both `qd_max` entries saturated — a real throw-scale result (Gen3's
  ceiling at its posture was 12cm).
- Full 23-azimuth table search (`find_throw_pose.py --robot franka_panda_dyn --out
  franka_panda_dyn_throw_pose_table.npy`) completed: **23/23 feasible**, uniform 55.8cm range,
  2.06 m/s, elev 50°, both `qd_max` saturated, at every azimuth −33°..+33° (independently
  searched per azimuth, not rotation-built — the uniformity is a genuine finding, consistent with
  rotation-invariance holding for Panda too, not assumed this time). 5m30s wall time.

## 8. New/changed files this session

`find_throw_pose.py` (rewritten: all-roll-frozen LP, `static_feasible`/`windup_path_feasible`
real torque checks, `is_natural_posture` elevation cap, generalized to `set_robot()`/`--robot`,
`_pad`/`N_FULL` DOF padding); `throw_pose_table.npy` (rebuilt, Gen3, 23/23 feasible);
`train_mc_pilot_pb_arm.py` (`release_xy` real-FK anchoring fix for `opt_pose` mode);
`robot_arm/robot_profiles.py::franka_panda_dyn` (new); `make_trained_kinetic_chain_video.py`
(new — loads real trained weights + real rollout, no oracle/assigned speed);
`gui_joint_control.py` (real PyBullet GUI, per-joint sliders, used for direct pose
specification); `tests/test_throw_pose_search.py`, `tests/test_optimized_release_freezebase.py`
updated; `tests/test_kinetic_chain_rollout.py` — **STALE**, references the old table structure,
will likely fail against the rebuilt table, not yet updated. Scratchpad (not committed):
`tossingbot_style.py`, `widen_release.py`.

## 9. Open items (priority)

1. **Fix `arm_controller.py`'s torque-mode DOF handling** (§7) — required before any
   `franka_panda_dyn` training can run. Port the same `_pad`/`N_FULL` zero-padding approach from
   `find_throw_pose.py` into `ArmController.step()`'s torque branch and relax/fix the
   constructor's `n_dofs != len(joint_ids)` guard.
2. Confirm the Panda 23-azimuth table search finished; build the table (rotation-invariance
   should generalize the same way it did for Gen3 — verify, don't assume); train; **video-verify
   frame-by-frame before presenting anything as done** (this session's recurring lesson).
3. Gen3 12cm/slow-speed ceiling: decide whether to search a higher-range posture (bigger
   windup/elevation, different azimuth-table geometry) or accept it as the honest result of this
   release mechanism and move on.
4. `tests/test_kinetic_chain_rollout.py` needs updating against the rebuilt table.
5. Commit `results_kinetic_chain_gen3/`, `make_trained_kinetic_chain_video.py`,
   `gui_joint_control.py` (currently uncommitted).
6. Tune Panda's `kp`/`kd` torque gains — copied from Gen3 untested, mass/inertia differ.

---

# Session Handoff (2026-07-21) — MC-PILOT Throwing Arm

_Last updated: **2026-07-21** (supersedes the 2026-07-20 handoff, kept below the divider).
This session = five arcs: (1) TossingBot **residual physics — a NEGATIVE result**, both ways;
(2) the **high-drag drag crossover** (the real publishable sim finding); (3) **hardware
bring-up** code for the real Gen3; (4) a **throwing-speed investigation** (the 15 cm lob wasn't
the arm's limit — it was `pinv` + a neutral pose); (5) a **systematic-debugging pass on the
throwing POSE** (user's intuition: my "1 m throw" was an un-aimable base-spin; corrected to a
direction-constrained AIMED throw ~35–49 cm that actually hits targets — but the *trained*
version's control still diverges). Ends with honest failures the next session must NOT paper
over. Living results ledger: `paper/results_ledger.md` — update it every run._

---

## 0. READ FIRST — what is REAL vs ASSIGNED vs NOT-WORKING (the theme of this session)

The user (correctly, angrily) caught me showing an assigned-velocity throw as if it were real
dynamics. Be scrupulous about this — it recurs throughout.

| Claim / artifact | Status |
|---|---|
| Policy training + GP learning; accuracy from real rollouts (gravity+drag) | **REAL** |
| Dynamic short throw ~15 cm, trained (`kinova_gen3_dyn`, 2.68 cm) | **REAL DYNAMICS** — torque, grip constraint, ball keeps momentum. Honest trained baseline. |
| **Single-shot** optimized-posture throw 2.33 m/s / **104 cm** (`real_dynamics_throw.py`, posture 4) | **REAL, MEASURED** — no assignment. But ONE open-loop shot, not a trained policy. |
| Long-range **variety** video + kinematic long-range numbers (2.77 cm, 0.7–1.5 m) | **ASSIGNED VELOCITY** — `kinova_gen3` kinematic profile sets ball = `v_cmd`; arm cosmetic. Idealized, NOT a physical throw. |
| `gen3_optimized_throw_p0/p4.mp4` | Halfway — ball assigned the arm's `J·q̇` at the release config; kinematic windup. Not torque-driven. |
| **AIMED** single-shot throw (`make_aimed_throw_video.py`, `find_throw_pose.py` pose): aims +20°→+20°, −20°→−20°, ~35–49 cm, 0.88–1.2 m/s **measured** at dt=0.005 | **REAL + AIMED**, validated single-shot. NOT trained. |
| **Trained** policy hitting varied targets at ~40 cm–1 m under **real dynamics** | **NOT WORKING** — training-loop integration diverges (release 4–30 m/s). See §6. |
| The ~1 m "throw" (postures 0/2/4) | **UN-AIMABLE** — got speed from a base spin (sideways), can't hit a target. A mirage. The honest *aimed* max is ~35–49 cm. |

---

## 1. Who / what / goal
Rohit (2nd-yr BTech intern, FDP Lab IIT Mandi; mentor Deepak Raina). Reproduce+extend MC-PILOT
(arXiv:2502.05595). Lab arm = **Kinova Gen3 7-DOF**. **Venue reality (settled): ReScience C is
the realistic primary; ICRA 2027 (deadline ~15 Sep 2026) needs real hardware — sim ablations
alone won't make it.**

## 2. Residual physics (TossingBot) — NEGATIVE, do not re-attempt as the method
**Decision: ship PLAIN MC-PILOT.** Code kept behind flags only for the ablation figure.
- Policy-level (`Residual_Throwing_Policy`, `--residual_physics`): worse (franka 2.14→2.85).
  δ optimized through the *biased model* → inherits the bias.
- Model-level (`Ballistic_SemiParametric_Model_learning_RBF`, `--residual_dynamics`): GP learns
  Δv−gravity; within noise, no gain.
- Tests: `tests/test_residual_policy.py` (6), `tests/test_residual_dynamics.py` (4) — pass.

## 3. High-drag CROSSOVER — the real publishable sim finding
Whiffle ball (m=0.004, r=0.06, ~19 %g), `eval_baseline.py` n=30. franka: baseline 4.40 /
plain **0.62** / resdyn 0.85. kuka: 5.74 / **1.69** / 1.77. Low drag (tennis): baseline BEATS
MC-PILOT (franka 1.53 v 2.14, kuka 1.10 v 1.97, kinova_dyn 2.46 v 2.71). → **learning matters
exactly when the analytical model fails**; the low+high crossover is the clean figure. Residual
helps in neither. `eval_baseline.py` reads `ball_mass/ball_radius` from config; trainer has
`--ball_mass/--ball_radius`.

## 4. Hardware bring-up (ICRA lever) — written, UNTESTED on hardware
`robot_arm/kinova_hardware.py` (`HardwareThrowExecutor`+`SafetyLimits`: dry-run default,
`speed_scale`=0.15 time-stretch, hard `qd_max` clamp, whole-trajectory pre-check fails closed,
guaranteed stop; Kortex isolated in `_KortexBackend` — verify names vs installed `kortex_api`),
`run_hardware_throw.py` (staged `plan→connect→home→gripper→throw`; `plan` dry-run PASSES;
`throw` needs `--arm`+`--confirm`), `HARDWARE_SETUP.md`. Dominant real error = **gripper delay**.

## 5. Throwing-speed investigation — Gen3 revived (with caveats)
The 15 cm throw was NOT the arm's limit — the planner used `pinv(J)` (min-norm, joints idle)
from the neutral posture. Fix: `q̇ = q̇_max·sign(d·Jᵢ)` from a high-manipulability posture.
- Technique suite (`scratchpad/throw_techniques.py`): T0 baseline 0.60 m/s/16.7 cm → T3
  optimized posture+angle **2.46 m/s/109 cm**. **T4 whip WORSE on a rigid arm (52 cm)** —
  needs a compliant DOF. T5 elastic sling (hardware) 5–7 m — only way past the rigid ceiling.
- **~2.5 m/s / ~1.1 m = hard rigid ceiling** (joint-velocity LP).
- **Torque feasibility:** top-5 postures — **0/1/3 INFEASIBLE** (shoulder 40–65>39 Nm), 2 & 4
  feasible with ~2× windup. So my earlier "1.1 m (posture 0)" was **infeasible**; realizable
  best = posture 4.
- **REAL single-shot** (`real_dynamics_throw.py`): posture 4 → **2.33 m/s measured, 104 cm**;
  posture 2 → 2.08 m/s, 83 cm. Videos `gen3_REAL_dynamics_p{2,4}.mp4`. **⚠ These are the
  UN-AIMABLE base-spin postures — real velocity but flies sideways, can't hit a target. §6
  corrects this: the AIMED throw is ~35–49 cm.**
- Refs: Senoo & Ishikawa (kinetic chain, IEEE 4651142); arXiv:2405.19001; Yoshikawa; SEA lit;
  Pontryagin optimal control; Modern Robotics §5.4 (manipulability ellipsoid). In `results_ledger.md` §3b/§8.

## 6. HONEST FAILURES + the pose investigation (systematic-debugging session)
1. **Assigned-velocity videos.** Long-range variety used the kinematic path → ball set to
   `v_cmd`, arm cosmetic. I presented it as a throw; user caught it. Always label kin vs dyn.

2. **The throwing POSE was wrong (user's intuition — correct).** My "optimized posture" maximized
   an *unconstrained* speed bound `Σ q̇_max·|d·Jᵢ|`, which the math satisfied via **base rotation
   (j1, vertical axis) = a horizontal spin** → ball flies sideways, cannot aim. The ~1 m throw was
   an **un-aimable mirage**. Correct throwing principle (now used): **base = azimuth only; shoulder/
   elbow sweep the vertical plane; the velocity VECTOR must point along the launch direction; arm
   extended (lever).** Implemented as a **direction-constrained LP** (`find_throw_pose.py`,
   `_optimized_release`): maximize s s.t. `J·q̇ = s·d`, `|q̇ᵢ|≤q̇_max`. **Result: the aimed throw
   AIMS** — single-shot real dynamics lands +20°→+20°, −20°→−20°, **~35–49 cm, 0.88–1.2 m/s
   measured** (`make_aimed_throw_video.py`, video `gen3_aimed_throw_current.mp4`). **The honest
   aimable Gen3 throw is ~35–49 cm, ~2–3× the 15 cm baseline — NOT 1 m.**

3. **Real-dynamics TRAINING integration still diverges (unsolved).** Systematic-debugging found
   three real root causes, fixed the first two, third unresolved:
   - **(a) aiming** → fixed by the direction-constrained LP (works standalone).
   - **(b) joint-velocity overshoot**: unwrapped posture joints (6.2 rad = −0.06+2π) make the
     windup→throw cubic traverse a huge excursion, commanding **4.9 rad/s vs the 1.4 limit**
     (`plan_throw` checks torque but NOT intermediate velocity). Fixed by **wrapping joints to
     nearest neutral** (4.9→2.2 rad/s) in `_optimized_release`.
   - **(c) torque-PD instability at dt=0.02**: `kp=400` is stable on the gentle normal throw but
     diverges on the aggressive aimed throw at the GP sampling dt; the standalone single-shot is
     clean at **dt=0.005**. Tried **sub-stepping** (`nsub`, fine physics / coarse recording) in
     `_simulate_pybullet` — it did NOT fix the divergence (release still 4–10 m/s) and added a
     free-flight blowup, so it's **disabled (`nsub=1`)**. There is a residual, un-isolated
     difference between the clean standalone loop and `_simulate_pybullet`. **Concrete leads for
     next time:** wrap reference (`arm._q_neutral` in `_optimized_release` vs the hardcoded neutral
     the standalone used), `plan_throw` torque **time-stretching** changing the release-step timing,
     and the payload-compensation / follow-through interaction. All opt-in (default None); **normal
     pipeline verified intact** (v_rel 0.44, lands sane). **The trained aimed-throw policy does not
     exist yet.**

   **Debugging conclusion (Phase 4.5):** 3 fixes deep, problem persists → this is an architecture
   question, not fix #4. The aggressive real-dynamics throw does not robustly fit the training loop.
   **Recommendation: bank the aimed throw as a validated open-loop feasibility result; keep the
   trained ~15 cm `kinova_gen3_dyn` as the real trained baseline; put energy into ReScience C +
   hardware, not chasing the trained long throw (a slow-arm limitation).**

## 7. New files this session
`policy_learning/Policy.py::Residual_Throwing_Policy`;
`model_learning/Model_learning.py::Ballistic_SemiParametric_Model_learning_RBF`;
`robot_arm/arm_controller.py::plan_throw` (+overrides);
`simulation_class/model_pybullet.py` (opt_posture mode — **unstable, §6**);
`train_mc_pilot_pb_arm.py` (+ `--opt_pose` and 5 other flags); `eval_baseline.py` (residual-aware + ball params);
`robot_arm/kinova_hardware.py`, `run_hardware_throw.py`, `HARDWARE_SETUP.md`;
`make_basket_video.py`, `make_basket_video_hgen.py`, `eval_and_video_long.py`, `real_dynamics_throw.py`;
`find_throw_pose.py` (direction-constrained AIMED pose search → `throw_pose.npy`);
`make_aimed_throw_video.py` (current working aimed throw video);
`tests/test_residual_{policy,dynamics}.py`; `paper/results_ledger.md`; `status_update/email_update4.md`.
`simulation_class/model_pybullet.py` `opt_posture` mode: aimed-LP `_optimized_release` + joint-wrap
(good); sub-stepping `nsub` disabled (=1). `robot_arm/arm_controller.py` `plan_throw` overrides.
Scratchpad (not committed): `optimize_throw.py`, `throw_techniques.py`, `optimized_throw_video.py`,
`domain_probe.py`, `best_postures.npy`. In-repo: `throw_pose.npy` (the AIMED pose, elev 20°).

## 8. Checkpoints
`results_mc_pilot_pb_A_{franka_panda_flight,kuka_iiwa_flight,kinova_gen3_dyn}_{residual,resdyn}`,
`..._{franka_panda_flight,kuka_iiwa_flight}_whiffle_{plain,resdyn}`,
`results_mc_pilot_pb_A_kinova_gen3_long` (kinematic/ASSIGNED, 2.77 cm, 0.7–1.5 m).

## 9. Gotchas this session
- Assigned (`kinova_gen3`, kinematic) vs dynamic (`kinova_gen3_dyn`, torque) release — never conflate.
- **Feasibility ≠ realization**: LP + single-cubic torque check both passed posture 0, but the
  real `plan_throw` windup structure makes it torque-INFEASIBLE. Validate through the real
  planner + real dynamics (model-belief-trap family).
- PyBullet stdout warnings lack trailing newline → merge with `print()`; recover via `tr '\r' '\n'`.
- Rigid-arm max throw speed = joint-velocity LP; posture (not joint-use alone) is the win.
- Gen3: 0.6→15 cm, 2.3→~1 m. UR5 ≈ ₹45 lakh, xArm6 ≈ ¼. Not buying; throwing=sim fast arms, precision=Gen3.

## 10. Open items (priority)
1. **Decide the throw arc (recommendation: STOP chasing the trained long throw).** The AIMED throw
   is a validated open-loop feasibility result (~35–49 cm, real, aims). Making it a *trained* policy
   needs the §6(3c) training-integration divergence solved — real debugging, uncertain payoff, on a
   slow arm with a modest ceiling. Recommend banking it as-is and moving to ReScience C + hardware.
2. If pursuing #1 anyway: isolate the standalone-vs-`_simulate_pybullet` difference (leads in §6:
   wrap ref `arm._q_neutral`, `plan_throw` time-stretching of `t_r`, payload/follow-through), then
   retrain `--opt_pose throw_pose.npy` on `kinova_gen3_dyn` and eval real-dynamics aimed accuracy.
3. Multi-seed everything (all this session's arm results are single-seed).
4. Hardware bring-up stages 0→5; measure gripper delay. **The real ICRA lever.**
5. Send an email — draft `email_update4.md` exists; honest throw story: trained = ~15 cm; aimed
   ~40 cm = validated single shot (not trained); the 1 m was an un-aimable base-spin artifact.
6. Windup→throw accel-continuous handoff (older 1.54→2.68 regression) — still open.

## 11. Deliverables to read
`paper/results_ledger.md` (living record — keep updating); `paper/change_history.md`,
`paper/paper_comparison.md`; videos in `status_update/vids/` (check §0 for real vs assigned first).

---

# Session Handoff (2026-07-20) — MC-PILOT Throwing Arm

_Last updated: 2026-07-20 (supersedes the 2026-07-17 handoff below this point).
Covers everything since email_update2.md (sent ~17 Jul) — a full "velocity-from-dynamics"
study plus a paper reality-check, run across a single long session with heavy autonomous
debugging. Read `paper/change_history.md` "Exploration 6" and "Exploration 7" and
`paper/paper_comparison.md` for the full technical detail behind every claim below._

## Who / what / goal

Rohit (2nd-year BTech, civil major) — intern in FDP Lab under Deepak Raina (mentor) +
prof Dharmendra Sharma. **Lab hardware target: Kinova Gen3 7-DOF.** End goal: first-author
publication (arXiv → ReScience C → ICRA 2027, deadline ~Sept 15 2026) + hardware deployment.

## The one-paragraph version

The arm's release was cosmetic (ball velocity assigned directly, `resetBaseVelocity`) in
every study before this one. Built real torque control with gravity compensation and an
analytic payload-mass correction. Debugging that control loop surfaced **four independent,
root-caused systematic biases** — a test-timing artifact, an unmodeled-payload feedforward
gap, a target-domain reachability bug, and (the deepest one) a release-position mismatch
that had been causing the trained policy to systematically overshoot every target by
17-28%. Fixing all four collapsed kinematic-mode accuracy to **0.34cm mean** (best in the
project) and gave a *derivable* (not just measured) **1.54cm mean** for the real,
physically-grounded hardware configuration. Then read the actual paper PDF end-to-end,
built a rigorous comparison (some wins, some real gaps, one claim of ours corrected), and
started closing gaps: data augmentation, an analytical-baseline comparison (mixed, honest
result), height-generalization for kinova, and — triggered by the user watching a video and
correctly saying "that doesn't look like a throw" — found and fixed a genuine zero-amplitude
windup bug that's been in every kinova demo this project has ever made.

## State of results (all independently re-verified against real physics with fresh RNG
seeds, not just training logs — see "the model-belief trap" below for why that distinction
matters)

| Result | Status |
|---|---|
| Torque control (Gen3) | real, physically-derived release; joint tracking 0.004-0.009 rad |
| Kinova kinematic-trained → kinematic release | **0.34cm mean / 1.00cm max** (5 seeds × 30) |
| Kinova dynamic-trained → dynamic release (hardware config) | **1.54cm mean / 3.23cm max** (5 seeds × 30) — *derivable* from measured controller noise, not just observed |
| Multi-arm (kuka/franka/xarm6), release-fixed | 1.72-1.87cm mean (3 seeds × 20 each) |
| Kinova height-generalization (H_MAX=0.10m) | 0.34-0.39cm (kinematic) / 1.83-1.88cm (dynamic), **flat across all height bands** |
| Analytical baseline (paper Eq. 13) vs MC-PILOT | **mixed** — MC-PILOT wins on kinova-kinematic (~5x) and xarm6 (~2x); baseline wins on kinova-dynamic/kuka/franka (see below) |
| Data augmentation (`Na`, paper's rotation trick) | implemented + tested; whether it improves *our* data efficiency is still unverified |
| Windup motion (visual + physical) | fixed a real 0deg-swing bug; costs ~1.1cm of accuracy, not yet recovered |

## The four root-caused biases (velocity-from-dynamics study, chronological)

1. **Test-timing artifact**: the tracking-error gate sampled the arm's state *after*
   `p.stepSimulation()` but compared it to the setpoint computed *before* that step — bakes
   in a `|qd|·dt` term that isn't real error. Fixed by sampling before the step (commit
   `38b1655`).
2. **Speed-envelope miscalibration**: `kinova_gen3`'s declared `speed_bounds=(0.3, 1.0)` was
   measured on-axis only; the real zero-clipping ceiling across the full ±30° wedge is
   **u=0.61 m/s**. Recalibrated to `(0.3, 0.6)`, target range `(0.67, 0.74)` (commit
   `276c9bf`).
3. **Unreachable polar target wedge**: targets sampled as (distance-from-origin, angle)
   ignore the release-point offset; off-axis cells needed up to 3x more flight than
   on-axis at the same nominal distance — at the recalibrated speed ceiling, nothing
   beyond ~15° was reachable at all. Fixed with `--flight_targets` (flight-distance
   annulus around the release point, not the origin) (commit `632a7ce`). **Corrected
   framing (commit `dd2b578`, after reading the actual paper)**: this is NOT a flaw in the
   paper's own convention — their release point rotates with target azimuth (Eq. 5), so
   flight distance is always `ℓ-ℓr` by construction. It's a gap in our fixed-release-point
   implementation, exposed because our `ℓr` (0.55m) is comparable to our target range
   (0.67-0.74m) while the paper's `ℓr` (0.07m) is negligible next to theirs (0.7-2.4m).
4. **The deepest one — release-position mismatch (commit `86164d5`)**: `cost_trial_list`
   ("Final trial cost") is computed by simulating particles through the *learned GP
   model*, never against real physics — a genuinely useful methodology finding on its own.
   Oracle-bisection against true physics showed the trained policy commanding **17-28%
   excess speed on 12/12 targets**, unchanged by 2.5x more training trials (ruled out
   data-starvation). Root cause: particles started at the *nominal* release position, but
   reality launches ~4-5cm elsewhere (safe-release teleport + tracking residual). Fix:
   propagate particles from the *empirically observed* mean release position. Kinematic
   accuracy collapsed from ~2.5cm to 0.34cm — the biggest single improvement of the
   session. Also fully explains an earlier, wrongly-celebrated finding ("dynamic beats
   kinematic") as two opposite biases cancelling, not a real effect.

## Paper reality check (`paper/paper_comparison.md`, read the actual PDF, not summaries)

**Where we're ahead**: real validated torque control + payload compensation (paper's sim
doesn't debug at this depth), the model-belief-vs-ground-truth methodology finding,
4-platform generalization (paper: 1 platform), systematic + mechanistically-explained
object sweep, statistically-powered noise dose-response, 21-test regression suite.

**Real gaps, not spin**: no real hardware (the big one — everything above is sim-only);
gripper-delay estimation doesn't exist for kinova (`ReleaseTimingJitter` class defined,
never applied — confirmed by grep); our sub-2cm accuracy is not comparable to the paper's
**10cm real-hardware hit-radius** bar without the vision/gripper-desync noise that
dominates their real error; height-adaptation currently retrains from scratch instead of
the paper's demonstrated zero-new-trials reuse-model trick (Sec 6.3.3); `Nexp=5` used for
kinova matches the paper's *simulation* setting, not its *real-hardware* one (`Nexp=10,
Na=2`) despite kinova being the real-hardware target; no non-spherical object testing.

**Analytical-baseline comparison result (genuinely mixed, not a clean win — done this
session, `eval_baseline.py`)**: MC-PILOT beats the paper's Eq. 13 closed-form baseline on
kinova-kinematic (0.54cm vs 2.71cm) and xarm6 (1.71cm vs 3.54cm), but the baseline wins on
kinova-dynamic (1.32cm vs 1.59cm), kuka (1.10cm vs 1.97cm), and franka (1.53cm vs 2.14cm).
Traced to real drag being tiny everywhere (≤1% of gravity even at kuka's 2.5 m/s) — MC-
PILOT's advantage over the no-drag formula is inherently modest, and only shows through
when policy resolution (250 RBF centers) is tight enough not to swamp it. Kinova's domain
is 7cm (tight coverage); kuka/franka's are 40-50cm (same 250 centers, thin coverage).
**Actionable**: kuka/franka were never hyperparameter-tuned the way kinova was forced to
be — real headroom there, untested.

## The windup bug (found 2026-07-20, triggered by watching the video)

User watched a dynamic-throw video and correctly said the arm "isn't even trying to
throw." Verified: `q_release` computed by `plan_throw`'s IK came out **exactly equal** to
`q_neutral` for kinova (diff = 0.0 rad exactly; kuka swings 44°, xarm6 38°, franka 8.6° for
comparison) — because `default_release_pos` was defined as precisely where the neutral
pose's own forward kinematics already sits. The windup formula collapses to zero for any
multiplier when the thing it's scaling is already zero. This bug has been in every kinova
video/demo the whole project has produced, not something new.

**Fix** (commit `7d8ffa4`): new `windup_delta` profile field gives kinova an explicit
cocked-back pose, independent of `q_release`/`q_neutral` (doesn't touch either — both are
load-bearing for the calibration above). First attempt (0.5/0.6 rad in 0.4s) demanded 134%
of `qd_max` — torque-infeasible, caught correctly, since the windup phase had no torque-
feasibility check before this (only the throw phase did; added one, mirroring it). Sized
down properly using the rest-to-rest cubic's known peak-velocity formula
(`1.5×delta/duration`) instead of guessing again — 0.2/0.25 rad (54% of `qd_max`), verified
torque-feasible and visually confirmed (frame extraction shows genuine rise-back-then-
forward motion).

**Real cost, not hidden**: kinematic mode unaffected (windup is cosmetic there). Dynamic-
trained (hardware config) regresses **1.54cm → 2.68cm mean** (5 seeds × 15 throws,
consistent, not noise). Root cause understood but not fixed: joint-level tracking stays
within gate, but the windup phase now has genuinely nonzero acceleration at its endpoint,
creating a discontinuity at the windup/throw handoff that the Jacobian-transpose payload
correction is more sensitive to than plain joint-angle tracking shows.

**Important physics point established in this same discussion**: a bigger windup CANNOT
make the arm throw farther — release speed is `qd_release = pinv(Jacobian) @ v_cmd`,
evaluated purely at the release configuration, with zero dependence on trajectory history.
The 0.6 m/s ceiling is a joint-*velocity* limit (`qd_max`), not a torque/momentum one, so
no amount of run-up increases it (unlike a human throw, where muscle force over distance
is the bottleneck). The only real lever is *accuracy*: smoothing the windup→throw
transition (matching acceleration across the boundary, not just position/velocity) should
recover some or all of the lost 1.1cm while keeping the real motion. **Not yet
implemented — proposed, agreed as the next step, not started.**

## Deliverables

- `paper/change_history.md` — "Exploration 6" and "Exploration 7" have the full technical
  narrative, every number, every commit reference.
- `paper/paper_comparison.md` — the full paper comparison (Section A/B/C/D as described
  above), parameter-by-parameter table included.
- `status_update/email_update3.md` — **STALE, needs a rewrite before sending.** Drafted
  mid-session (~commit `42cd3f2`) — covers the four-bias narrative and the 5-seed kinova
  results, but predates: multi-arm regeneration, height-generalization, the full paper
  comparison, `Na`/baseline implementation, and the windup fix. Also still has the
  incorrect "boundary condition of the paper's convention" framing that was corrected
  later (see item 3 above) — needs that specific paragraph rewritten before sending.
- `status_update/vids/mc_pilot_kinova_dynamic_throws.mp4` — dynamic (torque-controlled)
  throw video, regenerated with the windup fix; shows the real backswing-then-throw
  motion. `make_dynamic_video.py` regenerates it (uses the existing `frame_hook` +
  TinyRenderer method, same as every prior project video).
- Checkpoints (real, committed to results dirs, not scratch): `results_mc_pilot_pb_A_
  kinova_gen3/{1..5}`, `..._kinova_gen3_dyn/{1..5}`, `..._kinova_gen3_hgen/{1,2,3}`,
  `..._kinova_gen3_dyn_hgen/1`, `..._kuka_iiwa_flight/{1,2,3}`, `..._franka_panda_flight/
  {1,2,3}`, `..._xarm6_flight/{1,2,3}`. Superseded generations kept for before/after
  evidence: `..._uncalibrated/`, `..._releasebias/`.
- 21 pytest tests, `mc-pilot-pybullet/tests/` — first regression suite in the repo.

## New scripts this session

`measure_tracking_error.py`, `eval_sim2sim_gap.py`, `validate_dynamics.py`,
`eval_generalization.py`, `eval_noise_stress.py`, `eval_heightgen.py`, `eval_baseline.py`,
`make_dynamic_video.py` — all in `mc-pilot-pybullet/`. All real-physics evaluators (never
trust `cost_trial_list` alone — see the model-belief trap below).

## Gotchas learned this session

- **The model-belief trap**: `cost_trial_list` / "Final trial cost" is computed via
  particle simulation through the *learned GP model*. It can be near-zero while real
  accuracy is off by 17-28%. Never report training cost as an accuracy claim — always
  verify against the true rollout pipeline (`PyBulletThrowingSystem.rollout`), and prefer
  fresh, previously-unused RNG seeds when re-checking a number, not the same ones that
  produced it.
- PyBullet's `F.dropout(x, p=0.0)` is an exact no-op (rules out a dropout-scaling
  hypothesis quickly if it ever comes up again).
- Torque saturation and insufficient-gain-stiffness look similar at first (both show large
  tracking error) but behave oppositely under a gain sweep: insufficient stiffness
  improves with higher gains, saturation gets *worse* (bang-bang oscillation). If
  increasing kp/kd makes things worse, check torque headroom before tuning further.
  (Directly caused the windup-gain-tuning detour this session.)
- A rest-to-rest cubic's peak velocity is exactly `1.5×displacement/duration` and peak
  acceleration is `6×displacement/duration²` — use this to size any new windup/trajectory
  segment against `qd_max`/`tau_max` *before* running it, not after debugging a failure.
- PyBullet's stdout warnings (`b3Warning[...]`) don't end in a newline, so they can merge
  with the next `print()` call and get silently eaten by a `grep -v` filter tuned to
  remove them. If a print statement seems to have "vanished," check for this before
  assuming a crash.
- GPU (CUDA) is slower than CPU for this workload (215.6s vs 156.6s / 3 kinova trials) —
  small tensors, PyBullet itself is CPU-only regardless. Confirmed again this session;
  don't re-litigate it.

## Agreed next milestone / open items, roughly in priority order

1. **Smooth the windup→throw transition** (acceleration-continuous handoff) — agreed next
   step with the user, not started. Should recover some/all of the 1.1cm windup-fix cost.
2. **Rewrite `email_update3.md`** — stale, missing ~60% of this session's work, has one
   known-wrong paragraph.
3. Kuka/franka hyperparameter re-tuning (Nb/lengthscale/Nexp) + re-run the baseline
   comparison — direct test of whether MC-PILOT can be made to beat the analytical formula
   there too (currently it doesn't).
4. Gripper-delay estimation for kinova (paper Sec 5, Bayesian Optimization) — no real
   hardware yet, so build a sim-validated version first.
5. Reuse-model height-adaptation (paper's actual zero-new-trials trick, Sec 6.3.3) instead
   of the current full-retrain approach.
6. `Nexp=5 → 10` retrain for the hardware-facing kinova config (matches the paper's real-
   hardware setting, not its sim one).
7. Verify whether `Na` (now implemented) actually improves *our* data efficiency — the
   mechanism is built and tested, the claim itself is still open.
8. Non-spherical object testing (cube/cylinder-like, paper tests these).
9. Real hardware: Kortex driver + ~10 calibration throws, once the above sim-side items
   are in better shape.

---

# Superseded handoff (2026-07-17 evening) — kept for historical reference only

Everything below this line predates email_update2.md and the entire velocity-from-dynamics
study above. Left in place for continuity; do not treat any number below as current.

## State of results (all validated, all in status_update/)

| Result | Status |
|---|---|
| NumPy baseline seed study | random explor: 1/5 seeds converge → stratified: 3/5 → + lengthscale fix (ℓs=0.15×range): **5/5** |
| PyBullet release-collision bug | found (arm strikes released ball → GP data poisoned → lengthscale collapse 250→8.4), fixed, validated **5 seeds 50/50** |
| Proper eval (KUKA, ground) | 250 fresh targets: **100% <5cm, mean 1.86cm, max 3.81cm** |
| Variable basket height (new capability) | per-height policies h=0.25 (10/10), h=0.45 (9/10) |
| **Height-generalized single policy** | target=(Px,Py,h), 9-D state, 25 trials: **100/100 fresh throws at random h∈[0,0.45], mean 2.0cm** |
| Kinova Gen3 sim | profile + envelope measured (≤1.0 m/s, targets 0.67–0.87m); 2 seeds 9/10, converge 2–3cm |

## Gotchas learned (2026-07-17 session)

- PyBullet reuses client id 0 after disconnect → per-world init logic must reset manually
- OOM kills (no traceback) when Chrome eats RAM — free up before big batches; runs died twice
- Video: GUI/Xvfb capture path stalls; snapshot method (TinyRenderer) is fast and reliable
- Eval must use the true pipeline (PyBulletThrowingSystem.rollout) — hand-rolled demo re-implementations
  had ~10cm systematic discrepancy
- Prof cares about seed methodology (consecutive-seeds question) — protocol answers in meeting_prep.md
- Physical bucket collision walls deflect near-horizontal approaches — buckets in videos are visual-only
