# MC-PILOT Throwing — Results Ledger (living doc)

**Purpose:** one place that accumulates every result, decision, file, and negative
finding as we go, so paper-writing is assembly, not archaeology. **Update this after
every run or finding** — new numbers → a dated row; new file → the index; new
conclusion → the decisions log; dead end → negative results.

_Last updated: 2026-07-27_

---

## 0. Snapshot / strategy

- **Goal:** reproduce + extend MC-PILOT (Turcato et al., arXiv:2502.05595) — model-based RL that throws a ball into a bin from ~10 real trials. Lab hardware target: **Kinova Gen3 7-DOF** (FDP Lab, IIT Mandi).
- **Venue strategy (honest):**
  - **Primary — ReScience C:** the rigorous sim reproduction + ablations. Achievable, first-author, real. This is the bankable deliverable.
  - **Stretch — ICRA 2027 (deadline ~15 Sep 2026):** *lives or dies on real Gen3 hardware results.* Sim ablations do not make it ICRA-grade; a real-robot demonstration of something the paper didn't do (multi-arm / high-drag / vision on hardware) does.
- **Method decision:** ship **plain MC-PILOT**. Residual physics (both variants) is a **negative result** — kept in-repo only for the crossover ablation figure + honest reporting, not as the method.

---

## 1. Decisions log

| Date | Decision | Why |
|---|---|---|
| 2026-07-27 | **One release solver, shared by sim and hardware** (`simulation_class/release_solver.py`) | The hardware entry point had been planning a legacy IK+`pinv` throw — a different motion from the trained/validated one — while reporting `PRECHECK: PASS`. A copy would drift again; every historical throw bug has lived in these ~100 lines. Sim/hardware equality is now asserted to 1e-12 in CI. |
| 2026-07-27 | **Derive the hardware release box from the pose table, not a hardcoded box** | The hardcoded box (z ≤ 0.9 m) excluded the overhead release at z ≈ 1.137 m and would have refused every valid plan. Deriving it makes the check assert something true ("the release is where this table says") instead of arbitrary. |
| 2026-07-27 | **Precheck fails closed on velocity clamping** | Clamping keeps the arm safe but silently slows the throw; the ball lands short with nothing in the logs. A plan needing a clamp will not do what it claims. |
| 2026-07-27 | **Use ray–plane intersection, not stereo depth, for D415 position** | D415 depth error is ~2% of range = 2–4 cm at 1–2 m — the same magnitude as the 3.15 cm landing error being measured. Sim's `depth_camera.py` back-projects depth only because sim depth is exact. |
| 2026-07-21 | Use **plain MC-PILOT**, drop residual physics as the method | Neither policy-level nor model-level residual beat plain MC-PILOT in any regime (see §3, §4). |
| 2026-07-21 | Keep residual code + `--residual_*` flags in-repo | Needed to regenerate the drag **crossover** ablation figure and to report the honest negative. |
| 2026-07-21 | Hardware bring-up is the ICRA lever, not more sim ablations | ICRA weights real-robot results; sim-only reproduction ≠ contribution. |
| 2026-07-20 | (prior) Windup fix accepted despite 1.54→2.68 dynamic-mode regression | Real backswing motion required; accuracy recovery via accel-continuous handoff still open. |

---

## 2. Result: low-drag baseline comparison (tennis ball)

Analytical no-drag baseline (paper Eq.13) vs learned policies. **Real physics eval**
via `eval_baseline.py` (`PyBulletThrowingSystem.rollout`, NOT model cost), n=30 fresh
targets, seed 246810, tennis ball (m=0.0577, r=0.0327). Single seed unless noted.

| Arm (flight targets) | Baseline Eq.13 | Plain MC-PILOT | Policy-residual | Resdyn (gravity-mean) |
|---|---|---|---|---|
| franka_panda | **1.53** | 2.14 | 2.85 ✗ | 1.91 |
| kuka_iiwa | **1.10** | 1.97 | 2.00 ✗ | 2.07 |
| kinova_gen3_dyn | **2.46** | 2.71 | 2.55 | 2.66 |

**Reading:** in low drag the **fixed analytical formula wins** on all three arms —
drag ≤1% of gravity, so the no-drag parabola is already near-optimal and there's
almost nothing for a learned model to add. No learned variant beats it here.

---

## 3. Result: high-drag crossover (whiffle ball) — the key figure

Same harness, whiffle ball (m=0.004, r=0.06 → **~19% g drag** at throw speed), n=30.

| Arm | Baseline Eq.13 | Plain MC-PILOT | Resdyn |
|---|---|---|---|
| franka_panda | 4.40 | **0.62** | 0.85 |
| kuka_iiwa | 5.74 | **1.69** | 1.77 |

**Reading — the crossover:** in high drag MC-PILOT beats the analytical baseline
**~7x** (franka) and **~3.4x** (kuka) — the *reverse* of §2. Learning matters
precisely when the analytical model fails. §2 + §3 side-by-side = the crossover
figure: *when is MC-PILOT worth it vs a closed-form?*

---

## 3b. Result: Gen3 long-range throwing via release-posture + velocity-optimal joint scheduling (2026-07-21)

The Gen3's ~0.6 m/s / ~15 cm "throw" was NOT a hard limit — it was the planner using
`pinv(J)` (min-norm joint velocities) from the neutral posture. Fixes:
- **Velocity-limit-optimal joint use**: `qd = qd_max·sign(d·Jᵢ)` (drive every joint to its
  limit in the throw direction) instead of `pinv`.
- **Optimized release posture** (max velocity-manipulability along the throw direction).

Technique test suite (sim, release speed / throw distance), literature-guided:
| Technique | Speed | Throw |
|---|---|---|
| T0 baseline (neutral + pinv) | 0.60 m/s | 16.7 cm |
| T1 velocity-optimal, neutral pose | 0.75 m/s | 21.5 cm |
| **T3 optimized posture + joint use + angle** | **2.46 m/s** | **109 cm** |
| T4 whip / kinetic-chain (rigid arm) | 1.53 m/s | 52 cm (WORSE — confirms whip needs a compliant DOF) |
| T5 elastic sling (spring 0.5–1.0 J) | 6.6–8.3 m/s | 4.9–7.2 m (HARDWARE add, idealized ½mv² estimate) |

- **~2.5 m/s / ~1.1 m is the hard rigid-Gen3 ceiling** (provable from the joint-velocity LP;
  no trajectory trick beats it). Torque-feasible for 4/5 top postures with a slow ~1.5–2 s windup.
- **Elastic hardware is the only way past it** (SEA/sling); everything else is rigid-arm-bounded.

**Retrained long-range Gen3 (kinematic, n=40 fresh targets):** overall **2.77 cm mean, 100%
<5 cm**, throw-distance span **0.71–1.49 m** (was a 7 cm band). By distance: 0.7–1.0 m → 1.27 cm,
1.0–1.3 m → 2.90 cm, 1.3–1.6 m → 4.19 cm (error grows with range). Azimuth-flat (2.3–3.3 cm, ±30°).
Checkpoint `results_mc_pilot_pb_A_kinova_gen3_long/1`. Video `status_update/vids/mc_pilot_kinova_long_variety.mp4`.
**Caveat: kinematic (idealized release); dynamic/torque accuracy at range is the follow-up.**
Scripts: `optimize_throw.py`, `throw_techniques.py`, `optimized_throw_video.py`, `eval_and_video_long.py` (scratch/repo).

Refs: Senoo & Ishikawa (kinetic chain, IEEE 4651142); arXiv:2405.19001 (passive-joint throwing);
Yoshikawa manipulability; SEA elastic-storage literature.

## 4. Negative results (do not re-litigate; report honestly)

| Finding | Evidence | Status |
|---|---|---|
| **Policy-level residual physics** (TossingBot on the policy: speed = v̂+δ) does NOT help | franka 2.14→2.85, kuka/kinova flat (§2). δ is optimised through the *biased learned model*, so it inherits the bias — the analytical prior on the policy never injects real-physics truth into the training signal. | Dead end as a method. Keep for reporting. |
| **Model-level residual physics** (gravity mean function on the Δv GP) does NOT help | Within noise of plain MC-PILOT in both low (§2) and high (§3) drag. Gravity is state-independent, so a zero-mean GP already fits it in-domain; the extrapolation benefit didn't materialise into accuracy. | Dead end as a method. Keep for reporting. |
| **Model-belief trap** (prior) | `cost_trial_list` (training cost) computed through the learned GP can be ~0 while real accuracy is off 17–28%. | Methodology finding — always eval via real rollout with fresh seeds. |

---

## 5. Full result history (chronological, from the very first email)

Every reported result since project start, with its source. Verify eval seeds before
quoting a single number in the draft; ranges/multi-seed values are safest.

### Email 1 — 2026-07-14 (`status_update/status_email.md`) — baseline reliability fix
- **Seed-reliability finding:** the original "5/5 hits in 5 trials" held for only **1 lucky seed**. Re-run across 5 seeds → hit rates **60/20/80/10/10%**.
- Root causes: (1) random exploration leaves the model blind to part of the speed range; (2) policy lengthscale too coarse for the target geometry (throws near max speed).
- **Fix:** stratified exploration + target-range-scaled lengthscale (**ℓs = 0.15 × range**).
- **Result:** all 5 seeds → **≥90% hit** (four at 100%), landing error <2–3 cm — including the two seeds that failed under every prior config.

### Proper eval, KUKA ground (`status_update/eval_matrix.md`)
250 fresh targets, 5 seeds: **100% <5 cm, mean 1.86 cm, median 1.77, P95 3.17, max 3.81 cm.**

### Email 2 — 2026-07-17 (`status_update/email_update2.md`) — heights, generalization, Gen3 sim
- **Variable basket heights** (separate policies): h=0.25 m → **10/10**, h=0.45 m → **9/10**, 2–4 cm.
- **Height-generalized single policy** (target = (Px,Py,h), 9-D state, 10 exp + 25 trials): **100/100** fresh throws at random h∈[0,0.45] (never-seen heights), **mean 2.0 cm, worst 5.0 cm, no trend with height.**
- **Kinova Gen3 integrated in sim** (official URDF, real joint limits 1.396/1.222 rad/s, kinematic mode — wrists only 9 Nm). Envelope: ~1.0 m/s EE, reachable band **0.67–0.87 m**. Training: **9/10 both seeds, 2–3 cm.**
- **Noise study:** at 20% velocity slip, **100% (learning) vs 0% (curve inversion)** — where the learning method earns its value.
- **Realization:** the arm was *cosmetic* pre-this-work (ball velocity assigned via `resetBaseVelocity`).

### Email 3 — 2026-07-20 (`status_update/email_update3.md`) — real dynamics, 3 biases, accuracy floor
- **Torque control, real release:** computed-torque + gravity comp + analytic Jacobian-transpose payload term. Joint tracking **0.004–0.009 rad**, 9 Nm wrists respected. First physically-real release in the project.
- **Bias — target convention (flight-space fix):** polar (distance,angle) sampling ignores the release offset → off-axis needs ~3× flight; at Gen3's honest 0.61 m/s off-axis ceiling nothing beyond ~15° was reachable. Fixed via flight-annulus sampling → trains on **4 arms (KUKA/Franka/xArm6/Kinova), 1.4–2.7 cm mean, multi-seed.**
- **Bias — model-belief trap:** trial cost is the model's belief, not accuracy; trained policy commanded **17–28% excess speed on every target**. Root cause: particles start at *nominal* release pos but reality launches ~4 cm downrange. Fix: propagate from **empirical mean release position**.
- **Results after fix (5 seeds × 30, real-physics eval):**
  - **Kinematic release: 0.34 ± 0.07 cm mean, worst 1.00 cm** — best in project; overshoot gone (was 12/12 over, now centred).
  - **Dynamic (torque) release — hardware config: 1.54 ± 0.09 cm mean, worst 3.23 cm.** *Derivable* floor: tracking scatter 0.06 m/s × flight slope 0.29 m/(m/s) ≈ 1.4–1.8 cm; all 5 seeds inside. Irreducible 50 Hz controller scatter — real 1 kHz Kortex should beat it.
  - Earlier "dynamic beats kinematic" was two biases cancelling; each mode now calibrates cleanly.
- **Object generalization:** flat **1.4 cm** across 30–150 g and 2–4.5 cm balls, no retrain (drag negligible at these speeds).
- **Noise dose-response** (n=50/condition): zero-mean noise degrades monotonically, uncompensatable — confirms the taxonomy.
- First **regression test suite** (17 tests).

### 2026-07-20 later (`HANDOFF.md`) — windup fix + paper comparison
- **Windup bug:** kinova `q_release == q_neutral` exactly → 0° swing in every kinova demo ever. Fixed via explicit `windup_delta`. Cost: dynamic **1.54 → 2.68 cm** (accel-continuous handoff to recover is open).
- **Multi-arm regenerated** (release-fixed): **1.72–1.87 cm** (3 seeds × 20).
- **Height-gen (kinova):** 0.34–0.39 (kin) / 1.83–1.88 (dyn), flat across bands.
- **Analytical-baseline comparison (mixed):** MC-PILOT wins kinova-kinematic (~5×) + xarm6 (~2×); baseline wins kinova-dyn/kuka/franka. → motivated this session's residual + crossover work (§2–4).

### 2026-07-21 (this session) — residual physics + crossover + hardware
See **§2 (low-drag), §3 (high-drag crossover), §4 (negatives)** above, and §6 for the hardware bring-up.

Full narrative: `paper/change_history.md` (Explorations 1–7), `paper/paper_comparison.md`.

---

## 6. File index (what each thing is)

### Added this session (residual physics — kept for ablation only)
- `mc-pilot-pybullet/policy_learning/Policy.py` :: `Residual_Throwing_Policy` — policy-level residual (speed = clamp(v̂+δ)); **negative result**.
- `mc-pilot-pybullet/model_learning/Model_learning.py` :: `Ballistic_SemiParametric_Model_learning_RBF` — GP learns Δv−gravity; **negative result**.
- `mc-pilot-pybullet/train_mc_pilot_pb_arm.py` — added flags `--residual_physics`, `--residual_dynamics`, `--delta_max_frac`, `--ball_mass`, `--ball_radius`.
- `mc-pilot-pybullet/eval_baseline.py` — residual-aware policy load + reads ball params from config (so high-drag eval matches training).
- `mc-pilot-pybullet/tests/test_residual_policy.py` — 6 tests (policy residual).
- `mc-pilot-pybullet/tests/test_residual_dynamics.py` — 4 tests (model residual).

### Added this session (hardware bring-up — the ICRA lever)
- `mc-pilot-pybullet/robot_arm/kinova_hardware.py` — safety-gated Kortex executor (`HardwareThrowExecutor`, `SafetyLimits`); structural safety (dry-run default, speed_scale, qd clamp, position precheck, guaranteed stop).
- `mc-pilot-pybullet/run_hardware_throw.py` — staged bring-up CLI: `plan → connect → home → gripper → throw`.
- `mc-pilot-pybullet/HARDWARE_SETUP.md` — checklist, safety model, exact command sequence.

### Key existing files (reference)
- `mc-pilot-pybullet/policy_learning/MC_PILCO.py` — the MC-PILOT loop (model learn → policy improve → trial).
- `mc-pilot-pybullet/simulation_class/model_pybullet.py` — PyBullet ground truth (gravity + quadratic drag + wind hooks).
- `mc-pilot-pybullet/robot_arm/arm_controller.py` — IK + 3-phase throw trajectory (`plan_throw`, `get_setpoint`); shared by sim AND hardware.
- `mc-pilot-pybullet/robot_arm/robot_profiles.py` — per-arm profiles (qd_max, q_neutral, timing, speed_bounds).
- `mc-pilot-pybullet/eval_baseline.py` — the canonical real-physics evaluator (baseline vs policy).

### Progress record / source emails (chronological, §5 draws from these)
- `status_update/status_email.md` — Email 1 (Jul 14): baseline seed-reliability fix.
- `status_update/eval_matrix.md` — KUKA ground 250-target eval table.
- `status_update/email_update2.md` — Email 2 (Jul 17): heights, generalization, Gen3 sim.
- `status_update/email_update3.md` — Email 3 (Jul 20): real dynamics, 3 biases, accuracy floor.
- `status_update/email_update3.md` is **stale** re: later work (windup, this session) — rewrite before sending.
- `status_update/HANDOFF.md` — full session handoff (windup fix, paper comparison, gotchas).
- `status_update/meeting_prep.md`, `meeting_notes_pybullet_fix.md` — meeting material / seed-methodology answers.

### Result directories (checkpoints, don't hand-edit)
- Low-drag residual: `results_mc_pilot_pb_A_{franka_panda_flight,kuka_iiwa_flight,kinova_gen3_dyn}_residual/`, `..._resdyn/`
- High-drag: `results_mc_pilot_pb_A_{franka_panda_flight,kuka_iiwa_flight}_whiffle_{plain,resdyn}/`
- Plain baselines: `results_mc_pilot_pb_A_{franka_panda_flight,kuka_iiwa_flight,kinova_gen3_dyn}/`, `..._kinova_gen3/`

---

## 6a. Overhead throw — MULTI-SEED (2026-07-28). Use these numbers.

Supersedes the single-seed 3.15 cm quoted in email 5 (that was one eval draw on
seed 1 alone). Nothing regressed — the earlier figure was just one sample.

**3 training seeds × 3 independent eval target sets × 30 throws = 270 throws.**
`eval_adapted_height.py`, real `PyBulletThrowingSystem.rollout`, never training cost.

| Training seed | mean err (avg over 3 eval sets) |
|---|---|
| 1 | 2.83 cm |
| 2 | 2.75 cm |
| 3 | 3.09 cm |

- **Headline: 2.89 ± 0.18 cm, 100 % hit < 10 cm (all 270 throws).**
- All 9 runs: 2.89 ± 0.20 cm, min 2.61, max 3.26. Worst single throw 6.61 cm.
- Eval-set effect is mild but real (2.75 / 2.89 / 3.04 cm across sets). Seed 3 is
  weakest on *every* set → genuine seed variation, not target-draw noise.
- Release speed 1.16–1.54 m/s throughout, scaling with distance (not saturated).

**Why three eval sets:** with one shared set, the ± would be training variance
seen through a single draw of targets. Two spreads that agree (0.18 vs 0.20) is
the evidence that the number is stable.

### Height adaptation, multi-seed — paper Sec 6.4, ZERO new robot trials

Each height reuses the trained GP verbatim and re-optimises the policy only.
**No additional throws at any height.** 3 training seeds × 3 heights × 3 eval
sets × 30 throws = **810 throws**; 1080 including the ground rows.

| Task | seed 1 | seed 2 | seed 3 | across-seed |
|---|---|---|---|---|
| ground | 2.83 | 2.75 | 3.09 | **2.89 ± 0.18** |
| h = 0.10 m | 3.05 | 3.15 | 3.61 | **3.27 ± 0.30** |
| h = 0.20 m | 3.43 | 3.46 | 3.90 | **3.60 ± 0.26** |
| h = 0.30 m | 3.76 | 3.83 | 4.25 | **3.95 ± 0.26** |

- **100 % hit < 10 cm in all 36 runs (1080 throws).** Worst single throw 7.96 cm.
- All 27 height runs pooled: 3.61 ± 0.37 cm (min 2.95, max 4.29).
- Error grows monotonically with height (2.89 → 3.27 → 3.60 → 3.95 cm). Honest
  reading: adaptation degrades gracefully but is **not** free — the flight span
  shrinks as the plane rises, which is also why H_MAX = 0.30 (0.45 would leave
  only ~5.3 cm of span).
- **Seed 3 is worst in every single cell** (ground and all three heights). That
  is a coherent property of that seed's learned model, not eval noise — report
  it, don't average it away.

Supersedes the single-seed 2.95 / 3.41 / 3.80 cm in email 5 (those were seed 1
on eval seed 2024 — reproduced exactly, so nothing regressed; they were just one
draw). Checkpoints `results_kinetic_chain_gen3_h{10,20,30}/{1,2,3}`.

Checkpoints `results_kinetic_chain_gen3/{1,2,3}` (commit `72161de`). Seeds 2–3
record `opt_pose` in their config; seed 1 predates that and needs `--opt_pose`.

---

## 6b. Hardware-readiness pass (2026-07-27) — no new results, 5 defects fixed

Reliability audit ahead of bring-up. **No sim number changed**: the release-logic
refactor reproduces the 30-throw eval to `0.000e+00` max abs diff on every
landing/error/speed field. Suite 54 → **65 passing**.

Verified geometry of the shipped Gen3 table (FK'd, not quoted from an email): all
23 azimuth entries release at **r = 0.035 m, z = 1.137 m**, **1.628 m/s**, **5.0°**
elevation, **0.800 m** range, elbow and wrist both saturated at their velocity
limits, roll joints exactly 0. Range decreases monotonically 0°→70° at these
speeds, so the release *height* — not the launch angle — is what buys the distance.

Defects found and fixed (all hardware-side, none affecting reported sim results):
hardware planned a legacy IK+pinv throw rather than the trained overhead one;
phase timings read from `profile.timing` instead of the trained config (≈2× peak
joint velocity); release box excluded the overhead release; `max_traj_seconds`
refused the real 8.5 s trajectory; precheck silently clamped velocity instead of
failing. Control rate 100 Hz → 1 kHz with absolute-deadline pacing (dry run:
1000 Hz over 56,682 ticks, worst tick 0.16 ms late).

**Caveat for the paper:** every headline number — 3.15 cm and all three
height-adaptation figures — remains **single-seed**. There is no ± yet.

Plan: `docs/superpowers/plans/2026-07-27-hardware-cold-start.md` (Phases A–H,
gated, incl. D415 placement and frame conventions).

---

## 6c. Run-day readiness pass (2026-08-05) — no new sim claims, 4 findings

Pre-flight before hardware day. Everything below was executed, not quoted.
Runbook written: `mc-pilot-pybullet/HARDWARE_RUNBOOK.md` (the run-day page;
`HARDWARE_SETUP.md` stays the safety model + reference).

**Verified green:**
- `pytest tests/ -q` → **66 passed** (~9 s); **68** after the homing guard below.
- `plan`, seed 2, target (0.75, 0.05), `--speed_scale 1.0 --u_cap 1.60`: release speed
  1.498 m/s, release pos in safe box, **PRECHECK PASS**, T = 8.528 s, peak |qd|
  **1.30/1.396 (93 %)**, peak |τ| **8.6/39.0 (22 %)**.
- Full dry `throw`: 1.0 → 8529 ticks, 1000 Hz, worst tick **0.00 ms** late, release at
  s = 4.928 s. 0.15 → wall 56.85 s, 56 854 ticks, 1000 Hz, worst tick **2.11 ms** late,
  release at wall 32.85 s.
- Real-physics eval on a **fresh unused seed (987654)**, 30 throws, `eval_adapted_height.py`:
  seed 2 → **2.84 cm** (max 5.43), seed 1 → **2.92 cm** (max 5.49), both 100 % hit < 10 cm,
  speed 1.17–1.55 m/s. Consistent with §6a's 2.89 ± 0.18 cm — nothing has drifted.
- **Kortex symbols now statically verified** against installed `kortex_api` 2.6.0.post3:
  all 8 groups resolve (`SendJointSpeedsCommand`, `SendGripperCommand`, `RefreshFeedback`,
  session setup, `JointSpeeds` fields, `GripperCommand`/`GRIPPER_POSITION`). Names exist;
  arm acceptance still unproven.

**Four findings, all fixed in the docs:**
1. **Ball mismatch.** `HARDWARE_SETUP.md` specified a ping-pong/foam ball; the GP was trained
   at `ball_mass = 0.0577 kg`, `ball_radius = 0.0327 m` — a **tennis ball**. Different drag
   regime ⇒ the policy would not transfer. Corrected.
2. **Stale bring-up commands.** The staged commands pointed at
   `results_mc_pilot_pb_A_kinova_gen3/1` with `--robot kinova_gen3` — a *kinematic*-mode
   profile (`tau_max=None`, so precheck **skips the torque check**) and a `uM = 0.6`
   legacy checkpoint, not the trained overhead throw. Corrected to
   `results_kinetic_chain_gen3/2` + `kinova_gen3_dyn` + `--u_cap 1.60`.
3. **Readback angle convention is an untested first-motion risk.** `read_joint_state()` does a
   bare `deg2rad`; Gen3 continuous joints report in [0, 360) and `q_neutral` contains small
   negative angles (−0.6°, −2.3°, −0.7°) that would read back near 359°. `home()`'s P-servo
   would then see a −6.28 rad error and drive the long way for the full 4 s. Dry-run cannot
   catch it (it fakes the readback). **Fixed in code**: `home()` now fails closed on any joint
   error > π rad with a `WRAP/UNIT mismatch` message instead of servoing (suite 66 → **68**,
   both new tests in `tests/test_hardware_planner.py`). Refusing to move is the only safe
   response — the intended direction cannot be inferred from a wrapped reading.
4. **`plan` reports PASS on a bad release position.** Measured: seed 1 without `--opt_pose`
   falls back to the legacy IK+pinv throw (|v| 1.496 → **0.471 m/s**, release outside the safe
   box) and still prints `PRECHECK: PASS` — the box check is a separate line. `throw` does fail
   closed on it. Runbook says read both lines.

---

## 6d. First contact with the real arm (2026-08-07) — READ-ONLY, no command sent

New tool: `mc-pilot-pybullet/hw_readonly_check.py`. Opens a Kortex session, reads,
closes. Zero writes (`connect` by contrast writes one thing — the teardown
`stop()`). Final run: **38 checks, 0 FAIL**.

**The arm, measured:** Gen3 **L53K**, SN **WO545410-1**, 7 actuators, fw 872547072,
at **192.168.1.101**. `RUN_MODE` / `SINGLE_LEVEL_SERVOING`, stationary, motors
32–39 °C, 23.1–23.3 V, no faults. tcp/10000 + tcp/80 open (10001 is UDP; a closed
TCP probe there is expected).

**Our limits are confirmed by the arm itself** — velocity 80.002/80.002/80.002/
80.002/70.004/70.004/70.004 deg/s = `qd_max` 1.3963/1.2218 rad/s, torque
39/39/39/39/9/9/9 Nm = `tau_max`, both to float32 precision. Every feasibility
check in the repo rests on these; they are real.

**Four findings, all fixed:**
1. **The default IP was this control PC.** `enp108s0` is configured `192.168.1.10/24`
   — the exact address `run_hardware_throw.py` defaulted to, so a `connect` would
   have targeted ourselves and the "ping succeeded" proved nothing. Arm found by
   subnet sweep at `.101`; default corrected in code and both docs.
2. **R1 CONFIRMED, and the 2026-08-05 guard was wrong.** Kortex reports **every**
   joint on [0, 360) — including LIMITED ones: joint 3 read 247.37° (4.318 rad)
   against its own ±2.57 rad limit. But the blanket "error > π ⇒ refuse" guard was
   *also* wrong: joint 3 legitimately needed a 4.040 rad (231°) sweep through zero,
   which a π threshold would have blocked. Correct rule is per joint type —
   `read_joint_state()` wraps to (−π, π]; `home()` takes the **shortest path for
   continuous joints (0,2,4,6)** and the **direct difference for limited ones
   (1,3,5)**; `_assert_readback_sane` fails closed on a readback outside the URDF
   range. 4 regression tests built on the real readback. Suite 68 → **72**.
3. **Homing was sized wrong.** 4.040 rad at the 0.25·qd_max cap needs ~14.5 s; the
   `duration=4.0` default would have stopped part-way, leaving an undefined pose as
   the *starting point of a throw*. `home()` now sizes its window from the measured
   distance and prints the extension.
4. **Two-master hazard, observed live.** Between two runs the arm went
   `ARMSTATE_SERVOING_READY` → `ARMSTATE_SERVOING_MANUALLY_CONTROLLED` (web UI /
   joystick in use). Streaming joint speeds into that is unsafe; the checker now
   FAILs unless the arm reads `SERVOING_READY`.

Also: `GetControlMode` and both `*SoftLimitation` calls answer UNSUPPORTED_METHOD on
this firmware — neither is used by the throw. **No write path has run yet**; stage 2
(`home`) remains the first.

### 6d-bis. Cross-check against Kinova's published docs — one more defect, the costly one

Checking the arm's own reports against Kinova's documentation rather than trusting
either alone.

**Confirmed by the docs:** the [0, 360) joint convention is official
("the valid range of joint angle for the Kinova Gen3 is 0 to 360 degrees"), so the
R1 fix is right, not a workaround. Model **L53K = Gen3 7-DOF spherical + vision**,
902 mm reach — matches the `GEN3-7DOF-VISION` URDF the profile loads.

**Not confirmed, and it matters — `control_hz` was wrong by 25×.** From Kinova's own
driver readme (`Kinovarobotics/ros_kortex`, `kortex_driver/readme.md`):

> "The robot's high level commands function at a rate of 40Hz."
> "The base high level commands are treated every 25 ms inside the robot."
> "High level control cannot be achieved at a rate faster than 40 Hz for now."

Our executor streams `Base.SendJointSpeedsCommand` with the arm in
`SINGLE_LEVEL_SERVOING` (read back from the arm) — **high-level**, ceiling 40 Hz. The
`control_hz = 1000.0` default was justified in-code by Kinova's 1 kHz figure, which
belongs to **`LOW_LEVEL_SERVOING`** (per-actuator `BaseCyclic.Refresh`), a path this
code does not use. So the 2026-07-27 "100 Hz → 1 kHz" change was chasing a rate the
API cannot accept, and 100 Hz was *already* 2.5× above it.

Not a hazard (`JointSpeeds` with `duration=0` holds until superseded, so surplus
commands are coalesced), but two real consequences:
1. **The 1 kHz result was not a hardware measurement.** "1000 Hz achieved over 56,682
   ticks, worst tick 0.16 ms late" measured our own loop; the arm was consuming 40
   commands/s throughout. Retract that framing from the record.
2. **Release quantisation is now a first-order error term.** 25 ms at the measured
   1.498 m/s release speed is **up to 3.7 cm of undershoot — larger than the entire
   2.89 ± 0.18 cm sim accuracy.** It is *not* reducible by looping faster.

Fixed: `HIGH_LEVEL_MAX_HZ = 40.0` with the citation, `SafetyLimits` clamps and warns,
`precheck` prints the quantisation as a landing-error term. Suite 72 → **74**. Dry throw
now 342 ticks @ 40 Hz (was "8529 @ 1000 Hz"); 0.15 rehearsal 2275 ticks @ 40 Hz.

**Open decision for the paper:** the high-level path has a ~3.7 cm release-timing floor
before gripper latency is counted, so sim-vs-real will be dominated by it. Buying back
millisecond release timing requires moving the throw to `LOW_LEVEL_SERVOING`
(`BaseCyclic.Refresh` at 1 kHz, per-actuator commands, no kinematic library, and a
missed frame faults the arm). That is a real piece of work and should be scoped
deliberately, not improvised on run day.

### 6d-ter. Gripper release latency MEASURED (2026-08-07) — the dominant error term

First real calibration on hardware. `measure_gripper_latency.py`, 1 kHz UDP
feedback, fingers unloaded, arm stationary.

| n | command -> fingers move | command -> motion done |
|---|---|---|
| 5  | 70.5 ± 6.4 ms | 823.3 ± 6.6 ms |
| 15 | **67.9 ± 6.4 ms** (range 59.5–80.0) | 821.0 ± 15.3 ms |

Feedback held 1000.8 Hz mean throughout — the command/feedback asymmetry works
exactly as the docs promised.

**Why this matters more than anything else measured so far.** Uncompensated,
67.9 ms at the 1.498 m/s release speed is **10.2 cm of undershoot — 3.5× the
entire 2.89 ± 0.18 cm sim accuracy**, and nearly 3× the 3.7 cm command-
quantisation term. On a first hardware run it would not have looked like a
timing bug; it would have looked like the policy failing to transfer, and the
obvious (wrong) response would have been to retrain.

**It is compensable, and the data says so.** The 6.4 ms scatter is almost
exactly what 25 ms of uniform command quantisation predicts on its own
(25/√12 = 7.2 ms), so the jitter is the command path and the gripper's own
mechanics are highly repeatable. Decomposition: ~12.5 ms mean quantisation +
~55 ms deterministic gripper latency.

Fixed: `GRIPPER_RELEASE_LATENCY_S = 0.0679` and `SafetyLimits.gripper_lead_s`;
`rehearse_or_throw` now fires OPEN at `s_fire = t_r − lead·speed_scale` so the
FINGERS move at `t_r`. The lead is wall-clock, so it must be *multiplied* by
speed_scale, not divided — dividing would over-lead the 0.15 rehearsal 6.7×
and drop the ball 0.45 s before the swing. Verified at both scales: lead stays
67.9 ms of wall at 1.0 and at 0.15. Set `gripper_lead_s = 0.0` to reproduce an
uncompensated baseline.

Expected residual after compensation: **~1.0 cm**, inside the sim accuracy.

**Caveat, stated plainly:** measured STATIC and UNLOADED. In a real throw the
fingers hold a ball and the arm is decelerating, both of which load the
mechanism. This is a calibrated starting point to be validated against real
landings, not a final constant. The with-ball measurement has not been done.

Also found while probing (read-only, ControlConfig):
- Joint **acceleration** hard limit 297.94 deg/s² = 5.20 rad/s². Never checked
  before; our planned throw peaks at 1.779 rad/s² = **34.2 %**. Not a blocker.
- Tool configured as **0.831 kg at z = 0.12 m**, mass centre z = 0.047 — worth
  reconciling with the sim's payload term.
- **`twist_linear` hard limit reads 0.500 m/s while our release needs 1.498 m/s
  (3.0×).** We command joint speeds, not twist, and stay inside the 80/70 deg/s
  joint limits — but whether the arm enforces a Cartesian ceiling in joint-speed
  mode is UNKNOWN and untested. This is the top open risk for stage 5.


### 6e. Literature cross-check on the low-level / release-timing question (2026-08-07)

Searched the official Kortex docs, Kinova's issue tracker, working third-party
drivers, and the throwing literature before committing to a low-level rewrite.
Conclusion: **do not go low-level. Reproduce the paper's delay model instead.**

**The paper we are reproducing already solves our exact problem, and it is one
of its headline contributions.** MC-PILOT Sec. 3.2 + Sec. 5 + Algorithm 1:

  t_r = t_rcmd + t_d ,  t_d ~ U(a, a+b)

- "the opening command should be forwarded at time t_rcmd, before the nominal
  release time t_r, to compensate for the delay; namely, t_rcmd < t_r" — exactly
  the lead we implemented, independently arrived at.
- But they do NOT stop at a mean. Algorithm 1 estimates (a, b) after model
  learning, then samples `t_d^(m) ~ U(a, a+b)` per particle during policy
  optimization, so the policy is optimised to be robust to the spread.
  "Properly selecting the t_d distribution is crucial to the algorithm's
  success."
- Their numbers (Franka + elastic prosthetic tooltips): compensation **240 ms**,
  found by sweeping t_r − t_rcmd over 0–0.30 s at 1.2 / 2.0 / 2.8 m/s and
  picking the value whose landing distance best matched the ballistic nominal.
  Fig. 10 puts a ≈ 0.24 s, b ≈ 0.018 s.

**Where we are ahead of the paper.** They write: "In most commercial systems,
available measurements are not adequate to directly estimate this distribution
in a data-driven fashion since the gripper and the arm are not synchronized" —
which is why Sec. 5 exists at all. On our setup that premise does not hold: the
Kortex UDP feedback channel is NOT subject to the 40 Hz command ceiling, and the
interconnect reports gripper finger position, so we measured t_d **directly** at
1 kHz: 67.9 ± 6.4 ms. That is a methodological improvement over the paper's
indirect estimate and is worth reporting as one.

External validation: arXiv 2506.16986 (2025) cites gripper detach latency as
"between 50–100 ms" and not determinable a priori. Our 67.9 ms sits mid-band.

**Gap we have not closed.** We compensate the MEAN only. The paper's stochastic
treatment is not implemented — `noise_models.ReleaseTimingJitter` exists in this
repo but has never been applied to kinova. Reproducing Sec. 5 (or substituting
our direct measurement for it) and sampling t_d in policy optimisation is the
single highest-value remaining piece of the reproduction.

**Why NOT low-level servoing** (the alternative we were considering):
- Kinova, official: "This servoing mode is not meant to be run under Python.
  C++ is a much more suitable language for low-level control" — Python is
  "sensitive to jitter and will not guarantee a 1 kHz refresh rate". Our entire
  stack is Python.
- Low-level is not "the same commands, faster": working drivers
  (empriselab/kortex_hardware, ros2_kortex) use high-level 40 Hz for
  position/velocity and low-level 1 kHz for **effort/torque only**, with the
  client owning gravity compensation (Pinocchio). We would inherit the whole
  control law — and our sim gains are already known stiffness-limited.
- Low-level *velocity* control has an open, unresolved Kinova issue (#42): the
  arm drifts down under gravity while commanding velocity. Closed as not planned.
- It buys ~25 ms of command quantisation (~7 ms std). Our gripper's own
  irreducible jitter is 6.4 ms. So the ceiling on the gain is small, and the
  paper's answer to residual jitter is to model it, not to engineer it away.

Also noted for bring-up: switching to position mode for the first time makes the
arm move to the candlestick pose (kortex_hardware README) — do not be surprised
by it, and do not have anything in the way.


### 6f. BLOCKER — we planned against HARD limits; the arm enforces SOFT limits (2026-08-07)

Found by escalating the dry rehearsal 0.15 → 0.30 → 0.60 → 1.00 with planned-vs-
actual joint position logged at 1 kHz. Everything was clean to 0.60 and broke at
1.00.

| speed_scale | peak J0 cmd | drift at release |
|---|---|---|
| 0.15 | 11.2 °/s | 0.0044 rad (0.25°) |
| 0.30 | 22.3 °/s | 0.0056 rad |
| 0.60 | 44.7 °/s | 0.0119 rad (0.7°) |
| **1.00** | **74.4 °/s** | **0.5008 rad = 28.7°** |

A 42x jump, entirely on **J0** (base rotation, which sweeps ~178° during the
throw); every other joint stayed under 2°. Not lag — an 80 ms lag fit leaves
28.1° residual. It is a sustained velocity shortfall that accumulates 0.5→2.5 s
and then holds.

**Cause.** The arm runs `SendJointSpeedsCommand` in control mode
`ANGULAR_JOYSTICK`, and that mode's SOFT limits are well below the hard limits
this repo plans against:

|  | our `robot_profiles.py` (hard) | arm SOFT, ANGULAR_JOYSTICK |
|---|---|---|
| joint speed | 80.00 °/s (1.3963 rad/s) | **50.00 °/s (0.8727 rad/s)** |
| joint accel | 297.94 °/s² | **57.3 °/s²** (J1–4), 573 (J5–7) |

We plan **1.60x over the speed soft limit** and **1.78x over the acceleration
soft limit**. The threshold behaviour is exact: every scale whose peak J0 command
stays under 50 °/s shows ~zero drift; the one that crosses it shows 28°.

**Why it was missed.** `hw_readonly_check.py` compared our limits against
`GetAllJointsSpeedHardLimitation` and correctly reported "ours is within the
arm's on all joints" — true, but of the wrong limit.
`Base.GetAllJointsSpeedSoftLimitation` answers UNSUPPORTED_METHOD on this
firmware, and I treated that as "soft limits unavailable" instead of looking
further. They live on `ControlConfig.GetKinematicSoftLimits(control_mode)` and
require the mode as an argument. Read the limits of the mode you actually
command in.

**Consequences.** The trained throw cannot execute on this arm as configured.
Everything upstream — the release LP, `find_throw_pose`, the shipped pose table,
and training — used `qd_max = 1.3963 rad/s`. At the soft limit the kinematic
release-speed ceiling drops from 1.628 to ~1.018 m/s, and the reachable landing
band from 0.60–0.80 m to roughly 0.37–0.49 m. Running at `speed_scale ≤ 0.67`
respects the limit but does NOT give the trained throw — it time-stretches to a
slower release and the ball lands short.

**Two routes, unresolved, user decision:**
1. Raise the soft limits toward the hard ones —
   `ControlConfig.SetJointSpeedSoftLimits` / `SetJointAccelerationSoftLimits`
   exist. Legitimate (the hard limits are the arm's real capability and remain
   underneath) but it is deliberately raising a safety setting on the lab's
   hardware. **Not done; not to be done without an explicit decision.**
2. Re-plan against 50 °/s — re-run `find_throw_pose`, rebuild the pose table,
   retrain, and accept the shorter range.

Also measured en route: open-loop velocity streaming is otherwise sound. Below
the soft limit, drift at release is ≤0.7°, worth ≤0.68 cm of landing error
against a 2.89 cm target. The design is fine; the limit is the problem.


### 6g. RESOLVED — per-phase timing + soft limits raised (2026-08-07)

Two fixes, measured on the arm. Drift at release, `speed_scale = 1.0`:
**0.5008 rad (28.7°) → 0.0171 rad (0.98°), a 29× reduction.**

| scale | before | after |
|---|---|---|
| 0.15 | 0.0044 | 0.0056 |
| 0.30 | 0.0056 | 0.0111 |
| 0.60 | 0.0119 | 0.0236 |
| **1.00** | **0.5008** | **0.0171** |

**1. `speed_scale` now applies to the THROW phase only.** The insight is the
user's: only the three axis-perpendicular joints (1, 3, 5) carry throw velocity.
J0 sweeps its entire +178.7° azimuth during windup, is frozen
(`qd_release[0] = 0`) through the throw, and sweeps back during follow-through —
it contributes nothing to release speed, and the duration of those phases is
irrelevant. Uniform scaling was driving it to 74.4 °/s for no reason. Windup and
follow now run at `positioning_scale` (default 1.0, already sized against
`qd_max` by `_windup_pose_and_time`); only the throw dilates. Side benefit: the
0.15 rehearsal went 56.85 s → 14.76 s, and escalation now isolates the throw
instead of re-testing positioning every time.

**2. Soft limits raised to the arm's own hard limits.** 50.0 → 80/70 °/s,
57.3 → 297.9/573 °/s², for `ANGULAR_JOYSTICK` specifically. New
`SoftLimitManager` + `run_hardware_throw.py limits` subcommand: clamps every
request element-wise to the hard limits (never touched, still enforced
underneath), reads back and fails closed on mismatch, and backs the originals up
to `results_soft_limits_backup.json` because the API has no restore-defaults.
`--restore --confirm` puts them back. **The lab arm must not be left raised.**

**3. The readiness check now tests the enforced limit.** `hw_readonly_check.py`
compared only against HARD limits and said "ours is within the arm's on all
joints" — true and useless. It now reads `GetKinematicSoftLimits` for the ACTIVE
control mode and FAILs when our `qd_max` exceeds it. 42/42 OK after the raise;
it would have failed before, which is the point.

Interpretation note: **max** drift is no longer the figure to read. With
positioning at full speed it reaches ~0.09 rad during windup/follow, which is
irrelevant — those phases carry no ball. **Drift at release** is the number:
0.0171 rad ≈ 1.0 cm of landing error, against a 2.89 cm sim accuracy.

Also observed: J3 repeatedly parks at −2.66 rad, outside the URDF's ±2.57, and
the readback guard correctly refuses to plan from there (it fired mid-session and
blocked a run until J3 was jogged back). The hardware's real range is wider than
the model's; reconciling them is an open item.


---

## 7. Open items (priority order)

1. **Hardware bring-up** (stages 0→5 in `HARDWARE_SETUP.md`). Stage 0 dry-run passes today. ← ICRA lever.
2. Measure gripper release delay on real arm; calibrate (paper §5).
3. Close the loop: RealSense (reuse `mc-pilot-pybullet-yolo` detection) → 10-throw MC-PILOT adaptation → sim-vs-real section.
4. Multi-seed the low-drag arms (is franka's 1.91 real signal or noise?) — only if time; low priority given §2 conclusion.
5. Windup→throw accel-continuous handoff (recover the 1.1 cm dynamic-mode regression).
6. Rewrite `status_update/email_update3.md` (stale).

---

## 8. Reference papers
- **MC-PILOT** — Turcato et al., arXiv:2502.05595. The method we reproduce/extend.
- **TossingBot** — Zeng et al., IEEE T-RO 2020 (arXiv:1903.11239). Source of the residual-physics idea (which did not help here); their high-drag object results motivate our crossover framing. PDF in repo root.
- **MC-PILCO** — upstream reference impl (MERL, AGPL-3.0), vendored in `MC-PILCO/`.

---

## 9. This doc's changelog
- 2026-07-21 — created; low-drag baseline comparison, high-drag crossover, residual-physics negatives, hardware bring-up files, decisions log.
- 2026-07-21 — §5 expanded into full chronological history from Email 1 (Jul 14) through this session, with source-email references; added emails/status files to the file index.
