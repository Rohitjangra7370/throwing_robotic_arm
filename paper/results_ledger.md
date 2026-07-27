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
