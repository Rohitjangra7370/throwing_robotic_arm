# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Course project turned research effort (AR525, IIT Mandi; FDP Lab) reproducing and extending **MC-PILOT** (Turcato et al., arXiv:2502.05595) — model-based RL that teaches a robot arm to throw a ball into a target bin from ~10 real trials. `MC-PILCO/` is the vendored upstream reference implementation (MERL, AGPL-3.0) — **do not modify it**; all project work happens in the sibling `mc-pilot*` directories, each a full fork extended for one study.

The original 5 studies are **done and frozen**. Active work is a hardware-facing track inside `mc-pilot-pybullet/` targeting the lab's **Kinova Gen3 7-DOF**: real torque control, throw-pose search, height adaptation, and hardware bring-up. Venue strategy: ReScience C primary (sim reproduction + ablations); ICRA 2027 stretch, which lives or dies on real Gen3 results.

## Read these before doing anything

| Doc | What it is |
|---|---|
| `status_update/HANDOFF.md` | **Read first.** Session-by-session state, newest on top. Every session opens with a REAL vs ASSIGNED vs NOT-WORKING table and closes with open items. |
| `paper/results_ledger.md` | Living record of every result, decision, negative finding, file index. **Update it after every run or finding.** |
| `paper/change_history.md` | Full experiment-by-experiment narrative ("Exploration 1–7"), with commit refs |
| `paper/paper_comparison.md` | Parameter-by-parameter comparison against the actual MC-PILOT paper: where we're ahead, real gaps |
| `paper/current_config.md` | Canonical hyperparameter table (Nexp, Nopt, M, Nb, uM, Ts, T, lc, lm, lM, γM) with deltas from the paper |
| `docs/superpowers/plans/` | Written execution plans (velocity-from-dynamics, kinetic-chain pose, hardware bring-up) |

Note: the top-level `README.md` is stale in two places — it points to `change_history.md`/`current_config.md` as repo-root files (they live under `paper/`), and it describes a `.venv/` that does not exist.

## Repository layout

Each `mc-pilot*/` directory is a **self-contained fork** (own `gpr_lib/`, `simulation_class/`, `policy_learning/`, `model_learning/`, `envs/`) — no shared library. A GP/policy fix must be ported to each variant by hand if it should apply everywhere.

| Directory | Study | Sim backend | Key addition |
|---|---|---|---|
| `MC-PILCO/` | — | — | Upstream reference, unmodified |
| `mc-pilot/` | Baseline | NumPy ballistic | Ground targets, z_release = 0.5 m |
| `mc-pilot-elevated/` | 1 | NumPy ballistic | Elevated release (z = 1.0/1.5/2.0 m), stratified exploration |
| `mc-pilot-pybullet/` | 2 + **active work** | PyBullet | Real arm physics, noise models, multi-arm, torque control, Gen3 hardware track |
| `mc-pilot-pb-elevated/` | 3 | PyBullet | Elevated targets + arm physics |
| `mc-pilot-wind/` | 4 | NumPy + wind models | Constant wind / gusts / OU turbulence, blind vs wind-aware GP |
| `mc-pilot-pybullet-yolo/` | 5 | PyBullet + OpenCV/YOLOv8 | HSV segmentation + YOLO bin detection feeding the target estimate |
| `paper/`, `status_update/`, `docs/` | — | — | Report source, progress docs/emails/videos, execution plans |
| `old/` | — | — | Superseded, historical only |

`results_*/` directories are run outputs (checkpoints, logs, plots) — read them, don't hand-edit.

## Environment

Dependencies (torch 2.9, pybullet, numpy, scipy, matplotlib, pytest, ultralytics/opencv for Study 5) are installed in the **system `python3`** (3.11) — there is no venv despite what `README.md` and the per-variant `environment.yaml` files say. Use `python3`.

Run all commands from *inside* the relevant variant directory — imports resolve via `sys.path.append("..")`, so scripts assume CWD = that variant's root.

## Tests

`mc-pilot-pybullet/tests/` is a real pytest regression suite (54 tests, seconds to run, no GPU). It is the **only** variant with one; elsewhere "tests" means `test_*.py` training scripts.

```bash
cd mc-pilot-pybullet/
python3 -m pytest tests/ -q                              # full suite
python3 -m pytest tests/test_torque_control.py -q        # one file
python3 -m pytest tests/ -k follow_through -q            # one test
```

Tests encode past bugs as regressions (mid-ramp torque, Coriolis-at-release, follow-through torque+velocity, table direction alignment, monotonic windup, dynamic release). Run them before claiming a change to `arm_controller.py` / `model_pybullet.py` / `find_throw_pose.py` is safe.

## Running the studies

```bash
# Baseline
cd mc-pilot/ && python3 test_mc_pilot.py -seed 1 -num_trials 10

# Study 1 — elevated release (NumPy).  _strat scripts = stratified exploration, use these
cd mc-pilot-elevated/ && python3 test_mc_pilot_b_strat.py -seed 1 -num_trials 10   # z=1.0m (c=1.5, d=2.0)

# Study 2 — PyBullet arm + noise + multi-arm
cd mc-pilot-pybullet/
python3 test_mc_pilot_pb_A.py -seed 1 -num_trials 10                    # baseline, KUKA iiwa7
python3 test_mc_pilot_pb_A_noisy.py -seed 1 -num_trials 10 -alpha 0.20  # noise-aware
python3 run_pb_noise_paper_multiseed.py                                 # full noise sweep, 3 seeds
python3 demo_pybullet_gui.py --log_path results_mc_pilot_pb_A/1         # GUI replay

# Study 3 — PyBullet + elevated
cd mc-pilot-pb-elevated/ && python3 test_mc_pilot_pbe_B.py -seed 1 -num_trials 10   # z=1.0m

# Study 4 — wind
cd mc-pilot-wind/
python3 run_all_wind_experiments.py --num_trials 15   # all 9 configs
python3 analyze_wind_results.py                       # summary table

# Study 5 — vision
cd mc-pilot-pybullet-yolo/
python3 test_mc_pilot_pb_A.py -seed 1 -num_trials 10
python3 demo_pybullet_gui.py --log_path results_mc_pilot_pb_A/1 --num_throws 5
```

Study scripts take `-seed`/`-num_trials` (single dash); results land in `results_<config_name>/<seed>/`.

## The active track: `mc-pilot-pybullet/`

`train_mc_pilot_pb_arm.py` is the general trainer (double-dash flags, unlike the legacy `test_*.py` scripts). The per-arm `train_mc_pilot_pb_A_*.py` scripts are thin wrappers.

```bash
cd mc-pilot-pybullet/
# train any profiled arm
python3 train_mc_pilot_pb_arm.py --robot kinova_gen3_dyn --seed 1 --num_trials 10 --flight_targets

# search a hardware-valid throw release state, then build the azimuth->pose table
python3 find_throw_pose.py --robot kinova_gen3_dyn --mode overhead --out throw_pose_table.npy

# train through that table (real measured dynamic release, not assigned velocity)
python3 train_mc_pilot_pb_arm.py --robot kinova_gen3_dyn --opt_pose throw_pose_table.npy --flight_targets

# zero-new-trials height adaptation (paper Sec 6.4): reuse the trained GP, re-optimize policy only
python3 adapt_policy_height.py --log_path results_kinetic_chain_gen3/1 --height 0.20 --out results_kinetic_chain_gen3_h20

# real-physics evaluation (NEVER judge accuracy from training cost — see gotchas)
python3 eval_baseline.py --log_path <ckpt> --robot kinova_gen3_dyn --num_throws 30 --seed 246810
python3 eval_heightgen.py --log_path <ckpt> --robot kinova_gen3

# hardware bring-up, staged: plan -> connect -> home -> gripper -> throw.
# Dry-run by default; talking to the real arm needs --arm, and `throw` also needs --confirm.
python3 run_hardware_throw.py plan --log_path <ckpt> --target 0.9 0.0   # see HARDWARE_SETUP.md
```

Trainer flags worth knowing: `--flight_targets` (sample targets by flight distance from the release point, not polar-from-origin — required for arms whose release offset is comparable to the target range, i.e. Gen3), `--opt_pose` (throw through a searched pose table), `--Na` (paper's rotation data augmentation), `--ball_mass`/`--ball_radius` (drag regime; whiffle = 0.004 kg / 0.06 m), `--residual_physics`/`--residual_dynamics` (ablation only — both are negative results, see below), `--target_height`, `--device`.

`find_throw_pose.py --mode`: `overhead` (search release state directly, then rotate across the azimuth wedge — current approach), `rotate` (one verified entry rotated), `independent` (per-azimuth search; slower, leaves gaps).

## Core algorithm architecture

Every variant runs the same three-module MC-PILOT loop, orchestrated by `policy_learning/MC_PILCO.py`:

1. **Model learning** (`model_learning/Model_learning.py`) — `Ballistic_Model_learning_RBF`: 3 independent sparse GPs (one per Δvx/Δvy/Δvz), input `[x,y,z,vx,vy,vz]`, Adam on marginal log-likelihood. Propagation reconstructs `v_{t+1} = v_t + Δv`, `p_{t+1} = p_t + Ts·v_t + (Ts/2)·Δv` (paper Eq. 18). `Ballistic_SemiParametric_Model_learning_RBF` subtracts gravity as a mean function (negative result, ablation only).
2. **Policy** (`policy_learning/Policy.py`) — `Throwing_Policy`: RBF net mapping target `(Px,Py)` (or `(Px,Py,h)` height-generalized) → release speed, squashed to `[0,uM]` via `(uM/2)(tanh(x)+1)`. `Residual_Throwing_Policy` adds the analytical Eq. 13 speed (negative result, ablation only).
3. **Cost** (`policy_learning/Cost_function.py`) — distance-to-target with lengthscale `lc`; drives both the policy gradient and the trial score.

`simulation_class/` is the physics backend:
- `model.py` — pure NumPy ballistic (baseline, elevated, wind)
- `model_pybullet.py` — PyBullet arm + rigid-body ball. Owns release-state logic: `_optimized_release` (direction-constrained LP: maximize speed s.t. `J·q̇ = s·d`, `|q̇ᵢ| ≤ q̇ᵢᵐᵃˣ`, roll/twist joints frozen), `opt_posture_table` (azimuth→pose table mode), `opt_posture` (legacy single pose).
- `model_mujoco.py` — unused upstream leftover

**`simulation_class/release_solver.py` is shared by sim and hardware.** `OptimizedReleaseSolver.solve()` turns (policy speed, target) into the release state (`q_release`, `qd_release`, release position) via the azimuth→pose table, turret-aiming correction and the direction-constrained LP. `PyBulletThrowingSystem._optimized_release` and `run_hardware_throw.py::plan_throw_for_target` both call it, and `tests/test_hardware_planner.py` asserts they agree to 1e-12. Never inline or copy this logic — every historical throw bug has lived in it, and a second copy will drift.

`robot_arm/` (PyBullet variants):
- `robot_profiles.py` — per-arm `RobotProfile` dataclass: URDF, joint ids, EE link, `q_neutral`, `qd_max`, `tau_max`, `kp`/`kd`, `windup_delta`, `speed_bounds`, `control_mode`. **`control_mode` is the thing to check first**: `kinematic` (ball velocity assigned — idealized, NOT a physical throw), `position`, `torque` (computed-torque + gravity comp + Jacobian-transpose payload term — the real one). Profiles: `kuka_iiwa`, `franka_panda`, `franka_panda_dyn`, `kinova_gen3`, `kinova_gen3_dyn`, `xarm6`. The `_dyn` suffix means torque mode.
- `arm_controller.py` — IK, `plan_throw` (3-phase neutral→windup→release + follow-through, with torque/velocity feasibility checks on **all** phases), gripper. Shared by sim **and** hardware.
- `kinova_hardware.py` / `run_hardware_throw.py` / `HARDWARE_SETUP.md` — safety-gated Kortex executor (dry-run default, speed_scale time-stretch, hard qd clamp, whole-trajectory precheck that fails closed). Written, **untested on real hardware**.
- `noise_models.py` — velocity slip, salt-and-pepper, `ReleaseTimingJitter` (defined, never applied to kinova).

`gpr_lib/` is the upstream GP math layer — library code, rarely edited.

## Things that will bite you

**Methodology**

- **The model-belief trap.** `cost_trial_list` / "Final trial cost" is computed by simulating particles through the *learned GP model*, not real physics. It can sit near zero while real accuracy is off by 17–28%. Never report training cost as accuracy — always re-evaluate through `PyBulletThrowingSystem.rollout` (`eval_baseline.py` etc.) with **fresh, previously-unused RNG seeds**.
- **Assigned vs real dynamics.** `kinematic`-mode profiles set the ball's velocity directly (`resetBaseVelocity`); the arm is cosmetic. Never present those numbers or videos as a physical throw. Label kin vs dyn explicitly, every time.
- **Visually verify renders.** Extract and view actual frames before claiming a motion looks right. Numeric checks alone have missed a corkscrew throw, a zero-amplitude windup, and an arm that never moved — repeatedly.
- **Feasibility ≠ realization.** An LP plus a single endpoint torque check can pass a pose that the real `plan_throw` windup structure makes torque-infeasible. Validate through the real planner and real dynamics.

**Physics / control**

- **Only axis-perpendicular (pitch) joints carry throw velocity.** For Gen3 and Panda the roll/twist joints are indices `(0,2,4,6)` — their axes point along their links, so letting an LP use them produces a corkscrew, not a throw. Freeze `q̇=0` there (their static *angles* stay free search parameters).
- **All three trajectory phases need feasibility checks** — windup, throw, and follow-through. Follow-through was unchecked for a long time and silently shipped 3.2× torque / 1.9× velocity violations. Sample the whole path, not just endpoints (mid-swing gravity and mid-ramp inertial torque both exceed the endpoints).
- A rest-to-rest cubic has peak velocity exactly `1.5·Δ/duration` and peak acceleration `6·Δ/duration²` — size any new segment against `qd_max`/`tau_max` with this *before* running it.
- A bigger windup cannot make the arm throw farther: release speed is evaluated purely at the release configuration (`q̇ = pinv(J)·v_cmd`), with no dependence on trajectory history. The ceiling is a joint-*velocity* limit. Windup only affects accuracy and how the motion reads on video.
- Torque saturation and insufficient gain stiffness both look like large tracking error, but respond oppositely to a gain sweep: stiffness improves with higher `kp`, saturation gets worse. If raising gains hurts, check torque headroom first.
- **Rigid-arm speed ceiling is provable** from the joint-velocity LP (~2.5 m/s / ~1.1 m for Gen3 unaimed; the honest *aimed* number is much lower). No trajectory trick beats it — only a compliant/elastic DOF does.

**Training config**

- **Stratified exploration required** — random seeds can leave zero GP coverage over part of the release-speed range. `_strat` scripts partition `[0, uM]` into `Nexp` bands. Non-strat scripts silently fail to converge.
- **RBF lengthscale scales with target range**: `ℓs ≈ 0.15 × target_range`. The paper's `ℓs = 1.0` collapses the policy to near-constant speed on narrow (<0.4 m) domains.
- **Zero-mean noise is not learnable** — symmetric Gaussian noise doesn't shift the optimal policy. Only biased noise (velocity slip, timing jitter, salt-and-pepper) produces an aware-vs-naive gap.
- **PyBullet and NumPy physics diverge** — a config tuned in the NumPy sim does not transfer to the arm variants unchanged.
- **GPU is slower than CPU here** (small tensors, PyBullet is CPU-only regardless). Confirmed twice; don't re-litigate.

**Settled results — don't re-litigate**

- **Residual physics is a negative result at both levels.** Policy-level (`--residual_physics`) is worse; model-level (`--residual_dynamics`) is within noise. Ship plain MC-PILOT; the code stays only for the ablation figure.
- **The drag crossover is the publishable sim finding.** Low drag (tennis): the analytical Eq. 13 baseline *beats* MC-PILOT. High drag (whiffle): MC-PILOT wins ~3–7×. Learning matters exactly when the analytical model fails.
- Study 4: the *blind* 2-D GP beats the *wind-aware* 4-D GP at 15 trials — explicit wind conditioning needs ~40+ trials to pay for the extra dimensions.

**Known broken / in progress**

- Sim runs physics *and* control at 50 Hz (`nsub=1`, `dt_phys = dt = Ts = 0.02`) — that's where the 1.54 cm dynamic-mode noise floor comes from. Hardware streams at 1 kHz, so it should track *better* than sim, not worse.
- `kortex_api` is **not installed** and the Kortex method names in `_KortexBackend` are unverified against any real release. Everything hardware-side is dry-run-tested only.
- Panda's `kp`/`kd` were copied from Gen3 untuned — different mass/inertia, verify before trusting torque numbers. (Panda's 9-DOF zero-padding *is* fixed and verified in both `find_throw_pose.py` and `ArmController`; the constructor now only rejects non-contiguous `dof_ids`.)
- PyBullet's `b3Warning[...]` stdout lines lack a trailing newline and merge with the next `print()` — a print can appear to "vanish" into a `grep -v` filter. Check for this before assuming a crash. Recover with `tr '\r' '\n'`.
