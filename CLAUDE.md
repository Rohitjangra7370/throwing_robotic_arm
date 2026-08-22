# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Course project turned research effort (AR525, IIT Mandi; FDP Lab) reproducing and extending **MC-PILOT** (Turcato et al., arXiv:2502.05595) — model-based RL that teaches a robot arm to throw a ball into a target bin from ~10 real trials. `MC-PILCO/` is the vendored upstream reference implementation (MERL, AGPL-3.0) — **do not modify it**; all project work happens in the sibling `mc-pilot*` directories, each a full fork extended for one study.

**Provenance — get this right before writing anything for publication.** This repo is a fork of
`github.com/dnfy502/ar525_project`. Commits `526520d` (2026-03-25) through `cd4f265` (2026-04-29)
are **AR525 Group 3's** (Aarya Agarwal, Bhumika Gupta, Rishang Yadav, Yajesh Chandra); their write-up
is `rl_forked_paper.pdf` (their original `paper/Group_3_Report.pdf` was deleted from the working
tree, uncommitted — recoverable with `git show HEAD:paper/Group_3_Report.pdf > out.pdf` if needed).
**The 5 studies and their findings are theirs, not ours.** Our work starts at `22d0f03`
(2026-07-02). Treat their results as prior work to cite — and read `PROGRESS_REPORT.md` §2 and §6
first, which document where their claims do not survive checking (the ℓs rule and the multi-arm
study respectively; despite what older text in this file may say, there is no separate
`paper/forked_paper_review.md` — that file never existed on disk or in git history).

The 5 inherited studies are **done and frozen**. Active work is a hardware-facing track inside
`mc-pilot-pybullet/` targeting the lab's **Kinova Gen3 7-DOF**: real torque control, throw-pose
search, height adaptation, and hardware bring-up. Venue strategy: ReScience C primary (sim
reproduction + ablations); ICRA 2027 stretch, which lives or dies on real Gen3 results.

## Read these before doing anything

| Doc | What it is |
|---|---|
| `status_update/HANDOFF.md` | **Read first.** Session-by-session state, newest on top. Every session opens with a REAL vs ASSIGNED vs NOT-WORKING table and closes with open items. |
| `PROGRESS_REPORT.md` (root) | Full experiment-by-experiment narrative since the fork, section-numbered (§1 baseline audit … §6 multi-arm/drag crossover), with commit refs. Successor to the old `paper/change_history.md`. |
| `COMPARISON_VS_ORIGINAL_PAPER.md` (root) | Parameter-by-parameter comparison against the actual MC-PILOT paper: where we're ahead, real gaps. Successor to the old `paper/paper_comparison.md`. |
| `paper_icra2027/draft.tex` | Current paper draft (+ `paper_icra2027/results/`, `figs/`). Successor to the old `paper/main.tex`. |
| `docs/superpowers/plans/` | Written execution plans (velocity-from-dynamics, kinetic-chain pose, hardware bring-up) |

**The entire `paper/` directory (LaTeX source, `results_ledger.md`, `current_config.md`, `change_history.md`, `paper_comparison.md`, `.bib`, everything) was deleted from the working tree, uncommitted, before this file was last edited.** `change_history.md` and `paper_comparison.md` have on-disk successors (above); **`results_ledger.md` and `current_config.md` do not** — nothing currently plays their role. Any file under old `paper/` is still recoverable with `git show HEAD:paper/<name> > out`, since the deletion was never committed — check `git status` before assuming it's gone for good, and ask the user before treating this as intentional.

Separately, the top-level `README.md` is stale — it points to `change_history.md`/`current_config.md` as repo-root files (a location that predates even the `paper/` move) and describes a `.venv/` that does not exist.

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
| `status_update/`, `docs/`, `paper_icra2027/` | — | — | Progress docs/emails/videos, execution plans, ICRA draft source. Old `paper/` is gone — see the note under "Read these before doing anything" |
| `old/` | — | — | Superseded, historical only |

`results_*/` directories are run outputs (checkpoints, logs, plots) — read them, don't hand-edit.

## Environment

Dependencies (torch 2.9, pybullet, numpy, scipy, matplotlib, pytest, ultralytics/opencv for Study 5) are installed in the **system `python3`** (3.10.12, packages under `~/.local/lib/python3.10/`) — there is no venv despite what `README.md` and the per-variant `environment.yaml` files say. Use `python3`.

**Never install with bare `pip` on this machine.** `pip`/`pip3` on PATH resolve to the Blender snap's Python 3.13 (`#!/snap/blender/.../python3.13`), so packages land somewhere `python3` cannot see them and the import still fails. Always use `python3 -m pip`.

`kortex_api` (2.6.0.post3, for the real Gen3) is installed and its nine call sites are verified against the wheel. Two things to know: it pins **protobuf 3.5.1**, which downgrades protobuf system-wide and breaks onnx/tensorboard/wandb (the throw pipeline is unaffected — full suite passes); and protobuf 3.5.1 needs the `collections.MutableMapping` shim in `kinova_hardware.py::_patch_collections_abc()` to import at all on Python 3.10. If those other tools are needed here, put the hardware stack in its own venv.

Run all commands from *inside* the relevant variant directory — imports resolve via `sys.path.append("..")`, so scripts assume CWD = that variant's root.

**Bare `python3` on PATH may resolve to a Conda base env instead of the system interpreter.** If `conda`/miniconda is initialized in the shell profile, `which python3` can point at `~/miniconda3/bin/python3` (observed: 3.14, no torch/pybullet installed) ahead of `/usr/bin/python3` (3.10.12, has everything). `import torch` failing with `ModuleNotFoundError` despite this doc saying deps are installed means you're on the wrong `python3` — check with `python3 -c "import torch, pybullet"` before concluding something is actually broken, and fall back to `/usr/bin/python3` explicitly if needed.

## Tests

`mc-pilot-pybullet/tests/` is a real pytest regression suite (103 tests, ~20 s, no GPU). It is the **only** variant with one; elsewhere "tests" means `test_*.py` training scripts.

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

**Which checkpoint.** The current hardware-facing training run is `results_kinetic_chain_gen3/{1,2,3}` with `--robot kinova_gen3_dyn`. Seeds 2–3 record `opt_pose` in their config; **seed 1 predates that and silently falls back to a legacy IK+pinv throw (1.496 → 0.471 m/s) unless you pass `--opt_pose`** — and `plan` still prints `PRECHECK: PASS` when it does, because the release-box verdict is a separate line. Read both lines. Do not use `results_mc_pilot_pb_A_kinova_gen3/1` for anything hardware-facing: it is a `kinematic`-mode profile (so precheck **skips the torque check** entirely) at a legacy `uM = 0.6`.

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
- `kinova_hardware.py` / `run_hardware_throw.py` — safety-gated Kortex executor (dry-run default, speed_scale time-stretch, hard qd clamp, whole-trajectory precheck that fails closed). `HARDWARE_SETUP.md` is the safety model + reference; `HARDWARE_RUNBOOK.md` is the run-day page.
- `hw_readonly_check.py` — opens a Kortex session, reads, closes. **Zero writes** (`connect` by contrast writes the teardown `stop()`). Run it first at the lab. `measure_gripper_latency.py` — 1 kHz UDP feedback latency calibration.
- **Read/write status (updated 2026-08-22):** read-only paths, the gripper, and **the full throw trajectory as a joint-speed stream** have all run on the real arm — escalated `speed_scale` 0.15 → 0.30 → 0.60 → 1.00, **gripper empty, no ball, no landing**. Open-loop velocity streaming is viable: drift at release 0.0171 rad ≈ 1.0 cm of landing error against a 2.89 cm sim accuracy (2026-08-22 re-run after a gripper-loop fix, see below: 0.0318 rad, still within budget but ~1.9× the 08-10 figure — noted, not yet explained). `connect`, `home`, and `gripper` have all now actually been accepted and executed by the arm — the "no write path has ever been accepted" claim below is stale. **Still never executed: a throw with a ball in the hand.** It is intentionally paused, not untested — see the TCP-offset blocker below, found and quantified 2026-08-22, which must be fixed before a ball throw is meaningful.
- **`HardwareThrowExecutor.set_gripper()` fixed 2026-08-22.** It used to fire-and-forget `SendGripperCommand` and was observed to produce **zero motion** via the standalone `gripper` CLI (session teardown raced the async motor). Now blocks and confirms via `read_gripper()` feedback by default. The in-loop release call inside `rehearse_or_throw` must keep `confirm=False` — confirming there stalls the 40 Hz control loop (measured: a single 693 ms late tick, 0.38 rad drift) since polling blocks the same thread that has to keep streaming joint speeds.
- **BLOCKER, found 2026-08-22, bigger than anything else in this list: the release point/velocity model has always assumed a zero-length end effector.** `profile.ee_link` (`end_effector_link`, the bare wrist flange) is what `release_solver.py`'s LP solves against, and the sim's ball attaches there too (`model_pybullet.py:256`, literally zero offset) — consistent between sim and hardware, but neither has ever accounted for the real Robotiq 2F-85's reach. **The offset is not a guess — it's already in the arm's firmware.** `ControlConfig.GetToolConfiguration()` (read-only) returns `tool_transform = (0, 0, 0.12) m, zero rotation`, `tool_mass = 0.831 kg` (matches a real 2F-85 — configured deliberately, not a default). At the real trained release state, `v_true = v_flange + ω×r_offset` (ω=3.25 rad/s at release, `r_offset = R(q)·[0,0,0.12]`) computes to **0.39 m/s (26% of release speed) and 12.0 cm position delta** — an order of magnitude bigger than every previously-known error term combined (gripper latency, quantisation, drift). This same reading also confirmed the `GetMeasuredCartesianPose` `theta_x/y/z` Euler convention (intrinsic XYZ degrees) to 1.23°, previously unverified. A reference URDF with real Robotiq 2F-85 kinematics exists at `robot_arm/_urdf_cache/gen3_robotiq2f85.329d9bba3a1a.urdf` (primitive collision geometry, not wired into `N`/`N_FULL`/`joint_ids` anywhere) — its `gripper_mount` joint is kept at identity and is **not** the source of truth for the offset (that link chain's own ~13cm reach is from a different, uncertain generic-gripper source; stacking it on the firmware's 0.12m would double-count). Fix, not yet coded: fold `r_offset = R(q)·[0,0,0.12]` into `release_solver.py`'s release position/velocity, re-run `find_throw_pose.py`, re-verify feasibility, **then** resume the ball-throw bring-up. Do not throw a ball through this checkpoint until this is done.
- `noise_models.py` — velocity slip, salt-and-pepper, `ReleaseTimingJitter` (wired only into `test_mc_pilot_pb_C.py`, the paper-faithful KUKA noise demo; never applied to kinova — the real gripper latency is compensated in `kinova_hardware.py` instead).
- **Camera-base extrinsic calibration (new, in progress)** — prep for the "close the loop" step in `HARDWARE_SETUP.md` (RealSense detects basket/ball position, feeds landing error back into the GP). `make_aruco_targets.py` / `make_aruco_printable.py` generate print-ready, exact-scale ArUco/ChArUco PDFs with a verification ruler (print scaling silently corrupts marker size otherwise — measure the ruler before trusting a print). Top-level `../cam_snapshot.py` (repo root, not this directory) is a D435i aiming/exposure diagnostic, not a calibration step itself. `plan_camera_mount.py` plans where to put the camera. No extrinsic calibration has been completed yet.

`gpr_lib/` is the upstream GP math layer — library code, rarely edited.

## Things that will bite you

**Methodology**

- **The model-belief trap.** `cost_trial_list` / "Final trial cost" is computed by simulating particles through the *learned GP model*, not real physics. It can sit near zero while real accuracy is off by 17–28%. Never report training cost as accuracy — always re-evaluate through `PyBulletThrowingSystem.rollout` (`eval_baseline.py` etc.) with **fresh, previously-unused RNG seeds**.
- **Assigned vs real dynamics.** `kinematic`-mode profiles set the ball's velocity directly (`resetBaseVelocity`); the arm is cosmetic. Never present those numbers or videos as a physical throw. Label kin vs dyn explicitly, every time.
- **Visually verify renders.** Extract and view actual frames before claiming a motion looks right. Numeric checks alone have missed a corkscrew throw, a zero-amplitude windup, and an arm that never moved — repeatedly.
- **Feasibility ≠ realization.** An LP plus a single endpoint torque check can pass a pose that the real `plan_throw` windup structure makes torque-infeasible. Validate through the real planner and real dynamics.

**Physics / control**

- **An arm on a base plate is modelled by raising the base, never by a negative `target_height`.** The real Gen3 sits on a **0.433 m** plate and throws to the floor. `plane.urdf` is a real collision plane at world z=0 and the landing test only fires on a *descending crossing* of `target_height`, so `target_height < 0` means the ball rests on the floor without ever crossing, the loop runs out, and the rollout returns the ball's resting position as if it were a landing — plausible numbers, no error. Use `--base_height` (trainer) / `base_height=` (`PyBulletThrowingSystem`); negative `target_height` now raises. **Frames:** world = floor at 0, base at `+base_height`; base frame (hardware, pose tables) = base at 0, floor at `-base_height`.
- **A pose table's `range` is meaningful only against the floor it was searched for.** `find_throw_pose.py`'s range objective hardcoded the floor at base-frame z=0 until 2026-08-12, so every table shipped before then assumes a floor *level with the base*. Re-searched at the real `--floor_z -0.433`, the winning **release state came back bit-identical** (`max|Δq| = max|Δq̇| = 0`, same 1.6281 m/s, same 5.0° elevation) and only `range` moved, 0.7999 → 0.9355 m: the optimum is a corner solution with elbow and wrist saturated at their velocity limits, so dropping the floor rescales the objective without reordering candidates. **Measured for this arm and wedge, not a general law** — re-check it for any other arm. What the stale floor did corrupt is the trained **target band** (0.60–0.80 m against a true reachable ~0.935 m). Tables now carry a `floor_z` stamp and the trainer refuses a stamp that disagrees with `--base_height`.
- **Only axis-perpendicular (pitch) joints carry throw velocity.** For Gen3 and Panda the roll/twist joints are indices `(0,2,4,6)` — their axes point along their links, so letting an LP use them produces a corkscrew, not a throw. Freeze `q̇=0` there (their static *angles* stay free search parameters).
- **All three trajectory phases need feasibility checks** — windup, throw, and follow-through. Follow-through was unchecked for a long time and silently shipped 3.2× torque / 1.9× velocity violations. Sample the whole path, not just endpoints (mid-swing gravity and mid-ramp inertial torque both exceed the endpoints).
- A rest-to-rest cubic has peak velocity exactly `1.5·Δ/duration` and peak acceleration `6·Δ/duration²` — size any new segment against `qd_max`/`tau_max` with this *before* running it.
- A bigger windup cannot make the arm throw farther: release speed is evaluated purely at the release configuration (`q̇ = pinv(J)·v_cmd`), with no dependence on trajectory history. The ceiling is a joint-*velocity* limit. Windup only affects accuracy and how the motion reads on video.
- Torque saturation and insufficient gain stiffness both look like large tracking error, but respond oppositely to a gain sweep: stiffness improves with higher `kp`, saturation gets worse. If raising gains hurts, check torque headroom first.
- **Rigid-arm speed ceiling is provable** from the joint-velocity LP (~2.5 m/s / ~1.1 m for Gen3 unaimed; the honest *aimed* number is much lower). No trajectory trick beats it — only a compliant/elastic DOF does.

**Real Gen3 (measured 2026-08-07, arm at `192.168.1.101` — the old `.10` default was this control PC's own NIC)**

- **Kortex reports every joint on [0, 360), including the limited ones.** Joint 3 read 247.37° against its own ±2.57 rad limit. `read_joint_state()` wraps to (−π, π]; `home()` takes the **shortest path for continuous joints (0,2,4,6)** and the **direct difference for limited ones (1,3,5)**. A blanket "error > π ⇒ refuse" guard is wrong — it blocks legitimate 231° sweeps.
- **Gripper release latency is the dominant sim-to-real error term**: 67.9 ± 6.4 ms measured, = 10.2 cm undershoot at the 1.498 m/s release speed, 3.5× the entire sim accuracy. Compensated via `GRIPPER_RELEASE_LATENCY_S` / `SafetyLimits.gripper_lead_s`. The lead is **wall-clock, so multiply it by `speed_scale`, never divide** — dividing over-leads the 0.15 rehearsal 6.7× and drops the ball before the swing.
- **Two-master hazard is real and was observed live**: the arm flipped to `ARMSTATE_SERVOING_MANUALLY_CONTROLLED` when someone touched the web UI. Streaming joint speeds into that state is unsafe; checks fail unless it reads `SERVOING_READY`.
- The arm confirms `qd_max` (1.3963/1.2218 rad/s) and `tau_max` (39/39/39/39/9/9/9 Nm) to float32. Every feasibility check rests on these and they are real. Joint acceleration limit 5.20 rad/s² (planned throw peaks at 34%). `GetControlMode` and both `*SoftLimitation` calls answer UNSUPPORTED_METHOD on this firmware.

**Training config**

- **Stratified exploration required** — random seeds can leave zero GP coverage over part of the release-speed range. `_strat` scripts partition `[0, uM]` into `Nexp` bands. Non-strat scripts silently fail to converge.
- **RBF lengthscale must be initialized small on narrow domains.** `train_mc_pilot_pb_arm.py:249` defaults to `0.15 × (lM − lm)`. That works, but the "ℓs ≈ 0.15 × target_range" *rule* is inherited from the forked group and does not hold as stated — their own configs use ratios 0.43–1.00, and the sensitivity value they quote for it (S ≈ 0.066) actually corresponds to 0.43×range, not 0.15×. It is also only an **initialization**: `log_lengthscales` is a trainable parameter (`Policy.py:189`, `flg_train_lengthscales=True`). The real failure it avoids is a dead gradient at init — the paper's `ℓs = 1.0` saturates the tanh on narrow (<0.4 m) domains so every RBF centre fires identically. Do not cite the 0.15 constant as a derived rule; see `PROGRESS_REPORT.md` §2.
- **Zero-mean noise is not learnable** — symmetric Gaussian noise doesn't shift the optimal policy. Only biased noise (velocity slip, timing jitter, salt-and-pepper) produces an aware-vs-naive gap.
- **PyBullet and NumPy physics diverge** — a config tuned in the NumPy sim does not transfer to the arm variants unchanged.
- **GPU is slower than CPU here** (small tensors, PyBullet is CPU-only regardless). Confirmed twice; don't re-litigate.

**Settled results — don't re-litigate**

- **Residual physics is a negative result at both levels.** Policy-level (`--residual_physics`) is worse; model-level (`--residual_dynamics`) is within noise. Ship plain MC-PILOT; the code stays only for the ablation figure.
- **The drag crossover is the publishable sim finding.** Low drag (tennis): the analytical Eq. 13 baseline *beats* MC-PILOT. High drag (whiffle): MC-PILOT wins ~3–7×. Learning matters exactly when the analytical model fails.
- Study 4 (inherited, **underpowered**): the *blind* 2-D GP beats the *wind-aware* 4-D GP at 15 trials. The margins are 1–2 throws out of 15 at a single seed, and the "~40+ trials" crossover is asserted from an `Nb^(1/d)` argument, never run. Usable as motivation, not as a result.

**Known broken / in progress**

- Sim runs physics *and* control at 50 Hz (`nsub=1`, `dt_phys = dt = Ts = 0.02`) — that's where the 1.54 cm dynamic-mode noise floor comes from. **Hardware is slower, not faster:** the throw uses `Base.SendJointSpeedsCommand` in `SINGLE_LEVEL_SERVOING`, whose documented ceiling is **40 Hz** (`HIGH_LEVEL_MAX_HZ`, `kinova_hardware.py:93`). The 1 kHz figure in Kinova's docs belongs to `LOW_LEVEL_SERVOING` (`BaseCyclic.Refresh`), a path this code does not use. Any historical "1 kHz control loop" claim measured our own loop, not the arm.
- **`kortex_api` symbols are statically verified, and write paths now ARE accepted (updated 2026-08-22).** All 8 call groups resolve against the installed 2.6.0.post3 wheel; `connect`/`home`/`gripper`/empty-gripper `throw` at all four speed scales have all been executed live and accepted by the arm. Only a throw with a ball gripped remains unexecuted (see the TCP-offset blocker above).
- **`twist_linear` Cartesian ceiling — RESOLVED, was not a blocker.** Read directly off the arm (2026-08-22): `ANGULAR_JOYSTICK` (what the throw actually streams via `SendJointSpeedsCommand`) has soft `twist_linear = 0.0`, i.e. unset/not-applicable — the field is a Cartesian-mode concept, and the throw is joint-space. The 0.500 m/s hard ceiling belongs to `CARTESIAN_JOYSTICK` (the mode the arm happens to idle in), not the streaming mode. Confirmed both by this direct read and by the empty-gripper full-speed rehearsal completing clean with no fault.
- **Open hardware risks, in order:** (1) **the gripper TCP offset above — now the top blocker**, dwarfs everything else on this list; (2) ~~the 67.9 ms gripper latency was measured static and unloaded~~ — **re-measured 2026-08-22 with a real ball loaded, 15 trials: onset 73.2 ± 10.3 ms, statistically indistinguishable from the static/unloaded figure.** `GRIPPER_RELEASE_LATENCY_S` is unchanged and now validated for onset specifically — still not validated against an actual real landing measurement, which nothing but a real landing can substitute for; (3) 25 ms command quantisation is a ~3.7 cm landing-error floor that looping faster cannot fix.
- **First real ball throws executed 2026-08-22** (`speed_scale=0.15` only, twice — once baseline, once with a +90° `--wrist_roll_offset_deg` wrist-roll rotation, both clean, no fault). Landing was not rigorously measured (still resting on the uncorrected TCP-offset release model above) — do not treat either as an accuracy data point. `pickup_pose.json` records a repeatable arm pose for reloading the ball between throws. `--wrist_roll_offset_deg` (new CLI flag on `plan`/`throw`) rotates joint 6 at release for finger/release-path clearance — provably free (that joint is frozen at qd=0 throughout the throw and is last in the chain, so it cannot change release position/velocity, only the gripper's own orientation) and goes through the normal precheck since windup must travel further to reach it.
- **`HardwareThrowExecutor.set_gripper()` needed a second fix same day**: the original confirm-via-feedback logic expected the gripper to reach its literal open/closed target, but grasping a real object means the motor legitimately stalls partway (measured: 58.08% closed on a tennis ball, position and velocity both dead stable) — the fix now also accepts "moved meaningfully from its start position, then held still" as success, and separately handles being asked to re-close a gripper that's already stalled on an object from a prior call (no further motion occurs, so there's no "progress" left to detect).
- **A real `ROBOT_IN_FAULT` was hit 2026-08-22, root-caused to a power supply issue** (reproduced independently via the arm's own physical controller, not through any script here — confirms it wasn't application-level). Re-sending a gripper `close` command to an already-stalled gripper (pushing against the ball again) was the proximate trigger and is worth avoiding, but the underlying cause was electrical, not code. `hw_readonly_check.py` confirmed clean recovery once power was stable — always re-run it after any fault before sending another command.
- Panda's `kp`/`kd` were copied from Gen3 untuned — different mass/inertia, verify before trusting torque numbers. (Panda's 9-DOF zero-padding *is* fixed and verified in both `find_throw_pose.py` and `ArmController`; the constructor now only rejects non-contiguous `dof_ids`.)
- PyBullet's `b3Warning[...]` stdout lines lack a trailing newline and merge with the next `print()` — a print can appear to "vanish" into a `grep -v` filter. Check for this before assuming a crash. Recover with `tr '\r' '\n'`.
