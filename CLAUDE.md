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

`mc-pilot-pybullet/tests/` is a real pytest regression suite (294 tests, ~29 s, no GPU, updated 2026-09-06). It is the **only** variant with one; elsewhere "tests" means `test_*.py` training scripts.

```bash
cd mc-pilot-pybullet/
python3 -m pytest tests/ -q                              # full suite
python3 -m pytest tests/test_torque_control.py -q        # one file
python3 -m pytest tests/ -k follow_through -q            # one test
```

Tests encode past bugs as regressions (mid-ramp torque, Coriolis-at-release, follow-through torque+velocity, table direction alignment, monotonic windup, dynamic release, the wrist-camera tool-frame error). Run them before claiming a change to `arm_controller.py` / `model_pybullet.py` / `find_throw_pose.py` is safe.

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

**Which checkpoint.** **Current preferred: `results_kinetic_chain_gen3_tcp/1`** (`--opt_pose throw_pose_table_tcp.npy --tool_offset_z 0.12 --base_height 0.433`, all three required together — trained under TCP-offset-corrected physics, see the gripper-TCP-offset entry below). Fresh-seed eval (never used in training): **mean 1.90 cm, max 4.20 cm** — beats the old checkpoint's best-of-3. Seed 2 also trained (2.10/4.58 cm, worse); seed 3 was killed mid-run and is unusable.

The pre-TCP-offset-fix training run, `results_kinetic_chain_gen3/{1,2,3}` with `--robot kinova_gen3_dyn`, is now historical — **do not use it for a real throw**: it was trained assuming the ball leaves at the bare wrist flange, which understates real release speed by ~26% (see below). Seeds 2–3 record `opt_pose` in their config; seed 1 predates that and silently falls back to a legacy IK+pinv throw (1.496 → 0.471 m/s) unless you pass `--opt_pose` — and `plan` still prints `PRECHECK: PASS` when it does, because the release-box verdict is a separate line. Read both lines. Do not use `results_mc_pilot_pb_A_kinova_gen3/1` for anything hardware-facing either: it is a `kinematic`-mode profile (so precheck **skips the torque check** entirely) at a legacy `uM = 0.6`.

**Hardware throw-session app (`hardware_session.py` / `hardware_learning.py`, built 2026-09-01/02, 27 commits, 269→274 tests).** `python3 hardware_session.py` is a Tk GUI that runs a full run-day session end to end — start-of-day gates, camera bring-up, N real throws with landing measurement, then two model-update buttons, gated in order by `SessionState` (`tests/test_hardware_session.py`; GUI/hardware paths stay untested like everywhere else in this repo, only the state machine and the pure functions are). `hardware_learning.py::ingest_throws` appends each qualifying real throw's **raw RANSAC-inlier triangulated points** (never the fitted parabola — feeding that back would just teach the GP its own gravity-only assumption, the model-belief trap again) to the flight GP, and `fit_release_model` fits commanded→measured release speed/direction. **Both exclude any throw logged at `speed_scale != 1.0`, reporting the exclusion count rather than dropping it silently** — `speed_scale` is a time-stretch, so a bring-up rehearsal at 0.15 physically releases at ~0.15× commanded speed, and treating it as data would have dragged the fitted release gain toward 0.15 instead of ~0.9. Every real throw's raw dual-IR recording is saved to `throws/throw_<idx>.npz` **before** `measure_landing` runs (a save failure degrades to a logged refusal, never a lost throw) — the permanent regression fixture `HARDWARE_RUNBOOK.md`'s "keep every recording" rule refers to.
"Update model" (button 1) runs both and reports a verdict on the flight GP: `above_noise = (mean_dev > k_se·SE) AND (mean_dev > systematic_floor)`, where `SE = σ_v/√n` and `systematic_floor` is the per-step Δv a measured 0.56° extrinsic rotation error alone would fake (0.0019 m/s — 2.5× the tennis ball's drag signal, and does not shrink with more samples). **`k_se` is not applied directly to `mean_dev`** — `mean_dev` is the norm of a 3-axis mean vector, and under pure noise that norm is χ(3)-distributed with a nonzero mean (~1.6·SE), so a bare `k·SE` scalar comparison was found (final review, verified independently 3 times via Monte Carlo) to have a **~26% false "ABOVE NOISE" rate at any sample count**. Fixed: `k` is converted to its intended one-sided normal tail probability, then the matching χ²(3) critical value gives the real threshold (`k=2` → multiplier ≈3.09, not 2) — empirically restores the ~2.3% target rate. Both `above_se` and `above_sys` are required; when `above_sys` is the blocker, the report states plainly that no sample count can resolve it (a systematic floor, unlike random noise, does not average down) rather than printing a misleading "N more throws needed."
"Re-optimize policy" (button 2, `reoptimize_policy()`) unlocks only after button 1 has run, re-optimizes the policy against the now-updated GP by forwarding the caller's `reinforce_policy(**kwargs)` verbatim (never inventing the ~13-argument set itself), and writes the result to a **new, timestamped checkpoint directory** — refuses outright (`FileExistsError`) if that directory already exists and is non-empty, so `results_kinetic_chain_gen3_tcp/1` can never be overwritten. **The new checkpoint is never thrown automatically** — restart at start-of-day and the speed-scale ladder like any other. **`T_control` must be passed to `reinforce_policy` in seconds, not a pre-divided step count** — `reinforce_policy` divides by `T_sampling` internally. `train_mc_pilot_pb_arm.py` always passed seconds correctly; `adapt_policy_height.py` passed an already-divided step count (a control horizon ~50× too long) and **was fixed 2026-09-03 16:40**. Every height-adaptation checkpoint on disk postdates the fix (`results_kinetic_chain_gen3_tcp_h{10,20,30}` written 16:41–16:44, and their `config_log.pkl` records `T` as a duration in seconds, e.g. 0.5808), so **no reported height-generalization number was affected** — verified 2026-09-10. Nothing here is still open.
For a tennis ball at this rig's actual release speed (≈1.44 m/s) and range, the measured numbers say **flight-drag learning sits below the noise floor** (≈5 mm drag deviation over the flight vs. ≈18 mm extrinsic + ≈10 mm triangulation noise, well under even the corrected threshold) while the **release discrepancy sits well above it** (25 ms command quantisation alone is 2.9–3.7 cm of landing error, 6–7× the flight-noise floor) — so the flight GP update is expected, and reported, as a `BELOW NOISE` no-op on this data, and the release-model fit is where a real correction is expected to come from. A high-drag ball (whiffle) would move drag above the noise floor but needs its own trained checkpoint first — out of scope here.
**Two items this build left explicitly open, both procedural rather than code:** (1) `--wrist_roll_offset_deg` (finger clearance) was tuned for the old checkpoint's 5° release; this checkpoint releases at 15° and the angle **must be re-verified visually on the arm** at `speed_scale=0.15` with an empty gripper before any ball is loaded — the numeric precheck passing is feasibility, not proof the fingers clear. (2) `tune_ir_exposure.py` has never been run — `HARDWARE_RUNBOOK.md` §6's exposure/emitter table is still blank; run it once, camera-only, before the first real throw of a session.

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
- `robot_profiles.py` — per-arm `RobotProfile` dataclass: URDF, joint ids, EE link, `q_neutral`, `qd_max`, `tau_max`, `kp`/`kd`, `windup_delta`, `speed_bounds`, `control_mode`. **`control_mode` is the thing to check first**: `kinematic` (ball velocity assigned — idealized, NOT a physical throw), `position`, `torque` (computed-torque + gravity comp + Jacobian-transpose payload term — the real one). Profiles: `kuka_iiwa`, `franka_panda`, `franka_panda_dyn`, `kinova_gen3`, `kinova_gen3_dyn`, `ur7e`, `ur7e_dyn`, `xarm6`. The `_dyn` suffix means torque mode. Also carries **`roll_idx`** — the joints the release LP freezes at qd=0; see the UR7e section below for why that stopped being a constant.
- `arm_controller.py` — IK, `plan_throw` (3-phase neutral→windup→release + follow-through, with torque/velocity feasibility checks on **all** phases), gripper. Shared by sim **and** hardware.
- `kinova_hardware.py` / `run_hardware_throw.py` — safety-gated Kortex executor (dry-run default, speed_scale time-stretch, hard qd clamp, whole-trajectory precheck that fails closed). `HARDWARE_SETUP.md` is the safety model + reference; `HARDWARE_RUNBOOK.md` is the run-day page.
- `hw_readonly_check.py` — opens a Kortex session, reads, closes. **Zero writes** (`connect` by contrast writes the teardown `stop()`). Run it first at the lab. `measure_gripper_latency.py` — 1 kHz UDP feedback latency calibration.
- **Operator launch tools (added 2026-08-22/27) — check here before assuming a capability needs building.** `throw_gui.py` (Aug 22): minimal Tkinter form wrapping `pickup_and_lift.py` then `run_hardware_throw.py throw --arm --confirm` — fields for IP/checkpoint/target/speed_scale, a confirm checkbox that **unchecks itself after every run** (re-affirm each cycle, not just once), scrolling subprocess-output log. `pickup_and_lift.py` (Aug 22): moves to the recorded `pickup_pose.json` pose, closes the gripper, and **verifies a real grasp happened** (58-59% closed on a real ball vs ~99-100% closing on nothing) before allowing a throw — refuses to proceed on a false grasp rather than throwing an empty hand. `run_closed_loop_throws.py` (Aug 26/27): one-throw-per-invocation CLI — dashboard → confirm → throw → append a structured JSONL record (`throw_index`, `ball_id`, `q_release`/`qd_release`, `exec_stats`, `capture_file`, `landing_xy`). **Decoupled by default**: `landing_xy` stays `null`, filled in by a separate offline `measure_landing.py` pass matched by `throw_index` — pass `--measure --extrinsic <file>` to opt into polling `throws_dir` for `throw_capture.py`'s recording and filling it in immediately instead. `closed_loop_gui.py` (Aug 27): same shape as `throw_gui.py`, wrapping `run_closed_loop_throws.py`. None of these duplicate `run_hardware_throw.py`'s planning/execution — all of them import and call it directly.
- **Read/write status (updated 2026-08-22):** read-only paths, the gripper, and **the full throw trajectory as a joint-speed stream** have all run on the real arm — escalated `speed_scale` 0.15 → 0.30 → 0.60 → 1.00, **gripper empty, no ball, no landing**. Open-loop velocity streaming is viable: drift at release 0.0171 rad ≈ 1.0 cm of landing error against a 2.89 cm sim accuracy (2026-08-22 re-run after a gripper-loop fix, see below: 0.0318 rad, still within budget but ~1.9× the 08-10 figure — noted, not yet explained). `connect`, `home`, and `gripper` have all now actually been accepted and executed by the arm — the "no write path has ever been accepted" claim below is stale. **Still never executed: a throw with a ball in the hand.** The TCP-offset blocker below (found 2026-08-22) is **fixed in code as of 2026-08-27** — it no longer blocks a ball throw on methodological grounds. What's still missing is procedural, not a code fix: (1) `--wrist_roll_offset_deg` (finger clearance) was tuned for the old checkpoint's 5° release; the new checkpoint's release posture is a substantially different 15° elevation and this **must be re-verified visually on the arm**, not assumed safe just because the numeric precheck passes; (2) no `T_B_C` camera extrinsic exists on disk yet, so `--measure` can't close the loop even though the throw itself doesn't strictly need it.
- **`HardwareThrowExecutor.set_gripper()` fixed 2026-08-22.** It used to fire-and-forget `SendGripperCommand` and was observed to produce **zero motion** via the standalone `gripper` CLI (session teardown raced the async motor). Now blocks and confirms via `read_gripper()` feedback by default. The in-loop release call inside `rehearse_or_throw` must keep `confirm=False` — confirming there stalls the 40 Hz control loop (measured: a single 693 ms late tick, 0.38 rad drift) since polling blocks the same thread that has to keep streaming joint speeds.
- **Gripper TCP offset — found 2026-08-22, FIXED 2026-08-27.** The release point/velocity model had always assumed a zero-length end effector: `profile.ee_link` (`end_effector_link`, the bare wrist flange) is what `release_solver.py`'s LP solved against, and the sim's ball attached there too (zero offset) — consistent between sim and hardware, but neither accounted for the real Robotiq 2F-85's reach. The offset was not a guess — `ControlConfig.GetToolConfiguration()` (read-only) reports `tool_transform = (0, 0, 0.12) m, zero rotation`, `tool_mass = 0.831 kg` (a real 2F-85, configured deliberately). At the trained release state, checked empirically (not just the manual `v_true = v_flange + ω×r_offset` estimate below): `r_offset` in world frame came back **99.98% vertical** (`[0.002, 0.0001, 0.12]`), so position-wise the release point simply moves up ~12cm — but the wrist is *rotating* at release (ω≈3.25 rad/s), and a point rigidly offset from a rotating body picks up an independent velocity term regardless of how small the position shift's horizontal component is: **+0.39 m/s (+26%) speed, only 1.24° direction change** — confirming the original firmware-based estimate almost exactly.
  **The fix, and why the obvious-looking alternative was wrong:** the first attempt was to fully re-run `find_throw_pose.py --tool_offset_z 0.12` and swap in the resulting table — this *seemed* like the direct fix but was actually a **different, incompatible throw**: the search re-optimizes for max range under the new physics and picked a different corner solution (elevation 5.0°→15°, kinematic max 1.628→2.07 m/s), so a checkpoint trained on the old table's *direction* would launch on a completely different trajectory shape if fed through the new table — reproducing the exact "hardware executes a different throw than the simulator trained" bug `release_solver.py`'s own docstring warns about, just via a different table instead of a different planner. **The real fix was retraining under corrected sim physics, not patching around uncorrected sim physics**: `arm_controller.py::attach_ball` had a real (previously invisible, since the offset was always exactly zero) bug — it computed the weld offset as a world-frame delta but passed it to `createConstraint`'s `parentFramePosition`, which PyBullet requires in the *link's local frame*; fixed via `p.invertTransform`. `model_pybullet.py::PyBulletThrowingSystem` gained a `tool_offset` param (default zero — every existing checkpoint/table/test is unaffected) that now welds the ball at the TCP, not the flange, so the sim's own rigid-body physics produces the `ω×r_offset` boost **for free**, no manual formula anywhere in the code. `OptimizedReleaseSolver` (`release_solver.py`) and `find_throw_pose.py`'s search both take the same `tool_offset`, applied via PyBullet's own `calculateJacobian(..., localPosition=tool_offset, ...)` — no manual cross-product needed there either. Tables are now stamped with `tool_offset` (mirroring the existing `floor_z` stamp pattern) and `run_hardware_throw.py`/`eval_adapted_height.py` refuse a mismatched `--tool_offset_z` against a table's stamp, fail-closed.
  Retrained checkpoint (`results_kinetic_chain_gen3_tcp/1`, table `throw_pose_table_tcp.npy`) evaluated **better** than the old one on a fresh unused seed — see "Which checkpoint" above. A real (currently-invisible, since it's always exactly zero) safety-check bug was also found and fixed along the way: `release_box_from_table` was building the safe-release box from the flange position while `solve()` now correctly reports the TCP — a correctly-solved TCP release was failing the box check purely because the box itself hadn't moved.
  A reference URDF with real Robotiq 2F-85 kinematics exists at `robot_arm/_urdf_cache/gen3_robotiq2f85.329d9bba3a1a.urdf` (primitive collision geometry, not wired into `N`/`N_FULL`/`joint_ids` anywhere) — its `gripper_mount` joint is kept at identity and was **not** the source of truth for the offset (that link chain's own ~13cm reach is from a different, uncertain generic-gripper source; stacking it on the firmware's 0.12m would double-count). Still unresolved: `--wrist_roll_offset_deg` needs re-verification on the real arm for the new 15° release posture (see above), and no `T_B_C` camera extrinsic exists on disk.
- `noise_models.py` — velocity slip, salt-and-pepper, `ReleaseTimingJitter` (wired only into `test_mc_pilot_pb_C.py`, the paper-faithful KUKA noise demo; never applied to kinova — the real gripper latency is compensated in `kinova_hardware.py` instead).
- **Camera-base extrinsic calibration — DONE 2026-08-31, `calib/T_B_C.npz` exists.** This was the last blocker on "close the loop" (`HARDWARE_SETUP.md`). **`start_of_day.py` is now the run-day entry point**: one command, one GO/NO-GO, covering env → arm (`hw_readonly_check`) → cameras → calibration → gates → `plan`. It writes the canonical extrinsic (plus a timestamped archive and an audit `.json`) and **refuses to write it if any gate fails**, so nothing downstream can silently pick up a bad calibration. First real result: board recovered to **1.2 cm** of the true floor plane, PnP-vs-depth **0.8 cm**, reprojection 0.20/0.15 px, `plan` PASS + safe-box True.
  **The measured accuracy bound is REPEAT ≈ 1.8 cm / 0.56°** (5 frame pairs solved independently off a stationary rig) — not the 0.15 px reprojection error, and not the sub-mm synthetic numbers in the ball-tracking bullet below. That is the same order as the checkpoint's own 1.90 cm sim accuracy, so **one real landing cannot currently resolve a sim-vs-real gap smaller than ~2 cm**. Improving it needs physical change (bigger board, closer camera, multiple arm poses), not more frames.
  `perception/wrist_chain.py` is the single implementation of the chain (FK → board detect → compose → gates), shared by `start_of_day.py`, `calibrate_via_wrist_camera.py` (ChArUco board) and `scripts/calibrate_marker_tf.py` (single marker) — same rule as `release_solver.py`: never inline or copy it. `tests/test_wrist_chain.py` covers it (16 tests). `calibrate_camera_extrinsics.py` (tape-measured board placement) is superseded; prefer the FK path, per `feedback_extrinsic_calibration_method` in project memory. `make_aruco_targets.py` / `make_aruco_printable.py` still generate the print-ready exact-scale targets (measure the printed ruler — the SCALE gate now catches a mis-scaled print, but only after the fact). `plan_camera_mount.py` plans where to put the camera; `../cam_snapshot.py` is an aiming diagnostic, not a calibration step.
  **Two real bugs were found doing this, both silent:** (1) the tool-frame error above — see "Things that will bite you"; (2) `CharucoDetector.detectBoard` returns **zero** corners when any marker id appears twice in frame, which reads as "board not visible" rather than "ambiguous id". The overhead D435i sees the board *and* the loose check-point markers `make_aruco_targets.py` tells you to lay out, one of which duplicated a board id — 15 markers, 0 corners. `detect_board_pose` now drops both copies of a duplicated id before interpolating (17 corners, 0.13 px after).
- **Ball-tracking landing measurement (`perception/`, new 2026-08-25, first real-frame validation 2026-08-26).** Measures where a thrown ball **first contacts the floor**, not where it settles — on a tile floor a ball thrown at ~1.6 m/s bounces and rolls, so the resting position `perception/ball_detector.py` finds is a systematic offset from first contact, and first contact is the quantity the policy's landing error is defined against. Does this by triangulating two **independently detected** IR blob centroids from the D435i's rectified `infrared,1`/`infrared,2` pair (89.7°x58.8° FOV, global shutter, distortion exactly zero at 848x480 — see `docs/superpowers/specs/2026-08-25-ball-tracking-design.md`), then Gauss-Newton fitting a parabola to **pixel reprojection error** (not the triangulated 3D points, which carry correlated range-dependent noise) and solving for the descending root of `z = z_floor + ball_radius`. Modules: `perception/stereo.py` (rectified pair → camera-frame 3D, no undistortion needed), `perception/ball_track.py` (frame-diff-against-median-background blob centroid), `perception/trajectory.py` (RANSAC ballistic association — rejects arm-motion and reflection outliers because they don't fit `g=9.81`, then the fit + impact solve), `perception/ir_capture.py` (dual-IR ring-buffer recorder). **This is explicitly NOT the block-matching depth map this repo already rejects as a position source** (`ray_plane.py`, `HARDWARE_SETUP.md`: ~2-4 cm at 1-2 m, worse for a small textureless sphere) — triangulating two independently-detected 2D centroids is a different operation with a different error model; depth stays usable only as a coarse segmentation gate.
  **CLI tooling has moved on from what's below** — `throw_capture.py` (live dual-IR viewfinder with a range-gated auto-trigger: only records a candidate detected in BOTH imagers with disparity between `--min_range`/`--max_range`, which rejects a hand/forearm sweeping through frame on arithmetic, not tuning) has superseded `record_throw_ir.py`'s fixed-countdown capture; `calibrate_camera_extrinsics.py` is the actual extrinsic tool (see above). `measure_landing.py` and `tune_ir_exposure.py` (exposure/emitter A/B, not yet run — see `HARDWARE_RUNBOOK.md` §6) are unchanged.
  **First real camera frames went through the pipeline 2026-08-26**: 20 real dual-IR events captured (`mc-pilot-pybullet/throws/`, untracked). Best tracks: `throw_003.npz` (119/130 frames paired), `throw_001` (117/130), `throw_002` (105/130). Rendered as an annotated video + 3D triangulated scatter and **visually confirmed** — the detection circle sits on the ball in both IR streams throughout, the 3D track is a smooth continuous arc. This validates the 2D-detect / stereo-triangulate stages on real data for the first time; **`ransac_track`/`fit_ballistic`/`solve_impact` (i.e. the full `measure_landing.py` path) are still untested on real data**, since no current-mount `T_B_C` exists yet (see above) — running that end-to-end is the next real step once a fresh extrinsic is saved to a file. Separately, OSS research found no drop-in replacement for this pipeline; the one candidate worth a look is [MyPTV](https://github.com/ronshnapp/MyPTV) (MIT) for `stereo.py`'s triangulation specifically.
  **2026-08-31: the full `measure_landing.py` path ran on real data for the first time and correctly refused all 20 recordings** (best inlier fraction 0.49 vs the 0.60 gate). Root cause is the data, not the fitter: fitting acceleration freely in the camera frame — where `T_B_C` cannot enter — gives `|a|` = 0.16–2.54 m/s² across spans of 1.4–1.7 s, against 9.81 for free flight, and `throw_001` fits a *straight line* to 0.9 cm median residual over 128 points. The 08-26 clips are a hand-carried or rolling ball. Detection and triangulation are confirmed excellent by exactly that fit; `ransac_track`/`fit_ballistic`/`solve_impact` remain untested on real data because **no recording of a genuinely airborne ball exists yet**. A deliberate hand toss with `throw_capture.py` running would close this out without any arm motion.
  Prior to 2026-08-26, this section was **verified synthetically only**: 147 tests pass, an end-to-end synthetic parabola projected through the real measured intrinsics/baseline recovers the landing point to 0.18 mm, Gauss-Newton at 0.15 px pixel noise over 40 frames/30 seeds averages 0.47 mm, RANSAC separates 40/40 true detections from 12 injected arm-like outliers. `achieved_fps` (the ~44-usable-frames-at-90fps budget) and the `IRRecorder` exposure/emitter defaults (`exposure_us=2000`, `emitter=True`) are still reasoned, not measured. **Absolute accuracy is still bounded by `T_B_C`, not by the vision** — every millimetre figure above (synthetic or the 2026-08-26 real-frame numbers) says nothing about absolute correctness against the arm's base frame until a current, on-disk extrinsic exists.

## Second arm: UR7e (sim model + trained sim checkpoints 2026-09-06; no hardware)

Universal Robots UR7e is being brought up alongside the Gen3. **Only the sim model, profiles and freeze-set plumbing exist** — no pose table, no checkpoint, no hardware layer. Do not treat any UR7e number below as a result.

- **`scripts/install_ur7e_urdf.sh` builds the model and IS the provenance record.** Fetches `UniversalRobots/Universal_Robots_ROS2_Description` **tag 4.3.1** (pin the tag — the repo's default `ros2` branch has no `ur7e` at all, and that is where most search results and stale checkouts land), xacro-expands it against a throwaway ament overlay, and vendors the result to `pybullet_data/ur7e/` the same way `kinova_gen3/` is vendored. Idempotent; verifies the load and the mass at the end. Needs `/opt/ros/humble` for `xacro` (override with `ROS_SETUP=`).
- **The upstream `config/ur7e` is largely inherited from the UR5e, and that is mostly legitimate.** `physical_parameters.yaml` is byte-identical to ur5e's, `default_kinematics.yaml` is ur5e's numbers, `visual_parameters.yaml` points at `meshes/ur5e/` (there is no `meshes/ur7e/`), and `joint_limits.yaml`'s header cites the *UR5e* manual. But UR themselves ship one shared "UR5e/UR7e" JT file and one shared working-area PDF — same 850 mm reach, 20.6 kg, ⌀151 mm footprint. Same mechanics, hotter joints. **Verified for the UR7e specifically: `qd_max` = 180 °/s = 3.1416 rad/s on all six** (matches the tech sheet). **Provisional, UR5e-derived: `tau_max` 150/150/150/28/28/28 Nm and every link mass/inertia.** UR's public max-joint-torque article has no UR7e row; the 7.5 kg payload says the real limits are higher, so the precheck fails closed — but measure via RTDE `actual_current`/`target_moment` before publishing any torque-headroom number for this arm.
- **The release LP's freeze set is no longer a constant.** `(0, 2, 4, 6)` is the *7-DoF alternating roll-pitch-roll* layout of the Gen3 and Panda, and it was hardcoded in both `release_solver.py` and `find_throw_pose.py`. It is now `RobotProfile.roll_idx`, read via `robot_profiles.roll_indices()` and passed by **both** `PyBulletThrowingSystem` and `run_hardware_throw.py` (they must agree — same rule as everything else in `release_solver.py`). Every 7-DoF profile still resolves to `(0,2,4,6)` bit-identically, so no existing checkpoint, table or result moved. **On a 6-DoF arm the old constant was a latent silent bug**: index 6 is the LP's *speed slack variable*, not a joint, so freezing it pins the release speed to exactly 0 m/s and the LP still reports success. `tests/test_ur7e_profile.py` (11 tests) pins all of this.
- **UR7e freeze set is `(0, 4, 5)`, classified numerically off the PyBullet Jacobian** at a candidate release pose — pan (all motion out-of-plane), wrist_2 (all out-of-plane), wrist_3 (`|Jv|` exactly 0). Carriers are shoulder_lift / elbow / wrist_1 — three, same count as the Gen3's, so the LP structure is unchanged. Note: the axis-vs-base→EE-vector method used for the Gen3/Panda gave the *wrong* answer here through a frame-convention slip; the Jacobian columns are convention-free, use those.
- **`ee_link = 10` is `tool0`.** `flange`, `ft_frame` and `tool0` are co-located but differently rotated, and only `tool0` gives the z-out-along-the-tool convention that `tool_offset=(0,0,L)` assumes. `wrist_3` is the exact analogue of the Gen3 joint that `--wrist_roll_offset_deg` exploits — provably free for finger clearance, but only while the TCP offset stays purely axial.
- **Phantom mass, again.** The generated URDF declares six bodyless links (`world`, `base_link`, `ft_frame`, `base`, `flange`, `tool0`); PyBullet gives each 1 kg, three of them at the wrist. Raw load is **26.700 kg** against 21.700 kg of real links. `urdf_fixup.repair_massless_links` handles it and `ArmController` already routes through it — the same defect the Gen3's camera frames had.
- **This arm is TCP-speed-limited, not joint-velocity-limited — the opposite of the Gen3.** The forward-release LP sweep gives **4.157 m/s**, which *exceeds* the tech sheet's 4 m/s max TCP speed. Capped at 4 m/s that is **1.93× the Gen3's 2.07 m/s and 2.14× its range** (2.00 m vs 0.9355 m off the same 0.433 m plate). Consequences: every Gen3-trained target band, cost lengthscale and RBF lengthscale init is invalid here, the safety config will clamp before the joints do, and the lab needs ~2 m of clear floor.
- **The throw plane does not pass through the base axis** — UR's wrist carries a 0.1333 m y-offset. Train with `--flight_targets`; that flag exists for exactly this geometry.
- **`kp`/`kd`, `timing`, `windup_delta`, `speed_bounds` are all unvalidated placeholders.** `kp`/`kd` are scaled from the Gen3's 400/60 by the `tau_max` ratio (~3.85) on a 21.7 kg arm vs ~7 kg — same warning as `franka_panda_dyn`: gain-sweep before trusting a torque number, and remember saturation and insufficient stiffness look identical until you raise `kp`. `timing`/`windup_delta` are carried over from the Gen3 and must be checked through the real planner on all three phases.
- **Not built yet, in order:** pose search (`find_throw_pose.py --robot ur7e_dyn`) → retrain from scratch (Gen3 checkpoints are invalid — different arm, different speed regime, and "re-search ≠ correct a checkpoint" applies doubly) → an RTDE hardware backend. On the hardware side `kinova_hardware.py` already has a clean backend seam (`_DryRunBackend`/`_KortexBackend`: `connect`/`disconnect`/`read_joint_state`/`send_joint_velocities`/`send_gripper`/`read_gripper`/`open_realtime_feedback`/`close_realtime_feedback`/`stop`), so `ur_rtde`'s `speedJ` slots in — but `HIGH_LEVEL_MAX_HZ = 40.0` is a Kinova number (RTDE is 500 Hz on e-Series, which would remove the 25 ms / 2.9–3.7 cm quantisation floor), `SoftLimitManager` is Kortex-only (UR's equivalent is pendant-gated safety config, not remotely writable), the `[0,360)` joint wrap and `home()`'s continuous-vs-limited split do not apply (all six UR joints are ±360°), and `perception/wrist_chain.py` hardcodes `gen3.urdf` links 7/10.


### Five arm-specific assumptions the UR7e exposed

All five were hardcoded 7-DoF Gen3 facts. **None produced an error** -- each silently planned a
different throw. Every one is now derived from the profile or measured, with the legacy value as
the default, and every 7-DoF arm is bit-identical (verified: the raw Gen3 ablation reproduces
published Table I digit for digit after all five changes).

| Where | Was | Now |
|---|---|---|
| `release_solver.py`, `find_throw_pose.py` | freeze set `(0,2,4,6)` | `RobotProfile.roll_idx` via `roll_indices()`; UR7e `(0,4,5)`. On a 6-DoF arm index 6 is the LP's **speed slack**, so the old constant pinned release speed to exactly 0 and reported success |
| `find_throw_pose._fkj`, joint-limit reads | `range(N)` joint indices | `JOINT_IDS` from the profile. A UR URDF leads with two FIXED joints, so its arm joints are 2..7 |
| `find_throw_pose`, `paper_ablation_feasibility` | literal `[0,s2,0,s4,0,s6,0]` grid | `pitch_indices()` / `sagittal_q()` |
| `train_mc_pilot_pb_arm.py:332` | `for _j in range(7)` reading the table's release posture | indexes `profile.joint_ids`; also ran off the end of a 6-element entry |
| `build_table_by_rotation`, `release_solver` | base rotation `q[0] -= az` | `base_rotation_sign()` measures it, stamped into the table like `floor_z`/`tool_offset`, solver defaults to -1 for unstamped legacy tables |

**The base-rotation sign is the one to remember.** Measured: **Gen3 -1, but Panda +1, KUKA +1,
UR7e +1** -- so the "verified sign" comment in both files was a Gen3 fact stated as a general one.
Backwards, it aims the throw the wrong way round: on the UR7e landing y moved OPPOSITE target y,
11 cm on-axis degrading to 147 cm at +-30 deg azimuth. **No published number is affected** -- all
23 pose-table checkpoints on disk are Gen3. `franka_panda_dyn_throw_pose_table.npy` IS latently
wrong (unstamped, so read as -1, but the Panda is +1); it was never trained through, and must be
re-searched before use.

### Cartesian TCP-speed ceiling (`RobotProfile.v_tcp_max`)

The release LP was bounded by joint velocity only. On the UR7e that returns **4.65 m/s against a
rated 4.0** -- the first search produced 23/23 table entries the controller would clamp or refuse,
i.e. the arm executing a different throw than the simulator trained. Never bound on the Gen3
(2.07 m/s), so it went unnoticed until a second arm. `v_tcp_max` is 4.0 for the UR7e and **None
everywhere else**, leaving every existing LP bit-identical. Applied in BOTH LP implementations
(`aimed_speed` and `OptimizedReleaseSolver`) and passed by both the sim and hardware call sites.

Useful consequence: once the cap binds, the UR7e search is speed-limited rather than
torque-limited, so its raw and repaired pose tables come out **bit-identical**. UR7e results do not
depend on the phantom-mass decision below.

### kp/kd came from a real sweep, and kd was the trap

The first guess (Gen3's 400/60 scaled by the tau_max ratio to 1500/150) was **unstable**: measured
release speed 9.5 m/s against a commanded 4.0 -- the controller injecting energy, not tracking,
with `time_scale`/`clip_scale` both reporting 1.0 and nothing flagging it. A 2-D sweep shows
**kd, not kp, is the sensitive axis**: every kd >= 160 blows up at every kp from 100-800, while kp
is nearly flat over 200-800 once kd <= 80. Shipped values keep the Gen3's proven kd/kp = 0.15 and
scale the wrists by the tau ratio. Re-sweep if the timing or pose table changes.

Note this is the documented "raising gains hurts => check torque headroom" diagnostic giving a
*third* answer: neither saturation nor stiffness, but integrator instability at 50 Hz.

`gpr_lib/` is the upstream GP math layer — library code, rarely edited.

## Things that will bite you

**Methodology**

- **The pose search runs under PHANTOM MASS, and it changes the paper (found 2026-09-06).**
  `find_throw_pose.py` and `paper_ablation_feasibility.py` load the URDF **raw**, while
  `ArmController` (and therefore every rollout, eval and hardware precheck) repairs bodyless links.
  On the Gen3 that is **+3.00 kg / +46%** hung off the wrist: 9.491 kg vs a real 6.491, mean peak
  gravity torque 0.696 of limit vs 0.324 (**2.15x**, matching the 2.1-2.4x measured against the
  arm's own torque sensors), and 16.8% of postures rejected at the very first gate that are in fact
  feasible. **The KUKA and Panda have no bodyless links at all**, so the paper's Gen3-vs-Panda
  feasibility contrast is partly an artifact of a defect present on only one side.
  Measured A/B (the raw column reproduces published Table I digit for digit):
  survival after the full cascade **16.6% -> 82.3%**, windup-path rejection **54.4% -> 0.0%**,
  best safe-throw range **0.824 -> 1.031 m** (i.e. equal to the instant-only best -- the "range
  decreases after whole-trajectory validation" claim does not survive). The torque sweep moves too:
  saturation begins at **k=1.00 instead of k=2.25**, so the real Gen3 is *already at* the kinematic
  residual rather than torque-starved. **What survives exactly** is the section's actual
  conclusion -- rejection saturates at **17.7%**, entirely follow-through, kinematic in origin --
  and the corrected sweep shows the windup->ramp->follow-through handoff that the published
  version could not (its k=0.50/0.75 rows had 0% survival, so no decomposition existed).
  The Panda control run reproduces its published numbers **exactly** (152,844 -> 152,844, 100%),
  confirming phantom mass is the only variable. `REPAIR_INERTIALS` / `--repair_inertials` toggles
  it, **default False = bug-compatible with every existing table and paper number**; the error is
  conservative, so no shipped checkpoint is unsafe. Regenerated JSONs live in
  `mc-pilot-pybullet/paper_icra2027_ablation/`. **ADOPTED into the paper 2026-09-10**:
  Tables I/II, Fig. 3 and the range-ceiling figure now all report the repaired
  numbers, with the raw ones kept in an explicit provenance paragraph. Two
  further provenance gaps were found and closed while adopting them:
  `paper_ablation_feasibility.py` did not record its `t_throw` and defaulted to
  1.1 s while the deployed planner uses `T_R` = 1.6 s (now a `--t_throw` flag,
  recorded in the output JSON), and `paper_range_speed_sweep{,_tcp}.py` called
  `loadURDF` directly instead of `find_throw_pose.load_arm()`, so they could not
  honour `REPAIR_INERTIALS` at all (now routed through the loader, with a
  `--repair_inertials` flag). Figure provenance is selected by
  `ABLATION_VARIANT` in `paper_icra2027/figures/make_figures.py`.
- **Training cost is not accuracy, and neither is a clean planner report.** On the UR7e the planner
  reported `time_scale=1.0`, `clip_scale=1.0` and `v_planned == v_cmd` while the arm actually
  released 11-24 cm away from the intended point with the velocity pointing *downward*. Three
  independently-trained seeds then evaluated to byte-identical numbers -- the tell that the policy
  was not reaching the ball at all. Check `last_release_info`'s `release_pos_err` and
  `v_release` vs `v_planned`, not just the scale factors.

- **The model-belief trap.** `cost_trial_list` / "Final trial cost" is computed by simulating particles through the *learned GP model*, not real physics. It can sit near zero while real accuracy is off by 17–28%. Never report training cost as accuracy — always re-evaluate through `PyBulletThrowingSystem.rollout` (`eval_baseline.py` etc.) with **fresh, previously-unused RNG seeds**.
- **Assigned vs real dynamics.** `kinematic`-mode profiles set the ball's velocity directly (`resetBaseVelocity`); the arm is cosmetic. Never present those numbers or videos as a physical throw. Label kin vs dyn explicitly, every time.
- **Visually verify renders.** Extract and view actual frames before claiming a motion looks right. Numeric checks alone have missed a corkscrew throw, a zero-amplitude windup, and an arm that never moved — repeatedly.
- **Feasibility ≠ realization.** An LP plus a single endpoint torque check can pass a pose that the real `plan_throw` windup structure makes torque-infeasible. Validate through the real planner and real dynamics.
- **"Re-run the search under corrected physics" is not the same operation as "correct a trained checkpoint."** Found 2026-08-27 fixing the gripper-TCP-offset bug: a full `find_throw_pose.py` re-search under newly-correct physics re-*optimizes* for max range and can pick a different corner solution entirely (here: elevation 5.0°→15°) — which silently produces a *different throw*, not a corrected version of the old one, and a checkpoint trained on the old geometry is not valid against it. If the goal is "make an already-trained checkpoint execute correctly under a previously-missing physical effect," the checkpoint has to be retrained under the corrected sim, not have its execution table swapped post-hoc.

**Physics / control**

- **An arm on a base plate is modelled by raising the base, never by a negative `target_height`.** The real Gen3 sits on a **0.433 m** plate and throws to the floor. `plane.urdf` is a real collision plane at world z=0 and the landing test only fires on a *descending crossing* of `target_height`, so `target_height < 0` means the ball rests on the floor without ever crossing, the loop runs out, and the rollout returns the ball's resting position as if it were a landing — plausible numbers, no error. Use `--base_height` (trainer) / `base_height=` (`PyBulletThrowingSystem`); negative `target_height` now raises. **Frames:** world = floor at 0, base at `+base_height`; base frame (hardware, pose tables) = base at 0, floor at `-base_height`.
- **A pose table's `range` is meaningful only against the floor it was searched for.** `find_throw_pose.py`'s range objective hardcoded the floor at base-frame z=0 until 2026-08-12, so every table shipped before then assumes a floor *level with the base*. Re-searched at the real `--floor_z -0.433`, the winning **release state came back bit-identical** (`max|Δq| = max|Δq̇| = 0`, same 1.6281 m/s, same 5.0° elevation) and only `range` moved, 0.7999 → 0.9355 m: the optimum is a corner solution with elbow and wrist saturated at their velocity limits, so dropping the floor rescales the objective without reordering candidates. **Measured for this arm and wedge, not a general law** — re-check it for any other arm. What the stale floor did corrupt is the trained **target band** (0.60–0.80 m against a true reachable ~0.935 m). Tables now carry a `floor_z` stamp and the trainer refuses a stamp that disagrees with `--base_height`.
- **`GetMeasuredCartesianPose` reports the TOOL frame, not the flange (found 2026-08-31).** This arm has `tool_transform = (0, 0, 0.12) m` configured for the 2F-85, so that call returns a pose 12 cm past `end_effector_link`. Both calibration scripts composed the URDF's *flange*->camera offset onto it and shipped a 12 cm error into `T_B_C`; the board came out 14.0 cm below the floor it was lying on. Verified against PyBullet FK: the reported pose sits `[-0.0035, -0.0052, +0.1251]` m from the flange, in the flange frame. **The fix was to stop using that call**, not to subtract the offset from it -- `perception/wrist_chain.base_to_wrist_camera` goes measured joint angles -> FK -> `camera_color_frame`, which also retires the never-verified "theta_x/y/z are intrinsic-XYZ degrees" assumption. This is the **third** time the 2F-85's 12 cm has cost something (release speed, release box, extrinsic): any frame that is off by ~0.12 m, suspect it first.
- **Reprojection error cannot validate a calibration.** A planar target's PnP absorbs a wrong principal point, a wrong Euler convention, or a "fit to page" printout into the *pose* and still reports a fraction of a pixel: the 12 cm tool-frame bug above reprojected at **0.18 px**. Only a fact from outside the model can catch it. The board lying on a floor whose height we already know is that fact, and it is what `start_of_day.py`'s FLOOR gate is; the depth sensor measuring the same board independently of the printed square size is the SCALE gate. Never accept "low reprojection error" as evidence a calibration is right.
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
- **`kortex_api` symbols are statically verified, and write paths now ARE accepted (updated 2026-08-22).** All 8 call groups resolve against the installed 2.6.0.post3 wheel; `connect`/`home`/`gripper`/empty-gripper `throw` at all four speed scales have all been executed live and accepted by the arm. Only a throw with a ball gripped remains unexecuted — no longer blocked on the TCP-offset fix (now shipped, see above), just not yet attempted since that fix landed.
- **`twist_linear` Cartesian ceiling — RESOLVED, was not a blocker.** Read directly off the arm (2026-08-22): `ANGULAR_JOYSTICK` (what the throw actually streams via `SendJointSpeedsCommand`) has soft `twist_linear = 0.0`, i.e. unset/not-applicable — the field is a Cartesian-mode concept, and the throw is joint-space. The 0.500 m/s hard ceiling belongs to `CARTESIAN_JOYSTICK` (the mode the arm happens to idle in), not the streaming mode. Confirmed both by this direct read and by the empty-gripper full-speed rehearsal completing clean with no fault.
- **Open hardware risks, in order:** (1) **`--wrist_roll_offset_deg` needs re-verification on the arm** — the new TCP-offset-aware checkpoint's release posture (15° elevation) differs substantially from the old one (5°) that the current 90° default was tuned against; passes the numeric precheck but that is feasibility, not realization (see "Feasibility ≠ realization" above) — must be checked visually, not assumed; (2) ~~the gripper TCP offset — was the top blocker~~ **fixed 2026-08-27** (see above), no longer open; (3) ~~the 67.9 ms gripper latency was measured static and unloaded~~ — **re-measured 2026-08-22 with a real ball loaded, 15 trials: onset 73.2 ± 10.3 ms, statistically indistinguishable from the static/unloaded figure.** `GRIPPER_RELEASE_LATENCY_S` is unchanged and now validated for onset specifically — still not validated against an actual real landing measurement, which nothing but a real landing can substitute for; (4) 25 ms command quantisation is a ~2.9-3.7 cm landing-error floor (scales with release speed) that looping faster cannot fix.
- **First real ball throws executed 2026-08-22** (`speed_scale=0.15` through `1.00`, several). Landing was not rigorously measured (still resting on the uncorrected TCP-offset release model above) — do not treat any as an accuracy data point. **Also: do not trust these as clean-release data points either** — see the release-during-streaming bug immediately below, found the same day after these throws. It is entirely possible the ball left the hand via arm momentum/deceleration g-forces rather than a completed OPEN command; nothing at throw time distinguished the two (no exception, no elevated tick latency — the command was simply never acted on). `pickup_pose.json` records a repeatable arm pose for reloading the ball between throws. `--wrist_roll_offset_deg` (new CLI flag on `plan`/`throw`) rotates joint 6 at release for finger/release-path clearance — provably free (that joint is frozen at qd=0 throughout the throw and is last in the chain, so it cannot change release position/velocity, only the gripper's own orientation) and goes through the normal precheck since windup must travel further to reach it.
- **BLOCKER, found and fixed 2026-08-22: `SendGripperCommand` is silently ignored by the arm for as long as `SendJointSpeedsCommand` is being actively streamed.** Reproduced with an isolated, fast test — arm not even moving, zero velocities streamed — the gripper does not move AT ALL while the stream is continuous; the identical command with the arm idle opens cleanly in under a second. No exception, no elevated tick latency, nothing in any prior log flagged it, which is why it went unnoticed through several earlier "successful" throws (see above). A **separate TCP session dedicated to gripper commands was tried and is wrong** — the arm explicitly rejects it (`KServerException ERROR_DEVICE/SESSION_NOT_IN_CONTROL`), it does not just ignore it; only the session currently in control may write anything, gripper included. **The actual fix**: `GRIPPER_RELEASE_PAUSE_S` (0.20 s, `kinova_hardware.py`) — a brief gap in the SAME session's `SendJointSpeedsCommand` stream right after the release-time OPEN command, giving the gripper write a real chance to land. During the pause the arm coasts at the exact release-instant velocity (not the planned follow-through deceleration) — checked against joint limits at the real release state: worst-case margin is 76.4° at a 300ms pause, so 200ms is very conservative. Verified 2/2 on the real arm (empty gripper): gripper reads fully open (0.87%) after both runs, no fault. Re-check the joint-limit margin for any different checkpoint/pose table — it's a property of that specific release configuration, not a general law.
- **`HardwareThrowExecutor.set_gripper()` needed a second fix same day**: the original confirm-via-feedback logic expected the gripper to reach its literal open/closed target, but grasping a real object means the motor legitimately stalls partway (measured: 58.08% closed on a tennis ball, position and velocity both dead stable) — the fix now also accepts "moved meaningfully from its start position, then held still" as success, and separately handles being asked to re-close a gripper that's already stalled on an object from a prior call (no further motion occurs, so there's no "progress" left to detect).
- **A real `ROBOT_IN_FAULT` was hit 2026-08-22, root-caused to a power supply issue** (reproduced independently via the arm's own physical controller, not through any script here — confirms it wasn't application-level). Re-sending a gripper `close` command to an already-stalled gripper (pushing against the ball again) was the proximate trigger and is worth avoiding, but the underlying cause was electrical, not code. `hw_readonly_check.py` confirmed clean recovery once power was stable — always re-run it after any fault before sending another command.
- Panda's `kp`/`kd` were copied from Gen3 untuned — different mass/inertia, verify before trusting torque numbers. (Panda's 9-DOF zero-padding *is* fixed and verified in both `find_throw_pose.py` and `ArmController`; the constructor now only rejects non-contiguous `dof_ids`.)
- PyBullet's `b3Warning[...]` stdout lines lack a trailing newline and merge with the next `print()` — a print can appear to "vanish" into a `grep -v` filter. Check for this before assuming a crash. Recover with `tr '\r' '\n'`.
