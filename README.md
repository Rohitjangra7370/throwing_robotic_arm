# MC-PILOT Throwing Arm

Model-based RL that teaches a robot arm to throw an object into a target bin
from a handful of real trials — a reproduction and hardware extension of
**MC-PILOT** (Turcato et al., "Data-Efficient Robotic Object Throwing with
Model-Based Reinforcement Learning," [arXiv:2502.05595](https://arxiv.org/abs/2502.05595)).

Course project (AR525, IIT Mandi — Reinforcement Learning in Robotics; FDP
Lab) that grew into a research track built around a real **Kinova Gen3 7-DOF**
arm.

**Start here:** [`REPRODUCING.md`](REPRODUCING.md) has copy-paste commands for
every number marked "re-run" below, with the exact output each should print.

## Results at a glance

| Result | Value | Status |
|---|---|---|
| **Real Kinova Gen3**, camera-aimed throws at a movable bin (2026-09-11) | **mean 1.91 cm** error vs the aim point, median 1.83, max 3.68 cm; 12/14 within 3 cm, 14/14 within 5 cm. **14 measured of 22 thrown** | Recomputed from the committed `results.csv` — see caveats below |
| Sim, Gen3 deployed checkpoint (`results_kinetic_chain_gen3_tcp/1`), 30 throws, fresh seed 24681012 | **mean 1.90 cm**, max 4.20 cm, 100% within 10 cm | **Re-run 2026-10-16**, reproduces exactly |
| Sim, 4 manipulators (KUKA iiwa7, Franka Panda, xArm6, Kinova Gen3) | 1.4–2.7 cm mean | Reported in `PROGRESS_REPORT.md`; not re-run for this release |
| Sim, target-height generalization (single policy) | 2.0 cm mean, 5.0 cm worst, 100 targets over h ∈ [0, 0.45 m] | Reported in `RESULTS_AND_METRICS.md`; not re-run |
| Sim, aerodynamic drag (tennis → whiffle ball) | learned dynamics beat a no-drag analytical baseline **3–7×** at high drag; the analytical baseline *wins* at low drag | Reported in `PROGRESS_REPORT.md`; not re-run |
| Sim, whole-trajectory-safe throw range, Gen3 | 1.03 m at floor level, 1.26 m for the deployed TCP-offset throw | Reported in `COMPARISON_VS_ORIGINAL_PAPER.md` §3.2; not re-run |

### Caveats that must travel with the hardware number

The full account is in
[`mc-pilot-pybullet/results_bin_game/session_20260911_024918/README.md`](mc-pilot-pybullet/results_bin_game/session_20260911_024918/README.md).
In short:

- **22 thrown, 14 measured.** The 8 others were refused by the
  landing-measurement pipeline (ball still rising in its last detected frame,
  no ballistic arc found, ball never entered view). The operator watched every
  one land on the marker, so none was a failed throw — but the refused throws
  skew toward negative y, so that side of the workspace is under-sampled.
- **The analysis pipeline was revised after the recordings were inspected**
  (14 measured is after three fixes; the original analysis recovered 4). This is
  disclosed, mitigated (each fix is pinned by regression tests on real
  recordings, and the recovered throws are slightly *worse* than the originals, not
  better) but not eliminated. One fresh session analysed with the pipeline
  frozen would remove the concern.
- **It is not the raw checkpoint's accuracy.** The trained policy still plans as
  though it throws ~45 cm short of where it lands; the aim comes from inverting a
  fitted release model (tool offset + release-speed gain) on top of it.
- **The error is a miss distance in one camera's frame** — marker seen vs ball
  seen to land, same camera, same extrinsic. One rig, one session, one tennis
  ball. The calibration's own repeatability is ~1.8 cm, so this cannot resolve
  sim-vs-real differences below roughly that.
- **The raw dual-IR recordings (22 × ~106 MB) are not in this repo.** The summary
  statistics are reproducible from the committed per-throw table; the
  recording → landing step is not, though SHA-256 prefixes of every recording are
  committed so a re-derivation can be checked against the same data.

### Caveats for the simulation numbers

- Training cost is computed through the *learned* model and can sit near zero while
  real accuracy is off by 17–28%. Every accuracy figure here is a re-evaluation
  through PyBullet rigid-body physics on seeds not used in training.
- `kinematic`-mode profiles assign the ball's velocity directly (the arm is
  cosmetic). Only `*_dyn` (torque-control) numbers describe a physical throw.
- `RESULTS_AND_METRICS.md` is a snapshot that predates the 2026-09-10 phantom-mass
  repair; its banner lists the rows that are superseded.

## Provenance and credits

This repo is a fork of [`dnfy502/ar525_project`](https://github.com/dnfy502/ar525_project).
Two phases:

- **Pre-fork (2026-03-25 → 2026-04-29), AR525 Group 3** (Aarya Agarwal,
  Bhumika Gupta, Rishang Yadav, Yajesh Chandra): five studies — baseline,
  elevated release, PyBullet arm physics + multi-arm + noise, wind
  robustness, vision (OpenCV + YOLOv8). **Done, frozen, cited as prior work**
  — not this project's results. Where this project re-checked their claims and
  found they did not hold, `PROGRESS_REPORT.md` §2 and §6 say so.
- **Post-fork (2026-07-02 → present)**: reliability fixes to the inherited
  baseline, a constraint-aware release-state search (posture + direction +
  speed, validated over the whole windup/throw/follow-through trajectory, not
  just the release instant), cross-manipulator and target-height
  generalization, an aerodynamic-drag study, and a full real-hardware bring-up
  on a Kinova Gen3.

`MC-PILCO/` is the unmodified upstream reference implementation by Mitsubishi
Electric Research Laboratories (AGPL-3.0, with some files under MIT — see
`MC-PILCO/LICENSES/`). Every `mc-pilot*/` directory is derived from it.

## Repository layout

Each `mc-pilot*/` directory is a **self-contained fork** — own `gpr_lib/`,
`simulation_class/`, `policy_learning/`, `model_learning/`, `envs/`. No
shared library; a fix to one must be ported by hand to the others if it
should apply everywhere.

| Directory | Study | Sim backend | Key addition |
|---|---|---|---|
| `MC-PILCO/` | — | — | Upstream reference implementation (MERL), unmodified |
| `mc-pilot/` | Baseline | NumPy ballistic | Ground targets |
| `mc-pilot-elevated/` | 1 | NumPy ballistic | Elevated release, stratified exploration |
| `mc-pilot-pybullet/` | 2 + **active work** | PyBullet | Real arm physics, torque control, multi-arm, Gen3 hardware track — see its own `README.md`, `HARDWARE_SETUP.md`, `HARDWARE_RUNBOOK.md` |
| `mc-pilot-pb-elevated/` | 3 | PyBullet | Elevated targets + arm physics |
| `mc-pilot-wind/` | 4 | NumPy + wind models | Constant wind / gusts / OU turbulence |
| `mc-pilot-pybullet-yolo/` | 5 | PyBullet + OpenCV/YOLOv8 | Vision-based bin detection |
| `status_update/` | — | — | Session handoffs (`HANDOFF.md`, newest entry on top), older progress docs |
| `paper_icra2027/` | — | — | ICRA draft source and figure scripts. **Work in progress, revised separately from this release** — the tracked copy is not final and may not match the results above |
| `old/` | — | — | Superseded, historical only |

Run everything from *inside* the relevant `mc-pilot*/` directory — each
resolves imports and data paths (`sys.path.append("..")`, `results_*/`,
`data/*.json`) relative to its own root.

**Size.** The repo is large (~1.7 GB of tracked files) on purpose: trained
checkpoints, training datasets and run logs are committed so the numbers above can
be re-derived without retraining.

## Setup

Python **3.10** (developed and tested on 3.10.12, Linux x86_64).

```bash
python3.10 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt            # simulation, training, evaluation, tests
python3 mc-pilot-pybullet/scripts/install_robot_assets.py   # Kinova Gen3 model -- required, see below
python3 -c "import torch, pybullet, cv2; print('ok')"

# optional extras
python3 -m pip install -r requirements-vision.txt     # YOLOv8 (Study 5 only)
python3 -m pip install -r requirements-hardware.txt   # RealSense SDK (+ notes on Kinova's kortex_api)
```

**The `install_robot_assets.py` step is not optional.** Every Gen3 code path loads the
arm as `pybullet_data/kinova_gen3/gen3.urdf`, from inside the *installed*
`pybullet_data` package, and the stock `pybullet` wheel does not ship that directory.
On the original development machine the model had been copied there by hand, outside
the repo, so nothing failed until the environment was rebuilt from scratch. The
vendored copy lives in `mc-pilot-pybullet/robot_assets/kinova_gen3/` (Kinova's
BSD-3-Clause `ros_kortex` model; see its `NOTICE.md` for provenance and the two edits
made to the upstream URDF). Run the script once per Python environment; it is
idempotent and refuses to overwrite a differing install without `--force`.

Use `python3 -m pip` and `python3 -m pytest`, never bare `pip`/`pytest`: bare
invocations can silently resolve to a different interpreter (Conda, a
Blender-bundled Python, ROS plugins on `PATH`). If `import torch` fails after you
installed it, check `which python3` first.

`kortex_api` (needed only for the real arm) is Kinova's own wheel and is not on PyPI;
it pins `protobuf==3.5.1`, which breaks other tools in the same environment.
`requirements-hardware.txt` explains how to install it — keep it in its own venv.

## Verify the install

```bash
cd mc-pilot-pybullet
python3 -m pytest tests/ -q        # ~35 s, no GPU, no hardware attached
```

| Environment | Result |
|---|---|
| `requirements.txt` + `install_robot_assets.py` | **303 passed, 15 skipped** |
| + `requirements-hardware.txt` (RealSense SDK) | 355 passed, 13 skipped |
| + Kinova's `kortex_api` wheel + the UR7e model (`scripts/install_ur7e_urdf.sh`, needs ROS 2 Humble `xacro`) | **368 passed**, 0 skipped |

Every skip names its reason (`pytest -rs`). If the Gen3 model is missing, pytest stops
immediately with the command to fix it rather than reporting dozens of URDF errors.

## Running the studies

```bash
# Baseline
cd mc-pilot/ && python3 test_mc_pilot.py -seed 1 -num_trials 10

# Study 1 — elevated release (NumPy), stratified exploration
cd mc-pilot-elevated/ && python3 test_mc_pilot_b_strat.py -seed 1 -num_trials 10

# Study 2 — PyBullet arm + noise + multi-arm
cd mc-pilot-pybullet/
python3 test_mc_pilot_pb_A.py -seed 1 -num_trials 10
python3 run_pb_noise_paper_multiseed.py

# Study 3 — PyBullet + elevated
cd mc-pilot-pb-elevated/ && python3 test_mc_pilot_pbe_B.py -seed 1 -num_trials 10

# Study 4 — wind
cd mc-pilot-wind/ && python3 run_all_wind_experiments.py --num_trials 15

# Study 5 — vision
cd mc-pilot-pybullet-yolo/ && python3 test_mc_pilot_pb_A.py -seed 1 -num_trials 10
```

Results land in `results_<config_name>/<seed>/`.

## The hardware track (`mc-pilot-pybullet/`)

```bash
cd mc-pilot-pybullet/

# search a release-state table for the Gen3, then train through it
python3 find_throw_pose.py --robot kinova_gen3_dyn --mode overhead --out throw_pose_table.npy
python3 train_mc_pilot_pb_arm.py --robot kinova_gen3_dyn --opt_pose throw_pose_table.npy --flight_targets

# real-physics evaluation on a fresh, previously-unused seed
python3 eval_adapted_height.py --log_path results_kinetic_chain_gen3_tcp/1 \
    --opt_pose throw_pose_table_tcp.npy --tool_offset_z 0.12 --num_throws 30 --seed 24681012

# dry-run a throw plan against the real-arm safety checks (touches no hardware).
# --robot kinova_gen3_dyn matters: the default kinematic profile skips the torque check.
python3 run_hardware_throw.py plan --robot kinova_gen3_dyn \
    --log_path results_kinetic_chain_gen3_tcp/1 --opt_pose throw_pose_table_tcp.npy \
    --tool_offset_z 0.12 --wrist_roll_offset_deg 90 --target 0.9 0.0
```

**Hardware safety.** Every hardware command is a dry run unless you pass `--arm`,
and `throw` additionally needs `--confirm`. The scripts default to Kinova's
factory credentials (`admin`/`admin`) and this lab's arm address
(`192.168.1.101`); override `--ip`, `--username`, `--password` for your setup.
Read `mc-pilot-pybullet/HARDWARE_SETUP.md` (the safety model) and
`HARDWARE_RUNBOOK.md` (the run-day procedure) before connecting to a real arm. A
checkpoint is only valid with the pose table, `--tool_offset_z` and `--base_height`
it was trained under — the loaders refuse mismatches rather than throw a different
trajectory than the one that was trained.

## Documentation map

| Doc | What it is |
|---|---|
| [`REPRODUCING.md`](REPRODUCING.md) | Verified commands and expected output for each re-runnable result |
| `PROGRESS_REPORT.md` | Narrative writeup, section by section from the fork onward |
| `COMPARISON_VS_ORIGINAL_PAPER.md` | Parameter-by-parameter comparison against the MC-PILOT paper — where this project is ahead and where it is honestly behind (§4) |
| `RESULTS_AND_METRICS.md` | Quick-reference results table (see its banner for superseded rows) |
| `status_update/HANDOFF.md` | Session-by-session state, newest on top; opens with a REAL / ASSIGNED / NOT-WORKING table |
| `CLAUDE.md` | Working notes for the Claude Code assistant used on this project; doubles as a dense, dated engineering log of every bug found and number measured |
| `mc-pilot-pybullet/HARDWARE_SETUP.md`, `HARDWARE_RUNBOOK.md` | Gen3 safety model and run-day procedure |

## Key findings beyond the original paper

1. **Reliability isn't free.** The inherited baseline's "5/5 hits" was
   single-seed luck (60/20/80/10/10% across 5 seeds); stratified exploration
   + a lengthscale sized to the target domain fixes it to 5/5 across seeds.
2. **A searched, whole-trajectory-feasible release state, not a fixed
   analytical one, is what makes a weaker-actuator arm (Gen3, 9 Nm wrist)
   throwable at all.** Feasibility checked at the release instant alone
   passes poses the real windup structure makes torque-infeasible.
3. **The drag crossover is the sharpest sim finding:** an analytical
   ballistic baseline beats learned dynamics at low drag (tennis ball);
   learned dynamics wins 3–7× as drag rises (whiffle ball) — learning
   matters exactly when the analytical model's assumptions fail.
4. **Zero-mean noise doesn't move the optimal policy** — only biased noise
   (velocity slip, timing jitter) produces an aware-vs-naive gap.
5. Negative results, reported rather than discarded: residual-physics
   policy/model terms do not help; GPU training is slower than CPU here
   (small tensors, PyBullet is CPU-only regardless).
6. **Audit findings that changed earlier numbers.** A pose search that loaded the
   Gen3 URDF raw hung a phantom +3 kg off the wrist (bodyless links default to 1 kg
   in PyBullet), understating feasibility; a real tool-centre-point offset
   (0.12 m configured, ~0.22–0.27 m fitted) cost the first hardware landings 45 cm.
   Both are documented with before/after numbers in `PROGRESS_REPORT.md` and
   `CLAUDE.md`.

## License

[GNU AGPL-3.0-or-later](LICENSE), matching the vendored upstream `MC-PILCO/`
(© 2023 Mitsubishi Electric Research Laboratories), from which the `mc-pilot*/`
code is derived. If you run a modified version as a network service, AGPL §13
requires offering its source to users.

## Citing

If you use this work, please cite the MC-PILOT paper (Turcato et al.,
arXiv:2502.05595), the MC-PILCO package it builds on, and credit AR525 Group 3 for
the five inherited studies. This repository:
<https://github.com/Rohitjangra7370/throwing_robotic_arm>.
