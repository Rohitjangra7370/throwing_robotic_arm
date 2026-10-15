# Reproducing the results

Every command below was run on 2026-10-16 (Python 3.10.12, Linux x86_64, CPU only)
unless the section says otherwise, and the output shown is what it printed. §§2–4 were
run twice: on the original development machine, and in a clean virtual environment
built from `requirements.txt` plus the installer in §1 (fresh torch 2.9.0, pinned
opencv 4.11.0.86). The output was identical. Results that were **not** re-run for this
release are listed in §5 with where they come from, so you can tell the two apart.

Conventions:

- Run each command from inside the directory shown (`cd mc-pilot-pybullet`). Every
  variant resolves imports and `results_*/` paths relative to its own root.
- Use `python3 -m pytest` / `python3 -m pip`, never bare `pytest` / `pip`.
- No GPU is needed or helpful (small tensors; PyBullet is CPU-only regardless).

## 1. Verify the environment

```bash
python3.10 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 mc-pilot-pybullet/scripts/install_robot_assets.py    # Gen3 model; required
cd mc-pilot-pybullet
python3 -m pytest tests/ -q
```

The installer step exists because the Gen3 URDF and meshes are loaded from inside the
installed `pybullet_data` package, which the stock `pybullet` wheel does not ship. On
the original machine they had been copied there by hand, so a from-scratch install
failed 78 tests with `Cannot load URDF file` until this was found by building a clean
venv. The model is now vendored in `mc-pilot-pybullet/robot_assets/kinova_gen3/` and
`install_robot_assets.py --check` verifies an install byte-for-byte.

What a clean environment should print (measured 2026-10-16 in a fresh venv built from
these requirement files, Python 3.10.12):

| Environment | `pytest tests/ -q` |
|---|---|
| `requirements.txt` + installer | **303 passed, 15 skipped** |
| + `requirements-hardware.txt` (RealSense SDK) | 355 passed, 13 skipped |
| + Kinova's `kortex_api` wheel and the UR7e model | **368 passed** (the original development machine) |

The skipped tests are never silent: `pytest -rs` names each reason. They are 52 tests
in `test_hardware_session.py` / `test_session_camera.py` (need `pyrealsense2`), 12 in
`test_ur7e_profile.py` (need `bash scripts/install_ur7e_urdf.sh`, which requires ROS 2
Humble's `xacro`), and 1 in `test_hardware_planner.py` (needs Kinova's `kortex_api`
wheel). No arm or camera needs to be attached for any of them.

The 368 figure is from the original development machine, which differs from the
fresh-venv rows by more than the two extra packages: it runs `opencv-python` 4.12.0.88
where the requirements pin 4.11.0.86, and it carries whatever else had accumulated in
its system Python.

## 2. Simulation headline: the deployed Gen3 checkpoint

`results_kinetic_chain_gen3_tcp/1` is the checkpoint the real-arm session used. The
claim is its accuracy under PyBullet rigid-body physics on a seed that was never used
in training (24681012), 30 throws:

```bash
cd mc-pilot-pybullet
python3 eval_adapted_height.py --log_path results_kinetic_chain_gen3_tcp/1 \
    --opt_pose throw_pose_table_tcp.npy --tool_offset_z 0.12 \
    --num_throws 30 --seed 24681012
```

```
u_cap = 1.600 m/s (profile safe_u_cap)
adapted policy: h=0.00m  band 0.670-0.740m  T=0.600s  new_trials_used=?
n=30  mean 1.90cm  median 2.04cm  max 4.20cm  hit<10cm 100%  speed 1.32-1.49 m/s
```

(`new_trials_used=?` is a cosmetic gap in the print for a checkpoint that was never
height-adapted; `h=0.00m` is the unadapted, floor-level case.)

Things that matter when you vary this:

- The checkpoint, the pose table and `--tool_offset_z` are one unit. The table is
  stamped with the tool offset and floor height it was searched under, and the
  loaders refuse a mismatch — evaluating a checkpoint against a differently searched
  table silently measures a different throw.
- Use a **fresh seed** for any new evaluation. Training cost is computed through the
  learned model, not real physics, and can sit near zero while real accuracy is off
  by 17–28%, so it is never a substitute for this step.
- Pre-TCP checkpoints (`results_kinetic_chain_gen3/{1,2,3}`) are historical and must
  not be used for a real throw: they assume the ball leaves at the bare wrist flange.

## 3. Hardware headline: aimed throws on the real Gen3

The raw per-throw table from the 2026-09-11 session is committed. Recompute its
summary (standard library only):

```bash
cd mc-pilot-pybullet
python3 scripts/summarize_bin_game.py results_bin_game/session_20260911_024918/results.csv
```

```
throws attempted : 22
throws measured  : 14
error vs aim     : mean 1.91 cm | median 1.83 cm | max 3.68 cm | sd 1.06 cm
within 3 / 5 cm  : 12/14 | 14/14
refused (5): the ball is still RISING (# m/s) in the last frame it was detected in (t = # s)
refused (2): no ballistic arc found: the largest set of observations agreeing on one g = # parabola
refused (1): no left/right ball candidates paired in any frame
```

The session's own README rounds these to 1.9 / 1.8 / 3.7 / 1.0 cm; the unrounded sd
is 1.06.

**What this does and does not reproduce.** It reproduces the statistics from the
committed per-throw table. It does **not** re-derive each landing from camera data:
the 22 raw dual-IR recordings (~106 MB each) are not in the repo.
`recording_sha256_prefix.json` in the same directory holds a SHA-256 prefix of each,
so a re-derivation from the original files can be checked against the same data, and
`analysis_pipeline.patch` is the exact diff of the landing-measurement code against
commit `0f11e2d` that produced the numbers. The pipeline's behaviour on real
recordings is pinned by the fixtures in `mc-pilot-pybullet/tests/fixtures/`.

Read the session README before quoting the number: 14-of-22, a pipeline revised after
the recordings were inspected, and a fitted release-model calibration on top of the
trained policy are all part of what the 1.91 cm means.

## 4. Dry-run a throw plan against the real-arm safety checks

This touches no hardware (it is a dry run unless `--arm` is passed):

```bash
cd mc-pilot-pybullet
python3 run_hardware_throw.py plan --robot kinova_gen3_dyn \
    --log_path results_kinetic_chain_gen3_tcp/1 --opt_pose throw_pose_table_tcp.npy \
    --tool_offset_z 0.12 --wrist_roll_offset_deg 90 --target 0.9 0.0
```

```
=== PLAN (dry-run) robot=kinova_gen3_dyn target=[0.9, 0.0] ===
policy release speed: 1.592 m/s   v_cmd EE: [ 1.537 -0.042  0.412]
release pos in safe box: True
--- trajectory precheck ---
trajectory T=8.511s, speed_scale=0.15
release quantisation: 25.0 ms at 40 Hz  ->  up to 4.0 cm of undershoot at 1.592 m/s
peak |tau| (Nm): 0.7/39.0, 14.3/39.0, 0.7/39.0, 7.6/39.0, 0.1/9.0, 1.1/9.0, 0.1/9.0
PRECHECK: PASS
```

(Excerpt; the full output also prints joint positions and velocities.)

`--robot kinova_gen3_dyn` is required for the torque check. Under the default
kinematic `kinova_gen3` profile the same command prints
`peak |tau|: NOT CHECKED` and still ends `PRECHECK: PASS`.

Everything beyond `plan` needs the physical arm; see
`mc-pilot-pybullet/HARDWARE_SETUP.md` and `HARDWARE_RUNBOOK.md`.

## 5. Not re-run for this release

These were **not executed** for this release. The commands come from `CLAUDE.md`,
the existing study READMEs and each script's `--help`; they are starting points, not
verified one-line reproductions. Each committed checkpoint's `config_log.pkl` and each
pose table's stamp (`tool_offset`, `floor_z`) record what it was actually produced
with — check those before assuming a flag.

Training is stochastic across seeds: the committed checkpoints are the artifacts the
reported numbers belong to, and retraining gives numbers in the same range, not the
same digits (seed 2 of the deployed configuration evaluated to 2.10 cm mean / 4.58 cm
max against seed 1's 1.90 / 4.20).

| What | Entry point (run from `mc-pilot-pybullet/` unless noted) | Source |
|---|---|---|
| Search a Gen3 release-state table | `python3 find_throw_pose.py --robot kinova_gen3_dyn --mode overhead --floor_z -0.433 --tool_offset_z 0.12 --out throw_pose_table_tcp.npy` | `CLAUDE.md`, TCP-offset entry |
| Retrain the deployed checkpoint | `python3 train_mc_pilot_pb_arm.py --robot kinova_gen3_dyn --seed 1 --num_trials 10 --flight_targets --opt_pose throw_pose_table_tcp.npy --tool_offset_z 0.12 --base_height 0.433` — the last three flags must go together | `CLAUDE.md`, "Which checkpoint" |
| Zero-new-trial height adaptation | `python3 adapt_policy_height.py --log_path results_kinetic_chain_gen3_tcp/1 --height 0.20 --out results_kinetic_chain_gen3_tcp_h20`, then `eval_adapted_height.py` as in §2 | `CLAUDE.md` |
| Noise-aware vs naive policy | `python3 run_pb_noise_paper_multiseed.py` | study README |
| NumPy baseline (`mc-pilot/`) | `python3 test_mc_pilot.py -seed 1 -num_trials 10` trains **one** seed; the reported 1.86 cm is an aggregate of 5 seeds × 50 targets (`RESULTS_AND_METRICS.md` #1) | study README |
| Wind study (inherited, underpowered; `mc-pilot-wind/`) | `python3 run_all_wind_experiments.py --num_trials 15` | study README |
| Height generalization (2.0 cm, `RESULTS_AND_METRICS.md` #3), 4-manipulator generalization (1.4–2.7 cm), drag crossover (3–7×) | Trainer is `train_mc_pilot_pb_arm.py` (`--robot`, `--flight_targets`, `--ball_mass 0.004 --ball_radius 0.06` for the whiffle regime); the exact per-result invocations and evaluation scripts are in `PROGRESS_REPORT.md` §§3–6. No single verified command is given here | `PROGRESS_REPORT.md` |

Retrained runs write to `results_<name>/<seed>/` (see `--results_root`); do not point
them at a committed checkpoint directory unless you intend to overwrite it.

## 6. What cannot be reproduced from this repository

- **The real-arm session itself.** It needs the Kinova Gen3, a RealSense D435i, a
  camera extrinsic calibration for your mounting, and the run-day procedure in
  `HARDWARE_RUNBOOK.md`. The calibration archives under `mc-pilot-pybullet/calib/`
  are specific to the original rig.
- **The raw camera recordings** behind §3 (see above).
- **The ICRA paper.** `paper_icra2027/` is a work in progress that is revised
  separately; the tracked copy is not final and is not covered by anything in this
  document.

## 7. If a number does not match

- `import torch` / `import pybullet` failing despite a successful install: check
  `which python3`. Conda, a Blender-bundled Python or ROS plugins on `PATH` have each
  caused this on the original machine.
- A different `n=` or a different mean on the §2 command: confirm the seed, the
  checkpoint directory and `--tool_offset_z 0.12`. The §2 command takes no `--robot`
  or `--base_height`, so the evaluator is getting the arm and base height from
  somewhere other than its command line (presumably the checkpoint's own config).
- This was verified on one machine (and the unit tests additionally in one fresh
  venv, §1). Last-digit differences across PyBullet or torch builds would not be
  surprising; a materially different mean is worth investigating rather than
  rounding away.
