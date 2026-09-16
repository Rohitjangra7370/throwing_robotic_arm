# MC-PILOT Throwing Arm

Model-based RL that teaches a robot arm to throw an object into a target bin
from a handful of real trials — a reproduction and hardware extension of
**MC-PILOT** (Turcato et al., "Data-Efficient Robotic Object Throwing with
Model-Based Reinforcement Learning," arXiv:2502.05595).

Course project (AR525, IIT Mandi — Reinforcement Learning in Robotics; FDP
Lab) that grew into an active research track, aimed at ReScience C
(simulation reproduction) with an ICRA 2027 stretch goal built on real
**Kinova Gen3 7-DOF** hardware results.

## Headline results

| | |
|---|---|
| Simulation, 4 manipulators (KUKA iiwa7, Franka Panda, xArm6, Kinova Gen3) | **1.4–2.7 cm** mean landing error, no per-arm policy tuning beyond torque/velocity limits |
| Simulation, target-height generalization (KUKA iiwa) | single policy, **2.1 cm** mean error at unseen heights 0–45 cm |
| Simulation, aerodynamic drag (tennis → whiffle ball) | learned dynamics beat a no-drag analytical baseline **3–7×** as drag rises to ~19% of object weight |
| Simulation, low-torque Gen3 release-state search | overhead throw reaches **1.26 m**, 1.7× the arm's own floor-level placement radius |
| **Real Kinova Gen3, closed-loop aimed throws** | **1.9 cm mean landing error, n = 14** measured throws, camera-in-the-loop bin aiming |

The real-hardware result is the thing to verify first if you're deciding
whether this repo is worth reading further — it's not a simulation claim.
See `paper_icra2027/overleaf/main.tex` §Hardware Validation for the full
writeup and `results_bin_game/session_20260911_024918.json` for the raw
per-throw data (regenerate the summary with
`mc-pilot-pybullet/compile_bin_game.py`).

## Provenance

This repo is a fork of `github.com/dnfy502/ar525_project`. Two phases:

- **Pre-fork (2026-03-25 → 2026-04-29), AR525 Group 3** (Aarya Agarwal,
  Bhumika Gupta, Rishang Yadav, Yajesh Chandra): five studies — baseline,
  elevated release, PyBullet arm physics + multi-arm + noise, wind
  robustness, vision (OpenCV + YOLOv8). **Done, frozen, cited as prior work**
  — not this project's results.
- **Post-fork (2026-07-02 → present)**: reliability fixes to the inherited
  baseline, a constraint-aware release-state search (posture + direction +
  speed, validated over the whole windup/throw/follow-through trajectory,
  not just the release instant), cross-manipulator and target-height
  generalization, an aerodynamic-drag study, and a full real-hardware bring-up
  on a Kinova Gen3 ending in the closed-loop result above.

## Repository layout

Each `mc-pilot*/` directory is a **self-contained fork** — own `gpr_lib/`,
`simulation_class/`, `policy_learning/`, `model_learning/`, `envs/`. No
shared library; a fix to one must be ported by hand to the others if it
should apply everywhere.

| Directory | Study | Sim backend | Key addition |
|---|---|---|---|
| `MC-PILCO/` | — | — | Upstream reference implementation (MERL, AGPL-3.0), unmodified |
| `mc-pilot/` | Baseline | NumPy ballistic | Ground targets |
| `mc-pilot-elevated/` | 1 | NumPy ballistic | Elevated release, stratified exploration |
| `mc-pilot-pybullet/` | 2 + **active work** | PyBullet | Real arm physics, torque control, multi-arm, Gen3 hardware track — see its own `README.md`/`HARDWARE_SETUP.md`/`HARDWARE_RUNBOOK.md` |
| `mc-pilot-pb-elevated/` | 3 | PyBullet | Elevated targets + arm physics |
| `mc-pilot-wind/` | 4 | NumPy + wind models | Constant wind / gusts / OU turbulence |
| `mc-pilot-pybullet-yolo/` | 5 | PyBullet + OpenCV/YOLOv8 | Vision-based bin detection |
| `status_update/` | — | — | Session handoffs (`HANDOFF.md`, newest entry on top), older progress docs |
| `paper_icra2027/` | — | — | ICRA draft source and figure-generation scripts. Under active revision; not the focus of this checkout |
| `old/` | — | — | Superseded, historical only |

Run everything from *inside* the relevant `mc-pilot*/` directory — each
resolves imports and data paths (`sys.path.append("..")`, `results_*/`,
`data/*.json`) relative to its own root.

## Read these for the full story

| Doc | What it is |
|---|---|
| `CLAUDE.md` | The dense, dated engineering log — every bug found and fixed, every measured number, session by session. The most complete record; start here for depth on any specific claim. |
| `PROGRESS_REPORT.md` | Narrative writeup: what was done, how, what was found, section by section from the fork onward. |
| `COMPARISON_VS_ORIGINAL_PAPER.md` | Parameter-by-parameter comparison against the actual MC-PILOT paper — where this project is ahead, where it's honestly still behind. |
| `status_update/HANDOFF.md` | Session-by-session state for picking the work back up; newest entry on top. |
| `mc-pilot-pybullet/HARDWARE_SETUP.md`, `HARDWARE_RUNBOOK.md` | The Gen3 hardware safety model and run-day procedure. |

## Environment

Python 3.10, dependencies (`torch`, `pybullet`, `numpy`, `scipy`,
`matplotlib`, `pytest`, plus `ultralytics`/`opencv-python` for the vision
study) installed system-wide — there is no repo-level virtualenv on the
machine this was developed on. To set one up fresh:

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install torch numpy scipy matplotlib pybullet pytest ultralytics opencv-python
```

**Always use `python3 -m pip` / `python3 -m pytest`, never bare `pip`/`pytest`**
if there is any other Python on `PATH` (Conda, a Blender-bundled snap, etc.)
— bare invocations have silently resolved to the wrong interpreter on the
original dev machine more than once. Verify with
`python3 -c "import torch, pybullet"` before assuming a failure is real.

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

## The active hardware track (`mc-pilot-pybullet/`)

```bash
cd mc-pilot-pybullet/
python3 -m pytest tests/ -q                 # 294-test regression suite, ~29s, no GPU

# search a release-state table for the Gen3, then train through it
python3 find_throw_pose.py --robot kinova_gen3_dyn --mode overhead --out throw_pose_table.npy
python3 train_mc_pilot_pb_arm.py --robot kinova_gen3_dyn --opt_pose throw_pose_table.npy --flight_targets

# real-physics evaluation on a fresh, previously-unused seed
python3 eval_baseline.py --log_path <ckpt> --robot kinova_gen3_dyn --num_throws 30 --seed 246810

# operator run-day app for the physical arm (start-of-day gates -> N real
# throws with camera-based landing measurement -> model update)
python3 hardware_session.py
```

Full flag reference, checkpoint provenance, and every hardware gotcha found
along the way are in the nested `mc-pilot-pybullet/CLAUDE.md` section of the
project root's `CLAUDE.md` — read it before touching the release-solver,
torque-control, or hardware-execution code; nearly every historical bug in
this project lived in one of those three.

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
5. Two negative results, reported rather than discarded: residual-physics
   policy/model terms do not help; GPU training is slower than CPU here
   (small tensors, PyBullet is CPU-only regardless).

See `COMPARISON_VS_ORIGINAL_PAPER.md` for the full delta against the paper
this project reproduces and extends.
