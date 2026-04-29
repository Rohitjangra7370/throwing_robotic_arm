# MC-PILOT: Data-Efficient Robotic Throwing with Model-Based RL
**AR525 — Reinforcement Learning in Robotics | IIT Mandi, Group-3**

Members: Rishang Yadav · Bhumika Gupta · Aarya Agarwal · Yajesh Chandra

Paper reproduced: Turcato et al., "Data-Efficient Robotic Object Throwing with Model-Based Reinforcement Learning," arXiv:2502.05595

---

## What this project does

We reproduce the core MC-PILOT algorithm — which teaches a robot arm to throw a ball into a target bin using only ~10 real attempts — and extend it across five studies using a pure Python + PyBullet environment (no ROS, no Gazebo).

| Study | What was investigated | Best result |
|-------|-----------------------|-------------|
| Baseline | Reproduce MC-PILOT in a NumPy ballistic sim | 5/5 hits, cost 0.74 → 0.001 in 5 trials |
| 1. Elevated release | z = 1.0 / 1.5 / 2.0 m platforms, stratified exploration | 10/10 hits at all heights |
| 2. Multi-arm + noise | Franka Panda, KUKA iiwa7, xArm6; slip and salt-pepper noise | 100% all arms; aware beats naive by up to 100 pp |
| 3. PyBullet elevated | Elevated targets with PyBullet arm physics | 10/10 (Config B, z=1.0 m) |
| 4. Wind conditions | Constant wind, gusts, OU turbulence — blind vs aware GP | Blind GP wins; curse of dimensionality |
| 5. Vision pipeline | OpenCV HSV segmentation + YOLOv8 bin detection | End-to-end detect → throw |

Demo videos: `xarm6.mp4` (xArm6 throw), `yolo (1).mp4` (YOLO vision pipeline).

---

## Repository layout

```
MC-PILCO/               Upstream MC-PILCO codebase — reference only, do not modify
mc-pilot/               Baseline: NumPy sim, ground targets, z_release = 0.5 m
mc-pilot-elevated/      Study 1: elevated release heights (z = 1.0/1.5/2.0 m), NumPy sim
mc-pilot-pybullet/      Study 2: PyBullet arm + arm-noise study + multi-arm comparison
mc-pilot-pb-elevated/   Study 3: PyBullet arm + elevated release heights
mc-pilot-wind/          Study 4: wind models (constant / gusts / turbulence)
mc-pilot-pybullet-yolo/ Study 5: OpenCV + YOLO bin detection pipeline
paper/                  LaTeX source for the group report (main.tex + sections/)
ppt/                    Beamer presentation source and figures
2502.05595v1.pdf        Original MC-PILOT paper
xarm6.mp4               Demo: xArm6 throw in PyBullet
yolo (1).mp4            Demo: YOLO vision pipeline in PyBullet
change_history.md       Full experiment log — every run, every failure, root causes
current_config.md       Canonical Table 1 parameters for the baseline
results_mc_pilot/       Baseline results (seed 1) — see also mc-pilot/results_mc_pilot/
```

---

## Environment setup

Python 3.11 (see `.python-version`). A virtualenv is at `.venv/`.

**Activate the existing venv:**
```bash
source .venv/bin/activate
```

**Or create from scratch:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy scipy matplotlib pybullet ultralytics
```

All `python` commands below assume the venv is active and are run from inside the relevant study folder.

---

## Running the studies

### Baseline — NumPy sim, ground targets

```bash
cd mc-pilot/
python test_mc_pilot.py -seed 1 -num_trials 10
```

Results: `mc-pilot/results_mc_pilot/1/`. Expected: 5/5 hits by trial 5, final cost ≈ 0.001.

---

### Study 1 — Elevated release heights (NumPy)

```bash
cd mc-pilot-elevated/
python test_mc_pilot_b_strat.py -seed 1 -num_trials 10   # z = 1.0 m
python test_mc_pilot_c_strat.py -seed 1 -num_trials 10   # z = 1.5 m
python test_mc_pilot_d_strat.py -seed 1 -num_trials 10   # z = 2.0 m
python test_mc_pilot_e_strat5.py -seed 1 -num_trials 10  # z = 0.0 m, narrow range (ℓs=0.15 m)
```

Results: `results_mc_pilot_b_strat/`, `_c_strat/`, `_d_strat/`. Expected: 10/10 hits with stratified exploration.

---

### Study 2 — PyBullet arm + noise + multi-arm

```bash
cd mc-pilot-pybullet/

# Baseline (KUKA iiwa7, no noise):
python test_mc_pilot_pb_A.py -seed 1 -num_trials 10

# Noise-aware policy (velocity slip, α=0.20):
python test_mc_pilot_pb_A_noisy.py -seed 1 -num_trials 10 -alpha 0.20

# Multi-arm (Franka Panda, xArm6):
python train_mc_pilot_pb_A_franka_panda.py -seed 1
python train_mc_pilot_pb_A_xarm6.py -seed 1

# Full noise paper sweep (3 seeds × all noise configs):
python run_pb_noise_paper_multiseed.py

# PyBullet GUI demo (replay trained policy):
python demo_pybullet_gui.py --log_path results_mc_pilot_pb_A/1
```

Noise report: `results_mc_pilot_pb_noise_multiseed_report/`.

---

### Study 3 — PyBullet arm + elevated release

```bash
cd mc-pilot-pb-elevated/
python test_mc_pilot_pbe_B.py -seed 1 -num_trials 10   # z = 1.0 m
python test_mc_pilot_pbe_C.py -seed 1 -num_trials 10   # z = 1.5 m
python test_mc_pilot_pbe_D.py -seed 1 -num_trials 10   # z = 2.0 m
```

Results: `results_mc_pilot_pbe_B/`, `_C/`, `_D/`.

---

### Study 4 — Wind conditions

```bash
cd mc-pilot-wind/
python test_wind_W1.py              # constant wind (4 speeds + aware comparison)
python test_wind_W2.py              # random gusts
python test_wind_W3.py              # OU turbulence
# Or run all 9 configs in one go:
python run_all_wind_experiments.py --num_trials 15
python analyze_wind_results.py      # print summary table
```

Results: `results_wind_W1/`, `results_wind_W2/`, `results_wind_W3/`. Summary: `final_analysis.txt`.

---

### Study 5 — Vision pipeline (OpenCV + YOLO)

```bash
cd mc-pilot-pybullet-yolo/

# Step 1 — train the throwing policy:
python test_mc_pilot_pb_A.py -seed 1 -num_trials 10

# Step 2 — GUI demo with OpenCV vision:
python demo_pybullet_gui.py --log_path results_mc_pilot_pb_A/1 --num_throws 5

# (Optional) Train YOLO detector on synthetic data:
python generate_yolo_target_dataset.py
python train_yolo_target_detector.py
```

---

## Key findings beyond the paper

Three convergence rules not stated in the original paper but necessary for reliable learning:

1. **Stratified exploration** — random seeds can produce zero GP coverage over part of the speed range; partitioning `[0, uM]` into `Nexp` equal bands guarantees full coverage in every seed.

2. **RBF lengthscale rule** — `ℓs ≈ 0.15 × target_range`; the paper's default ℓs = 1.0 m makes the policy output a nearly constant speed for all targets when the target domain is narrow (< 0.4 m).

3. **Zero-mean noise invariance** — symmetric Gaussian noise does not shift the optimal policy; only biased noise (velocity slip, timing jitter, salt-and-pepper) is learnable by RL.

**Unexpected result (Study 4):** The blind GP (2-D policy) consistently outperforms the wind-aware GP (4-D policy) across all wind conditions with 15 training trials. Explicit wind state conditioning requires ~40+ trials to overcome the curse of dimensionality.

---

## Reference files

| File | Contents |
|------|----------|
| `change_history.md` | Every experiment — what was tried, what failed, root causes, lessons |
| `current_config.md` | Canonical hyperparameters for the baseline (Table 1 equivalent) |
| `paper/main.tex` | Full group report source |
| `2502.05595v1.pdf` | Original MC-PILOT paper |
