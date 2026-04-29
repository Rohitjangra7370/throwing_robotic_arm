# mc-pilot-pybullet — Study 2: PyBullet Arm + Noise + Multi-Arm

Replaces the NumPy ballistic simulator with a PyBullet arm (KUKA iiwa7 by default). Introduces arm-side noise and tests three robot arms.

**Two research questions:**
1. Can a noise-aware RL policy outperform a naive policy when the arm delivers biased or stochastic velocity errors?
2. Does the trained policy transfer across different robot URDFs (Franka Panda, KUKA iiwa7, xArm6)?

---

## New components (not in mc-pilot/)

| File | What it does |
|------|-------------|
| `simulation_class/model_pybullet.py` | `PyBulletThrowingSystem` — URDF arm + ball physics, bin geometry |
| `robot_arm/arm_controller.py` | IK solver, cubic spline trajectory, EE velocity extraction |
| `robot_arm/noise_models.py` | `VelocitySlipNoise` (biased undershoot), `SaltAndPepperNoise` (random spikes) |
| `robot_arm/robot_profiles.py` | URDF paths and joint configs for Franka Panda, KUKA iiwa7, xArm6 |

---

## Entry points

```bash
# Baseline — KUKA iiwa7, no noise
python test_mc_pilot_pb_A.py -seed 1 -num_trials 10

# Noise-aware policy (velocity slip, α=0.20 — sharpest separation)
python test_mc_pilot_pb_A_noisy.py -seed 1 -num_trials 10 -alpha 0.20

# Multi-arm: Franka Panda
python train_mc_pilot_pb_A_franka_panda.py -seed 1

# Multi-arm: xArm6
python train_mc_pilot_pb_A_xarm6.py -seed 1

# Full noise paper sweep (3 seeds × all slip + salt-pepper configs):
python run_pb_noise_paper_multiseed.py

# PyBullet GUI demo (replay trained policy with arm animation):
python demo_pybullet_gui.py --log_path results_mc_pilot_pb_A/1 --num_throws 5

# Vision-guided throw (OpenCV depth camera — see also mc-pilot-pybullet-yolo/):
python test_mc_pilot_pb_A_vision.py -seed 1 -num_trials 5
```

---

## Results

**Multi-arm (commanded velocity):** 100% hit rate for all three arms — the RL algorithm is arm-agnostic.

**Noise study (3 seeds, averaged):**

| Noise type | Level | Aware hit rate | Naive hit rate |
|------------|-------|---------------|---------------|
| Slip | α = 0.10 | 100% | 50% |
| Slip | α = 0.20 | **100%** | **0%** |
| Salt-pepper | p = 0.05 | 100% | 75% |
| Salt-pepper | p = 0.10 | 75% | 25% |

Full multi-seed report: `results_mc_pilot_pb_noise_multiseed_report/`

**Key theorem:** Zero-mean symmetric Gaussian noise cannot be compensated by RL — the optimal policy is unchanged. Only biased noise (slip, salt-pepper, timing jitter) is learnable.

---

## File guide

- `test_mc_pilot_pb_A.py` — **main baseline entry point**
- `test_mc_pilot_pb_A_noisy.py` — **main noise study entry point**
- `run_pb_noise_paper_multiseed.py` — **multi-seed noise sweep**
- `train_mc_pilot_pb_A_franka_panda.py`, `train_mc_pilot_pb_A_xarm6.py` — **multi-arm runs**
- `demo_pybullet_gui.py` — **visual demo**
- `compare_robot_arms.py`, `inspect_robot_arms.py` — analysis utilities
- `tmp_split_run_diagnosis.py`, `tmp_terminal_policy_check.py` — debugging scripts (ignore)
- `test_mc_pilot_pb_B.py`, `test_mc_pilot_pb_C.py`, `test_mc_pilot_pb_noise_study.py` — earlier study variants (ignore)
- `apply_mcpilco_policy.py`, `log_plot_cartpole.py`, `test_mcpilco_cartpole*.py` — upstream MC-PILCO cartpole boilerplate (ignore)
