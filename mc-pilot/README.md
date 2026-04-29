# mc-pilot — MC-PILOT Baseline

Implements the MC-PILOT throwing algorithm in a pure NumPy ballistic simulator — no ROS, no Gazebo, no PyBullet.

**Study:** Baseline reproduction. Ground-plane targets, z_release = 0.5 m, Franka Panda geometry.

Paper: Turcato et al., arXiv:2502.05595

---

## Our code vs MC-PILCO boilerplate

This folder was copied from `../MC-PILCO/` and modified. The files we wrote or changed are:

| File | What it does |
|------|-------------|
| `simulation_class/model.py` | `ThrowingSystem` — NumPy ballistic ODE with drag, particle landing freeze |
| `model_learning/Model_learning.py` | `Ballistic_Model_learning_RBF` — GP on 6-D ball state, velocity-increment output |
| `policy_learning/Policy.py` | `Throwing_Policy` (RBF, π(Px,Py)→speed), `Random_Throwing_Exploration`, `Stratified_Throwing_Exploration` |
| `policy_learning/Cost_function.py` | Landing-based cost with ℓc = 0.5 m |
| `policy_learning/MC_PILCO.py` | `MC_PILOT` — single-shot policy application, augmented 8-D state |
| `test_mc_pilot.py` | Main entry point |

Everything else (`apply_mcpilco_policy.py`, `log_plot_cartpole.py`, `test_mcpilco_cartpole*.py`, `MC_PILCO_Software_Package.pdf`, etc.) is upstream MC-PILCO boilerplate for the cartpole task — not used here.

---

## Run

```bash
python test_mc_pilot.py -seed 1 -num_trials 10
```

Results saved to `results_mc_pilot/<seed>/`. Expected: 5/5 hits by trial 5, cost converging from 0.74 to ≈ 0.001.

---

## Key parameters

| Parameter | Our value | Paper value | Why different |
|-----------|-----------|-------------|---------------|
| `lc` | 0.5 m | 0.1 m | lc=0.1 saturates cost for all early misses (0.3–1.5 m), killing gradients |
| `Ts` | 0.02 s | 0.01 s | Doubled for ~3× speedup; no accuracy regression |
| `T` | 0.7 s | 1.0 s | Ball lands by ~0.58 s; trimmed horizon saves compute |
| `Nexp` | 5 | 5 | Matches paper |
| `M` | 400 | 400 | Matches paper |
| `uM` | 3.5 m/s | 3.5 m/s | Matches paper |

---

## Four silent bugs fixed during reproduction

| # | Bug | Symptom | Fix |
|---|-----|---------|-----|
| 1 | Speed channel stripped from GP input | No gradient reaches policy | Embed speed in particle initial velocity |
| 2 | Underground particles continue propagating | Cost stuck at 0.998 | Freeze particle state when z ≤ 0 |
| 3 | Cost lengthscale ℓc = 0.1 m too tight | Gradient < 1e-3 for all early errors | Set ℓc = 0.5 m |
| 4 | Simulation horizon unit mismatch | 10,000-step tensor → RAM crash | Pass integer steps, not float seconds |
