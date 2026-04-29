# mc-pilot-pb-elevated — Study 3: PyBullet Arm + Elevated Release

Combines PyBullet arm physics with elevated release heights. Validates that the stratified exploration and lengthscale rules from Study 1 carry over when the arm is physically modelled.

**Design:** KUKA iiwa7 arm mounted on a pedestal (`base_z = z_release − 0.5 m`) so the end-effector reaches the target release height. Ball velocity is set explicitly at release via `resetBaseVelocity` — the arm motion is cosmetic and provides the visualisation.

---

## Entry points

```bash
python test_mc_pilot_pbe_B.py -seed 1 -num_trials 10   # z = 1.0 m
python test_mc_pilot_pbe_C.py -seed 1 -num_trials 10   # z = 1.5 m
python test_mc_pilot_pbe_D.py -seed 1 -num_trials 10   # z = 2.0 m

# PyBullet GUI demo (replay trained policy):
python demo_pybullet_gui.py --log_path results_mc_pilot_pbe_B/1 --num_throws 5
```

Results saved to `results_mc_pilot_pbe_B/`, `_C/`, `_D/`.

---

## Results

Config PBE-B (z = 1.0 m, stratified exploration): **10/10 hits**. Results in `results_mc_pilot_pbe_B/`.

---

## New components (vs mc-pilot/)

| File | What it does |
|------|-------------|
| `simulation_class/model_pybullet.py` | `PyBulletThrowingSystem` — URDF arm on pedestal, explicit velocity release |
| `robot_arm/arm_controller.py` | IK solver and trajectory controller |
| `demo_pybullet_gui.py` | GUI replay script |

---

## File guide

- `test_mc_pilot_pbe_B.py`, `_C.py`, `_D.py` — **use these** (PyBullet elevated configs)
- `test_mc_pilot_pb_A.py` — PyBullet ground baseline (parity check with mc-pilot-pybullet/)
- `test_mc_pilot.py` — NumPy baseline (carried over for reference)
- `apply_mcpilco_policy.py`, `log_plot_cartpole.py`, `test_mcpilco_cartpole*.py` — upstream MC-PILCO cartpole boilerplate (ignore)
