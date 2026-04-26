# Graph Report - AR525 MC-PILOT Repo Map (2026-04-25)

## Summary
- 7 nodes and 10 directed relationships
- Scope: verified repository structure plus report-described sub-project relationships
- This report separates:
  - `present in repository` items
  - `report-mentioned but not found as standalone folders/files in this branch`

## Sub-Projects (Repo-Verified)
- Present:
  - `mc-pilot/`
  - `mc-pilot-elevated/`
  - `MC-PILCO/` (upstream/reference tree)
- Report-mentioned, not found here as standalone folders:
  - `mc-pilot-pybullet/`
  - `mc-pilot-pb-elevated/`

## File Mapping by Sub-Project

### `mc-pilot/` (baseline)
- Main training script: `mc-pilot/test_mc_pilot.py`
- Core policy: `mc-pilot/policy_learning/Policy.py`
- GP model: `mc-pilot/model_learning/Model_learning.py`
- MC rollout / trainer: `mc-pilot/policy_learning/MC_PILCO.py`
- Ballistic physics simulator: `mc-pilot/simulation_class/model.py`
- PyBullet utilities currently present in this branch:
  - `mc-pilot/realtime_throw_pybullet.py`
  - `mc-pilot/visualize_pybullet.py`

### `mc-pilot-elevated/` (height study)
- Main scripts:
  - `mc-pilot-elevated/test_mc_pilot_b.py`
  - `mc-pilot-elevated/test_mc_pilot_c.py`
  - `mc-pilot-elevated/test_mc_pilot_d.py`
  - `mc-pilot-elevated/test_mc_pilot_e.py`
  - stratified variants: `test_mc_pilot_*_strat*.py`
- Core modules mirror baseline:
  - `mc-pilot-elevated/policy_learning/Policy.py`
  - `mc-pilot-elevated/model_learning/Model_learning.py`
  - `mc-pilot-elevated/policy_learning/MC_PILCO.py`
  - `mc-pilot-elevated/simulation_class/model.py`

### `mc-pilot-pybullet/` (report-claimed)
- Not found as standalone folder in this checkout.
- Closest present files:
  - `mc-pilot/realtime_throw_pybullet.py`
  - `mc-pilot/visualize_pybullet.py`
- Report-mentioned files not found here:
  - `model_pybullet.py`
  - `test_mc_pilot_pb_A.py`, `test_mc_pilot_pb_B.py`, `test_mc_pilot_pb_C.py`
  - dedicated `ArmController` class file

### `mc-pilot-pb-elevated/` (report-claimed)
- Not found as standalone folder in this checkout.
- Report-mentioned files not found here:
  - `demo_pybullet_gui.py`
  - `standalone_throw.py`

## Four Failure Modes - Fix Locations

### 1) Gradient path fix (policy -> rollout differentiability)
- `mc-pilot/policy_learning/MC_PILCO.py`
  - custom `MC_PILOT.apply_policy` embedding speed into initial particle velocity and keeping gradient chain
- mirrored in:
  - `mc-pilot-elevated/policy_learning/MC_PILCO.py`

### 2) Landing mask / freeze after touchdown
- `mc-pilot/policy_learning/MC_PILCO.py`
  - landed boolean mask and freeze logic (`z <= 0`)
- mirrored in:
  - `mc-pilot-elevated/policy_learning/MC_PILCO.py`

### 3) Cost lengthscale fix (`lc`)
- baseline:
  - `mc-pilot/test_mc_pilot.py` (`lc = 0.5`)
- elevated configs:
  - `mc-pilot-elevated/test_mc_pilot.py`
  - `mc-pilot-elevated/test_mc_pilot_b.py`
  - `mc-pilot-elevated/test_mc_pilot_c.py`
  - `mc-pilot-elevated/test_mc_pilot_d.py`
  - `mc-pilot-elevated/test_mc_pilot_e.py`
  - targeted experiments: `test_mc_pilot_e_strat4.py`, `test_mc_pilot_e_strat5.py` (`lc = 0.25`)

### 4) Horizon / indexing fixes (`T_control`, `opt_steps_list`)
- baseline:
  - `mc-pilot/test_mc_pilot.py` (`T = 0.7`, `T_control: T`, `n_list = Nexp + num_trials`, `opt_steps_list = [Nopt] * n_list`)
- elevated:
  - `mc-pilot-elevated/test_mc_pilot*.py` (same pattern, per-config `T`)
- trainer logic using those values:
  - `mc-pilot/policy_learning/MC_PILCO.py`
  - `mc-pilot-elevated/policy_learning/MC_PILCO.py`

## Shared Core Modules
- `Policy.py`:
  - `mc-pilot/policy_learning/Policy.py`
  - `mc-pilot-elevated/policy_learning/Policy.py`
- `Model_learning.py`:
  - `mc-pilot/model_learning/Model_learning.py`
  - `mc-pilot-elevated/model_learning/Model_learning.py`
- `MC_PILCO.py`:
  - `mc-pilot/policy_learning/MC_PILCO.py`
  - `mc-pilot-elevated/policy_learning/MC_PILCO.py`

## Stratified Exploration
- Implementation class:
  - `mc-pilot-elevated/policy_learning/Policy.py` (`Stratified_Throwing_Exploration`)
- Usage in scripts:
  - `mc-pilot-elevated/test_mc_pilot_b_strat.py`
  - `mc-pilot-elevated/test_mc_pilot_c_strat.py`
  - `mc-pilot-elevated/test_mc_pilot_d_strat.py`
  - `mc-pilot-elevated/test_mc_pilot_e_strat.py`
  - `mc-pilot-elevated/test_mc_pilot_e_strat2.py`
  - `mc-pilot-elevated/test_mc_pilot_e_strat3.py`
  - `mc-pilot-elevated/test_mc_pilot_e_strat4.py`
  - `mc-pilot-elevated/test_mc_pilot_e_strat5.py`

## Wind-Aware State Augmentation (8-D -> 11-D)
- Present in this checkout:
  - air-relative drag and wind parameter in simulators:
    - `mc-pilot/simulation_class/model.py`
    - `mc-pilot-elevated/simulation_class/model.py`
- Not found in this checkout (as report describes):
  - `infer_throwing_state_layout` helper
  - explicit 11-D state propagation in `Policy.py`, `Model_learning.py`, `MC_PILCO.py`
  - `model_pybullet.py`, `demo_pybullet_gui.py` wind hooks

## God Nodes
1. `mc-pilot/` - anchors baseline training and contains the core modules plus currently present PyBullet utilities.
2. `mc-pilot-elevated/` - carries most experiment scripts and stratified exploration usage.
3. `failure-fixes` - links to both baseline and elevated trainer/config code.

## Surprising Connections
- The report frames separate `mc-pilot-pybullet/` and `mc-pilot-pb-elevated/` projects, but in this branch PyBullet code is currently under `mc-pilot/` (`realtime_throw_pybullet.py`, `visualize_pybullet.py`).
- Wind physics updates are already in `simulation_class/model.py` (air-relative drag), while the broader 11-D pipeline wiring described in the report is not fully visible in this checkout.
- The most critical algorithmic fixes (gradient path and landing mask) are implemented in the custom `MC_PILOT.apply_policy` override, mirrored across both baseline and elevated trees.