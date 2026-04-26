# Wind Force Experiment: MC-PILOT Wind Extension

Extension of MC-PILOT to study wind effects on ball throwing accuracy. Three wind scenarios (constant, gusts, turbulence) with systematic comparison of implicit GP absorption vs. explicit wind-state augmentation.

## Motivation

MC-PILOT's drag equation (`_ball_accel()`) already accepts a `wind` vector but it's **static** — set once at construction. Real semi-outdoor deployments face:
- Steady crosswinds shifting landing laterally
- Gusts that change mid-flight (step changes in `w(t)`)
- Turbulence creating continuous stochastic perturbation

The core scientific question: **Can the GP model implicitly absorb wind effects by treating them as noise in the dynamics, or does it require explicit wind-state input to generalize?**

## User Review Required

> [!IMPORTANT]
> **New directory**: This creates `mc-pilot-wind/` as a standalone variant (not modifying existing `mc-pilot/` or `mc-pilot-pybullet/`). The baseline throwing system + GP model are copied and extended in-place.

> [!IMPORTANT]
> **Wind magnitudes**: Proposed wind speeds are calibrated to a tennis ball (57g, 3.27cm radius) thrown at 1.5–3.5 m/s over ~1m range. A 1 m/s crosswind produces ~5–8cm lateral deflection. Stronger wind (2+ m/s) would dominate the throw and make the task trivially impossible. The plan uses 0.3–1.0 m/s winds.

## Open Questions

1. **Wind observability**: In "wind-aware" mode, should the policy receive the current wind measurement as input (making it a 10-D augmented state `[ball(6), target(2), wind(2)]`), or should it receive a running average? The plan uses instantaneous wind.
2. **Number of trials**: The plan uses 15 trials (vs. 10 in baseline) to give the GP more data under wind. Is this acceptable runtime-wise (~15 min per config on CPU)?

## Proposed Changes

### New Directory: `mc-pilot-wind/`

This is a complete standalone copy of `mc-pilot/` with wind-specific modifications. All shared library code (`gpr_lib/`, base `MC_PILCO`, base `Policy`, base `Cost_function`) is symlinked from `mc-pilot/` to avoid duplication.

---

### Wind Models Module

#### [NEW] [wind_models.py](file:///home/rishang/ar525_project/mc-pilot-wind/simulation_class/wind_models.py)

Three wind model classes, all implementing the same `WindModel` interface:

```python
class WindModel:
    def __call__(self, t: float) -> np.ndarray:  # returns [wx, wy, wz]
    def reset(self):                              # new episode
    def describe(self) -> str:                    # human-readable summary
```

| Class | Behavior | Parameters |
|-------|----------|------------|
| `ConstantWind` | `w(t) = w_0` (fixed 3D vector) | `velocity=[wx, wy, wz]` |
| `GustWind` | `w(t)` = piecewise-constant, changes every `T_gust` seconds to a new random vector within `[-w_max, w_max]^3` | `w_max`, `T_gust`, `seed` |
| `TurbulentWind` | `w(t) = w_mean + N(0, σ²)` per-component Gaussian noise at each call, with optional temporal correlation via exponential smoothing | `w_mean`, `sigma`, `alpha` (smoothing), `seed` |

All models produce **2D wind only** (wx, wy) with wz=0 — vertical wind is negligible for indoor/warehouse scenarios.

---

### Modified Throwing System

#### [NEW] [model_wind.py](file:///home/rishang/ar525_project/mc-pilot-wind/simulation_class/model_wind.py)

`WindThrowingSystem` — extends the baseline `ThrowingSystem` with a time-varying wind model:

- **Constructor** takes a `WindModel` instance instead of a static `wind` vector
- **`_simulate()`** calls `self.wind_model(t)` at each Euler step, passing the result to `_ball_accel()`
- **`rollout()`** calls `self.wind_model.reset()` at the start of each episode, then records the wind history `w_traj` alongside `pos_traj` and `vel_traj`
- Returns **extended state arrays** that optionally include wind at each timestep

The key change in `_simulate()`:
```python
for i in range(num_steps - 1):
    t = i * dt
    w = self.wind_model(t)           # ← time-varying wind
    accel = _ball_accel(pos[i], vel[i], self.mass, self.radius, w)
    ...
    wind_traj[i] = w                 # record for optional GP input
```

---

### GP Model Variants

#### [NEW] [Model_learning_wind.py](file:///home/rishang/ar525_project/mc-pilot-wind/model_learning/Model_learning_wind.py)

Two GP model classes for the A/B comparison:

1. **`Ballistic_Model_learning_RBF`** (inherited unchanged from mc-pilot)
   - GP input: `[x, y, z, vx, vy, vz]` (6-D) — **wind-blind**
   - Tests whether the GP can absorb wind effects as increased prediction variance

2. **`WindAware_Ballistic_Model_learning_RBF`** (new subclass)
   - GP input: `[x, y, z, vx, vy, vz, wx, wy]` (8-D) — **wind-aware**
   - Augmented state layout: `s = [x,y,z, vx,vy,vz, Px,Py, wx,wy]` (10-D)
   - `data_to_gp_input()` extracts `[ball(6), wind(2)]` = dims 0:6 + 8:10
   - `get_next_state_from_gp_output()` carries target AND wind dims forward unchanged

The wind dims (8:10) are populated from the wind model's output at each simulation step.

---

### Extended MC_PILOT

#### [NEW] [MC_PILCO.py](file:///home/rishang/ar525_project/mc-pilot-wind/policy_learning/MC_PILCO.py)

`MC_PILOT_Wind` — extends `MC_PILOT` with wind-aware data collection and particle propagation:

- **`get_data_from_system()`**: After rollout, appends wind trajectory columns to the state array (10-D state for wind-aware mode)
- **`apply_policy()`**: In wind-aware mode, samples a random wind vector per particle and includes it in the initial particle state. During GP propagation, the wind dims are carried forward (constant per throw, matching the physics)
- **`__init__()`** accepts `wind_aware: bool` flag to switch between 8-D (blind) and 10-D (aware) state layouts

---

### Policy Extension

#### [NEW] [Policy.py](file:///home/rishang/ar525_project/mc-pilot-wind/policy_learning/Policy.py)

The `Throwing_Policy` stays unchanged — it extracts target from `state[-target_dim:]` and this still works because:
- Wind-blind (8-D): `[ball(6), target(2)]` → target at dims 6:8 ✓
- Wind-aware (10-D): `[ball(6), target(2), wind(2)]` → target at dims 6:8... **WRONG**

> [!WARNING]
> The wind-aware state layout puts wind AFTER target, so `state[-target_dim:]` would grab wind instead of target. Fix: `Throwing_Policy` uses `target_indices` (explicit dim list) instead of `[-target_dim:]`.

New `WindAware_Throwing_Policy` subclass:
- Extracts `(Px, Py)` from explicit indices `[6, 7]` (not relative to end)
- Optionally also reads wind `(wx, wy)` from indices `[8, 9]` as additional RBF input — this lets the policy output **different speeds for the same target under different winds**

---

### Test Scripts

#### [NEW] [test_wind_W1.py](file:///home/rishang/ar525_project/mc-pilot-wind/test_wind_W1.py) — Constant Wind

Baseline comparison: constant crosswind at different speeds.

| Variant | Wind | GP Mode | Policy |
|---------|------|---------|--------|
| W1-calm | `[0, 0, 0]` | blind | standard |
| W1-light | `[0.3, 0, 0]` crosswind | blind | standard |
| W1-moderate | `[0.7, 0, 0]` crosswind | blind | standard |
| W1-strong | `[1.0, 0, 0]` crosswind | blind | standard |
| W1-aware | `[0.7, 0, 0]` crosswind | **wind-aware** | **wind-aware** |

Expected outcome: wind-blind GP absorbs constant wind as a bias in `Δv` predictions — should work reasonably since the wind is the same every throw.

#### [NEW] [test_wind_W2.py](file:///home/rishang/ar525_project/mc-pilot-wind/test_wind_W2.py) — Random Gusts

| Variant | Wind | GP Mode | Policy |
|---------|------|---------|--------|
| W2-blind | Gusts `w_max=0.5`, `T_gust=0.15s` | blind | standard |
| W2-aware | Same gusts | **wind-aware** | **wind-aware** |

Expected outcome: wind-blind GP fails — each throw sees different wind, so the GP must explain varying Δv from identical ball states. Wind-aware GP should recover by conditioning on `(wx, wy)`.

#### [NEW] [test_wind_W3.py](file:///home/rishang/ar525_project/mc-pilot-wind/test_wind_W3.py) — Turbulence

| Variant | Wind | GP Mode | Policy |
|---------|------|---------|--------|
| W3-blind | Turbulent `σ=0.3`, `w_mean=[0.3, 0, 0]` | blind | standard |
| W3-aware | Same turbulence | **wind-aware** | **wind-aware** |

Expected outcome: turbulence adds irreducible noise. Wind-aware GP helps with the mean component but variance remains high.

#### [NEW] [run_all_wind_experiments.py](file:///home/rishang/ar525_project/mc-pilot-wind/run_all_wind_experiments.py)

Orchestration script that runs all wind configs sequentially and collects results into a summary table.

#### [NEW] [analyze_wind_results.py](file:///home/rishang/ar525_project/mc-pilot-wind/analyze_wind_results.py)

Post-hoc analysis script:
- Loads `log.pkl` from each experiment
- Computes landing error statistics (mean, std, hit rate at 10cm threshold)
- Generates comparison plots: cost curves, landing scatter, wind effect vs. accuracy
- Produces a summary table for the report

---

### Cost Function

#### [MODIFY] [Cost_function.py](file:///home/rishang/ar525_project/mc-pilot-wind/policy_learning/Cost_function.py)

No changes needed — `Throwing_Cost` uses explicit `position_indices` and `target_indices` which work with both 8-D and 10-D states.

---

## File Structure

```
mc-pilot-wind/
├── simulation_class/
│   ├── model.py                  # symlink → mc-pilot/simulation_class/model.py
│   ├── model_wind.py             # NEW: WindThrowingSystem
│   └── wind_models.py            # NEW: ConstantWind, GustWind, TurbulentWind
├── model_learning/
│   ├── Model_learning.py         # symlink → mc-pilot/model_learning/Model_learning.py
│   └── Model_learning_wind.py    # NEW: WindAware_Ballistic_Model_learning_RBF
├── policy_learning/
│   ├── MC_PILCO.py               # NEW: MC_PILOT_Wind (extends MC_PILOT)
│   ├── Policy.py                 # symlink + NEW: WindAware_Throwing_Policy
│   └── Cost_function.py          # symlink → mc-pilot/policy_learning/Cost_function.py
├── gpr_lib/                      # symlink → mc-pilot/gpr_lib/
├── test_wind_W1.py               # Constant wind experiments
├── test_wind_W2.py               # Gust wind experiments
├── test_wind_W3.py               # Turbulence experiments
├── run_all_wind_experiments.py   # Orchestration
└── analyze_wind_results.py       # Results analysis
```

## Verification Plan

### Automated Tests

1. **Sanity check**: Run W1-calm (zero wind) and verify results match `mc-pilot/test_mc_pilot.py` baseline (same cost curve, same hit rate)
2. **Wind physics**: Run `WindThrowingSystem` with constant 1 m/s crosswind, verify landing position shifts ~5–8cm laterally compared to calm
3. **GP input dimensions**: Verify wind-aware GP accepts 8-D input without shape errors
4. **State layout**: Verify `Throwing_Policy` extracts correct target dims in both 8-D and 10-D modes

### Experiments (the actual study)

1. Run all W1/W2/W3 configs with seed=1
2. Compare cost convergence curves (blind vs. aware)
3. Compute landing error statistics
4. Generate comparison table for the report

### Expected ~2 hours total runtime

| Config | Estimated time |
|--------|---------------|
| W1 (5 variants × 15 trials) | ~75 min |
| W2 (2 variants × 15 trials) | ~30 min |
| W3 (2 variants × 15 trials) | ~30 min |
