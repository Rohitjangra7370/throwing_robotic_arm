# Velocity-from-Dynamics — Design

_Date: 2026-07-19 · Study: Kinova Gen3 sim, `mc-pilot-pybullet/` · Status: approved_

## Goal

Release velocity must come from tracked arm motion under torque control, not
`resetBaseVelocity` injection. Then measure the command-vs-actual release-velocity
error distribution and fit a noise model from it, ready for later noise-robust
Gen3 training (that training is the *next* milestone, out of scope here).

Approach chosen: **computed-torque control + torque-feasibility time scaling**
(over PD+gravity-comp-only, which risks infeasible tracking at the 9 Nm wrists,
and over full velocity-space trajectory optimization, which is deferred until
time scaling proves insufficient).

Hardware context: real deployment will use Kortex **low-level 1 kHz joint
control**; sim runs at 240 Hz — note the rate gap in the report, it is one of
the sim-vs-real deltas the calibration throws will absorb.

## Scope

- Done when: torque-tracked Gen3 throws work end-to-end; tracking-error
  distribution measured over the release-speed range; fitted noise model added
  to `robot_arm/noise_models.py`.
- Not in scope: retraining MC-PILOT under dynamic release (only a 10-throw
  sim2sim gap check), hardware driver work, other robot profiles.

## Components (all in `mc-pilot-pybullet/`)

### 1. Profile `kinova_gen3_dyn` (`robot_arm/robot_profiles.py`)

Copy of `kinova_gen3` with `control_mode="torque"` and new `RobotProfile` field
`tau_max=(39, 39, 39, 39, 9, 9, 9)` Nm (Gen3 actuator limits; big joints 1–4,
small wrists 5–7). New fields `kp`, `kd` for per-joint feedback gains, initial
value (100, 20) for every joint, tuned via the slow-tracking test. `tau_max`,
`kp`, `kd` default to `None` on other profiles — nothing else changes for them.
Existing `kinova_gen3` (kinematic) profile stays untouched so prior results and
the running multi-seed batch remain reproducible.

### 2. `ArmController` torque branch (`robot_arm/arm_controller.py`)

- `_eval_cubic` / `get_setpoint` additionally return `qdd_des` (second
  derivative of the piecewise cubic).
- One-time setup when `control_mode == "torque"`: disable PyBullet default
  velocity motors (`setJointMotorControlArray(VELOCITY_CONTROL, forces=0)`).
- `step(q_target, qd_target, qdd_target)` torque mode, each sim step:

  ```
  e  = q_des  − q_meas
  ė  = qd_des − qd_meas
  τ  = calculateInverseDynamics(q_meas, qd_meas, qdd_des + Kp·e + Kd·ė)
  τ  = clip(τ, ±tau_max)   # then TORQUE_CONTROL
  ```

  Inverse dynamics is computed for the **arm alone** — the 57.7 g ball and the
  grip-constraint forces are deliberately unmodeled. That mismatch (plus the
  torque clamp) is the physical source of the speed-dependent tracking error
  this study measures. Gravity compensation is inherent in the inverse-dynamics
  term, matching Kinova firmware behavior under low-level torque mode.
- NaN guard on τ → raise immediately with joint state dump.

### 3. Torque-feasibility time scaling (`plan_throw`)

After planning the throw-phase cubic: sample it (~50 points), evaluate demanded
τ via inverse dynamics; if any joint exceeds `tau_max`, stretch the throw-phase
duration ×1.2 and re-plan (endpoint velocity `qd_release` is preserved — a
longer phase lowers acceleration peaks). Max 6 iterations; on failure raise
with a per-joint torque report. Log the final `t_r` shift in the plan output.
This is the pragmatic core of "velocity-space trajectory optimization"; a full
optimal-profile planner is a later upgrade only if scaling proves insufficient.

### 4. Dynamic release (`release_ball`)

New flag `dynamic=True`: remove the grip constraint and **skip
`resetBaseVelocity` entirely** — the ball keeps the velocity the physics engine
gave it while dragged by the constraint. Keep `keep_collision_disabled=True`
(the release-collision fix). Actual release velocity recorded via
`getBaseVelocity(ball_id)` immediately after constraint removal.

### 5. Measurement sweep (`measure_tracking_error.py`, new script)

Sim is deterministic → the error distribution comes from command diversity, not
repeats. Grid: release speed u ∈ [0.3, 1.0] in 25 steps × target angle ∈
[−30°, 30°] in 9 steps ≈ 225 throws on `kinova_gen3_dyn`.

Per throw record: `v_cmd`, planned `v_achieved` (after qd_max clip), actual
ball velocity at release, release-position error, landing point, applied
time-scaling factor. Throws with release-position error > 5 cm are flagged and
excluded from the fit (reported, not silently dropped).

Output: `.npz` of all records, figure of Δv components vs u, stats table
(mean/std bias per component, per u-band) printed to stdout.

### 6. Noise model (`robot_arm/noise_models.py`)

`TrackingErrorNoise`:
- Fit: per-component linear bias `Δv_i = a_i·u + b_i` by least squares +
  Gaussian residual covariance (full 3×3).
- Constructor `TrackingErrorNoise.from_measurements(npz_path)`.
- Exposes the same `pybullet_release_vel(v_cmd, ee_vel)` interface as the
  existing arm-noise classes so it drops into later noise-robust training
  unchanged.

## Validation

1. **Gravity-hold**: zero-velocity setpoint at neutral pose for 2 s of sim;
   EE drift < 1 cm. Proves torque path + inverse-dynamics wiring.
2. **Slow-tracking**: single u = 0.3 throw; joint tracking error < ~0.02 rad
   through the throw phase. Gates gain tuning before the full sweep.
3. **Sim2sim gap**: replay the trained kinova policy (seed 1 checkpoint,
   `results_mc_pilot_pb_A_kinova_gen3/1`) under dynamic release, 10 throws;
   report landing error vs the kinematic baseline. Quantifies what
   noise-robust training must later absorb. Must use the true
   `PyBulletThrowingSystem.rollout` pipeline (hand-rolled replays had ~10 cm
   systematic discrepancy in earlier work).

## Error handling summary

- Time-scaling failure after 6 iterations → raise with per-joint torque report.
- NaN torque → raise with state dump.
- Release-position error > 5 cm in sweep → flag + exclude from fit, report count.

## Risks / notes

- 240 Hz sim vs 1 kHz hardware control rate — documented delta, not fixed here.
- Kp/Kd initial (100, 20) may need retuning; slow-tracking test is the gate.
- If time scaling stretches t_r far at high u, the release pose stays the same
  but timing-derived noise characteristics change — the sweep records the
  scaling factor so this is visible in analysis.
