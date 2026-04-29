# MC-PILOT Implementation Plan — PyBullet Arm + Noise Extension

_Previous plan archived at `old/implementation_plan.md` (covered the baseline numpy-sim implementation, which is now complete and validated in `mc-pilot/` and `mc-pilot-elevated/`). This document is the forward-looking plan for the next extension only._

---

## Motivation

The existing `mc-pilot/` (ground targets) and `mc-pilot-elevated/` (Exploration 1) implementations both work — see `change_history.md`. The RL learns to throw to within ~0.1m of the target across the reachable workspace in ~10 throws.

But the existing setup has a structural weakness as a **demonstration** of RL's value:

- The policy outputs a release velocity and that *exact* velocity is applied to the ball. There is no gap between "what the policy commands" and "what physically happens."
- Ball free-flight is deterministic given release velocity. A non-RL solver could invert the ballistic equation numerically once and achieve near-perfect accuracy.
- So the current setup only shows RL can learn a known physics model — not that it can cope with any *actual uncertainty*.

The paper (MC-PILOT, Turcato et al. 2025) handles this correctly: its Modification 2 (gripper delay estimation) exists precisely because the arm's gripper opens with unknown random delay `t_d ~ U(a, a+b)`, which perturbs the actual release moment along the arm's moving trajectory — and therefore perturbs the actual release velocity. Bayesian Optimization calibrates this delay from hardware data. **Our simulation-only work skipped this module**, which is why our setup has no RL-justifying stochasticity.

### Goal of this extension

Introduce a PyBullet arm + ball simulator with injected arm-side noise so that:

1. **The policy can no longer "just solve the inverse problem"** — the arm delivers a *noisy realisation* of the commanded velocity, and the RL must learn a policy robust to that noise.
2. **We get a visualisation pipeline** — a PyBullet GUI that can be extended for demos, slides, and video.
3. **Existing work is untouched** — `mc-pilot/` and `mc-pilot-elevated/` stay as-is, validated baselines. All new code lives in a parallel folder.

---

## Plan Verification (against the three constraints)

### 1. Does it give you an expandable visualisation system?

**Yes.** PyBullet ships with Python bindings and built-in URDFs for Kuka iiwa and Franka Panda (via the `pybullet_data` package). The visualisation layer is pure PyBullet API calls: camera, lighting, overlays, recording, marker shapes, trajectory traces. Everything visible in the GUI is a Python object you can extend without touching the RL stack.

Extension points we will expose explicitly:
- Camera spec (position, yaw, pitch, distance) as a config.
- Target/landing markers as `loadURDF` visual shapes — easy to restyle.
- Trajectory trace rendering via `addUserDebugLine` — user can toggle on/off.
- Video capture via `startStateLogging(STATE_LOGGING_VIDEO_MP4, ...)`.
- Swap arm URDF in one place (`arm_controller.py`) if we want Franka instead of Kuka.

### 2. Does it stay faithful to the paper?

**Yes — to the spirit, not the letter.** The paper's claim is that MC-PILOT learns a throw policy that is robust to a specific real-world source of uncertainty (gripper-delay-induced release perturbation). Our extension reproduces that *structural* story with a simulated analogue. We do not need to match the paper's exact delay distribution — we need to introduce arm-side stochasticity that forces the RL to do non-trivial work. That is exactly what injected arm noise does.

Concretely:
- **Paper's real-world uncertainty:** gripper delay `t_d` → perturbed (release_pos, release_vel) at the actual release instant.
- **Our sim analogue:** release-timing jitter (directly matches) OR joint-torque noise (slightly different mechanism, same effect class).
- **Both produce the same signal** to the GP: slightly different actual release state across throws, consistent free-flight dynamics thereafter.

Crucially, the paper's data flow is preserved: **GP learns free-flight ball dynamics; arm is outside the learned model**. Arm noise enters via initial-state variability in the training data and via initial-state sampling in the MC policy loop.

### 3. Does it leave existing work untouched?

**Yes.** The entire extension lives in a new folder `mc-pilot-pybullet/` created by copying `mc-pilot/` verbatim. No edits to `mc-pilot/` or `mc-pilot-elevated/`. All experimental results, change history, and validated configs for both existing implementations remain unaffected.

---

## One Crucial Constraint You Must Preserve

MC-PILOT's policy optimisation runs `M=400` particles through `apply_policy()` for `Nopt=1500` steps per trial. That inner loop **must stay in torch/GP-space** — it is not a full simulator call. PyBullet is roughly 100× too slow to run inside that loop.

PyBullet enters only at **data-collection time** (once per real throw — roughly 10 exploration + 10 trial throws = ~20 rollouts per seed). This mirrors the paper's use of Gazebo.

The design below respects this: PyBullet is the data-collection simulator and the visualisation engine; the MC inner loop is unchanged except for a tiny initial-state perturbation step.

---

## File Structure

```text
project/
  mc-pilot/                          # UNTOUCHED — validated baseline (ground targets)
  mc-pilot-elevated/                 # UNTOUCHED — validated Exploration 1 (elevated targets)
  mc-pilot-pybullet/                 # NEW — full copy of mc-pilot, then extended
    simulation_class/
      model.py                       # UNCHANGED (numpy ThrowingSystem kept as ablation baseline)
      model_pybullet.py              # NEW — PyBulletThrowingSystem, same rollout() signature
    robot_arm/                       # NEW module
      __init__.py
      arm_controller.py              # IK + Jacobian-based throw trajectory planner
      noise_models.py                # release-timing jitter, torque noise, velocity bias
      assets/                        # URDF (optional — fall back to pybullet_data)
    policy_learning/
      MC_PILCO.py                    # MODIFY MC_PILOT: add arm_noise kwarg + perturb v3d in apply_policy
      Policy.py                      # UNCHANGED
      Cost_function.py               # UNCHANGED
    model_learning/                  # UNCHANGED
    gpr_lib/                         # UNCHANGED
    test_mc_pilot_pb_A.py            # NEW — Config A reproduction with PyBullet, zero arm noise
    test_mc_pilot_pb_A_noisy.py      # NEW — Config A with arm noise — the core demo
    demo_pybullet_gui.py             # NEW — replay trained policy in GUI mode
```

---

## Components

### 1. `PyBulletThrowingSystem` (simulation_class/model_pybullet.py)

Drop-in replacement for the numpy `ThrowingSystem`. Same constructor parameters, same `rollout(s0, policy, T, dt, noise)` signature, same 3-tuple return. Callers in `MC_PILOT.get_data_from_system()` need zero changes.

**Per-rollout sequence:**
1. Reset PyBullet world in `p.DIRECT` mode (no GUI during training).
2. Load ground plane, arm URDF, ball.
3. Reset arm to a fixed neutral pre-throw joint configuration.
4. Attach ball to end-effector via `p.createConstraint` (rigid weld).
5. Call `policy(s0, t=0)` once → scalar release speed (exactly as today).
6. Compute commanded release velocity vector from target geometry (same math as current `_speed_to_velocity`).
7. `arm_controller.plan_throw(v_cmd)` → list of `(t, q_target, qd_target)` waypoints.
8. Step PyBullet at `1/dt` Hz, commanding joints via `p.setJointMotorControlArray(..., POSITION_CONTROL, ...)` with noise injection at each step.
9. At the planned release step `(± t_d sampled from noise_model)`: `p.removeConstraint(grip)` — ball is now free.
10. Continue stepping until ball `z ≤ 0` or `T` elapsed. Record ball state each step.
11. Return `(noisy_states, inputs, clean_states)` in the same 8-D augmented layout: `[x,y,z, vx,vy,vz, Px,Py]`.

**Why this matches the existing interface exactly:** the GP consumes `(state, input) → next_state` pairs. As long as we emit state samples in the 8-D layout and inputs as `[speed, 0, 0, ...]`, `Ballistic_Model_learning` is none the wiser about the underlying simulator.

### 2. `arm_controller.py` — Detailed Design

This is the load-bearing, non-trivial new component. The rest of the plan is configuration and wiring. This section is deliberately long.

#### 2.1 Arm choice and world setup

| Choice | Value | Why |
| :--- | :--- | :--- |
| Arm | **KUKA iiwa7** (7-DOF) | Ships with `pybullet_data`. Path: `pybullet_data.getDataPath() + '/kuka_iiwa/model.urdf'`. Simpler kinematics than Panda; well-tested URDF. |
| Mount | Base at world origin, z=0 | Workspace roughly a 1.1m sphere. Floor plane at z=0. |
| EE link index | 6 (last link, flange) | PyBullet auto-numbering; verify with `p.getJointInfo` on startup. |
| Ball | Sphere, m=0.0577, r=0.0327 (tennis ball) | Matches existing numpy sim. |
| Gripper | Not modeled — replaced by rigid weld constraint | Gripper kinematics are visual noise; MC-PILOT only cares about the release event. |

Alternative if the iiwa ever hits a wall: **Franka Panda**, also in `pybullet_data/franka_panda/panda.urdf`. Swap is one line in `arm_controller.py`.

#### 2.2 Velocity reachability — a real constraint

KUKA iiwa7 nominal joint-velocity limits (rad/s): `[1.71, 1.71, 1.75, 2.27, 2.44, 3.14, 3.14]`.

Heuristic peak end-effector speed for a shoulder-only swing: `qd_max * lever_arm ≈ 1.71 * 0.7 ≈ 1.2 m/s`. With multi-joint coordination (shoulder + elbow), realistic peak is **~2.5 m/s**. The paper's `uM = 3.5 m/s` is **aggressive** for an iiwa.

**Three ways to handle this:**
1. **Retarget the experiments** (recommended): set `uM = 2.5 m/s` for the PyBullet scripts, and pick `lm/lM` accordingly so all targets remain reachable. The RL algorithm doesn't care about absolute velocity scales.
2. **Uncap joint limits in simulation**: PyBullet's `setJointMotorControl2` only enforces `maxVelocity` if you pass it. Omit it and the solver will command whatever torques are needed. Physically unrealistic but visually fine.
3. **Accept systematic undershoot** as a source of arm-side "noise" the RL must cope with. This is legitimate but muddies the demo.

Plan default: **option 1**. Write `test_mc_pilot_pb_A.py` with `uM=2.5, lm=0.5, lM=1.1`.

**Downstream consequence — lengthscales must rescale.** With `lm=0.5, lM=1.1`, the target range is 0.6m — different from the existing configs. Per the finding established in `change_history.md` (the E-Strat5 rule: `ls ≈ 0.15 × target_range`), the new test script needs:

```python
lengthscales_init = np.array([0.09, 0.09])   # 0.15 * 0.6m target range
```

Leaving this at the paper's default `[1.0, 1.0]` will reproduce the E-Strat1–4 failure modes (RBF policy blindness, policy collapses to a constant speed). Do not skip this step.

#### 2.3 Throw motion phases (5-phase schedule)

```
  t = 0         : neutral pose, ball gripped, arm stationary
  t ∈ [0, t_w]  : windup — arm moves to cocked pose
  t ∈ [t_w, t_r]: throw — arm accelerates to release pose
  t = t_r       : RELEASE — constraint removed
  t ∈ [t_r, T]  : follow-through — arm decelerates; ball in free flight
```

Default timing: `t_w = 0.3s`, `t_r = 0.6s`, `T = 1.0s`. These fit the existing `T = 0.55–1.0` horizons in current test scripts; data-collection `T` may be set longer than ball flight time to allow arm follow-through to be rendered, without extending the free-flight sim (the GP only trains on post-release steps).

Windup is mostly cosmetic but it matters for two things:
- Makes the throw **visually recognisable** (GUI demo).
- Gives the trajectory solver **a wider range of joint velocities** to realise `qd_release` without violating joint limits.

#### 2.4 Pose definitions

**`q_neutral`** — hardcoded upright pose, arm relaxed.
```
q_neutral = [0, -π/4, 0, π/2, 0, -π/4, 0]   # iiwa joints, radians — tune in Stage 0
```

**`q_release`** — arm extended toward target, EE at `release_pos`, EE z-axis oriented along `v_cmd`.
- Compute via PyBullet IK:
  ```python
  q_release = p.calculateInverseKinematics(
      arm_id, ee_link,
      targetPosition=release_pos,
      targetOrientation=quat_aligned_with(v_cmd),   # EE z-axis along v_cmd
      lowerLimits=q_lo, upperLimits=q_hi,
      restPoses=q_neutral,                          # bias IK toward a reachable solution
      maxNumIterations=200, residualThreshold=1e-4,
  )
  ```
- IK may return a solution with EE velocity component along `v_cmd` being blocked by a near-singular Jacobian — detect by checking condition number of `J_lin` and re-IK with perturbed orientation if bad.

**`q_windup`** — arm cocked opposite the throw direction, biased away from `q_release` along the throw axis. Simple construction:
```
q_windup = q_neutral  +  (q_release - q_neutral) * (-0.5)
```
i.e. mirror past neutral by 50% of the release displacement. Then clip to joint limits. This is a heuristic — tune visually in Stage 0.

#### 2.5 Trajectory generation (cubic polynomials, per joint)

For each joint `i` independently, solve a cubic polynomial on `[t_w, t_r]`:

```
q_i(t)  = a₀ + a₁·τ + a₂·τ² + a₃·τ³     where τ = t - t_w
qd_i(t) = a₁ + 2a₂·τ + 3a₃·τ²
```

Four boundary conditions:
```
q_i(t_w)  = q_windup_i                    → a₀ = q_windup_i
qd_i(t_w) = 0                             → a₁ = 0
q_i(t_r)  = q_release_i                   → a₀ + a₂·Δ² + a₃·Δ³ = q_release_i
qd_i(t_r) = qd_release_i                  → 2a₂·Δ + 3a₃·Δ² = qd_release_i
```
with `Δ = t_r - t_w`. Closed-form for `a₂, a₃`:
```
a₃ = (qd_release_i·Δ - 2(q_release_i - q_windup_i)) / Δ³
a₂ = (3(q_release_i - q_windup_i) - qd_release_i·Δ) / Δ²
```

**Neutral → windup** segment `[0, t_w]`: same math with `qd_i(0)=0`, `qd_i(t_w)=0`. Reduces to a cubic with both velocities zero — a standard "rest-to-rest" move.

**Follow-through** segment `[t_r, T]`: rest-to-rest cubic from `(q_release, qd_release)` to `(q_followthrough, 0)`. `q_followthrough` is a hand-picked relaxed pose past the release — cosmetic only.

#### 2.6 Computing `qd_release` — Jacobian pseudoinverse

Given:
- `q_release` (7-vector)
- Desired EE linear velocity `v_cmd ∈ ℝ³`

Compute:
```python
# PyBullet Jacobian: 6×7 (linear 3 rows + angular 3 rows)
J_lin, J_ang = p.calculateJacobian(
    arm_id, ee_link,
    localPosition=[0, 0, 0],        # EE origin in link-local frame
    objPositions=list(q_release),
    objVelocities=[0.0]*7,
    objAccelerations=[0.0]*7,
)
J_lin = np.array(J_lin)              # shape (3, 7)

# Minimum-norm joint velocity producing v_cmd at EE (linear only):
# qd = J_lin⁺ · v_cmd  = J_linᵀ (J_lin J_linᵀ)⁻¹ v_cmd
qd_release = np.linalg.pinv(J_lin) @ v_cmd
```

**Feasibility check**:
```python
qd_ratio = np.abs(qd_release) / qd_max        # qd_max = iiwa nominal limits
if qd_ratio.max() > 1.0:
    qd_release /= qd_ratio.max()              # uniform scale-down
    v_cmd_achieved = J_lin @ qd_release       # log the actual velocity
    warn(f"commanded {np.linalg.norm(v_cmd):.2f} m/s, achievable {np.linalg.norm(v_cmd_achieved):.2f} m/s")
```

The 4-D null space of `J_lin` (7 joints − 3 linear constraints) is unused. A future refinement could use it to avoid joint-limit collisions, but for the initial implementation zero null-space is fine.

#### 2.7 Ball gripping and release

**Grip creation** (once per rollout, before `t=0`):
```python
ball_start = ee_position_at(q_neutral) + ee_grip_offset   # slightly forward of EE
ball_id = p.createMultiBody(
    baseMass=0.0577,
    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.0327),
    baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.0327, rgbaColor=[1,1,0,1]),
    basePosition=ball_start,
)
grip = p.createConstraint(
    parentBodyUniqueId=arm_id, parentLinkIndex=ee_link,
    childBodyUniqueId=ball_id, childLinkIndex=-1,
    jointType=p.JOINT_FIXED, jointAxis=[0,0,0],
    parentFramePosition=ee_grip_offset, childFramePosition=[0,0,0],
)
```

**Release** (exactly one step at `t = t_r`):
```python
# Record actual ball state at the release instant
pos, _ = p.getBasePositionAndOrientation(ball_id)
lin_vel, _ = p.getBaseVelocity(ball_id)

# Remove the weld — ball is now free
p.removeConstraint(grip)

# Optional: override the ball's linear velocity to exactly match the EE velocity
# at release (cleaner story than trusting constraint-induced velocity).
# Can also be used to inject "release velocity noise":
p.resetBaseVelocity(ball_id, linearVelocity=ee_velocity_at_release + dv_noise)
```

**Free flight** (after release): **do not use PyBullet's built-in damping for the ball**. We reuse the existing `_ball_accel()` from `simulation_class/model.py`, which implements the paper's Eq. 35 (Reynolds-dependent quadratic drag). This keeps ball physics apples-to-apples with `mc-pilot/` and `mc-pilot-elevated/`, so all three implementations produce comparable numbers.

```python
# On ball creation, disable PyBullet's default damping:
p.changeDynamics(ball_id, -1, linearDamping=0.0, angularDamping=0.0)

# Each sim step after release, inject the paper's drag force manually:
pos, _ = p.getBasePositionAndOrientation(ball_id)
lin_vel, _ = p.getBaseVelocity(ball_id)
a_total = _ball_accel(np.array(pos), np.array(lin_vel),
                      mass=ball_mass, radius=ball_radius, wind=np.zeros(3))
# Strip gravity (PyBullet applies it natively) and convert accel → force
a_drag = a_total - np.array([0, 0, -9.81])
F_drag = ball_mass * a_drag
p.applyExternalForce(ball_id, -1, F_drag, [0, 0, 0], p.WORLD_FRAME)
```

Record `(pos, vel)` each sim step for the trajectory log. `_ball_accel` is imported, not reimplemented — the existing numpy function is the single source of truth for ball aerodynamics across all three implementations.

#### 2.8 Sub-step timing and the dt mismatch

PyBullet defaults to 240 Hz physics. Our `dt = 0.02s` = 50 Hz. Two options:

1. **Match them**: `p.setTimeStep(dt); p.stepSimulation()` once per `dt`. Simple. Less accurate for rigid-body contacts, but the arm's cubic trajectory is smooth and the ball uses our hand-coded `_ball_accel` (§2.7) — so integration accuracy is dominated by our own Euler step, not PyBullet's solver.
2. **Sub-step**: run PyBullet at 240 Hz internally, record ball state every `dt`. More accurate for arm motor tracking. More code.

**Plan default: option 1** for the first pass. Revisit if arm motor tracking is visibly laggy.

Because ball aerodynamics are now handled by `_ball_accel` (§2.7), ball-trajectory accuracy is identical to the numpy sim regardless of which option is picked here — this knob only affects arm-side integration.

#### 2.9 Noise injection — where and which layer

Four possible injection points, increasing in physicality and complexity:

| Layer | Mechanism | Matches `apply_policy`? | Complexity |
| :--- | :--- | :--- | :--- |
| (a) Release velocity override | `resetBaseVelocity(ball, v_ee + dv)` at release | Exact — `dv ~ N(0, σ²)` in both paths | Trivial |
| (b) Release timing jitter | Shift release step by `round(U(0, b)/dt)` | Approximate — calibrate Gaussian from rollouts | Easy |
| (c) Joint tracking noise | Add Gaussian to `targetPosition` / `targetVelocity` | Approximate — calibrate from rollouts | Medium |
| (d) Joint torque noise | Add Gaussian to `force=` in motor control | Approximate — calibrate from rollouts | Medium |

**Plan default**: Start with **(a)** — exact match between data collection and `apply_policy`, zero calibration. Once working, add **(b)** as a second noise source for visual realism (the arm visibly releases at slightly different points).

(c) and (d) are better science (full arm-in-the-loop noise) but need calibration rollouts to map joint noise → EE velocity noise distribution. Defer unless time permits.

#### 2.10 `ArmController` API (final)

```python
class ArmController:
    def __init__(self, sim_client, urdf_path, mount_pose, q_neutral):
        """Load URDF, record EE link index, joint limits, velocity limits."""

    def reset(self):
        """Set all joints to q_neutral, zero velocities. Call once per rollout."""

    def attach_ball(self, ball_id):
        """Create fixed constraint between EE and ball. Store grip id."""

    def plan_throw(self, v_cmd, release_pos, t_w, t_r, T):
        """Return dict of polynomial coefficients per joint for phases:
           neutral→windup on [0, t_w], windup→release on [t_w, t_r], release→rest on [t_r, T].
           Also return (q_release, qd_release, v_cmd_achieved) for logging."""

    def get_setpoint(self, coeffs, t) -> (q, qd):
        """Evaluate the piecewise-cubic trajectory at time t."""

    def step(self, q_target, qd_target, noise=None):
        """One PyBullet sim step. Commands joints via POSITION_CONTROL with optional noise."""

    def release_ball(self, dv_noise=None):
        """Remove grip constraint. Optionally override ball velocity with v_ee + dv_noise."""

    def ee_state(self) -> (pos, lin_vel, orient, ang_vel):
        """Query end-effector state in world frame."""
```

#### 2.11 Failure modes — what to watch for in Stage 0

| Symptom | Likely cause | Fix |
| :--- | :--- | :--- |
| Ball drops at t=0, doesn't move with arm | Constraint not created, or created after ball has already fallen | Place ball exactly at grip frame BEFORE `createConstraint` |
| Ball swings with arm but flies sideways at release | EE orientation wrong — `qd_release` produces velocity in unintended direction | Verify `J_lin @ qd_release ≈ v_cmd` numerically in Stage 1 |
| Ball released with large spin | Angular-velocity coupling through constraint | Explicit `resetBaseVelocity(linearVelocity=..., angularVelocity=[0,0,0])` on release |
| Arm hits joint limit mid-swing | Cubic trajectory overshoots | Shorten `Δ = t_r - t_w` OR scale down `qd_release` OR pick `q_windup` with more margin |
| Landing position wildly off from commanded | EE velocity ≠ commanded velocity at the release instant | Log `ee_state()` at release, compare `lin_vel` to `v_cmd`. Likely cause: cubic reaches `qd_release` at `t=t_r` but PyBullet's tracking lags — increase position-control gains |
| PyBullet tracking error makes throws weak | Default POSITION_CONTROL gains too soft | Pass larger `positionGain`, `velocityGain` to `setJointMotorControl2`, or use hybrid position+velocity control |

#### 2.12 De-risked progression — staged plan for this component alone

| Stage | Deliverable | Success criterion | Est. effort |
| :--- | :--- | :--- | :--- |
| **0.** Load arm and visualise | Script that shows KUKA iiwa in a `p.GUI` window, ball welded to EE, 5-second static hold | Arm loads, ball stays attached | 2 hrs |
| **1.** Scripted single throw | Hardcode `v_cmd = (2, 0, 2)`, run `plan_throw`, step trajectory, release, watch ball fly | Ball released from EE, lands somewhere near `(vx·t + ..., ..., 0)` | 1 day |
| **2.** Parameterise by `v_cmd` | Replace hardcode with function call; test 5 different `v_cmd` values | Landing position scales correctly with `v_cmd` | 0.5 day |
| **3.** Verify `v_cmd` tracking accuracy | At release instant, compare `ee_state().lin_vel` to `v_cmd` | Match within 10%, or document bias | 0.5 day |
| **4.** Wrap as `PyBulletThrowingSystem` | Class with `rollout(s0, policy, T, dt, noise)` signature | Numpy-sim test script runs drop-in with PyBullet backend | 0.5–1 day |
| **5.** Noise injection (level a) | `ArmNoise.VelocityBiasNoise` wired in | Distribution of landing points widens with σ; matches noise injected in `apply_policy` | 0.5 day |
| **6.** GUI demo | `demo_pybullet_gui.py` loads trained policy, renders throws | Visually recognisable throwing motion; ball lands near target | 1 day |

**Total estimated effort for the arm component: 4–6 working days** for someone with PyBullet familiarity; add 2–3 days for the first-time PyBullet user. The hardest stage is (1)→(3) — getting the cubic trajectory to produce the right EE velocity at the right instant. Everything downstream of that is straightforward wiring.

This staged plan is **independent of MC-PILOT integration** — stages 0–3 can be done in a standalone script before touching any of the existing RL code. Fail fast there; don't try to debug arm kinematics through the full RL pipeline.

### 3. `noise_models.py`

Plug-in noise classes sharing a common interface. Users can write their own by subclassing.

**Base class:**
```python
class ArmNoise:
    def perturb_command(self, q, qd, step) -> (q_noisy, qd_noisy): ...
    def sample_release_offset(self) -> int:       # in timesteps
    def sample_initial_velocity_noise(self, n) -> np.ndarray:  # [n, 3] — for apply_policy
```

**Built-ins:**
- `ReleaseTimingJitter(b)` — release step shifted by `round(U(0, b)/dt)` timesteps. Direct analogue of paper's gripper delay. `sample_initial_velocity_noise` returns a Gaussian approximation calibrated from rollouts (see below).
- `JointTorqueNoise(sigma)` — adds Gaussian torque each step. More physical, couples nonlinearly into EE velocity.
- `VelocityBiasNoise(sigma)` — adds Gaussian directly to commanded EE velocity. Simplest, deterministic mapping to `apply_policy` — recommended for the first pass.

**Recommended starting choice:** `VelocityBiasNoise(sigma=0.15 m/s)`. Mathematically identical between data collection and `apply_policy` (no calibration required), gives a clean demo.

### 4. `MC_PILOT` changes (policy_learning/MC_PILCO.py)

Two small edits to the existing `MC_PILOT` class. Everything else — GP model learning, cost function, particle freezing, policy RBF — is unchanged.

**Edit 1 — `__init__` signature**: add an optional `arm_noise` kwarg, store it on the instance.

```python
def __init__(self, ..., arm_noise=None, ...):
    ...
    self.arm_noise = arm_noise   # shared with PyBulletThrowingSystem; both paths sample from the same instance
```

When `arm_noise=None`, the class behaves identically to the current `mc-pilot/` implementation — no regression for the zero-noise ablation.

**Edit 2 — `apply_policy` body**: one block inserted after `v3d` is computed from policy speed and before it becomes the initial particle state:

```python
# Arm-side noise: policy commands v_cmd; arm delivers v_cmd + dv.
# Must match the noise distribution used by PyBulletThrowingSystem during data collection.
if self.arm_noise is not None:
    dv_np = self.arm_noise.sample_initial_velocity_noise(num_particles)   # [M, 3], numpy
    dv = torch.tensor(dv_np, dtype=self.dtype, device=self.device)         # detached from graph
    v3d = v3d + dv
```

Gradient chain stays intact: `v3d` has `requires_grad=True` via the policy; `dv` is a constant w.r.t. policy params; the sum retains its gradient through the policy path. This is the standard treatment of observation-like noise in reparameterised MC objectives.

These two edits are the **only algorithmic changes** in the entire extension.

### 5. `demo_pybullet_gui.py`

Standalone script. No training. Loads a trained policy from `results_*/seed/log.pkl`, spins up `p.GUI`, runs N throws to demo targets, renders:
- Arm executing the throw
- Ball trajectory trace (`addUserDebugLine` between consecutive positions)
- Target marker (coloured sphere at `(Px, Py, 0)`)
- Landing marker and error text overlay

Optional CLI flags: `--record video.mp4`, `--targets path.csv`, `--slow N` (slowdown factor).

---

## Execution Order

| Step | What | Files | Verification |
| :--- | :--- | :--- | :--- |
| 0 | `cp -r mc-pilot mc-pilot-pybullet` | shell | `python test_mc_pilot.py -seed 1 -num_trials 1` still runs |
| 1 | `pip install pybullet` in project venv | — | `python -c "import pybullet, pybullet_data"` |
| 2 | Stand up arm standalone: load URDF, IK, `plan_throw`, single throw in `p.GUI` | `robot_arm/arm_controller.py` | Arm visibly throws, ball lands. No MC-PILOT yet. |
| 3 | Wrap as `PyBulletThrowingSystem` with numpy-sim interface | `simulation_class/model_pybullet.py` | Swap into one of the existing `test_mc_pilot_*.py` scripts with `arm_noise=None`; training converges comparably to numpy sim (hit rate within ±10%). |
| 4 | Add noise models; run data collection with `VelocityBiasNoise(sigma=0.15)` but **no** change to `apply_policy` | `robot_arm/noise_models.py` | Policy still learns but landing error inflated. Demonstrates that arm noise hurts a naive policy. |
| 5 | Inject noise into `apply_policy` via the `arm_noise` kwarg | `policy_learning/MC_PILCO.py` | Policy learns robust behaviour; hit rate recovers vs step 4. This is the scientific result. |
| 6 | Sweep `sigma` across {0.0, 0.05, 0.10, 0.15, 0.20, 0.30} | `test_mc_pilot_pb_A_noisy.py` (parameterised) | Produces a hit-rate-vs-noise curve — main experimental output. |
| 7 | Build `demo_pybullet_gui.py` | new file | Visual demo for slides/video. |

Steps 0–5 must happen in order. Steps 6 and 7 are independent and can swap.

---

## Interface Contract (so the extension stays decoupled)

The `MC_PILOT` class accepts a simulator via `throwing_system=...`. Any object implementing:

```python
def rollout(self, s0, policy, T, dt, noise) -> (noisy_states, inputs, clean_states):
    # shapes: [n, 8], [n, 1], [n, 8]  where n = int(T/dt)+1 or earlier on landing
```

is a valid simulator. `PyBulletThrowingSystem` satisfies this contract. The numpy `ThrowingSystem` satisfies this contract. We can switch between them by changing one line of the test script.

Similarly, `arm_noise=...` accepts any object implementing the `ArmNoise` interface above. Users (you) can drop in new noise models without touching the RL code.

---

## Risks and Mitigations

| Risk | Mitigation |
| :--- | :--- |
| PyBullet's built-in drag (via `linearDamping`) ≠ paper's Reynolds-dependent drag | Resolved by design — ball damping is disabled and the paper's `_ball_accel` (Eq. 35) is applied manually each sim step (§2.7). Ball physics match the numpy sim exactly; landing distances across `mc-pilot/`, `mc-pilot-elevated/`, and `mc-pilot-pybullet/` are directly comparable. |
| IK / plan_throw fails near workspace boundary | Arm reachability is a function of `(lm, lM)` and release point. For Config A–E target ranges, a KUKA iiwa mounted at origin easily covers everything. If an IK solve fails, warn and skip that rollout — the exploration-policy distribution will sample a reachable one on the next draw. |
| PyBullet is slow relative to numpy sim | Only run in `p.DIRECT` during training. Each rollout is ~1–2s. For 20 rollouts per seed this adds ~30s total. Fine. |
| Arm-grip detach timing is finicky | Use `p.createConstraint` (rigid weld) + `p.removeConstraint` at the release step rather than friction-based grasp. Deterministic, no tuning. |
| PyBullet RNG not controlled → training not reproducible | Seed explicitly each rollout: `np.random.seed(...)`, `torch.manual_seed(...)` and pass same seed to the noise model's internal RNG. PyBullet's physics is deterministic given constant inputs; the only stochastic input is our injected noise. |
| `apply_policy` noise distribution must match data-collection noise, or the GP and policy disagree | Both paths source their noise from the *same* `ArmNoise` instance. Enforce by construction — `PyBulletThrowingSystem` and `MC_PILOT` both receive the same `arm_noise` object. |

---

## Success Criteria

1. **Zero-noise parity:** `PyBulletThrowingSystem` with `arm_noise=None` reproduces Config A hit rate within ±10% of the numpy-sim baseline (currently 5/5 per change_history).
2. **Noise demonstrates RL's value:** At `sigma=0.15 m/s`, a policy trained *without* arm-noise awareness (step 4) gets measurably worse hit rate than one trained *with* arm-noise awareness (step 5). This is the scientific result.
3. **Visual demo:** `demo_pybullet_gui.py` renders a recognisable throw with arm, ball trajectory, target, and landing marker. User can swap camera and markers by editing one config block.
4. **No regressions:** `mc-pilot/` and `mc-pilot-elevated/` run unchanged; their experimental results in `change_history.md` remain valid.

---

## What This Plan Deliberately Does NOT Do

- Does not touch `mc-pilot/` or `mc-pilot-elevated/`.
- Does not rewrite the GP model learning, cost function, or RBF policy.
- Does *not* trust PyBullet's built-in ball damping. The paper's Eq. 35 drag is injected manually via the existing `_ball_accel` function (§2.7), keeping ball physics identical across all three implementations.
- Does not implement the paper's Bayesian Optimization delay estimation (Modification 2). We inject noise with a *known* distribution because we control the simulator; no calibration needed.
- Does not extend to elevated targets. That can be a follow-up: once arm+noise is working here, port the changes into a parallel `mc-pilot-pybullet-elevated/` folder using the same pattern.
