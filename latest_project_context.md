# Project Context — Model-Based RL for Robotic Throwing (MC-PILOT / Kinova Gen3)

Generated 2026-08-24 from a direct scan of the source tree, config files, pose
tables, result archives and the passing test suites. Every number below was read
out of code or result data in this repository, not from prior write-ups.

---

## 1. Project Name & Core Purpose

**Data-efficient robotic throwing via model-based reinforcement learning
(MC-PILOT), extended from simulation to a real Kinova Gen3 7-DOF arm.**

The system teaches a robot arm to throw a ball into a target bin at an arbitrary
position within a reachable wedge, learning the ball's flight dynamics from a
handful of trials (~10) rather than thousands. A Gaussian-Process model of
free-flight dynamics is learned from data; a policy mapping *target position →
release speed* is then optimized entirely inside that learned model by
Monte-Carlo policy gradient, with no additional robot interaction.

Scope of the work in this repository:

1. **Reproduce** the MC-PILOT algorithm (Turcato et al., arXiv:2502.05595) from
   the MC-PILCO reference implementation.
2. **Extend** it along five simulation axes — elevated release, real arm
   physics + actuator noise, elevated targets, wind fields, vision-estimated
   targets — each as an independent full fork.
3. **Transfer** it to physical hardware: a throw-pose search that respects real
   actuator limits, a torque-controlled dynamic release, and a safety-gated
   Kortex executor that streams the identical planned trajectory to a real
   Kinova Gen3.

Scale: **~77,000 lines of project Python** across 7 sibling forks (upstream
MC-PILCO reference is a separate, unmodified 7,000 lines). The active
hardware-facing fork is **~31,000 lines**.

---

## 2. Tech Stack

**Languages / Core**
- Python 3.10 (system interpreter; float64 numerics throughout)
- NumPy, SciPy (`scipy.optimize.linprog`, HiGHS solver)

**Machine Learning**
- **PyTorch 2.9** — GP hyperparameter optimization, RBF policy network,
  Monte-Carlo policy-gradient backprop through particle rollouts (Adam)
- Custom sparse-GP library (`gpr_lib`): RBF/Squared-Exponential kernels,
  exact / SOR (Subset of Regressors) / SOD (Subset of Data) / L1-selected
  inference, Gaussian likelihood, marginal-log-likelihood training

**Simulation & Physics**
- **PyBullet** — rigid-body arm dynamics, URDF loading, inverse dynamics
  (`calculateInverseDynamics`), analytical Jacobians, damped-least-squares IK,
  torque control (`TORQUE_CONTROL`), constraint-based grasping
- Custom NumPy ballistic integrator with quadratic aerodynamic drag
- URDFs: Kinova Gen3 (ros_kortex GEN3-7DOF-VISION V12), Franka Panda,
  KUKA iiwa7, UFactory xArm6

**Robotics / Control**
- Computed-torque control (inverse dynamics feedforward + PD, `kp=400`,
  `kd=60`, ζ≈1.5, ωₙ≈20 rad/s) with Jacobian-transpose payload compensation
- Piecewise-cubic joint-space trajectory generation (C¹), 3-phase
  windup → throw → follow-through
- Linear programming for direction-constrained release-velocity allocation
- Forward kinematics / analytic Jacobian / pseudo-inverse velocity IK
- **Kinova Kortex API** (`kortex_api` 2.6.0.post3) — `Base`, `BaseCyclic`,
  `ControlConfig`, `DeviceConfig` services; TCP command channel (port 10000)
  + UDP real-time feedback channel (port 10001); `SendJointSpeedsCommand`
  streaming in `SINGLE_LEVEL_SERVOING`

**Perception / Vision**
- **OpenCV** — ArUco / ChArUco detection, `solvePnP`, Brown-Conrady distortion
- **Intel RealSense D435i** via `pyrealsense2` — RGB-D capture, intrinsics
- **Ultralytics YOLOv8** — target-bin detection (Study 5), with a synthetic
  dataset generator rendering labelled RGB from PyBullet
- Pinhole ray–plane intersection geometry with sphere-radius correction
- ReportLab — exact-scale printable calibration-target PDFs

**Tooling**
- pytest (two independent regression suites), Matplotlib, imageio (video),
  Tkinter (joint-jog GUI), Git

---

## 3. System Architecture / Core Logic

### 3.1 The learning loop (`policy_learning/MC_PILCO.py::MC_PILOT`)

A three-module loop, run for N trials:

```
 real/sim throw  ──▶  GP model update  ──▶  policy optimization in-model  ──┐
        ▲                                                                   │
        └───────────────────────── new policy ──────────────────────────────┘
```

**(a) Model learning — `model_learning/Model_learning.py::Ballistic_Model_learning_RBF`**

Three *independent* sparse GPs, one per velocity-delta channel:

- GP input: `[x, y, z, vx, vy, vz]` (6-D ball state; target and control input
  are stripped, since flight dynamics do not depend on them)
- GP output: `[Δvx, Δvy, Δvz]`
- Trained by Adam on the marginal log-likelihood
- Propagation, per step:
  `v_{t+1} = v_t + Δv`,  `p_{t+1} = p_t + Ts·v_t + (Ts/2)·Δv`
- A semi-parametric variant subtracts gravity as an analytic mean function
  (retained as an ablation)

**(b) Policy — `policy_learning/Policy.py::Throwing_Policy`**

- RBF network: `Nb = 250` Gaussian basis functions over the target position
- Input `(Px, Py)`, or `(Px, Py, h)` for the height-generalized variant
- Output squashed to `[0, uM]` by `(uM/2)·(tanh(x)+1)`
- Emits a non-zero command **only at t = 0** (release); free flight is
  uncontrolled
- Lengthscales are trainable parameters, initialized at `0.15 × (lM − lm)`
- `Residual_Throwing_Policy` adds the closed-form no-drag ballistic speed
  (TossingBot-style residual) — kept as an ablation

**(c) Cost — `policy_learning/Cost_function.py::Throwing_Cost`**

`c = 1 − exp(−‖p_land − P‖² / lc²)`, evaluated **only at the terminal step**.
This single scalar drives the policy gradient.

**(d) Optimization**

`M = 400` particles are propagated through the learned GP, **each with its own
randomly sampled target**, so one gradient step generalizes across the whole
target domain. `Nopt = 1500` Adam steps per trial, with dropout, exponentially
smoothed cost monitoring, learning-rate reduction and policy re-initialization
on NaN.

**(e) Exploration & augmentation**

- **Stratified exploration** partitions `[0, uM]` into `Nexp` bands so initial
  trials guarantee GP coverage across the full speed range.
- **SO(2) data augmentation** (`Na` per real trial): free-flight ballistics are
  rotationally symmetric about gravity, so each real trajectory is rotated by a
  random yaw and added as an additional, physically valid training example — no
  extra robot interaction.

### 3.2 Physics backends (`simulation_class/`)

| Module | Role |
|---|---|
| `model.py` | Pure-NumPy ballistic integrator with quadratic drag (baseline / elevated / wind studies) |
| `model_pybullet.py` | PyBullet arm executes the throw; ball is a rigid body; drag applied as an explicit external force each step so results stay comparable to the NumPy sim. Owns landing detection (descending plane crossing + linear interpolation onto the plane) |
| `release_solver.py` | **Single shared release solver for simulation *and* hardware** |
| `wind_models.py` | Constant wind, exponential-decay gusts, Ornstein–Uhlenbeck turbulence |

Frame convention is explicit and enforced: world frame has the floor at z = 0
and the arm base raised by `base_height`; the base frame (used by hardware and
pose tables) has the base at 0 and the floor at `−base_height`. Landing planes
below the floor are rejected at construction time rather than silently returning
a resting position.

### 3.3 The release solver (`simulation_class/release_solver.py`)

The most accuracy-critical computation, deliberately extracted into one module
that both `PyBulletThrowingSystem._optimized_release` and
`run_hardware_throw.py::plan_throw_for_target` call — a regression test asserts
the two paths agree to `1e-12`.

Given a commanded release speed and a target, it produces
`(release_pos, q_release, q̇_release, v_release)`:

1. **Azimuth → posture table lookup** (nearest of 23 entries spanning ±33° in 3°
   steps). Base-rotating a single posture does *not* correctly aim this arm
   (`J(base+az) ≠ Rz·J(base)`), so a fresh posture is searched per azimuth.
2. **Turret-aiming correction.** The ball departs from the *release point*, not
   the base origin, and that point sits at a fixed polar offset that rotates
   with the posture. The aim heading solves
   `|T|·sin(φ − β) = −r_off·sin(α_off)`. Uncorrected, the measured geometry
   (`r_off = 0.54 m`, `α_off = −51.7°`) gives ~32° of aim error at 0.8 m.
3. **2π wrapping applied to the base joint only** — it is the sole continuous
   joint; the other six have hard mechanical limits and are in-range by
   construction.
4. **Direction-constrained velocity LP** (SciPy/HiGHS):
   `maximize s subject to J·q̇ = s·d, |q̇ᵢ| ≤ q̇ᵢᵐᵃˣ`, with roll/twist joints
   `(0, 2, 4, 6)` pinned to `q̇ = 0`. Only axis-perpendicular pitch joints
   `(1, 3, 5)` carry throw velocity; the roll joints' *static* angles remain
   free search parameters (pinning them to zero collapses the achievable
   velocity set to a line).

### 3.4 Robot abstraction (`robot_arm/`)

- **`robot_profiles.py`** — a frozen `RobotProfile` dataclass registry covering
  6 arms (`kuka_iiwa`, `franka_panda`, `franka_panda_dyn`, `kinova_gen3`,
  `kinova_gen3_dyn`, `xarm6`). Each carries URDF path, joint indices, EE link,
  neutral pose, `qd_max`, `tau_max`, PD gains, windup delta, speed bounds and
  **`control_mode`**: `kinematic` (ball velocity assigned — idealized),
  `position`, or `torque` (full computed-torque dynamics). Gen3 limits
  (1.3963/1.2218 rad/s, 39/39/39/39/9/9/9 Nm) were verified against the physical
  arm to float32.

- **`arm_controller.py`** — profile-driven planner and executor:
  - URDF load with automatic repair of massless links (PyBullet silently
    substitutes 1 kg per inertia-less link, which added 3 kg of phantom wrist
    mass to the shipped Gen3 URDF and inflated every computed torque ~2.3×)
  - IK, Jacobian, pseudo-inverse velocity solve
  - **3-phase piecewise cubic** trajectory with *proximal-to-distal
    (kinetic-chain) stagger*: joint *i*'s ramp begins at
    `stagger_frac[i]·dt_throw` and all joints arrive at release together, so
    distal joints stay cocked and fire late while each joint's own peak velocity
    remains exactly `q̇_release,i ≤ q̇ᵢᵐᵃˣ`
  - **Feasibility checking on all three phases**, sampled densely along the path
    (not just endpoints) for both torque and velocity, with automatic
    time-scaling retry (up to 6× growth) and a scanned follow-through duration
    ladder
  - Computed-torque `step()`: `τ = ID(q, q̇, q̈_cmd) + Jᵀ·m_payload·(a_ee − g)`,
    clipped to `tau_max`, where `q̈_cmd = q̈_ref + kp·e + kd·ė`

- **`noise_models.py`** — actuator/release noise: velocity bias, velocity slip,
  salt-and-pepper spikes, release-timing jitter, and a `TrackingErrorNoise`
  fitted from measured sim tracking residuals.

### 3.5 Throw-pose search (`find_throw_pose.py`)

An offline search producing the azimuth → posture table. Candidate postures are
filtered by a cascade of *real* physical tests, each with its own torque margin:

| Filter | Margin | What it rejects |
|---|---|---|
| Static gravity feasibility | `tau_max` | Postures the arm cannot even hold still |
| Direction-constrained LP | — | Postures that cannot aim at the required elevation |
| Windup kinematics | joint limits | Cock-back poses that leave joint range |
| Windup path dynamics | 0.90 | Torque-infeasible approach paths |
| Release-instant dynamics | 0.99 | Infeasible release states |
| Throw ramp | 0.75 | Infeasible acceleration ramps |
| Follow-through | 0.85 | Unbounded deceleration phases |

Plus a natural-posture gate (minimum forward reach 0.25 m, arm elevation
5–50°) and a 25°/step continuity cap between adjacent azimuth entries. Tables
are stamped with the `floor_z` they were searched against, and the trainer
refuses a stamp that disagrees with `--base_height`.

### 3.6 Hardware execution (`robot_arm/kinova_hardware.py`, `run_hardware_throw.py`)

The trajectory is planned by the **same** `ArmController` used in simulation;
only the executor is swapped ("step PyBullet" → "stream joint velocities to
Kortex"). Six independent safety layers:

1. **Dry-run by default** — no Kortex connection, no motion, and the module
   imports and runs with no arm and no `kortex_api` present (stub backend).
2. **Per-phase time scaling.** `speed_scale` (default 0.15) stretches the
   *throw phase only*; windup and follow-through run at `positioning_scale`.
   Geometry is preserved, wall-clock velocity is scaled down. Scaling all
   phases together was measurably worse: the base joint sweeps its full 178.7°
   azimuth during windup, contributes nothing to release speed, and uniform
   scaling drove it to 74.4 °/s and 0.5008 rad of position error at release.
3. **Hard per-joint velocity clamp** to measured `qd_max`. Never scales up.
4. **Whole-trajectory precheck (400 samples), fails closed** — soft joint
   envelope, inverse-dynamics torque against `tau_max` with a 0.90 margin,
   detection of any instant where clamping would activate (treated as a hard
   failure, because a clamped throw silently lands short), and a reported
   release-quantisation error term.
5. **Watchdog stop** — any exception, Ctrl-C or loop exit sends zero velocity
   and releases servoing in a `finally` block.
6. **Single-point Kortex isolation** — every API call lives in one backend
   class.

Additional hardware machinery:

- **Gripper-latency compensation.** The open command is fired early by
  `GRIPPER_RELEASE_LATENCY_S`; because the lead is wall-clock, it converts into
  trajectory time as `s_fire = t_r − lead·speed_scale` (multiplied, not
  divided).
- **Absolute-deadline pacing** at the documented 40 Hz high-level ceiling
  (`Base.SendJointSpeedsCommand` in `SINGLE_LEVEL_SERVOING`), with achieved-rate
  and worst-tick-lateness recorded as results.
- **Separate feedback channel for instrumentation.** Drift tracking reads joint
  state over the UDP real-time channel (~0.9 ms, measured 1122 Hz) rather than
  the TCP command channel (~25 ms, which halved the achieved control rate).
- **Joint-convention handling.** Kortex reports every joint on [0, 360),
  including limited ones; `read_joint_state()` wraps to (−π, π], and `home()`
  takes the shortest path for continuous joints `(0,2,4,6)` and the direct
  difference for limited ones `(1,3,5)`.
- **Soft-limit manager** with backup/restore to JSON.
- **Staged CLI** (`plan → connect → limits → home → gripper → throw`) where
  real motion requires `--arm` and a throw additionally requires `--confirm`.

### 3.7 Perception & calibration (`perception/`, `calibrate_*.py`)

- **`ray_plane.py`** — pixel → base-frame position by ray–plane intersection.
  Depth is deliberately *not* the position source (D435i stereo error is ~2% of
  range = 2–4 cm at 1–2 m, the same size as the landing error being measured);
  the landing surface is a plane of known height, so pixel + intrinsics +
  extrinsics determines position exactly. Includes the **sphere-radius
  correction**: intersecting against `z_plane + r` instead of `z_plane`, which
  removes a systematic `r·tan(θ)` offset (1.4 cm for a 1.9 m overhead mount at
  0.8 m off-axis).
- **`calibrate_camera_extrinsics.py`** — solves `T_B_C` from a live ChArUco
  frame via `solvePnP` composed with the board's measured base-frame pose;
  orientation is supplied as two explicit direction vectors rather than an
  implicit convention. Quality-gated on detected corner count and marker edge
  size.
- **`calibrate_via_wrist_camera.py`** — cross-check using the arm's own FK plus
  a marker seen by both cameras.
- **`make_aruco_targets.py` / `make_aruco_printable.py`** — exact-scale ArUco
  and ChArUco PDFs with a printed verification ruler (printer scaling silently
  corrupts marker size otherwise).

### 3.8 `throw_lab/` — trajectory-profile research package

A self-contained study of *how the arm should reach the release state*,
importing the shipped code read-only so every measurement runs through the same
controller and the same 50 Hz PyBullet world.

| Module | Role |
|---|---|
| `shapes.py` | Normalized acceleration/velocity shapes (const-accel, min-jerk quintic, trapezoidal-accel, plateau) with closed-form peak expressions |
| `feasibility.py` | Dense, payload-aware whole-phase torque/velocity/jerk/limit checking |
| `planner.py` | Shape-pluggable 4-phase planner: windup → throw → **brake** → return |
| `dynopt.py` | Per-joint time allocation ("the whip"), two objectives: min-time and closed-loop landing error |
| `harness.py` | Closed-loop executor + measurement (release velocity, landing, saturation) |
| `bench.py` / `sweep.py` | Profile comparison and throw-window duration sweep CLIs |

### 3.9 Repository structure — 7 independent forks

Each `mc-pilot*/` directory is a **self-contained fork** (its own `gpr_lib/`,
`simulation_class/`, `policy_learning/`, `model_learning/`), so studies can
diverge without cross-contamination.

| Directory | Study | Backend | Addition |
|---|---|---|---|
| `MC-PILCO/` | — | — | Upstream reference, unmodified |
| `mc-pilot/` | Baseline | NumPy ballistic | Ground targets, z_release = 0.5 m |
| `mc-pilot-elevated/` | 1 | NumPy | Elevated release (1.0/1.5/2.0 m), stratified exploration |
| `mc-pilot-pybullet/` | 2 + **active** | PyBullet | Arm physics, noise models, multi-arm, torque control, Gen3 hardware track |
| `mc-pilot-pb-elevated/` | 3 | PyBullet | Elevated targets + arm physics |
| `mc-pilot-wind/` | 4 | NumPy + wind | Constant / gust / OU turbulence; blind vs wind-aware GP |
| `mc-pilot-pybullet-yolo/` | 5 | PyBullet + OpenCV/YOLOv8 | HSV segmentation + YOLO bin detection feeding the target estimate |

---

## 4. Key Engineering Achievements

### Algorithms implemented from paper to working code

- Full MC-PILOT loop reproduced from the MC-PILCO reference: 3-GP ballistic
  model, RBF release-speed policy with tanh squashing, saturated terminal
  distance cost, Monte-Carlo policy gradient over 400 particles with per-particle
  target randomization.
- **Zero-new-trial task adaptation** (paper §6.4) implemented in
  `adapt_policy_height.py`: the GP models *ball* dynamics, which are independent
  of basket placement, so a new basket height is handled by re-running policy
  optimization alone — no robot interaction. Correctly updates both the target
  domain (measured reachable-band slope 0.382 m per m of height) and the particle
  control horizon (the rollout is a fixed-length loop with a terminal cost, so a
  stale horizon optimizes for a landing that happens after the ball passed the
  basket).
- **SO(2) trajectory augmentation** exploiting the rotational symmetry of
  free-flight ballistics to multiply training data without new trials.
- **Analytical baseline** (closed-form no-drag ballistic release speed)
  implemented as a first-class comparator, not an afterthought.

### Physical-throw generation on a torque-limited 7-DOF arm

- **Kinetic-chain throw synthesis**: proximal-to-distal joint stagger with a
  per-joint monotonic ramp, so every joint peaks exactly at its own
  `q̇_release ≤ q̇ᵐᵃˣ` and the whole-trajectory velocity guarantee survives the
  stagger.
- **Exhaustive hardware-valid pose search** for the Gen3: 69,505 candidate
  postures, 713,670 LP solves, 74,230 fully feasible trajectories, ~33 min of
  compute. Winning release state: **1.6281 m/s at 5° elevation**, 0.824 m
  landing range from a base-level floor (0.9355 m re-searched against the real
  0.433 m base plate). Same pipeline run on Franka Panda produced 3.27 m/s /
  1.649 m from an independent search, demonstrating the search is
  robot-agnostic.
- **Proved release-speed ceiling.** Release speed depends only on the release
  *configuration* and joint-velocity limits through the LP; a
  0.2 s → 2.4 s throw-window sweep at three azimuths held peak torque ratio at
  **0.18–0.49** while the joint-*velocity* ratio sat pinned at **0.92**. No
  trajectory shape raises the ceiling on a rigid arm — established by
  measurement rather than assumed.
- **Single-source release solver** shared by simulation and hardware, with a
  regression test asserting the two planners agree to `1e-12`.

### Quantified simulation results (all re-evaluated through real physics, never from training cost)

| Result | Measurement |
|---|---|
| Torque-mode (real dynamics) landing error | **1.50 ± 0.68 cm** — 75 throws, 5 seeds |
| Kinematic-mode (assigned velocity) landing error | 2.52 ± 0.65 cm — 75 throws, 5 seeds |
| Release-velocity tracking fidelity | **90.3 %** of commanded speed over 225 throws; mean release-position error **4.5 mm** |
| Object generalization (unseen ball) | **1.38–1.46 cm** across a 4×3 grid: mass 0.03–0.15 kg × radius 0.020–0.045 m |
| Height generalization (dynamic release) | **3.6 cm** mean over 40 targets spanning 0–0.3 m basket height |
| Noise robustness — velocity slip α = 0.10 | hit rate **8.3 % → 100 %**, mean error **13.1 cm → 3.8 cm** (naive → noise-aware, 3 seeds) |
| Noise robustness — salt-and-pepper p = 0.10 | hit rate **66.7 % → 91.7 %**, mean error **5.9 cm → 3.4 cm** (3 seeds) |
| Multi-arm comparison (achieved-velocity mode) | KUKA iiwa / xArm6 / Panda benchmarked on identical targets with joint-utilization and speed-ratio metrics |

Methodology enforced throughout: policy accuracy is *never* reported from the
in-model training cost (which is computed by simulating particles through the
learned GP and can sit near zero while true accuracy is off by tens of percent).
Every accuracy figure comes from re-running the policy through the physics
backend on fresh, previously unused RNG seeds.

### Real Kinova Gen3 bring-up

- **Full throw trajectory executed on the physical arm** as an open-loop
  joint-speed stream, escalated `speed_scale` 0.15 → 0.30 → 0.60 → 1.00, clean at
  every stage. Real ball throws executed at 0.15.
- **Open-loop drift quantified**: 0.0171 rad at release ≈ 1.0 cm of landing
  error, against a 2.89 cm simulation accuracy — measured by instrumenting the
  UDP feedback channel, which costs 3.6 % of the control period instead of the
  100 % the command channel would.
- **Gripper release latency characterized at 1 kHz over UDP**: 67.9 ± 6.4 ms
  static/unloaded (15 trials), re-validated at 73.2 ± 10.3 ms with a real ball
  loaded (15 trials) — statistically indistinguishable. Uncompensated this is
  10.2 cm of undershoot at 1.498 m/s, 3.5× the entire simulation accuracy; it is
  now compensated in the executor. The 6.4 ms scatter matches the 25 ms command
  quantum's predicted 7.2 ms std, locating the jitter in the command path rather
  than the hardware.
- **Control-rate claim corrected against the vendor's own driver docs**: the
  throw path is high-level (`SendJointSpeedsCommand`, 40 Hz ceiling, 25 ms
  command quantum), not the 1 kHz `LOW_LEVEL_SERVOING`/`BaseCyclic.Refresh` path.
  The 25 ms quantum is a ~3.7 cm landing-error floor that no faster loop can fix
  — a constraint now surfaced in every precheck report rather than buried.
- **Actuator limits verified against the physical arm** to float32 precision
  (`qd_max` 1.3963/1.2218 rad/s, `tau_max` 39/39/39/39/9/9/9 Nm, joint
  acceleration limit 5.20 rad/s²; the planned throw peaks at 34 % of it).
- **Tool-frame error budget quantified from firmware**:
  `ControlConfig.GetToolConfiguration()` reports `tool_transform = (0,0,0.12) m`
  and `tool_mass = 0.831 kg` for the Robotiq 2F-85. At the trained release state
  (|ω| = 3.25 rad/s), `v_true = v_flange + ω × r_offset` is **0.39 m/s — 26 % of
  release speed — and 12 cm of release position**, an order of magnitude larger
  than every other known error term combined. The same read confirmed the
  `GetMeasuredCartesianPose` Euler convention (intrinsic XYZ, degrees) to 1.23°.
- **Read-only-first tooling**: `hw_readonly_check.py` opens a Kortex session,
  reads, and closes with **zero writes** — run before any command path.

### Trajectory-profile study (`throw_lab`)

Benchmarking six jerk-limited alternatives against the shipped piecewise-cubic
throw, all planning to the *same* release state through the same closed loop:

- **`trap_accel(β = 0.25)` beats the shipped cubic by 27× on landing error**
  (2.16 cm → 0.08 cm), **8.4× on release-timing sensitivity**
  (3.37 → 0.40 cm/s of release-speed change per 20 ms shift of the release
  instant), at half the direction error and 0.56 s sooner to release.
- Proved analytically and by regression test that the shipped "piecewise cubic"
  throw phase is in fact a **constant-acceleration ramp** (with the monotonic
  windup's cock-back distance `Δq = q̇ₑ·T/2`, the cubic coefficient
  `a₃ = (q̇ₑT − 2Δq)/T³` vanishes identically) — torque-optimal for a fixed
  `(Δq, q̇ₑ, T)`, but the worst possible shape for release timing.
- **Time-optimal ("min-time whip") trajectory optimization is a documented
  negative result on this arm**: because torque never binds, the optimizer drives
  the throw window to its floor and the closed loop then *overshoots* — 111.4 %
  of planned release speed, 9.5 cm landing error, and a realized torque ratio of
  1.31 that the plan-time check never saw (the PD correction sits on top of the
  feedforward). Continuous-time feasibility is not sufficient for a sampled
  closed loop; the planner now carries a hard control-bandwidth floor derived
  from the loop's own 33 ms time constant.
- **Closed-loop landing-error optimization works**: 2.16 cm → 0.00 cm landing
  error and 4.51 s → 3.52 s time-to-release at 99.39 % speed fidelity, ~240
  harness evaluations per release state.
- Structural finding: when torque *is* the binding constraint, the optimal joint
  stagger is **not** monotone proximal-to-distal — the Gen3's 9 Nm wrists need
  the *longest* window against the shoulder's 39 Nm, inverting the classic
  "cock the wrist, fire it late" intuition, which applies to elastic/underactuated
  links rather than rigid ones with weak distal actuators.

### Engineering rigor

- **Two independent pytest regression suites, both green**:
  - `mc-pilot-pybullet/tests/` — **103 tests, 21 s**, no GPU. Covers torque
    control, mid-ramp torque, Coriolis-at-release, follow-through
    torque + velocity feasibility, monotonic windup, dynamic release,
    base-height geometry, pose search, data augmentation, residual ablations,
    ray-plane perception geometry, and sim/hardware planner equivalence.
  - `mc-pilot-pybullet/throw_lab/tests/` — **76 tests, 4.7 s**. Covers shape
    math, control-grid release snapping, the brake-phase velocity bound, and the
    bandwidth floor.
- Physical bugs are encoded as regressions, not fixed and forgotten (e.g.
  `test_brake_phase_never_exceeds_release_velocity`,
  `test_release_instant_lands_on_the_control_grid`,
  `test_min_feasible_duration_respects_the_bandwidth_floor`,
  `test_const_accel_reproduces_shipped_cubic_to_velocity`).
- **Dependency archaeology handled cleanly**: `kortex_api` 2.6.0.post3 pins
  protobuf 3.5.1, whose Python implementation calls `collections.MutableMapping`
  — removed in Python 3.10. A scoped ABC re-export shim confined to the single
  hardware module makes the whole vendor API importable, and all Kortex call
  groups are statically verified against the installed wheel by test.
- **URDF sanitation** (`urdf_fixup.py`): PyBullet substitutes `mass = 1 kg` for
  inertia-less links, which added 3 kg of phantom wrist mass to the shipped Gen3
  URDF and inflated every computed torque ~2.3×. Repaired at load time; motion
  is bit-identical, only torque changes.

---

## 5. Provenance

The 5 simulation studies were completed by AR525 Group 3 (Aarya Agarwal, Bhumika
Gupta, Rishang Yadav, Yajesh Chandra) between 2026-03-25 and 2026-04-29 and are
frozen. Work from commit `22d0f03` (2026-07-02) onward — the PyBullet real-arm
track, torque control, throw-pose search, hardware bring-up, perception and
calibration, `throw_lab`, and the test suites — is the current author's.

Venue target: ReScience C (simulation reproduction + ablations) as primary,
ICRA 2027 as a stretch contingent on real Gen3 results. Paper draft at
`paper_icra2027/draft.tex`.
