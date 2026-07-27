# Hardware Cold Start — Kinova Gen3 Overhead Throw + D415

_Written 2026-07-27, after Deepak's "let's plan hardware experiments" reply.
Everything below is verified against the code at commit `4a949fb` + working tree,
not quoted from the status emails. Where a doc and the code disagreed, the code won._

---

## 1. Key points since Email 2 (17 Jul)

Email 2 ended with: Gen3 in sim, kinematic release, ~9/10 hits at 2–3 cm, and the
realisation that the arm was cosmetic (ball velocity assigned via `resetBaseVelocity`).
Everything below happened after that.

**1. The release became physical.** Computed-torque control with gravity compensation
and an analytic Jacobian-transpose payload term. Release velocity is now *measured* at
separation from tracked arm motion. `control_mode="torque"` profiles (`kinova_gen3_dyn`,
`franka_panda_dyn`) are the real ones; `kinematic` profiles still assign velocity and
must never be presented as throws.

**2. Four root-caused systematic biases**, in order found: a test-timing artifact
(sampling state after `stepSimulation` but comparing to the pre-step setpoint); a
miscalibrated speed envelope (declared 1.0 m/s measured on-axis only, real off-axis
ceiling 0.61 m/s); an unreachable polar target wedge (fixed by `--flight_targets`,
sampling an annulus around the release point instead of polar-from-origin); and the
deepest — particles propagating from the *nominal* release position while reality
launched 4–5 cm elsewhere, causing a 17–28% speed overshoot on every target. Fixing all
four took kinematic accuracy to 0.34 cm and gave a derivable 1.54 cm for the dynamic
(hardware) configuration.

**3. The model-belief trap** — a methodology finding worth its own paragraph in the
paper. `cost_trial_list` is computed by propagating particles through the *learned GP*,
never against real physics. It sat near zero while true accuracy was off by 17–28%. All
accuracy claims now come from `PyBulletThrowingSystem.rollout` with fresh unused seeds.

**4. Residual physics is a negative result at both levels.** Policy-level (TossingBot
style, speed = analytical + learned δ) is *worse* — δ is optimised through the biased
model so it inherits the bias. Model-level (gravity as a GP mean function) is within
noise. Ship plain MC-PILOT; the flags stay only for the ablation figure.

**5. The drag crossover — the publishable sim finding.** Low drag (tennis ball): the
paper's own analytical Eq. 13 *beats* MC-PILOT on 3/3 arms. High drag (whiffle,
~19% g): MC-PILOT wins ~7× on Panda, ~3.4× on KUKA. Learning matters exactly where the
analytical model fails, and we can show both sides of the crossing.

**6. The throw pose was wrong twice, and both corrections are real physics.** First: the
"1 m throw" got its speed from base rotation — a horizontal spin that cannot be aimed. A
direction-constrained LP fixed the aiming. Second: that LP still let the roll/twist
joints spin, producing a corkscrew. Only axis-perpendicular (pitch) joints carry throw
velocity; on Gen3 and Panda the roll joints are indices (0,2,4,6) and are frozen at
q̇ = 0 during the throw, free only as static setup angles.

**7. The overhead throw.** Rebuilt the search to optimise the *release state* directly
(joint angles + velocities at release, under real 39/9 Nm and 1.396/1.222 rad/s limits)
instead of scoring poses by distance. This is what the shipped table contains — verified
by FK just now, not quoted:

| Quantity | Value (all 23 azimuth entries) |
|---|---|
| Release position | r = **0.035 m** from base axis, **z = 1.137 m** — essentially straight overhead |
| Release speed | **1.628 m/s** |
| Launch elevation | **5.0°** |
| Ballistic range | **0.800 m** |
| Azimuth coverage | **−33° … +33°**, 23 entries, uniform by construction |
| Joint velocities | elbow **−1.396** and wrist **−1.222** rad/s — both saturated at limit; roll joints exactly 0 |

The low elevation is not a bug: at these speeds range *decreases* monotonically from 0°
to 70°, so the height (1.137 m), not the launch angle, is what buys the distance.

**8. Whole-trajectory safety.** Follow-through had *no* feasibility check and was
silently commanding 3.2× torque / 1.9× joint-speed limits after release. Found three
times over (an infeasible plan; the arm free-falling because the rollout stopped
commanding it post-release; then a torque-only check that missed a velocity violation).
All three phases — windup, throw, follow-through — are now sampled along the whole path,
and `plan_throw` raises rather than shipping an infeasible trajectory.

**9. Honest ceiling.** With follow-through genuinely enforced, safe landing caps at
**0.83 m — inside the arm's own 0.86 m reach.** Quantified range-vs-recoverability
trade-off. Report it as a finding; do not let it get rounded into "throws beyond reach."

**10. Zero-new-trials height adaptation (paper Sec 6.4), reproduced.** Reused the
trained GP verbatim and re-optimised the policy alone: 2.95 / 3.41 / 3.80 cm at
h = 0.10 / 0.20 / 0.30 m, 100% hit < 10 cm, **zero additional throws** — matching the
ground baseline (3.15 cm) and beating a full 9-D retrain (3.63 cm).

**11. Also worth knowing:** a bigger windup cannot increase range (release speed is
evaluated purely at the release configuration; the ceiling is a joint-*velocity* limit).
GPU is slower than CPU for this workload. The rigid-arm speed ceiling is provable from
the joint-velocity LP — only a compliant/elastic DOF beats it.

**Caveat to state before anyone quotes it:** the headline 3.15 cm is **single-seed**
(`results_kinetic_chain_gen3/1` only), as are all three height-adaptation numbers. There
is no ± yet.

---

## 2. Codebase audit — layer by layer

Verified this session: **54/54 tests pass in 7.29 s** (`python3 -m pytest tests/ -q`).

### Layer map and the one seam that matters

```
policy (RBF, target -> speed)
        |
        v
_speed_to_velocity(speed, release_pos, target)      generic 35 deg ballistic vector
        |
        v
_optimized_release(arm, v_cmd)                      <-- OVERHEAD THROW LIVES HERE
   turret-corrected azimuth -> table lookup -> base rotation
   -> rotate stored v_dir -> scale stored qd -> FK release_pos
        |
        v
arm.plan_throw(v_cmd, release_pos, t_w, t_r, T,
               q_release_override=..., qd_release_override=...,
               monotonic_windup=True)   -> coeffs {windup, throw, follow, stagger_frac}
        |
        v
arm.get_setpoint(coeffs, t)  ---->  sim: arm.step()      (PyBullet torque)
                             \--->  hw:  backend.send_joint_velocities()  (Kortex)
```

`get_setpoint(coeffs, t)` is the single sim/hardware seam, and it is a **good** one:
pure function of the cubic coefficients and time, no PyBullet dependency, evaluated at
arbitrary t (so a 1 kHz consumer works without resampling). `kinova_hardware.py` already
consumes it correctly.

### Per-layer verdict

| Layer | File | Verdict |
|---|---|---|
| Robot data | `robot_arm/robot_profiles.py` | **Solid.** Frozen dataclass, one dict, 6 profiles. `control_mode` is the load-bearing field. Panda `kp`/`kd` still copied from Gen3 untuned. |
| Arm control | `robot_arm/arm_controller.py` | **Solid.** All 3 phases feasibility-checked along the whole path. 9-DOF zero-padding verified working for `franka_panda_dyn` (constructs + steps; guard now only rejects non-contiguous `dof_ids`, which is correct). *The 2026-07-22 handoff's "Panda cannot train" note is stale — I tested it.* |
| Release logic | `simulation_class/model_pybullet.py::_optimized_release` | **Correct but trapped.** ~100 lines of the project's most important logic, reachable only as a method of `PyBulletThrowingSystem`. This is the integration blocker (§3). |
| Pose search | `find_throw_pose.py` | **Solid and robot-generic** (`--robot`). Real `calculateInverseDynamics` checks, not proxies. |
| Learning | `policy_learning/`, `model_learning/` | **Solid**, unchanged upstream shape. Residual variants present but negative — ablation only. |
| Evaluation | `eval_baseline.py`, `eval_heightgen.py`, `adapt_policy_height.py` | **Solid.** All go through real rollout. Discipline of fresh seeds is followed. |
| Hardware | `robot_arm/kinova_hardware.py` | **Structurally good, three real gaps** (§3). Safety model is genuinely defence-in-depth. |
| Hardware CLI | `run_hardware_throw.py` | **Plans the wrong throw** (§3, blocker H1). |
| Vision | `mc-pilot-pybullet-yolo/robot_arm/depth_camera.py` | Works in sim; **its depth back-projection method does not transfer to a real D415** (§5). |

### The four defects, ranked

**H1 — BLOCKER: hardware plans the legacy throw, not the overhead throw.**
`run_hardware_throw.py:103`:

```python
coeffs, q_release, qd_release, v_ach = arm.plan_throw(v_cmd, rel, t_w=t_w, t_r=t_r, T=T)
```

No `q_release_override`, no `qd_release_override`, no `monotonic_windup`, no table. It
aims with a hardcoded `launch_angle_deg=35.0` from `cfg["release_pos"]` and lets IK +
`pinv` choose the release state. That is the near-horizontal min-norm throw the whole
overhead argument replaced. **Everything in Email 3 — the release-state search, the
1.137 m release, the 5× gain — is unreachable from the hardware path today.**

Fix, and it must be a *refactor not a copy*: lift `_optimized_release` out of
`PyBulletThrowingSystem` into a free function (it already takes `arm` as an argument;
its only other state is `_opt_table`, `_opt_polar`, `_cur_target_xy`, `launch_angle`).
Sim and hardware then call the identical function. Duplicating those 100 lines into the
hardware script guarantees the two paths drift apart, and this is precisely the logic
where every past bug has hidden.

**H2 — precheck is position-only.** `HardwareThrowExecutor.precheck` samples the joint
envelope (±6.10 rad) and computes peak velocity, but performs **no torque check** and
**never fails on clamping**. `clamp_velocity` only clamps down — safe for the hardware,
silently wrong for the throw: at `speed_scale=1.0` a clamp changes release speed and the
ball misses with no warning. Must (a) fail closed if any clamp is active, (b) re-run the
same whole-trajectory torque check `plan_throw` already implements.

**H3 — rate mismatch, both directions.** `SafetyLimits.control_hz = 100.0` (hardcoded
again in `make_limits`), streamed with `time.sleep` in Python. Gen3 low-level is 1 kHz.
Separately, **sim runs physics *and* control at 50 Hz** (`nsub = 1`, `dt_phys = dt =
Ts = 0.02`) — this is the origin of the 1.54 cm dynamic noise floor, and it means real
1 kHz hardware should track *better* than sim, not worse. Release timing quantisation
goes 20 ms → 1 ms. At 1.63 m/s, 20 ms = 3.3 cm, so this matters.

**H4 — open-loop velocity streaming.** The backend sends joint *speeds* with no position
feedback correction, while sim closes the loop with computed torque (kp = 400,
kd = 60). Positional drift over a ~1.2 s trajectory is unmeasured. This is exactly what
Phase C's tracking gate exists to catch; if drift is significant, add an outer P-term on
position error to the streamed velocity.

**Not defects, but flagged:** `kortex_api` is not installed on this machine
(`ModuleNotFoundError`), and the Kortex method names in `_KortexBackend` are unverified
against any real release.

---

## 3. Phase 0 — code work before anyone touches the arm

Gate: nothing in Phases A+ starts until these are done and tests are green.

**P0.1–P0.5 are DONE (2026-07-27).** Suite is **65 passing** (was 54).

- [x] **P0.1** Extracted `_optimized_release` → `simulation_class/release_solver.py::OptimizedReleaseSolver`.
  `PyBulletThrowingSystem` delegates; the duplicate body is deleted, not left dead.
  **Regression gate met: 30-throw eval reproduces bit-for-bit** (max abs diff 0.000e+00
  across every landing / error / speed field; mean 2.69 cm, seed 2024).
- [x] **P0.2** `run_hardware_throw.py::plan_throw_for_target` now calls the shared solver
  with the pose table and passes `q_release_override` / `qd_release_override` /
  `monotonic_windup`. Verified equal to the sim planner to **1e-12** on 5 targets
  across the wedge. Two further bugs found and fixed while doing it:
  - phase timings came from `profile.timing` (0.4/0.8) instead of the trained
    `cfg["T_W"]/["T_R"]` (0.5/1.6) — would have roughly **doubled** commanded peak
    joint velocity;
  - the hardcoded release box (z ≤ 0.9 m) **excluded the overhead release** at
    z ≈ 1.137 m, so `throw` would have refused every valid plan. The box is now
    derived from the pose table's own FK release locus, which also makes the check
    meaningful rather than arbitrary.
- [x] **P0.3** `precheck` now fails closed on any active velocity clamp (a clamp keeps
  the arm safe but silently slows the throw and lands short with nothing in the logs)
  and runs whole-trajectory inverse dynamics against `tau_max`. Reuses a new public
  `ArmController.inverse_dynamics()`, which `_throw_peak_torque_ratio` also calls — one
  torque path, not two. Real plan reports **peak 8.1 / 39.0 Nm (21%)**.
- [x] **P0.4** `control_hz` → 1000. `time.sleep(dt)` pacing replaced with an
  absolute-deadline loop (sleep-plus-body-time drift accumulates over thousands of
  ticks). Achieved rate, worst tick lateness and release wall-time are recorded in
  `last_exec_stats` and warned on if >10% below target. Dry run holds
  **1000 Hz over 56,682 ticks, worst tick 0.16 ms late.**
- [x] **P0.5** `tests/test_hardware_planner.py` — 11 tests: sim-vs-hardware planner
  equality across the wedge, "is actually the overhead throw" (release height, roll
  joints frozen), trained-timings-not-profile-defaults, precheck passes on the real
  plan, precheck fails on clamping, precheck fails on torque violation, derived release
  box contains the release and still rejects a displaced one.
- [ ] **P0.6** Install `kortex_api`, confirm every method name in `_KortexBackend`
  against that exact version. Fix in that one class. **Blocked: wheel not available here.**
- [ ] **P0.7** Multi-seed the sim result (seeds 2, 3) so there is a ± to quote.

Also fixed in passing: `eval_adapted_height.py` gained `--opt_pose`, so checkpoints
written before the trainer recorded `opt_pose` in `config_log.pkl` (including
`results_kinetic_chain_gen3/1`) can still be evaluated. Newly trained checkpoints record
it automatically.

---

## 4. Frame conventions — fix these before any calibration

Everything in this project is already expressed in one frame. Write it down once and
never re-derive it.

### B — base frame (= sim world frame)

Sim mounts the arm with `basePosition=(0,0,0)`, `useFixedBase=True`,
`setGravity(0,0,-9.81)`. Therefore:

- **Origin**: centre of the Gen3 mounting flange, on the table/pedestal surface.
- **+X**: the azimuth-0 throw direction (downrange).
- **+Z**: vertically up, opposite gravity.
- **+Y**: Z × X (to the left when standing behind the arm looking downrange).
- Units metres, angles radians, right-handed.

Everything the policy touches lives in B: targets `(Px, Py)`, `target_height`, release
position, landing points. **Physically mark +X on the table with tape at setup.** Every
downstream number depends on that one arrow being right.

Verified geometry in B for the shipped table: release at
`(≈0.03, ≈0.00, 1.137)`, ball flies at 5° elevation, lands on the
annulus sector **r ∈ [0.60, 0.83] m, θ ∈ [−33°, +33°]**. Widest lateral extent
= 2 · 0.83 · sin 33° = **0.90 m**.

### C — D415 optical frame

librealsense/ROS optical convention: **+Z out of the lens, +X right in the image,
+Y down.** Depth and colour have *separate* optical frames — always `rs2::align` depth
to colour and then use the **colour** intrinsics only. Never mix.

### T_B_C — the one unknown

4×4 rigid transform, camera pose expressed in B. `p_B = R · p_C + t`. Solved in Phase D.
Store it as a YAML/JSON next to the checkpoints, with the date and the residual it was
solved to. Re-solve if the camera is bumped — and assume it *will* be bumped.

### Do not use the arm's own camera

The Gen3 URDF carries `camera_link`, `camera_color_frame`, `camera_depth_frame` (the
Vision module in the wrist — the loader warns about them on every run). It moves with
the arm, so its extrinsic changes every pose, and during a throw it is moving at
1.6 m/s. Bin detection uses the **external fixed D415 only**.

---

## 5. Camera plan — D415

### Choose the mount for lateral accuracy, not depth

**D415 depth error is ~2% of range: 2–4 cm at 1–2 m.** Our landing error is 3.15 cm.
Raw stereo depth is therefore *the same size as the thing we are trying to measure* and
cannot be the primary sensor. The sim `depth_camera.py` back-projects depth because sim
depth is exact; **that method does not transfer.**

Use **ray–plane intersection** instead: the bin rim and the floor are planes of known
height in B, so a pixel plus a calibrated `T_B_C` and colour intrinsics gives a
lateral position at millimetre scale, with no dependence on stereo depth.

```
d_C = normalize(K⁻¹ · [u, v, 1])          # ray in optical frame
d_B = R · d_C ;  o_B = t                  # ray in base frame
λ   = (z_plane − o_B.z) / d_B.z           # intersect plane z = z_plane
p_B = o_B + λ · d_B
```

Depth is still useful — as a segmentation gate and a sanity check that the detection is
at roughly the right range — just never as the position source.

Second consequence: **the D415 is a rolling-shutter camera.** It is fine for a static
bin and a settled ball; it will smear a ball moving at 1.6 m/s. Do not plan on
tracking the ball in flight with it. Measure the **landing position after the ball
settles**, which is all the MC-PILOT loop needs anyway.

### Placement

Constraints: cover a 0.90 m wide landing zone; stay out of the ±33° wedge and out of the
ball's path (which stays below z = 1.14 m); avoid the arm occluding the bin; respect
D415 min-Z ≈ 0.45 m; stay close enough for good angular resolution.

Field of view needed: to see 0.90 m with the 65° horizontal depth FOV, distance
≥ (0.45 / tan 32.5°) = **0.71 m**. So ~1.2–1.4 m is comfortable with margin.

**Recommended: side-oblique mount.**

| Parameter | Value (base frame B) |
|---|---|
| Camera position | **(0.75, −1.05, 0.90) m** |
| Aim point | landing-zone centroid **(0.72, 0.00, 0.00)** |
| Optical axis | (−0.03, +1.05, −0.90), normalised — yaw toward +Y, **pitch down ≈ 40.6°** |
| Distance to centroid | **1.38 m** |
| Coverage at that range | ≈ 1.76 m wide × 1.00 m tall (65° × 40°) — the 0.90 m zone fits with ~2× margin |
| Resolution | 1280 px over 1.76 m ≈ **1.4 mm/px** |
| Clearance from ball path | ball's max lateral excursion is 0.83 · sin 33° = 0.45 m; camera sits at |y| = 1.05 m → **0.60 m clear** |
| Height vs ball | ball never exceeds 1.14 m; camera at 0.90 m is below it and outside the wedge |

Mount rigidly — a tripod with a sandbag or, better, bolted to the bench frame. Any flex
invalidates `T_B_C`, and re-calibration costs 20 minutes.

If the bench cannot take a side mount, the fallback is a true overhead mount at
`(0.72, 0.00, 2.20)` looking straight down. Geometry is cleanest (the landing plane is
fronto-parallel, so ray–plane is exact and perspective error vanishes), but it needs
rigging above the arm and puts hardware over the workspace — only do it if the frame is
solid.

### Streaming config

- Colour **1280×720 @ 30 fps**, depth 1280×720 @ 30, `rs2::align` depth→colour.
- Disable auto-exposure once lighting is fixed; lock white balance. Auto-exposure hunting
  between throws changes the HSV thresholds under you.
- Record the factory intrinsics from `rs2_get_video_stream_intrinsics` **and** verify
  with a checkerboard — factory values are usually good but confirm before trusting.

---

## 6. The cold start — staged, gated

**Every stage has a pass gate. A stage that does not pass does not advance.** If a real
reading contradicts sim, stop and measure — the gap goes in the paper, it does not get
tuned away silently.

### Phase A — bench prep, arm unpowered (~1 h)

- [ ] A1 Mount the arm rigidly. Confirm the base is level (spirit level on the flange).
- [ ] A2 **Tape the +X axis** on the table, and mark the ±33° wedge edges out to 0.9 m.
- [ ] A3 Measure and mark the landing annulus: arcs at r = 0.60 and r = 0.83 m.
- [ ] A4 Clearance survey: ≥ 1.35 m of free space above the base (release at 1.137 m
  plus ball plus margin); nothing inside the wedge out to 1.0 m; no people downrange.
- [ ] A5 E-stop physically in reach of whoever runs the laptop. Second person present.
- [ ] A6 Soft floor / catch mat in the landing zone; ball is foam or ping-pong.

**Gate A:** a photo of the marked workspace, and the e-stop tested (press it, confirm it
latches) before power-on.

### Phase B — power and comms, zero motion (~1 h)

- [ ] B1 Power on, Ethernet up, confirm arm IP; web interface reachable.
- [ ] B2 `python3 run_hardware_throw.py connect --arm --ip <ip>` → reads joint state.
- [ ] B3 Compare the arm's reported joint angles against Kortex's web UI — units and
  sign convention must agree with our profile (**a sign flip here silently mirrors the
  whole throw**).
- [ ] B4 Dry-run stage 0 with the *new* planner: `plan --log_path <ckpt> --target 0.72 0.0`.
  Confirm the printed release pose matches the table (z ≈ 1.137 m, speed ≈ 1.63 m/s,
  5° elevation) — if it prints a 35° low throw, P0.2 was not done.

**Gate B:** joint angles agree with the pendant to < 0.01 rad; `PRECHECK: PASS`; printed
release geometry matches the table.

### Phase C — motion, no ball (~2 h)

- [ ] C1 `home` — gentle capped move to `q_neutral`. Watch it. Hand on e-stop.
- [ ] C2 Full trajectory at `--speed_scale 0.25`. Log commanded vs measured joint
  positions every tick.
- [ ] C3 **Tracking gate:** max |q_cmd − q_meas| < 0.05 rad at every joint, every phase.
- [ ] C4 **Torque gate:** log Kortex torque feedback; compare against the sim's
  `calculateInverseDynamics` prediction for the same trajectory. Expect < 80% of
  39/9 Nm. Plot both — this plot is a paper figure.
- [ ] C5 Escalate 0.25 → 0.50 → 0.75 → 1.00, re-checking C3/C4 at each. **Abort at
  > 90% of any torque limit** and re-derive the table ceiling at lower uM.
- [ ] C6 Confirm the follow-through actually decelerates the arm — this is the phase
  that was shipping 3.2× torque violations in sim. Watch it, do not just read numbers.

**Gate C:** clean 1.0× run, no ball, tracking and torque within gates, follow-through
visually smooth.

### Phase D — camera install and calibration (~2 h)

- [ ] D1 Mount the D415 per §5. Rigid. Do not move it again.
- [ ] D2 Stream check, lock exposure and white balance, record intrinsics.
- [ ] D3 **Solve `T_B_C` using the arm as the calibration rig:**
  1. Grip the ball (or a high-contrast marker) in the gripper.
  2. Command **≥ 12 static poses** spread over the workspace and depth range, all inside
     the camera FOV. Use slow moves; let the arm settle before capturing.
  3. Per pose record `p_B` from Kortex tool pose (or FK on the measured joints), and the
     marker centroid in the colour image → `p_C` (depth median over the marker mask is
     acceptable *here*, because Kabsch averages the noise over 12 points).
  4. Solve with Kabsch/Umeyama for `R, t` minimising `‖p_B − (R p_C + t)‖`.
- [ ] D4 **Calibration gate: RMS residual < 5 mm.** Between 5 and 10 mm, add poses and
  re-solve. Over 10 mm, something is wrong — check for a mirrored axis before adding data.
- [ ] D5 **Independent validation:** tape 5 known points in the landing zone, place the
  ball on each, run ray–plane, compare against the tape measure. **Gate: < 1 cm each.**
- [ ] D6 Save `T_B_C` + intrinsics + date + residual to a versioned file.

**Gate D:** D4 and D5 both pass. Until then there is no measurement system, and without a
measurement system there are no results.

### Phase E — gripper and release-delay calibration (~1.5 h)

The dominant real-world error term. The arm decelerates while the gripper opens, so
release happens *late* and every throw undershoots systematically.

- [ ] E1 Gripper open/close test, `gripper --arm --close` / `--open`.
- [ ] E2 Measure the delay: command release at a known trajectory time, film at **240 fps**
  (a phone is sufficient), count frames from command to actual ball separation. **5 repeats.**
- [ ] E3 Report mean and spread. **At 1.63 m/s, 20 ms = 3.3 cm** — the same size as our
  entire error budget. This measurement is not optional.
- [ ] E4 Compensate: advance `release_step` by the measured mean delay. The spread (not
  the mean) becomes the honest irreducible noise floor, and it is what
  `noise_models.ReleaseTimingJitter` was written for but never applied to Gen3.
- [ ] E5 Re-verify after compensation with 3 more filmed throws.

**Gate E:** mean delay known to ±5 ms, compensation applied, spread documented.

### Phase F — pipeline integration, still no throw (~1 h)

Prove the full loop end to end before adding ballistics.

- [ ] F1 Place the bin at a measured position. Detect it with the camera → `(Px, Py)` in B.
  **Gate: camera estimate vs tape measure < 1 cm.**
- [ ] F2 Feed that into the policy → release speed. Sanity: should land in 1.16–1.49 m/s
  for targets in range. A saturated or out-of-range speed means the frame convention or
  the target is wrong.
- [ ] F3 Plan the trajectory → precheck → **confirm it passes with no clamping** (H2).
- [ ] F4 Execute at `speed_scale=0.25` with the gripper held **closed** — validates
  payload compensation on real dynamics with no projectile.
- [ ] F5 Move the bin, repeat 3×. The detected target must track the bin.

**Gate F:** the loop runs bin → detection → policy → plan → precheck → motion, three
times, no manual intervention.

### Phase G — first throws (~2 h)

- [ ] G1 First throw at `speed_scale=0.5`, ball in gripper, bin removed, everyone clear.
  Expect a short landing (velocity scales down); **verify the ball leaves the hand near
  the top of the arc**, i.e. that this is an overhead release and not a drop.
- [ ] G2 Landing detection: background-subtract before/after frames, ray–plane the ball
  centroid to the floor plane. Compare with tape. **Gate: < 1 cm.**
- [ ] G3 `speed_scale=1.0`, 3 throws at a fixed target. Record measured landing.
- [ ] G4 **Sim-vs-real gate: if |real − sim predicted| > 15 cm, STOP and diagnose.** The
  likely causes, in order: release-delay miscalibration (Phase E), a frame/sign error
  (Phase B3/D5), real torque or velocity saturation (Phase C4).
- [ ] G5 Film everything. Frame-check the video before claiming anything works — numeric
  checks alone have missed a corkscrew throw, a zero-amplitude windup, and an arm that
  never moved.

**Gate G:** 3 consecutive 1.0× throws, landing measured, within 15 cm of prediction.

### Phase H — MC-PILOT on hardware (~1 day)

- [ ] H1 ~10 calibration throws at fixed speeds (u ≈ 1.2 / 1.35 / 1.5, a few each),
  landing measured for each. **This is the paper's real exploration data** — feed the
  real `(u, landing)` pairs into the GP exactly as sim trials are fed. Use stratified
  bands, not random speeds.
- [ ] H2 Train GP + policy on the real trials. Consider the paper's real-hardware setting
  (`Nexp=10, Na=2`) rather than the sim one (`Nexp=5, Na=0`) — Gen3 *is* the hardware target.
- [ ] H3 Evaluate on 10–15 held-out targets across the wedge, real bin. **This is the
  headline number.** Hit radius < 10 cm is the paper's own bar.
- [ ] H4 Height adaptation on hardware: re-optimise the policy offline for a new bin
  height, **zero new throws**, then evaluate. This is the cleanest reproduction claim we
  have and it costs almost nothing once H3 works.
- [ ] H5 Record per-throw `(target, u, measured landing, Kortex torques)` for every run.
  Sim-vs-real gap table + the torque comparison plot are the paper's hardware section.

---

## 7. Standing rules (carried from every prior session)

- Velocity claims: **measured at separation**, never commanded.
- Any "it works": verify **frames/video and numbers**, both.
- Feasibility checks: **whole trajectory, all three phases**, never endpoint-only.
- Landing distance: `hypot(actual_landing_xy)`, never release-radius + range.
- Never quote `cost_trial_list` as accuracy.
- Label kinematic vs dynamic explicitly, every single time.
- If real contradicts sim: **measure it and publish the gap.**

---

## 8. Time estimate

| Phase | Effort | Needs arm | Needs camera |
|---|---|---|---|
| 0 — code | 1 day | no | no |
| A — bench | 1 h | no (unpowered) | no |
| B — comms | 1 h | yes | no |
| C — motion | 2 h | yes | no |
| D — camera cal | 2 h | yes | yes |
| E — gripper delay | 1.5 h | yes | no (phone) |
| F — integration | 1 h | yes | yes |
| G — first throws | 2 h | yes | yes |
| H — MC-PILOT | 1 day | yes | yes |

**≈ 2 days of lab time plus 1 day of code**, assuming no surprises. Phase 0 and Phase A
can happen before any arm booking. Phase E is the one most likely to overrun.
