# Kinetic-Chain Throw Pose — Design

_Date: 2026-07-21 · Study: Kinova Gen3 sim, `mc-pilot-pybullet/` · Status: approved_

## Motivation — a corrected root cause

The `--opt_pose` aimed-throw training path was documented in `HANDOFF.md` §6.3c
as an unsolved "release 4–30 m/s divergence" caused by "torque-PD instability at
dt=0.02" and an "un-isolated residual difference between the standalone loop and
`_simulate_pybullet`," concluded to be architectural.

That diagnosis is wrong. This session isolated it (scratchpad repro matrix,
2026-07-21):

- The runaway is **dt-independent** (identical at dt=0.005 and dt=0.02).
- Root cause: `_optimized_release` teleports the arm to the release pose to read
  its FK, then **never resets it to neutral**. The throw trajectory starts at
  neutral (windup), so step 0 begins with a ~3 rad position error → `kp=400` →
  torque saturates → joints run away to PyBullet's 100 rad/s clamp → ball ejected
  at a chaotic 0.7–60 m/s. The standalone scripts reset to neutral after their FK
  (`make_aimed_throw_video.py:57`); `_simulate_pybullet` did not.
- **Fix applied** (`model_pybullet.py`, `_optimized_release`): reset joints to
  neutral after the FK query. Verified in the real `rollout`: release speed now
  matches command, aims within ~2° across ±39°.

But even fixed, the committed `throw_pose.npy` is **not hardware-executable**: it
commands ~3× the real `qd_max` (1.22 rad/s) mid-throw, and its LP solution
saturates the base joint. Grounding shows the base contributes **0.0%** to
release speed (2.052 vs 2.062 m/s with/without it) — the contortion bought
nothing. This spec rebuilds the pose on the correct throwing principle.

## Principle (the goal)

- **Base joint (j1, vertical z-axis) = azimuth only.** Its static angle sets which
  vertical plane the throw lies in; it is held **still during the throw**:
  `qd[0] = 0`. It is not a speed source.
- **Shoulder/elbow/wrist sweep the vertical plane**, producing a release velocity
  vector pointing exactly along the launch direction `d` (aimable).
- **Arm extended** (long lever, `v = ω·r`).
- **Every joint velocity ≤ `qd_max` over the ENTIRE windup→throw motion**, not
  just at the release instant — the property that makes it real-hardware-valid.

Grounding (base frozen, `scratchpad/ground_nobase.py`): best aimed throw
**2.05 m/s / 105 cm** at elev 25°, all release joint velocities within `qd_max`.
So the hardware-valid throw is ~3× the current 35 cm — a strict win.

### CORRECTION (2026-07-21, during execution): azimuth needs a posture TABLE

A single posture optimized at azimuth 0 does **not** transfer to other azimuths by
rotating the base. Rotating joint 1 does not cleanly rotate this arm's throw
geometry (`J_rot ≠ Rz·J_0`; verified — `ee(q_rot)` y-sign flips vs `Rz·ee(q0)`), so
freezing the base collapses off-axis aimed speed: **2.14 m/s at 0° → 0.24 at 10° →
0.10 at 27°**. The MC-PILOT training wedge is **±30°** (`--gM_deg` default 30), so
this breaks the whole domain.

Fix (verified): search a fresh frozen-base posture **per azimuth** — recovers
**1.90 m/s at 15°, 1.55 m/s at 30°**. The pose search therefore produces an
**azimuth→posture table** (`throw_pose_table.npy`), and `_optimized_release` looks
up the nearest-azimuth posture for the target and re-solves the frozen LP for the
exact launch direction. Tasks 2 (base-freeze LP) and 3 (monotonic windup) are
unaffected — they were already committed and correct.

## Scope

- **Done when:** a new hardware-valid `throw_pose.npy` exists; the real `rollout`
  runs without blowup, aims within ~2° across ±39°, and **max |qd| over the whole
  trajectory ≤ qd_max**; distance reported vs the old 35 cm.
- **Not in scope:** retraining a full GP+policy on the new pose (a separate later
  step); other robot profiles; hardware driver work.

## The monotonic kinetic-chain sweep (math)

The release-instant velocities are ≤ `qd_max` by LP construction. The mid-stroke
overshoot is eliminated by cocking the windup **backward along the throw axis** by
exactly half the ballistic:

    q_windup = q_release − qd_release · (t_throw / 2)

Then the rest-to-velocity cubic from `q_windup` (rest) to `q_release` (velocity
`qd_release`) over `t_throw` reduces to a **linear velocity ramp**
`qd(t) = qd_release · t/t_throw` — constant acceleration, monotonic, peak =
`qd_release ≤ qd_max`. No overshoot by construction (mimics windup-then-whip).

Coupling to handle: `plan_throw`'s torque-feasibility loop stretches `t_throw`. A
linear ramp has constant accel `qd_release/t_throw`, so a larger `t_throw` lowers
torque **and** grows the windup cock `dq = qd_release·t_throw/2`. Therefore
`q_windup` must be **recomputed whenever `t_throw` is stretched**. The
windup phase (neutral→windup) peak velocity `1.5·|q_windup − q_neutral|/t_windup`
must also stay ≤ `qd_max`; size `t_windup` to satisfy it (recompute on stretch).

## Components (all in `mc-pilot-pybullet/`)

### 1. Pose search — rewrite `find_throw_pose.py` (azimuth TABLE)

- Aimed LP with the base pinned: `max s s.t. J·qd = s·d, qd[0]=0, |qd_i|≤qd_max`.
  (`freeze_base` bound `(0,0)` on `qd[0]`.)
- **For each azimuth** in a grid over the training wedge (every 3° across ±33°):
  set `base = azimuth`, search extended forward/up postures over shoulder/elbow/
  wrist, score by ballistic throw distance (drag, `model._ball_accel`), and reject
  poses whose windup cock `q_release − qd_release·(t_throw/2)` leaves joint limits.
- Output `throw_pose_table.npy`: array of `{azimuth_deg, q, elev_deg, speed}` sorted
  by azimuth; every entry has all release joint velocities ≤ `qd_max`.

### 1b. Azimuth lookup in `_optimized_release`

- Load the table onto the system. `_optimized_release` picks the **nearest-azimuth**
  entry to the target azimuth, uses its posture `q` (base already ≈ azimuth), and
  re-solves the frozen LP for the **exact** launch direction `d` (nearest table
  azimuth is within 1.5°, so the re-solve stays feasible and aims exactly).

### 2. Monotonic sweep — `simulation_class/model_pybullet.py` + `robot_arm/arm_controller.py`

- `_optimized_release`: add the `qd[0]=0` bound to the LP; keep the neutral-reset
  fix (already applied). Compute and return `q_windup` for the monotonic ramp.
- `plan_throw`: accept a `q_windup_override`. When given, build the throw phase as
  the `q_windup → q_release` rest-to-`qd_release` ramp, and recompute `q_windup`
  from the stretched `t_throw` inside the torque-feasibility loop. Keep the
  existing torque check and the windup-phase feasibility check.
- Base rotates to azimuth during windup (`q_windup[0] = q_release[0]` since
  `qd_release[0]=0`), then held still.

### 3. Verification — `scratchpad/` real-rollout check

- Assert over a target/speed sweep: no blowup; land azimuth matches target
  azimuth within ~2° across ±39°; `max |qd|` over the whole trajectory ≤ `qd_max`
  (the guarantee the old pose failed). Report throw distance vs old 35 cm.

## Risks / notes

- If no posture satisfies both aiming and the windup-limit rejection at useful
  distance, relax the elevation range or the posture grid before weakening the
  `qd_max` guarantee — the guarantee is the point.
- The old contorted `throw_pose.npy` is overwritten; the base-using LP path in
  `_optimized_release` gains a `qd[0]=0` bound (no other mode calls it, so no
  cross-mode impact).
