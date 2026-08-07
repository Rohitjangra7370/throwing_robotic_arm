# Kinova Gen3 hardware bring-up — MC-PILOT throw

**Status: code written, UNTESTED on real hardware. Bring up in stages. Never skip stages.**

> **On run day, follow `HARDWARE_RUNBOOK.md`** — it carries the exact checkpoint/profile/ball
> configuration, the pre-flight checks with their verified outputs, and the per-stage gates.
> This file is the safety model and the reference.

The trained policy is a tiny RBF (target → release speed). The throw trajectory is
planned by the *same* sim `ArmController` (IK + 3-phase cubic) used everywhere in
this repo, so **hardware executes an identical motion to simulation** — only the
executor changes (PyBullet step → Kortex joint-velocity stream).

## Files
- `robot_arm/kinova_hardware.py` — safety-gated Kortex executor (`HardwareThrowExecutor`, `SafetyLimits`).
- `run_hardware_throw.py` — staged bring-up CLI (`plan` / `connect` / `home` / `gripper` / `throw`).

## Hardware / software checklist
- Kinova Gen3 7-DOF + gripper (Robotiq 2F-85 or Kinova), powered, **Ethernet to control PC**.
- Control PC: Ubuntu, **system `python3`** (there is no venv on this machine). `kortex_api` 2.6.0.post3 is already installed; if reinstalling, use `python3 -m pip install <kinova-kortex-wheel>` — bare `pip` here resolves to the Blender snap's Python 3.13. Confirm the arm's IP (default assumed `192.168.1.101`).
- 1× Intel RealSense D435/D455 overhead (perception, added AFTER the arm throws — not needed for stages 0–5).
- Small basket on the **same plane as the robot base** (`target_height=0.0`), 0.60–0.80 m out.
- **Tennis ball: 57.7 g, 65.4 mm diameter** — the trained GP's `ball_mass=0.0577`,
  `ball_radius=0.0327`. A ping-pong or foam ball is a different drag regime and the policy
  will not transfer. Mark it brightly for the vision stage.
- **Physical e-stop within reach. Workspace cleared. Ball secured. Bystanders back.**

## The safety model (why this can't over-drive the arm)
1. **Dry-run by default.** No `--arm` → no connection, no motion. `plan` never touches the arm.
2. **`speed_scale ∈ (0,1]`, default 0.15.** The whole throw is time-stretched by `1/speed_scale`: positions follow real geometry, wall-clock velocities scale *down*. 0.15 = a safe 15% rehearsal (ball dribbles); 1.0 = the real throw.
3. **Hard velocity clamp to `qd_max`** (the profile's measured Gen3 limits: 1.396 rad/s J1–4, 1.222 J5–7). Commands are only ever clamped *down*. A bug cannot over-speed a joint.
4. **Whole-trajectory position pre-check** before any motion — every joint verified inside a soft envelope; **fails closed** (aborts).
5. **Guaranteed stop** — any exception / Ctrl-C / normal exit sends zero-velocity and drops servoing in a `finally` block (context manager). The arm stops.
6. **`throw` interlock** — real motion needs `--arm` **and** `--confirm` (you assert workspace clear + e-stop in hand). Default speed is a slow rehearsal, not a throw.
7. **Onboard controller** enforces Kinova's own hard limits beneath ours (second independent net).

## Staged bring-up — do these in order, watch every run

Use `--robot kinova_gen3_dyn`, not `kinova_gen3`: the latter is `control_mode=kinematic` with
`tau_max=None`, so precheck silently skips the torque check. Use a `results_kinetic_chain_gen3/`
checkpoint — that is the trained overhead throw. Seed 1 predates `opt_pose` being recorded in the
config and additionally needs `--opt_pose throw_pose_table.npy`; seeds 2–3 self-describe.

```bash
# 0. DRY-RUN plan only. No arm. Verify BOTH "release pos in safe box: True" AND "PRECHECK: PASS".
#    Plan at --speed_scale 1.0: the 0.15 default reports a 7x-smaller qd and hides the headroom.
python3 run_hardware_throw.py plan --robot kinova_gen3_dyn --speed_scale 1.0 \
    --log_path results_kinetic_chain_gen3/2 --target 0.75 0.05 --u_cap 1.60

# 1. Connect + read joint state (no motion). Compare readback against q_neutral before stage 2.
python3 run_hardware_throw.py connect --arm --ip 192.168.1.101 --robot kinova_gen3_dyn

# 2. Gentle homing to neutral (slow, P-servo, speed-capped).
python3 run_hardware_throw.py home --arm --ip 192.168.1.101 --robot kinova_gen3_dyn

# 3. Gripper open/close test.
python3 run_hardware_throw.py gripper --arm --ip 192.168.1.101 --close
python3 run_hardware_throw.py gripper --arm --ip 192.168.1.101 --open

# 4. SLOW rehearsal throw @ 15% (~57 s wall clock; ball dribbles; validates motion + timed release).
python3 run_hardware_throw.py throw --arm --ip 192.168.1.101 --robot kinova_gen3_dyn \
    --log_path results_kinetic_chain_gen3/2 --target 0.75 0.05 \
    --speed_scale 0.15 --u_cap 1.60 --confirm

# 5. Escalate ONLY after each is clean: 0.15 → 0.30 → 0.60 → 1.00
#    ... --speed_scale 0.30 --confirm   (then 0.60, then 1.00)
```

`--u_cap 1.60` on every planning command: the table's kinematic max 1.628 m/s is not
follow-through recoverable. The trajectory is 8.53 s at `speed_scale=1.0` (release at 4.93 s),
56.9 s at 0.15 (release at 32.9 s) — the tail is commanded deceleration, do not abort into it.

## Three things that WILL need work during bring-up (flagged honestly)
- **Kortex API names — now verified, statically.** All Kortex calls are centralised in
  `_KortexBackend`, and all eight symbol groups (`SendJointSpeedsCommand`, `SendGripperCommand`,
  `RefreshFeedback`, session setup, `JointSpeeds`, `GripperCommand`/`GRIPPER_POSITION`) resolve
  against the installed `kortex_api` 2.6.0.post3 (2026-08-05). That proves the names exist, not
  that the arm accepts the commands. Any fix still goes in that one class.
- **Joint-angle convention on readback.** `read_joint_state()` applies a bare `deg2rad` to Kortex
  feedback. Gen3 continuous joints report in **[0, 360)**, and our neutral has small *negative*
  angles (−0.6°, −2.3°, −0.7°), which would read back near 359°. `home()`'s P-servo would then
  see a −6.28 rad error and drive that joint the long way around for the full 4 s. **Check the
  stage-1 readback against `profile.q_neutral` before stage 2**; if any joint reads > π rad, add
  unwrapping here first. Dry-run cannot catch this — it fakes the readback.
- **Gripper release delay** is the dominant real-world error (arm decelerates while the gripper
  opens → systematic undershoot; this is the paper's §5 `ReleaseTimingJitter`). Measure it
  during stage 4–5 (command-to-open-time vs actual release), then either advance the gripper
  trigger by that delay or feed it into the MC-PILOT model update. Do NOT expect the sim
  policy to hit targets on hardware until this is calibrated.

## After the arm throws reliably: close the loop
Add the RealSense: detect basket `(Px,Py)` before each throw (policy input) and ball landing
`(x,y)` after (background-subtract before/after frames) → landing error → feed into the
MC-PILOT model-learning loop (~10 real throws) → the GP adapts the sim policy to real physics.
That adaptation + a sim-vs-real comparison is the ICRA-relevant result. The
`mc-pilot-pybullet-yolo` variant already has the HSV/YOLO detection pipeline to reuse.
