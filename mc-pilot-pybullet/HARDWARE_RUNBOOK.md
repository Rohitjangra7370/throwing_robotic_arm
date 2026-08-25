# Hardware run day — one-page runbook (Kinova Gen3)

Companion to `HARDWARE_SETUP.md` (safety model, wiring). **This page is the order of operations.**
Everything runs from `mc-pilot-pybullet/` with `python3`. Bench evidence 2026-08-05 (§1); on-arm read-only evidence 2026-08-07 (§1b).

## 0. The exact configuration we are running

| Thing | Value | Why it matters |
|---|---|---|
| Checkpoint | `results_kinetic_chain_gen3/2` | Seed 2 (2.75 cm, best of 3). Its config **self-describes** `opt_pose`. Seed 1 works too but **must** add `--opt_pose throw_pose_table.npy`. |
| Robot profile | `kinova_gen3_dyn` | **Not** `kinova_gen3` — that one is `control_mode=kinematic`, has `tau_max=None`, so precheck **silently skips the torque check**. |
| Ball | **Tennis ball, 57.7 g, 65.4 mm dia** | The GP was trained at `ball_mass=0.0577, ball_radius=0.0327`. A ping-pong/foam ball is a different drag regime; the policy will not transfer. (`HARDWARE_SETUP.md` says ping-pong — that line is wrong.) |
| Target plane | **z = 0 = the robot base plane** | `target_height=0.0`. The bin sits on whatever surface the base is bolted to, not the floor below it. |
| Target band | **0.60 – 0.80 m** horizontal from base | Outside this the policy extrapolates. Start at `--target 0.75 0.05`. |
| Speed cap | `--u_cap 1.60` on every command | Table's kinematic max 1.628 m/s is **not follow-through recoverable**. Policy asks 1.17–1.55, so the cap rarely binds — pass it anyway. |
| Motion duration | **8.53 s** @ `speed_scale=1.0`, **56.9 s** @ 0.15 | Release fires at **s = 4.93 s** (58% in). The remaining ~3.6 s is *commanded* follow-through deceleration. **Do not abort after the ball leaves** — that is the arm braking itself. |

## 1. Bench sanity checks — do these before the arm is powered (~10 min)

Run each; do not proceed on a failure. All four passed on 2026-08-05:

```bash
python3 -m pytest tests/ -q                       # -> 87 passed in ~20s
python3 run_hardware_throw.py plan --robot kinova_gen3_dyn --speed_scale 1.0 \
    --log_path results_kinetic_chain_gen3/2 --target 0.75 0.05 --u_cap 1.60
#   plan at 1.0, NOT the 0.15 default -- 0.15 shows a 7x-smaller qd and hides the headroom
#   -> policy release speed 1.498 m/s   release pos in safe box: True   PRECHECK: PASS
#   -> peak |qd| 1.30/1.40 (93%)   peak |tau| 8.6/39.0 (22%)   T=8.528s
python3 run_hardware_throw.py throw --robot kinova_gen3_dyn \
    --log_path results_kinetic_chain_gen3/2 --target 0.75 0.05 --speed_scale 1.0 --u_cap 1.60
#   full DRY-RUN (no --arm = no motion) -> 342 ticks, 40Hz, worst tick 0.00ms late
#   -> release quantisation: 25.0 ms at 40 Hz -> up to 3.7 cm undershoot (see R0)
python3 eval_adapted_height.py --log_path results_kinetic_chain_gen3/2 --num_throws 30 --seed 987654
#   fresh unused seed -> mean 2.84cm, max 5.43cm, hit<10cm 100%, speed 1.17-1.55 m/s
```

Optional 5th (takes ~57 s, exercises exactly what stage 4 will run): the same dry `throw` at
`--speed_scale 0.15` → `wall_T=56.85s`, release at wall **32.88 s**, 2275 ticks, 40 Hz,
worst tick 0.00 ms late.

**Read two lines on `plan`, not one.** `PRECHECK: PASS` only covers the *trajectory*. The
`release pos in safe box:` line is separate and `plan` prints PASS even when it says `False`.
(`throw` does fail closed on it — but catch it at `plan`.)

## 1b. On-arm facts, measured read-only (2026-08-07) — no command was sent

| Fact | Value |
|---|---|
| **Arm IP** | **192.168.1.101** — *not* the old `192.168.1.10` default, which is **this PC's own address** on `enp108s0`. Pinging it "succeeded" against ourselves. Default now corrected everywhere. |
| Identity | Gen3 **L53K**, SN **WO545410-1**, PN KR10944, 7 actuators, fw `872547072` |
| Ports | tcp/10000 open (Kortex session), tcp/80 open (web UI), tcp/22 open. `10001` is **UDP** — a closed TCP probe there is expected, not a fault. |
| Health | `RUN_MODE`, `SINGLE_LEVEL_SERVOING`, stationary, motors 32–39 °C, actuators 23.1–23.3 V, no faults |
| **Velocity limits (arm's own)** | **80.002 / 80.002 / 80.002 / 80.002 / 70.004 / 70.004 / 70.004 deg/s** = exactly our `qd_max` 1.3963 / 1.2218 rad/s ✅ |
| **Torque limits (arm's own)** | **39 / 39 / 39 / 39 / 9 / 9 / 9 Nm** = exactly our `tau_max` ✅ |

That last pair is the important one: every feasibility check in this repo — the release LP, the
whole-trajectory precheck, the follow-through guard — is built on those numbers, and the arm
confirms them to float32 precision. They were not fiction.

## 2. Staged bring-up — never skip a stage, never skip a gate

```bash
# 0.5 READ-ONLY BENCH CHECK -- sends NO command. Run this before `connect`.
#     Must end "0 FAIL". Verified 38/38 OK on 2026-08-07.
python3 hw_readonly_check.py --ip 192.168.1.101
# 1. CONNECT (writes one thing: zero joint speeds on teardown).  GATE: §3 R1/R6.
python3 run_hardware_throw.py connect --arm --ip 192.168.1.101 --robot kinova_gen3_dyn
# 2. HOME (slow P-servo, capped at 25% qd_max).  GATE: arm reaches neutral, no long-way rotation.
python3 run_hardware_throw.py home --arm --ip 192.168.1.101 --robot kinova_gen3_dyn
# 3. GRIPPER, no arm motion.  GATE: time open command -> fingers actually clear (§3 R3).
python3 run_hardware_throw.py gripper --arm --ip 192.168.1.101 --close
python3 run_hardware_throw.py gripper --arm --ip 192.168.1.101 --open
# 4. REHEARSAL @15% (~57 s, ball dribbles).  GATE: no collision, no fault, release visibly fires.
python3 run_hardware_throw.py throw --arm --ip 192.168.1.101 --robot kinova_gen3_dyn \
    --log_path results_kinetic_chain_gen3/2 --target 0.75 0.05 \
    --speed_scale 0.15 --u_cap 1.60 --confirm
# 5. ESCALATE one step at a time, clean run required between each: 0.30 -> 0.60 -> 1.00
```

E-stop in hand for every stage from 2 onward. Workspace clear for the **whole 8.5 s**, not just
the swing. Film every run — the standing rule on this project is that numbers alone have missed
real motion bugs three separate times.

## 3. The things most likely to bite

- **R0 — the control rate was wrong, and it costs accuracy. FIXED, but read this.**
  Kinova's own driver docs: *"The base high level commands are treated every 25 ms inside the
  robot. High level control cannot be achieved at a rate faster than 40 Hz for now."* We stream
  `Base.SendJointSpeedsCommand` with the arm in `SINGLE_LEVEL_SERVOING` — that is **high-level**,
  so the ceiling is **40 Hz**. The old `control_hz = 1000.0` was justified by Kinova's 1 kHz
  figure, which belongs to **`LOW_LEVEL_SERVOING`** (per-actuator `BaseCyclic.Refresh`), a path
  this code does not use. Not a hazard — `JointSpeeds` with `duration=0` is held until superseded,
  so surplus commands were merely coalesced — but *"1000 Hz achieved, worst tick 0.00 ms"* was
  measuring our own loop, not the arm. Now clamped to 40 Hz with a printed warning.
  **The consequence is real: 25 ms of release quantisation = up to 3.7 cm of undershoot at
  1.498 m/s — larger than the entire 2.89 cm sim accuracy.** `precheck` now prints it as a
  landing-error term. It is *not* reducible by looping faster; only a move to `LOW_LEVEL_SERVOING`
  buys back millisecond release timing. Treat 3.7 cm as the current high-level error floor and
  expect sim-vs-real to be dominated by it plus gripper latency (R3).

- **R1 — joint-angle convention. CONFIRMED ON THE ARM, AND FIXED (2026-08-07).** Kortex reports
  **every** joint on **[0, 360)**, limited joints included — measured: joint 3 came back at
  `247.37°` (4.318 rad) against its own ±2.57 rad limit. The old bare `deg2rad` in
  `read_joint_state()` would have made `home()` servo the long way for the whole window.
  Now: `read_joint_state()` wraps to (−π, π], and `home()` computes error **per joint type** —
  shortest path for the continuous joints (0, 2, 4, 6), direct difference for the limited ones
  (1, 3, 5), whose error may legitimately exceed π. A single π threshold was wrong in *both*
  directions and would have blocked a legal stage 2. 4 regression tests, built on the real
  readback. `_assert_readback_sane` still fails closed if a joint reads outside its URDF range.
- **R6 — the arm can be held by another master.** Observed live: between two runs the state went
  `ARMSTATE_SERVOING_READY` → **`ARMSTATE_SERVOING_MANUALLY_CONTROLLED`** because someone was
  driving it from the web UI. Streaming joint speeds into that is a two-master situation.
  `hw_readonly_check.py` now FAILs on it — **it must read `SERVOING_READY` before stage 2.**
- **R7 — homing is a big slow move from the current pose, not a nudge.** Measured 4.040 rad
  (231°) on joint 3, needing **~14.5 s** at the 0.25·qd_max cap. The old `duration=4.0` default
  would have stopped part-way and left an undefined pose as the *starting point of a throw*.
  `home()` now sizes its own window from the measured distance and says so. Expect ~15 s and
  let it finish.
- **R2 — the wrong-throw trap.** Seed 1's config predates `opt_pose` being recorded. Omit
  `--opt_pose` and the planner **silently falls back to a legacy IK+pinv throw**: measured
  tonight, |v| drops 1.496 → 0.471 m/s and the release moves outside the safe box, while `plan`
  still prints `PRECHECK: PASS`. Prefer seed 2/3; if using seed 1, pass the table.
- **R3 — gripper release latency: MEASURED and COMPENSATED (2026-08-07).** 1 kHz UDP feedback,
  15 trials: **67.9 ± 6.4 ms** command→fingers-move (range 59.5–80.0). Uncompensated that is
  **10.2 cm** at 1.498 m/s — 3.5× the whole 2.89 cm sim accuracy, and it would have read as the
  policy failing to transfer, not as a timing bug. The scatter (6.4 ms) is almost exactly what
  25 ms of command quantisation predicts alone, so the gripper itself is repeatable and the
  latency is compensable. `rehearse_or_throw` now fires OPEN at `t_r − lead·speed_scale`;
  expected residual **~1.0 cm**. **Still static and unloaded** — during a throw the fingers hold
  a ball and the arm decelerates. Validate against real landings before trusting it; re-measure
  with the ball gripped. `gripper_lead_s = 0.0` reproduces the uncompensated baseline.
- **R4 — Kortex names verified statically AND against the arm.** All 8 symbol groups
  (`SendJointSpeedsCommand`, `SendGripperCommand`, `RefreshFeedback`, session setup, `JointSpeeds`,
  `GripperCommand`/`GRIPPER_POSITION`) resolve against the installed `kortex_api` 2.6.0.post3.
  Read paths are now exercised on the real arm (`hw_readonly_check.py`, 38/38). Two calls are
  UNSUPPORTED on this firmware — `GetControlMode` and the `*SoftLimitation` pair — neither is
  used by the throw. **No write path has ever run**: stage 2 is still the first one.
- **R5 — headroom is thin at full speed.** At `speed_scale=1.0` peak commanded velocity is **93%
  of `qd_max`** (torque is comfortable at 22%). Any joint that clamps means the ball lands short
  with nothing in the log — precheck fails closed on this, so a `VELOCITY CLAMPING ACTIVE` report
  is a stop, not a warning.

## 4. Record per throw (this is the dataset, not a debug log)

Target `(Px,Py)` · commanded release speed · `speed_scale` · landing `(x,y)` measured on the plane ·
`ex.last_exec_stats` (ticks, achieved Hz, **worst_late_ms**, `release_wall_s`) · video file · ball ID.
At 1.5 m/s, **1 ms of release-timing error = 1.5 mm of landing error**, and the high-level path
quantises release to 25 ms (**~3.7 cm**, R0) before the gripper's own latency (R3) is even counted.
The timing record belongs next to the landing point, not in a console scrollback — it is the
largest known term in the error budget, so it has to be measured per throw, not assumed.

Once ~10 real throws are logged, that is the input to the real MC-PILOT model update
(`HARDWARE_SETUP.md` §"close the loop") — the ICRA-relevant result.

## 5. Environment gotcha

`kortex_api` pins **protobuf 3.5.1**, already installed here and already downgraded system-wide.
The throw pipeline is unaffected (74/74 tests pass), but **onnx / tensorboard / wandb are broken
on this machine**. Do not try to fix that on run day. Never use bare `pip` — it resolves to the
Blender snap's Python 3.13. Use `python3 -m pip`.

## 6. Landing measurement (vision) — NOT YET RUN ON REAL HARDWARE

Everything in this section is **verified synthetically only** (see `CLAUDE.md`'s ball-tracking
bullet and `docs/superpowers/specs/2026-08-25-ball-tracking-design.md`). No frame from the real
camera has ever entered this pipeline — the D435i is currently unplugged. This is the procedure
to run once it is mounted and `T_B_C` is re-measured; do not treat any number below as measured
until it has actually been produced on run day.

**Step 0, once, before the first real throw of the day: pick exposure and emitter.**
`IRRecorder`'s defaults (`exposure_us=2000`, `emitter=True`) are reasoned, not measured — the
A/B that would validate them (spec §9: 2 ms exposure risks under-exposure indoors with the
emitter off; the emitter's static floor pattern should subtract out in background diff but may
saturate the ball) has never been run. Run `tune_ir_exposure.py` **first**, on run day, before
any throw is recorded, and write the chosen values here:

| Setting | Value | Chosen on | Notes |
|---|---|---|---|
| `exposure_us` | *(not yet run)* | | |
| `emitter` | *(not yet run)* | | |

**Per throw:**

```bash
python3 record_throw_ir.py ...     # ring-buffers the dual-IR window to disk; keep every recording,
                                    # it is a permanent regression fixture, not a scratch file
python3 measure_landing.py ...     # offline: recording -> first-contact (x, y) in base frame
```

**Refusal conditions — a refusal means RE-THROW, never hand-tune a threshold to make a bad
track pass:**

- fewer than **12** usable frames on the track (of ~44 expected)
- RANSAC inlier fraction below **0.6**
- RMS reprojection residual above **1.0 px**

Each of these raises with a diagnostic identifying which one fired, in the style of
`ball_detector.py`'s `detect_ball_bgsub` — the pipeline is built to fail loudly rather than
report a plausible-looking wrong number, and this project has a documented history of exactly
that failure mode. If a track is refused, re-throw; do not lower the thresholds to force a
number out of a bad recording.

Independent cross-check (spec §7.3, Task 11 Step 3 — **not yet run**, needs the overhead mount
and a re-measured `T_B_C`): on throws where the ball does not bounce far, compare
`measure_landing.py`'s first-contact point against the existing static `ball_detector.py` +
`ray_plane.ball_center_on_plane` resting measurement. Agreement to within a couple of
centimetres is expected; a systematic offset in one direction implicates `T_B_C`, not the
fitter, since both paths share the extrinsic. Record both numbers and their difference — do not
adjust anything to make them agree.
