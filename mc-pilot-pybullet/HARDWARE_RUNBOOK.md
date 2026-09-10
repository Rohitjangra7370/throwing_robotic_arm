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

### 0.1 — TCP-offset checkpoint, current preferred config (added 2026-08-27)

The table above describes the checkpoint used through 2026-08-22. Since then the release solver
was fixed to account for the real Robotiq 2F-85's TCP offset (firmware `GetToolConfiguration`:
`tool_transform=(0,0,0.12)m` — see `CLAUDE.md`'s gripper-TCP-offset blocker), which the sim, the
LP, and `find_throw_pose.py`'s search had all always ignored (0.39 m/s / 12 cm error at the trained
release state, quantified 2026-08-22, fixed 2026-08-27). **A checkpoint trained under the old,
uncorrected physics is not safe to reinterpret under the new solver without retraining** — the
release *direction* changes (5.0° → 15° elevation is a re-optimization under corrected physics,
not a small correction), so a stale checkpoint's speed commands would launch on a different
trajectory shape than it was trained for.

| Thing | Value | Why it matters |
|---|---|---|
| Checkpoint | `results_kinetic_chain_gen3_tcp/1` | Retrained under TCP-correct physics (ball welded at the real 12cm offset in sim, not the flange). Fresh-seed eval (seed 24681012, never used in training): **mean 1.90cm, max 4.20cm** — beats the old checkpoint's best-of-3 (2.84cm/5.43cm). Seed 2 also trained (2.10cm/4.58cm, worse); seed 3 was interrupted mid-run and is not usable. |
| Pose table | `throw_pose_table_tcp.npy` | Re-searched with `--tool_offset_z 0.12 --floor_z -0.433`. Stamped with both `floor_z` and `tool_offset` — `run_hardware_throw.py`/`eval_adapted_height.py` refuse a mismatched `--tool_offset_z` against this stamp, mirroring the existing `floor_z` refusal pattern. |
| Required flags | `--opt_pose throw_pose_table_tcp.npy --tool_offset_z 0.12 --base_height 0.433` | All three needed together — the checkpoint's `config_log.pkl` self-describes `opt_pose` and `base_height` but **not** `tool_offset_z` (not recorded by the trainer), so it must always be passed explicitly or the run is refused (fail-closed, not silently wrong). |
| Speed cap | `--u_cap 2.00` recommended | New table's kinematic max is 2.07 m/s (all azimuths corner-solution-saturated at `qd_max`, same character as the old table's 1.628). Policy asks ~1.3–1.5 m/s in practice — cap rarely binds, pass it anyway. Trained with `--uMin 0.5 --uM 2.0`. |
| PRECHECK swept | 9 points across the full `[0.68,0.74]×[-0.25,+0.25]` band at `speed_scale=1.0`: all PASS, peak `|qd|` up to 99% at the ±0.25 extremes (not saturated), peak `|tau|` ≤ 36% everywhere. Not yet re-verified with `--wrist_roll_offset_deg` values other than 0/90 — the release posture changed substantially (different elevation/corner-solution), so the finger-clearance angle that worked for the old 5° throw **must be re-checked visually on the arm**, not assumed. |
| New tooling | `run_closed_loop_throws.py` (CLI, one throw per invocation: dashboard → confirm → throw → append a structured JSONL record — `throw_index`, `ball_id`, `q_release`/`qd_release`, `exec_stats`, `capture_file`, `landing_xy`). **Decoupled by default** — `landing_xy` stays `null`, filled in by a separate offline `measure_landing.py` pass matched by `throw_index`, matching that script's own "offline half of the vision pipeline" design. Pass `--measure --extrinsic <file>` to opt into polling `throws_dir` for `throw_capture.py`'s recording and filling `capture_file`/`landing_xy` immediately instead. `closed_loop_gui.py` wraps it in the same shape as `throw_gui.py` (pickup → grasp-verify → lift → throw → log, confirm checkbox resets every run, auto-incrementing `throw_index`). Neither duplicates `run_hardware_throw.py`'s planning/execution — both import and call it directly. |
| Sim sanity video | 3-throw render (`make_trained_kinetic_chain_video.py --tool_offset_z 0.12`), errors 0.6/1.6/1.8cm, ball visibly lands in the bin — visually confirmed, not just the numbers (this project's own rule, given past corkscrew/zero-amplitude-windup regressions that only showed up on frames). |

**DONE 2026-08-31 — `calib/T_B_C.npz` now exists.** Calibrated against the overhead mount with the
board on the floor: board recovered to **1.2 cm** of the real floor plane, PnP-vs-depth agreement
**0.8 cm**, wrist/D435i reprojection 0.20/0.15 px. See §0.2 — run `start_of_day.py`, which produces
it. `run_closed_loop_throws.py`'s `load_extrinsic_any()` reads either the `.npz` or the `.json`, so
`--extrinsic calib/T_B_C.npz` works directly and `--measure` is no longer blocked.

## 0.2 — START HERE on a run day: `start_of_day.py` (added 2026-08-31)

One command, one GO/NO-GO. Put the ChArUco board flat on the floor where the overhead D435i sees it,
jog the arm so its **wrist camera sees the same board**, leave it there, and run:

```bash
/usr/bin/python3 start_of_day.py --ip 192.168.1.101
#   -> env, arm (hw_readonly_check 42/42), cameras, calibration, gates, plan
#   -> writes calib/T_B_C.npz (+ timestamped archive and an audit .json)
#   -> last line is GO or NO-GO; it refuses to write the extrinsic if a gate fails
```

**The ChArUco calibration board is glued to the floor, permanently, and cannot be removed
between calibration and throwing** (corrected 2026-09-02 — an earlier version of this note
wrongly assumed it was a placeable board; it is not. The single ArUco tag inside the movable
target bin is a separate, unrelated marker and is fine where it is). Found 2026-09-02: a real
full-speed throw's landing measurement was refused (`no ballistic arc found`, best consensus
18-20/76-79 frames) even though the ball visibly flew. Root cause: the board's checkerboard
texture beats against the IR emitter's dot pattern and produces persistent per-frame diff noise
at the same fixed pixel locations for the whole recording, which `detect_candidates` reads as
ball-like blobs every frame — confirmed by plotting candidate `(u,v)` across a real recording
(clustered at a handful of unmoving coordinates for the full 1.4s window) and by visually
rendering the actual frames. This diluted the genuine ball track (visually confirmed: a single
bright blob entering top-of-frame and moving smoothly down-left) below RANSAC's 60%-inlier gate.

**Fixed in code, not by workaround**, since the board can't move: `perception/ball_track.py`'s
`reject_static_candidates` drops, per camera stream and per recording, any candidate that recurs
at nearly the same pixel location across many frames — computed fresh from each recording's own
candidates (no hand-drawn ROI, no per-mount tuning), wired into `build_observations` as the
default (`reject_static=False` to see the raw, unfiltered candidates for detector diagnosis).

**This alone was not enough, and the first follow-up diagnosis was WRONG — corrected same day.**
Filtering `throw_001.npz`'s board noise let RANSAC find a technically clean 18/24-frame consensus
(0.52 px RMS), but the fitted speed came out to 7.08 m/s against a commanded 1.41 m/s. This was
first (wrongly) attributed to the overhead camera's short stereo baseline making depth/vertical
velocity fundamentally unmeasurable over a short track. **The real cause, found by decomposing
the fitted velocity into `|v0_xy|` (0.87 m/s, plausible) vs `v0_z` (7.23 m/s alone, the entire
error): `p0`/`v0` are fit parameters at the recording's local `t=0`, which for
`session_camera.RingBuffer`'s capture is `PRE_S=0.45s` BEFORE the ball is ever released (`t` is
zeroed to the window's first frame, and the window starts at `t_release - PRE_S`).** Comparing
raw `v0` to a commanded RELEASE speed compares the wrong instant — backward-extrapolating a
correctly-measured post-release arc through 0.45s of gravity the ball never experienced in free
flight (it was still in the gripper) inflates the vertical component by `g * PRE_S ≈ 4.4 m/s` on
its own, on top of `mark_release()` itself timestamping the commanded gripper OPEN rather than the
ball's actual mechanical departure (`GRIPPER_RELEASE_LATENCY_S = 0.0679s`, `kinova_hardware.py`).
Using `release_t_offset = PRE_S + GRIPPER_RELEASE_LATENCY_S ≈ 0.518s` to evaluate `v0 + g·offset`
before comparing brought `throw_000`'s real release-time speed to 2.32 m/s against a 1.47 m/s
commanded — inside tolerance, where the naive comparison (7.28 m/s) was 5x off. The short-baseline
depth-degeneracy story may still contribute some residual noise, but it was not the dominant
cause and should not be assumed without re-testing if this resurfaces.

`measure_landing()` now takes `commanded_speed` **and requires `release_t_offset` alongside it**
(raises `ValueError` if one is given without the other — silently assuming `t=0` is release is
exactly the bug this guards against) and refuses when the release-time speed is more than 2x off.
`hardware_session.py` passes both automatically using the constants above.
**`run_closed_loop_throws.py` does NOT pass `commanded_speed`** — its recordings come from
`throw_capture.py`'s range-gated trigger (arms on ball detection, not on commanded release), a
different capture mechanism whose local-`t=0`-to-release relationship has not been established;
guessing `PRE_S` there would silently reintroduce the same bug in a different code path. Whoever
wires this up for that path needs to work out what its `t=0` actually means first.

One real question this did NOT resolve: even with the corrected offset, `throw_000`'s solved
landing (0.940, -0.110) vs. target (0.724, -0.009) is **24 cm off** — well beyond the checkpoint's
~1.9-2.4 cm sim/repeat accuracy. `solve_impact`'s (x, y) math is unaffected by which `t=0`
reference `v0` uses (it solves forward from the fit self-consistently), so this error is real, not
an artifact of the offset bug above. It may be a genuine first-real-ball-loaded-throw sim-to-real
gap, or a residual measurement/extrinsic issue — open, not yet investigated.

Tests: `tests/test_ball_track.py` (static-filter unit tests), `tests/test_landing_pipeline.py`
(board-noise integration test, and `release_t_offset` regression tests including one that
reproduces this exact incident: a shifted-time-origin recording that the naive offset=0 comparison
wrongly refuses and the correct offset accepts).

**Use `/usr/bin/python3` explicitly.** Bare `python3` on this machine resolves to a Conda base env
(3.14, no cv2/torch/pyrealsense2) — every tool on this page will fail with `ModuleNotFoundError`
that has nothing to do with the tool.

Gates, and why reprojection error is not one of them: a planar PnP absorbs a wrong principal point,
a wrong Euler convention, or a mis-scaled printout into the *pose* and still reports a fraction of a
pixel — the 12 cm tool-frame bug below reprojected at 0.18 px. So the gates are facts from outside
the model:

| Gate | What it is | Measured 2026-08-31 |
|---|---|---|
| FLOOR | board is on the floor and we know where the floor is, so its calibrated height + tilt must agree — catches the whole arm-side chain | −1.2 cm, 0.9° tilt |
| SCALE | the D435i's own depth measures board distance independently of intrinsics **and of the printed square size** — the only thing here that catches a "fit to page" printout | PnP 1.606 m vs depth 1.614 m (+0.8 cm) |
| REPEAT | 5 frame pairs solved separately; the spread *is* the measurement noise | **1.8 cm, 0.56°** |
| DRIFT | vs the stored extrinsic — a knocked mount | 0.2 cm, 0.1° |
| THROW | `run_hardware_throw.py plan`, reading **both** the `PRECHECK:` and `release pos in safe box:` lines | PASS / True |

**The REPEAT number is the honest accuracy bound on any landing measurement: ~1.8 cm**, not the
0.15 px reprojection error and not the sub-millimetre synthetic figures in §6. It is the same order
as the checkpoint's own 1.90 cm sim accuracy, so a single real landing cannot currently resolve a
sim-vs-real gap smaller than that. Averaging more frames does not fix it (the noise is not zero-mean
across a static scene); a bigger board, a closer camera, or multiple arm poses would.

## 0.3 — The training-session app: `hardware_session.py` (added 2026-09-01/02)

One Tk window for the whole run-day flow, instead of one CLI invocation per throw. §7's
`run_closed_loop_throws.py` / `closed_loop_gui.py` still work and are unaffected — this is a
different, more complete tool that wraps start-of-day, N throws, and the model update in one place,
calling the exact same planner/executor as §1/§2/§7 (`run_hardware_throw.py`, `pickup_and_lift.py`,
`HardwareThrowExecutor`) rather than a second implementation of any of it.

```bash
/usr/bin/python3 hardware_session.py --ip 192.168.1.101 --robot kinova_gen3_dyn \
    --log_path results_kinetic_chain_gen3_tcp/1 --opt_pose throw_pose_table_tcp.npy \
    --tool_offset_z 0.12 --base_height 0.433
#   --dry_run forces args.arm=False, but ONLY for the throw cycle's plan/execute path --
#   step_pickup() always calls the real pickup_and_lift(), which always moves the real
#   arm and grasps for real, regardless of this flag. Read --help before trusting it.
```

**Four buttons, four separate gates.** `SessionState` (`hardware_session.py`) is the actual
authority — read it, not this table, if the two ever disagree:

| Button | Enabled when | What it does |
|---|---|---|
| **Run start-of-day** | Always (only greys out while its own worker is mid-run) | Runs `start_of_day.py`'s stages verbatim (§0.2) — env, arm read-only check, calibration, throw-readiness plan. GO → `CALIBRATED`, then the camera thread's own confirm brings the stage to `READY`. NO-GO → `BLOCKED`; every throw-gated button stays disabled until start-of-day is re-run clean. |
| **Pick up & throw** | Stage is `READY` or `MODEL_UPDATED` (`can_throw()`) | Runs one `ThrowCycle` — see the per-throw sequence below. |
| **Update model** | ≥ `--min_throws_for_update` (default **5**) *qualifying* throws logged (`can_update_model()`) | Fits the release model from this session's own logged throws. **Qualifying means full-speed only** — see the new rule below; `n_throws`/`n_measured` (the throws table, "how many landings") still count every throw, rehearsals included. |
| **Re-optimize policy** | Only after "Update model" has actually run once this session (`can_reoptimize_policy()`) | Locked at session start no matter how many throws are logged — there is no path to this button that skips a model update. Once unlocked (added 2026-09-02): re-optimizes the policy against the SAME in-memory model "Update model" left behind (real throws already folded into its GP, not a fresh reload from disk) via `reoptimize_policy()`, mirroring `adapt_policy_height.py`'s `mc.reinforce_policy(...)` call. Writes a **new, timestamped** checkpoint directory — refuses outright (`FileExistsError`) if asked to write into an existing, non-empty one, so `results_kinetic_chain_gen3_tcp/1` can never be overwritten. **The new checkpoint is not thrown automatically** — its release posture may differ from the one visually verified for finger clearance, so it goes back through §0.2 and restarts the speed-scale ladder at 0.15, same as any other checkpoint; the report pane says this explicitly. |

**Per-throw sequence** (`ThrowCycle`, `hardware_session.py`):

1. Place a ball at the recorded pickup pose.
2. Click **Pick up & throw** → `step_pickup()`. Grasp is verified, not assumed: **58–59% closed on
   a real ball** (position and velocity both stall) vs **99–100% closing on nothing**
   (`pickup_and_lift.py`, §2's R3). A false grasp refuses right here, before any throw motion —
   reload the ball and retry.
3. **Plan** (`step_plan()`) → read **two separate lines**, the same trap §1/§2/§7 already warn
   about: `PRECHECK: PASS/FAIL` (trajectory feasibility only) and `release pos in safe box:
   True/False` (a different check — `PRECHECK: PASS` alone does not mean throw). Both must be true.
4. **Escalation gate** (`check_scale()` / `hardware_learning.scale_allowed`) — the requested
   `speed_scale` needs a clean logged run at the rung below it first (0.15 → 0.30 → 0.60 → 1.00).
   Refuses outright; it does not silently clamp to a lower rung for you.
5. **Re-tick the confirm checkbox.** It clears after every single throw, on purpose — "confirmed
   once" is not a safety property this app offers. If you don't see it checked, it is not checked.
6. **Throw.** 8.5 s total at `speed_scale=1.0`; release fires at **4.93 s (58% in)**. **The
   remaining ~3.6 s is commanded braking — do not abort on it.** That is the arm decelerating
   exactly as planned, not a fault.
7. **Measure** — `measure_landing.py` against the dual-IR track the camera thread captured,
   triggered off the `on_release` timestamp.
8. **Log.** A row is written no matter what happens past this point — including a refused
   measurement, or an exception raised anywhere in the execution block itself (`set_gripper`,
   `home`, `rehearse_or_throw`) — as long as the ball has physically left the hand:
   `landing_xy: None` plus a `refusal_reason`, never a silently dropped throw.
9. Reload — place the next ball, go back to step 2.

**New rule (2026-09-02): ladder throws are rehearsals, not data.** `speed_scale` is a time-stretch
on the streamed joint speeds (`qd_cmd = qd * ds_dwall`, `kinova_hardware.py`) — a throw logged at
0.15 really did release at roughly 0.15× speed, physically, not "the same throw measured noisily."
**Only throws logged at `speed_scale == 1.0` count** toward "Update model"'s threshold
(`SessionState.n_measured_full_speed`) and only they are used by
`hardware_learning.fit_release_model` — everything else is excluded, and the exclusion is *reported*
(`n_excluded_rehearsal`, named directly in the fit's own `text`), never silently dropped. A record
missing `speed_scale` entirely does not default to counting, either. Fitting a rehearsal as if it
were a full-speed throw pulls the release gain toward ~0.15 instead of ~0.9 — with a
confident-looking residual sigma sitting right next to the wrong number.

**A refused measurement means RE-THROW.** Same rule as §6: never loosen a threshold to force a
number out of a bad recording. A refusal is itself logged (step 8 above) — it is evidence about
the rig, not a gap you patch over in the dataset.

**Why this batches instead of updating after every trial, the way the paper does.** MC-PILOT's own
loop (`policy_learning/MC_PILCO.py`) updates the model and re-optimizes the policy after *every*
single trial — that is the algorithm. This app deliberately does not: policy re-optimization takes
minutes, and the operator should be able to see what the real throws actually did to the model —
gain, offset, residual sigma, release-direction spread — before the policy moves and changes what
the next throw even targets. **"Update model" reports and stops, by design.** It does not chain
into "Re-optimize policy" automatically, even once both are fully wired end to end. Read the
report; decide whether to re-optimize.

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
- **R8 — `GetMeasuredCartesianPose` reports the TOOL frame, not the flange. FIXED 2026-08-31.**
  This arm has `tool_transform = (0, 0, 0.12) m` configured for the Robotiq 2F-85, so that call
  returns a pose 12 cm beyond `end_effector_link`. Both calibration scripts composed the URDF's
  **flange**→camera offset onto it, putting the wrist camera 12 cm out of place and feeding the error
  straight into `T_B_C`. Caught by putting the board on the floor and noticing the calibration placed
  it 14.0 cm *underground*. Verified against PyBullet FK on the same URDF the planner uses: the
  reported pose sits `[-0.0035, -0.0052, +0.1251]` m from the flange in the flange frame, versus the
  firmware's own `0.120` — same vector. **The fix is not to subtract the tool transform** but to skip
  that call entirely: `perception/wrist_chain.base_to_wrist_camera` goes from measured *joint angles*
  through FK to `camera_color_frame`, which also retires the never-verified assumption that
  `theta_x/y/z` are intrinsic-XYZ degrees. Board error after the fix: **1.2 cm**. Regression:
  `tests/test_wrist_chain.py::test_tool_frame_pose_is_not_the_flange_pose`.
  This is the **third** time the 2F-85's 12 cm has cost this project something (release speed, then
  the release box, now the extrinsic). When a frame is off by ~0.12 m here, suspect it first.

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

**`run_closed_loop_throws.py` automates this whole row** — target, commanded speed, `speed_scale`,
`q_release`/`qd_release`, tick timing, `ball_id`, and (with `--measure`) landing (x,y) and the
recording filename — appended as one JSON line per throw to `closed_loop_throw_log.jsonl`. See
§0.1. Manual recording as described above still applies if running `run_hardware_throw.py throw`
directly instead.

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
| `exposure_us` | `4000` | 2026-09-02 | `tune_ir_exposure.py` sweep: 1000/2000us detected the ball in 0/N frames (too dark) at both emitter settings; 4000us was the shortest exposure with usable detections. `emitter=True` at 4000us: 127/134 frames_with_ball (94.8%), 0% saturation, med_circ 0.59. 8000us scored higher circularity (0.80) but is a longer exposure, so per the tool's own selection rule (shortest exposure, most frames_with_ball, sat%~0) 4000us wins — the circularity dip is noted but not saturation-driven. |
| `emitter` | `True` | 2026-09-02 | Same sweep: `emitter=True` beat `emitter=False` at every exposure that detected anything (4000us: 127/134 vs 94/133; 8000us: 98/134 vs 0/135). |

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

## 7. Closed-loop throw session (added 2026-08-26) — TCP-offset checkpoint + camera, per-throw structure

**Context.** The gripper TCP-offset blocker (`CLAUDE.md`) is fixed at the code level: `release_solver.py`,
`find_throw_pose.py`, and `model_pybullet.py` all accept a `tool_offset` and default to zero (every prior
checkpoint/table/test is unaffected). Measured at the trained release state: the offset is 99.98% vertical
(no aim/collision issue) but the wrist's rotation at release couples into a **+0.39 m/s / +26% speed**
effect the old checkpoint never modeled (`ω × r_offset`, confirmed both analytically and by real PyBullet
constraint physics — see `tests/test_tool_offset.py`). Decision made: retrain from scratch with the ball
physically welded at the TCP (`--tool_offset_z 0.12`), rather than hand-correcting the old checkpoint —
a changed release geometry needs a policy trained on it, not a runtime patch.

**Sequence for this checkpoint, in order:**
1. `find_throw_pose.py --tool_offset_z 0.12 --floor_z -0.433 --out throw_pose_table_tcp.npy` — done
   2026-08-26 (~30 min). Table re-optimizes under the corrected physics: elevation 5.0°→15°, kinematic max
   1.628→2.07 m/s. **This table is only valid for training a NEW checkpoint, never for planning through the
   OLD `results_kinetic_chain_gen3` checkpoint** — different release geometry, incompatible policy mapping.
2. `train_mc_pilot_pb_arm.py --robot kinova_gen3_dyn --opt_pose throw_pose_table_tcp.npy --tool_offset_z 0.12
   --base_height 0.433 --flight_targets --uMin 0.30 --uM 2.00 --lm 0.30 --lM 0.90 --results_root
   results_kinetic_chain_gen3_tcp` — the old profile-default speed bounds (0.30–0.60 m/s) don't reach this
   table's much faster release; `--uM`/`--lm`/`--lM` had to be re-derived for the new geometry (the trainer's
   own reachable-band check catches a bad guess before wasting compute — read its error, don't skip it).
3. `eval_adapted_height.py --log_path results_kinetic_chain_gen3_tcp/1 --opt_pose throw_pose_table_tcp.npy
   --tool_offset_z 0.12 --num_throws 30 --seed <fresh, unused>` — compare against the old checkpoint's
   ~2.84 cm / 5.43 cm baseline before touching hardware. **Never skip this** — training cost is not accuracy
   (see "Things that will bite you" in `CLAUDE.md`).
4. Re-run the bench sanity checks (§1 above) against the new checkpoint/table before any arm motion.
5. Only then: hardware bring-up staging (§2) against the new checkpoint, starting again at `speed_scale=0.15`
   even though the arm itself was already bring-up-verified on 2026-08-22 — the release state changed, torque/
   velocity margins have to be re-read for real, not assumed carried over.

**Per-throw execution: `run_closed_loop_throws.py`.** One throw per invocation (matches this project's
"never skip a stage, e-stop in hand for every run" culture — no batch-confirm-once mode). Wraps
`run_hardware_throw.py`'s plan/precheck/throw path unchanged; adds a live console dashboard (target,
commanded speed, `PRECHECK: PASS/FAIL` **and** `release pos in safe box: True/False` as two separate lines,
per the R2-style trap this doc has already warned about once) and a structured append-only log
(`closed_loop_throw_log.jsonl` by default — one JSON object per throw: index, timestamp, target, commanded
speed, `speed_scale`, `q_release`/`qd_release`, precheck result, `ex.last_exec_stats`, `ball_id`,
`capture_file`, `landing_xy`).

```bash
python3 run_closed_loop_throws.py --log_path results_kinetic_chain_gen3_tcp/1 \
    --opt_pose throw_pose_table_tcp.npy --tool_offset_z 0.12 \
    --target 0.75 0.05 --throw_index 0 --ball_id tennis-01 \
    --arm --speed_scale 0.15 --confirm     # escalate 0.15 -> 0.30 -> 0.60 -> 1.00, same as §2
```

**Data flow — two decoupled streams, correlated offline, not live.** This mirrors `measure_landing.py`'s own
stated design ("a changed fitter can be re-run against a real throw from weeks ago"):

```
ARM side (this script)              CAMERA side (separate, already running)
  plan -> precheck -> confirm         throw_capture.py --out throws/ ...
  -> throw -> exec_stats                (ring-buffer, auto- or manually-triggered,
  -> append JSONL record                 one throw_XXX.npz per detected event)
        |                                        |
        `------------------.    .----------------'
                             v  v
              operator matches throw_index <-> throw_XXX.npz by time/order
                             |
              measure_landing.py --recording throws/throw_XXX.npz
                             --extrinsic calib/T_B_C.npz    (offline, anytime after)
                             |
              landing (x,y) written back into that throw's JSONL record
                             |
              ~10 throws with landing_xy filled in = the closed-loop dataset
              (HARDWARE_SETUP.md "close the loop" — feeds the real MC-PILOT
              model-learning update; **that ingestion/update script does not
              exist in this repo yet** — the dataset this pipeline produces is
              the input it will need, not the update itself)
```

**Camera readiness, as of 2026-08-31:** `throw_capture.py` (dual-IR ring-buffer recorder) works and the
2D-detect/stereo-triangulate stages are real-frame-validated (20 throws captured 2026-08-26,
`throw_003.npz` visually confirmed). **`calib/T_B_C.npz` now exists** (2026-08-31, via
`start_of_day.py` — see §0.2), so `measure_landing.py` and `landing_xy` are unblocked and
`--extrinsic calib/T_B_C.npz` can be passed for real. What has still never run on real data is
`perception/trajectory.py`'s `ransac_track`/`fit_ballistic`/`solve_impact` — i.e. the second half of
`measure_landing.py`. That is now purely a matter of pointing it at one of the existing recordings;
nothing blocks it.

**A calibration produced before 2026-08-31 is wrong by ~12 cm and must not be reused.** Both
calibration scripts built the chain off `GetMeasuredCartesianPose`, which reports the TOOL frame
(0.12 m out, the 2F-85's `tool_transform`), and composed the URDF's *flange*→camera offset onto it —
see §3 R8. Any `camera_extrinsics*.json` on disk from before that date carries the error. Re-run
`start_of_day.py`.
