# Hardware throw session: live tracking, guided throw cycle, real-data model update — design

_2026-08-31. Depends on `2026-08-25-ball-tracking-design.md` (the perception pipeline) and on the
extrinsic calibration completed 2026-08-31 (`calib/T_B_C.npz`, `perception/wrist_chain.py`)._

## Goal

One application that takes the rig from cold to a real MC-PILOT model update:

1. run the start-of-day checks and calibration, GO/NO-GO;
2. show a live annotated view of the ball, its path, and its computed landing point;
3. drive N real throws through a guided cycle — pick up, verify grasp, plan, confirm, throw,
   measure, log, reload — with the operator replacing the ball between throws;
4. on a button, update the model on that real data, report what it learned, and only then, on a
   second button, re-optimize the policy into a new checkpoint.

Today every piece of this except (4) exists as a separate script that has to be run by hand in the
right order with the right flags, and the pieces cannot share the camera.

## The constraint that shapes the architecture

**The D435i can be opened by exactly one process.** That single fact is why
`run_closed_loop_throws.py` and `throw_capture.py` are currently decoupled through files: the
capture process owns the camera and the throw process polls a directory for what it wrote.

This design instead makes the session app own the camera for the whole session. The payoff is not
tidiness, it is correctness: **the app knows the exact release instant** (it issued the throw), so
it records a window around that instant instead of inferring a throw from a disparity gate. The
current range-gated auto-trigger has to guess from geometry alone whether a moving blob is a thrown
ball; an app that just threw the ball does not have to guess.

The cost is honest and worth stating: one process means a camera fault ends the session. Per-throw
recordings are written as they happen, so a crash costs the remaining throws, never the logged ones.

## What the evidence says the model can actually learn

This is the part that changes what is worth building, so it is settled here rather than after ten
throws.

**Flight aerodynamics are below the noise floor.** Tennis ball, m = 57.7 g, r = 32.7 mm, release
≈ 1.44 m/s, flight ≈ 0.51 s:

| Quantity | Value |
|---|---|
| drag deceleration, `½ρC_dAv²/m` at 1.5 m/s (C_d ≈ 0.5) | 0.039 m/s² |
| resulting deviation from a pure-gravity trajectory over the flight | **≈ 5 mm** |
| extrinsic repeatability (measured, 5 solves, static rig) | **18 mm** |
| stereo triangulation noise at ~1.6 m | ≈ 10 mm |

So a GP trained on these flights cannot distinguish drag from measurement noise. It will learn
gravity, which the propagation already assumes.

**Release error is well above the noise floor**, and is where the sim-to-real gap actually lives:

| Term | Value | Source |
|---|---|---|
| 25 ms high-level command quantisation | 2.9–3.7 cm of landing error | `HARDWARE_RUNBOOK.md` R0 |
| gripper-latency residual after compensation | ≈ 1.0 cm | R3 |

`measure_landing()` already returns the ball's **measured** release state `p0`, `v0` in the base
frame. Commanded-vs-measured `v0` is therefore directly observable and is 6–7× the noise floor.

**Consequence for the design:** do both, and report them separately with their uncertainties. The
flight GP is run faithfully so that its being a no-op is *demonstrated* rather than assumed — that
is a real result, and the honest version of "we closed the loop". The release model is where a
usable correction will come from.

A high-drag ball (whiffle, 4 g / 60 mm) would put drag far above the noise floor and is the natural
follow-up, but it is a different drag regime and needs its own trained checkpoint first. Out of
scope here; recorded so it is not forgotten.

## 1. Module boundaries

Two new files. The split is not cosmetic: everything in the second file must be testable without a
window, an arm, or a camera.

| File | Contains | Depends on |
|---|---|---|
| `hardware_session.py` | Tk dashboard, camera thread + live overlay window, worker thread, session state machine | everything below |
| `hardware_learning.py` | trajectory resampling, GP ingestion, release-model fit, above-noise statistic, target spread | numpy, torch, the existing model/policy classes |

Everything else is **imported and called, never reimplemented**: `start_of_day.py` (stage functions),
`pickup_and_lift.pickup_and_lift`, `run_hardware_throw.py` (planning, precheck, execution),
`run_closed_loop_throws.py` (`build_throw_record`, `append_log`, `format_status_dashboard`),
`measure_landing.measure_landing`, `perception/ir_capture.py`, `perception/ball_track.py`,
`perception/trajectory.py`. This repo's standing rule — one implementation of the release solver,
one of the calibration chain — extends to these.

## 2. Session state machine

```
COLD ──start-of-day──▶ CALIBRATED ──arm camera──▶ READY
                          │(NO-GO)                  │
                          ▼                         ▼
                       BLOCKED            ┌── THROW CYCLE ──┐
                                          │ pickup+grasp    │
                                          │ plan            │
                                          │ confirm         │
                                          │ throw+record    │
                                          │ measure+log     │
                                          │ reload prompt   │
                                          └────────┬────────┘
                                                   │ N throws logged
                                                   ▼
                                        MODEL UPDATED ──▶ POLICY RE-OPTIMIZED
```

Stage 0 runs **before** the camera thread starts. Calibration needs colour at 1920×1080; tracking
needs IR at 848×480/90 fps. Running them sequentially avoids stream reconfiguration mid-session and
lets `start_of_day.py` be reused exactly as it is, opening and closing the camera itself.

A NO-GO from stage 0 leaves the app in BLOCKED with the failing gates listed. The throw controls are
disabled, not merely discouraged.

## 3. Live view

The camera thread runs a continuous dual-IR ring buffer (`perception/ir_capture.py`) and draws an
OpenCV window at display rate, independent of the 90 fps capture. Overlay, in layers:

- per-frame detected blob in both IR images (`detect_candidates`), so a detection failure is visible
  as it happens rather than after the fit refuses;
- the triangulated 3D path of the current or last event, projected back into the left image;
- after a throw: the fitted arc, the solved impact point, and the landing `(x, y)` in base-frame
  coordinates with its σ, drawn at the impact pixel;
- a status line: armed / recording / measuring, and the last refusal reason if any.

Tk owns the main thread; the cv2 window lives entirely in the camera thread and is never touched
from Tk callbacks. Frames cross to the GUI as immutable snapshots through a single-slot queue, so a
slow GUI drops frames instead of stalling capture.

## 4. Throw cycle

Each step is a gate, and the gates already exist — this design wires them, it does not soften them.

1. **Pick up and verify grasp** — `pickup_and_lift`. A real ball stalls the gripper at 58–59 %
   closed; closing on nothing reads 99–100 %. Refuses to continue on a false grasp.
2. **Plan** — through `run_hardware_throw.py`'s planner. The dashboard shows `PRECHECK:` and
   `release pos in safe box:` as **two separate lines**, because `plan` prints PASS on the first
   even when the second is False.
3. **Confirm** — a checkbox that unchecks itself after every run. Re-affirmed per throw, never once
   per session.
4. **Throw and record** — the camera thread is told the release instant and keeps
   `[t_release − pre, t_release + post]` from the ring buffer.
5. **Measure and log** — `measure_landing` on that window; append one JSONL record.
6. **Reload prompt** — explicit "place the ball at the pickup pose" step before the next throw is
   armed.

**Speed-scale escalation is enforced, not advised:** the app refuses a speed_scale until a clean run
has been logged at the one below it (0.15 → 0.30 → 0.60 → 1.00), matching `HARDWARE_RUNBOOK.md` §2.

**A refusal from `measure_landing` does not abort the session.** The throw is logged with
`landing_xy: null` and the refusal reason, and the operator is told to re-throw. Per the runbook:
never hand-tune a threshold to make a bad track pass.

## 5. Data model

One append-only JSONL, the schema `run_closed_loop_throws.py` already writes, plus the fields the
vision now provides:

```
throw_index, timestamp, ball_id, target[2], commanded_speed, speed_scale,
q_release[7], qd_release[7], precheck_ok, release_in_box, exec_stats{...},
capture_file, landing_xy | null, sigma_xy_m, n_frames, n_inliers, rms_px,
measured_p0[3], measured_v0[3], refusal_reason | null
```

`measured_v0` is the new one that matters: it is what makes the release model fittable.

## 6. The model update (button 1)

Loads the checkpoint, then for each logged throw with a successful measurement:

**Flight GP.** Resample the **raw RANSAC-inlier triangulated points** onto the `Ts = 0.02` grid and
append via `model_learning.add_data`, with the same `Na` rotation augmentation the simulated path
applies in `MC_PILOT.get_data_from_system`.

> **Never resample the fitted parabola into the GP.** `fit_ballistic` fits a gravity-only model, so
> feeding its output back would teach the GP `Δv = g·dt` — the fit's own assumption, returned as if
> it were evidence. This repo has a documented history of exactly this shape of error (the
> model-belief trap), and it would be undetectable in the resulting cost curve.

**Release model.** Fit commanded release speed → measured `v0` (magnitude and direction), reporting
coefficients, residual σ, and the count behind them.

**Then report and stop.** No policy change. The report gives per-throw predicted-vs-measured landing,
the GP hyperparameter deltas, the release-model fit, and one explicit verdict: **is the flight GP's
learned deviation distinguishable from the measurement noise?**

Defined concretely, so it cannot drift into a judgement call: let `d_k` be the GP's predicted `Δv`
correction at sample `k` minus the pure-gravity `Δv`, i.e. the non-ballistic part the GP claims to
have found. Let `σ_k` be the per-sample velocity uncertainty implied by the position noise
(extrinsic 18 mm ⊕ triangulation 10 mm, propagated through the `Ts` differencing). The verdict is
`ABOVE NOISE` only if `RMS(d_k) > 2 · RMS(σ_k)`, reported with both numbers and the sample count.
On the evidence in the table above the expected answer is `BELOW NOISE`, and the report must say so
plainly rather than presenting a small number as a discovery. Both branches are tested (§8).

## 7. Policy re-optimization (button 2)

Enabled only after button 1 has run. Re-optimizes the policy against the updated model following the
pattern `adapt_policy_height.py` already establishes (reuse the GP, re-optimize the policy only),
and writes a **new** checkpoint directory. It never overwrites `results_kinetic_chain_gen3_tcp/1`.

The new checkpoint is not thrown automatically. It has to go back through stage 0 and the escalation
ladder like any other, because its release state may differ from the one the operator visually
verified for finger clearance.

## 8. Testing

Hardware and GUI paths stay untested, as everywhere else in this repo. Everything else is pure and
gets tests in `tests/test_hardware_learning.py`:

- resampling a synthetic track onto the `Ts` grid preserves a known velocity profile;
- **feeding a pure-parabola track produces a GP that has learned nothing beyond gravity** — the
  tautology guard, as a test rather than a comment;
- a synthetic track with an injected known drag *is* recovered when noise is set below it, and is
  correctly reported as indistinguishable when noise is set above it (both directions of the
  above-noise statistic);
- release-model fit recovers known gain/offset;
- JSONL round-trip, including a refused throw with `landing_xy: null`;
- target spread covers the band and stays inside it;
- escalation gate rejects a jump from 0.15 to 1.00 and accepts the ladder.

## 9. Deliberately out of scope

- The whiffle-ball high-drag regime (needs its own checkpoint).
- Automatic ball reloading — the operator replaces the ball, by design.
- Any change to `release_solver.py`, the pose tables, or the trained checkpoint.
- Multi-camera or moving-camera calibration.

## 10. Risks

| Risk | Handling |
|---|---|
| Camera fault ends the session | per-throw recordings are written as they happen; logged throws survive |
| Ten throws teach the flight GP nothing | expected, quantified above, and reported as a result rather than discovered late |
| Extrinsic drifts mid-session | stage 0 stores it; a re-run compares and warns at 3 cm / 5° |
| GUI thread contention with 90 fps capture | single-slot frame queue; GUI drops frames, capture never blocks |
| Operator fatigue across 10 throws | confirm checkbox resets every run; escalation ladder enforced in code |
