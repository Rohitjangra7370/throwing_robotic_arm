# throw_lab — trajectory-profile research sandbox

Separate, self-contained study of **how the arm should get to the release
state**. Nothing in the shipped `mc-pilot-pybullet` pipeline imports this
package; it imports the shipped code (`ArmController`,
`OptimizedReleaseSolver`, the Eq. 35 drag model) read-only, so every number
below comes out of the same arm model, the same computed-torque controller and
the same 50 Hz PyBullet world the trainer uses.

```bash
cd mc-pilot-pybullet/
/usr/bin/python3 -m pytest throw_lab/tests -q          # 76 tests, ~5 s
/usr/bin/python3 -m throw_lab.bench  --quick --jitter  # profile comparison
/usr/bin/python3 -m throw_lab.bench  --release_mode legacy --t_w 0.4 --dt_throw 0.4
/usr/bin/python3 -m throw_lab.sweep                    # throw-window duration sweep
/usr/bin/python3 -m throw_lab.bench  --whip --jitter   # + both whip optimizers
```

| file | what it is |
|---|---|
| `shapes.py` | normalized acceleration/velocity shapes + closed-form peaks. Pure math, no PyBullet. |
| `feasibility.py` | dense whole-phase torque/velocity/jerk/limit checking, payload-aware |
| `planner.py` | shape-pluggable 4-phase planner (windup → throw → brake → return) |
| `dynopt.py` | the "whip": per-joint time allocation, two objectives (min-time, closed-loop) |
| `harness.py` | closed-loop executor + measurement (release velocity, landing, saturation) |
| `bench.py` | profile comparison CLI |
| `sweep.py` | throw-window duration sweep |
| `results/` | saved `.npz` from the runs quoted below |

---

## 0. The premise, corrected before anything else

The starting hypothesis was: *cubic splines have infinite jerk; go quintic for
accuracy or bang-bang/S-curve for speed; then use optimal control to find a
whip.* Two parts of that survive contact with this arm, one does not.

**Confirmed — the cubics really are C¹, not C².** `_cubic_rest_to_rest` starts
at `qdd = 6·Δ/T²` from a standing start, and `_cubic_to_velocity` starts at
`2·a₂ ≠ 0` immediately after the windup ends at `-6·Δ/T²`. Measured
acceleration step across the phase joins: **3.5 rad/s² (table release) and
28.2 rad/s² (legacy release)**, versus exactly 0 for every jerk-limited shape
here (`bench.py`'s `dQddJoin` column, and
`test_planner.py::test_smooth_shapes_have_no_acceleration_step_at_any_join`).

**Corrected — the shipped throw phase is not really a cubic.** With
`monotonic_windup=True` (which is what the kinetic-chain/`opt_pose` path uses)
the planner cocks back by exactly `Δq = q̇ₑ·T/2`, and at that cock distance the
cubic term vanishes identically:

```
a3 = (qd_e*T - 2*dq) / T**3 = (qd_e*T - qd_e*T) / T**3 = 0
```

so the "piecewise cubic" throw is a **constant-acceleration ramp**. That is
proved in `test_shapes.py::test_const_accel_reproduces_shipped_cubic_to_velocity`
and it matters twice over: constant acceleration is *torque-optimal* for a
fixed (Δq, q̇ₑ, T) triple, and it is the *worst possible* shape for release
timing, because the arm is still at full acceleration at the instant the ball
leaves.

**Refuted for this arm — "S-curve extracts the maximum velocity out of your
joint limits."** That premise assumes torque is the binding constraint. On the
Gen3 at the trained release states it is not. Across a 0.2 s → 2.4 s
throw-window sweep at three azimuths, peak torque ratio stayed in
**0.18–0.49** while the joint-*velocity* ratio sat pinned at **0.92**
(`sweep.py`, `results/duration_sweep.npz`). Release speed is capped by the
direction-constrained LP in `release_solver.py`
(`max s s.t. J q̇ = s·d, |q̇ᵢ| ≤ q̇ᵢᵐᵃˣ`), and no profile changes `J(q_release)`
or `q̇ᵐᵃˣ`. **No trajectory shape in this study raises the release speed
ceiling, and none can.** What they change is how much of that ceiling the arm
actually realizes, and how robustly.

---

## 1. What the profiles actually buy

Both benchmarks below plan to the **same** `(q_release, q̇_release)` from the
same `OptimizedReleaseSolver`, so rows differ only in the trajectory that gets
there.

### 1a. Legacy IK + pinv release — `results/bench_legacy.npz`

Eight release states (3 speeds × 3 azimuths, one refused — see below),
`t_w = 0.4`, `dt_throw = 0.4` — the same regime `measure_tracking_error.py`
sampled, where the shipped answer is 90.3 % of commanded speed over 225
throws. This lab reproduces it at **91.98 %**, which is the cross-check that
the harness is measuring the same thing the shipped pipeline does.

| profile | speed % | dir err | landErr | dv per 20 ms step | dLand per step | Δq̈ at joins |
|---|---|---|---|---|---|---|
| **SHIPPED cubic** | **91.98 %** | 1.72° | 0.71 cm | **21.32 cm/s** | **6.10 cm** | 28.2 |
| lab const-accel | 97.19 % | 0.72° | 0.23 cm | 4.18 cm/s | 2.25 cm | 7.9 |
| lab min-jerk (quintic) | 97.15 % | 0.65° | 0.23 cm | 0.28 cm/s | **0.95 cm** | **0** |
| lab trap-accel β=0.25 | 97.88 % | 0.65° | **0.16 cm** | **0.14 cm/s** | 0.98 cm | **0** |
| lab trap-accel β=0.10 | **99.66 %** | 0.67° | 0.40 cm | 0.22 cm/s | 1.07 cm | **0** |
| lab plateau f=0.15 | 97.25 % | 0.64° | 0.24 cm | 0.12 cm/s | 1.03 cm | **0** |

Single-state whip rows at 1.0 m/s (`results/bench_legacy_whip.npz`):
closed-loop **98.08 %, 0.00 cm** landing error at `t_r = 0.96 s`; min-time
**112.55 %, 2.96 cm** — the same overshoot failure as in §3.

### 1b. Kinetic-chain `opt_pose` release — `results/bench_table.npz`

Six release states (speeds 1.2 / 1.5 m/s × azimuths −30 / 0 / +30°),
`t_w = 0.5`, `dt_throw = 1.1` — the hardware-facing path, planning through
`throw_pose_table.npy`.

| profile | speed % | dir err | landErr | dv per step | dLand per step | t_r | Δq̈ at joins |
|---|---|---|---|---|---|---|---|
| **SHIPPED cubic** | **97.75 %** | 2.36° | **2.16 cm** | **3.37 cm/s** | 4.03 cm | 4.51 s | 3.54 |
| lab const-accel | 98.97 % | 1.25° | 0.19 cm | 3.36 cm/s | 3.97 cm | 4.30 s | 4.30 |
| lab min-jerk (quintic) | 99.48 % | 1.20° | 0.09 cm | 0.37 cm/s | 2.46 cm | 5.11 s | **0** |
| lab trap-accel β=0.25 | 99.37 % | 1.21° | 0.08 cm | 0.40 cm/s | 2.47 cm | 3.95 s | **0** |
| lab trap-accel β=0.10 | 99.27 % | 1.22° | **0.07 cm** | 0.39 cm/s | 2.47 cm | **3.48 s** | **0** |
| lab plateau f=0.15 | **99.78 %** | 1.21° | 0.18 cm | **0.34 cm/s** | 2.46 cm | 3.48 s | **0** |
| lab whip (closed-loop) | 99.39 % | 1.21° | **0.00 cm** | 0.40 cm/s | 2.47 cm | 3.52 s | **0** |
| lab whip (min-time) | 111.37 % | 1.91° | 9.46 cm | 4.95 cm/s | 4.64 cm | 2.78 s | 0 |

Against the shipped baseline, `trap_accel(β=0.25)` gives **27× lower landing
error** (2.16 → 0.08 cm), **8.4× lower release-timing sensitivity**
(3.37 → 0.40 cm/s), 1.6 % more of the commanded release speed, half the
direction error, and reaches release **0.56 s sooner**. Separating the causes:
const-accel alone (i.e. just the planner rewrite — grid-snapped release,
per-joint `min_tw`, brake+return) accounts for the landing-error and
direction-error collapse; the jerk-limited *shape* is what buys the
timing-sensitivity column, and const-accel does not move it at all
(3.36 vs 3.37 cm/s).

**The one that matters for hardware is `dv/step`.** The real Gen3 releases
through a gripper with 67.9 ± 6.4 ms latency on a 25 ms command quantum, so
the release instant is uncertain by roughly one 50 Hz control step. A profile
still accelerating at release converts that uncertainty straight into release
speed error; one that arrives with `q̈ = 0` does not. Measured reduction:
**8.4× (table, 3.37 → 0.40 cm/s) to 152× (legacy, 21.32 → 0.14 cm/s)**, and it is a first-order-vs-second-order
difference, not a tuning win — see `shapes.py`'s `ds_at_release`.

**Recommendation: `trap_accel` with β = 0.25** — `min_jerk` costs 1.875×
peak acceleration for the same (q̇ₑ, T), β = 0.25 costs only 1.333× and gets
the same zero release acceleration. β = 0.10 is faster still but its peak jerk
is 2 100 rad/s³ against β = 0.25's 400, which is not something to hand a real
harmonic drive without a jerk limit set.

---

## 2. Two structural bugs found on the way

### 2a. The release step fires a whole control step early, sometimes two

`model_pybullet.py:283` computes `release_step = int(t_r_actual / dt_phys)`.
That floors — so it already releases up to a full 20 ms early — and it is
float-fragile on top: for `t_r = 4.18`, `t_r/0.02` evaluates to
`208.99999999999997` and `int()` drops a *further* whole step. Measured cost
of that single lost step at one release state: **2.5° of direction error and
2.6 cm of landing error**, purely because the shipped profile is at full
acceleration when the ball leaves.

`planner.py` snaps `t_r` onto the control grid (growing the windup, never
shortening it), and `harness.py` floors with a `1e-9` epsilon so a profile
comparison compares profiles rather than rounding luck. Regression:
`test_planner.py::test_release_instant_lands_on_the_control_grid`.

### 2b. The follow-through has no velocity bound; brake-then-return does

`plan_throw` sends a single cubic from `(q_release, q̇_release)` straight to
`q_neutral` at rest. That cubic has to cover a fixed distance in a fixed time
starting at full release speed, so a short window forces a mid-phase velocity
blow-up (the code's own comment records 5.6× `qd_max` at the nominal 0.6 s)
while a long one raises peak gravity torque — hence the non-monotone
feasible band and the candidate-ladder *scan* it needs.

`throw_lab` splits it: **brake** runs the same shape time-reversed, `q̇_release
→ 0`, going wherever that takes the arm, so `|q̇| ≤ |q̇_release|` holds *by
construction*; **return** is then an ordinary rest-to-rest move. Both are
monotone in duration, so both solve by growth instead of a scan. Regression:
`test_planner.py::test_brake_phase_never_exceeds_release_velocity`.

This is not theoretical. Sweeping the legacy release states, **`plan_throw`
raised on 1 of 9** — `Follow-through infeasible over the scanned durations
(best: torque 0.35x, velocity 1.00x of limit)` — i.e. it refused by a rounding
margin, on the velocity ratio the phase has no bound on. Two of the other
eight needed the horizon walked before it would plan them, reproducing the
shipped code's own comment that `T=2.2` succeeds where `T=2.0` fails for the
same release state. `bench.py::legacy_release_state` walks that ladder and
`bench.py` records refusals rather than crashing.

One caveat on that count, stated plainly: in legacy mode the release state
itself comes *out* of `plan_throw`, so a refusal skips the whole state and the
lab profiles were never evaluated on it. The claim is "the shipped planner
refused a release state it generated", **not** "the lab planner handled a state
the shipped one couldn't" — that comparison was not run.

Two smaller ones, not fixed anywhere, just measured:

- **The shipped windup is over-stretched by up to 14 %.** `plan_throw`'s
  `min_tw = 1.5*span/np.min(self._qd_max)` divides by the slowest joint's limit
  regardless of which joint actually moves the furthest. On the Gen3 the
  windup is dominated by the base joint (`qd_max` 1.3963) while `min()` returns
  a wrist's 1.2218 — a 1.143× penalty. `throw_lab` checks the real per-joint
  ratio instead and gets `t_r` 4.90 s → 4.54 s on the same trajectory family.
- **The planner's torque check omits the PD correction and the payload.**
  `_throw_peak_torque_ratio` calls `inverse_dynamics`, which sees the arm URDF
  only, while `step()` adds `kp·e + kd·ė` *and* a `Jᵀm(a_ee − g)` payload term
  and clips the **sum**. Measured realized peak torque ratio ran 0.65–0.84
  where the plan-time check said ~0.40. It never saturated here, so nothing
  broke — but the margin is roughly half what the check reports.
  `feasibility.py` includes the payload term; the PD term is state-dependent
  and is only observable in the harness.

---

## 3. The whip, solved two ways — and one of them is a negative result

`dynopt.py` optimizes the per-joint time allocation of the throw phase: joint
*i*'s ramp runs over `[fᵢ·dt_throw, dt_throw]`, so a large `fᵢ` means that
joint stays cocked and fires late and hard. The shipped planner assumes
`stagger_frac = linspace(0, 0.5, n)` — a guess, taking no account of which
actuator is saturating.

**Objective A, minimize time-to-release subject to torque/velocity/jerk.**
This is the textbook time-optimal formulation (the scipy + PyBullet equivalent
of the CasADi/Drake direct-transcription setup), and on this arm it is a
**negative result**. Because torque never binds, the optimizer walks the throw
window straight down to whatever floor it is given, and the closed loop then
fails to track it:

| `dt_throw` floor | realized speed | landing error | peak τ ratio | |
|---|---|---|---|---|
| none — bisection reached 0.019 s | 0.31 % | 17.8 cm | 0.60 | single state |
| 0.075 s | 112.2 % | 10.6 cm | 1.15 | single state |
| 0.40 s = 20 control steps | **111.4 %** | **9.5 cm** | **1.31** | 6-state mean |
| 1.10 s (the shipped window) | 99.4 % | 0.08 cm | 0.63 | 6-state mean |

The arm **overshoots** its own plan — >100 % of planned release speed, and a
realized torque ratio above 1.0 that the plan-time check never saw, because
the PD correction is on top of the feedforward. The closed loop here is
computed-torque with `kp = 400, kd = 60`, i.e. `ωₙ = 20 rad/s, ζ = 1.5`, so
its time constant is `1/(ζωₙ) = 33 ms`; a trajectory that ramps its
acceleration faster than a few of those simply is not trackable. **Continuous-
time feasibility is not a sufficient condition for a sampled closed loop**, and
`min_feasible_duration` now carries a hard control-bandwidth floor with that
reasoning inline. (An earlier version had a bisection bug that bracketed
straight through the floor and returned `dt_throw = 0.019 s` — one control
step. Regression:
`test_planner.py::test_min_feasible_duration_respects_the_bandwidth_floor`.)

**Objective B, minimize the measured landing error.**
`optimize_closed_loop` optimizes `(dt_throw, f)` against

```
J = |land − land_ideal| + |land(+1 step) − land(−1 step)| / 2
```

with every term produced by actually running the throw through the harness —
no proxy. This works. Over the same six kinetic-chain release states: landing error
**2.16 cm (shipped) → 0.00 cm**, time-to-release **4.51 s → 3.52 s**, at
99.39 % speed fidelity and 0.40 cm/s timing sensitivity — i.e. it matches the
best fixed shape on every axis and wins outright on landing error, for ~240
harness evaluations per release state. On the legacy release states it lands
98.08 % / 0.00 cm at `t_r = 0.96 s`.

**The interesting structural finding**: when torque *is* made the binding
constraint, the optimizer's preferred stagger is **not** monotone
proximal-to-distal. On the Gen3 the wrists have 9 Nm against the shoulder's
39 Nm, so the torque-poor distal joints need the *longest* window, not the
shortest — the opposite of the "cock the wrist and fire it late" intuition
the linear stagger encodes. The whip story is about *elastic* or underactuated
links; a rigid arm with weak distal actuators wants the reverse.

---

## 4. Where the real error actually is (context, not this study's result)

Keep the numbers above in proportion. The dominant sim-to-real term for this
arm is **not** the trajectory profile: it is the Robotiq 2F-85 tool offset that
neither `release_solver.py` nor the sim has ever modelled. The firmware's own
`ControlConfig.GetToolConfiguration()` reports `tool_transform = (0, 0, 0.12) m`,
and at the trained release state `|ω| = 3.25 rad/s`, so
`v_true = v_flange + ω × r_offset` is **0.39 m/s — 26 % of the release speed**,
and 12 cm of release position. See `CLAUDE.md`'s TCP blocker.

That is why `bench.py` reports `|ω|` at release in every row. It is not a
trajectory-quality metric; it is the multiplier on that offset. Worth knowing:
the shape profiles leave it essentially alone (2.85 → 2.89-2.91 rad/s over the
six table states, since `q̇_release` is fixed by the pose table), and so does
the closed-loop whip (2.90), but the **min-time whip pushes it to 3.22 rad/s,
+11 %** — i.e. the naive time-optimal throw makes the largest error term in the
system worse while gaining nothing. At one 1.5 m/s state it reached 3.82 rad/s
against the shipped 3.20, +19 %.

Fixing the TCP offset is worth ~10× more than anything in this directory. The
profile work is still worth landing, because it is the difference between a
0.13 cm/s and a 21 cm/s release-timing sensitivity, and that error does not go
away when the TCP offset is fixed.

---

## 5. If any of this is to be adopted

Nothing here touches the shipped pipeline; adoption would mean porting into
`arm_controller.py`. In descending order of value-per-risk:

1. **Snap `t_r` to the control grid** and epsilon the floor in
   `model_pybullet.py:283`. Two lines, removes a systematic and occasionally
   doubled release-timing loss. No trajectory change.
2. **Swap the throw phase to `trap_accel(β=0.25)`.** Same cock-back pose, same
   duration, same release state — every symmetric shape integrates to ½, which
   is what makes this a drop-in. Buys the 8.4–152× timing-sensitivity reduction
   and C² joins.
3. **Split the follow-through into brake + return.** Removes the scan and gives
   an a-priori velocity bound on the phase that has historically shipped
   violations.
4. **Fix `min_tw`'s `np.min(qd_max)`** to the per-joint ratio.
5. Leave the whip alone unless the arm becomes torque-bound. If it does, use
   the closed-loop objective, never min-time.

Everything above is measured on `kinova_gen3_dyn` in sim, at 50 Hz, with the
`throw_pose_table.npy` release states. **No part of it has run on the real
arm**, and the hardware control rate is 40 Hz (`SendJointSpeedsCommand`), not
50 — so the control-bandwidth floor is *tighter* there, not looser.
