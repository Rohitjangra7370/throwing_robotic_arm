"""
Dynamics-aware time allocation for the throw phase -- the "whip", solved.

WHAT THIS DOES AND DOES NOT BUY YOU
-----------------------------------
It does NOT raise the release speed.  For a rigid arm the release speed is set
entirely by the release CONFIGURATION and the joint-velocity limits, through
the direction-constrained LP in `simulation_class/release_solver.py`
(maximize s s.t. J q_dot = s*d, |q_dot_i| <= qd_max_i).  No trajectory shape,
no whip, no bang-bang profile beats that bound, because none of them changes
either J(q_release) or qd_max.  The often-quoted "the whip lets the wrist
exceed its own limits" is about ELASTIC or underactuated links, which this arm
does not have.  `CLAUDE.md` already records the same conclusion; this module
does not relitigate it.

What the whip DOES buy on a torque-limited arm is TIME and TORQUE MARGIN.  The
shipped planner hands every joint the same throw window and staggers their
starts on a fixed linear ramp (`stagger_frac = linspace(0, 0.5, n)`), which is
a guess, not an allocation: it takes no account of which joint's actuator is
actually saturating.  On the Gen3 the wrists have 9 Nm against the shoulder's
39 Nm, so a uniform window loads them very unevenly.  When the resulting peak
torque exceeds the limit, `plan_throw` simply stretches the WHOLE window by
1.2x and retries -- up to 2.07x on the shipped training runs (measured across
`results_tracking_error/tracking_error.npz`).  Stretching the window is what
raises the tracking error, because the trajectory then spends longer in the
region where the model-mismatch terms (unmodelled payload Coriolis,
50 Hz discretisation) integrate up.

So the optimization solved here is:

    minimize    dt_throw
    over        f_i in [0, f_max]        (per-joint start fraction)
    subject to  |tau(t)|  <= tau_max     for all t in the phase
                |qd(t)|   <= qd_max      (holds by construction)
                |qddd(t)| <= jerk_max    (if given)
                q(t) within joint limits
                q(dt_throw) = q_release, qd(dt_throw) = qd_release  (exactly)

Joint i's ramp runs over [f_i * dt_throw, dt_throw], so a large f_i means that
joint stays cocked and fires late and hard -- the proximal-to-distal
"summation of speed" sequencing (Senoo & Ishikawa, ICRA 2008), but with the
sequencing DERIVED from the actuator limits instead of assumed.

The inner problem (smallest feasible dt_throw for a given allocation) is solved
by bracketing + bisection rather than a 1.2x ladder; the outer problem is
gradient-free (Powell), because torque feasibility through PyBullet's inverse
dynamics has no usable analytic gradient here.

This is the scipy + PyBullet equivalent of the CasADi/Drake direct-transcription
formulation.  It is deliberately a TIME-ALLOCATION problem rather than a free
collocation over every knot: with q(T), qd(T) pinned by the release solver and
the shape fixed, the only remaining freedom that changes the dynamics is when
each joint spends its window -- and that keeps the "peak |qd| = qd_release by
construction" guarantee, which a free collocation would lose.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from throw_lab import shapes as sh


def _alloc_eval_factory(q_release, qd_release, stagger, local, shape):
    """(q, qd, qdd, qddd) sampler for one time allocation, and its windup pose."""
    q_w = q_release - qd_release * local * shape.S1

    def ev(t):
        tau = np.clip((t - stagger) / np.maximum(local, 1e-12), 0.0, 1.0)
        active = (t >= stagger) & (t <= stagger + local)
        return (
            q_w + qd_release * local * shape.Sint(tau),
            qd_release * shape.s(tau),
            np.where(active, qd_release * shape.ds(tau) / np.maximum(local, 1e-12), 0.0),
            np.where(
                active,
                qd_release * shape.dds(tau) / np.maximum(local, 1e-12) ** 2,
                0.0,
            ),
        )

    return ev, q_w


def worst_ratio(chk, q_release, qd_release, frac, dt_throw, shape, n_samples=60):
    """Worst constraint ratio over the throw phase for one (frac, dt_throw)."""
    stagger = np.asarray(frac, dtype=float) * dt_throw
    local = dt_throw - stagger
    ev, _ = _alloc_eval_factory(q_release, qd_release, stagger, local, shape)
    return chk.check_span(ev, 0.0, dt_throw, n_samples=n_samples)


def min_feasible_duration(
    chk,
    q_release,
    qd_release,
    frac,
    shape,
    lo=0.40,
    hi_cap=12.0,
    tol=0.01,
    n_samples=60,
):
    """Smallest dt_throw whose whole-phase constraint ratios are all <= 1.

    Bracket by doubling, then bisect to `tol` seconds.  Returns
    (duration, PhaseCheck) or (None, worst_check) when even `hi_cap` fails --
    which happens when GRAVITY alone exceeds a joint's torque limit somewhere
    on the path, a duration-independent infeasibility.

    `lo` IS A CONTROL-BANDWIDTH FLOOR, NOT A CONVENIENCE.  Torque feasibility is
    checked in continuous time, so the optimizer will happily drive the throw
    window down until torque binds -- measured, that lands at dt_throw = 0.075 s
    for the Gen3, which is 3.75 ticks of the 50 Hz control loop.  At that point
    the check no longer predicts anything: the closed loop is a computed-torque
    PD with kp = 400, kd = 60, i.e. omega_n = 20 rad/s and zeta = 1.5, so its
    settling time constant is 1/(zeta*omega_n) = 33 ms.  A trajectory that ramps
    its acceleration faster than a few of those cannot be tracked, and the arm
    OVERSHOOTS it (measured: 112% of planned release speed and 10.6 cm of
    landing error at 0.075 s).  0.40 s is ~12 time constants and 20 control
    ticks.  Raise it, never lower it, unless the control rate goes up too.
    """
    hi = float(lo)
    last = None
    doubled = False
    while hi <= hi_cap:
        c = worst_ratio(chk, q_release, qd_release, frac, hi, shape, n_samples)
        last = c
        if c.ok:
            break
        hi *= 2.0
        doubled = True
    else:
        return None, last
    # The bracket's lower end is the FLOOR when the floor itself was feasible.
    # Setting it to 0 here (an earlier version did) let the bisection walk
    # straight through the control-bandwidth floor -- measured, it returned
    # dt_throw = 0.019 s, one control step, and the resulting "throw" left the
    # ball at 0.3% of the planned speed pointing 136 deg the wrong way.
    low = hi / 2.0 if doubled else lo
    while hi - low > tol:
        mid = 0.5 * (low + hi)
        c = worst_ratio(chk, q_release, qd_release, frac, mid, shape, n_samples)
        if c.ok:
            hi, last = mid, c
        else:
            low = mid
    return hi, last


def optimize_allocation(
    chk,
    q_release,
    qd_release,
    shape,
    frac0=None,
    f_max=0.85,
    n_samples=60,
    maxiter=40,
    min_dur=0.40,
    verbose=False,
):
    """Powell search over per-joint start fractions, minimizing time-to-release.

    Only joints that actually carry release velocity are optimized; the frozen
    roll/twist joints have qd_release = 0, so their start time cannot affect
    anything and including them would just add flat directions to the search.

    Returns dict with `frac`, `dt_throw`, `check`, `n_eval`, `frac0_dt`.
    """
    qd_release = np.asarray(qd_release, dtype=float)
    n = len(qd_release)
    movers = np.flatnonzero(np.abs(qd_release) > 1e-9)
    if frac0 is None:
        frac0 = np.linspace(0.0, 0.5, n)          # the shipped linear stagger
    frac0 = np.asarray(frac0, dtype=float)

    base_dur, base_chk = min_feasible_duration(
        chk, q_release, qd_release, frac0, shape, lo=min_dur, n_samples=n_samples
    )
    if base_dur is None:
        raise RuntimeError(
            f"baseline allocation infeasible at every duration up to the cap "
            f"({base_chk}) -- the release pose itself is torque-infeasible"
        )

    n_eval = [0]
    cache = {}

    def objective(x):
        key = tuple(np.round(x, 4))
        if key in cache:
            return cache[key]
        frac = frac0.copy()
        frac[movers] = np.clip(x, 0.0, f_max)
        pen = float(np.sum(np.maximum(0.0, np.abs(x) - f_max) ** 2)) * 100.0
        dur, c = min_feasible_duration(
            chk, q_release, qd_release, frac, shape, lo=min_dur, n_samples=n_samples
        )
        n_eval[0] += 1
        # Tie-break on torque margin.  When the duration is pinned at the
        # control-bandwidth floor -- which it is whenever torque is NOT the
        # binding constraint, i.e. every kinetic-chain release state measured
        # on this arm -- the time objective is exactly flat and Powell would
        # otherwise return whichever arbitrary allocation it happened to try
        # last (observed: one that raised peak torque 0.21 -> 0.99 for no gain).
        val = (base_dur * 4.0 if dur is None
               else dur + 0.02 * c.worst) + pen
        cache[key] = val
        if verbose:
            print(f"    f={np.round(frac[movers], 3)} -> dt={val:.4f}s", flush=True)
        return val

    x0 = np.clip(frac0[movers], 0.0, f_max)
    res = minimize(
        objective,
        x0,
        method="Powell",
        bounds=[(0.0, f_max)] * len(movers),
        options={"maxiter": maxiter, "xtol": 1e-2, "ftol": 1e-3},
    )
    frac = frac0.copy()
    frac[movers] = np.clip(res.x, 0.0, f_max)
    dur, chk_out = min_feasible_duration(
        chk, q_release, qd_release, frac, shape, lo=min_dur, n_samples=n_samples
    )
    if dur is None or dur > base_dur:
        # never ship a worse allocation than the one we started from
        frac, dur, chk_out = frac0, base_dur, base_chk
    stagger = frac * dur
    return {
        "frac": frac,
        "stagger": stagger,
        "local_dur": dur - stagger,
        "dt_throw": float(dur),
        "check": chk_out,
        "n_eval": n_eval[0],
        "frac0_dt": float(base_dur),
        "frac0_check": base_chk,
        "shape": shape.name,
        "min_dur": float(min_dur),
    }


def report(alloc):
    return (
        f"whip allocation: dt_throw {alloc['frac0_dt']:.3f}s -> "
        f"{alloc['dt_throw']:.3f}s "
        f"({100.0 * (alloc['dt_throw'] / alloc['frac0_dt'] - 1.0):+.1f}%), "
        f"tau {alloc['frac0_check'].tau_ratio:.3f} -> {alloc['check'].tau_ratio:.3f}, "
        f"start frac {np.round(alloc['frac'], 3)}, {alloc['n_eval']} inner solves"
    )


# --------------------------------------------------------------------------
# closed-loop optimization: optimize the MEASURED outcome, not a proxy
# --------------------------------------------------------------------------
def optimize_closed_loop(
    harness,
    planner,
    q_release,
    qd_release,
    shape,
    throw_dur0=1.1,
    f_max=0.85,
    dur_bounds=(0.4, 2.6),
    jitter_weight=1.0,
    maxiter=8,
    maxfev=240,
    windup_shape="trap_vel",
    windup_kw=None,
    verbose=False,
):
    """Powell search over (dt_throw, per-joint start fractions) minimizing the
    landing error the arm ACTUALLY produces, plus its release-timing spread.

        J(x) = |land(0) - land_ideal| + w * 0.5 * |land(+1 step) - land(-1 step)|

    Both terms come out of `ThrowHarness`, i.e. out of the same closed-loop
    torque controller and the same 50 Hz world the trainer uses, with the ball
    released by removing the grip constraint.  Nothing here is a feasibility
    proxy: `min_feasible_duration`'s time-optimal objective is only a stand-in
    for accuracy, and on this arm it is a BAD one -- torque never binds at the
    kinetic-chain release states (peak ratio 0.18-0.49 across a 0.2-2.4 s
    duration sweep), so minimizing time just walks the window down to the
    control-bandwidth floor, where the arm overshoots its own plan.

    The second term is what buys down the hardware error budget directly: the
    real Gen3 releases through a gripper with 67.9 +- 6.4 ms latency on a 25 ms
    command quantum, so the landing spread over +-1 control step is a direct
    proxy for the landing spread the real arm will show.

    Returns dict with `dt_throw`, `frac`, `plan`, `J`, `J0`, `n_eval`.
    """
    qd_release = np.asarray(qd_release, dtype=float)
    n = len(qd_release)
    movers = np.flatnonzero(np.abs(qd_release) > 1e-9)
    frac0 = np.linspace(0.0, 0.5, n)
    n_eval = [0]

    def build(dur, frac):
        st = np.asarray(frac, dtype=float) * dur
        return planner.plan(
            q_release, qd_release,
            dt_throw=dur, throw_shape=shape,
            windup_shape=windup_shape, windup_kw=windup_kw or {"beta": 0.10},
            stagger=st, local_dur=dur - st,
            label="whip_cl",
        )

    def score(dur, frac):
        n_eval[0] += 1
        try:
            plan = build(dur, frac)
        except RuntimeError:
            return 1e3, None
        res = harness.run(plan)
        ideal, _ = harness.free_flight(res.release_pos_planned, res.v_planned)
        if not (np.all(np.isfinite(res.land_xy)) and np.all(np.isfinite(ideal))):
            return 1e3, plan
        err = float(np.linalg.norm(res.land_xy - ideal))
        hi = harness.run(plan, release_bias_steps=1)
        lo = harness.run(plan, release_bias_steps=-1)
        if np.all(np.isfinite(hi.land_xy)) and np.all(np.isfinite(lo.land_xy)):
            spread = 0.5 * float(np.linalg.norm(hi.land_xy - lo.land_xy))
        else:
            spread = 1.0
        return err + jitter_weight * spread, plan

    def objective(x):
        dur = float(np.clip(x[0], *dur_bounds))
        frac = frac0.copy()
        frac[movers] = np.clip(x[1:], 0.0, f_max)
        j, _ = score(dur, frac)
        if verbose:
            print(f"    dt={dur:.3f} f={np.round(frac[movers], 3)} -> J={j:.5f}",
                  flush=True)
        return j

    x0 = np.concatenate([[float(throw_dur0)], np.clip(frac0[movers], 0.0, f_max)])
    j0, _ = score(float(throw_dur0), frac0)
    res = minimize(
        objective, x0, method="Powell",
        bounds=[dur_bounds] + [(0.0, f_max)] * len(movers),
        options={"maxiter": maxiter, "maxfev": maxfev,
                 "xtol": 5e-3, "ftol": 1e-4},
    )
    dur = float(np.clip(res.x[0], *dur_bounds))
    frac = frac0.copy()
    frac[movers] = np.clip(res.x[1:], 0.0, f_max)
    j, plan = score(dur, frac)
    if j > j0:                       # never ship worse than the starting point
        dur, frac = float(throw_dur0), frac0
        j, plan = score(dur, frac)
    return {
        "dt_throw": dur,
        "frac": frac,
        "plan": plan,
        "J": float(j),
        "J0": float(j0),
        "n_eval": n_eval[0],
        "shape": shape.name,
    }


def report_closed_loop(alloc):
    return (
        f"closed-loop whip: J {100 * alloc['J0']:.2f}cm -> {100 * alloc['J']:.2f}cm "
        f"({100.0 * (alloc['J'] / max(alloc['J0'], 1e-9) - 1.0):+.1f}%), "
        f"dt_throw {alloc['dt_throw']:.3f}s, "
        f"start frac {np.round(alloc['frac'], 3)}, {alloc['n_eval']} harness evals"
    )
