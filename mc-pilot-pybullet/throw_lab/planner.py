"""
Shape-pluggable 4-phase throw planner (research lab -- nothing in the shipped
mc-pilot-pybullet pipeline imports this).

    neutral --windup--> q_w --throw--> (q_release, qd_release)
                                  --brake--> q_stop --return--> neutral

The first three phases mirror `ArmController.plan_throw` so the two planners can
be compared on exactly the same release state, produced by the same
`OptimizedReleaseSolver`.  Two structural changes:

1. THE THROW PHASE IS PARAMETRISED BY AN ACCELERATION SHAPE, not by polynomial
   coefficients.  With `monotonic_windup=True` the shipped planner cocks back by
   dq = qd_release * T / 2, and at that exact cock distance
   `_cubic_to_velocity`'s cubic term vanishes identically
   (a3 = (qd_e*T - 2*dq)/T^3 = 0).  The shipped "cubic" throw IS a
   constant-acceleration ramp.  Writing it as qd(t) = qd_e * s(t/T) makes that
   explicit and turns every alternative into a one-line substitution.
   Every symmetric shape integrates to 1/2, so `const_accel`, `min_jerk` and
   `trap_accel` share the SAME cock-back pose and duration -- the comparison
   isolates the acceleration distribution and nothing else.

2. THE FOLLOW-THROUGH IS SPLIT INTO BRAKE + RETURN.  The shipped planner uses a
   single cubic from (q_release, qd_release) straight to q_neutral at rest, and
   that cubic has no a-priori velocity bound: it has to cover a fixed distance
   in a fixed time starting from full release speed, so making the window short
   forces a mid-phase velocity blow-up (measured 5.6x qd_max at the nominal
   0.6 s) while making it long raises peak gravity torque.  `plan_throw` has to
   SCAN a candidate ladder because the feasible band is narrow and
   non-monotone.  Braking first -- qd_release -> 0 through the same shape
   family, going wherever that takes the arm -- bounds |qd| by |qd_release| by
   construction, and the return leg is then an ordinary rest-to-rest move.
   Both sub-phases are monotone in their duration, so both can be solved by
   growth instead of a scan.

The release instant is SNAPPED to the control grid.  The executor releases at
`int(t_r / dt)`, so an unsnapped t_r silently releases up to one full control
step (20 ms at 50 Hz) early -- and since the shipped profile is still at full
acceleration at release, that is a systematic speed loss that varies from plan
to plan.  Leaving it unsnapped makes every cross-profile comparison a
comparison of rounding luck.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from throw_lab import shapes as sh
from throw_lab.feasibility import FeasibilityChecker, grow_until_feasible


@dataclass
class LabPlan:
    """An executable throw. `eval(t) -> (q, qd, qdd, qddd)` for any t >= 0."""

    q_neutral: np.ndarray
    q_windup: np.ndarray
    q_release: np.ndarray
    qd_release: np.ndarray
    qdd_release: np.ndarray
    q_stop: np.ndarray
    t_w: float
    t_r: float
    t_b: float                   # end of the braking phase
    T: float
    stagger: np.ndarray          # (n,) throw-phase start offset per joint [s]
    local_dur: np.ndarray        # (n,) throw-phase duration per joint [s]
    windup_shape: sh.RestShape
    throw_shape: sh.VelocityShape
    checks: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def dt_throw(self):
        return self.t_r - self.t_w

    @property
    def join_times(self):
        """Internal phase boundaries, where acceleration may step."""
        return (0.0, self.t_w, self.t_r, self.t_b)

    # -- evaluation -------------------------------------------------------
    def eval(self, t):
        t = float(t)
        if t <= self.t_w:
            return self._eval_windup(t)
        if t <= self.t_r:
            return self._eval_throw(t)
        if t <= self.t_b:
            return self._eval_brake(t - self.t_r)
        return self._eval_return(min(t - self.t_b, self.T - self.t_b))

    def _eval_windup(self, t):
        tw = max(self.t_w, 1e-12)
        tau = np.clip(t / tw, 0.0, 1.0)
        dq = self.q_windup - self.q_neutral
        s = self.windup_shape
        return (
            self.q_neutral + dq * s.Vint(tau),
            dq * s.v(tau) / tw,
            dq * s.dv(tau) / tw**2,
            dq * s.ddv(tau) / tw**3,
        )

    def _eval_throw(self, t):
        s = self.throw_shape
        rel = t - self.t_w
        loc = np.maximum(self.local_dur, 1e-12)
        tau = np.clip((rel - self.stagger) / loc, 0.0, 1.0)
        active = (rel >= self.stagger) & (rel <= self.stagger + self.local_dur)
        return (
            self.q_windup + self.qd_release * loc * s.Sint(tau),
            self.qd_release * s.s(tau),
            np.where(active, self.qd_release * s.ds(tau) / loc, 0.0),
            np.where(active, self.qd_release * s.dds(tau) / loc**2, 0.0),
        )

    def _eval_brake(self, dt):
        s = self.throw_shape
        d1 = max(self.t_b - self.t_r, 1e-12)
        tau = np.clip(dt / d1, 0.0, 1.0)
        # time-reversed shape: qd falls qd_release -> 0 monotonically, so the
        # phase peak velocity is |qd_release| by construction
        return (
            self.q_release + self.qd_release * d1 * (tau - s.Sint(tau)),
            self.qd_release * (1.0 - s.s(tau)),
            -self.qd_release * s.ds(tau) / d1,
            -self.qd_release * s.dds(tau) / d1**2,
        )

    def _eval_return(self, dt):
        d2 = max(self.T - self.t_b, 1e-12)
        tau = np.clip(dt / d2, 0.0, 1.0)
        dq = self.q_neutral - self.q_stop
        s = self.windup_shape
        return (
            self.q_stop + dq * s.Vint(tau),
            dq * s.v(tau) / d2,
            dq * s.dv(tau) / d2**2,
            dq * s.ddv(tau) / d2**3,
        )

    # -- reporting --------------------------------------------------------
    def summary(self):
        c = self.checks
        return (
            f"{self.meta.get('label', '?'):<20} "
            f"t_w {self.t_w:5.2f} t_throw {self.dt_throw:5.2f} "
            f"brake {self.t_b - self.t_r:4.2f} return {self.T - self.t_b:4.2f} | "
            + "  ".join(f"{k} {v.tau_ratio:.2f}/{v.qd_ratio:.2f}"
                        for k, v in c.items())
        )


class LabThrowPlanner:
    """Builds `LabPlan`s for a given arm + release state."""

    def __init__(
        self,
        arm,
        payload_mass=0.0,
        jerk_max=None,
        n_samples=120,
        control_dt=0.02,
    ):
        self.arm = arm
        self.chk = FeasibilityChecker(
            arm, payload_mass=payload_mass, jerk_max=jerk_max, n_samples=n_samples
        )
        self.n = len(arm._qd_max)
        self.control_dt = float(control_dt)
        self.torque_mode = arm._control_mode == "torque"

    # -- geometry ---------------------------------------------------------
    def _windup_pose(self, q_release, qd_release, local_dur, S1):
        """Cock-back pose implied by the shape's own displacement integral.

        For any symmetric shape S1 = 1/2, giving exactly the q_w the shipped
        `monotonic_windup` path computes.  A `Plateau` shape has S1 > 1/2 and
        cocks back correspondingly further.
        """
        qw = q_release - qd_release * local_dur * S1
        return np.clip(qw, self.arm._q_lo, self.arm._q_hi)

    def _stagger_frac(self, stagger_max):
        """Proximal-to-distal start offsets as fractions of the throw window.

        Joint index is the physical base->wrist chain order for the Gen3 and
        Panda URDFs, so a monotone ramp in index IS proximal-to-distal
        ("summation of speed") sequencing.  stagger_max = 0.5 reproduces the
        shipped `monotonic_windup` stagger exactly.
        """
        return np.linspace(0.0, float(stagger_max), self.n)

    # -- phase samplers ---------------------------------------------------
    @staticmethod
    def _throw_sampler(q_w, q_release, qd_release, stagger, local, shape):
        def ev(t):
            loc = np.maximum(local, 1e-12)
            tau = np.clip((t - stagger) / loc, 0.0, 1.0)
            active = (t >= stagger) & (t <= stagger + local)
            return (
                q_w + qd_release * loc * shape.Sint(tau),
                qd_release * shape.s(tau),
                np.where(active, qd_release * shape.ds(tau) / loc, 0.0),
                np.where(active, qd_release * shape.dds(tau) / loc**2, 0.0),
            )

        return ev

    @staticmethod
    def _rest_sampler(q0, q1, dur, shape):
        dq = q1 - q0

        def ev(t):
            tau = np.clip(t / dur, 0.0, 1.0)
            return (
                q0 + dq * shape.Vint(tau),
                dq * shape.v(tau) / dur,
                dq * shape.dv(tau) / dur**2,
                dq * shape.ddv(tau) / dur**3,
            )

        return ev

    @staticmethod
    def _brake_sampler(q_release, qd_release, dur, shape):
        def ev(t):
            tau = np.clip(t / dur, 0.0, 1.0)
            return (
                q_release + qd_release * dur * (tau - shape.Sint(tau)),
                qd_release * (1.0 - shape.s(tau)),
                -qd_release * shape.ds(tau) / dur,
                -qd_release * shape.dds(tau) / dur**2,
            )

        return ev

    # -- the plan ---------------------------------------------------------
    def plan(
        self,
        q_release,
        qd_release,
        t_w=0.5,
        dt_throw=1.1,
        brake_dur=0.4,
        return_dur=0.6,
        throw_shape="const_accel",
        windup_shape="cubic",
        stagger_max=0.5,
        shape_kw=None,
        windup_kw=None,
        label=None,
        stagger=None,
        local_dur=None,
        snap_release=True,
    ):
        """Plan a throw reaching exactly (q_release, qd_release).

        `stagger` / `local_dur` let `dynopt` inject an optimized per-joint time
        allocation; when omitted the proximal-to-distal linear stagger is used.
        """
        q_release = np.asarray(q_release, dtype=float)
        qd_release = np.asarray(qd_release, dtype=float)
        q_neutral = np.asarray(self.arm._q_neutral, dtype=float)
        tshape = (
            throw_shape
            if isinstance(throw_shape, sh.VelocityShape)
            else sh.get_velocity_shape(throw_shape, **(shape_kw or {}))
        )
        wshape = (
            windup_shape
            if isinstance(windup_shape, sh.RestShape)
            else sh.get_rest_shape(windup_shape, **(windup_kw or {}))
        )
        frac0 = self._stagger_frac(stagger_max)
        fixed_alloc = stagger is not None and local_dur is not None
        if fixed_alloc:
            span = float(np.max(np.asarray(stagger) + np.asarray(local_dur)))
            frac0 = np.asarray(stagger, dtype=float) / max(span, 1e-12)

        # ---- throw phase: grow until torque/velocity/jerk feasible --------
        def build_throw(dur):
            st = frac0 * dur
            lo = dur - st
            q_w = self._windup_pose(q_release, qd_release, lo, tshape.S1)
            ev = self._throw_sampler(q_w, q_release, qd_release, st, lo, tshape)
            bits = dict(
                q_windup=q_w,
                stagger=st,
                local_dur=lo,
                qdd_release=qd_release * tshape.ds_at_release / np.maximum(lo, 1e-12),
            )
            return bits, self.chk.check_span(ev, 0.0, dur)

        if self.torque_mode:
            dt_throw, bits, throw_chk, _ = grow_until_feasible(build_throw, dt_throw)
        else:
            bits, throw_chk = build_throw(dt_throw)
        q_windup = bits["q_windup"]

        # ---- windup phase --------------------------------------------------
        def build_windup(dur):
            ev = self._rest_sampler(q_neutral, q_windup, dur, wshape)
            return None, self.chk.check_span(ev, 0.0, dur)

        if self.torque_mode:
            t_w, _, windup_chk, _ = grow_until_feasible(build_windup, t_w)
        else:
            _, windup_chk = build_windup(t_w)

        # ---- snap the release instant onto the control grid ----------------
        # The executor releases at int(t_r / dt).  Growing t_w (never shrinking)
        # keeps the windup feasible -- it only slows a rest-to-rest move down.
        if snap_release and self.control_dt > 0:
            t_r_raw = t_w + dt_throw
            k = int(np.ceil(t_r_raw / self.control_dt - 1e-9))
            t_w_snap = k * self.control_dt - dt_throw
            if t_w_snap > t_w:
                t_w = t_w_snap
                _, windup_chk = build_windup(t_w)

        # ---- brake: qd_release -> 0 through the same shape ------------------
        def build_brake(dur):
            ev = self._brake_sampler(q_release, qd_release, dur, tshape)
            q_stop = q_release + qd_release * dur * (1.0 - tshape.S1)
            return q_stop, self.chk.check_span(ev, 0.0, dur)

        if self.torque_mode:
            brake_dur, q_stop, brake_chk, _ = grow_until_feasible(
                build_brake, brake_dur
            )
        else:
            q_stop, brake_chk = build_brake(brake_dur)

        # ---- return: q_stop -> neutral, ordinary rest-to-rest ---------------
        def build_return(dur):
            ev = self._rest_sampler(q_stop, q_neutral, dur, wshape)
            return None, self.chk.check_span(ev, 0.0, dur)

        if self.torque_mode:
            return_dur, _, return_chk, _ = grow_until_feasible(build_return, return_dur)
        else:
            _, return_chk = build_return(return_dur)

        t_r = t_w + dt_throw
        t_b = t_r + brake_dur
        return LabPlan(
            q_neutral=q_neutral,
            q_windup=q_windup,
            q_release=q_release,
            qd_release=qd_release,
            qdd_release=bits["qdd_release"],
            q_stop=q_stop,
            t_w=float(t_w),
            t_r=float(t_r),
            t_b=float(t_b),
            T=float(t_b + return_dur),
            stagger=bits["stagger"],
            local_dur=bits["local_dur"],
            windup_shape=wshape,
            throw_shape=tshape,
            checks={
                "windup": windup_chk,
                "throw": throw_chk,
                "brake": brake_chk,
                "return": return_chk,
            },
            meta={
                "label": label or f"{tshape.name}/{wshape.name}",
                "throw_shape": tshape.name,
                "windup_shape": wshape.name,
                "stagger_max": stagger_max,
                "fixed_alloc": fixed_alloc,
            },
        )
