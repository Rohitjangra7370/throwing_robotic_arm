"""
Normalized trajectory SHAPES -- pure math, no PyBullet, no robot.

Everything in this module is scalar and dimensionless so it can be unit-tested
without a physics engine, and so the analytic peak values (which is what the
feasibility checks and the time-scaling loops really need) are available in
closed form instead of by sampling.

Two families, because a throw needs two different boundary-condition sets:

  VelocityShape  rest -> velocity     (the THROW phase: start at rest at the
                 cocked pose, arrive at exactly qd_release)
      qd(t)   = qd_e * s(tau)
      q(t)    = q_w + qd_e * T * Sint(tau)
      qdd(t)  = (qd_e / T)  * ds(tau)
      qddd(t) = (qd_e / T^2)* dds(tau)
    with tau = t/T, s(0)=0, s(1)=1, and s monotone non-decreasing so the phase
    peak velocity is exactly qd_e (this is what makes "|qd| <= qd_max holds by
    construction" true -- see arm_controller.plan_throw's comment).

  RestShape      rest -> rest         (the WINDUP phase)
      qd(t)   = (dq / T)   * v(tau)     with  integral_0^1 v = 1
      q(t)    = q0 + dq    * Vint(tau)
      qdd(t)  = (dq / T^2) * dv(tau)
      qddd(t) = (dq / T^3) * ddv(tau)

KEY FACT used all over the planner: every SYMMETRIC velocity shape
(s(tau) + s(1-tau) = 1) integrates to exactly 1/2, so the required cock-back
distance is dq = qd_e * T / 2 for ALL of them. That means constant-accel,
min-jerk and trapezoidal-accel throws can be compared at the SAME windup pose
and the SAME duration -- an apples-to-apples comparison in which the only thing
that changes is the acceleration shape.

Reference for the bounded-jerk ("double-S") family:
  Biagiotti & Melchiorri, *Trajectory Planning for Automatic Machines and
  Robots*, Springer 2008, ch. 3.4.
Min-jerk quintic:
  Flash & Hogan, J. Neurosci. 5(7):1688-1703, 1985.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "VelocityShape",
    "ConstAccel",
    "MinJerkVel",
    "TrapAccel",
    "Plateau",
    "RestShape",
    "CubicRest",
    "MinJerkRest",
    "TrapVelRest",
    "quintic_bc",
    "eval_quintic",
    "get_velocity_shape",
    "get_rest_shape",
]


# --------------------------------------------------------------------------
# rest -> velocity shapes (THROW phase)
# --------------------------------------------------------------------------
class VelocityShape:
    """s(tau): 0 -> 1, monotone. Subclasses supply s / ds / dds / Sint."""

    name = "base"

    #: integral of s over [0, 1]; 0.5 for every symmetric shape
    S1 = 0.5
    #: max |ds| over [0, 1]  -> peak acceleration  = peak_ds * qd_e / T
    peak_ds = 1.0
    #: max |dds| over [0, 1] -> peak jerk          = peak_dds * qd_e / T^2
    peak_dds = np.inf
    #: |ds(1)| -- acceleration AT RELEASE, in units of qd_e / T.  This is the
    #: first-order sensitivity of the realized release velocity to release
    #: timing error, and it is the single most hardware-relevant number here
    #: (gripper latency jitter on the real Gen3 is +-6.4 ms, 1 sigma).
    ds_at_release = 1.0

    def s(self, tau):
        raise NotImplementedError

    def ds(self, tau):
        raise NotImplementedError

    def dds(self, tau):
        raise NotImplementedError

    def Sint(self, tau):
        raise NotImplementedError

    def __repr__(self):
        return f"<{self.name}>"


class ConstAccel(VelocityShape):
    """s = tau.  Constant acceleration.

    This is EXACTLY what `arm_controller._cubic_to_velocity` degenerates to
    whenever the caller cocks back by dq = qd_e*T/2 (which is what
    `monotonic_windup=True` does): a3 = (qd_e*T - 2*dq)/T^3 = 0, so the "cubic"
    is a quadratic and qdd is constant.  Torque-optimal for a fixed (dq, qd_e,
    T) triple; worst possible release-timing behaviour, because it is still at
    FULL acceleration at t = T.
    """

    name = "const_accel"
    S1 = 0.5
    peak_ds = 1.0
    peak_dds = np.inf  # step in acceleration at both ends
    ds_at_release = 1.0

    def s(self, tau):
        return np.asarray(tau, dtype=float)

    def ds(self, tau):
        return np.ones_like(np.asarray(tau, dtype=float))

    def dds(self, tau):
        return np.zeros_like(np.asarray(tau, dtype=float))

    def Sint(self, tau):
        tau = np.asarray(tau, dtype=float)
        return 0.5 * tau**2


class MinJerkVel(VelocityShape):
    """Quintic min-jerk velocity ramp: s = 10t^3 - 15t^4 + 6t^5.

    ds(0) = ds(1) = 0, so acceleration is continuous into and out of the phase
    and, crucially, is ZERO at release: realized release velocity is then
    second-order insensitive to release-timing error.  Cost: peak acceleration
    is 1.875x the constant-accel profile for the same (qd_e, T), so a
    torque-limited arm must stretch T by up to 1.875x to keep the same margin.
    """

    name = "min_jerk"
    S1 = 0.5
    peak_ds = 1.875                 # 30/16 at tau = 1/2
    peak_dds = 30.0 / np.sqrt(27.0)  # = 5.7735, at tau = (1 -+ 1/sqrt(3))/2
    ds_at_release = 0.0

    def s(self, tau):
        t = np.asarray(tau, dtype=float)
        return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5

    def ds(self, tau):
        t = np.asarray(tau, dtype=float)
        return 30.0 * t**2 * (1.0 - t) ** 2

    def dds(self, tau):
        t = np.asarray(tau, dtype=float)
        return 60.0 * t - 180.0 * t**2 + 120.0 * t**3

    def Sint(self, tau):
        t = np.asarray(tau, dtype=float)
        return 2.5 * t**4 - 3.0 * t**5 + t**6


class TrapAccel(VelocityShape):
    """Trapezoidal acceleration with jerk-limited ramps of width `beta`.

    The tunable middle ground between ConstAccel (beta -> 0, infinite jerk,
    minimum peak torque) and a triangular-acceleration profile (beta = 0.5).
    Peak acceleration  = qd_e / (T * (1 - beta))
    Peak jerk          = qd_e / (T^2 * beta * (1 - beta))
    Acceleration at release is 0 for any beta > 0, so this keeps min-jerk's
    release-timing robustness at a fraction of its torque cost -- beta = 0.25
    costs only 1.333x peak acceleration versus min-jerk's 1.875x.
    """

    name = "trap_accel"

    def __init__(self, beta=0.25):
        beta = float(beta)
        if not (0.0 < beta <= 0.5):
            raise ValueError(f"beta must be in (0, 0.5], got {beta}")
        self.beta = beta
        self.A = 1.0 / (1.0 - beta)
        self.S1 = 0.5                        # symmetric shape
        self.peak_ds = self.A
        self.peak_dds = self.A / beta
        self.ds_at_release = 0.0
        self.name = f"trap_accel(b={beta:g})"

    # s is built piecewise from the trapezoidal ds
    def ds(self, tau):
        t = np.asarray(tau, dtype=float)
        b, A = self.beta, self.A
        out = np.full_like(t, A)
        lo = t < b
        hi = t > 1.0 - b
        out = np.where(lo, A * t / b, out)
        out = np.where(hi, A * (1.0 - t) / b, out)
        return np.clip(out, 0.0, None)

    def dds(self, tau):
        t = np.asarray(tau, dtype=float)
        b, A = self.beta, self.A
        out = np.zeros_like(t)
        out = np.where(t < b, A / b, out)
        out = np.where(t > 1.0 - b, -A / b, out)
        return out

    def s(self, tau):
        t = np.clip(np.asarray(tau, dtype=float), 0.0, 1.0)
        b, A = self.beta, self.A
        s1 = A * t**2 / (2.0 * b)                       # ramp up
        s2 = A * (t - b / 2.0)                          # cruise
        s3 = 1.0 - A * (1.0 - t) ** 2 / (2.0 * b)       # ramp down
        return np.where(t < b, s1, np.where(t > 1.0 - b, s3, s2))

    def Sint(self, tau):
        t = np.clip(np.asarray(tau, dtype=float), 0.0, 1.0)
        b, A = self.beta, self.A
        # region 1: integral of A t^2/(2b)
        i1 = A * t**3 / (6.0 * b)
        i1b = A * b**2 / 6.0                            # value at t = b
        # region 2: integral of A (t - b/2)
        i2 = i1b + A * ((t**2 - b**2) / 2.0 - (b / 2.0) * (t - b))
        i2c = i1b + A * (((1 - b) ** 2 - b**2) / 2.0 - (b / 2.0) * (1 - 2 * b))
        # region 3: integral of 1 - A(1-t)^2/(2b)
        i3 = i2c + (t - (1 - b)) + A * ((1.0 - t) ** 3 - b**3) / (6.0 * b)
        return np.where(t < b, i1, np.where(t > 1.0 - b, i3, i2))


class Plateau(VelocityShape):
    """Wrap a shape so it reaches qd_e early and then HOLDS it.

    `frac` is the fraction of the phase spent at constant velocity, at the end.
    Acceleration and jerk are both exactly zero over the whole plateau, so the
    realized release velocity is completely insensitive to release-timing error
    anywhere inside it -- the strongest available answer to the real Gen3's
    67.9 +- 6.4 ms gripper latency and to the 25 ms command quantisation.

    The price is paid in POSITION, not velocity: the end-effector keeps moving
    at the full release speed through the plateau, so the release POINT drifts
    by |v| * jitter (about 1 cm per 6.4 ms at 1.5 m/s) and the cock-back
    distance grows from qd_e*T/2 to qd_e*T*(1+frac)/2.
    """

    def __init__(self, base: VelocityShape, frac=0.15):
        frac = float(frac)
        if not (0.0 <= frac < 1.0):
            raise ValueError(f"frac must be in [0, 1), got {frac}")
        self.base = base
        self.frac = frac
        self.k = 1.0 - frac                       # tau scale of the active part
        self.S1 = base.S1 * self.k + frac
        self.peak_ds = base.peak_ds / self.k
        self.peak_dds = base.peak_dds / self.k**2
        self.ds_at_release = 0.0
        self.name = f"plateau({base.name},f={frac:g})"

    def _u(self, tau):
        return np.clip(np.asarray(tau, dtype=float) / self.k, 0.0, 1.0)

    def s(self, tau):
        return self.base.s(self._u(tau))

    def ds(self, tau):
        t = np.asarray(tau, dtype=float)
        return np.where(t < self.k, self.base.ds(self._u(t)) / self.k, 0.0)

    def dds(self, tau):
        t = np.asarray(tau, dtype=float)
        return np.where(t < self.k, self.base.dds(self._u(t)) / self.k**2, 0.0)

    def Sint(self, tau):
        t = np.clip(np.asarray(tau, dtype=float), 0.0, 1.0)
        active = self.k * self.base.Sint(self._u(t))
        held = self.k * self.base.S1 + (t - self.k)
        return np.where(t < self.k, active, held)


# --------------------------------------------------------------------------
# rest -> rest shapes (WINDUP phase)
# --------------------------------------------------------------------------
class RestShape:
    """v(tau) with v(0)=v(1)=0 and integral_0^1 v = 1."""

    name = "base"
    peak_v = 1.5
    peak_dv = 6.0
    peak_ddv = np.inf

    def v(self, tau):
        raise NotImplementedError

    def dv(self, tau):
        raise NotImplementedError

    def ddv(self, tau):
        raise NotImplementedError

    def Vint(self, tau):
        raise NotImplementedError

    def __repr__(self):
        return f"<{self.name}>"


class CubicRest(RestShape):
    """Baseline: exactly `arm_controller._cubic_rest_to_rest`.

    Vint = 3t^2 - 2t^3, i.e. q = q0 + dq*(3t^2 - 2t^3), which is the same
    polynomial that function builds.  Acceleration steps from 0 to 6*dq/T^2 at
    t=0 (infinite jerk) and from -6*dq/T^2 to whatever the throw phase starts
    with at t=T.
    """

    name = "cubic"
    peak_v = 1.5
    peak_dv = 6.0
    peak_ddv = np.inf

    def v(self, tau):
        t = np.asarray(tau, dtype=float)
        return 6.0 * t * (1.0 - t)

    def dv(self, tau):
        t = np.asarray(tau, dtype=float)
        return 6.0 - 12.0 * t

    def ddv(self, tau):
        return np.full_like(np.asarray(tau, dtype=float), -12.0)

    def Vint(self, tau):
        t = np.asarray(tau, dtype=float)
        return 3.0 * t**2 - 2.0 * t**3


class MinJerkRest(RestShape):
    """Quintic min-jerk rest-to-rest (Flash & Hogan)."""

    name = "min_jerk"
    peak_v = 1.875
    peak_dv = 30.0 / np.sqrt(27.0)   # 5.7735
    peak_ddv = 60.0

    def v(self, tau):
        t = np.asarray(tau, dtype=float)
        return 30.0 * t**2 * (1.0 - t) ** 2

    def dv(self, tau):
        t = np.asarray(tau, dtype=float)
        return 60.0 * t - 180.0 * t**2 + 120.0 * t**3

    def ddv(self, tau):
        t = np.asarray(tau, dtype=float)
        return 60.0 - 360.0 * t + 360.0 * t**2

    def Vint(self, tau):
        t = np.asarray(tau, dtype=float)
        return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


class TrapVelRest(RestShape):
    """Trapezoidal velocity with smoothstep ramps of width `beta`.

    Lowest peak velocity of the three (1/(1-beta) vs 1.5 / 1.875), which is
    what matters when the windup is velocity-limited rather than torque-limited
    -- e.g. the long cock-backs the kinetic-chain stagger asks for.
    """

    name = "trap_vel"

    def __init__(self, beta=0.25):
        beta = float(beta)
        if not (0.0 < beta <= 0.5):
            raise ValueError(f"beta must be in (0, 0.5], got {beta}")
        self.beta = beta
        self.V = 1.0 / (1.0 - beta)
        self.peak_v = self.V
        self.peak_dv = 1.5 * self.V / beta
        self.peak_ddv = 6.0 * self.V / beta**2
        self.name = f"trap_vel(b={beta:g})"

    @staticmethod
    def _g(u):
        return 3.0 * u**2 - 2.0 * u**3

    @staticmethod
    def _dg(u):
        return 6.0 * u - 6.0 * u**2

    @staticmethod
    def _ddg(u):
        return 6.0 - 12.0 * u

    @staticmethod
    def _Gint(u):
        return u**3 - 0.5 * u**4

    def v(self, tau):
        t = np.clip(np.asarray(tau, dtype=float), 0.0, 1.0)
        b, V = self.beta, self.V
        up = V * self._g(t / b)
        dn = V * self._g((1.0 - t) / b)
        return np.where(t < b, up, np.where(t > 1.0 - b, dn, V))

    def dv(self, tau):
        t = np.clip(np.asarray(tau, dtype=float), 0.0, 1.0)
        b, V = self.beta, self.V
        up = V * self._dg(t / b) / b
        dn = -V * self._dg((1.0 - t) / b) / b
        return np.where(t < b, up, np.where(t > 1.0 - b, dn, 0.0))

    def ddv(self, tau):
        t = np.clip(np.asarray(tau, dtype=float), 0.0, 1.0)
        b, V = self.beta, self.V
        up = V * self._ddg(t / b) / b**2
        dn = V * self._ddg((1.0 - t) / b) / b**2
        return np.where(t < b, up, np.where(t > 1.0 - b, dn, 0.0))

    def Vint(self, tau):
        t = np.clip(np.asarray(tau, dtype=float), 0.0, 1.0)
        b, V = self.beta, self.V
        i1 = V * b * self._Gint(t / b)
        i1b = V * b * self._Gint(1.0)                      # = V*b/2
        i2 = i1b + V * (t - b)
        i2c = i1b + V * (1.0 - 2.0 * b)
        # ramp-down: integral of V*g((1-t)/b) from 1-b to t
        i3 = i2c + V * b * (self._Gint(1.0) - self._Gint((1.0 - t) / b))
        return np.where(t < b, i1, np.where(t > 1.0 - b, i3, i2))


# --------------------------------------------------------------------------
# generic quintic with full C2 boundary conditions (used for the follow phase)
# --------------------------------------------------------------------------
def quintic_bc(q0, qd0, qdd0, qT, qdT, qddT, T):
    """Coefficients (n, 6) of the unique quintic meeting all six BCs per joint.

    Returned in ascending power order so `eval_quintic` is a plain Horner-free
    polynomial evaluation, matching the (n, 4) layout `_eval_cubic` uses.
    """
    q0 = np.asarray(q0, dtype=float)
    qd0 = np.asarray(qd0, dtype=float)
    qdd0 = np.asarray(qdd0, dtype=float)
    qT = np.asarray(qT, dtype=float)
    qdT = np.asarray(qdT, dtype=float)
    qddT = np.asarray(qddT, dtype=float)
    T = float(T)
    a0 = q0
    a1 = qd0
    a2 = 0.5 * qdd0
    d = qT - (a0 + a1 * T + a2 * T**2)
    dd = qdT - (a1 + 2.0 * a2 * T)
    ddd = qddT - 2.0 * a2
    # [T^3 T^4 T^5; 3T^2 4T^3 5T^4; 6T 12T^2 20T^3] [a3 a4 a5]^T = [d dd ddd]^T
    a3 = (10.0 * d - 4.0 * dd * T + 0.5 * ddd * T**2) / T**3
    a4 = (-15.0 * d + 7.0 * dd * T - 1.0 * ddd * T**2) / T**4
    a5 = (6.0 * d - 3.0 * dd * T + 0.5 * ddd * T**2) / T**5
    return np.stack([a0, a1, a2, a3, a4, a5], axis=1)


def eval_quintic(coeffs, t):
    """(q, qd, qdd, qddd) for the (n, 6) coefficient block at scalar time t."""
    a = coeffs
    t = float(t)
    pw = np.array([1.0, t, t**2, t**3, t**4, t**5])
    q = a @ pw
    qd = a[:, 1:] @ (np.arange(1, 6) * pw[:5])
    qdd = a[:, 2:] @ (np.arange(2, 6) * np.arange(1, 5) * pw[:4])
    qddd = a[:, 3:] @ (np.arange(3, 6) * np.arange(2, 5) * np.arange(1, 4) * pw[:3])
    return q, qd, qdd, qddd


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
_VEL_SHAPES = {
    "const_accel": lambda **kw: ConstAccel(),
    "min_jerk": lambda **kw: MinJerkVel(),
    "trap_accel": lambda beta=0.25, **kw: TrapAccel(beta),
    "plateau": lambda beta=0.25, frac=0.15, **kw: Plateau(TrapAccel(beta), frac),
    "plateau_minjerk": lambda frac=0.15, **kw: Plateau(MinJerkVel(), frac),
}

_REST_SHAPES = {
    "cubic": lambda **kw: CubicRest(),
    "min_jerk": lambda **kw: MinJerkRest(),
    "trap_vel": lambda beta=0.25, **kw: TrapVelRest(beta),
}


def get_velocity_shape(name, **kw) -> VelocityShape:
    if name not in _VEL_SHAPES:
        raise ValueError(
            f"Unknown velocity shape '{name}'. Available: {sorted(_VEL_SHAPES)}"
        )
    return _VEL_SHAPES[name](**kw)


def get_rest_shape(name, **kw) -> RestShape:
    if name not in _REST_SHAPES:
        raise ValueError(
            f"Unknown rest shape '{name}'. Available: {sorted(_REST_SHAPES)}"
        )
    return _REST_SHAPES[name](**kw)
