"""
Whole-trajectory feasibility checking for the throw lab.

Same contract as `ArmController._throw_peak_torque_ratio` / `._peak_qd_ratio`
(dense sampling of the WHOLE phase, never just the endpoints -- endpoint-only
checking is how a 3.2x-over-limit follow-through shipped), with three
deliberate differences:

1. It also checks JERK, which the shipped planner has no notion of.  A cubic
   segment has unbounded jerk at its endpoints by construction, so the shipped
   planner cannot even represent the constraint.

2. It optionally includes the GRIPPED BALL in the torque estimate.  `step()`
   adds a Jacobian-transpose point-mass term `J^T m (a_ee - g)` to the
   commanded torque and then clips the SUM to tau_max, but
   `_throw_peak_torque_ratio` calls `inverse_dynamics`, which sees the arm URDF
   only.  The shipped feasibility check is therefore optimistic by exactly that
   term.  Small for a 58 g ball, but it is a real bias and it is free to fix.

3. It reports the full ratio vector, so a caller can see WHICH joint binds --
   the input the whip/time-allocation optimizer needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pybullet as p

_GRAVITY = np.array([0.0, 0.0, -9.81])


@dataclass
class PhaseCheck:
    """Worst-case ratios over one phase. <= 1.0 everywhere means feasible."""

    tau_ratio: float
    qd_ratio: float
    q_ratio: float          # joint-limit usage, 0 = at neutral-ish, >1 = outside
    jerk_peak: float        # rad/s^3, informational unless jerk_max is set
    jerk_ratio: float
    worst_tau: np.ndarray = field(default=None)
    per_joint_tau_ratio: np.ndarray = field(default=None)

    @property
    def ok(self):
        return max(self.tau_ratio, self.qd_ratio, self.q_ratio, self.jerk_ratio) <= 1.0

    @property
    def worst(self):
        return max(self.tau_ratio, self.qd_ratio, self.q_ratio, self.jerk_ratio)

    def __str__(self):
        return (
            f"tau {self.tau_ratio:.3f}  qd {self.qd_ratio:.3f}  "
            f"q {self.q_ratio:.3f}  jerk {self.jerk_peak:.1f} rad/s^3"
        )


class FeasibilityChecker:
    """Wraps an ArmController with limits and a payload-aware torque model."""

    def __init__(self, arm, payload_mass=0.0, jerk_max=None, n_samples=120):
        self.arm = arm
        self.payload_mass = float(payload_mass)
        self.jerk_max = None if jerk_max is None else np.asarray(jerk_max, dtype=float)
        self.n_samples = int(n_samples)
        self.qd_max = np.asarray(arm._qd_max, dtype=float)
        self.tau_max = (
            None if arm._tau_max is None else np.asarray(arm._tau_max, dtype=float)
        )
        self.q_lo = np.asarray(arm._q_lo, dtype=float)
        self.q_hi = np.asarray(arm._q_hi, dtype=float)

    # -- torque -----------------------------------------------------------
    def torque(self, q, qd, qdd):
        """Inverse dynamics + (optionally) the gripped ball's contribution.

        The payload term mirrors `ArmController.step` exactly: a point mass
        rigidly held at the EE contributes J^T * m * (a_ee - g) with
        a_ee = J qdd.  The Jacobian-derivative (Coriolis) part is omitted there
        too, so omitting it here keeps the check and the controller consistent.
        """
        tau = self.arm.inverse_dynamics(q, qd, qdd)
        if self.payload_mass > 0.0:
            j_lin = self.jacobian(q)
            a_ee = j_lin @ np.asarray(qdd, dtype=float)
            tau = tau + j_lin.T @ (self.payload_mass * (a_ee - _GRAVITY))
        return tau

    def jacobian(self, q):
        arm = self.arm
        q_full = arm._ik_q_neutral.copy()
        for local_i, dof_id in enumerate(arm._dof_ids):
            q_full[dof_id] = q[local_i]
        j_lin_raw, _ = p.calculateJacobian(
            arm._arm_id,
            arm._ee_link,
            localPosition=[0, 0, 0],
            objPositions=q_full.tolist(),
            objVelocities=[0.0] * arm._n_dofs,
            objAccelerations=[0.0] * arm._n_dofs,
            physicsClientId=arm._cid,
        )
        return np.array(j_lin_raw)[:, arm._dof_ids]

    # -- phase sweep ------------------------------------------------------
    def check_samples(self, samples):
        """`samples` = iterable of (q, qd, qdd, qddd) tuples."""
        worst_tau_ratio = 0.0
        worst_qd_ratio = 0.0
        worst_q_ratio = 0.0
        worst_jerk = 0.0
        worst_jerk_ratio = 0.0
        worst_tau_vec = None
        per_joint = np.zeros(len(self.qd_max))
        for q, qd, qdd, qddd in samples:
            if self.tau_max is not None:
                tau = self.torque(q, qd, qdd)
                r_vec = np.abs(tau) / self.tau_max
                per_joint = np.maximum(per_joint, r_vec)
                r = float(r_vec.max())
                if r > worst_tau_ratio:
                    worst_tau_ratio, worst_tau_vec = r, tau
            worst_qd_ratio = max(
                worst_qd_ratio, float(np.max(np.abs(qd) / self.qd_max))
            )
            # joint-limit usage: 0 inside, >1 outside (symmetric normalisation
            # against the half-range so the number is comparable across joints)
            mid = 0.5 * (self.q_hi + self.q_lo)
            half = np.maximum(0.5 * (self.q_hi - self.q_lo), 1e-9)
            worst_q_ratio = max(worst_q_ratio, float(np.max(np.abs(q - mid) / half)))
            jm = float(np.max(np.abs(qddd)))
            worst_jerk = max(worst_jerk, jm)
            if self.jerk_max is not None:
                worst_jerk_ratio = max(
                    worst_jerk_ratio, float(np.max(np.abs(qddd) / self.jerk_max))
                )
        return PhaseCheck(
            tau_ratio=worst_tau_ratio,
            qd_ratio=worst_qd_ratio,
            q_ratio=worst_q_ratio,
            jerk_peak=worst_jerk,
            jerk_ratio=worst_jerk_ratio,
            worst_tau=worst_tau_vec,
            per_joint_tau_ratio=per_joint,
        )

    def check_span(self, eval_fn, t0, t1, n_samples=None):
        """Dense check of `eval_fn(t) -> (q, qd, qdd, qddd)` over [t0, t1]."""
        n = int(n_samples or self.n_samples)
        ts = np.linspace(float(t0), float(t1), n)
        return self.check_samples(eval_fn(t) for t in ts)


def grow_until_feasible(build_fn, dur0, max_iter=24, growth=1.12, cap=30.0):
    """Smallest duration >= dur0 on a geometric ladder that passes `build_fn`.

    `build_fn(dur) -> (obj, PhaseCheck)`.  Returns (dur, obj, check, n_iter).
    Raises RuntimeError if nothing on the ladder is feasible.

    Growth is 1.12 rather than the shipped planner's 1.2 with 6 iterations:
    that ladder can only reach 2.99x and lands on coarse rungs (a phase needing
    1.25x pays 1.44x).  24 rungs at 1.12 reach 15.2x with <= 12% overshoot.
    """
    dur = float(dur0)
    best = None
    for i in range(max_iter):
        obj, chk = build_fn(dur)
        if best is None or chk.worst < best[2].worst:
            best = (dur, obj, chk)
        if chk.ok:
            return dur, obj, chk, i
        dur *= growth
        if dur > cap:
            break
    raise RuntimeError(
        f"no feasible duration on ladder [{dur0:.3f}, {dur:.3f}] s; "
        f"best {best[2]} at {best[0]:.3f}s"
    )


def scan_until_feasible(build_fn, candidates):
    """First feasible duration in `candidates` (order matters).

    Used for phases whose worst-case ratio is NON-MONOTONE in duration -- the
    follow-through is the known case: peak |qd| falls with duration while peak
    |tau| bottoms out and then rises again as the arm spends longer hanging in
    high-gravity-torque configurations.
    """
    candidates = list(candidates)
    best = None
    for dur in candidates:
        obj, chk = build_fn(dur)
        if best is None or chk.worst < best[2].worst:
            best = (dur, obj, chk)
        if chk.ok:
            return dur, obj, chk
    raise RuntimeError(
        f"no feasible duration among {len(candidates)} candidates; "
        f"best {best[2]} at {best[0]:.3f}s"
    )
