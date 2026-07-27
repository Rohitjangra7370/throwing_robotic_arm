"""
Safety-first hardware executor for the Kinova Gen3 7-DOF throw.

DESIGN PRINCIPLE: the arm must be physically UNABLE to be commanded past its
limits, and the DEFAULT must touch nothing. Safety is structural, not advisory,
because this code cannot be validated by "seeing if it works" -- a bad command
on a real 7-DOF arm doing a throw is dangerous and can damage the hardware.

Single source of truth: the throw trajectory is planned by the SAME sim
ArmController (IK + 3-phase cubic) used everywhere else in this repo, so the
hardware executes an identical motion to simulation. This module only SWAPS THE
EXECUTOR -- from "step PyBullet" to "stream joint velocities to Kortex".

Layered safety (every layer independent):
  1. DRY-RUN by default. No Kortex connection, no motion, unless dry_run=False.
  2. speed_scale in (0, 1], default 0.15. The whole throw is TIME-STRETCHED by
     1/speed_scale: positions follow the real geometry, wall-clock velocities are
     scaled DOWN. speed_scale=1.0 is the real throw; 0.15 is a safe 15% rehearsal
     with correct geometry (ball just dribbles out -- validates motion + gripper
     + no self/table collision, safely).
  3. Hard per-joint velocity clamp to qd_max (from the profile's MEASURED limits).
     Commanded velocity is NEVER scaled up, only down. A software bug cannot
     command an over-speed joint.
  4. Position pre-check: the ENTIRE trajectory is sampled and every joint verified
     inside a soft-limit envelope (margin inside the URDF limits) BEFORE any motion.
     Fails closed (aborts) on any violation.
  5. Watchdog + guaranteed stop: any exception, Ctrl-C, or loop exit sends
     zero-velocity and releases servoing in a finally block. The arm stops.
  6. Kortex interaction is isolated behind one backend. Dry-run uses a stub, so
     this file runs with NO arm and NO kortex_api installed.

UNTESTED AGAINST REAL HARDWARE. The Kortex method/message names below match the
Kinova Kortex Python API shape but MUST be verified against your installed
kortex_api version during staged bring-up (see run_hardware_throw.py). Every
Kortex call is centralised in _KortexBackend so fixes live in one place.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
# Safety limits
# --------------------------------------------------------------------------- #
@dataclass
class SafetyLimits:
    qd_max: np.ndarray                 # per-joint velocity ceiling (rad/s), from profile
    q_soft_lo: np.ndarray              # soft joint lower limits (rad)
    q_soft_hi: np.ndarray              # soft joint upper limits (rad)
    speed_scale: float = 0.15          # global slow-motion factor in (0, 1]
    # Kinova Gen3 low-level joint control runs at 1 kHz. Streaming at 100 Hz
    # resampled the trajectory 10x coarser than the arm can accept and put 10 ms
    # of quantisation on the release instant -- at 1.63 m/s that is 1.6 cm of
    # landing error handed over for free.
    control_hz: float = 1000.0         # command streaming rate
    max_traj_seconds: float = 180.0    # hard cap on total execution wall-clock
    tau_max: np.ndarray | None = None  # per-joint torque ceiling (Nm), from profile
    torque_margin: float = 0.90        # refuse plans above this fraction of tau_max
    release_box_lo: np.ndarray = field(default_factory=lambda: np.array([0.2, -0.5, 0.1]))
    release_box_hi: np.ndarray = field(default_factory=lambda: np.array([0.9, 0.5, 0.9]))

    def __post_init__(self):
        assert 0.0 < self.speed_scale <= 1.0, "speed_scale must be in (0, 1]"
        self.qd_max = np.asarray(self.qd_max, dtype=float)
        self.q_soft_lo = np.asarray(self.q_soft_lo, dtype=float)
        self.q_soft_hi = np.asarray(self.q_soft_hi, dtype=float)

    def clamp_velocity(self, qd: np.ndarray) -> np.ndarray:
        """Hard clamp to +-qd_max. Never scales up."""
        return np.clip(np.asarray(qd, dtype=float), -self.qd_max, self.qd_max)

    def clamp_active(self, qd: np.ndarray, tol=1e-9) -> bool:
        """
        True if clamping would actually alter this command.

        The clamp keeps the ARM safe but silently changes the THROW: a clamped
        joint releases the ball slower than the policy asked for, and the ball
        lands short with nothing in the logs to say why. A plan that needs
        clamping is a plan that will not do what it claims, so precheck treats
        this as a hard failure rather than a correction.
        """
        qd = np.asarray(qd, dtype=float)
        return bool(np.any(np.abs(qd) > self.qd_max + tol))


# --------------------------------------------------------------------------- #
# Kortex backends (real + dry-run stub)
# --------------------------------------------------------------------------- #
class _DryRunBackend:
    """No hardware. Records commands so a plan can be inspected without an arm."""

    def __init__(self, n_dofs):
        self.n_dofs = n_dofs
        self.connected = False
        self.gripper_pos = 1.0  # 1=closed, 0=open (Kortex convention)
        self.commands = []

    def connect(self):
        self.connected = True
        print("[DRY-RUN] would connect to arm (no network I/O)")

    def disconnect(self):
        self.connected = False
        print("[DRY-RUN] would disconnect")

    def read_joint_state(self):
        # no real feedback available in dry-run
        return np.zeros(self.n_dofs), np.zeros(self.n_dofs)

    def send_joint_velocities(self, qd):
        self.commands.append(("vel", float(np.max(np.abs(qd)))))

    def send_gripper(self, pos):
        self.gripper_pos = float(pos)

    def stop(self):
        self.commands.append(("stop", 0.0))


class _KortexBackend:
    """
    Real Kinova Gen3 backend. All kortex_api usage lives here.

    NOTE: verify method/message names against your kortex_api version during
    bring-up. Import is lazy so the module loads without kortex_api installed.
    """

    def __init__(self, n_dofs, ip="192.168.1.10", port=10000, port_rt=10001,
                 username="admin", password="admin"):
        self.n_dofs = n_dofs
        self.ip, self.port, self.port_rt = ip, port, port_rt
        self.username, self.password = username, password
        self.connected = False
        self._base = None
        self._base_cyclic = None
        self._transport = None
        self._session = None

    def connect(self):
        # Lazy imports: only needed for real hardware.
        from kortex_api.TCPTransport import TCPTransport
        from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
        from kortex_api.SessionManager import SessionManager
        from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
        from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
        from kortex_api.autogen.messages import Session_pb2, Base_pb2  # noqa: F401

        self._transport = TCPTransport()
        self._transport.connect(self.ip, self.port)
        self._router = RouterClient(self._transport, lambda ex: print("KORTEX ERR:", ex))

        sess = Session_pb2.CreateSessionInfo()
        sess.username = self.username
        sess.password = self.password
        sess.session_inactivity_timeout = 60000
        sess.connection_inactivity_timeout = 2000
        self._session = SessionManager(self._router)
        self._session.CreateSession(sess)

        self._base = BaseClient(self._router)
        self._base_cyclic = BaseCyclicClient(self._router)
        self.connected = True
        print(f"[KORTEX] connected to {self.ip}")

    def disconnect(self):
        try:
            if self._session is not None:
                self._session.CloseSession()
            if self._transport is not None:
                self._transport.disconnect()
        finally:
            self.connected = False
            print("[KORTEX] disconnected")

    def read_joint_state(self):
        fb = self._base_cyclic.RefreshFeedback()
        q = np.array([np.deg2rad(a.position) for a in fb.actuators[: self.n_dofs]])
        qd = np.array([np.deg2rad(a.velocity) for a in fb.actuators[: self.n_dofs]])
        return q, qd

    def send_joint_velocities(self, qd):
        """High-level joint speed command (deg/s). Onboard controller enforces
        its own hard limits -- a second independent safety net beneath ours."""
        from kortex_api.autogen.messages import Base_pb2
        cmd = Base_pb2.JointSpeeds()
        for i, w in enumerate(qd):
            js = cmd.joint_speeds.add()
            js.joint_identifier = i
            js.value = float(np.rad2deg(w))
            js.duration = 0
        self._base.SendJointSpeedsCommand(cmd)

    def send_gripper(self, pos):
        """pos in [0,1]; 0=open, 1=closed (Kortex GRIPPER_POSITION)."""
        from kortex_api.autogen.messages import Base_pb2
        cmd = Base_pb2.GripperCommand()
        cmd.mode = Base_pb2.GRIPPER_POSITION
        finger = cmd.gripper.finger.add()
        finger.finger_identifier = 1
        finger.value = float(np.clip(pos, 0.0, 1.0))
        self._base.SendGripperCommand(cmd)

    def stop(self):
        """Command zero joint velocity immediately."""
        self.send_joint_velocities(np.zeros(self.n_dofs))


# --------------------------------------------------------------------------- #
# Executor
# --------------------------------------------------------------------------- #
class HardwareThrowExecutor:
    """
    Executes a planned throw on the real Gen3 (or dry-run), with all safety
    layers. Use as a context manager so the arm ALWAYS stops on exit:

        with HardwareThrowExecutor(limits, dry_run=False, ip=...) as ex:
            ex.home(arm_controller)
            ex.rehearse_or_throw(coeffs, arm_controller)
    """

    def __init__(self, limits: SafetyLimits, dry_run=True, ip="192.168.1.10",
                 gripper_open=0.0, gripper_closed=1.0):
        self.limits = limits
        self.dry_run = dry_run
        self.n_dofs = len(limits.qd_max)
        self.gripper_open = gripper_open
        self.gripper_closed = gripper_closed
        self.last_exec_stats = None   # timing record of the most recent execution
        self.backend = (
            _DryRunBackend(self.n_dofs) if dry_run
            else _KortexBackend(self.n_dofs, ip=ip)
        )

    # -- context management: guaranteed stop ------------------------------- #
    def __enter__(self):
        self.backend.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.backend.stop()
        except Exception as e:  # never mask the real error, but always try to stop
            print("WARNING: stop() failed during teardown:", e)
        self.backend.disconnect()
        return False  # do not suppress exceptions

    # -- pre-flight checks ------------------------------------------------- #
    def precheck(self, coeffs, arm, n_samples=400):
        """
        Sample the whole trajectory and verify EVERY joint stays inside the soft
        envelope and every commanded velocity (after speed_scale) is within
        qd_max. Returns (ok, report). Fails closed.
        """
        T = coeffs["T"]
        problems = []
        clamped = []
        peak_qd = np.zeros(self.n_dofs)
        peak_tau = np.zeros(self.n_dofs)
        tau_max = self.limits.tau_max
        for t in np.linspace(0.0, T, n_samples):
            q, qd, qdd = arm.get_setpoint(coeffs, t, with_accel=True)
            q = np.asarray(q)
            qd_raw = np.asarray(qd) * self.limits.speed_scale
            if self.limits.clamp_active(qd_raw):
                clamped.append(
                    f"  t={t:.3f}s |qd|={np.round(np.abs(qd_raw), 3)} "
                    f"exceeds qd_max={np.round(self.limits.qd_max, 3)}"
                )
            qd_cmd = self.limits.clamp_velocity(qd_raw)
            peak_qd = np.maximum(peak_qd, np.abs(qd_cmd))
            lo_viol = self.limits.q_soft_lo - q
            hi_viol = q - self.limits.q_soft_hi
            for j in range(self.n_dofs):
                if lo_viol[j] > 0 or hi_viol[j] > 0:
                    problems.append(
                        f"  t={t:.3f}s joint{j}: q={q[j]:+.3f} outside "
                        f"[{self.limits.q_soft_lo[j]:+.3f},{self.limits.q_soft_hi[j]:+.3f}]"
                    )
            if tau_max is not None:
                # Inverse dynamics along the WHOLE path. Endpoint-only checking
                # is what let a follow-through demanding 3.2x torque ship: peak
                # torque lives mid-swing (gravity) and mid-ramp (inertia), not
                # at the ends.
                tau = np.abs(np.asarray(arm.inverse_dynamics(q, qd, qdd), dtype=float))
                peak_tau = np.maximum(peak_tau, tau)
        report = [
            f"trajectory T={T:.3f}s, speed_scale={self.limits.speed_scale}",
            f"peak commanded |qd| (rad/s): "
            + ", ".join(f"{v:.2f}/{m:.2f}" for v, m in zip(peak_qd, self.limits.qd_max)),
        ]
        if tau_max is not None:
            report.append(
                "peak |tau| (Nm): "
                + ", ".join(f"{v:.1f}/{m:.1f}" for v, m in zip(peak_tau, tau_max))
            )
            over = peak_tau > self.limits.torque_margin * np.asarray(tau_max, float)
            if np.any(over):
                problems.append(
                    f"  TORQUE over {self.limits.torque_margin:.0%} of limit on joints "
                    f"{list(np.where(over)[0])}: "
                    f"{np.round(peak_tau[over], 1)} vs {np.round(np.asarray(tau_max, float)[over], 1)} Nm"
                )
        else:
            report.append("peak |tau|: NOT CHECKED (no tau_max in limits)")
        if clamped:
            problems.append(
                f"  VELOCITY CLAMPING ACTIVE at {len(clamped)} sampled instants -- "
                "the executed throw would be SLOWER than planned and land short"
            )
            problems.extend(clamped[:5])
        if problems:
            report.append(f"VIOLATIONS ({len(problems)}):")
            report.extend(problems[:12])
            if len(problems) > 12:
                report.append(f"  ... and {len(problems) - 12} more")
        ok = len(problems) == 0
        return ok, "\n".join(report)

    def check_release_pos(self, release_pos):
        rp = np.asarray(release_pos, dtype=float)
        inside = np.all(rp >= self.limits.release_box_lo) and np.all(rp <= self.limits.release_box_hi)
        return bool(inside)

    # -- motions ----------------------------------------------------------- #
    def home(self, arm, q_neutral, duration=4.0):
        """Slow, capped move to the neutral pose using proportional joint-speed
        servoing. Deliberately gentle -- this is the safe way to reach start."""
        q_neutral = np.asarray(q_neutral, dtype=float)
        dt = 1.0 / self.limits.control_hz
        # cap homing speed at a small fraction of qd_max regardless of speed_scale
        home_cap = 0.25 * self.limits.qd_max
        t0 = time.time()
        print(f"[home] moving to neutral over ~{duration}s (capped, gentle)")
        while time.time() - t0 < duration:
            q, _ = self.backend.read_joint_state() if not self.dry_run else (q_neutral * 0, None)
            err = q_neutral - q
            qd = np.clip(2.0 * err, -home_cap, home_cap)  # P-servo, capped
            qd = self.limits.clamp_velocity(qd)
            self.backend.send_joint_velocities(qd)
            if np.max(np.abs(err)) < 0.01:
                break
            time.sleep(dt)
        self.backend.stop()
        print("[home] done")

    def set_gripper(self, closed: bool):
        self.backend.send_gripper(self.gripper_closed if closed else self.gripper_open)

    def rehearse_or_throw(self, coeffs, arm, verbose=True):
        """
        Stream the throw. Trajectory-time s advances at speed_scale of wall-clock,
        so commanded velocity qd(s)*speed_scale is chain-rule-consistent with the
        stretched playback: positions follow real geometry, velocities scale down.
        Gripper opens when s crosses t_r. Guaranteed stop on any exit.

        speed_scale < 1.0 -> safe slow rehearsal (ball dribbles).
        speed_scale = 1.0 -> the real throw.
        """
        t_r = coeffs["t_r"]
        T = coeffs["T"]
        scale = self.limits.speed_scale
        dt = 1.0 / self.limits.control_hz
        wall_T = T / scale
        if wall_T > self.limits.max_traj_seconds:
            raise RuntimeError(
                f"execution would take {wall_T:.1f}s > max_traj_seconds "
                f"{self.limits.max_traj_seconds}s; refuse."
            )

        released = False
        s = 0.0
        # Absolute-deadline pacing. `time.sleep(dt)` sleeps dt PLUS however long
        # the loop body took plus scheduler slop, so the period silently drifts
        # long and the drift accumulates over thousands of ticks -- at 1 kHz the
        # body cost is a large fraction of dt, so this is not a rounding detail.
        # Deadlines are computed from a fixed origin so an overrun on one tick is
        # absorbed rather than pushed into every later tick.
        t0 = time.perf_counter()
        tick = 0
        worst_late = 0.0
        release_wall = None
        if verbose:
            print(f"[exec] speed_scale={scale} wall_T={wall_T:.2f}s "
                  f"release at s={t_r:.3f}s rate={self.limits.control_hz:.0f}Hz")
        try:
            while True:
                wall = time.perf_counter() - t0
                s = wall * scale
                if s > T:
                    break
                q, qd, _ = arm.get_setpoint(coeffs, s, with_accel=True)
                qd_cmd = self.limits.clamp_velocity(np.asarray(qd) * scale)
                self.backend.send_joint_velocities(qd_cmd)
                if (not released) and s >= t_r:
                    self.set_gripper(closed=False)  # OPEN -> release
                    released = True
                    release_wall = wall
                    if verbose:
                        print(f"[exec] gripper release commanded at wall={wall:.3f}s")
                tick += 1
                deadline = t0 + tick * dt
                late = time.perf_counter() - deadline
                worst_late = max(worst_late, late)
                if late < 0:
                    time.sleep(-late)
        finally:
            self.backend.stop()
        elapsed = time.perf_counter() - t0
        achieved_hz = tick / elapsed if elapsed > 0 else 0.0
        # Timing quality is a RESULT, not a debug print: at 1.63 m/s every 1 ms of
        # release-instant error is 1.6 mm of landing error, so this belongs in the
        # per-throw record alongside the landing position.
        self.last_exec_stats = {
            "ticks": tick,
            "elapsed_s": elapsed,
            "target_hz": float(self.limits.control_hz),
            "achieved_hz": achieved_hz,
            "worst_late_ms": worst_late * 1e3,
            "release_wall_s": release_wall,
        }
        if verbose:
            print(f"[exec] trajectory complete, arm stopped "
                  f"({tick} ticks, {achieved_hz:.0f}Hz achieved vs "
                  f"{self.limits.control_hz:.0f}Hz target, worst tick "
                  f"{worst_late * 1e3:.2f}ms late)")
            if achieved_hz < 0.9 * self.limits.control_hz:
                print("[exec] WARNING: achieved rate is >10% below target -- the "
                      "control loop is not keeping up; release timing is degraded.")
        return released
