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

import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np


def _patch_collections_abc():
    """
    Make kortex_api importable on Python 3.10+.

    kortex_api 2.6.0.post3 pins protobuf==3.5.1, whose Python implementation
    still does `collections.MutableMapping`. Those ABCs moved to
    collections.abc in 3.3 and were REMOVED from `collections` in 3.10, so a
    bare `import kortex_api...` dies with

        AttributeError: module 'collections' has no attribute 'MutableMapping'

    on this machine (Python 3.10.12). Re-exporting the ABCs is the standard
    workaround and is confined to this module, which is the only place Kortex
    is touched. Verified: with this applied, all nine Kortex symbols this
    backend calls import and resolve correctly.

    Note the wheel's protobuf 3.5.1 pin also downgrades protobuf system-wide,
    which breaks onnx/tensorboard/wandb. It does NOT affect the throw pipeline
    (torch/pybullet/numpy/scipy; full test suite still passes). If those tools
    are needed on the same machine, put the hardware stack in its own venv.
    """
    import collections
    import collections.abc
    for _name in ("MutableMapping", "Mapping", "MutableSequence", "Sequence",
                  "Callable", "Iterable", "MutableSet", "Set"):
        if not hasattr(collections, _name):
            setattr(collections, _name, getattr(collections.abc, _name))


# Ceiling on the HIGH-LEVEL command rate, from Kinova's own driver docs
# (Kinovarobotics/ros_kortex, kortex_driver/readme.md):
#
#   "The robot's high level commands function at a rate of 40Hz."
#   "The base high level commands are treated every 25 ms inside the robot."
#   "High level control cannot be achieved at a rate faster than 40 Hz for now."
#
# This executor is high-level: it streams Base.SendJointSpeedsCommand while the
# arm sits in SINGLE_LEVEL_SERVOING (read back from the lab arm, 2026-08-07).
# Sending faster is not an error and not a hazard -- JointSpeeds with duration=0
# is held until superseded, so surplus commands are simply coalesced -- but it
# buys nothing and it makes the achieved-rate log a measurement of our own loop
# rather than of the arm. The 1 kHz figure in Kinova's docs belongs to
# LOW_LEVEL_SERVOING (per-actuator BaseCyclic.Refresh), which this file does not
# use. Reaching 1 ms release timing therefore needs a servoing-mode change, not
# a faster loop.
HIGH_LEVEL_MAX_HZ = 40.0

# Wall-clock delay from "gripper open commanded" to "fingers actually move".
#
# MEASURED on the lab arm 2026-08-07 at 1 kHz over the UDP feedback channel,
# 15 trials, fingers unloaded: 67.9 +- 6.4 ms (range 59.5-80.0). A second run of
# 5 gave 70.5 +- 6.4, so the mean is stable to a few ms.
#
# Uncompensated this is the single largest error in the system: 67.9 ms at the
# 1.498 m/s release speed is 10.2 cm of undershoot, 3.5x the entire 2.89 cm sim
# accuracy. It would not look like a timing bug on the first hardware run -- it
# would look like the policy failing to transfer.
#
# It is compensable because it is repeatable. The 6.4 ms of scatter is almost
# exactly what the 25 ms command quantisation alone predicts for a uniform
# delay (std = 25/sqrt(12) = 7.2 ms), so the gripper's own mechanics contribute
# very little jitter -- the spread is the command path, not the hardware.
# Leading the trigger by this much leaves ~1.0 cm of residual, inside the sim
# accuracy.
#
# RE-MEASURED WITH A REAL BALL LOADED (2026-08-22, 15 trials, same 1 kHz UDP
# method): onset 73.2 +- 10.3 ms, statistically indistinguishable from the
# original static/unloaded 67.9 +- 6.4 ms (higher spread, not a shifted mean).
# The value below is unchanged; treat the "static and unloaded" caveat as
# closed for ONSET specifically. Still not validated against an actual real
# landing measurement -- that's the one thing this can't substitute for.
GRIPPER_RELEASE_LATENCY_S = 0.0679


# Wall-clock window, after the release-time gripper OPEN command, during which
# NO new SendJointSpeedsCommand is sent.
#
# FOUND 2026-08-22: SendGripperCommand is silently ignored by the arm's
# embedded controller for as long as SendJointSpeedsCommand is being actively
# streamed -- reproduced with an isolated test (arm not even moving, zero
# velocities streamed): the gripper does not move AT ALL while the stream is
# continuous, and only resumes once the stream pauses. This is not a session/
# channel issue (a fully separate TCP session for the gripper command was
# tried and made no difference) -- it is the controller's own real-time
# scheduling declining to service a gripper write while it's busy servicing
# joint-speed writes.
#
# Measured travel achieved (STATIC, no ball, no throw inertia) for a pause
# starting right when the command is sent: 100ms -> 83.8% closed (was 99%),
# 150ms -> 76.0%, 200ms -> 69.0%, 300ms -> 51.5%. A real throw has centrifugal/
# inertial assistance ejecting the ball well before "fully open" is needed, so
# this is a conservative choice, not a tight one.
#
# SAFETY: during the pause, JointSpeeds "hold until superseded" (see
# HIGH_LEVEL_MAX_HZ), so the arm coasts at the exact release-instant velocity
# -- not accelerating further, but also not running the planned follow-through
# deceleration -- for this whole window. Checked against joint limits at the
# real release state (q_release, qd_release, results_kinetic_chain_gen3/2):
# worst-case margin at 300ms is 76.4 deg, so 200ms leaves an enormous safety
# margin. Re-check this margin for any different checkpoint/pose table --
# it is a property of THIS release configuration, not a general law.
GRIPPER_RELEASE_PAUSE_S = 0.20


# --------------------------------------------------------------------------- #
# Safety limits
# --------------------------------------------------------------------------- #
@dataclass
class SafetyLimits:
    qd_max: np.ndarray                 # per-joint velocity ceiling (rad/s), from profile
    q_soft_lo: np.ndarray              # soft joint lower limits (rad)
    q_soft_hi: np.ndarray              # soft joint upper limits (rad)
    speed_scale: float = 0.15          # global slow-motion factor in (0, 1]
    # Command streaming rate. MUST NOT exceed HIGH_LEVEL_MAX_HZ -- see that
    # constant. The previous 1000.0 here was justified by "Gen3 joint control
    # runs at 1 kHz", which is true only of LOW_LEVEL_SERVOING; this executor
    # runs high-level (SINGLE_LEVEL_SERVOING, confirmed on the arm), where the
    # base treats commands every 25 ms.
    control_hz: float = HIGH_LEVEL_MAX_HZ
    max_traj_seconds: float = 180.0    # hard cap on total execution wall-clock
    tau_max: np.ndarray | None = None  # per-joint torque ceiling (Nm), from profile
    torque_margin: float = 0.90        # refuse plans above this fraction of tau_max
    # Wall-clock seconds to fire the gripper EARLY so the fingers move at t_r.
    # See GRIPPER_RELEASE_LATENCY_S. Set to 0.0 to disable compensation (e.g.
    # to reproduce an uncompensated baseline for the record).
    gripper_lead_s: float = GRIPPER_RELEASE_LATENCY_S
    # Time scale for the NON-THROW phases (windup and follow-through).
    # `speed_scale` applies to the throw phase alone -- see rehearse_or_throw.
    # 1.0 is safe by construction: ArmController._windup_pose_and_time already
    # grows the windup until its rest-to-rest cubic peak fits inside qd_max.
    positioning_scale: float = 1.0
    release_box_lo: np.ndarray = field(default_factory=lambda: np.array([0.2, -0.5, 0.1]))
    release_box_hi: np.ndarray = field(default_factory=lambda: np.array([0.9, 0.5, 0.9]))

    def __post_init__(self):
        assert 0.0 < self.speed_scale <= 1.0, "speed_scale must be in (0, 1]"
        assert 0.0 < self.positioning_scale <= 1.0, \
            "positioning_scale must be in (0, 1]"
        if self.control_hz > HIGH_LEVEL_MAX_HZ:
            # Clamp rather than raise: a slower stream is always the safe
            # direction, and refusing here would block a bring-up over a config
            # value that cannot cause harm. But say so loudly -- the old default
            # made "1000 Hz achieved" look like a hardware result when the arm
            # was only ever consuming 40 of those commands per second.
            print(f"[limits] control_hz {self.control_hz:.0f} exceeds the "
                  f"high-level ceiling {HIGH_LEVEL_MAX_HZ:.0f} Hz (Kinova: base "
                  f"treats high-level commands every 25 ms); clamping.")
            self.control_hz = HIGH_LEVEL_MAX_HZ
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

    def open_realtime_feedback(self):
        print("[DRY-RUN] would open 1 kHz UDP feedback (no network I/O)")

    def close_realtime_feedback(self):
        pass

    def read_gripper(self):
        # Instantaneous and exact, which is the point: a dry run can prove the
        # plumbing but can never measure a latency. Any number this produces is
        # zero by construction, and the tool says so rather than reporting it.
        return self.gripper_pos * 100.0, 0.0

    def stop(self):
        self.commands.append(("stop", 0.0))


class _KortexBackend:
    """
    Real Kinova Gen3 backend. All kortex_api usage lives here.

    NOTE: verify method/message names against your kortex_api version during
    bring-up. Import is lazy so the module loads without kortex_api installed.
    """

    def __init__(self, n_dofs, ip="192.168.1.101", port=10000, port_rt=10001,
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
        _patch_collections_abc()
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

    def open_realtime_feedback(self):
        """
        Second, UDP session carrying BaseCyclic feedback at 1 kHz.

        COMMANDS are capped at 40 Hz (HIGH_LEVEL_MAX_HZ) -- FEEDBACK is not.
        Kinova: "UDPTransport can only be used to read the robot's feedback at
        1kHz with the BaseCyclic service." That asymmetry is what makes the
        dominant error term measurable: we cannot command the gripper open at a
        precise instant, but we CAN observe exactly when the fingers started
        moving, and the difference is the latency we need to calibrate out.

        Separate transport/router/session from the TCP command channel -- they
        are independent connections to the same arm, not a mode switch, so this
        changes nothing about how the arm is commanded.
        """
        from kortex_api.UDPTransport import UDPTransport
        from kortex_api.RouterClient import RouterClient
        from kortex_api.SessionManager import SessionManager
        from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
        from kortex_api.autogen.messages import Session_pb2

        self._rt_transport = UDPTransport()
        self._rt_transport.connect(self.ip, self.port_rt)
        self._rt_router = RouterClient(self._rt_transport,
                                       lambda ex: print("KORTEX RT ERR:", ex))
        sess = Session_pb2.CreateSessionInfo()
        sess.username, sess.password = self.username, self.password
        sess.session_inactivity_timeout = 60000
        sess.connection_inactivity_timeout = 2000
        self._rt_session = SessionManager(self._rt_router)
        self._rt_session.CreateSession(sess)
        self._rt_cyclic = BaseCyclicClient(self._rt_router)
        print(f"[KORTEX] real-time feedback open on udp/{self.port_rt}")

    def close_realtime_feedback(self):
        try:
            if getattr(self, "_rt_session", None) is not None:
                self._rt_session.CloseSession()
            if getattr(self, "_rt_transport", None) is not None:
                self._rt_transport.disconnect()
        finally:
            self._rt_cyclic = None
            print("[KORTEX] real-time feedback closed")

    # NOTE: a "separate TCP session for gripper commands" was tried and
    # removed 2026-08-22 -- the arm explicitly REJECTS a write command from
    # any session other than the one currently in control
    # (KServerException ERROR_DEVICE/SESSION_NOT_IN_CONTROL), it does not
    # just ignore it. Only one session may write at a time; see
    # GRIPPER_RELEASE_PAUSE_S for the fix that actually works (a brief gap
    # in the SAME session's command stream).

    def read_gripper(self):
        """
        (position_percent, velocity) of the gripper finger motor.

        Uses the 1 kHz UDP channel when open, else the TCP one. Position is the
        arm's own percent-closed reading; `Base.GetMeasuredGripperMovement`
        reports the same quantity normalised to [0, 1] (verified on the lab arm:
        0.873 % vs 0.00873).
        """
        cyclic = getattr(self, "_rt_cyclic", None) or self._base_cyclic
        fb = cyclic.RefreshFeedback()
        motors = fb.interconnect.gripper_feedback.motor
        if not len(motors):
            raise RuntimeError("no gripper motor in interconnect feedback -- "
                               "is an end effector attached?")
        return float(motors[0].position), float(motors[0].velocity)

    def disconnect(self):
        try:
            if getattr(self, "_rt_cyclic", None) is not None:
                self.close_realtime_feedback()
            if self._session is not None:
                self._session.CloseSession()
            if self._transport is not None:
                self._transport.disconnect()
        finally:
            self.connected = False
            print("[KORTEX] disconnected")

    def read_joint_state(self):
        """
        Joint state in radians, wrapped to (-pi, pi].

        MEASURED on the lab arm (Gen3 L53K, SN WO545410-1, 2026-08-07): Kortex
        reports actuator position in degrees on **[0, 360)**, for limited joints
        as well as continuous ones. A bare deg2rad therefore returns e.g. 6.199
        rad for a joint physically at -4.8 deg, and 4.318 rad for a LIMITED
        joint (+-2.57 rad) physically at -112.6 deg -- a value outside that
        joint's own range, which no downstream check can interpret.

        Wrapping to (-pi, pi] is correct for every joint on this arm: the three
        limited joints span +-2.24 / +-2.57 / +-2.09 rad, all inside (-pi, pi],
        so the wrapped value is the physical angle. Continuous joints have no
        preferred representative and (-pi, pi] is as good as any -- `home()`
        takes the shortest path for those regardless.
        """
        # Prefer the UDP real-time channel when it is open. MEASURED on the lab
        # arm: this read costs 24.97 ms over TCP and 0.52 ms over UDP, because
        # the TCP RPC blocks until the base's next 25 ms command cycle. At a
        # 25 ms control period a TCP read HALVES the achieved command rate --
        # observed directly, 20 Hz against a 40 Hz target, when drift tracking
        # was first added. Instrumentation must never share the command channel.
        cyclic = getattr(self, "_rt_cyclic", None) or self._base_cyclic
        fb = cyclic.RefreshFeedback()
        pos = np.array([np.deg2rad(a.position) for a in fb.actuators[: self.n_dofs]])
        q = np.arctan2(np.sin(pos), np.cos(pos))          # -> (-pi, pi]
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
        """
        pos in [0,1]; 0=open, 1=closed (Kortex GRIPPER_POSITION).

        Always on the main command session -- the arm rejects a gripper write
        from any OTHER session while this one is in control
        (SESSION_NOT_IN_CONTROL), so a second session is not an option here.
        Sending this while SendJointSpeedsCommand is being actively streamed
        on this same session is silently ignored by the arm regardless; see
        GRIPPER_RELEASE_PAUSE_S in rehearse_or_throw for the actual fix (a
        brief gap in the stream, not a different channel).
        """
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
# Soft limits
# --------------------------------------------------------------------------- #
# The control mode Base.SendJointSpeedsCommand actually runs in. Soft limits are
# PER MODE, so reading or setting the wrong mode's limits tells you nothing about
# the commands you are sending.
JOINT_SPEED_CONTROL_MODE = "ANGULAR_JOYSTICK"


class SoftLimitManager:
    """
    Read / raise / restore the arm's SOFT kinematic limits.

    WHY THIS EXISTS. We planned the throw against the arm's HARD limits
    (80/70 deg/s, 297.9 deg/s^2) while the arm enforces the SOFT limits of the
    active control mode -- on this unit 50.0 deg/s and 57.3 deg/s^2 in
    ANGULAR_JOYSTICK. Commanding 74.4 deg/s on J0 against a 50 deg/s soft limit
    produced 0.5008 rad (28.7 deg) of position error at release. The threshold is
    exact: every speed_scale whose peak command stayed under 50 deg/s drifted
    ~0 rad; the one that crossed drifted 28 deg.

    Note `Base.GetAllJointsSpeedSoftLimitation` answers UNSUPPORTED_METHOD on
    this firmware, which is easy to misread as "soft limits unavailable". They
    live on ControlConfig and require the control mode as an argument.

    SAFETY. Raising a soft limit is deliberately not something this code does on
    its own:
      * requests are CLAMPED element-wise to the hard limits, which stay enforced
        by the arm underneath and are never touched;
      * every set is READ BACK and verified, failing closed on mismatch;
      * the previous values are written to a backup file, because the API has no
        notion of "restore defaults" -- if we do not record them, they are gone.
    Restore them when you are done. The next person to use the arm will not know
    it was left fast.
    """

    def __init__(self, backend, mode_name=JOINT_SPEED_CONTROL_MODE):
        _patch_collections_abc()
        from kortex_api.autogen.client_stubs.ControlConfigClientRpc import ControlConfigClient
        from kortex_api.autogen.messages import ControlConfig_pb2 as CC
        self._CC = CC
        self._cc = ControlConfigClient(backend._router)
        self._base = backend._base
        self.mode_name = mode_name
        self.mode = getattr(CC, mode_name)

    # -- reads ------------------------------------------------------------- #
    def active_mode(self):
        return str(self._cc.GetControlMode()).split(":")[-1].strip()

    def read_soft(self, mode_name=None):
        info = self._CC.ControlModeInformation()
        info.control_mode = getattr(self._CC, mode_name or self.mode_name)
        r = self._cc.GetKinematicSoftLimits(info)
        return {"speed": np.array(list(r.joint_speed_limits), float),
                "accel": np.array(list(r.joint_acceleration_limits), float),
                "twist_linear": float(r.twist_linear)}

    def read_hard(self):
        r = self._cc.GetKinematicHardLimits()
        return {"speed": np.array(list(r.joint_speed_limits), float),
                "accel": np.array(list(r.joint_acceleration_limits), float),
                "twist_linear": float(r.twist_linear)}

    # -- writes ------------------------------------------------------------ #
    def _set(self, speed_deg_s, accel_deg_s2):
        sp = self._CC.JointSpeedSoftLimits()
        sp.control_mode = self.mode
        for v in speed_deg_s:
            sp.joint_speed_soft_limits.append(float(v))
        self._cc.SetJointSpeedSoftLimits(sp)

        ac = self._CC.JointAccelerationSoftLimits()
        ac.control_mode = self.mode
        for v in accel_deg_s2:
            ac.joint_acceleration_soft_limits.append(float(v))
        self._cc.SetJointAccelerationSoftLimits(ac)

    def apply(self, speed_deg_s, accel_deg_s2, tol=1e-2):
        """Clamp to hard, set, read back, verify. Returns the achieved values."""
        hard = self.read_hard()
        speed = np.minimum(np.asarray(speed_deg_s, float), hard["speed"])
        accel = np.minimum(np.asarray(accel_deg_s2, float), hard["accel"])
        if np.any(np.asarray(speed_deg_s, float) > hard["speed"] + tol):
            print(f"[limits] request exceeded HARD speed limit; clamped to "
                  f"{np.round(hard['speed'], 2)}")
        if np.any(np.asarray(accel_deg_s2, float) > hard["accel"] + tol):
            print(f"[limits] request exceeded HARD accel limit; clamped to "
                  f"{np.round(hard['accel'], 1)}")
        self._set(speed, accel)
        got = self.read_soft()
        if (not np.allclose(got["speed"], speed, atol=1e-1)
                or not np.allclose(got["accel"], accel, atol=1.0)):
            raise RuntimeError(
                "soft-limit read-back does not match what was set.\n"
                f"  wanted speed {np.round(speed, 2)} accel {np.round(accel, 1)}\n"
                f"  got    speed {np.round(got['speed'], 2)} accel "
                f"{np.round(got['accel'], 1)}\n"
                "Refusing to continue on limits we cannot confirm.")
        return got

    def backup(self, path):
        cur = self.read_soft()
        rec = {"mode": self.mode_name,
               "speed_deg_s": cur["speed"].tolist(),
               "accel_deg_s2": cur["accel"].tolist()}
        with open(path, "w") as f:
            json.dump(rec, f, indent=2)
        return rec

    def restore(self, path):
        with open(path) as f:
            rec = json.load(f)
        if rec["mode"] != self.mode_name:
            raise RuntimeError(f"backup is for mode {rec['mode']}, not {self.mode_name}")
        return self.apply(rec["speed_deg_s"], rec["accel_deg_s2"])


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

    def __init__(self, limits: SafetyLimits, dry_run=True, ip="192.168.1.101",
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
    def precheck(self, coeffs, arm, n_samples=400, release_speed=None):
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
        # Release-instant quantisation is a LANDING-ERROR term, so it belongs in
        # the precheck next to torque and velocity, not in a footnote. The base
        # treats high-level commands every 1/control_hz s, so the gripper-open
        # command lands up to that late; at release speed v that is v/control_hz
        # metres of undershoot. On this arm (40 Hz, 1.5 m/s) it is ~3.7 cm --
        # larger than the whole 2.89 cm sim accuracy, and it is NOT reducible by
        # looping faster (see HIGH_LEVEL_MAX_HZ).
        dt_q = 1.0 / self.limits.control_hz
        quant = f"release quantisation: {dt_q * 1e3:.1f} ms at {self.limits.control_hz:.0f} Hz"
        if release_speed is not None:
            quant += (f"  ->  up to {float(release_speed) * dt_q * 100.0:.1f} cm "
                      f"of undershoot at {float(release_speed):.3f} m/s")
        report = [
            f"trajectory T={T:.3f}s, speed_scale={self.limits.speed_scale}",
            quant,
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
    @staticmethod
    def _homing_error(q_now, q_target, q_lo, q_hi):
        """
        Per-joint homing error that respects each joint's ACTUAL range.

        A single pi threshold is wrong here, and the lab arm proves both halves
        of why (measured 2026-08-07):

          * CONTINUOUS joints (URDF span >= 2pi; indices 0,2,4,6 on Gen3) can
            turn either way, so the error must be the SHORTEST angular path.
            Joint 2 sat at -177.4 deg with neutral at -2.3 deg: the direct
            difference sends it the long way round.
          * LIMITED joints (1,3,5) cannot wrap at all, so their error is the
            direct difference and may legitimately exceed pi. Joint 3 (+-2.57
            rad) sat at -1.966 rad with neutral at +1.460 -- a legal 3.426 rad
            (196 deg) sweep through zero. Wrapping that would drive it into its
            own limit, which is the opposite of safe.

        Returns (err, continuous_mask).
        """
        q_now = np.asarray(q_now, dtype=float)
        q_target = np.asarray(q_target, dtype=float)
        span = np.asarray(q_hi, dtype=float) - np.asarray(q_lo, dtype=float)
        continuous = span >= 2.0 * np.pi - 1e-6
        err = q_target - q_now
        shortest = np.arctan2(np.sin(err), np.cos(err))
        return np.where(continuous, shortest, err), continuous

    @staticmethod
    def _assert_readback_sane(q, q_lo, q_hi, margin=0.05):
        """
        Fail closed on a readback we cannot interpret -- distinguishing the TWO
        distinct causes, which need different fixes.

        (a) |q| > pi. Every joint on this arm is either continuous (where any
            representative in (-pi, pi] is valid) or limited to a range that
            fits inside (-pi, pi]. So a value past half a turn cannot be a pose;
            it is Kortex's [0, 360) reporting reaching us unwrapped. Fix in
            code, in read_joint_state.

        (b) |q| <= pi but outside this joint's URDF range. That IS a pose -- the
            arm is genuinely somewhere our kinematic model says it cannot be.
            Observed for real on 2026-08-07: joint 3 read -2.656 rad while being
            hand-guided, against a URDF limit of -2.570, i.e. the hardware's
            range is wider than the model's. Fix in the world (jog it back) or
            in the URDF -- not by unwrapping, which would corrupt a valid angle.

        Conflating these is not cosmetic: the first version of this guard
        reported (b) as a units bug, which would have sent someone editing
        working conversion code to chase a pose problem.
        """
        q = np.asarray(q, dtype=float)
        q_lo = np.asarray(q_lo, dtype=float)
        q_hi = np.asarray(q_hi, dtype=float)

        unwrapped = np.where(np.abs(q) > np.pi + 1e-9)[0]
        if unwrapped.size:
            raise RuntimeError(
                f"joint readback past pi on joints {list(unwrapped)}: "
                f"q={np.round(q[unwrapped], 3)} rad. This is a joint-angle "
                "WRAP/UNIT mismatch -- Kortex reports on [0,360) and the value "
                "reached us unwrapped. Fix read_joint_state, do not jog the arm."
            )

        lo, hi = q_lo - margin, q_hi + margin
        bad = np.where((q < lo) | (q > hi))[0]
        if bad.size:
            over = np.maximum(lo[bad] - q[bad], q[bad] - hi[bad])
            raise RuntimeError(
                f"joint readback OUTSIDE THE KINEMATIC MODEL on joints "
                f"{list(bad)}: q={np.round(q[bad], 3)} rad, outside URDF "
                f"[{np.round(q_lo[bad], 3)}, {np.round(q_hi[bad], 3)}] by "
                f"{np.round(over, 3)} rad (margin {margin}). The angles are "
                "well-formed, so this is NOT a units bug -- the arm is in a pose "
                "the model says is unreachable. Either jog it back inside range, "
                "or reconcile the URDF limits with the hardware's real ones "
                "(measured 2026-08-07: joint 3 reached -2.656 vs URDF -2.570). "
                "Refusing to plan from a pose the model cannot represent."
            )

    def home(self, arm, q_neutral, duration=4.0):
        """Slow, capped move to the neutral pose using proportional joint-speed
        servoing. Deliberately gentle -- this is the safe way to reach start."""
        q_neutral = np.asarray(q_neutral, dtype=float)
        q_lo, q_hi = np.asarray(arm._q_lo, float), np.asarray(arm._q_hi, float)
        dt = 1.0 / self.limits.control_hz
        # cap homing speed at a small fraction of qd_max regardless of speed_scale
        home_cap = 0.25 * self.limits.qd_max

        # Size the window against the ACTUAL distance to travel before starting.
        # Measured on the lab arm: neutral was 3.426 rad away on joint 3, which
        # needs 9.8 s at home_cap -- the 4.0 s default would have stopped the arm
        # part-way and left it in an undefined intermediate pose, which then
        # becomes the starting point of a throw. Extending a velocity-CAPPED move
        # does not make it faster, only complete.
        q0, _ = self.backend.read_joint_state() if not self.dry_run else (q_neutral * 0, None)
        self._assert_readback_sane(q0, q_lo, q_hi)
        err0, _ = self._homing_error(q0, q_neutral, q_lo, q_hi)
        t_needed = 1.25 * float(np.max(np.abs(err0) / home_cap))
        if t_needed > duration:
            print(f"[home] {np.max(np.abs(err0)):.3f} rad to travel needs ~{t_needed:.1f}s "
                  f"at the speed cap; extending duration {duration:.1f}s -> {t_needed:.1f}s")
            duration = t_needed

        t0 = time.time()
        print(f"[home] moving to neutral over ~{duration:.1f}s (capped, gentle)")
        while time.time() - t0 < duration:
            q, _ = self.backend.read_joint_state() if not self.dry_run else (q_neutral * 0, None)
            self._assert_readback_sane(q, q_lo, q_hi)
            err, _ = self._homing_error(q, q_neutral, q_lo, q_hi)
            qd = np.clip(2.0 * err, -home_cap, home_cap)  # P-servo, capped
            qd = self.limits.clamp_velocity(qd)
            self.backend.send_joint_velocities(qd)
            if np.max(np.abs(err)) < 0.01:
                break
            time.sleep(dt)
        self.backend.stop()
        print("[home] done")

    def set_gripper(self, closed: bool, confirm=True, timeout_s=3.0, tol_pct=3.0,
                    stall_window_s=0.3, min_progress_pct=10.0):
        """
        Command the gripper. With `confirm=True` (default), BLOCK until
        feedback shows it actually got there -- OR, if it's closing against
        a held object, until it stalls after genuine progress. Reaching the
        literal target percentage only happens when nothing is between the
        fingers; gripping a real object (measured 2026-08-22, tennis ball:
        stopped at 58.08%, position AND velocity both dead stable) means the
        motor stalls against the object well short of the "fully closed"
        value, by design -- treating that as a failure would refuse to ever
        throw a real ball. Success is therefore either (a) reaching within
        `tol_pct` of the target, or (b) having moved at least
        `min_progress_pct` from the starting position and then held still
        for `stall_window_s` -- (a) alone would still correctly reject the
        original bug below (zero motion, no progress, never satisfies (b)).

        `SendGripperCommand` is fire-and-forget: the arm's embedded controller
        keeps driving the motor after the call returns, asynchronously. The
        standalone `gripper` CLI command used to send this and immediately
        tear down the session in the same breath (`HardwareThrowExecutor.
        __exit__` -> disconnect) -- on the real arm that raced the motor and
        the command was observed to produce NO motion at all: 0.5-1.0 s later,
        with the session kept open, the same command converges cleanly
        (0.061 -> 0.86 at 0.5s -> 0.991 at 1.0s, measured 2026-08-22). The
        pre-throw grasp call in `rehearse_or_throw` (before `home()`) was
        accidentally safe against this -- `home()`'s ~10s gives the motor time
        regardless -- but that was luck, not a guarantee, and every other
        caller had none at all. `confirm=True` fails LOUDLY (raises) rather
        than reporting success on a gripper that never actually closed -- a
        throw with an unconfirmed grip either drops the ball early or never
        grips it at all, and both are silent failures if this just prints
        "commanded" and moves on.

        `confirm=False` is REQUIRED for the in-flight release call inside the
        40 Hz streaming loop (`rehearse_or_throw`, at s_fire). Confirming
        there polls `read_gripper()` in a loop on the SAME thread that must
        keep sending joint-speed commands every 25 ms -- measured directly
        (2026-08-22): confirming at release produced a single 693.55 ms late
        tick and 0.38 rad of open-loop drift right after release, because the
        arm coasted on its last commanded velocity, un-decelerated, for the
        entire stall instead of running the planned follow-through. The
        release-time call must stay fire-and-forget; only the pre-throw grasp
        and the standalone CLI command get to block.
        """
        target = self.gripper_closed if closed else self.gripper_open
        if not confirm:
            # Fire-and-forget path -- MUST NOT read feedback here. This is the
            # in-loop release call (rehearse_or_throw, s_fire): an extra
            # read_gripper() here was briefly introduced alongside the
            # already_holding check below and ran unconditionally, putting an
            # untested blocking call inside the 40 Hz command loop at the
            # exact release instant. Keep this branch a single send, nothing
            # else, always.
            self.backend.send_gripper(target)
            return
        opposite_pct = (self.gripper_open if closed else self.gripper_closed) * 100.0
        start_pct, start_vel = self.backend.read_gripper()
        # Already sitting closed-on-an-object from a PRIOR call (e.g. re-confirming
        # a grasp before a throw) -- there is no "progress" left to make this call,
        # so the progress-based stall check below would never fire. If we're already
        # stable and nowhere near the opposite (un-commanded) extreme, there's
        # nothing to do.
        already_holding = (closed and abs(start_pct - opposite_pct) > 15.0
                          and abs(start_vel) < 1e-3)
        self.backend.send_gripper(target)
        if already_holding:
            return
        target_pct = target * 100.0
        t0 = time.time()
        last_pct = start_pct
        last_move_t = t0
        pos_pct = start_pct
        while time.time() - t0 < timeout_s:
            pos_pct, _ = self.backend.read_gripper()
            if abs(pos_pct - target_pct) <= tol_pct:
                return
            if abs(pos_pct - last_pct) > 0.5:
                last_move_t = time.time()
            last_pct = pos_pct
            progressed = abs(pos_pct - start_pct) >= min_progress_pct
            stalled = (time.time() - last_move_t) >= stall_window_s
            if progressed and stalled:
                return  # stopped after real progress -- e.g. stalled against a held object
            time.sleep(0.05)
        raise RuntimeError(
            f"gripper did not reach {'CLOSED' if closed else 'OPEN'} "
            f"({target_pct:.1f}%) and never stalled after real progress, within "
            f"{timeout_s:.1f}s -- started {start_pct:.1f}%, stopped at {pos_pct:.1f}%. "
            f"Refusing to proceed; do not throw on an unconfirmed grip.")

    def rehearse_or_throw(self, coeffs, arm, verbose=True, track=None,
                          track_every=1):
        """
        Stream the throw with PER-PHASE time scaling.

        `speed_scale` applies to the THROW phase only; windup and follow-through
        run at `positioning_scale`. Scaling them together was a design flaw with
        a measured cost: J0 sweeps the full +178.7 deg azimuth during windup, is
        frozen (qd_release[0] = 0) through the throw, and sweeps back during
        follow-through -- it contributes NOTHING to release speed, yet uniform
        scaling drove it to 74.4 deg/s and produced 0.5008 rad (28.7 deg) of
        position error at release. Only the three axis-perpendicular joints
        (1, 3, 5) carry throw velocity; everything else is positioning, and
        positioning duration is irrelevant to the throw.

        Trajectory time s advances at a per-phase rate, and the commanded
        velocity is qd(s) * ds/dwall -- the same chain rule as before, applied
        piecewise:

            windup [0, t_w]   ds/dwall = positioning_scale
            throw  [t_w, t_r] ds/dwall = speed_scale
            follow [t_r, T]   ds/dwall = positioning_scale

        Escalating speed_scale now tests the throw alone and never re-tests
        positioning. Guaranteed stop on any exit.
        """
        t_w = float(coeffs["t_w"])
        t_r = float(coeffs["t_r"])
        T = float(coeffs["T"])
        scale = self.limits.speed_scale
        pos = self.limits.positioning_scale
        dt = 1.0 / self.limits.control_hz

        # Wall-clock duration of each phase, and the wall time at which each ends.
        w_windup = t_w / pos
        w_throw = (t_r - t_w) / scale
        w_follow = (T - t_r) / pos
        wall_T = w_windup + w_throw + w_follow
        wall_release = w_windup + w_throw

        def s_of_wall(wall):
            """Trajectory time and ds/dwall at a given wall time."""
            if wall < w_windup:
                return wall * pos, pos
            if wall < wall_release:
                return t_w + (wall - w_windup) * scale, scale
            return t_r + (wall - wall_release) * pos, pos

        if wall_T > self.limits.max_traj_seconds:
            raise RuntimeError(
                f"execution would take {wall_T:.1f}s > max_traj_seconds "
                f"{self.limits.max_traj_seconds}s; refuse."
            )

        # Fire the OPEN command early so the FINGERS move at trajectory time
        # t_r. The lead is a wall-clock delay, so in trajectory time it scales
        # with speed_scale: s advances at `scale` per wall-second, hence
        # s_fire = t_r - lead*scale. Getting this backwards would over-lead the
        # slow rehearsal by 1/scale and drop the ball before the swing.
        # Release sits at the END of the throw phase, so the lead converts at the
        # THROW phase's rate, not the positioning rate.
        lead = float(self.limits.gripper_lead_s)
        s_fire = t_r - lead * scale
        if s_fire < t_w:
            # lead reaches back past the start of the throw phase
            print(f"[exec] WARNING: gripper lead {lead:.3f}s exceeds the throw "
                  f"phase at speed_scale={scale}; firing at the start of the "
                  f"throw, release will be LATE.")
            s_fire = t_w

        released = False
        release_pause_until = None
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
            print(f"[exec] throw_scale={scale} pos_scale={pos} wall_T={wall_T:.2f}s "
                  f"(windup {w_windup:.2f} + throw {w_throw:.2f} + follow "
                  f"{w_follow:.2f}) release at s={t_r:.3f}s (wall {wall_release:.2f}s) "
                  f"rate={self.limits.control_hz:.0f}Hz")
        try:
            while True:
                wall = time.perf_counter() - t0
                s, ds_dwall = s_of_wall(wall)
                if s > T or wall > wall_T:
                    break
                q, qd, _ = arm.get_setpoint(coeffs, s, with_accel=True)
                qd_cmd = self.limits.clamp_velocity(np.asarray(qd) * ds_dwall)
                # Skip sending during the post-release pause window (see
                # GRIPPER_RELEASE_PAUSE_S) -- SendGripperCommand is silently
                # ignored by the arm while SendJointSpeedsCommand is being
                # actively streamed, so the OPEN command needs a real gap with
                # nothing competing on the command channel. The arm holds its
                # last commanded velocity for this whole window (checked
                # against joint limits at the real release state -- see that
                # constant's docstring for the margin).
                if release_pause_until is None or wall >= release_pause_until:
                    self.backend.send_joint_velocities(qd_cmd)
                # The throw is streamed OPEN-LOOP in velocity: q is computed and
                # then discarded. Any velocity-tracking error therefore
                # INTEGRATES into position error over the 8.5 s trajectory, and
                # at release that moves both the release point and the release
                # direction (v = J(q)*qd). Nothing corrects it and, until this
                # log existed, nothing measured it either. Recording planned vs
                # actual costs one feedback read per tick and turns an unknown
                # into a number.
                if track is not None and (tick % track_every) == 0:
                    try:
                        q_meas, _ = self.backend.read_joint_state()
                        track.append((s, np.asarray(q, float).copy(),
                                      np.asarray(q_meas, float).copy()))
                    except Exception:
                        pass
                if (not released) and s >= s_fire:
                    # confirm=False: this runs inside the 40 Hz streaming loop --
                    # blocking here to poll gripper feedback stalls the loop and
                    # the arm coasts uncommanded until it returns. See set_gripper.
                    self.set_gripper(closed=False, confirm=False)  # OPEN -> release
                    released = True
                    release_wall = wall
                    release_pause_until = wall + GRIPPER_RELEASE_PAUSE_S
                    if verbose:
                        print(f"[exec] gripper OPEN commanded at wall={wall:.3f}s "
                              f"(s={s:.3f}); fingers expected to move at "
                              f"s={t_r:.3f} after {lead:.3f}s lead; pausing joint-"
                              f"speed commands until wall={release_pause_until:.3f}s")
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
        if track:
            arr_s = np.array([t[0] for t in track])
            err = np.array([t[2][:self.n_dofs] - t[1][:self.n_dofs] for t in track])
            # wrap: continuous joints can cross +-pi mid-trajectory
            err = np.arctan2(np.sin(err), np.cos(err))
            i_rel = int(np.argmin(np.abs(arr_s - t_r)))
            self.last_exec_stats.update({
                "drift_max_rad": float(np.max(np.abs(err))),
                "drift_at_release_rad": float(np.max(np.abs(err[i_rel]))),
                "drift_final_rad": float(np.max(np.abs(err[-1]))),
                "drift_samples": len(track),
            })
            if verbose:
                print(f"[exec] OPEN-LOOP DRIFT (planned vs actual): "
                      f"max {np.max(np.abs(err)):.4f} rad, "
                      f"at release {np.max(np.abs(err[i_rel])):.4f} rad, "
                      f"end {np.max(np.abs(err[-1])):.4f} rad")
        if verbose:
            print(f"[exec] trajectory complete, arm stopped "
                  f"({tick} ticks, {achieved_hz:.0f}Hz achieved vs "
                  f"{self.limits.control_hz:.0f}Hz target, worst tick "
                  f"{worst_late * 1e3:.2f}ms late)")
            if achieved_hz < 0.9 * self.limits.control_hz:
                print("[exec] WARNING: achieved rate is >10% below target -- the "
                      "control loop is not keeping up; release timing is degraded.")
        return released
