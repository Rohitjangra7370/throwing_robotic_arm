"""
STRICTLY READ-ONLY Kinova Gen3 bench check. Sends NO command to the arm.

Stage 1 of the bring-up, done properly. `run_hardware_throw.py connect` writes
exactly one thing -- the teardown `stop()` (zero joint speeds) -- which is safe
but is still a write. This script performs zero writes: it opens a Kortex
session, reads, and closes. Nothing here can move the arm.

What it answers, in the order the answers matter:
  1. Is this a Gen3, and which one (serial / firmware / DOF count)?
  2. Is it faulted, and what servoing/operating mode is it sitting in?
  3. Do the arm's OWN velocity and torque limits agree with the numbers in
     robot_profiles.py? Every feasibility check in this repo -- the release LP,
     the whole-trajectory precheck, the follow-through guard -- is built on
     qd_max = 1.3963/1.2218 rad/s and tau_max = 39/9 Nm. If the arm disagrees,
     every one of those checks has been validating against fiction.
  4. Does the joint readback come back in the convention read_joint_state()
     assumes? (HARDWARE_RUNBOOK R1: Kortex reports Gen3 continuous joints in
     [0,360), q_neutral holds small negative angles, and a naive deg2rad would
     make home() servo the long way around.)

Usage:
    python3 hw_readonly_check.py --ip 192.168.1.101
"""

import argparse
import sys

import numpy as np

sys.path.append("..")

from robot_arm.kinova_hardware import (HardwareThrowExecutor, _KortexBackend,
                                       _patch_collections_abc)
from robot_arm.robot_profiles import get_robot_profile


def _urdf_joint_ranges(robot, n):
    """Per-joint (lo, hi) from the URDF -- the same source ArmController uses."""
    import pybullet as p
    import pybullet_data
    profile = get_robot_profile(robot)
    cid = p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
        aid = p.loadURDF(pybullet_data.getDataPath() + "/" + profile.urdf_rel_path,
                         useFixedBase=True, physicsClientId=cid)
        lo, hi = [], []
        for jid in list(profile.joint_ids)[:n]:
            ji = p.getJointInfo(aid, jid, physicsClientId=cid)
            lo.append(ji[8])
            hi.append(ji[9])
        return np.array(lo, float), np.array(hi, float)
    finally:
        p.disconnect(cid)

OK, WARN, FAIL = "  OK  ", " CHECK", " FAIL "
_results = []


def _record(level, msg):
    _results.append((level, msg))
    print(f"[{level}] {msg}")


def _probe(label, fn):
    """Run one read. A wrong method name is a finding, not a crash."""
    try:
        return fn()
    except Exception as e:
        _record(WARN, f"{label}: unavailable ({type(e).__name__}: {e})")
        return None


def section(title):
    print(f"\n--- {title} " + "-" * max(0, 60 - len(title)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ip", default="192.168.1.101")
    ap.add_argument("--robot", default="kinova_gen3_dyn")
    ap.add_argument("--username", default="admin")
    ap.add_argument("--password", default="admin")
    args = ap.parse_args()

    _patch_collections_abc()
    profile = get_robot_profile(args.robot)
    n = len(profile.qd_max)

    be = _KortexBackend(n, ip=args.ip)
    be.username, be.password = args.username, args.password
    be.connect()                      # session open; no command sent
    try:
        from kortex_api.autogen.client_stubs.DeviceConfigClientRpc import DeviceConfigClient
        dev = DeviceConfigClient(be._router)
        base = be._base

        # ---------------------------------------------------------------- #
        section("1. identity")
        for label, fn in (
            ("device type", lambda: dev.GetDeviceType()),
            ("model number", lambda: dev.GetModelNumber()),
            ("serial number", lambda: dev.GetSerialNumber()),
            ("part number", lambda: dev.GetPartNumber()),
            ("firmware", lambda: dev.GetFirmwareVersion()),
            ("MAC", lambda: dev.GetMACAddress()),
        ):
            r = _probe(label, fn)
            if r is not None:
                _record(OK, f"{label}: {str(r).strip().replace(chr(10), ' ')}")

        n_act = _probe("actuator count", lambda: base.GetActuatorCount().count)
        if n_act is not None:
            lvl = OK if n_act == n else FAIL
            _record(lvl, f"actuator count: {n_act} (profile '{args.robot}' expects {n})")

        # ---------------------------------------------------------------- #
        section("2. state / faults  (must be fault-free before ANY motion)")
        for label, fn in (
            ("arm state", lambda: base.GetArmState()),
            ("servoing mode", lambda: base.GetServoingMode()),
            ("operating mode", lambda: base.GetOperatingMode()),
        ):
            r = _probe(label, fn)
            if r is not None:
                _record(OK, f"{label}: {str(r).strip().replace(chr(10), ' ')}")

        # Seen for real on 2026-08-07: between two runs of this script the arm
        # went ARMSTATE_SERVOING_READY -> ARMSTATE_SERVOING_MANUALLY_CONTROLLED
        # because someone was driving it from the web UI / joystick. Streaming
        # joint speeds while another client holds the arm is a two-master
        # situation; the arm must be handed back before stage 2.
        st = _probe("arm state (gate)", lambda: base.GetArmState())
        if st is not None:
            name = st.active_state if isinstance(st.active_state, str) else str(st.active_state)
            ready = "SERVOING_READY" in str(st).upper()
            manual = "MANUALLY_CONTROLLED" in str(st).upper()
            # Anything that is not SERVOING_READY means some other client is
            # driving. Seen live on 2026-08-07: MANUALLY_CONTROLLED (web UI /
            # joystick) and PLAYING_SEQUENCE (a sequence running from the web
            # UI). Both are the same hazard -- streaming joint speeds into an
            # arm another master is already commanding -- so both are a stop,
            # not a warning. The enum may also arrive as a bare int.
            if ready:
                _record(OK, "arm state gate: SERVOING_READY -- free for us to drive")
            else:
                why = ("ARMSTATE_SERVOING_MANUALLY_CONTROLLED -- web UI / joystick "
                       "holds the arm" if manual else
                       "ARMSTATE_SERVOING_PLAYING_SEQUENCE -- a sequence is running"
                       if "PLAYING_SEQUENCE" in str(st).upper() else
                       f"state {name!r} is not SERVOING_READY")
                _record(FAIL, f"arm state gate: {why}. Another client is driving this "
                              "arm. Stop/release it there before any stage-2 motion; do "
                              "not stream joint speeds into a second master.")

        # ---------------------------------------------------------------- #
        section("3. arm's OWN limits vs robot_profiles.py")
        def _vals(resp, field):
            # The field is `joints_limitations` (plural/plural), each entry a
            # {joint_identifier, type, value}. Entry 0 omits joint_identifier
            # because protobuf 3 drops zero-valued scalars, so index by order.
            jl = getattr(resp, "joints_limitations", None)
            if jl is None:
                return None
            return [float(x.value) for x in jl]

        # SOFT limits answer UNSUPPORTED_METHOD on this firmware (measured); the
        # HARD ones are the safety-relevant pair anyway.
        for label, fn, ours, unit, conv in (
            ("speed HARD", lambda: base.GetAllJointsSpeedHardLimitation(),
             np.rad2deg(profile.qd_max), "deg/s", 1.0),
            ("torque HARD", lambda: base.GetAllJointsTorqueHardLimitation(),
             np.asarray(profile.tau_max, float), "Nm", 1.0),
        ):
            resp = _probe(label, fn)
            if resp is None:
                continue
            got = _vals(resp, None)
            if not got:
                _record(WARN, f"{label}: response had no joint_limits field: {resp}")
                continue
            got = np.asarray(got, float) * conv
            _record(OK, f"{label} (arm): {np.round(got, 2)} {unit}")
            _record(OK, f"{label} (ours): {np.round(ours, 2)} {unit}")
            if len(got) == len(ours):
                # ours must be <= the arm's, otherwise our checks pass plans the
                # arm will refuse or clamp. Tolerance is RELATIVE: the arm sends
                # float32 degrees (80.00209045410156) and ours is a float64
                # rad2deg of 1.3963, so an absolute 1e-6 threshold flags a
                # ~3e-6 deg/s difference as a limit violation. It is not one.
                over = np.where(ours > got * (1.0 + 1e-4))[0]
                if over.size:
                    _record(FAIL, f"{label}: OUR limit EXCEEDS the arm's on joints "
                                  f"{list(over)} -- our feasibility checks are optimistic")
                else:
                    _record(OK, f"{label}: ours is within the arm's on all joints")

        # ---------------------------------------------------------------- #
        section("4. live feedback + R1 angle-convention gate")
        fb = _probe("RefreshFeedback", lambda: be._base_cyclic.RefreshFeedback())
        if fb is not None:
            acts = list(fb.actuators)[:n]
            pos_deg = np.array([a.position for a in acts], float)
            vel_deg = np.array([a.velocity for a in acts], float)
            tau = np.array([a.torque for a in acts], float)
            temp = np.array([getattr(a, "temperature_motor", np.nan) for a in acts], float)
            volt = np.array([getattr(a, "voltage", np.nan) for a in acts], float)
            _record(OK, f"position (deg, raw): {np.round(pos_deg, 2)}")
            _record(OK, f"velocity (deg/s):    {np.round(vel_deg, 3)}")
            _record(OK, f"torque (Nm):         {np.round(tau, 2)}")
            _record(OK, f"motor temp (C):      {np.round(temp, 1)}")
            _record(OK, f"actuator voltage(V): {np.round(volt, 1)}")

            if np.max(np.abs(vel_deg)) > 0.5:
                _record(FAIL, f"arm is MOVING ({np.max(np.abs(vel_deg)):.2f} deg/s) -- "
                              "nothing here commanded it; stop and investigate")
            else:
                _record(OK, "arm is stationary (|qd| < 0.5 deg/s)")

            hot = np.where(temp > 60.0)[0]
            if hot.size:
                _record(WARN, f"actuators {list(hot)} above 60 C: {np.round(temp[hot],1)}")

            # --- the R1 gate, through the SHIPPED code path -------------- #
            # Deliberately calls the real read_joint_state / _homing_error /
            # _assert_readback_sane rather than reimplementing them, so this
            # check validates what stage 2 will actually execute.
            q_neutral = np.asarray(profile.q_neutral, float)
            q_lo, q_hi = _urdf_joint_ranges(args.robot, n)
            q_naive = np.deg2rad(pos_deg)
            q_shipped, _ = be.read_joint_state()

            _record(OK, f"raw deg2rad (pre-fix behaviour): {np.round(q_naive, 3)}")
            _record(OK, f"read_joint_state() returns:      {np.round(q_shipped, 3)}")
            _record(OK, f"profile q_neutral (rad):         {np.round(q_neutral, 3)}")

            past_pi = np.where(np.abs(q_naive) > np.pi)[0]
            if past_pi.size:
                _record(OK, f"R1 present on this arm: joints {list(past_pi)} report past pi "
                            f"({np.round(q_naive[past_pi],3)} rad) -- Kortex is on [0,360). "
                            "read_joint_state() wraps to (-pi,pi], so this is handled.")
            else:
                _record(OK, "R1 not exercised right now: no joint reads past pi")

            try:
                HardwareThrowExecutor._assert_readback_sane(q_shipped, q_lo, q_hi)
                _record(OK, "readback lies inside every joint's URDF range")
            except RuntimeError as e:
                _record(FAIL, f"readback range guard would REFUSE stage 2: {e}")

            err, cont = HardwareThrowExecutor._homing_error(q_shipped, q_neutral, q_lo, q_hi)
            for j in range(n):
                _record(OK, f"  j{j} {'CONT' if cont[j] else 'LIM '}: q={q_shipped[j]:+7.3f} "
                            f"-> neutral {q_neutral[j]:+7.3f}   err {err[j]:+7.3f} rad "
                            f"({np.rad2deg(err[j]):+7.1f} deg)")
            cap = 0.25 * np.asarray(profile.qd_max, float)
            t_needed = 1.25 * float(np.max(np.abs(err) / cap))
            _record(OK, f"max |err| = {np.max(np.abs(err)):.3f} rad on joint "
                        f"{int(np.argmax(np.abs(err)))}; homing needs ~{t_needed:.1f} s "
                        f"at the 0.25*qd_max cap (home() auto-extends its window)")
            if np.any(np.abs(err[cont]) > np.pi + 1e-6):
                _record(FAIL, "a CONTINUOUS joint's error exceeds pi after shortest-path "
                              "reduction -- that should be impossible; do not proceed")

        ang = _probe("GetMeasuredJointAngles", lambda: base.GetMeasuredJointAngles())
        if ang is not None and fb is not None:
            got = np.array([j.value for j in ang.joint_angles][:n], float)
            d = np.max(np.abs(((got - pos_deg + 180.0) % 360.0) - 180.0))
            lvl = OK if d < 1.0 else WARN
            _record(lvl, f"GetMeasuredJointAngles agrees with RefreshFeedback to {d:.3f} deg")

        pose = _probe("GetMeasuredCartesianPose", lambda: base.GetMeasuredCartesianPose())
        if pose is not None:
            _record(OK, f"EE pose (m/deg): x={pose.x:.4f} y={pose.y:.4f} z={pose.z:.4f} "
                        f"tx={pose.theta_x:.1f} ty={pose.theta_y:.1f} tz={pose.theta_z:.1f}")
    finally:
        be.disconnect()               # session close; still no command sent

    section("summary")
    nf = sum(1 for l, _ in _results if l == FAIL)
    nw = sum(1 for l, _ in _results if l == WARN)
    print(f"{len(_results)} checks: {nf} FAIL, {nw} CHECK, {len(_results)-nf-nw} OK")
    if nf:
        print("\nFAILures:")
        for l, m in _results:
            if l == FAIL:
                print("  -", m)
    print("\nNo command was sent to the arm.")
    return 1 if nf else 0


if __name__ == "__main__":
    sys.exit(main())
