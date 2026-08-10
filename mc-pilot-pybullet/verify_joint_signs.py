"""
Verify per-joint SIGN and DIRECTION against the real arm, one joint at a time.

WHY FK ALONE IS NOT ENOUGH
--------------------------
Comparing our URDF's forward kinematics against the arm's reported Cartesian
pose (done 2026-08-07, agreed to 0.82 mm) proves that the arm's POSITION
FEEDBACK maps onto our model correctly -- joint order, per-joint sign, zero
offsets, base frame, all at once.

It proves nothing about the direction the arm moves when we COMMAND it. We
drive the throw with Base.SendJointSpeedsCommand, and nothing so far has
checked that a positive commanded joint speed produces a positive position
change in that same convention. If any joint were inverted, the throw would
swing that joint backwards -- through the workspace, at up to 80 deg/s.

That is the single failure mode with the worst consequences and the least
coverage, so it gets its own test, at a speed where being wrong is harmless.

METHOD
------
For each joint independently:
  * read q0 over the 1 kHz UDP feedback channel
  * command +TEST_SPEED on that joint alone, zeros on the other six
  * hold for TEST_SECONDS, then command all-zero and let it settle
  * read q1, and check sign(q1 - q0) == +1 and the magnitude is sane
  * command the exact negative to return, and check it comes back

Defaults move each joint by about 1.4 deg at 2.9 deg/s -- 3.6% of the arm's own
80 deg/s limit. Every commanded value passes through the same SafetyLimits
clamp the throw uses.

REFUSES TO MOVE A JOINT if the motion would take it within JOINT_MARGIN of its
URDF limit, and aborts the whole run on any unexpected reading.

    python3 verify_joint_signs.py                      # dry-run
    python3 verify_joint_signs.py --arm --confirm      # real, ~1 min
"""

import argparse
import sys
import time

import numpy as np

sys.path.append("..")

import pybullet as p

import run_hardware_throw as H
from robot_arm.kinova_hardware import HardwareThrowExecutor, _patch_collections_abc

TEST_SPEED = 0.05        # rad/s  (2.9 deg/s, 3.6% of the 1.396 rad/s limit)
TEST_SECONDS = 0.5       # -> ~0.025 rad = 1.4 deg of travel
JOINT_MARGIN = 0.35      # rad of clearance required from any URDF limit
SETTLE = 0.6


def _read(ex, n):
    q, _ = ex.backend.read_joint_state()
    return np.asarray(q, float)[:n]


def _delta(q_after, q_before, continuous):
    """
    Angular change, wrapped for continuous joints.

    Caught in the act on 2026-08-07: joint 2 sits at -179.1 deg, right on the
    +-180 boundary, so its return move crossed the wrap and a raw subtraction
    reported +358.488 deg for what was actually -1.512 deg. This is the SAME
    trap already fixed in home() -- any difference of two angles needs it, and
    a fresh script is exactly where it comes back.
    """
    d = np.asarray(q_after, float) - np.asarray(q_before, float)
    wrapped = np.arctan2(np.sin(d), np.cos(d))
    return np.where(continuous, wrapped, d)


def _drive(ex, n, joint, speed, seconds, hz=40.0):
    """Command one joint, zeros elsewhere, then stop. Returns commands sent."""
    dt = 1.0 / hz
    qd = np.zeros(n)
    qd[joint] = speed
    qd = ex.limits.clamp_velocity(qd)          # same clamp the throw uses
    t0 = time.perf_counter()
    n_cmd = 0
    try:
        while time.perf_counter() - t0 < seconds:
            ex.backend.send_joint_velocities(qd)
            n_cmd += 1
            time.sleep(dt)
    finally:
        ex.backend.send_joint_velocities(np.zeros(n))
    time.sleep(SETTLE)
    return n_cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default="192.168.1.101")
    ap.add_argument("--robot", default="kinova_gen3_dyn")
    ap.add_argument("--arm", action="store_true")
    ap.add_argument("--confirm", action="store_true",
                    help="assert workspace clear and e-stop in hand")
    ap.add_argument("--joints", type=int, nargs="*", default=None,
                    help="subset to test (default: all 7)")
    ap.add_argument("--speed", type=float, default=TEST_SPEED)
    ap.add_argument("--seconds", type=float, default=TEST_SECONDS)
    args = ap.parse_args()

    if args.arm and not args.confirm:
        print("REFUSED: moving the arm needs --confirm (workspace clear, e-stop in hand).")
        return 2

    _patch_collections_abc()
    arm, profile, cid = H.build_arm(args.robot)
    n = len(profile.qd_max)
    q_lo, q_hi = np.asarray(arm._q_lo, float), np.asarray(arm._q_hi, float)
    limits = H.make_limits(profile, 1.0, arm=arm)
    joints = args.joints if args.joints is not None else list(range(n))

    rows, failures = [], []
    with HardwareThrowExecutor(limits, dry_run=not args.arm, ip=args.ip) as ex:
        if args.arm:
            ex.backend.open_realtime_feedback()
        try:
            expected = args.speed * args.seconds
            print(f"\ncommanding {args.speed:.3f} rad/s for {args.seconds:.2f}s "
                  f"-> expect ~{np.degrees(expected):.2f} deg per joint "
                  f"({'REAL ARM' if args.arm else 'DRY-RUN, no motion'})\n")
            for j in joints:
                q0 = _read(ex, n)
                room_hi, room_lo = q_hi[j] - q0[j], q0[j] - q_lo[j]
                if min(room_hi, room_lo) < JOINT_MARGIN:
                    print(f"  j{j}: SKIPPED -- only {min(room_hi,room_lo):.3f} rad "
                          f"from a limit (need {JOINT_MARGIN})")
                    continue
                _drive(ex, n, j, +args.speed, args.seconds)
                q1 = _read(ex, n)
                _drive(ex, n, j, -args.speed, args.seconds)
                q2 = _read(ex, n)

                cont = (q_hi - q_lo) >= 2.0 * np.pi - 1e-6
                d = _delta(q1, q0, cont)[j]
                back = _delta(q2, q1, cont)[j]
                others = np.max(np.abs(np.delete(_delta(q1, q0, cont), j)))
                ok_sign = d > 0
                ok_mag = 0.3 * expected < abs(d) < 3.0 * expected
                ok_back = back < 0
                ok_iso = others < 0.02
                good = ok_sign and ok_mag and ok_back and ok_iso
                rows.append((j, d, back, others, good))
                if not good:
                    failures.append(j)
                print(f"  j{j}: +cmd -> {np.degrees(d):+7.3f} deg  "
                      f"-cmd -> {np.degrees(back):+7.3f} deg  "
                      f"other joints max {np.degrees(others):.3f} deg   "
                      f"{'OK' if good else 'FAIL'}"
                      + ("" if ok_sign else "  <-- SIGN INVERTED")
                      + ("" if ok_mag else "  <-- MAGNITUDE OFF")
                      + ("" if ok_iso else "  <-- CROSS-TALK"))
                if not good and args.arm:
                    print("  aborting: unexpected response, not testing further joints.")
                    break
        finally:
            if args.arm:
                ex.backend.close_realtime_feedback()

    p.disconnect(cid)
    print("\n--- summary " + "-" * 50)
    if not args.arm:
        print("DRY-RUN: the stub backend does not integrate velocity, so no "
              "direction was measured. Plumbing only.")
        return 0
    print(f"{len(rows)} joint(s) tested, {len(failures)} FAILED")
    if failures:
        print(f"  FAILED joints: {failures}")
        print("  DO NOT RUN A THROW. A commanded sign that does not match the "
              "feedback convention would swing that joint backwards at speed.")
        return 1
    print("  every tested joint moves POSITIVE for a POSITIVE command, returns "
          "on the negative, and does not disturb its neighbours.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
