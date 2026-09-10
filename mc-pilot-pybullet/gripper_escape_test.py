"""
Find the gripper position at which the ball actually LEAVES the hand.

WHY THIS EXISTS
---------------
`measure_gripper_latency.py` measures when the FINGERS start moving (onset,
67-68 ms, already compensated by GRIPPER_RELEASE_LATENCY_S). It cannot measure
when the BALL leaves, and with the Robotiq 2F-85 in an ENCOMPASSING grip -- the
fingertips wrapped past the ball's equator, confirmed visually on this rig
2026-09-09 -- those are very different instants: the ball is caged and has to
wait for real finger travel.

That gap is not a detail. Measured 2026-09-09, integrating the arm forward at
its held joint velocity through GRIPPER_RELEASE_PAUSE_S and re-solving the
ballistic landing, the departure instant sets the SIGN of the whole release
error:

    departs at onset  -> -12.1 cm (short)      +76 ms -> 0.0 cm
    +30 ms            ->  -7.1 cm              +140ms -> +8.7 cm (long)

The arm's own 36 ms command lag shortens the throw; a late departure lengthens
it, because during the coast window the wrist keeps SWINGING and the release
elevation falls (20 deg -> 0 deg over 200 ms). They cancel near +76 ms. So
"apply the lag correction" is not safe to do until this number exists.

Gripper POSITION feedback alone cannot supply it: thresholds from +1 to +20
percentage points of finger travel span 4 ms to 140 ms. This needs a fact from
outside the gripper, which is what this test is.

HOW IT WORKS
------------
The ball must be held IN THE AIR so it can fall when it escapes.

  1. close fully -> the motor stalls on the ball (33.33% on this tennis ball)
  2. open to a probe position X
  3. close fully again and read where it stalls:
        stalls in the holding band  -> ball still caged at X
        runs to ~99-100%            -> ball escaped and fell

Step 3 is a normal grasp, not the 2026-08-22 fault shape: at X the fingers are
WIDER than at the stall, so the ball is loose in the cage and the motor is not
pushing into anything. The hazardous case -- commanding close at an already
stalled gripper -- is refused explicitly.

Binary search over X converges in ~5 probes; roughly half drop the ball, so
expect to re-seat it about three times.

    python3 gripper_escape_test.py                             # dry-run
    python3 gripper_escape_test.py --arm --confirm             # real
"""
import argparse
import sys
import time

import numpy as np

sys.path.append("..")

from robot_arm.kinova_hardware import (HardwareThrowExecutor, SafetyLimits,
                                       _patch_collections_abc)
from robot_arm.robot_profiles import get_robot_profile
from measure_gripper_latency import GRASP_THRESHOLD_PCT, OPEN_PCT, holding_something


def next_probe(lo, hi):
    """Midpoint of the current bracket, in percent-closed."""
    return 0.5 * (lo + hi)


def update_bracket(lo, hi, probe, caged):
    """
    Shrink the bracket after one probe.

    Percent-closed is MORE closed at higher values, so the ball is caged ABOVE
    the escape position and free below it. `lo` is therefore always a position
    known to release and `hi` one known to hold -- the opposite of the usual
    convention, and worth stating because getting it backwards silently
    converges on the wrong end of the bracket.
    """
    if caged:
        return lo, probe
    return probe, hi


def escape_time_from_traces(traces, x_escape, onset_eps=0.5):
    """
    Convert an escape POSITION into an escape TIME using recorded open traces.

    `traces` are (t, position, velocity) arrays from
    measure_gripper_latency.py's OPEN transitions, i.e. the same motion a real
    throw performs. Returns (t_escape, t_onset) means over the traces, in
    seconds, skipping any trace that never reaches x_escape.
    """
    t_esc, t_on = [], []
    for a in np.asarray(traces, dtype=object):
        a = np.asarray(a, dtype=float)
        if a.size == 0:
            continue
        t, pos = a[:, 0], a[:, 1]
        p0 = pos[0]
        moved = np.abs(pos - p0) > onset_eps
        if not moved.any():
            continue
        t_on.append(float(t[np.argmax(moved)]))
        # The trace must START more closed than the escape position, or it was
        # already past it before the command was even issued and `pos <= x`
        # is true at sample 0 -- which returns t[0], i.e. an "escape" BEFORE
        # the fingers moved. Refuse instead of reporting a negative remainder.
        if p0 <= x_escape:
            continue
        below = pos <= x_escape          # opening -> position falls
        if below.any():
            t_esc.append(float(t[np.argmax(below)]))
    mean_on = float(np.mean(t_on)) if t_on else None
    if not t_esc:
        return None, mean_on
    mean_esc = float(np.mean(t_esc))
    if mean_on is not None and mean_esc < mean_on:
        # Physically impossible: the ball cannot leave before the fingers move.
        return None, mean_on
    return mean_esc, mean_on


def _wait_for_ball(seat_delay):
    """
    Hand the operator the gripper between probes.

    Blocks on Enter by default. `--seat_delay` swaps that for a countdown so the
    run can be driven non-interactively -- the fingers are already open and
    stationary either way, so the only thing the delay controls is how long
    there is to place the ball.
    """
    if seat_delay and seat_delay > 0:
        print(f"  seat the ball in the OPEN fingers, hands CLEAR "
              f"({seat_delay:.0f}s)...", flush=True)
        for left in range(int(seat_delay), 0, -1):
            if left % 5 == 0 or left <= 3:
                print(f"    {left}s", flush=True)
            time.sleep(1.0)
        print("  closing.", flush=True)
    else:
        input("  seat the ball in the OPEN fingers, hands CLEAR, "
              "then press Enter... ")


def _settle(backend, timeout=3.0, vel_eps=0.5, min_samples=40):
    """Poll until the finger motor has clearly stopped, then return position."""
    t0 = time.perf_counter()
    n = 0
    pos, vel = backend.read_gripper()
    while time.perf_counter() - t0 < timeout:
        pos, vel = backend.read_gripper()
        n += 1
        if n > min_samples and abs(vel) < vel_eps:
            break
        time.sleep(0.002)
    return pos


def probe_once(ex, x_pct, settle_s, arm_live):
    """
    One probe: grasp -> open to x_pct -> re-grasp -> did the ball survive?

    Returns (caged, grasp_pct, regrasp_pct) or None if the initial grasp failed.
    """
    pre = _settle(ex.backend)
    if holding_something(pre) and arm_live:
        print(f"  REFUSED: gripper is at {pre:.1f}%, already stalled on "
              f"something. Commanding close again is the 2026-08-22 fault "
              f"shape. Open it first.")
        return None

    ex.backend.send_gripper(1.0)                  # grasp
    grasp = _settle(ex.backend)
    if not holding_something(grasp) and arm_live:
        print(f"  grasp = {grasp:.2f}% -> CLOSED ON NOTHING, no ball. "
              f"Skipping this probe rather than recording a false escape.")
        return None

    ex.backend.send_gripper(x_pct / 100.0)        # open to the probe position
    at_x = _settle(ex.backend)
    time.sleep(settle_s)                          # let a freed ball actually fall

    ex.backend.send_gripper(1.0)                  # re-grasp
    regrasp = _settle(ex.backend)
    caged = holding_something(regrasp)
    print(f"  probe X={x_pct:5.2f}%  (reached {at_x:5.2f}%)  re-grasp "
          f"{regrasp:6.2f}%  -> {'STILL CAGED' if caged else 'BALL ESCAPED'}")
    return caged, grasp, regrasp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default="192.168.1.101")
    ap.add_argument("--robot", default="kinova_gen3_dyn")
    ap.add_argument("--arm", action="store_true", help="talk to the REAL arm")
    ap.add_argument("--confirm", action="store_true",
                    help="assert the ball can FALL FREELY from the gripper and "
                         "hands are clear of the fingers")
    ap.add_argument("--probes", type=int, default=6)
    ap.add_argument("--lo", type=float, default=0.0,
                    help="percent-closed known to RELEASE (fully open)")
    ap.add_argument("--hi", type=float, default=None,
                    help="percent-closed known to HOLD; default = the measured "
                         "grasp position of the first successful probe")
    ap.add_argument("--seat_delay", type=float, default=0.0,
                    help="seconds to wait for the ball to be seated, instead of "
                         "blocking on Enter. Lets the run be driven "
                         "non-interactively; 0 = prompt.")
    ap.add_argument("--settle", type=float, default=1.0,
                    help="seconds to wait at the probe position for a freed "
                         "ball to fall clear before re-grasping")
    ap.add_argument("--traces", default="results_gripper_latency_loaded.npz",
                    help="open-transition traces used to convert the escape "
                         "POSITION into an escape TIME")
    args = ap.parse_args()

    if args.arm and not args.confirm:
        print("REFUSED: this drops the ball repeatedly and cycles the fingers. "
              "--confirm asserts the ball can fall freely and hands are clear.")
        return 2

    _patch_collections_abc()
    profile = get_robot_profile(args.robot)
    limits = SafetyLimits(qd_max=np.array(profile.qd_max, float),
                          q_soft_lo=-6.10 * np.ones(len(profile.qd_max)),
                          q_soft_hi=6.10 * np.ones(len(profile.qd_max)),
                          speed_scale=1.0)

    results = []
    with HardwareThrowExecutor(limits, dry_run=not args.arm, ip=args.ip) as ex:
        ex.backend.open_realtime_feedback()
        try:
            lo, hi = args.lo, args.hi
            for i in range(args.probes):
                print(f"\n[{i+1}/{args.probes}]")
                # Open BEFORE asking for the ball -- the fingers may still be
                # holding one from the previous probe, and "seat the ball in the
                # OPEN fingers" is not an instruction anyone can follow while
                # they are shut.
                ex.backend.send_gripper(0.0)
                _settle(ex.backend)
                if args.arm:
                    _wait_for_ball(args.seat_delay)
                if hi is None:
                    out = probe_once(ex, 0.0, args.settle, args.arm)
                    if out is None:
                        continue
                    hi = out[1]
                    print(f"  bracket seeded from the measured grasp: "
                          f"hi = {hi:.2f}% (holds), lo = {lo:.2f}% (releases)")
                    results.append((0.0, out[0]))
                    continue
                x = next_probe(lo, hi)
                out = probe_once(ex, x, args.settle, args.arm)
                if out is None:
                    continue
                caged = out[0]
                results.append((x, caged))
                lo, hi = update_bracket(lo, hi, x, caged)
                print(f"  bracket now: releases below {lo:.2f}%, "
                      f"holds above {hi:.2f}%  (width {hi-lo:.2f})")
        finally:
            ex.backend.close_realtime_feedback()

    if not results or hi is None:
        print("\nno usable probes.")
        return 2
    x_esc = 0.5 * (lo + hi)
    print(f"\n--- escape position " + "-" * 44)
    print(f"ball leaves between {lo:.2f}% and {hi:.2f}% closed  "
          f"-> X_escape = {x_esc:.2f} +- {(hi-lo)/2:.2f} %")

    try:
        tr = np.load(args.traces, allow_pickle=True)["traces"]
    except Exception as e:
        print(f"(no open traces at {args.traces}: {e}) -- position only.")
        return 0
    t_esc, t_on = escape_time_from_traces(tr, x_esc)
    if t_esc is None:
        print("open traces never reach that position; cannot convert to time.")
        return 0
    print(f"\n--- escape time, from {len(tr)} recorded open transitions " + "-" * 8)
    print(f"  t_onset  = {1e3*t_on:6.1f} ms   (compensated by GRIPPER_RELEASE_LATENCY_S)")
    print(f"  t_escape = {1e3*t_esc:6.1f} ms   <- the ball actually leaves HERE")
    print(f"  uncompensated remainder = {1e3*(t_esc-t_on):.1f} ms")
    print(f"\nFeed that remainder to the coast analysis to get the landing sign: "
          f"the arm keeps swinging during GRIPPER_RELEASE_PAUSE_S, so a later "
          f"departure LENGTHENS the throw and opposes the 36 ms command lag.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
