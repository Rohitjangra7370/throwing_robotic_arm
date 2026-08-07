"""
Measure gripper release latency on the real Gen3, at 1 kHz, with no camera.

WHY THIS IS THE NEXT THING TO MEASURE
-------------------------------------
Release timing is now the dominant term in our landing-error budget:

  * commands are HIGH-LEVEL, so the base treats them every 25 ms -> up to
    3.7 cm of undershoot at the measured 1.498 m/s release speed, which is
    already larger than the whole 2.89 cm sim accuracy (see HIGH_LEVEL_MAX_HZ);
  * on top of that, the fingers take their own time to physically clear the
    ball while the arm is still decelerating -- the paper's Sec 5
    ReleaseTimingJitter, and historically the largest sim-to-real gap in
    throwing work.

The second term has never been measured here. A 30 fps RealSense cannot measure
it either: 33 ms per frame is coarser than the 25 ms effect we are chasing.

But the arm already knows. FEEDBACK is not subject to the 40 Hz command
ceiling -- Kinova exposes BaseCyclic over UDP at 1 kHz -- and the interconnect
reports gripper finger position and velocity directly. So we can timestamp the
command, watch the fingers at 1 kHz, and get latency to ~1 ms with no camera,
no calibration and no extrinsics.

WHAT IT REPORTS
---------------
  t_onset     command -> fingers first move        (the number to advance the
                                                    release trigger by)
  t_clear     command -> fingers past the ball     (radius-dependent, the
                                                    instant the ball is free)
  t_settle    command -> motion finished

SAFETY: this CYCLES THE GRIPPER repeatedly. Fingers are a pinch hazard. It is
dry-run by default and needs --arm AND --confirm. It never moves the arm.

    python3 measure_gripper_latency.py                          # dry-run
    python3 measure_gripper_latency.py --arm --confirm -n 10    # real
"""

import argparse
import sys
import time

import numpy as np

sys.path.append("..")

from robot_arm.kinova_hardware import (HIGH_LEVEL_MAX_HZ, HardwareThrowExecutor,
                                       SafetyLimits, _patch_collections_abc)
from robot_arm.robot_profiles import get_robot_profile


def sample_transition(backend, target_closed, settle_eps=0.5, timeout=3.0,
                      onset_eps=0.5, hz=1000.0):
    """
    Command one gripper transition and record (t, position, velocity) at ~1 kHz.

    The command timestamp is taken as late as possible and the first sample as
    early as possible, so the reported latency is charged to the arm rather than
    to our own call overhead.
    """
    dt = 1.0 / hz
    p0, _ = backend.read_gripper()
    trace = []
    t_cmd = time.perf_counter()
    backend.send_gripper(1.0 if target_closed else 0.0)
    tick = 0
    while True:
        now = time.perf_counter()
        el = now - t_cmd
        if el > timeout:
            break
        p, v = backend.read_gripper()
        trace.append((el, p, v))
        # stop once it has clearly stopped moving away from the start
        if len(trace) > 50 and abs(v) < settle_eps and abs(p - p0) > onset_eps:
            break
        tick += 1
        deadline = t_cmd + tick * dt
        late = time.perf_counter() - deadline
        if late < 0:
            time.sleep(-late)
    return p0, np.array(trace, dtype=float)


def analyse(p0, trace, ball_radius_pct=None, onset_eps=0.5, vel_eps=1.0):
    """Onset / clear / settle from a position-velocity trace."""
    if trace.size == 0:
        return {}
    t, p, v = trace[:, 0], trace[:, 1], trace[:, 2]
    moved = np.abs(p - p0) > onset_eps
    moving = np.abs(v) > vel_eps
    out = {"n_samples": len(t), "span_s": float(t[-1]),
           "rate_hz": len(t) / float(t[-1]) if t[-1] > 0 else float("nan"),
           "p_start": p0, "p_end": float(p[-1])}
    out["t_onset"] = float(t[np.argmax(moved)]) if moved.any() else None
    out["t_onset_vel"] = float(t[np.argmax(moving)]) if moving.any() else None
    if moved.any():
        # last instant still moving = motion finished
        out["t_settle"] = float(t[len(moving) - 1 - np.argmax(moving[::-1])]) \
            if moving.any() else None
    if ball_radius_pct is not None and moved.any():
        cleared = np.abs(p - p0) > ball_radius_pct
        out["t_clear"] = float(t[np.argmax(cleared)]) if cleared.any() else None
    return out


def _probe(args):
    """
    READ-ONLY rate probe. Opens the UDP feedback channel and samples; sends no
    command of any kind. The whole latency measurement rests on the claim that
    feedback escapes the 40 Hz command ceiling, so that claim gets verified on
    its own, before anything is permitted to move.
    """
    _patch_collections_abc()
    profile = get_robot_profile(args.robot)
    limits = SafetyLimits(qd_max=np.array(profile.qd_max, float),
                          q_soft_lo=-6.10 * np.ones(len(profile.qd_max)),
                          q_soft_hi=6.10 * np.ones(len(profile.qd_max)),
                          speed_scale=1.0)
    with HardwareThrowExecutor(limits, dry_run=not args.arm, ip=args.ip) as ex:
        ex.backend.open_realtime_feedback()
        try:
            t0 = time.perf_counter()
            samples = []
            while time.perf_counter() - t0 < args.probe:
                samples.append((time.perf_counter() - t0,) + ex.backend.read_gripper())
            a = np.array(samples, float)
        finally:
            ex.backend.close_realtime_feedback()
    gaps = np.diff(a[:, 0]) * 1e3
    print(f"\n--- UDP feedback probe ({args.probe:.1f}s, NO command sent) ---")
    print(f"samples          : {len(a)}")
    print(f"achieved rate    : {len(a) / a[-1, 0]:.1f} Hz")
    print(f"inter-sample gap : mean {gaps.mean():.3f} ms  p50 {np.percentile(gaps,50):.3f}  "
          f"p99 {np.percentile(gaps,99):.3f}  max {gaps.max():.3f}")
    print(f"gripper position : {a[:,1].min():.3f} .. {a[:,1].max():.3f} %  "
          f"(|v| max {np.abs(a[:,2]).max():.3f})")
    ok = len(a) / a[-1, 0] > 500
    print(f"verdict          : {'1 kHz-class path CONFIRMED' if ok else 'NOT the 1 kHz path -- latency would be unmeasurable'}")
    print("\nNo command was sent to the arm.")
    return 0 if ok else 2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default="192.168.1.101")
    ap.add_argument("--robot", default="kinova_gen3_dyn")
    ap.add_argument("--arm", action="store_true", help="talk to the REAL arm (default: dry-run)")
    ap.add_argument("--confirm", action="store_true",
                    help="assert fingers are clear of hands and the ball is loose")
    ap.add_argument("-n", "--trials", type=int, default=10)
    ap.add_argument("--gap", type=float, default=1.5, help="seconds between cycles")
    ap.add_argument("--clear_pct", type=float, default=None,
                    help="finger travel %% at which the ball is free; if omitted, "
                         "t_clear is not reported")
    ap.add_argument("--out", default="results_gripper_latency.npz")
    ap.add_argument("--probe", type=float, default=None, metavar="SECONDS",
                    help="READ-ONLY: open the UDP channel, sample for SECONDS, "
                         "report the achieved rate, send NO command. Proves the "
                         "1 kHz path works before anything is allowed to move.")
    args = ap.parse_args()

    if args.probe is not None:
        return _probe(args)

    if args.arm and not args.confirm:
        print("REFUSED: cycling the gripper on the real arm needs --confirm "
              "(fingers are a pinch hazard; ball loose, hands clear).")
        return 2

    _patch_collections_abc()
    profile = get_robot_profile(args.robot)
    limits = SafetyLimits(qd_max=np.array(profile.qd_max, float),
                          q_soft_lo=-6.10 * np.ones(len(profile.qd_max)),
                          q_soft_hi=6.10 * np.ones(len(profile.qd_max)),
                          speed_scale=1.0)

    with HardwareThrowExecutor(limits, dry_run=not args.arm, ip=args.ip) as ex:
        ex.backend.open_realtime_feedback()
        try:
            print(f"\ncommand ceiling {HIGH_LEVEL_MAX_HZ:.0f} Hz (25.0 ms quantisation); "
                  f"feedback target 1000 Hz")
            print(f"cycling gripper {args.trials}x "
                  f"({'REAL ARM' if args.arm else 'DRY-RUN, latency is 0 by construction'})\n")
            rows, traces = [], []
            for i in range(args.trials):
                # close first (grasp), then the OPEN transition is the release
                sample_transition(ex.backend, target_closed=True)
                time.sleep(args.gap)
                p0, tr = sample_transition(ex.backend, target_closed=False)
                r = analyse(p0, tr, ball_radius_pct=args.clear_pct)
                traces.append(tr)
                rows.append(r)
                print(f"  [{i+1:2d}] rate {r.get('rate_hz', float('nan')):7.1f} Hz  "
                      f"onset {1e3*(r.get('t_onset') or float('nan')):6.1f} ms  "
                      f"settle {1e3*(r.get('t_settle') or float('nan')):6.1f} ms  "
                      f"travel {r['p_start']:.1f} -> {r['p_end']:.1f} %")
                time.sleep(args.gap)
        finally:
            ex.backend.close_realtime_feedback()

    def _stat(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return (np.mean(vals), np.std(vals), len(vals)) if vals else (None, None, 0)

    print("\n--- summary " + "-" * 50)
    rate = _stat("rate_hz")
    print(f"feedback rate: {rate[0]:.1f} Hz mean "
          f"({'1 kHz UDP path confirmed' if rate[0] and rate[0] > 500 else 'BELOW 500 Hz -- not the UDP path'})")
    for key, label in (("t_onset", "command -> fingers move  "),
                       ("t_clear", "command -> ball free     "),
                       ("t_settle", "command -> motion done   ")):
        m, s, n = _stat(key)
        if m is None:
            print(f"{label}: not measured")
            continue
        print(f"{label}: {1e3*m:6.1f} +- {1e3*s:.1f} ms  (n={n})")
        if key == "t_onset":
            print(f"{'':27}  -> advance the release trigger by this much;"
                  f" at 1.498 m/s it is {1.498*m*100:.1f} cm of landing error")

    np.savez(args.out, traces=np.array(traces, dtype=object), allow_pickle=True)
    print(f"\ntraces -> {args.out}")
    if not args.arm:
        print("DRY-RUN: plumbing only. No latency was measured.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
