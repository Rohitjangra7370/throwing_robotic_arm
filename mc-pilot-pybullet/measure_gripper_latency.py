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


# Below this the fingers have something between them. Same rule and same
# measured basis as pickup_and_lift.GRASP_THRESHOLD_PCT (58-59% on a real
# tennis ball against ~99-100% closing on nothing) -- kept as one number in
# one place would be better, but that module moves the arm on import-time
# defaults, so the constant is mirrored here with its provenance instead.
GRASP_THRESHOLD_PCT = 90.0
# Fully open reads ~0.87% on this gripper. Anything above this but below
# GRASP_THRESHOLD_PCT is the motor stalled PART-WAY, i.e. holding something.
OPEN_PCT = 5.0


def holding_something(pos_pct):
    """
    True iff the fingers are stalled on an object.

    NOT the same as "less than GRASP_THRESHOLD_PCT". An OPEN gripper is also
    below that threshold (~0.87%), so a bare `pos < 90` test refuses the one
    state it is safe to close from. The hazardous state is stalled PART-WAY:
    open at one end, closed-on-nothing (~99-100%) at the other, a held object
    in between (61.4% measured on this tennis ball, 58.08% on the previous one).
    """
    return OPEN_PCT < float(pos_pct) < GRASP_THRESHOLD_PCT


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
    ap.add_argument("--loaded", action="store_true",
                    help="BALL-LOADED mode. The saved runs to date are all EMPTY "
                         "(p_start 99.13%%, i.e. the fingers closed on nothing), so "
                         "the compensated onset has never been measured from the "
                         "state a real throw releases from: a 2F-85 gripping a "
                         "tennis ball STALLS at ~58%%. This mode prompts for a ball "
                         "reload before each cycle, verifies a real grasp happened "
                         "(pickup_and_lift.py's rule), and refuses to re-close on an "
                         "already-stalled gripper -- pushing again into a held ball "
                         "is the proximate trigger of the 2026-08-22 ROBOT_IN_FAULT.")
    ap.add_argument("--release_speed", type=float, default=1.424,
                    help="TCP release speed (m/s) used to convert a latency into "
                         "a landing error. Default 1.424 = what "
                         "results_kinetic_chain_gen3_tcp/1 commands at a 0.70 m "
                         "target, verified through run_hardware_throw.py plan. "
                         "The old hardcoded 1.498 predates the TCP-offset "
                         "retrain; check `plan` for your actual target.")
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
                if args.loaded:
                    pre, _ = ex.backend.read_gripper()
                    if holding_something(pre) and args.arm:
                        print(f"REFUSED: gripper at {pre:.1f}% is stalled part-way, "
                              f"i.e. still holding something (open is <{OPEN_PCT}%, "
                              f"closed-on-nothing is >{GRASP_THRESHOLD_PCT}%). "
                              f"Re-closing onto a stalled gripper is what preceded "
                              f"the 2026-08-22 ROBOT_IN_FAULT. Open it first:\n"
                              f"  run_hardware_throw.py gripper --open --arm "
                              f"--robot kinova_gen3_dyn")
                        return 2
                    input(f"\n  [{i+1}/{args.trials}] place the ball between the "
                          f"fingers, hands CLEAR, then press Enter... ")
                # close first (grasp), then the OPEN transition is the release
                sample_transition(ex.backend, target_closed=True)
                if args.loaded:
                    grip, _ = ex.backend.read_gripper()
                    ok = holding_something(grip)
                    print(f"       grasp check: {grip:.2f}% closed -> "
                          f"{'BALL HELD' if ok else 'CLOSED ON NOTHING'}")
                    if not ok and args.arm:
                        print("       SKIPPING this cycle: an empty close measures "
                              "the same thing the existing traces already measured.")
                        sample_transition(ex.backend, target_closed=False)
                        time.sleep(args.gap)
                        continue
                time.sleep(args.gap)
                p0, tr = sample_transition(ex.backend, target_closed=False)
                r = analyse(p0, tr, ball_radius_pct=args.clear_pct)
                traces.append(tr)
                rows.append(r)
                print(f"  [{i+1:2d}] rate {r.get('rate_hz', float('nan')):7.1f} Hz  "
                      f"onset {1e3*(r.get('t_onset') or float('nan')):6.1f} ms  "
                      f"clear {1e3*(r.get('t_clear') or float('nan')):6.1f} ms  "
                      f"settle {1e3*(r.get('t_settle') or float('nan')):6.1f} ms  "
                      f"travel {r['p_start']:.1f} -> {r['p_end']:.1f} %"
                      f"{'  <-- EMPTY, not a loaded release' if r['p_start'] > GRASP_THRESHOLD_PCT else ''}")
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
        v = args.release_speed
        if key == "t_onset":
            print(f"{'':27}  -> currently compensated (GRIPPER_RELEASE_LATENCY_S);"
                  f" at {v:.3f} m/s it is {v*m*100:.1f} cm of landing error")
        elif key == "t_clear":
            m_on, _, _ = _stat("t_onset")
            if m_on is not None:
                print(f"{'':27}  -> the ball leaves HERE, not at onset: "
                      f"{1e3*(m-m_on):.1f} ms past it, uncompensated.")
                print(f"{'':27}  -> SIGN IS NOT OBVIOUS, do not assume undershoot. "
                      f"During GRIPPER_RELEASE_PAUSE_S the arm COASTS at the held "
                      f"joint velocity rather than decelerating, so a late "
                      f"departure keeps the wrist SWINGING: release elevation "
                      f"falls and the throw gets LONGER. That opposes the arm's "
                      f"own command lag, which shortens it. Net sign depends on "
                      f"the release state -- integrate q forward at the held qd "
                      f"and re-solve the ballistic landing (measured 2026-09-09: "
                      f"the two cancel near +76 ms). The pre-pause docs saying "
                      f"'latency = undershoot' predate that coast window.")

    np.savez(args.out, traces=np.array(traces, dtype=object), allow_pickle=True)
    print(f"\ntraces -> {args.out}")
    if not args.arm:
        print("DRY-RUN: plumbing only. No latency was measured.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
