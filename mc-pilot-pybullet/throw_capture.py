"""
Live dual-IR viewfinder that records a throw when it sees one.

WHY THIS EXISTS ALONGSIDE record_throw_ir.py
--------------------------------------------
`record_throw_ir.py` prints a countdown and records a fixed window. That works
when the person throwing is the person reading the terminal. It does not work
when you need to stand where the arm is, and it gives you no way to check --
before committing to a throw -- that the ball is actually being DETECTED rather
than merely being in frame. Those are different questions, and the second one is
the one that decides whether the recording is worth anything.

So this tool does two things record_throw_ir.py does not:

  1. shows you a live overlay of what the detector sees, so aiming the camera and
     confirming detection are the same action; and
  2. arms a motion trigger, so the recording starts because a ball flew, not
     because a countdown expired.

THE TRIGGER IS A DISPARITY GATE, NOT A MOTION GATE
--------------------------------------------------
A naive "did enough pixels change" trigger fires on the thrower's own arm -- this
was tried, and every one of eight captured events turned out to be a forearm
sweeping through frame. The arm is bigger, closer and brighter than the ball, so
it wins every contrast-based test.

What the arm cannot fake is RANGE. This triggers only on a candidate that is
detected in BOTH imagers, pairs on a common row, and whose disparity puts it
between --min_range and --max_range metres from the camera. A hand passing under
the lens sits at 300-400 px of disparity (a few cm from the glass) and is
rejected on arithmetic, not on tuning. This is the same physical argument
`ransac_track` uses downstream, applied early enough to keep junk out of the file.

    python3 throw_capture.py --out throws/            # aim, then throw
    python3 throw_capture.py --out throws/ --range 0.4 2.2
    python3 throw_capture.py --out throws/ --manual   # space bar records instead

    q / ESC  quit      b  rebuild background      space  record now
    a        arm / disarm the trigger             +/-   exposure

Recordings are written in the exact format perception.ir_capture.load_recording
expects, so analyze_throw.py and measure_landing.py read them unchanged.
"""
import argparse
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from perception.ball_track import detect_candidates, median_background
from perception.ir_capture import save_recording
from perception.stereo import D435I_IR_BASELINE_M, pair_candidates
from perception.ray_plane import D435I_IR_848x480 as INTR

FRAME_TIMEOUT_MS = 1500
CONSECUTIVE_FAIL_LIMIT = 5      # stalled waits before we tear the link down
BG_FRAMES = 90                  # ~1 s of stills to build the median background


def open_pipeline(args):
    """Start a dual-IR pipeline with manual exposure. Returns (pipe, sensor)."""
    cfg = rs.config()
    cfg.enable_stream(rs.stream.infrared, 1, args.width, args.height,
                      rs.format.y8, args.fps)
    cfg.enable_stream(rs.stream.infrared, 2, args.width, args.height,
                      rs.format.y8, args.fps)
    pipe = rs.pipeline()
    profile = pipe.start(cfg)
    sensor = profile.get_device().first_depth_sensor()
    sensor.set_option(rs.option.enable_auto_exposure, 0)
    sensor.set_option(rs.option.exposure, float(args.exposure_us))
    sensor.set_option(rs.option.gain, float(args.gain))
    if sensor.supports(rs.option.emitter_enabled):
        sensor.set_option(rs.option.emitter_enabled, 1 if args.emitter else 0)
    # Auto-exposure re-enabling itself, or an exposure write that did not land,
    # silently ruins a recording. Confirm both rather than trusting the write.
    ae = sensor.get_option(rs.option.enable_auto_exposure)
    ex = sensor.get_option(rs.option.exposure)
    if ae != 0 or abs(ex - args.exposure_us) > 1:
        raise RuntimeError(f"camera did not accept manual exposure "
                           f"(auto_exposure={ae}, exposure={ex} us) -- restart it")
    return pipe, sensor


def disparity_window(min_range_m, max_range_m):
    """Range window -> the disparity band a real ball must fall inside."""
    fxb = INTR.fx * D435I_IR_BASELINE_M
    return fxb / max_range_m, fxb / min_range_m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="directory for event .npz files")
    ap.add_argument("--range", type=float, nargs=2, default=[0.35, 2.20],
                    metavar=("MIN_M", "MAX_M"),
                    help="range window a detection must fall in to count as the ball")
    ap.add_argument("--pre", type=float, default=0.45, help="seconds kept before trigger")
    ap.add_argument("--post", type=float, default=1.00, help="seconds kept after trigger")
    ap.add_argument("--max_events", type=int, default=20)
    ap.add_argument("--exposure_us", type=int, default=4000)
    ap.add_argument("--gain", type=float, default=16.0)
    ap.add_argument("--emitter", action="store_true",
                    help="enable the IR projector (measured to add nothing overhead)")
    ap.add_argument("--manual", action="store_true", help="start disarmed; space records")
    ap.add_argument("--no_window", action="store_true", help="headless, trigger only")
    ap.add_argument("--width", type=int, default=848)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=90)
    ap.add_argument("--detect_every", type=int, default=2,
                    help="run detection on every Nth frame while previewing")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    d_lo, d_hi = disparity_window(*args.range)
    ring_n = int((args.pre + args.post) * args.fps) + 20
    pre_n, post_n = int(args.pre * args.fps), int(args.post * args.fps)

    print(f"range {args.range[0]:.2f}-{args.range[1]:.2f} m  ->  accept disparity "
          f"{d_lo:.1f}-{d_hi:.1f} px")
    print(f"ring buffer {ring_n} frames ({(args.pre+args.post):.2f} s, "
          f"{2*ring_n*args.width*args.height/1e6:.0f} MB)")

    pipe, sensor = open_pipeline(args)
    r1 = np.empty((ring_n, args.height, args.width), np.uint8)
    r2 = np.empty((ring_n, args.height, args.width), np.uint8)
    rt = np.empty(ring_n, float)

    bg1 = bg2 = None
    bg_acc = []
    k = n = 0
    fails = reconnects = 0
    armed = not args.manual
    trig_at = None
    events = []
    t_prev = time.monotonic()
    fps_est = 0.0
    status = "building background"

    try:
        while True:
            ok, fs = pipe.try_wait_for_frames(FRAME_TIMEOUT_MS)
            if not ok:
                fails += 1
                if fails >= CONSECUTIVE_FAIL_LIMIT:
                    print("link stalled -- re-enumerating the camera")
                    try:
                        pipe.stop()
                    except Exception:
                        pass
                    time.sleep(1.0)
                    pipe, sensor = open_pipeline(args)
                    fails = 0
                    reconnects += 1
                continue
            fails = 0
            f1, f2 = fs.get_infrared_frame(1), fs.get_infrared_frame(2)
            if not f1 or not f2:
                continue

            a = np.asanyarray(f1.get_data())
            b = np.asanyarray(f2.get_data())
            ts = f1.get_timestamp() * 1e-3 + 0.5 * args.exposure_us * 1e-6

            r1[k] = a; r2[k] = b; rt[k] = ts
            k = (k + 1) % ring_n
            n += 1

            now = time.monotonic()
            fps_est = 0.9 * fps_est + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now

            if bg1 is None:
                bg_acc.append((a.copy(), b.copy()))
                if len(bg_acc) >= BG_FRAMES:
                    bg1 = median_background(np.stack([x for x, _ in bg_acc]))
                    bg2 = median_background(np.stack([y for _, y in bg_acc]))
                    bg_acc = []
                    status = "armed" if armed else "disarmed"
                    print("background built")
            elif trig_at is None and n % args.detect_every == 0:
                c1 = detect_candidates(a, bg1)
                c2 = detect_candidates(b, bg2)
                hit = None
                if c1 and c2:
                    lt = [c.as_uv_area() for c in c1]
                    rtl = [c.as_uv_area() for c in c2]
                    for li, ri in pair_candidates(lt, rtl):
                        disp = lt[li][0] - rtl[ri][0]
                        if d_lo <= disp <= d_hi:
                            hit = (lt[li][0], lt[li][1], disp,
                                   INTR.fx * D435I_IR_BASELINE_M / disp)
                            break
                last_hit = hit
                if hit and armed:
                    trig_at = n
                    status = "RECORDING"
                    print(f"  trigger: u {hit[0]:.0f} v {hit[1]:.0f} "
                          f"disparity {hit[2]:.1f} px -> {hit[3]:.2f} m", flush=True)
            if trig_at is not None and n - trig_at >= post_n:
                m = min(n, pre_n + post_n)
                idx = [(k - m + i) % ring_n for i in range(m)]
                tt = rt[idx]; tt = tt - tt[0]
                rec = {"t": tt, "ir1": r1[idx].copy(), "ir2": r2[idx].copy(),
                       "meta": {"width": args.width, "height": args.height,
                                "fps": args.fps, "exposure_us": args.exposure_us,
                                "gain": args.gain, "emitter": args.emitter,
                                "n_frames": int(m),
                                "achieved_fps": float((m - 1) / max(tt[-1], 1e-9))}}
                path = os.path.join(args.out, f"throw_{len(events):03d}.npz")
                save_recording(path, rec)
                events.append(path)
                print(f"  saved {path}  ({m} frames, "
                      f"{rec['meta']['achieved_fps']:.1f} fps)", flush=True)
                trig_at = None
                status = "armed" if armed else "disarmed"
                if len(events) >= args.max_events:
                    break

            if not args.no_window:
                vis = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
                if bg1 is not None and trig_at is None:
                    for c in detect_candidates(a, bg1):
                        cv2.circle(vis, (int(round(c.u)), int(round(c.v))),
                                   max(4, int(np.sqrt(c.area_px / np.pi)) + 3),
                                   (0, 200, 255), 1)
                cv2.drawMarker(vis, (int(INTR.ppx), int(INTR.ppy)), (90, 90, 90),
                               cv2.MARKER_CROSS, 18, 1)
                colour = (0, 0, 255) if status == "RECORDING" else (0, 220, 0)
                cv2.putText(vis, f"{status}  {fps_est:4.1f} fps  exp {args.exposure_us}us"
                            f"  mean {a.mean():5.1f}", (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
                cv2.putText(vis, f"events {len(events)}   reconnects {reconnects}   "
                            f"accept {d_lo:.0f}-{d_hi:.0f}px", (8, 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(vis, "q quit  b background  space record  a arm  +/- exposure",
                            (8, args.height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (160, 160, 160), 1, cv2.LINE_AA)
                cv2.imshow("throw_capture  (IR1)", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                elif key == ord('b'):
                    bg1 = bg2 = None; bg_acc = []; status = "building background"
                elif key == ord(' ') and trig_at is None and bg1 is not None:
                    trig_at = n; status = "RECORDING"
                elif key == ord('a'):
                    armed = not armed
                    status = "armed" if armed else "disarmed"
                elif key in (ord('+'), ord('=')):
                    args.exposure_us = min(20000, args.exposure_us + 500)
                    sensor.set_option(rs.option.exposure, float(args.exposure_us))
                elif key == ord('-'):
                    args.exposure_us = max(200, args.exposure_us - 500)
                    sensor.set_option(rs.option.exposure, float(args.exposure_us))
    finally:
        try:
            pipe.stop()
        except Exception:
            pass
        if not args.no_window:
            cv2.destroyAllWindows()

    print(f"\n{len(events)} event(s) written to {args.out}")
    for e in events:
        print(f"  python3 analyze_throw.py --recording {e}")


if __name__ == "__main__":
    main()
