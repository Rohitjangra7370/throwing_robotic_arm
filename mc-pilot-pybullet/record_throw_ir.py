"""
Capture one throw to disk.

    python3 record_throw_ir.py --out throws/2026-08-25_t01.npz
    python3 record_throw_ir.py --out throws/t02.npz --seconds 2.5 --countdown 5

Start this, then throw. The window only has to CONTAIN the flight -- the
median-background detector needs no separate empty-scene capture, and frames
before the ball appears cost nothing.
"""
import argparse
import os
import time

from perception.ir_capture import IRRecorder, save_recording


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--countdown", type=int, default=3)
    ap.add_argument("--exposure_us", type=int, default=2000)
    ap.add_argument("--no_emitter", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with IRRecorder(exposure_us=args.exposure_us, emitter=not args.no_emitter) as r:
        for k in range(args.countdown, 0, -1):
            print(f"  {k}...", flush=True)
            time.sleep(1.0)
        print("  THROW", flush=True)
        rec = r.record(args.seconds)

    save_recording(args.out, rec)
    m = rec["meta"]
    print(f"saved {args.out}: {m['n_frames']} frames at {m['achieved_fps']:.1f} fps "
          f"(exposure {m['exposure_us']} us, emitter {m['emitter']})")
    if m["achieved_fps"] < 0.9 * m["fps"]:
        print(f"  CHECK: achieved {m['achieved_fps']:.1f} fps against a requested "
              f"{m['fps']} -- dropped frames widen the error budget")


if __name__ == "__main__":
    main()
