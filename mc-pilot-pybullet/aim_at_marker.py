"""
Read the bin's ArUco marker off the camera and print the target to type.

The command-line half of the session GUI's "Aim at bin" button -- same
functions, same log file, no GUI and no session required. Use it when the app
is not running:

    python3 aim_at_marker.py --marker_size 0.080
    python3 aim_at_marker.py --marker_id 0 --marker_size 0.080
    python3 aim_at_marker.py --from throws/throw_020.npz     # no camera needed

THE CAMERA MUST BE FREE. A D435i admits exactly one process, and a live
session holds it for the whole run day (`session_camera.IRRecorder` opens
`infrared,1`/`infrared,2`). If the app is up, use its button instead -- this
will simply fail to open the device.

WHY IT AVERAGES FRAMES. One IR frame at 4 ms exposure is noisy enough to move
a corner by a few tenths of a pixel, which is a few millimetres on the floor.
The bin is not moving while you read it, so the per-pixel median over a short
burst is free accuracy -- the same argument `ball_track.median_background`
makes, for the same reason.

Reads IR1 rather than colour on purpose; see perception/floor_marker.py's
module docstring for why that choice is load-bearing and not convenience.
"""

import argparse
import datetime
import json
import sys

import numpy as np

from hardware_session import aim_at_bin, aimer_args


def grab_ir1(n_frames=15, warmup=10, width=848, height=480, fps=90,
             exposure_us=4000, emitter=True):
    """Median of a short live burst of IR1 frames."""
    from perception.ir_capture import IRRecorder

    frames = []
    with IRRecorder(width=width, height=height, fps=fps,
                    exposure_us=exposure_us, emitter=emitter) as rec:
        for i, (_ts, ir1, _ir2) in enumerate(rec.stream()):
            if i < warmup:          # let auto-anything settle before measuring
                continue
            frames.append(np.array(ir1, copy=True))
            if len(frames) >= n_frames:
                break
    return np.median(np.asarray(frames), axis=0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--marker_id", type=int, default=None)
    ap.add_argument("--marker_size", type=float, default=None,
                    help="MEASURED printed side in metres. Optional, but it is "
                         "the only check that catches a rescaled printout, a "
                         "wrong floor height or a stale extrinsic.")
    ap.add_argument("--marker_exclude", type=int, nargs="*", default=())
    ap.add_argument("--from", dest="from_recording", default=None,
                    help="read IR1 from a saved .npz instead of the camera")
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--exposure_us", type=int, default=4000)
    ap.add_argument("--no_emitter", action="store_true")
    ap.add_argument("--robot", default="kinova_gen3_dyn")
    ap.add_argument("--log_path", default="results_kinetic_chain_gen3_tcp/1")
    ap.add_argument("--opt_pose", default="throw_pose_table_tcp.npy")
    ap.add_argument("--tool_offset_z", type=float, default=0.12)
    ap.add_argument("--base_height", type=float, default=0.433)
    ap.add_argument("--ball_radius", type=float, default=0.0327)
    ap.add_argument("--u_cap", type=float, default=2.00)
    ap.add_argument("--wrist_roll_offset_deg", type=float, default=90.0)
    ap.add_argument("--aim_model", choices=("gain", "additive"), default="additive")
    ap.add_argument("--bin_game_log", default="bin_game_log.jsonl")
    args = ap.parse_args()

    if args.from_recording:
        from perception.ball_track import median_background
        from perception.ir_capture import load_recording
        ir1 = median_background(load_recording(args.from_recording)["ir1"])
        source = args.from_recording
        print(f"reading IR1 from {source} (median of the recording)")
    else:
        print(f"grabbing {args.frames} IR1 frames from the D435i ...")
        ir1 = grab_ir1(n_frames=args.frames, exposure_us=args.exposure_us,
                       emitter=not args.no_emitter)
        source = "live"

    from perception import base_frame
    extrinsic = base_frame.load_extrinsic()

    try:
        marker, target, pred, refusals = aim_at_bin(
            ir1, aimer_args(args), extrinsic,
            marker_id=args.marker_id, marker_size_m=args.marker_size,
            exclude_ids=tuple(args.marker_exclude or ()))
    except RuntimeError as e:
        print(f"\nNO READING: {e}")
        return 2

    size = ("" if marker.side_error_m != marker.side_error_m
            else f"   size check {marker.side_error_m * 1000:+.0f} mm "
                 f"(measures {marker.side_m * 1000:.0f} mm)")
    print(f"\n  marker {marker.marker_id} is at base ({marker.x:+.4f}, {marker.y:+.4f}) m")
    print(f"  {marker.pixel_side:.0f} px across in the image{size}")

    if refusals or target is None:
        print("\n  REFUSED -- do not throw this one:")
        for r in refusals:
            print(f"    - {r}")
    else:
        lx, ly = pred["predicted_landing"]
        print(f"\n  ==> TYPE Target X {target[0]:.4f}   Target Y {target[1]:+.4f}")
        print(f"      (untick auto-cycle first, or the fields are ignored)")
        print(f"      commanded speed {pred['commanded_speed']:.3f} m/s, "
              f"predicts landing ({lx:+.3f}, {ly:+.3f})")
        print(f"      solver miss {pred['solver_miss_m'] * 1000:.1f} mm; "
              f"expect the ball within ~3.4 cm of the marker")

    rec = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
           "source": source, "marker_id": marker.marker_id,
           "marker_xy": [marker.x, marker.y], "marker_side_m": marker.side_m,
           "marker_side_error_m": (None if marker.side_error_m != marker.side_error_m
                                   else marker.side_error_m),
           "marker_pixel_side": marker.pixel_side,
           "command_target": target,
           "predicted_landing": None if pred is None else pred["predicted_landing"],
           "commanded_speed": None if pred is None else pred["commanded_speed"],
           "aim_model": args.aim_model, "refusals": list(refusals)}
    with open(args.bin_game_log, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"\n  logged to {args.bin_game_log}")
    return 0 if not refusals else 1


if __name__ == "__main__":
    sys.exit(main())
