"""
Press record, get an annotated video back.

Wraps throw_capture.py's disparity-gated auto-trigger (unmodified, invoked as
a subprocess so this stays a thin combination of two already-trusted pieces)
with perception.visualize.render_annotated -- the same detect -> pair ->
triangulate -> overlay pipeline used to first verify this codebase's detector
against real IR frames instead of only synthetic ones.

    python3 record_and_annotate.py --out throws_live/
    python3 record_and_annotate.py --out throws_live/ --range 0.4 2.0

Move the ball into the camera's view after starting -- the recording starts
itself once a real stereo detection lands inside --range metres, the same
arm-motion-rejecting trigger throw_capture.py uses standalone.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

import numpy as np

# This lives in scripts/, one level below the mc-pilot-pybullet root that
# `perception`, throw_capture.py and measure_landing.py resolve against --
# put that root on sys.path regardless of the caller's cwd, rather than
# relying on it.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import measure_landing as ml
from perception.base_frame import CANONICAL_PATH, has_calibration, load_extrinsic, to_base
from perception.ir_capture import load_recording
from perception.visualize import render_annotated


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="directory for the recording + video/plot")
    ap.add_argument("--range", type=float, nargs=2, default=[0.35, 2.20],
                     metavar=("MIN_M", "MAX_M"))
    ap.add_argument("--timeout", type=float, default=60.0,
                     help="give up if nothing triggers within this many seconds")
    ap.add_argument("--exposure_us", type=int, default=2000)
    ap.add_argument("--gain", type=float, default=16.0)
    args = ap.parse_args()

    # Anchor to the invoking shell's cwd before the subprocess below runs with
    # cwd=ROOT -- otherwise a relative --out would silently land next to
    # throw_capture.py instead of where the user actually asked for it.
    args.out = os.path.abspath(args.out)
    os.makedirs(args.out, exist_ok=True)
    before = set(glob.glob(os.path.join(args.out, "throw_*.npz")))

    cmd = ["timeout", "-s", "INT", str(int(args.timeout)), sys.executable,
           "throw_capture.py", "--out", args.out, "--max_events", "1", "--no_window",
           "--range", str(args.range[0]), str(args.range[1]),
           "--exposure_us", str(args.exposure_us), "--gain", str(args.gain)]
    print(f"recording -- move the ball across the camera now "
          f"(triggers on a real detection {args.range[0]:.2f}-{args.range[1]:.2f} m away, "
          f"gives up after {args.timeout:.0f}s)")
    subprocess.run(cmd, cwd=ROOT)

    new = sorted(set(glob.glob(os.path.join(args.out, "throw_*.npz"))) - before)
    if not new:
        print(f"\nno event triggered within {args.timeout:.0f}s -- nothing came into range "
              f"{args.range[0]:.2f}-{args.range[1]:.2f} m, or the camera link stalled. "
              f"Re-run, or check with tune_ir_exposure.py.")
        sys.exit(1)

    rec_path = new[0]
    base = os.path.splitext(rec_path)[0]
    out_video, out_plot = base + "_annotated.mp4", base + "_3d.png"

    rec = load_recording(rec_path)
    stats = render_annotated(rec, out_video, out_plot, title=rec_path)

    print(f"\nrecorded : {rec_path}")
    print(f"video    : {out_video}  ({stats['n_paired']}/{stats['n_frames']} frames paired)")
    print(f"3d plot  : {out_plot}")
    if stats["n_paired"] == 0:
        print("WARNING: zero frames paired -- the trigger fired but the detector never "
              "found a stereo match. Check exposure/gain, or that the ball actually "
              "crossed both imagers' view.")
        return

    # Auto tf: base-frame coordinates print themselves the moment a
    # calibration exists at CANONICAL_PATH -- no flag, nothing else to wire.
    if not has_calibration():
        print(f"\nno base-frame calibration yet at {CANONICAL_PATH} -- run "
              f"scripts/calibrate_marker_tf.py to see these coordinates in the arm's base frame")
        return

    R_bc, t_bc = load_extrinsic()
    obs, _ = ml.build_observations(rec)
    rig = ml.default_rig()
    cam_pts = np.array([rig.triangulate(*row[1:]) for row in obs])
    base_pts = to_base(cam_pts, R_bc, t_bc)
    print(f"\nbase frame (via {CANONICAL_PATH}):")
    print(f"  x[{base_pts[:,0].min():+.3f},{base_pts[:,0].max():+.3f}]  "
          f"y[{base_pts[:,1].min():+.3f},{base_pts[:,1].max():+.3f}]  "
          f"z[{base_pts[:,2].min():+.3f},{base_pts[:,2].max():+.3f}]  ({len(base_pts)} pts)")
    try:
        land = ml.measure_landing(rec, R_bc, t_bc)
        print(f"  landing  : x={land['x']:+.4f} y={land['y']:+.4f}  "
              f"sigma={land['sigma_xy_m']*1e3:.1f}mm  "
              f"inliers={land['n_inliers']}/{land['n_frames']}  rms={land['rms_px']:.2f}px")
    except RuntimeError as exc:
        print(f"  landing solve skipped: {exc}")


if __name__ == "__main__":
    main()
