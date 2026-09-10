"""
Run the whole landing pipeline over a recorded throw, and show its work.

    python3 analyze_throw.py --recording throws/throw_000.npz
    python3 analyze_throw.py --recording throws/throw_000.npz --extrinsic ext.npz
    python3 analyze_throw.py --recording throws/throw_000.npz --range 0.4 2.2

`measure_landing.py` answers one question -- where did it land -- and refuses if it
cannot answer honestly. That is the right behaviour for the run loop and the wrong
behaviour for a bring-up session, because a bare refusal does not tell you WHICH of
detection, pairing, or association failed. This tool runs the same stages and prints
what each one produced, then writes a picture of every moving thing in the recording
so you can see the flight (or see that there wasn't one).

THE DISPARITY PREFILTER IS NOT A LOOSENED GATE
----------------------------------------------
`ransac_track` requires 60% of all observations to lie on one arc. In a clean
recording the ball dominates and that holds. In a real room it does not: a hand, a
sleeve, or the arm generates hundreds of rows, and a perfectly good 35-frame ball
track becomes 12% of the total and is refused.

The fix is to drop rows that CANNOT be a ball in flight, not to lower the bar for
what counts as an arc. A detection whose disparity puts it 5 cm from the lens is
not a thrown ball at any threshold, and removing it is arithmetic, not tuning. The
ballistic consensus test itself is left exactly as strict as it was.

--extrinsic is the real T_B_C. Without it, --height assumes a perfectly vertical
camera at that height above the floor, which is enough to test whether an arc
EXISTS and is NOT enough to trust the landing coordinates. Output says which was
used, every time.
"""
import argparse
import os

import numpy as np

from perception.ball_track import median_background
from perception.ir_capture import load_recording
from perception.ray_plane import D435I_IR_848x480 as INTR
from perception.stereo import D435I_IR_BASELINE_M, StereoRig
from perception.trajectory import (BALL_RADIUS, MIN_INLIER_FRAMES, Z_FLOOR_BASE,
                                   ballistic_position, ransac_track, solve_impact)
from measure_landing import build_observations


def approx_extrinsic(height_m):
    """
    Camera assumed perfectly vertical, looking straight down from `height_m`.
    Origin on the floor beneath it, z up. p_base = R @ p_cam + t.
    """
    R = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
    return R, np.array([0., 0., float(height_m)])


def load_extrinsic_checked(path):
    """Like measure_landing.load_extrinsic, but also rejects a mirrored frame."""
    z = np.load(path)
    R, t = np.asarray(z["R"], float), np.asarray(z["t"], float)
    if R.shape != (3, 3) or t.shape != (3,):
        raise ValueError(f"expected R (3,3) and t (3,), got {R.shape} and {t.shape}")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
        raise ValueError("R is not orthonormal -- this is not a rotation")
    det = float(np.linalg.det(R))
    if abs(det - 1.0) > 1e-6:
        raise ValueError(f"det(R) = {det:+.6f}, not +1 -- this is a reflection, not a "
                         f"rotation, and it will mirror the landing point")
    return R, t


def write_trace(rec, path):
    """Max-projection of |frame - background|: the path of everything that moved."""
    import cv2
    ir = rec["ir1"]
    bg = median_background(ir).astype(np.int16)
    mx = np.abs(ir.astype(np.int16) - bg).max(axis=0)
    img = np.clip(mx * 3, 0, 255).astype(np.uint8)
    cv2.imwrite(path, cv2.applyColorMap(img, cv2.COLORMAP_TURBO))


def write_overlay(rec, obs, inlier_mask, path):
    """Detections over the scene: inliers green, everything else dim red."""
    import cv2
    base = median_background(rec["ir1"]).astype(np.uint8)
    vis = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    for j in range(len(obs)):
        u, v = int(round(obs[j, 1])), int(round(obs[j, 2]))
        if inlier_mask is not None and inlier_mask[j]:
            cv2.circle(vis, (u, v), 4, (0, 230, 0), -1)
        else:
            cv2.circle(vis, (u, v), 2, (0, 0, 170), 1)
    cv2.imwrite(path, vis)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", required=True)
    ap.add_argument("--extrinsic", help=".npz with R (3x3) and t (3,) -- the real T_B_C")
    ap.add_argument("--height", type=float, default=1.6141,
                    help="fallback camera height when --extrinsic is absent")
    ap.add_argument("--range", type=float, nargs=2, default=[0.35, 2.20],
                    metavar=("MIN_M", "MAX_M"),
                    help="physical range window; rows outside it cannot be a ball")
    ap.add_argument("--no_prefilter", action="store_true",
                    help="skip the range prefilter and feed RANSAC everything")
    ap.add_argument("--z_floor", type=float, default=None,
                    help="floor height in base frame (default: 0 for --height, "
                         f"{Z_FLOOR_BASE} for a real extrinsic)")
    ap.add_argument("--ball_radius", type=float, default=BALL_RADIUS)
    ap.add_argument("--thresh_px", type=float, default=2.0)
    ap.add_argument("--min_inlier_frac", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=None, help="where to write the diagnostic images")
    args = ap.parse_args()

    rec = load_recording(args.recording)
    meta = rec["meta"]
    stem = os.path.splitext(os.path.basename(args.recording))[0]
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.recording))
    os.makedirs(outdir, exist_ok=True)

    print(f"recording {args.recording}")
    print(f"  {meta['n_frames']} frames, {meta['achieved_fps']:.1f} fps, "
          f"exposure {meta['exposure_us']} us, emitter {meta['emitter']}")
    if meta["achieved_fps"] < 0.9 * meta["fps"]:
        print(f"  WARNING dropped frames: {meta['achieved_fps']:.1f} of a requested "
              f"{meta['fps']} -- the error budget assumed no drops")

    if args.extrinsic:
        R_bc, t_bc = load_extrinsic_checked(args.extrinsic)
        z_floor = Z_FLOOR_BASE if args.z_floor is None else args.z_floor
        frame_note = f"CALIBRATED extrinsic from {args.extrinsic}"
    else:
        R_bc, t_bc = approx_extrinsic(args.height)
        z_floor = 0.0 if args.z_floor is None else args.z_floor
        frame_note = (f"APPROXIMATE extrinsic: camera assumed perfectly vertical at "
                      f"{args.height:.4f} m.\n           Landing COORDINATES from this "
                      f"are not trustworthy; the fit quality is.")
    print(f"  frame: {frame_note}")

    # ---- stage 1: detection and pairing -------------------------------------
    try:
        obs, max_frac = build_observations(rec)
    except RuntimeError as e:
        print(f"\nREFUSED at detection: {e}")
        return 1
    print(f"\ndetection: {len(obs)} paired rows over "
          f"{len(np.unique(obs[:, 0])) if len(obs) else 0} distinct frames "
          f"(max mask fraction {max_frac:.5f})")
    if len(obs) == 0:
        print("  nothing was detected at all -- wrong background, or nothing moved")
        write_trace(rec, os.path.join(outdir, f"{stem}_trace.png"))
        return 1

    disp = obs[:, 1] - obs[:, 3]
    rng = INTR.fx * D435I_IR_BASELINE_M / np.maximum(disp, 1e-9)
    print(f"  disparity {disp.min():.1f}..{disp.max():.1f} px  "
          f"=> range {rng.min():.2f}..{rng.max():.2f} m")

    # ---- stage 2: physical range prefilter ----------------------------------
    if args.no_prefilter:
        keep = np.ones(len(obs), bool)
    else:
        keep = (rng >= args.range[0]) & (rng <= args.range[1])
        print(f"  range prefilter [{args.range[0]:.2f}, {args.range[1]:.2f}] m keeps "
              f"{int(keep.sum())}/{len(obs)} rows "
              f"({int((~keep).sum())} cannot be a ball in flight)")
    obs_f = obs[keep]
    if len(obs_f) < MIN_INLIER_FRAMES:
        print(f"\nREFUSED: only {len(obs_f)} plausible rows survive, need "
              f">= {MIN_INLIER_FRAMES}")
        write_trace(rec, os.path.join(outdir, f"{stem}_trace.png"))
        write_overlay(rec, obs, None, os.path.join(outdir, f"{stem}_overlay.png"))
        return 1

    # ---- stage 3: ballistic association -------------------------------------
    rig = StereoRig(INTR, D435I_IR_BASELINE_M)
    try:
        inliers, fit = ransac_track(obs_f, rig, R_bc, t_bc, thresh_px=args.thresh_px,
                                    min_inlier_frac=args.min_inlier_frac, seed=args.seed)
    except RuntimeError as e:
        print(f"\nREFUSED at association: {e}")
        print("\n  what to check, in order:")
        print("   * open the trace image -- if there is no arc in it, the ball was")
        print("     never in frame and no threshold will conjure one")
        print("   * if the arc is there but short, the flight left the frame; the")
        print("     camera has to see release AND landing")
        print("   * if the arc is there and long, widen --range or raise --thresh_px")
        write_trace(rec, os.path.join(outdir, f"{stem}_trace.png"))
        write_overlay(rec, obs_f, None, os.path.join(outdir, f"{stem}_overlay.png"))
        return 1

    mask = np.zeros(len(obs_f), bool); mask[inliers] = True
    print(f"\nassociation: {len(inliers)}/{len(obs_f)} inliers, "
          f"reprojection rms {fit.rms_px:.3f} px")
    print(f"  p0 {np.round(fit.p0, 4)} m")
    print(f"  v0 {np.round(fit.v0, 4)} m/s  (speed {np.linalg.norm(fit.v0):.3f} m/s)")

    # ---- stage 4: impact ----------------------------------------------------
    try:
        x, y, t_imp = solve_impact(fit.p0, fit.v0, z_floor=z_floor,
                                   ball_radius=args.ball_radius)
    except RuntimeError as e:
        print(f"\nREFUSED at impact solve: {e}")
        return 1
    tin = obs_f[inliers, 0]
    print(f"\nLANDING  x {x:+.4f}  y {y:+.4f}  m   at t = {t_imp:.4f} s")
    print(f"  arc observed over {tin.min():.3f}..{tin.max():.3f} s "
          f"({tin.max()-tin.min():.3f} s of flight)")
    apex = ballistic_position(fit.p0, fit.v0, max(0.0, -fit.v0[2] / -9.81))
    print(f"  apex height {apex[2]:+.3f} m; floor taken as z = {z_floor:+.3f} m")
    if not args.extrinsic:
        print("  NOTE these coordinates rest on the APPROXIMATE extrinsic above.")
        print("       Treat the fit quality as the result, not the position.")

    tp = os.path.join(outdir, f"{stem}_trace.png")
    op = os.path.join(outdir, f"{stem}_overlay.png")
    write_trace(rec, tp); write_overlay(rec, obs_f, mask, op)
    print(f"\nwrote {tp}\n      {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
