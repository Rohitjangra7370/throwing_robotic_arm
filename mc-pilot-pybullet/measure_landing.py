"""
Recording -> the ball's first contact point with the floor, in the base frame.

This is the offline half of the vision pipeline: it never touches the camera.
`record_throw_ir.py` captures; this interprets. Keeping them apart means every
recording is a permanent regression fixture -- a changed fitter can be re-run
against a real throw from weeks ago, which matters in a project with this much
history of plausible-looking wrong numbers.

    from measure_landing import measure_landing
    result = measure_landing(rec, R_bc, t_bc)
"""

from __future__ import annotations

import numpy as np

from perception.ball_track import (detect_candidates, frame_diagnostics,
                                   median_background)
from perception.ray_plane import D435I_IR_848x480
from perception.stereo import D435I_IR_BASELINE_M, StereoRig, pair_candidates
from perception.trajectory import (BALL_RADIUS, Z_FLOOR_BASE,
                                   ransac_track, solve_impact)

__all__ = ["build_observations", "measure_landing", "default_rig"]


def default_rig():
    return StereoRig(D435I_IR_848x480, D435I_IR_BASELINE_M)


def build_observations(rec, bg1=None, bg2=None, **detect_kw):
    """
    Recording -> ((N, 5) observation array [t, u1, v1, u2, v2], max_mask_frac).

    Backgrounds default to the per-pixel temporal median of the recording
    itself, which needs no separate empty-scene capture and cannot drift
    relative to the throw.

    Frames with no detection are skipped silently -- every frame before the ball
    enters view is one of those, and it is not an error. Frames with several
    candidates contribute several rows; deciding which is the ball is
    `ransac_track`'s job, not this function's.
    """
    ir1, ir2, ts = rec["ir1"], rec["ir2"], np.asarray(rec["t"], float)
    if len(ir1) != len(ir2) or len(ir1) != len(ts):
        raise ValueError(f"ragged recording: {len(ir1)} ir1, {len(ir2)} ir2, "
                         f"{len(ts)} timestamps")
    if bg1 is None:
        bg1 = median_background(ir1)
    if bg2 is None:
        bg2 = median_background(ir2)

    rows, max_frac = [], 0.0
    for k in range(len(ts)):
        max_frac = max(max_frac,
                       frame_diagnostics(ir1[k], bg1)["mask_nonzero_frac"],
                       frame_diagnostics(ir2[k], bg2)["mask_nonzero_frac"])
        left = detect_candidates(ir1[k], bg1, **detect_kw)
        right = detect_candidates(ir2[k], bg2, **detect_kw)
        if not left or not right:
            continue
        lt = [c.as_uv_area() for c in left]
        rt = [c.as_uv_area() for c in right]
        for li, ri in pair_candidates(lt, rt):
            rows.append([ts[k], lt[li][0], lt[li][1], rt[ri][0], rt[ri][1]])

    if max_frac > 0.15:
        raise RuntimeError(
            f"a frame changed over {max_frac:.0%} of its pixels against the "
            f"background -- the camera was bumped, the lighting shifted, or "
            f"auto-exposure resettled mid-capture. Re-record; do not loosen the "
            f"detector to work around it")
    if not rows:
        raise RuntimeError(
            "no left/right ball candidates paired in any frame -- either the "
            "ball never entered view, the detector thresholds are wrong for "
            "this exposure, or the two streams are misaligned")
    return np.asarray(rows, float), max_frac


def measure_landing(rec, R_bc, t_bc, z_floor=Z_FLOOR_BASE,
                    ball_radius=BALL_RADIUS, seed=0, rig=None, **detect_kw):
    """
    The whole offline pipeline, in base-frame coordinates.

    `R_bc`, `t_bc` are the camera pose in base coordinates
    (p_base = R_bc @ p_cam + t_bc), i.e. the stored T_B_C.

    Returns a dict: x, y, t_impact, sigma_xy_m, n_frames, n_inliers, rms_px,
    p0, v0. Raises RuntimeError at the first stage that cannot honestly proceed.

    `sigma_xy_m` propagates the fit covariance to the landing point by linearising
    solve_impact around the estimate. A landing point WITHOUT a sigma is not
    eligible to become a GP datapoint, which is why this is returned and not
    merely logged.
    """
    rig = rig or default_rig()
    obs, max_frac = build_observations(rec, **detect_kw)
    inliers, fit = ransac_track(obs, rig, R_bc, t_bc, seed=seed)
    x, y, t_imp = solve_impact(fit.p0, fit.v0, z_floor=z_floor,
                               ball_radius=ball_radius)

    # Linearised propagation: d(x,y)/d(theta) by central differences on the same
    # impact solve the answer came from, so the sigma describes THIS estimator.
    theta = np.concatenate([fit.p0, fit.v0])
    Jl = np.zeros((2, 6))
    for k in range(6):
        step = 1e-6 * max(1.0, abs(theta[k]))
        tp, tm = theta.copy(), theta.copy()
        tp[k] += step; tm[k] -= step
        xp, yp, _ = solve_impact(tp[:3], tp[3:], z_floor=z_floor, ball_radius=ball_radius)
        xm, ym, _ = solve_impact(tm[:3], tm[3:], z_floor=z_floor, ball_radius=ball_radius)
        Jl[:, k] = [(xp - xm) / (2 * step), (yp - ym) / (2 * step)]
    cov_xy = Jl @ fit.cov @ Jl.T

    return {"x": x, "y": y, "t_impact": t_imp,
            "sigma_xy_m": float(np.sqrt(max(np.trace(cov_xy), 0.0))),
            "n_frames": int(obs.shape[0]), "n_inliers": int(inliers.size),
            "rms_px": fit.rms_px, "p0": fit.p0, "v0": fit.v0,
            "max_mask_frac": max_frac}


def load_extrinsic(path):
    """
    Load T_B_C from a .npz with `R` (3x3) and `t` (3,), the convention used
    throughout this project: p_base = R @ p_cam + t.
    """
    z = np.load(path)
    R, t = np.asarray(z["R"], float), np.asarray(z["t"], float)
    if R.shape != (3, 3) or t.shape != (3,):
        raise ValueError(f"expected R (3,3) and t (3,), got {R.shape} and {t.shape}")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
        raise ValueError("R is not orthonormal -- this is not a rotation")
    return R, t


def main():
    import argparse

    from perception.ir_capture import load_recording

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recording", required=True)
    ap.add_argument("--extrinsic", required=True, help=".npz with R (3x3), t (3,)")
    ap.add_argument("--z_floor", type=float, default=Z_FLOOR_BASE)
    ap.add_argument("--ball_radius", type=float, default=BALL_RADIUS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rec = load_recording(args.recording)
    R_bc, t_bc = load_extrinsic(args.extrinsic)
    out = measure_landing(rec, R_bc, t_bc, z_floor=args.z_floor,
                          ball_radius=args.ball_radius, seed=args.seed)

    print(f"landing (base frame): x = {out['x']:+.4f} m   y = {out['y']:+.4f} m")
    print(f"  sigma            : {out['sigma_xy_m'] * 1e3:.1f} mm")
    print(f"  impact at t      : {out['t_impact']:.4f} s")
    print(f"  frames / inliers : {out['n_frames']} / {out['n_inliers']}")
    print(f"  fit RMS          : {out['rms_px']:.3f} px")
    print(f"  max changed-px   : {out['max_mask_frac']:.2%}  (>15% = bumped camera)")
    print(f"  release p0       : {np.array2string(out['p0'], precision=4)}")
    print(f"  release v0       : {np.array2string(out['v0'], precision=4)}  "
          f"|v0| = {np.linalg.norm(out['v0']):.4f} m/s")
    print("\nNOTE: absolute accuracy is bounded by T_B_C, not by the vision. "
          "Confirm the extrinsic is current for the present mount.")


if __name__ == "__main__":
    main()
