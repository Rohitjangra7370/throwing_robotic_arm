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
                                   median_background, reject_static_candidates)
from perception.ray_plane import D435I_IR_848x480
from perception.stereo import D435I_IR_BASELINE_M, StereoRig, pair_candidates
from perception.trajectory import (BALL_RADIUS, G_BASE, MIN_INLIER_FRAMES,
                                   Z_FLOOR_BASE, ballistic_position,
                                   ransac_track, solve_impact)

__all__ = ["build_observations", "check_landing_is_observed", "measure_landing",
           "select_flight_arc", "default_rig", "MAX_UNOBSERVED_DROP_M",
           "MIN_OBSERVED_DROP_M", "MAX_SIGMA_XY_M"]


# How much of the drop to the floor may be left UNSEEN after the last frame the
# ball was detected in, and how much of the fall must have been SEEN before it.
# See check_landing_is_observed.
# RAISED 0.30 -> 0.60 on 2026-09-11, on evidence. The cap was never what
# rejected the known-bad recordings -- re-run with it at 0.30, 0.60 and 2.00,
# the hand-carried clip is still caught by `descending` and the stationary-ball
# clip by the observed-drop floor, at every setting. What it WAS doing was
# refusing real throws whose arc had been clipped short by a detector bug (see
# ball_track.reject_static_candidates' spare_area_ratio): 5 of one session's
# throws sat at 0.35-0.52 m of unseen drop while reporting a propagated
# landing sigma of 2.6-6.7 mm.
#
# The physics permits it. A tennis ball's drag is 0.7% of weight at these
# speeds (see perception/trajectory.py), so 0.5 m of unobserved fall at ~5 m/s
# accumulates ~0.3 mm of error from the drag this fit ignores. The real
# question is whether the ARC is well determined, and that is now asked
# directly: MAX_SIGMA_XY_M below gates the propagated uncertainty of the
# answer itself rather than a proxy for it.
#
# NOT valid for the whiffle ball of the drag-crossover study, where drag is the
# entire point -- there the fit needs a drag term before any extrapolation.
MAX_UNOBSERVED_DROP_M = 0.60
MIN_OBSERVED_DROP_M = 0.30

# The landing's own propagated 1-sigma, from the fit covariance through the
# impact solve. This is the honest "is this a measurement or a guess" test:
# 15 mm is already far worse than the 2.6-6.7 mm real arcs achieve and well
# inside the ~2 cm the extrinsic itself is good for, so it refuses an
# ill-conditioned fit (the depth/velocity degeneracy of a short, poorly
# spanned track) without refusing a long clean arc that simply ends early.
MAX_SIGMA_XY_M = 0.015


def check_landing_is_observed(obs_inliers, fit, z_floor=Z_FLOOR_BASE,
                              ball_radius=BALL_RADIUS,
                              max_unobserved_drop_m=MAX_UNOBSERVED_DROP_M,
                              min_observed_drop_m=MIN_OBSERVED_DROP_M):
    """
    Refuse a landing point that is extrapolated rather than measured.

    WHY THIS EXISTS (2026-09-10). `ransac_track`'s inlier fraction used to be
    taken over the whole recording, and was quietly doing two jobs at once:
    the one it was written for (is the arc clean?) and one nobody had named
    (did the ball actually fall?). Making the fraction span-local -- necessary,
    because the post-bounce arc is most of every real recording and was making
    the gate unreachable -- removes the second job, so it gets its own gate
    here, where `z_floor` is known.

    WHAT IT CATCHES. A ball carried past the camera by hand, or rolling on the
    floor, fits a g = 9.81 parabola to sub-pixel RMS with a span-local inlier
    fraction of 1.00: over a short arc at small disparity, "slow and near" and
    "fast and far" project almost identically, so the constrained fit slides
    down that degeneracy and lands on a fast arc that has to be extrapolated a
    long way to reach the floor. Measured on the real recordings, the drop
    still left after the last observed frame is 0.12 m for a genuine arm
    throw against 1.05 m and 0.81 m for the two known hand-carried clips --
    the one quantity that separates them, and an outside fact in the same
    sense as `start_of_day.py`'s FLOOR gate.

    THE SECOND HALF OF THE SAME QUESTION. "How much was left unseen" is not
    enough on its own: throws/throw_006.npz is a ball sitting on the floor,
    creeping 3 cm in 0.16 s, and it fits a constrained g = 9.81 parabola to
    0.55 px with a span-local inlier fraction of 0.83, DESCENDING, only 0.22 m
    above the impact plane -- it passes the cap above. It escapes by putting
    its apex inside the observed span, where a parabola is locally flat and a
    jittering stationary blob looks exactly like the top of a |v0| = 15.3 m/s
    arc. The thing it never does is FALL, so the arc must also be seen to drop
    at least `min_observed_drop_m`. A real Gen3 release sits ~0.73 m above the
    floor and the real throw drops 1.01 m across its observed frames.

    THE THRESHOLDS ARE CEILINGS AND FLOORS, NOT TARGETS. 0.30 m of unseen drop
    at a
    typical ~5.4 m/s impact speed is ~0.055 s, i.e. ~10 cm of horizontal
    travel with nothing observing it -- already useless against a 1.9 cm sim
    accuracy. It is set where it is to refuse the degenerate case with margin
    (2.7x below the closest known-bad recording, 2.4x above the known-good
    one), not to bound the landing error. A throw that only just passes is a
    camera-aim problem, not a good measurement.

    `obs_inliers` is `obs[inliers]`, the rows the fit was actually made from.
    Returns a dict of what it measured; raises RuntimeError if it refuses.
    """
    ts = np.asarray(obs_inliers, float)[:, 0]
    t_first, t_last = float(ts.min()), float(ts.max())
    p_first = ballistic_position(fit.p0, fit.v0, t_first)
    p_last = ballistic_position(fit.p0, fit.v0, t_last)
    v_last = np.asarray(fit.v0, float) + G_BASE * t_last
    drop = float(p_last[2] - (z_floor + ball_radius))
    seen_drop = float(p_first[2] - p_last[2])
    descending = bool(v_last[2] < 0.0)

    if not descending:
        raise RuntimeError(
            f"the ball is still RISING ({v_last[2]:+.2f} m/s) in the last "
            f"frame it was detected in (t = {t_last:.3f} s) -- everything "
            f"between there and the floor is unobserved, so this is not a "
            f"landing measurement. Re-aim the camera to cover the descent, "
            f"or re-throw.")
    if seen_drop < min_observed_drop_m:
        raise RuntimeError(
            f"the ball was only seen to fall {seen_drop:.2f} m over the "
            f"{ts.size} frames it was tracked in ({t_first:.3f} to "
            f"{t_last:.3f} s), under the {min_observed_drop_m:.2f} m minimum "
            f"-- a fit whose apex sits inside the observed span matches a "
            f"ball that is barely moving just as well as a throw, so this is "
            f"not evidence of a flight. Aim the camera at the descent, or "
            f"check the ball actually left the hand.")
    if drop > max_unobserved_drop_m:
        raise RuntimeError(
            f"{drop:.2f} m of unobserved drop: the ball was last detected at "
            f"t = {t_last:.3f} s, still {drop:.2f} m above the impact plane "
            f"(z = {z_floor + ball_radius:+.4f}), so the landing point is "
            f"extrapolated, not measured (limit {max_unobserved_drop_m:.2f} m). "
            f"A hand-carried or rolling ball fits a parabola just as cleanly "
            f"as a throw does and is refused here, not upstream. If the throw "
            f"was real, the camera is not seeing the ball down to the floor.")
    return {"unobserved_drop_m": drop, "observed_drop_m": seen_drop,
            "t_first_seen": t_first, "t_last_seen": t_last,
            "descending": descending,
            "v_last": np.asarray(v_last, float).copy()}


def default_rig():
    return StereoRig(D435I_IR_848x480, D435I_IR_BASELINE_M)


def build_observations(rec, bg1=None, bg2=None, reject_static=True, **detect_kw):
    """
    Recording -> ((N, 5) observation array [t, u1, v1, u2, v2], max_mask_frac).

    Backgrounds default to the per-pixel temporal median of the recording
    itself, which needs no separate empty-scene capture and cannot drift
    relative to the throw.

    Frames with no detection are skipped silently -- every frame before the ball
    enters view is one of those, and it is not an error. Frames with several
    candidates contribute several rows; deciding which is the ball is
    `ransac_track`'s job, not this function's.

    `reject_static` (default True) drops, per camera stream, any candidate
    that recurs at nearly the same pixel location across many frames of this
    recording before pairing -- see `reject_static_candidates`'s docstring
    (found 2026-09-02: a permanently-mounted calibration board in the fixed
    overhead FOV otherwise floods the observation set with non-ball detections
    that dilute the real track below RANSAC's inlier gate). Pass False to get
    the raw, unfiltered candidates (e.g. for diagnosing the detector itself).
    """
    ir1, ir2, ts = rec["ir1"], rec["ir2"], np.asarray(rec["t"], float)
    if len(ir1) != len(ir2) or len(ir1) != len(ts):
        raise ValueError(f"ragged recording: {len(ir1)} ir1, {len(ir2)} ir2, "
                         f"{len(ts)} timestamps")
    if bg1 is None:
        bg1 = median_background(ir1)
    if bg2 is None:
        bg2 = median_background(ir2)

    max_frac = 0.0
    per_left, per_right = [], []
    for k in range(len(ts)):
        max_frac = max(max_frac,
                       frame_diagnostics(ir1[k], bg1)["mask_nonzero_frac"],
                       frame_diagnostics(ir2[k], bg2)["mask_nonzero_frac"])
        per_left.append(detect_candidates(ir1[k], bg1, **detect_kw))
        per_right.append(detect_candidates(ir2[k], bg2, **detect_kw))

    if reject_static:
        per_left = reject_static_candidates(per_left)
        per_right = reject_static_candidates(per_right)

    rows = []
    for k in range(len(ts)):
        left, right = per_left[k], per_right[k]
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


def select_flight_arc(obs, rig, R_bc, t_bc, z_floor=Z_FLOOR_BASE,
                      ball_radius=BALL_RADIUS,
                      max_unobserved_drop_m=MAX_UNOBSERVED_DROP_M,
                      min_observed_drop_m=MIN_OBSERVED_DROP_M,
                      seed=0, max_arcs=6,
                      inlier_fracs=(0.6, 0.5, 0.45)):
    """
    The arc that is the THROW, not merely the biggest arc in the recording.

    WHY (2026-09-11). `ransac_track` returns the LARGEST consensus set, which
    is the right answer to "which points agree on one parabola" and the wrong
    answer to "which parabola is the flight". A tile floor bounces the ball
    back up into the same field of view, and the capture window runs
    POST_S = 1.0 s past release while the flight is only ~0.2 s of it -- so the
    bounce routinely has MORE detections than the flight and wins the vote.
    The landing gates then correctly refuse it ("the ball is still RISING"),
    and a perfectly good throw is thrown away. Measured on one run-day session:
    11 of 18 refusals were exactly this, on throws the operator watched land on
    the target.

    So: pull out successive disjoint consensus sets, keep the ones that pass
    `check_landing_is_observed`, and take the EARLIEST. Earliest is not a
    heuristic -- the ball is in flight before it bounces, by construction, and
    nothing before release fits g = 9.81 at all (in the gripper it moves with
    the arm). Taking the largest instead is what caused the bug.

    `inlier_fracs` is tried in order and the FIRST value that yields a
    qualifying arc wins. Loosening is safe here in a way it would not be for a
    single-arc fit: everything downstream of this still has to pass, and those
    are the real quality measures -- >= MIN_INLIER_FRAMES, RMS < MAX_RMS_PX,
    descending, >= MIN_OBSERVED_DROP_M of fall actually seen, and a propagated
    landing sigma under MAX_SIGMA_XY_M. The span-local fraction is a weak proxy
    next to those, and on a recording holding two ballistic arcs plus their
    debris it is pessimistic by construction. Checked directly: a throw the
    strict 0.6 path accepted is reproduced at 0.5 to within 2 mm.

    Returns (inlier_indices_into_obs, FitResult, seen_dict). `seen_dict`
    carries `inlier_frac_used`, so a landing recovered only by loosening is
    identifiable after the fact rather than silently equal to a strict one.
    Raises with the most informative refusal seen if no arc qualifies.
    """
    obs = np.asarray(obs, float)
    failures = []
    for frac in inlier_fracs:
        remaining = np.arange(obs.shape[0])
        passing = []
        for _ in range(int(max_arcs)):
            if remaining.size < MIN_INLIER_FRAMES:
                break
            try:
                local, fit = ransac_track(obs[remaining], rig, R_bc, t_bc,
                                          seed=seed, min_inlier_frac=frac)
            except RuntimeError as exc:
                failures.append(str(exc))
                break
            idx = remaining[local]
            try:
                seen = check_landing_is_observed(
                    obs[idx], fit, z_floor=z_floor, ball_radius=ball_radius,
                    max_unobserved_drop_m=max_unobserved_drop_m,
                    min_observed_drop_m=min_observed_drop_m)
                seen["inlier_frac_used"] = float(frac)
                passing.append((float(obs[idx, 0].min()), idx, fit, seen))
            except RuntimeError as exc:
                failures.append(str(exc))
            remaining = np.setdiff1d(remaining, idx)
        if passing:
            passing.sort(key=lambda z: z[0])
            _t0, idx, fit, seen = passing[0]
            return idx, fit, seen

    if failures:
        raise RuntimeError(
            failures[0] + (f"  [{len(failures)} candidate arc(s) examined; none "
                           f"was a measurable descent]" if len(failures) > 1 else ""))
    raise RuntimeError(
        f"no ballistic arc found in {obs.shape[0]} observations")


def measure_landing(rec, R_bc, t_bc, z_floor=Z_FLOOR_BASE,
                    ball_radius=BALL_RADIUS, seed=0, rig=None,
                    commanded_speed=None, release_t_offset=None,
                    speed_ratio_tol=2.0,
                    max_unobserved_drop_m=MAX_UNOBSERVED_DROP_M,
                    min_observed_drop_m=MIN_OBSERVED_DROP_M,
                    max_sigma_xy_m=MAX_SIGMA_XY_M, **detect_kw):
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

    `commanded_speed`, if given, sanity-checks the fitted release speed against
    the throw's own commanded release speed (already known and logged
    per-throw -- an outside fact, same principle `start_of_day.py`'s gates
    use, since RANSAC/RMS cannot catch this). `release_t_offset` is REQUIRED
    alongside it: `p0`/`v0` are fit parameters at the recording's local
    `t = 0`, which for a `session_camera.RingBuffer`-style capture is
    `PRE_S` seconds BEFORE the ball is ever released (`t` is zeroed to the
    window's first kept frame, and the window starts at `t_release - PRE_S`)
    -- NOT at release. Comparing raw `v0` to a commanded release speed is
    comparing the wrong instant: found 2026-09-02, a real throw's `v0`
    included ~0.45s of backward gravity extrapolation through a period the
    ball was still in the gripper, inflating the vertical component by
    `g * PRE_S =~ 4.4 m/s` on its own and making a fine measurement look like
    an 18x-wrong one. The check instead evaluates velocity at
    `t = release_t_offset` (`v0 + g * release_t_offset`) before comparing.
    Passing `commanded_speed` without `release_t_offset` raises ValueError --
    silently falling back to the wrong instant is exactly the bug this
    guards against. Both `None` (default) skips the check entirely for
    callers with no commanded speed to compare against (offline re-analysis,
    the standalone CLI).

    Refuses when the release-time speed is outside
    `[1/speed_ratio_tol, speed_ratio_tol] * commanded_speed`.
    """
    if commanded_speed is not None and release_t_offset is None:
        raise ValueError(
            "commanded_speed requires release_t_offset -- p0/v0 are fit at "
            "the recording's local t=0, which is PRE_S seconds BEFORE "
            "release for a RingBuffer-style capture, not at release itself. "
            "Comparing raw v0 to a commanded release speed compares the "
            "wrong instant (found 2026-09-02: inflated a fine measurement "
            "by ~g*PRE_S in the vertical component). Pass the offset from "
            "release to the window's local t=0 for this capture's own "
            "convention (e.g. session_camera.PRE_S).")

    rig = rig or default_rig()
    obs, max_frac = build_observations(rec, **detect_kw)
    inliers, fit, seen = select_flight_arc(
        obs, rig, R_bc, t_bc, z_floor=z_floor, ball_radius=ball_radius,
        max_unobserved_drop_m=max_unobserved_drop_m,
        min_observed_drop_m=min_observed_drop_m, seed=seed)

    if commanded_speed is not None:
        v_release = fit.v0 + G_BASE * release_t_offset
        measured_speed = float(np.linalg.norm(v_release))
        lo, hi = commanded_speed / speed_ratio_tol, commanded_speed * speed_ratio_tol
        if not (lo <= measured_speed <= hi):
            raise RuntimeError(
                f"fit passed RANSAC ({inliers.size}/{obs.shape[0]} frames, "
                f"{fit.rms_px:.2f} px RMS) but release-time speed "
                f"{measured_speed:.2f} m/s (v0 shifted by {release_t_offset:.3f}s) "
                f"is outside [{lo:.2f}, {hi:.2f}] m/s of the commanded "
                f"{commanded_speed:.2f} m/s -- likely a depth/velocity "
                f"degeneracy from a short or poorly-conditioned track, not a "
                f"real measurement. Refusing rather than reporting "
                f"a plausible-looking wrong number; re-throw.")

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
    sigma_xy = float(np.sqrt(max(np.trace(cov_xy), 0.0)))
    if sigma_xy > max_sigma_xy_m:
        raise RuntimeError(
            f"the landing point's own 1-sigma is {sigma_xy * 1000:.0f} mm, over "
            f"the {max_sigma_xy_m * 1000:.0f} mm limit -- the arc does not "
            f"determine where the ball hit. This is what a short or poorly "
            f"spanned track looks like once the fit covariance is propagated "
            f"through the impact solve, and no threshold elsewhere can "
            f"substitute for it. Re-throw; do not report this number.")

    return {"x": x, "y": y, "t_impact": t_imp,
            "sigma_xy_m": sigma_xy,
            "n_frames": int(obs.shape[0]), "n_inliers": int(inliers.size),
            "rms_px": fit.rms_px, "p0": fit.p0, "v0": fit.v0,
            "max_mask_frac": max_frac,
            "unobserved_drop_m": seen["unobserved_drop_m"],
            "observed_drop_m": seen["observed_drop_m"],
            "inlier_frac_used": seen.get("inlier_frac_used"),
            "t_first_seen": seen["t_first_seen"],
            "t_last_seen": seen["t_last_seen"]}


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
    print(f"  fall seen        : {out['observed_drop_m'] * 100:.1f} cm over "
          f"t = {out['t_first_seen']:.3f} to {out['t_last_seen']:.3f} s")
    print(f"  fall NOT seen    : {out['unobserved_drop_m'] * 100:.1f} cm between the "
          f"last frame and the floor")
    print(f"  release p0       : {np.array2string(out['p0'], precision=4)}")
    print(f"  release v0       : {np.array2string(out['v0'], precision=4)}  "
          f"|v0| = {np.linalg.norm(out['v0']):.4f} m/s")
    print("\nNOTE: absolute accuracy is bounded by T_B_C, not by the vision. "
          "Confirm the extrinsic is current for the present mount.")


if __name__ == "__main__":
    main()
