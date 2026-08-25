"""
Ballistic trajectory fitting and the floor-impact solve.

WHAT THIS MEASURES, AND WHY IT IS NOT THE RESTING POSITION
-----------------------------------------------------------
`ball_detector.py` measures where the ball SETTLES. On a tile floor a ball
thrown at ~1.6 m/s bounces and rolls, so the settled position is not the
first-contact point -- and it is first contact that the policy's landing error
is defined against. The gap is a systematic, and nothing in the static pipeline
bounds it. Fitting the flight and solving for the floor crossing measures first
contact directly.

WHY NO DRAG TERM
----------------
Tennis ball, 57.7 g, 65.4 mm diameter (fixed by the trained checkpoints and
HARDWARE_RUNBOOK.md). Drag at 2 m/s is 0.5*1.2*0.5*3.36e-3*v^2 = 4.0e-3 N
against 0.566 N of weight -- 0.7%. A drag-free parabola is valid HERE.

It is NOT valid for the whiffle ball (0.004 kg / 0.06 m) of the drag-crossover
study, where drag is the entire point. If this is ever pointed at that ball,
`g` alone will not describe the motion and a drag term has to be added to
`ballistic_position` and to the fit residual together.

FRAME. Base frame throughout: base at z = 0, floor at z = -base_height =
-0.433, g = (0, 0, -9.81).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["G_BASE", "Z_FLOOR_BASE", "BALL_RADIUS", "ballistic_position",
           "solve_impact_time", "solve_impact", "fit_two_points",
           "FitResult", "fit_ballistic", "MIN_INLIER_FRAMES", "MAX_RMS_PX",
           "ransac_track"]

G_BASE = np.array([0.0, 0.0, -9.81])
Z_FLOOR_BASE = -0.433      # measured base plate; see CLAUDE.md's frame note
BALL_RADIUS = 0.0327       # tennis ball, matches the trained checkpoints


def ballistic_position(p0, v0, t, g=G_BASE):
    """p(t) = p0 + v0*t + 0.5*g*t^2. Scalar t -> (3,); array t -> (N, 3)."""
    p0 = np.asarray(p0, float); v0 = np.asarray(v0, float); g = np.asarray(g, float)
    t = np.asarray(t, float)
    return p0 + v0 * t[..., None] + 0.5 * g * (t[..., None] ** 2)


def solve_impact_time(p0, v0, z_target, g=G_BASE):
    """
    Time of the DESCENDING crossing of the horizontal plane z = z_target.

    A trajectory that rises through the plane and falls back has TWO positive
    roots. The smaller one is the ball on its way up; reporting it would give a
    confident landing point for a ball that has not landed. Always the larger.

    Raises RuntimeError if the plane is never reached.
    """
    p0 = np.asarray(p0, float); v0 = np.asarray(v0, float); g = np.asarray(g, float)
    a = 0.5 * float(g[2])
    b = float(v0[2])
    c = float(p0[2]) - float(z_target)
    if a == 0.0:
        raise RuntimeError("zero vertical acceleration -- not a ballistic arc")
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        raise RuntimeError(
            f"trajectory never reaches z = {z_target:.4f} (discriminant "
            f"{disc:.4e}); apex is z = {float(p0[2]) - b * b / (4.0 * a):.4f}")
    root = np.sqrt(disc)
    # a < 0, so the LARGER time is (-b - root)/(2a). Both roots computed so the
    # failure message can say which ones existed.
    t_desc = (-b - root) / (2.0 * a)
    t_asc = (-b + root) / (2.0 * a)
    if t_desc <= 0.0:
        raise RuntimeError(
            f"no descending crossing at positive time (roots {t_asc:.4f}, "
            f"{t_desc:.4f} s) -- the ball is already past this plane")
    return float(t_desc)


def solve_impact(p0, v0, z_floor=Z_FLOOR_BASE, ball_radius=BALL_RADIUS, g=G_BASE):
    """
    First-contact point on the floor: returns (x, y, t).

    Solves for the ball's CENTRE reaching z = z_floor + ball_radius, because
    that is where the centre is at the instant the sphere touches down. For a
    sphere the centre's (x, y) is the contact (x, y). Identical convention to
    `ray_plane.ball_center_on_plane` -- the two must agree or comparing them
    means nothing.
    """
    t = solve_impact_time(p0, v0, float(z_floor) + float(ball_radius), g=g)
    p = ballistic_position(p0, v0, t, g=g)
    return float(p[0]), float(p[1]), float(t)


def fit_two_points(t_a, p_a, t_b, p_b, g=G_BASE, min_dt=0.02):
    """
    Exact (p0, v0) through two timed 3-D points. Six equations, six unknowns.

    This is RANSAC's minimal sample. `min_dt` rejects a degenerate pair: two
    nearly simultaneous points leave v0 = dp/dt numerically meaningless, and the
    resulting wild trajectory would be scored against the data as if it were a
    real hypothesis.
    """
    g = np.asarray(g, float)
    t_a, t_b = float(t_a), float(t_b)
    if abs(t_b - t_a) < min_dt:
        raise RuntimeError(f"sample points are separated by only "
                           f"{abs(t_b - t_a) * 1e3:.1f} ms (need >= "
                           f"{min_dt * 1e3:.0f} ms) -- degenerate")
    q_a = np.asarray(p_a, float) - 0.5 * g * t_a ** 2
    q_b = np.asarray(p_b, float) - 0.5 * g * t_b ** 2
    v0 = (q_b - q_a) / (t_b - t_a)
    p0 = q_a - v0 * t_a
    return p0, v0


MIN_INLIER_FRAMES = 12     # spec section 6: 2x redundancy over 6 unknowns
MAX_RMS_PX = 1.0           # spec section 6, vs 0.15 px assumed centroid precision


@dataclass
class FitResult:
    """A fitted trajectory and how much to trust it."""
    p0: np.ndarray
    v0: np.ndarray
    cov: np.ndarray        # 6x6 over [p0, v0]
    rms_px: float
    n_obs: int


def _residuals(theta, obs, rig, R_bc, t_bc, g):
    """Stacked (u1, v1, u2, v2) reprojection residuals, 4 per observation."""
    p0, v0 = theta[:3], theta[3:]
    p_b = ballistic_position(p0, v0, obs[:, 0], g=g)
    p_c = (p_b - t_bc) @ R_bc            # == (R_bc.T @ (p_b - t_bc).T).T
    u1, v1, u2, v2 = rig.project(p_c)
    pred = np.stack([u1, v1, u2, v2], axis=-1)
    return (pred - obs[:, 1:]).ravel()


def fit_ballistic(obs, rig, R_bc, t_bc, g=G_BASE, max_iter=50, tol=1e-10):
    """
    Fit (p0, v0) to timed stereo pixel observations by Gauss-Newton.

    `obs` is (N, 5): columns [t, u1, v1, u2, v2]. `R_bc`, `t_bc` are the camera
    pose in base coordinates (p_base = R_bc @ p_cam + t_bc).

    THE OBJECTIVE IS PIXEL ERROR, NOT ERROR AGAINST TRIANGULATED POINTS.
    Pixel noise is the iid quantity. Triangulated points carry correlated,
    strongly range-dependent noise (a fixed 0.2 px of disparity is 4 cm at 2 m
    and 1 cm at 1 m), so least-squares over them silently weights the far,
    noisier frames as heavily as the near ones. Fitting in pixels is the
    statistically correct thing and costs nothing offline.

    The Jacobian is computed by central differences ON PURPOSE. It is 6 columns
    over a few hundred residuals -- microseconds offline -- and this codebase has
    a long history of silent sign errors in hand-derived geometry. A wrong
    analytic Jacobian does not crash; it converges somewhere plausible.
    """
    obs = np.asarray(obs, float)
    if obs.ndim != 2 or obs.shape[1] != 5:
        raise ValueError(f"obs must be (N, 5) [t,u1,v1,u2,v2], got {obs.shape}")
    if obs.shape[0] < MIN_INLIER_FRAMES:
        raise RuntimeError(
            f"only {obs.shape[0]} observations, need >= {MIN_INLIER_FRAMES} "
            f"(2x redundancy over 6 unknowns) -- refusing to fit")

    R_bc = np.asarray(R_bc, float); t_bc = np.asarray(t_bc, float)
    g = np.asarray(g, float)
    order = np.argsort(obs[:, 0])
    obs = obs[order]

    # Initialise from the two most widely separated frames, triangulated.
    p_first = rig.triangulate(*obs[0, 1:])
    p_last = rig.triangulate(*obs[-1, 1:])
    to_base = lambda p_c: R_bc @ p_c + t_bc
    p0, v0 = fit_two_points(obs[0, 0], to_base(p_first),
                            obs[-1, 0], to_base(p_last), g=g)
    theta = np.concatenate([p0, v0])

    r = _residuals(theta, obs, rig, R_bc, t_bc, g)
    for _ in range(max_iter):
        J = np.empty((r.size, 6))
        for k in range(6):
            step = 1e-6 * max(1.0, abs(theta[k]))
            tp = theta.copy(); tp[k] += step
            tm = theta.copy(); tm[k] -= step
            J[:, k] = (_residuals(tp, obs, rig, R_bc, t_bc, g)
                       - _residuals(tm, obs, rig, R_bc, t_bc, g)) / (2.0 * step)
        delta, *_ = np.linalg.lstsq(J, -r, rcond=None)
        theta = theta + delta
        r = _residuals(theta, obs, rig, R_bc, t_bc, g)
        if np.linalg.norm(delta) < tol:
            break

    m = r.size
    dof = m - 6
    if dof <= 0:
        raise RuntimeError(f"{m} residuals cannot constrain 6 parameters")
    sigma2 = float(r @ r) / dof
    JtJ = J.T @ J
    try:
        cov = sigma2 * np.linalg.inv(JtJ)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(f"singular normal equations -- the trajectory is "
                           f"not observable from these frames: {exc}") from exc

    rms_px = float(np.sqrt(float(r @ r) / m))
    return FitResult(p0=theta[:3].copy(), v0=theta[3:].copy(), cov=cov,
                     rms_px=rms_px, n_obs=int(obs.shape[0]))


def ransac_track(obs, rig, R_bc, t_bc, g=G_BASE, thresh_px=2.0,
                 min_inlier_frac=0.6, iters=300, seed=0):
    """
    Pick out the frames that lie on one ballistic arc, then fit them.

    THIS IS THE ARM REJECTOR. The arm moves fast in exactly the region the ball
    starts in, and it is the dominant false-positive source. Rather than mask it
    out by hand -- which would need re-drawing for every mount and every pose,
    and would silently clip real detections -- this exploits the one property the
    ball has and the arm does not: the ball's positions fit a parabola with
    g = 9.81. Reflections and the second bounce fall out for the same reason.

    Minimal sample is 2 timed 3-D points (6 equations, 6 unknowns). Scoring is
    in pixels, consistent with `fit_ballistic`'s objective.

    Returns (inlier_indices_into_obs, FitResult).
    """
    obs = np.asarray(obs, float)
    if obs.ndim != 2 or obs.shape[1] != 5:
        raise ValueError(f"obs must be (N, 5) [t,u1,v1,u2,v2], got {obs.shape}")
    n = obs.shape[0]
    if n < MIN_INLIER_FRAMES:
        raise RuntimeError(f"only {n} observations, need >= {MIN_INLIER_FRAMES}")

    R_bc = np.asarray(R_bc, float); t_bc = np.asarray(t_bc, float)
    g = np.asarray(g, float)
    rng = np.random.default_rng(seed)

    # Triangulate once. A pair that cannot be triangulated at all (bad disparity,
    # mismatched row) is dropped here rather than poisoning the sampling.
    usable, pts_b = [], []
    for i in range(n):
        try:
            p_c = rig.triangulate(*obs[i, 1:])
        except RuntimeError:
            continue
        usable.append(i)
        pts_b.append(R_bc @ p_c + t_bc)
    usable = np.asarray(usable, int)
    if usable.size < MIN_INLIER_FRAMES:
        raise RuntimeError(
            f"only {usable.size} of {n} observations triangulate at all "
            f"(need >= {MIN_INLIER_FRAMES}) -- check left/right pairing")
    pts_b = np.asarray(pts_b, float)

    best_idx = np.empty(0, dtype=int)
    for _ in range(iters):
        a, b = rng.choice(usable.size, size=2, replace=False)
        try:
            p0, v0 = fit_two_points(obs[usable[a], 0], pts_b[a],
                                    obs[usable[b], 0], pts_b[b], g=g)
        except RuntimeError:
            continue
        try:
            err = np.abs(_residuals(np.concatenate([p0, v0]), obs[usable],
                                    rig, R_bc, t_bc, g)).reshape(-1, 4)
        except RuntimeError:
            # A wild minimal-sample hypothesis can put the arc behind the camera,
            # where project() has no answer. That is a rejected hypothesis, not an
            # error -- drop the sample and keep searching.
            continue
        inl = usable[np.max(err, axis=1) <= thresh_px]
        if inl.size > best_idx.size:
            best_idx = inl

    frac = best_idx.size / float(n)
    if best_idx.size < MIN_INLIER_FRAMES or frac < min_inlier_frac:
        raise RuntimeError(
            f"no ballistic arc found: best inlier consensus {best_idx.size}/{n} "
            f"frames (inlier fraction {frac:.2f}, need >= {min_inlier_frac} and "
            f">= {MIN_INLIER_FRAMES} frames) -- this recording does not contain "
            f"a clean throw")

    fit = fit_ballistic(obs[best_idx], rig, R_bc, t_bc, g=g)
    if fit.rms_px > MAX_RMS_PX:
        raise RuntimeError(
            f"fit RMS {fit.rms_px:.2f} px exceeds {MAX_RMS_PX} px -- the frames "
            f"agree on an arc but not a good one; do not use this landing point")
    return best_idx, fit
