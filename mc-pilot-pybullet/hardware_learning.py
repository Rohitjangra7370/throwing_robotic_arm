"""
The parts of the hardware throw session that are decisions or mathematics
rather than plumbing, kept free of Tk, PyBullet and pyrealsense2 so they can be
tested without a window, an arm, or a camera.

See docs/superpowers/specs/2026-08-31-hardware-session-design.md.
"""
from __future__ import annotations

import numpy as np

SCALE_LADDER = (0.15, 0.30, 0.60, 1.00)
TRAINED_BAND = ((0.68, 0.74), (-0.25, 0.25))


def propose_targets(n, band=TRAINED_BAND, seed=0):
    """
    `n` targets spread across the trained band.

    Stratified, not uniform-random: n random draws routinely cluster, and ten
    throws that land in the same place tell the GP almost nothing it does not
    already believe. y gets the stratification because it is the wide axis;
    x is jittered inside its much narrower range.
    """
    (x_lo, x_hi), (y_lo, y_hi) = band
    rng = np.random.default_rng(seed)
    edges = np.linspace(y_lo, y_hi, n + 1)
    ys = edges[:-1] + rng.uniform(0.0, 1.0, n) * np.diff(edges)
    xs = x_lo + rng.uniform(0.0, 1.0, n) * (x_hi - x_lo)
    order = rng.permutation(n)
    return np.stack([xs[order], ys[order]], axis=1)


def next_allowed_scale(logged_scales, ladder=SCALE_LADDER):
    """Highest rung reachable given the clean runs logged so far."""
    best = -1
    for s in logged_scales:
        for i, rung in enumerate(ladder):
            if abs(s - rung) < 1e-9:
                best = max(best, i)
    return ladder[min(best + 1, len(ladder) - 1)]


def scale_allowed(requested, logged_scales, ladder=SCALE_LADDER):
    """
    (ok, reason). The runbook's escalation ladder, enforced rather than advised:
    every rung needs a clean run at the rung below it first. Repeating a rung or
    dropping back down is always fine.
    """
    ceiling = next_allowed_scale(logged_scales, ladder)
    if requested <= ceiling + 1e-9:
        return True, ""
    return False, (f"speed_scale {requested:.2f} needs a clean logged run at "
                   f"{ceiling:.2f} first -- escalate one rung at a time")


TS_DEFAULT = 0.02          # T_sampling, must match the trained checkpoint


def track_to_state_samples(points_base, times, target_xy, commanded_speed,
                           ts=TS_DEFAULT):
    """
    A measured flight -> the (state_samples, input_samples) pair the model's
    `add_data` consumes, shaped exactly as `PyBulletThrowingSystem.rollout`
    returns them: states (n, 8) = [x, y, z, vx, vy, vz, Px, Py], inputs (n, 1)
    with the release speed at [0, 0] and zeros after.

    `points_base` MUST be the raw triangulated RANSAC inliers, never a
    resampled `fit_ballistic` output. The fit is gravity-only, so feeding it
    back would hand the GP `dv = g*dt` -- its own assumption returned as
    evidence. That failure is invisible in the cost curve, which is exactly why
    it gets a test (`test_pure_parabola_teaches_the_gp_nothing`).

    Position is resampled by local linear interpolation onto the Ts grid and
    velocity by central differences of the resampled positions, so the states
    obey the same `p_{t+1} = p_t + Ts*v_t + (Ts/2)*dv` relation the GP's
    propagation assumes (paper Eq. 18).
    """
    p = np.asarray(points_base, float)
    t = np.asarray(times, float).reshape(-1)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"expected (N, 3) points, got {p.shape}")
    if t.size < 4:
        raise ValueError(f"track too short to difference: {t.size} samples")

    t0 = t - t[0]
    grid = np.arange(0.0, t0[-1] + 1e-12, ts)
    if grid.size < 3:
        raise ValueError(f"track too short to difference: spans {t0[-1]:.3f} s at ts={ts}")

    pos = np.stack([np.interp(grid, t0, p[:, k]) for k in range(3)], axis=1)
    vel = np.gradient(pos, ts, axis=0, edge_order=2)

    n = grid.size
    states = np.zeros((n, 8))
    states[:, 0:3] = pos
    states[:, 3:6] = vel
    states[:, 6] = float(target_xy[0])
    states[:, 7] = float(target_xy[1])

    inputs = np.zeros((n, 1))
    inputs[0, 0] = float(commanded_speed)
    return states, inputs
