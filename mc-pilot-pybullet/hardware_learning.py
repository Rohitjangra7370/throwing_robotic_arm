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

    The first and last velocity samples (from one-sided differentiation) are the
    noisiest in the array. Do not use them as a release-velocity estimate; the
    fitted `v0` from `measure_landing` is the right source.
    """
    p = np.asarray(points_base, float)
    t = np.asarray(times, float).reshape(-1)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"expected (N, 3) points, got {p.shape}")
    if t.size < 4:
        raise ValueError(f"track too short to difference: {t.size} samples")

    t0 = t - t[0]
    if np.any(np.diff(t0) < 0.0):
        raise ValueError(
            "times must be non-decreasing -- np.interp silently returns garbage "
            "for unordered xp (measured: 3.8 cm position / 0.65 m/s velocity "
            "error from two swapped samples, with no exception). ransac_track "
            "preserves chronological order, so this means the caller reordered "
            "or merged tracks")

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


# 18 mm extrinsic repeatability (measured 2026-08-31, 5 solves, static rig)
# combined with ~10 mm stereo triangulation noise at 1.6 m.
POS_SIGMA_M = float(np.hypot(0.018, 0.010))


def velocity_noise_sigma(pos_sigma_m=POS_SIGMA_M, ts=TS_DEFAULT):
    """
    Position noise propagated into a central-difference velocity.

    v_k = (p_{k+1} - p_{k-1}) / (2*ts), so sigma_v = sqrt(2)*sigma_p / (2*ts).
    At 2.06 cm and 50 Hz this is ~0.73 m/s, which is half the release speed --
    the honest reason a per-sample velocity from this rig cannot resolve drag.
    """
    return float(np.sqrt(2.0) * pos_sigma_m / (2.0 * ts))


def deviation_verdict(dv_learned, sigma_v=None, k=2.0):
    """
    Is the non-ballistic correction the GP claims to have found bigger than the
    noise it was fitted through?

    `dv_learned` is the GP's predicted delta-v minus the pure-gravity delta-v,
    i.e. only the part that is not already assumed. ABOVE NOISE requires
    RMS(deviation) > k * RMS(sigma). Both numbers and the sample count go into
    the text, always -- a verdict without its evidence is how a noise-sized
    number becomes a claimed discovery.
    """
    d = np.asarray(dv_learned, float)
    sigma_v = velocity_noise_sigma() if sigma_v is None else float(sigma_v)
    rms_d = float(np.sqrt(np.mean(d ** 2)))
    ratio = rms_d / (k * sigma_v) if sigma_v > 0 else np.inf
    above = rms_d > k * sigma_v
    text = (f"{'ABOVE NOISE' if above else 'BELOW NOISE'}: "
            f"RMS deviation {rms_d:.4f} m/s vs {k:g}x RMS sigma "
            f"{sigma_v:.4f} m/s over {d.shape[0]} samples "
            f"(ratio {ratio:.2f}). "
            + ("The GP found structure the noise cannot explain."
               if above else
               "The GP learned nothing distinguishable from measurement noise -- "
               "expected for a tennis ball at this speed, where drag displaces "
               "~5 mm against ~21 mm of position noise. Report it as such."))
    return {"rms_deviation": rms_d, "rms_sigma": sigma_v, "ratio": ratio,
            "above_noise": bool(above), "n_samples": int(d.shape[0]), "text": text}
