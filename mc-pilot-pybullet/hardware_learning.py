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


POS_SIGMA_INDEPENDENT_M = 0.010   # per-frame stereo triangulation; survives differencing
POS_SIGMA_SYSTEMATIC_M = 0.018    # extrinsic translation repeatability; CANCELS in a difference,
                                  # so it must NOT enter velocity_noise_sigma -- kept for reference
EXTRINSIC_ROT_SIGMA_DEG = 0.56    # measured extrinsic rotation repeatability
G = 9.81


def velocity_noise_sigma(pos_sigma_m=POS_SIGMA_INDEPENDENT_M, ts=TS_DEFAULT):
    """
    Position noise propagated into a central-difference velocity.

    v_k = (p_{k+1} - p_{k-1}) / (2*ts), so sigma_v = sqrt(2)*sigma_p / (2*ts).

    Only per-frame-independent noise survives differencing; extrinsic translation
    (systematic) errors cancel in the difference p_{k+1} - p_{k-1}, so they must
    NOT enter this estimate. The default uses stereo triangulation noise (10 mm)
    only, not the extrinsic repeatability (18 mm). This is critical: including
    the systematic term would yield ~0.73 m/s noise, which is 640× the drag signal
    and makes a correct verdict impossible.

    At 10 mm and 50 Hz this is ~0.22 m/s, still ~280× the drag signal for a
    tennis ball. The ensemble test in deviation_verdict is the only way to
    resolve whether there is real structure in the noise.
    """
    return float(np.sqrt(2.0) * pos_sigma_m / (2.0 * ts))


def systematic_dv_floor(rot_sigma_deg=EXTRINSIC_ROT_SIGMA_DEG, ts=TS_DEFAULT, g=G):
    """
    The per-step Delta-v that an extrinsic ROTATION error alone would produce.

    A rotation error tilts the whole trajectory, so gravity in the calibrated
    frame is not quite vertical, and the residual reads as a constant horizontal
    acceleration -- indistinguishable in form from drag. Unlike random noise
    this does NOT average down with more samples, so it is a hard floor on any
    aerodynamic claim from this rig: 0.56 deg gives 0.0019 m/s per step, which
    is 2.5x the drag signal for a tennis ball at this speed.
    """
    return float(g * np.sin(np.radians(rot_sigma_deg)) * ts)


def deviation_verdict(dv_learned, sigma_v=None, k=2.0):
    """
    Is the non-ballistic correction the GP claims to have found bigger than the
    noise it was fitted through, AND bigger than what a mis-calibrated extrinsic
    could fake?

    `dv_learned` is the GP's predicted delta-v minus the pure-gravity delta-v,
    i.e. only the part that is not already assumed. Uses an ensemble test:
    mean_dev > k * SE AND mean_dev > systematic_floor. Both must be true for
    ABOVE NOISE. The first condition checks random-noise threshold, the second
    checks whether the result could be aliased extrinsic rotation error.

    Returns dict with keys: rms_deviation, rms_sigma, ratio, mean_deviation,
    standard_error, systematic_floor, above_noise, n_samples, text. When the
    verdict is BELOW NOISE and only the random-noise threshold is the blocker
    (systematic floor is cleared), the text includes the sample count that would
    be needed to resolve the measured signal. When the systematic floor is the
    blocker, the text explains that it is a hard limit requiring better calibration.
    """
    d = np.asarray(dv_learned, float)
    sigma_v = velocity_noise_sigma() if sigma_v is None else float(sigma_v)
    n = d.shape[0]

    # Per-sample diagnostics (kept for inspection, not the verdict)
    rms_d = float(np.sqrt(np.mean(d ** 2)))
    denom_per_sample = k * sigma_v
    ratio = rms_d / denom_per_sample if denom_per_sample > 0 else 0.0

    # Ensemble test
    mean_d = float(np.linalg.norm(np.mean(d, axis=0)))
    se = sigma_v / np.sqrt(n) if n > 0 else np.inf
    sys_floor = systematic_dv_floor()

    # Both conditions required for ABOVE NOISE
    above_se = mean_d > k * se
    above_sys = mean_d > sys_floor
    above = above_se and above_sys

    condition_text = ""
    remedy_text = ""
    if above:
        condition_text = f"Exceeds both {k:g}*SE={k*se:.4f} m/s and systematic floor {sys_floor:.4f} m/s."
    elif above_se and not above_sys:
        # SE cleared, but systematic floor is the blocker
        condition_text = f"Exceeds random-noise threshold ({k:g}*SE={k*se:.4f}) but NOT systematic floor ({sys_floor:.4f} m/s)."
        remedy_text = f"The systematic floor is a hard limit from extrinsic rotation error ({sys_floor:.4f} m/s); more samples cannot resolve it — the remedy is better calibration."
    elif above_sys and not above_se:
        # Systematic floor cleared, SE is the blocker
        condition_text = f"Exceeds systematic floor ({sys_floor:.4f} m/s) but NOT random-noise threshold ({k:g}*SE={k*se:.4f})."
        if mean_d > 1e-12:
            n_required = int(np.ceil((k * sigma_v / mean_d) ** 2))
            remedy_text = f"To resolve this mean deviation above the noise threshold would require ~{n_required} samples at this effect size."
    else:
        # Both conditions fail
        condition_text = f"Below both: {k:g}*SE={k*se:.4f} m/s, systematic floor {sys_floor:.4f} m/s."
        remedy_text = "The systematic floor (extrinsic rotation error) is the binding constraint. Sampling cannot resolve this — a better calibration is required."

    text = (f"{'ABOVE NOISE' if above else 'BELOW NOISE'}: "
            f"mean deviation {mean_d:.4f} m/s vs {k:g}*SE {k*se:.4f} m/s "
            f"over {n} samples. {condition_text} "
            + (remedy_text if remedy_text else ""))

    return {
        "rms_deviation": rms_d,
        "rms_sigma": sigma_v,
        "ratio": ratio,
        "mean_deviation": mean_d,
        "standard_error": float(se),
        "systematic_floor": sys_floor,
        "above_noise": bool(above),
        "n_samples": int(n),
        "text": text
    }
