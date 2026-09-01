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


SPEED_SCALE_TOL = 1e-6
FULL_SPEED = 1.0


def fit_release_model(records):
    """
    Commanded release speed -> measured release speed, over the logged throws
    -- but ONLY over throws executed at FULL SPEED (speed_scale == 1.0).

    This is the term worth fitting. At full speed the flight is ballistic to
    within ~5 mm while the release carries 2.9-3.7 cm of command quantisation
    plus ~1 cm of gripper-latency residual, so the discrepancy between what the
    policy asked for and what the ball actually left with is both large and
    directly observable in `measured_v0`.

    A logged throw at speed_scale < 1.0 is a REHEARSAL, required by
    HARDWARE_RUNBOOK.md's escalation ladder (0.15 -> 0.30 -> 0.60 -> 1.00)
    ahead of any full-speed throw. `speed_scale` is a time-stretch on the
    streamed joint speeds (`qd_cmd = qd * ds_dwall` in kinova_hardware.py), so
    a 0.15 throw really does release at roughly 0.15x speed -- it is not the
    same throw measured noisily, it is a physically slower throw. Fitting it
    as if it were data drags the gain toward the rehearsal ratio (observed
    failure mode: gain ~0.15 instead of ~0.9) while still reporting a
    confident-looking residual sigma beside it -- worse than refusing, because
    it *looks* trustworthy.

    A record with NO `speed_scale` key is treated as NON-QUALIFYING, the same
    as a sub-1.0 one -- it is deliberately NOT assumed to be a full-speed
    throw. Defaulting a missing field to "counts as data" would be exactly
    the kind of silent corruption this function already guards
    `measured_v0` against.

    Refused throws (no measurement) are skipped, not imputed.

    Returns a dict with an additional `n_excluded_rehearsal` key: throws that
    had a valid measurement but were excluded for being below full speed (or
    missing `speed_scale`). That count is also named in `text` whenever it is
    nonzero -- silently dropping rows is its own hazard in this codebase.
    """
    cmd, meas, dirs = [], [], []
    n_measured_total = 0
    n_excluded_rehearsal = 0
    for r in records:
        v0 = r.get("measured_v0")
        if v0 is None or r.get("commanded_speed") is None:
            continue
        # Silently skip any malformed measured_v0: not a 3-element vector, empty,
        # zero-magnitude, or non-finite (NaN, inf). A single bad record can corrupt
        # the fit (observed: zero vector among two good records produced gain=-6.5),
        # so guard tightly here.
        try:
            v = np.asarray(v0, float).ravel()
            if v.size != 3 or np.linalg.norm(v) <= 1e-9 or not np.all(np.isfinite(v)):
                continue
        except (ValueError, TypeError):
            continue
        n_measured_total += 1

        scale = r.get("speed_scale")
        try:
            is_full_speed = (scale is not None
                             and abs(float(scale) - FULL_SPEED) < SPEED_SCALE_TOL)
        except (TypeError, ValueError):
            is_full_speed = False
        if not is_full_speed:
            n_excluded_rehearsal += 1
            continue

        cmd.append(float(r["commanded_speed"]))
        meas.append(float(np.linalg.norm(v)))
        dirs.append(v / np.linalg.norm(v))

    if len(cmd) < 3:
        if n_measured_total >= 3:
            # "3 measured but all rehearsals" -- a different operator action
            # (run full-speed throws) than "fewer than 3 measured at all"
            # (run more throws, period). Kept as two distinct messages.
            raise ValueError(
                f"{n_measured_total} measured throws logged, but only "
                f"{len(cmd)} at full speed (speed_scale == 1.0) -- "
                f"{n_excluded_rehearsal} were rehearsal-speed (or missing "
                f"speed_scale) and excluded. Need at least 3 measured throws "
                f"AT FULL SPEED to fit a release model -- run more "
                f"full-speed throws, not more rehearsals.")
        raise ValueError(
            f"need at least 3 measured throws to fit a release model, have "
            f"{n_measured_total} measured total ({len(cmd)} at full speed)")
    # Why 3 is the floor: a line through 2 points fits exactly, yielding
    # residual_sigma = 0 with zero degrees of freedom. This reads as a perfect
    # model even for noisy data. The operator uses residual_sigma to decide
    # whether to trust the release correction, so a structural zero would
    # misreport a degenerate fit. Three points give 1 DoF and meaningful residual.

    c = np.asarray(cmd)
    m = np.asarray(meas)
    A = np.stack([c, np.ones_like(c)], axis=1)
    (gain, offset), *_ = np.linalg.lstsq(A, m, rcond=None)
    resid = m - (gain * c + offset)
    sigma = float(np.std(resid, ddof=min(2, len(c) - 1)))

    d = np.asarray(dirs)
    # Maximum pairwise angle: the honest measure of "release direction spread".
    # Previous code computed max deviation from mean direction, which produced
    # spurious results: normalizing a near-cancelling mean was numerically
    # unstable (on this test's 120°-apart input: ~180° from floating-point residue).
    spread = 0.0
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            angle = np.degrees(np.arccos(np.clip(np.dot(d[i], d[j]), -1, 1)))
            spread = max(spread, angle)

    excl_text = (f"; excluded {n_excluded_rehearsal} rehearsal throws at "
                f"speed_scale < 1.0" if n_excluded_rehearsal else "")
    return {"gain": float(gain), "offset": float(offset),
            "residual_sigma": sigma, "n": int(len(c)),
            "n_excluded_rehearsal": int(n_excluded_rehearsal),
            "direction_error_deg": float(spread),
            "text": (f"measured |v0| = {gain:.4f} * commanded + {offset:+.4f} m/s, "
                     f"residual sigma {sigma:.4f} m/s over {len(c)} full-speed "
                     f"throws{excl_text}; release direction spread {spread:.2f} deg")}


G_BASE_VEC = np.array([0.0, 0.0, -9.81])


def ingest_throws(mc, records, track_getter, na=0, ts=TS_DEFAULT):
    """
    Append real flights to the model exactly as the simulated loop does, and
    report what that changed.

    `track_getter(record)` returns `(points_base (N,3), times (N,))` -- the RAW
    RANSAC-inlier triangulated points for that throw. It is a callback so the
    caller owns file loading and this stays testable.

    Rotation augmentation (`na`) mirrors MC_PILOT.get_data_from_system, which
    applies the paper's Na augmentation to every trial it collects.

    Only throws executed at speed_scale == 1.0 may be ingested (amendment,
    2026-09-02, matching the rule `fit_release_model` above already
    implements). speed_scale is a time-stretch on the streamed joint speeds
    (`qd_cmd = qd * ds_dwall` in kinova_hardware.py), so a rehearsal at, say,
    0.15 releases the ball at ~0.15x the commanded speed --
    `track_to_state_samples` writes `commanded_speed` into
    `input_samples[0, 0]`, and for a rehearsal that number does not
    correspond to what the ball actually did. Feeding it in would teach the
    GP a false input->outcome mapping. A record with NO `speed_scale` key is
    treated as NON-QUALIFYING, the same rule `fit_release_model` uses --
    never default a missing field to "counts as data". Excluded throws are
    counted in `n_excluded_rehearsal` and named in `text` whenever nonzero,
    never silently dropped.
    """
    n_in = n_skip = n_excl = 0
    deviations = []
    for r in records:
        if r.get("landing_xy") is None:
            n_skip += 1
            continue

        scale = r.get("speed_scale")
        try:
            is_full_speed = (scale is not None
                             and abs(float(scale) - FULL_SPEED) < SPEED_SCALE_TOL)
        except (TypeError, ValueError):
            is_full_speed = False
        if not is_full_speed:
            n_excl += 1
            continue

        track = track_getter(r)
        if track is None:
            n_skip += 1
            continue
        pts, times = track
        states, inputs = track_to_state_samples(
            pts, times, r["target"], r["commanded_speed"], ts=ts)
        mc.model_learning.add_data(new_state_samples=states, new_input_samples=inputs)
        for _ in range(na):
            ang = np.random.uniform(0.0, 2.0 * np.pi)
            c, s = np.cos(ang), np.sin(ang)
            rot = states.copy()
            for sl in (slice(0, 2), slice(3, 5), slice(6, 8)):
                x, y = states[:, sl].T
                rot[:, sl] = np.stack([c * x - s * y, s * x + c * y], axis=1)
            mc.model_learning.add_data(new_state_samples=rot, new_input_samples=inputs)

        dv = np.diff(states[:, 3:6], axis=0)
        deviations.append(dv - np.tile(G_BASE_VEC * ts, (dv.shape[0], 1)))
        n_in += 1

    verdict = deviation_verdict(np.concatenate(deviations)) if deviations else \
        {"text": "no throws ingested", "above_noise": False, "n_samples": 0,
         "rms_deviation": 0.0, "rms_sigma": 0.0, "ratio": 0.0}
    try:
        release = fit_release_model(records)
    except ValueError as e:
        release = {"text": f"release model not fitted: {e}", "n": 0}

    excl_text = (f", excluded {n_excl} rehearsal throws at speed_scale < 1.0"
                if n_excl else "")
    return {"n_ingested": n_in, "n_skipped": n_skip, "n_excluded_rehearsal": n_excl,
            "verdict": verdict, "release_model": release,
            "text": (f"ingested {n_in} throws, skipped {n_skip}{excl_text}\n"
                     f"flight GP: {verdict['text']}\n"
                     f"release:   {release['text']}")}
