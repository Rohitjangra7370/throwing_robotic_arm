"""Tests for the pure decision + learning logic behind the hardware session."""
import numpy as np
import pytest

from hardware_learning import next_allowed_scale, propose_targets, scale_allowed

BAND = ((0.68, 0.74), (-0.25, 0.25))


def test_propose_targets_stays_inside_the_trained_band():
    t = propose_targets(10, BAND, seed=0)
    assert t.shape == (10, 2)
    assert np.all(t[:, 0] >= 0.68) and np.all(t[:, 0] <= 0.74)
    assert np.all(t[:, 1] >= -0.25) and np.all(t[:, 1] <= 0.25)


def test_propose_targets_actually_spreads():
    """10 near-identical throws teach the GP almost nothing -- that is the point."""
    t = propose_targets(10, BAND, seed=0)
    assert t[:, 1].max() - t[:, 1].min() > 0.30   # uses most of the y range
    assert t[:, 0].max() - t[:, 0].min() > 0.03   # and both ends of the narrow x range


def test_propose_targets_is_deterministic_for_a_seed():
    assert np.allclose(propose_targets(10, BAND, seed=7), propose_targets(10, BAND, seed=7))


def test_escalation_starts_at_the_bottom_of_the_ladder():
    assert next_allowed_scale([]) == pytest.approx(0.15)
    ok, why = scale_allowed(1.00, [])
    assert not ok and "0.15" in why


def test_escalation_advances_one_rung_per_clean_run():
    assert next_allowed_scale([0.15]) == pytest.approx(0.30)
    assert next_allowed_scale([0.15, 0.30]) == pytest.approx(0.60)
    assert next_allowed_scale([0.15, 0.30, 0.60]) == pytest.approx(1.00)


def test_escalation_refuses_skipping_a_rung():
    ok, why = scale_allowed(0.60, [0.15])
    assert not ok and "0.30" in why


def test_escalation_allows_repeating_or_dropping_back():
    assert scale_allowed(0.15, [0.15, 0.30])[0]
    assert scale_allowed(0.30, [0.15, 0.30])[0]


def test_track_to_state_samples_has_the_exact_shapes_the_model_expects():
    """(n, 8) = [x,y,z,vx,vy,vz,Px,Py] and (n, 1) with the speed only at t=0 --
    verified against PyBulletThrowingSystem.rollout, not assumed."""
    from hardware_learning import track_to_state_samples
    t = np.arange(0.0, 0.50, 1 / 90.0)
    p0, v0, g = np.array([0.3, 0.0, 0.02]), np.array([1.39, 0.0, 0.37]), np.array([0, 0, -9.81])
    pts = p0 + np.outer(t, v0) + 0.5 * np.outer(t ** 2, g)
    s, u = track_to_state_samples(pts, t, (0.71, 0.02), 1.44, ts=0.02)
    assert s.shape[1] == 8 and u.shape[1] == 1
    assert s.shape[0] == u.shape[0]
    assert u[0, 0] == pytest.approx(1.44)
    assert np.allclose(u[1:, 0], 0.0)
    assert np.allclose(s[:, 6], 0.71) and np.allclose(s[:, 7], 0.02)


def test_track_to_state_samples_recovers_a_known_velocity_profile():
    from hardware_learning import track_to_state_samples
    t = np.arange(0.0, 0.50, 1 / 90.0)
    p0, v0, g = np.array([0.3, 0.0, 0.02]), np.array([1.39, 0.0, 0.37]), np.array([0, 0, -9.81])
    pts = p0 + np.outer(t, v0) + 0.5 * np.outer(t ** 2, g)
    s, _ = track_to_state_samples(pts, t, (0.71, 0.02), 1.44, ts=0.02)
    assert np.allclose(s[0, 0:3], p0, atol=2e-3)
    assert np.allclose(s[0, 3:6], v0, atol=2e-2)
    dt = 0.02
    dv = (s[1:, 3:6] - s[:-1, 3:6]) / dt
    assert np.allclose(dv[:, 2].mean(), -9.81, atol=0.5)


def test_track_to_state_samples_is_sampled_at_ts_not_at_camera_rate():
    """90 fps in, 50 Hz out -- the GP's propagation assumes Ts spacing."""
    from hardware_learning import track_to_state_samples
    t = np.arange(0.0, 0.50, 1 / 90.0)
    pts = np.stack([t * 1.4, t * 0, 0.02 - 4.9 * t ** 2], axis=1)
    s, _ = track_to_state_samples(pts, t, (0.71, 0.0), 1.44, ts=0.02)
    assert 24 <= s.shape[0] <= 26        # 0.50 s / 0.02 s


def test_track_to_state_samples_rejects_a_track_too_short_to_difference():
    from hardware_learning import track_to_state_samples
    t = np.array([0.0, 0.01])
    pts = np.zeros((2, 3))
    with pytest.raises(ValueError, match="too short"):
        track_to_state_samples(pts, t, (0.71, 0.0), 1.44, ts=0.02)


def test_track_to_state_samples_rejects_non_monotonic_times():
    """np.interp silently produces garbage for unordered xp (measured: 3.8 cm
    position and 0.65 m/s velocity error from two swapped samples). This guard
    prevents silent corruption when a caller merges tracks or reorders."""
    from hardware_learning import track_to_state_samples
    t = np.arange(0.0, 0.50, 1 / 90.0)
    p0, v0, g = np.array([0.3, 0.0, 0.02]), np.array([1.39, 0.0, 0.37]), np.array([0, 0, -9.81])
    pts = p0 + np.outer(t, v0) + 0.5 * np.outer(t ** 2, g)
    # Swap two interior samples to break monotonicity
    t_bad = t.copy()
    t_bad[5], t_bad[20] = t_bad[20], t_bad[5]
    pts_bad = pts.copy()
    pts_bad[5], pts_bad[20] = pts_bad[20], pts_bad[5]
    with pytest.raises(ValueError, match="non-decreasing"):
        track_to_state_samples(pts_bad, t_bad, (0.71, 0.0), 1.44, ts=0.02)


def test_velocity_noise_sigma_uses_independent_noise_only():
    """Only per-frame stereo noise survives differencing; extrinsic translation
    (systematic) error cancels in p_{k+1} - p_{k-1}. The 10 mm independent default
    yields ~0.35 m/s, not the 0.73 m/s from the full ~20 mm when wrongly combined."""
    from hardware_learning import velocity_noise_sigma, POS_SIGMA_INDEPENDENT_M
    s = velocity_noise_sigma(pos_sigma_m=POS_SIGMA_INDEPENDENT_M, ts=0.02)
    assert 0.30 < s < 0.40          # ~0.35 m/s from 10 mm independent noise
    assert velocity_noise_sigma(POS_SIGMA_INDEPENDENT_M, 0.04) < s  # longer baseline, less noise


def test_verdict_is_below_noise_when_the_signal_is_smaller_than_sigma():
    """The expected real-world answer for a tennis ball: drag ~5mm, noise too large."""
    from hardware_learning import deviation_verdict
    v = deviation_verdict(dv_learned=np.full((40, 3), 0.01), sigma_v=0.5)
    assert not v["above_noise"]
    assert "BELOW NOISE" in v["text"]
    assert v["ratio"] < 1.0


def test_verdict_ensemble_threshold_re_pinned_to_se():
    """Boundary test re-pinned to ensemble SE, not per-sample RMS. With n=40,
    sigma_v=0.5, k=2.0: SE = 0.0791, k*SE = 0.1581. Systematic floor is ~0.0019.
    Mean deviation 0.15 is below k*SE; 0.17 is above. Both are way above the
    systematic floor, so the second condition is not the limiting one here."""
    from hardware_learning import deviation_verdict
    just_under = deviation_verdict(np.full((40, 1), 0.15), sigma_v=0.5, k=2.0)
    just_over = deviation_verdict(np.full((40, 1), 0.17), sigma_v=0.5, k=2.0)
    assert not just_under["above_noise"]
    assert just_over["above_noise"]
    assert "ABOVE NOISE" in just_over["text"]


def test_verdict_includes_both_conditions_and_names_failure():
    """RE-PINNED to verify the AND gate is actually checked. Exceeding k*SE is
    not enough if the result could be aliased extrinsic rotation. This test
    creates a case where above_se=True but above_sys=False, so removing the
    systematic-floor condition would flip the verdict. Previous version used
    dv=0.002, sigma_v=0.5, n=40 which gave above_se=False, above_sys=True
    (opposite of the docstring claim), so it passed for the wrong reason.

    Correct case: dv_learned=0.0015, sigma_v=0.001, n=250, k=2.0 yields:
    - k*SE = 0.000126 (cleared by 0.0015)
    - systematic_floor = 0.0019176 (NOT cleared by 0.0015)
    - Verdict: above_noise=False because the systematic floor fails."""
    from hardware_learning import deviation_verdict, systematic_dv_floor
    sys_floor = systematic_dv_floor()
    assert pytest.approx(sys_floor, abs=1e-6) == 0.001918

    # Case: mean_dev clears SE threshold but NOT systematic floor
    above_se_below_sys = deviation_verdict(np.full((250, 1), 0.0015), sigma_v=0.001, k=2.0)
    assert not above_se_below_sys["above_noise"], "Systematic floor not cleared; should be below-noise"
    assert above_se_below_sys["mean_deviation"] > 2.0 * above_se_below_sys["standard_error"], \
        "SE threshold cleared but systematic floor is not"
    assert "systematic floor" in above_se_below_sys["text"].lower()
    assert "calibration" in above_se_below_sys["text"].lower()
    # When systematic floor is the blocker, no n_required should appear; this assertion
    # catches regressions that re-add sample-count guidance in the floor-is-blocker branch
    assert "would require" not in above_se_below_sys["text"].lower(), \
        "n_required text should not appear when systematic floor is the blocker"


def test_verdict_k_zero_does_not_raise():
    """Zero guard must protect the denominator (k * sigma_v), not sigma_v alone."""
    from hardware_learning import deviation_verdict
    v = deviation_verdict(np.full((40, 1), 0.01), sigma_v=0.5, k=0.0)
    assert v["ratio"] == 0.0


def test_verdict_below_noise_text_includes_required_sample_count():
    """When the verdict is BELOW NOISE, the text must state the sample count
    needed to resolve this effect above the noise threshold -- so the reader
    learns whether the experiment was underpowered or the effect is absent."""
    from hardware_learning import deviation_verdict
    v = deviation_verdict(np.full((37, 3), 0.02), sigma_v=0.5)
    assert not v["above_noise"]
    assert "37" in v["text"]
    assert "sample" in v["text"].lower()
    # The text must include n_required (a large number for small mean deviations)
    # We can verify this exists by checking that a number > 1000 appears
    import re
    numbers = re.findall(r'\d+', v["text"])
    assert any(int(n) > 1000 for n in numbers)


def test_fit_release_model_recovers_a_known_gain_and_offset():
    # speed_scale=1.0 on every record -- Defect 1 (2026-09-02): fit_release_model
    # now excludes anything not at full speed, and a missing key does not
    # default to qualifying. See test_fit_release_model_excludes_rehearsal_throws
    # for the mixed-scale case this file did not previously cover.
    from hardware_learning import fit_release_model
    rng = np.random.default_rng(0)
    recs = []
    for c in np.linspace(1.2, 1.6, 12):
        actual = 0.90 * c + 0.05
        recs.append({"commanded_speed": float(c),
                     "measured_v0": [float(actual), 0.0, 0.0],
                     "landing_xy": [0.7, 0.0], "speed_scale": 1.0})
    out = fit_release_model(recs)
    assert out["gain"] == pytest.approx(0.90, abs=1e-6)
    assert out["offset"] == pytest.approx(0.05, abs=1e-6)
    assert out["n"] == 12
    assert out["n_excluded_rehearsal"] == 0


def test_fit_release_model_skips_refused_throws():
    """A throw with no measurement carries no release information.
    speed_scale=1.0 added to the measured records -- see the comment on
    test_fit_release_model_recovers_a_known_gain_and_offset."""
    from hardware_learning import fit_release_model
    recs = [{"commanded_speed": 1.4, "measured_v0": [1.31, 0, 0], "landing_xy": [0.7, 0], "speed_scale": 1.0},
            {"commanded_speed": 1.5, "measured_v0": None, "landing_xy": None},
            {"commanded_speed": 1.6, "measured_v0": [1.49, 0, 0], "landing_xy": [0.7, 0], "speed_scale": 1.0},
            {"commanded_speed": 1.5, "measured_v0": [1.40, 0, 0], "landing_xy": [0.7, 0], "speed_scale": 1.0}]
    assert fit_release_model(recs)["n"] == 3


def test_fit_release_model_refuses_to_fit_too_few_points():
    """Fewer than 3 measured throws at all -- the generic message, no mention
    of rehearsals (that is a different failure, see
    test_fit_release_model_all_rehearsals_raises_a_distinct_error)."""
    from hardware_learning import fit_release_model
    recs = [{"commanded_speed": 1.4, "measured_v0": [1.3, 0, 0], "landing_xy": [0.7, 0], "speed_scale": 1.0}]
    with pytest.raises(ValueError, match="at least 3") as exc_info:
        fit_release_model(recs)
    assert "rehearsal" not in str(exc_info.value).lower()


def test_fit_release_model_excludes_rehearsal_throws():
    """Defect 1 (2026-09-02): only speed_scale==1.0 throws feed the fit. Mix
    3 rehearsal-speed throws (a plausible fit if wrongly included -- they sit
    right on the same line) with 4 full-speed ones; the fit must recover the
    full-speed gain/offset exactly and report the exclusion."""
    from hardware_learning import fit_release_model
    recs = []
    for c, scale in [(1.2, 0.15), (1.4, 0.30), (1.6, 0.60)]:
        # Rehearsal: measured value follows a DIFFERENT (also linear) relation,
        # so if these leak into the fit the recovered gain/offset will be wrong.
        recs.append({"commanded_speed": c, "measured_v0": [0.15 * c, 0.0, 0.0],
                     "landing_xy": [0.1, 0.0], "speed_scale": scale})
    for c in np.linspace(1.2, 1.8, 4):
        actual = 0.90 * c + 0.05
        recs.append({"commanded_speed": float(c), "measured_v0": [float(actual), 0.0, 0.0],
                     "landing_xy": [0.7, 0.0], "speed_scale": 1.0})
    out = fit_release_model(recs)
    assert out["n"] == 4
    assert out["n_excluded_rehearsal"] == 3
    assert out["gain"] == pytest.approx(0.90, abs=1e-6)
    assert out["offset"] == pytest.approx(0.05, abs=1e-6)
    assert "excluded 3 rehearsal throws" in out["text"]
    assert "speed_scale < 1.0" in out["text"]


def test_fit_release_model_all_rehearsals_raises_a_distinct_error():
    """3 measured throws logged, but all rehearsals -- this needs a different
    operator action (throw at full speed) than "fewer than 3 measured at
    all" (throw more, period), so the message must say so."""
    from hardware_learning import fit_release_model
    recs = [{"commanded_speed": c, "measured_v0": [0.15 * c, 0.0, 0.0],
            "landing_xy": [0.1, 0.0], "speed_scale": 0.15}
           for c in (1.2, 1.4, 1.6)]
    with pytest.raises(ValueError) as exc_info:
        fit_release_model(recs)
    msg = str(exc_info.value).lower()
    assert "3 measured" in msg or "3 measured throws" in msg
    assert "rehearsal" in msg
    assert "full speed" in msg


def test_fit_release_model_missing_speed_scale_key_is_non_qualifying():
    """A record with no speed_scale key at all must NOT be silently assumed
    to be full-speed data -- it is excluded exactly like an explicit < 1.0."""
    from hardware_learning import fit_release_model
    recs = [{"commanded_speed": c, "measured_v0": [0.9 * c, 0.0, 0.0], "landing_xy": [0.7, 0.0]}
           for c in (1.2, 1.4, 1.6)]   # no speed_scale key anywhere
    with pytest.raises(ValueError) as exc_info:
        fit_release_model(recs)
    assert "rehearsal" in str(exc_info.value).lower()


def test_pure_parabola_teaches_the_gp_nothing():
    """
    THE TAUTOLOGY GUARD. A gravity-only track must produce a learned deviation
    that the verdict calls BELOW NOISE. If someone resamples fit_ballistic's
    output into add_data, this is what should catch it.
    """
    from hardware_learning import deviation_verdict, track_to_state_samples
    t = np.arange(0.0, 0.50, 1 / 90.0)
    p0, v0, g = np.array([0.3, 0.0, 0.02]), np.array([1.39, 0.0, 0.37]), np.array([0, 0, -9.81])
    pts = p0 + np.outer(t, v0) + 0.5 * np.outer(t ** 2, g)
    s, _ = track_to_state_samples(pts, t, (0.71, 0.0), 1.44)

    dv = np.diff(s[:, 3:6], axis=0)
    dv_gravity = np.tile(g * 0.02, (dv.shape[0], 1))
    verdict = deviation_verdict(dv - dv_gravity)
    assert not verdict["above_noise"], (
        "a pure-gravity track produced a deviation the verdict called real -- "
        "the guard against feeding the fitted parabola back has broken")


def test_injected_drag_is_recovered_when_noise_is_set_below_it():
    """The verdict must also be able to say yes, or it is not a test."""
    from hardware_learning import deviation_verdict
    dv = np.full((40, 3), 0.05)
    assert deviation_verdict(dv, sigma_v=0.001)["above_noise"]
    assert not deviation_verdict(dv, sigma_v=1.0)["above_noise"]


def test_fit_release_model_skips_malformed_measured_v0():
    """Zero-magnitude, non-finite, empty, non-3-vector measured_v0 must be
    skipped cleanly. A single bad record can corrupt the fit (observed: [0,0,0]
    among two good records produced gain=-6.5, offset=+10.65 — physically
    impossible, returned with no error). Guards tightly against silent corruption.
    speed_scale=1.0 added throughout -- see the comment on
    test_fit_release_model_recovers_a_known_gain_and_offset."""
    from hardware_learning import fit_release_model
    recs = [
        {"commanded_speed": 1.4, "measured_v0": [1.26, 0, 0], "landing_xy": [0.7, 0], "speed_scale": 1.0},
        {"commanded_speed": 1.5, "measured_v0": [0.0, 0.0, 0.0], "landing_xy": [0.7, 0], "speed_scale": 1.0},  # zero vector — skip
        {"commanded_speed": 1.6, "measured_v0": [1.44, 0, 0], "landing_xy": [0.7, 0], "speed_scale": 1.0},
        {"commanded_speed": 1.7, "measured_v0": [float('nan'), 0, 0], "landing_xy": [0.7, 0], "speed_scale": 1.0},  # non-finite — skip
        {"commanded_speed": 1.8, "measured_v0": [1.62, 0, 0], "landing_xy": [0.7, 0], "speed_scale": 1.0},
        {"commanded_speed": 1.9, "measured_v0": [], "landing_xy": [0.7, 0], "speed_scale": 1.0},  # empty — skip
        {"commanded_speed": 1.95, "measured_v0": 5.0, "landing_xy": [0.7, 0], "speed_scale": 1.0},  # scalar, not 3-vector — skip
        {"commanded_speed": 2.0, "measured_v0": [1.80, 0, 0], "landing_xy": [0.7, 0], "speed_scale": 1.0}
    ]
    result = fit_release_model(recs)
    # Should have skipped zero, NaN, empty, and scalar records; left with 4 good ones
    assert result["n"] == 4
    # With a linear relationship (measured ≈ 0.9*commanded), gain should be close to 0.9
    assert 0.85 < result["gain"] < 0.95


def test_fit_release_model_direction_spread_is_maximum_pairwise_angle():
    """Release direction spread must be the maximum pairwise angle, not max
    deviation from mean. The old code computed max deviation from a vector mean
    that often cancels to near-zero; normalizing that unstable vector produced
    spurious results (on this exact input: ~180° from opposite-side residual).
    This test must enforce the correct 120° ± 1°, not just > 100° which would
    pass the old broken code's spurious 180° output. speed_scale=1.0 added --
    see the comment on test_fit_release_model_recovers_a_known_gain_and_offset."""
    from hardware_learning import fit_release_model
    # Three release directions 120° apart in a plane: should report ~120°
    recs = [
        {"commanded_speed": 1.0, "measured_v0": [1.0, 0.0, 0.0], "landing_xy": [0.7, 0], "speed_scale": 1.0},
        {"commanded_speed": 1.1, "measured_v0": [-0.5, 0.866, 0.0], "landing_xy": [0.7, 0], "speed_scale": 1.0},  # 120° from first
        {"commanded_speed": 1.2, "measured_v0": [-0.5, -0.866, 0.0], "landing_xy": [0.7, 0], "speed_scale": 1.0},  # 120° from first, 120° from second
    ]
    result = fit_release_model(recs)
    # Must be within 1° of 120°. A looser bound like > 100° would pass the old
    # code's spurious ~180° output, making the test a false guard.
    assert abs(result["direction_error_deg"] - 120.0) < 1.0


def test_production_path_can_report_above_noise():
    """The real pipeline (track_to_state_samples → deviation_verdict at defaults)
    must be capable of returning ABOVE NOISE, not just negative verdicts. This test
    deliberately injects a large, obviously super-threshold extra acceleration to
    prove the instrument works in the positive direction. Real tennis-ball drag is
    provably below the noise floor at default settings; this test does not claim
    otherwise — only that the verdict can return True."""
    from hardware_learning import track_to_state_samples, deviation_verdict
    t = np.arange(0.0, 0.50, 1 / 90.0)
    p0 = np.array([0.3, 0.0, 0.02])
    v0 = np.array([1.39, 0.0, 0.37])
    g = np.array([0, 0, -9.81])
    # Extra acceleration: 10.0 m/s² is obviously super-threshold. With Ts=0.02,
    # this produces Δv = 0.2 m/s per step, ~3000× the drag signal. The noise floor
    # from stereo is ~0.22 m/s; this is way above it. With ~24 samples, this
    # clears both the random-noise threshold (2*SE) and systematic floor.
    a_extra = np.array([0.0, 0.0, 10.0])
    pts = p0 + np.outer(t, v0) + 0.5 * np.outer(t ** 2, (g + a_extra))

    s, _ = track_to_state_samples(pts, t, (0.71, 0.0), 1.44)
    dv = np.diff(s[:, 3:6], axis=0)
    dv_gravity = np.tile(g * 0.02, (dv.shape[0], 1))
    verdict = deviation_verdict(dv - dv_gravity)

    assert verdict["above_noise"], (
        "the production path with deliberately super-threshold extra acceleration "
        "failed to report ABOVE NOISE — the verdict is broken")
