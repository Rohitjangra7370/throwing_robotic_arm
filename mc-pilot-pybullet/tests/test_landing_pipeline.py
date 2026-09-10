"""
Spec section 7.1 -- synthetic end-to-end. Renders a known parabola into two
IMAGES (not just observations), runs the whole pipeline, and checks the landing
point against ground truth.

This is the test that would catch a detector/pairing/fit integration error that
every unit test passes individually.
"""
import cv2
import numpy as np
import pytest

from measure_landing import build_observations, measure_landing
from perception.ball_track import median_background
from perception.ray_plane import D435I_IR_848x480
from perception.stereo import D435I_IR_BASELINE_M, StereoRig
from perception.trajectory import (BALL_RADIUS, Z_FLOOR_BASE,
                                   ballistic_position, solve_impact)

RIG = StereoRig(D435I_IR_848x480, D435I_IR_BASELINE_M)
R_BC = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
T_BC = np.array([0.82, 0.0, 1.767])
TRUE_P0 = np.array([0.035, 0.0, 1.137])
TRUE_V0 = np.array([1.6218, 0.0, 0.1419])
H, W = 480, 848


SHIFT = 4  # cv2 sub-pixel fixed point: 1/16 px


def _disc(img, u, v, r):
    """
    Filled disc at a SUB-PIXEL centre.

    An integer centre would quantise the ground truth to 0.5 px -- 3.3x the
    0.15 px centroid precision the error budget assumes -- so the test would be
    measuring cv2's rounding, not the pipeline.
    """
    k = 1 << SHIFT
    cv2.circle(img, (int(round(u * k)), int(round(v * k))),
               int(round(max(2.0, r) * k)), 225, -1, shift=SHIFT)


def _render(times, seed=0):
    """
    Draw the ball into both IR images at its true projected position.

    Frames where the disc would touch an image edge are SKIPPED. Before
    t ~= 0.120 s the ball is off the left edge of IR2 (u2 = -10.9 px at
    t = 0.10), and a clipped disc has a biased centroid -- which would look
    exactly like a pipeline error.
    """
    rng = np.random.default_rng(seed)
    base = (40 + rng.integers(0, 12, size=(H, W))).astype(np.uint8)
    ts, ir1, ir2 = [], [], []
    for t in times:
        p_c = R_BC.T @ (ballistic_position(TRUE_P0, TRUE_V0, t) - T_BC)
        u1, v1, u2, v2 = RIG.project(p_c)
        r_px = D435I_IR_848x480.fx * BALL_RADIUS / p_c[2]
        inside = all(r_px <= u <= W - 1 - r_px for u in (u1, u2)) and \
                 all(r_px <= v <= H - 1 - r_px for v in (v1, v2))
        if not inside:
            continue
        f1, f2 = base.copy(), base.copy()
        _disc(f1, u1, v1, r_px)
        _disc(f2, u2, v2, r_px)
        ts.append(t); ir1.append(f1); ir2.append(f2)
    if len(ts) < 20:
        raise AssertionError(f"only {len(ts)} frames rendered in view -- "
                             f"the test time window is wrong, not the pipeline")
    return {"t": np.asarray(ts, float), "ir1": np.asarray(ir1),
            "ir2": np.asarray(ir2), "meta": {"synthetic": True}}


# One real throw at 90 fps, from the frame the ball clears IR2's edge to impact.
FLIGHT_TIMES = np.arange(0.12, 0.58, 1.0 / 90.0)


def test_end_to_end_landing_point_matches_truth():
    rec = _render(FLIGHT_TIMES)
    got = measure_landing(rec, R_BC, T_BC)
    x_true, y_true, _ = solve_impact(TRUE_P0, TRUE_V0, z_floor=Z_FLOOR_BASE)
    err = np.hypot(got["x"] - x_true, got["y"] - y_true)
    assert err < 0.02, f"landing error {err * 1e3:.1f} mm"
    assert got["n_inliers"] >= 0.8 * len(rec["t"])
    assert got["rms_px"] < 1.0
    assert got["sigma_xy_m"] > 0


def test_observations_are_built_for_most_rendered_frames():
    rec = _render(FLIGHT_TIMES)
    obs, max_frac = build_observations(rec)
    assert max_frac < 0.01, "a synthetic scene must not look like lighting drift"
    assert obs.shape[1] == 5
    assert obs.shape[0] >= 0.9 * len(rec["t"]), \
        f"only {obs.shape[0]} of {len(rec['t'])} rendered frames paired"


def test_a_recording_with_no_ball_raises_rather_than_inventing_a_landing():
    rng = np.random.default_rng(1)
    flat = (40 + rng.integers(0, 12, size=(20, H, W))).astype(np.uint8)
    rec = {"t": np.linspace(0, 0.2, 20), "ir1": flat, "ir2": flat.copy(), "meta": {}}
    with pytest.raises(RuntimeError):
        measure_landing(rec, R_BC, T_BC)


def _add_static_noise(rec, u=460.0, v=234.0, r=6.0):
    """
    Overlay a disc at the same pixel location in every frame of both streams --
    a stand-in for the permanently-mounted calibration board (found
    2026-09-02): its candidates recur at a handful of fixed (u, v) locations
    across nearly the whole window, unlike the ball, which moves every frame.

    The caller must supply an EXTERNAL background (one built before this noise
    was added) to `build_observations`/`measure_landing` via `bg1`/`bg2`,
    rather than letting them compute their own per-pixel median from this
    noisy recording. A disc present in 100% of frames wins its own temporal
    median and vanishes into the "background" before detection ever runs --
    real board noise survives median subtraction only because the IR
    emitter's dot pattern gives it genuine frame-to-frame texture jitter,
    which a filled disc does not have. Supplying the pre-noise background
    isolates the thing this test actually checks (rejecting a recurring
    candidate) from a different, already-covered behavior (temporal-median
    robustness), rather than trying to out-render the real speckle noise.
    """
    ir1, ir2 = rec["ir1"].copy(), rec["ir2"].copy()
    for k in range(len(ir1)):
        _disc(ir1[k], u, v, r)
        _disc(ir2[k], u, v, r)
    return {**rec, "ir1": ir1, "ir2": ir2}


def test_static_board_noise_does_not_block_the_real_track():
    """
    Without static rejection, a persistent extra candidate in every frame
    dilutes the real ball's inlier fraction; with it (the default), the real
    track is still recovered cleanly. Matches the real-recording finding
    2026-09-02: the board was diluting a genuine ball track below RANSAC's
    60% inlier gate.
    """
    clean = _render(FLIGHT_TIMES)
    bg1, bg2 = median_background(clean["ir1"]), median_background(clean["ir2"])
    noisy = _add_static_noise(clean)
    got = measure_landing(noisy, R_BC, T_BC, bg1=bg1, bg2=bg2)   # reject_static=True by default
    x_true, y_true, _ = solve_impact(TRUE_P0, TRUE_V0, z_floor=Z_FLOOR_BASE)
    err = np.hypot(got["x"] - x_true, got["y"] - y_true)
    assert err < 0.02, f"landing error {err * 1e3:.1f} mm"


def test_disabling_static_rejection_lets_the_noise_back_in():
    """reject_static=False is an escape hatch for diagnosing the detector itself."""
    clean = _render(FLIGHT_TIMES)
    bg1, bg2 = median_background(clean["ir1"]), median_background(clean["ir2"])
    noisy = _add_static_noise(clean)
    obs_filtered, _ = build_observations(noisy, bg1=bg1, bg2=bg2)
    obs_raw, _ = build_observations(noisy, bg1=bg1, bg2=bg2, reject_static=False)
    assert obs_raw.shape[0] > obs_filtered.shape[0]


TRUE_SPEED = float(np.linalg.norm(TRUE_V0))


def test_commanded_speed_requires_release_t_offset():
    """
    Found 2026-09-02: p0/v0 are fit at the recording's local t=0, which for a
    RingBuffer-style capture is PRE_S seconds BEFORE release, not at release.
    Silently comparing raw v0 to a commanded release speed compares the wrong
    instant -- this must be a loud error, never an implicit "assume t=0 is
    release".
    """
    rec = _render(FLIGHT_TIMES)
    with pytest.raises(ValueError, match="release_t_offset"):
        measure_landing(rec, R_BC, T_BC, commanded_speed=TRUE_SPEED)


def test_commanded_speed_gate_accepts_a_matching_fit():
    # FLIGHT_TIMES already starts at the true release instant (TRUE_P0/TRUE_V0
    # are defined AT t=0), so this recording's local t=0 IS release: offset 0.
    rec = _render(FLIGHT_TIMES)
    got = measure_landing(rec, R_BC, T_BC, commanded_speed=TRUE_SPEED,
                          release_t_offset=0.0)
    assert got["x"] is not None


def test_commanded_speed_gate_refuses_a_fit_far_from_commanded():
    """
    An accurate fit against the WRONG commanded speed must still be refused --
    this is an outside-fact check (what the arm was actually told to do), not
    a fit-quality check (start_of_day.py's gates use the same principle: a
    perfect internal fit does not prove the answer is right).
    """
    rec = _render(FLIGHT_TIMES)
    with pytest.raises(RuntimeError, match="outside"):
        measure_landing(rec, R_BC, T_BC, commanded_speed=TRUE_SPEED * 10,
                        release_t_offset=0.0)


def test_release_t_offset_recovers_a_good_throw_pre_release_capture_window():
    """
    Regression for the actual 2026-09-02 incident: a RingBuffer-style
    recording whose local t=0 sits PRE_S seconds before release. Using the
    correct offset must accept a genuinely good fit; using 0 (the bug) must
    wrongly refuse it, because v0 at t=0 has picked up ~g*PRE_S of backward
    gravity extrapolation through a period the ball was not yet in flight.
    """
    pre_s = 0.45
    rec = _render(FLIGHT_TIMES)
    rec = {**rec, "t": rec["t"] + pre_s}   # shift: release is now at local t=pre_s

    got = measure_landing(rec, R_BC, T_BC, commanded_speed=TRUE_SPEED,
                          release_t_offset=pre_s)
    x_true, y_true, _ = solve_impact(TRUE_P0, TRUE_V0, z_floor=Z_FLOOR_BASE)
    assert np.hypot(got["x"] - x_true, got["y"] - y_true) < 0.02

    with pytest.raises(RuntimeError, match="outside"):
        measure_landing(rec, R_BC, T_BC, commanded_speed=TRUE_SPEED,
                        release_t_offset=0.0)
