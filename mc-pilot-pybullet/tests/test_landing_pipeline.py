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
