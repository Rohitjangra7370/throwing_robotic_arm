"""
Ball detector tests on synthetic frames -- pure image logic, no camera.

Runs before any real capture is trusted, same reasoning as test_ray_plane.py:
this is code every real landing measurement flows through.
"""
import numpy as np
import pytest

from perception.ball_detector import detect_ball_bgsub, detect_ball_hsv


def _blank(h=200, w=200, val=180):
    """Uniform light-tan-ish frame, standing in for the tile floor."""
    return np.full((h, w, 3), val, dtype=np.uint8)


def _with_ball(frame, cx, cy, r=8, val=250):
    out = frame.copy()
    yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    out[mask] = val
    return out


def test_bgsub_finds_a_ball_that_appeared():
    before = _blank()
    after = _with_ball(before, cx=120, cy=70, r=8)
    u, v, diag = detect_ball_bgsub(before, after)
    assert u == pytest.approx(120, abs=1.0)
    assert v == pytest.approx(70, abs=1.0)
    assert diag["n_candidates"] == 1
    assert diag["mask_nonzero_frac"] < 0.05


def test_bgsub_ignores_a_small_appearing_speck_below_min_area():
    before = _blank()
    after = _with_ball(before, cx=50, cy=50, r=1)  # ~3px area, below default min
    with pytest.raises(RuntimeError, match="no changed region"):
        detect_ball_bgsub(before, after, min_area_px=15)


def test_bgsub_raises_on_no_change_rather_than_fabricating_a_centroid():
    before = _blank()
    after = _blank()
    with pytest.raises(RuntimeError, match="no changed region"):
        detect_ball_bgsub(before, after)


def test_bgsub_flags_lighting_drift_as_such_not_a_silent_wrong_answer():
    before = _blank(val=150)
    after = _blank(val=220)   # whole frame brighter -- exposure drift, not a ball
    with pytest.raises(RuntimeError, match="lighting drift"):
        detect_ball_bgsub(before, after)


def test_bgsub_picks_the_larger_of_two_changed_regions():
    before = _blank()
    after = _with_ball(before, cx=30, cy=30, r=3)     # small spurious change
    after = _with_ball(after, cx=150, cy=100, r=10)   # the real ball
    u, v, diag = detect_ball_bgsub(before, after, min_area_px=5)
    assert u == pytest.approx(150, abs=1.0)
    assert v == pytest.approx(100, abs=1.0)
    assert diag["n_candidates"] == 2


def test_bgsub_rejects_shape_mismatch():
    before = _blank(200, 200)
    after = _blank(100, 100)
    with pytest.raises(ValueError, match="shape mismatch"):
        detect_ball_bgsub(before, after)


def test_hsv_finds_a_coloured_blob():
    frame = _blank(val=180)
    # paint a green blob (BGR) for HSV thresholding
    yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
    mask = (xx - 60) ** 2 + (yy - 90) ** 2 <= 9 * 9
    frame = frame.copy()
    frame[mask] = (0, 200, 0)  # BGR green
    u, v, diag = detect_ball_hsv(frame, hsv_lower=(45, 70, 70), hsv_upper=(90, 255, 255))
    assert u == pytest.approx(60, abs=1.0)
    assert v == pytest.approx(90, abs=1.0)


def test_hsv_raises_when_nothing_matches():
    frame = _blank()
    with pytest.raises(RuntimeError, match="no HSV blob"):
        detect_ball_hsv(frame, hsv_lower=(45, 70, 70), hsv_upper=(90, 255, 255))
