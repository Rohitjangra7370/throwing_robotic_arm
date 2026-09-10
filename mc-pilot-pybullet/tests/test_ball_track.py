"""
Ball detection on synthetic images. No camera.

Synthetic rather than recorded because the quantity being validated is centroid
BIAS -- a systematic that a real recording cannot bound, since it has no ground
truth. A drawn disc has one exactly.
"""
import cv2
import numpy as np
import pytest

from perception.ball_track import (Candidate, detect_candidates,
                                   frame_diagnostics, median_background,
                                   reject_static_candidates)

H, W = 480, 848


def _bg(level=40):
    """Textured background -- a flat one would make detection unrealistically easy."""
    rng = np.random.default_rng(0)
    return (level + rng.integers(0, 12, size=(H, W))).astype(np.uint8)


def _with_ball(bg, u, v, r=7.0, level=220):
    img = bg.copy()
    cv2.circle(img, (int(round(u)), int(round(v))), int(round(r)), int(level), -1)
    return img


def test_median_background_ignores_a_ball_that_moves():
    bg = _bg()
    frames = [_with_ball(bg, 100 + 40 * i, 200) for i in range(9)]
    got = median_background(frames)
    assert np.abs(got.astype(int) - bg.astype(int)).max() <= 1


def test_detects_a_single_ball_with_subpixel_accuracy():
    bg = _bg()
    frame = _with_ball(bg, 423.0, 217.0, r=7.0)
    cands = detect_candidates(frame, bg)
    assert len(cands) == 1
    c = cands[0]
    assert c.u == pytest.approx(423.0, abs=0.3)
    assert c.v == pytest.approx(217.0, abs=0.3)
    assert c.radius_px == pytest.approx(7.0, rel=0.15)
    assert c.circularity > 0.7


def test_returns_empty_when_nothing_changed():
    """No ball must mean no candidates -- never a fabricated one."""
    bg = _bg()
    assert detect_candidates(bg.copy(), bg) == []


def test_rejects_blobs_outside_the_size_gate():
    bg = _bg()
    tiny = _with_ball(bg, 400.0, 200.0, r=1.0)
    huge = _with_ball(bg, 400.0, 200.0, r=60.0)
    assert detect_candidates(tiny, bg) == []
    assert detect_candidates(huge, bg) == []


def test_finds_both_balls_when_two_are_present():
    """Detection does not decide which blob is the ball; that is RANSAC's job."""
    bg = _bg()
    frame = _with_ball(_with_ball(bg, 200.0, 150.0), 600.0, 300.0)
    assert len(detect_candidates(frame, bg)) == 2


def test_frame_diagnostics_flags_lighting_drift():
    """
    Spec section 6 wants the one number that catches a flooded frame -- the same
    role mask_nonzero_frac plays in detect_ball_bgsub. A ball is a few tenths of
    a percent of the frame; a lighting shift or a bumped camera is tens of
    percent, and would otherwise surface as a silent detection failure.
    """
    bg = _bg()
    quiet = frame_diagnostics(_with_ball(bg, 423.0, 217.0), bg)
    assert quiet["mask_nonzero_frac"] < 0.01
    flooded = frame_diagnostics(np.full_like(bg, 200), bg)
    assert flooded["mask_nonzero_frac"] > 0.9


def test_candidate_is_a_plain_tuple_of_floats_for_pairing():
    """pair_candidates() consumes (u, v, area_px); keep them interoperable."""
    bg = _bg()
    c = detect_candidates(_with_ball(bg, 423.0, 217.0), bg)[0]
    assert isinstance(c.as_uv_area(), tuple)
    assert len(c.as_uv_area()) == 3
    assert all(isinstance(x, float) for x in c.as_uv_area())


def _cand(u, v, area=100.0):
    return Candidate(u=u, v=v, area_px=area, radius_px=7.0, circularity=0.9)


def test_reject_static_candidates_drops_an_unmoving_blob():
    """
    A permanently-mounted board (found 2026-09-02, real recordings) produces a
    candidate at ~the same (u, v) every frame. A real ball's (u, v) changes
    every frame. The filter must tell these apart using only recurrence, with
    no knowledge of which one is "the board".
    """
    n = 20
    per_frame = []
    for k in range(n):
        static = _cand(423.0 + 0.3 * (k % 2), 217.0)   # board: jitters <1px, every frame
        moving = _cand(100.0 + 15.0 * k, 200.0)          # ball: moves 15px/frame
        per_frame.append([static, moving])

    out = reject_static_candidates(per_frame)
    for k, cands in enumerate(out):
        assert len(cands) == 1, f"frame {k}: expected only the moving candidate to survive"
        assert cands[0].u == pytest.approx(100.0 + 15.0 * k)


def test_reject_static_candidates_is_a_noop_on_a_ball_only_recording():
    """The filter must not eat a real track just because it is the only thing present."""
    per_frame = [[_cand(100.0 + 15.0 * k, 200.0)] for k in range(20)]
    out = reject_static_candidates(per_frame)
    assert sum(len(c) for c in out) == 20


def test_reject_static_candidates_ignores_a_brief_coincidence():
    """
    Two different moving objects passing through the same bin on different
    frames must not be flagged -- persistence is about ONE bin recurring
    across many frames, not about how many candidates a bin sees in total.
    """
    per_frame = [[_cand(100.0 + 15.0 * k, 200.0)] for k in range(20)]
    per_frame[5].append(_cand(250.0, 300.0))   # unrelated, single-frame coincidence
    out = reject_static_candidates(per_frame)
    assert sum(len(c) for c in out) == 21
