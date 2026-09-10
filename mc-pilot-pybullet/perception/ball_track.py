"""
Find the ball in one IR image by differencing against a static background.

WHY NOT YOLO
------------
The ball spans ~12-20 px in the 848x480 IR stream, which sits at the floor of
YOLOv8's stride-8 P3 head. The camera is static and the ball is high contrast
against a differenced background, so a threshold plus a contour finds it in
~1 ms of CPU with no model, no training set and no GPU -- which also matters
because `torch.cuda` is currently unusable on this machine. A YOLO backend
behind this same interface is a reasonable fallback; it is not the primary.

WHY THIS RETURNS EVERY CANDIDATE, NOT "THE BALL"
------------------------------------------------
The arm moves fast in exactly the region the ball starts in, and it is the main
false-positive source. This module deliberately does NOT try to pick the ball
out -- no hand-drawn ROI mask, no "largest blob" heuristic. It reports every
plausible blob and lets the RANSAC ballistic association in `trajectory.py`
reject the arm for free: arm pixels do not fit a parabola with g = 9.81. That
also handles reflections and the second bounce, which no mask would.

Contrast this with `ball_detector.detect_ball_bgsub`, which DOES pick a single
largest blob -- correct there, because it looks at a settled scene with one
changed object, and there is no trajectory available to arbitrate.

SUBPIXEL CENTROID
-----------------
Intensity-weighted over the difference image, not the binary mask. For a
symmetric motion blur the intensity centroid is unbiased at MID-EXPOSURE, which
is why `ir_capture.py` timestamps frames at mid-exposure rather than at arrival.
Getting that pairing wrong is a systematic: at 5.8 m/s an 8.5 ms exposure is
5 cm of travel.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["Candidate", "median_background", "detect_candidates",
           "frame_diagnostics", "reject_static_candidates"]


@dataclass(frozen=True)
class Candidate:
    """One plausible ball blob in one image."""
    u: float
    v: float
    area_px: float
    radius_px: float
    circularity: float

    def as_uv_area(self):
        """The triple `stereo.pair_candidates` consumes."""
        return (float(self.u), float(self.v), float(self.area_px))


def median_background(frames):
    """
    Per-pixel median over a stack of frames.

    A ball occupies any given pixel for at most a couple of frames out of the
    ~44 in a throw window, so the temporal median IS the empty scene -- no
    separate background capture, and no drift between a "before" shot and the
    throw. This is strictly more robust than the two-shot approach in
    `ball_detector.py`, which is vulnerable to auto-exposure resettling between
    the two captures.
    """
    stack = np.asarray(frames)
    if stack.ndim != 3:
        raise ValueError(f"expected a stack of 2-D frames, got shape {stack.shape}")
    if stack.shape[0] < 3:
        raise ValueError(f"need >= 3 frames for a median background, got {stack.shape[0]}")
    return np.median(stack, axis=0).astype(np.uint8)


def frame_diagnostics(frame, background, diff_thresh=18):
    """
    The one number that says whether this frame is interpretable at all.

    `mask_nonzero_frac` is the fraction of pixels that changed. A ball is a few
    tenths of a percent. Tens of percent means the lighting shifted, the camera
    was bumped, or auto-exposure resettled -- in which case the detector will
    either find nothing or find garbage, and either way the answer is to fix the
    capture, not to loosen a threshold. Same role, same name, as
    `ball_detector.detect_ball_bgsub`'s diagnostic.
    """
    diff = cv2.absdiff(frame, background)
    mask = diff >= diff_thresh
    return {"mask_nonzero_frac": float(np.count_nonzero(mask)) / mask.size,
            "diff_max": int(diff.max()), "diff_mean": float(diff.mean())}


def detect_candidates(frame, background, diff_thresh=18, min_area_px=20,
                      max_area_px=2000, min_circularity=0.30, morph_kernel=3):
    """
    One image + its background -> every plausible ball blob.

    Parameters
    ----------
    diff_thresh : int
        Absolute per-pixel difference counting as "changed".
    min_area_px, max_area_px : int
        The ball spans ~12-20 px across at 1.5-2.4 m, i.e. ~113-314 px of area.
        The window is deliberately wider than that: motion blur elongates the
        blob, so the gate rejects sensor speckle and whole-frame lighting drift,
        not marginal balls.
    min_circularity : float
        4*pi*area/perimeter^2. LOOSE on purpose -- a blurred ball is an ellipse,
        not a disc, and tightening this is how you silently drop the fastest
        (most informative) frames. It only exists to reject long thin edges.

    Returns a list of Candidate, possibly empty. Returning [] is a valid,
    meaningful answer -- it does NOT raise, because "no ball in this frame" is
    expected for every frame before the ball enters view.
    """
    if frame.shape != background.shape:
        raise ValueError(f"frame shape {frame.shape} != background shape "
                         f"{background.shape} -- the camera must not have moved")

    diff = cv2.absdiff(frame, background)
    mask = (diff >= diff_thresh).astype(np.uint8) * 255
    kernel = np.ones((morph_kernel, morph_kernel), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if not (min_area_px <= area <= max_area_px):
            continue
        perim = float(cv2.arcLength(c, True))
        if perim <= 0:
            continue
        circ = 4.0 * np.pi * area / (perim * perim)
        if circ < min_circularity:
            continue

        blob = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(blob, [c], -1, 255, -1)
        w = diff.astype(np.float64) * (blob > 0)
        total = w.sum()
        if total <= 0:
            continue
        ys, xs = np.nonzero(blob)
        u = float((w[ys, xs] * xs).sum() / total)
        v = float((w[ys, xs] * ys).sum() / total)
        out.append(Candidate(u=u, v=v, area_px=area,
                             radius_px=float(np.sqrt(area / np.pi)),
                             circularity=float(circ)))
    return out


def reject_static_candidates(per_frame, bin_px=6.0, persistence_frac=0.12,
                             min_persistent_frames=3, dilate=1):
    """
    Drop candidates that recur at nearly the same pixel location across many
    frames of ONE recording -- found 2026-09-02: a permanently-mounted ChArUco
    calibration board sitting in the overhead camera's fixed field of view
    (glued to the floor, not something that can be removed between throws)
    beats against the IR emitter's dot pattern and produces persistent diff
    noise at its corners every single frame, which `detect_candidates` reads
    as ball-like blobs. A real flying ball's (u, v) changes every frame; this
    board noise does not.

    This is deliberately NOT a hand-drawn ROI mask (the thing this module's
    own docstring says not to do for the arm) -- it is computed fresh from
    THIS recording's own candidates, so it needs no per-mount tuning and
    generalises to any other static false-positive source, not just this one
    board. Confirmed on real recordings: filtering this way turns a "no
    ballistic arc found" refusal (real ball detections diluted below the 60%
    RANSAC inlier gate by ~50+ static detections) into a clean single-object
    track, visually verified frame-by-frame against the raw IR video.

    `persistence_frac` is relative to the number of frames that had ANY
    candidate at all (not total frames), since a real ball is only visible in
    a fraction of the window. A bin hit in >= max(min_persistent_frames,
    persistence_frac * n_active) frames is flagged static; `dilate` grows the
    flagged region by that many bins to catch the board's own subpixel jitter
    around its true position.

    `per_frame`: list of lists of Candidate, one list per frame (single
    camera stream -- call once per IR sensor). Returns a same-shaped list
    with static candidates removed.
    """
    from collections import defaultdict

    n_active = sum(1 for c in per_frame if c)
    if n_active == 0:
        return [list(c) for c in per_frame]

    bin_frames = defaultdict(set)
    for k, cands in enumerate(per_frame):
        for c in cands:
            b = (round(c.u / bin_px), round(c.v / bin_px))
            bin_frames[b].add(k)

    threshold = max(min_persistent_frames, persistence_frac * n_active)
    bad_bins = set()
    for b, frames_hit in bin_frames.items():
        if len(frames_hit) >= threshold:
            for dx in range(-dilate, dilate + 1):
                for dy in range(-dilate, dilate + 1):
                    bad_bins.add((b[0] + dx, b[1] + dy))

    return [[c for c in cands
             if (round(c.u / bin_px), round(c.v / bin_px)) not in bad_bins]
            for cands in per_frame]
