"""
Find the thrown ball's pixel position via before/after background subtraction.

WHY SUBTRACTION, NOT COLOR
-----------------------------
`robot_arm/depth_camera.py` (the sim-only detector, Study 5) finds the target
BIN by HSV colour threshold -- reasonable there, the bin is a saturated green
against a plain floor. The ball is a different problem: a WHITE table-tennis
ball against a light tan/beige tile floor has almost no colour or brightness
separation (both are light, low-saturation), so a colour threshold would
false-positive on the tile itself as readily as it finds the ball. This is
exactly why `HARDWARE_SETUP.md`'s own plan calls for background-subtracting
before/after frames, not colour segmentation -- subtraction only needs the
ball's ARRIVAL to change pixels, and does not care what colour it is.

This still works for a coloured ball (e.g. a tennis ball) -- subtraction is a
strict superset of what colour segmentation catches, since any object that
changes the frame produces a diff regardless of its colour.

WHAT THIS DOES NOT DO
------------------------
No depth back-projection -- `perception/ray_plane.py`'s docstring already
covers why: D435i stereo depth error (~2% of range) is the same order as the
landing accuracy being measured. This module only finds the pixel; turning
that pixel into a real-world position is `ray_plane.ball_center_on_plane`,
which needs the camera's calibrated intrinsics/extrinsics and the known
landing-plane height -- not depth.

    from perception.ball_detector import detect_ball_bgsub
    px, py, diag = detect_ball_bgsub(frame_before, frame_after)
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["detect_ball_bgsub", "detect_ball_hsv"]


def detect_ball_bgsub(frame_before, frame_after, diff_thresh=25, min_area_px=15,
                      max_area_px=8000, morph_kernel=5):
    """
    Pixel centroid of what changed between two frames.

    Parameters
    ----------
    frame_before, frame_after : BGR arrays, same shape, same camera pose.
        `frame_before` should be the empty landing zone (captured once,
        before the throw); `frame_after` is captured once the ball has
        landed and settled -- NOT mid-flight, motion blur defeats the whole
        approach.
    diff_thresh : int
        Per-pixel absolute grayscale difference to count as "changed".
        25 is a starting point for indoor lighting; raise it if the floor's
        own lighting drifts between the two captures (e.g. auto-exposure
        resettling) and produces false diff everywhere.
    min_area_px, max_area_px : int
        Reject blobs outside this changed-pixel-count range -- catches both
        sensor noise (too small) and a lighting-drift false positive that
        floods the whole frame (too large), neither of which is the ball.

    Returns
    -------
    (u, v, diag) : pixel centroid (float, float) and a diagnostics dict with
    'area_px', 'bbox', 'mask_nonzero_frac' -- inspect these before trusting
    the centroid, especially mask_nonzero_frac, which is the single number
    that catches lighting drift (should be a few percent, not double digits).

    Raises RuntimeError if nothing in the plausible size range is found --
    fails loudly rather than returning a fabricated centroid, matching this
    project's standing rule (see CLAUDE.md's model-belief-trap note): a wrong
    detection that looks like a normal number is worse than an explicit
    failure.
    """
    if frame_before.shape != frame_after.shape:
        raise ValueError(f"frame shape mismatch: before={frame_before.shape} "
                         f"after={frame_after.shape} -- camera must not move "
                         f"between the two captures")

    gray_before = cv2.cvtColor(frame_before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(frame_after, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_before, gray_after)
    mask = (diff >= diff_thresh).astype(np.uint8) * 255

    kernel = np.ones((morph_kernel, morph_kernel), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    mask_nonzero_frac = float(np.count_nonzero(mask)) / mask.size

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [c for c in contours if min_area_px <= cv2.contourArea(c) <= max_area_px]
    if not candidates:
        raise RuntimeError(
            f"no changed region in [{min_area_px}, {max_area_px}] px found -- "
            f"mask_nonzero_frac={mask_nonzero_frac:.3f}, "
            f"{'looks like lighting drift flooded the frame' if mask_nonzero_frac > 0.15 else 'ball may not have landed in frame, or diff_thresh is too high'}")

    # Largest surviving candidate -- a real ball produces one solid blob;
    # sensor noise produces many tiny scattered ones already filtered by
    # min_area_px, so "largest" is a real ball, not a coin-flip pick.
    contour = max(candidates, key=cv2.contourArea)
    area_px = float(cv2.contourArea(contour))
    bbox = cv2.boundingRect(contour)
    M = cv2.moments(contour)
    if M["m00"] == 0:
        raise RuntimeError("degenerate contour (zero moment) -- treat as a failed detection")
    u = M["m10"] / M["m00"]
    v = M["m01"] / M["m00"]

    diag = {"area_px": area_px, "bbox": bbox, "mask_nonzero_frac": mask_nonzero_frac,
           "n_candidates": len(candidates)}
    return float(u), float(v), diag


def detect_ball_hsv(frame, hsv_lower, hsv_upper, min_area_px=15, max_area_px=8000,
                    morph_kernel=5):
    """
    Colour-threshold fallback -- only reliable for a ball with real colour
    separation from the floor (e.g. a yellow-green tennis ball). Do NOT use
    this for the white TT ball; see the module docstring for why. Same
    contour/moment logic as detect_ball_bgsub, single-frame input instead of
    a before/after pair.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.asarray(hsv_lower, dtype=np.uint8),
                       np.asarray(hsv_upper, dtype=np.uint8))
    kernel = np.ones((morph_kernel, morph_kernel), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [c for c in contours if min_area_px <= cv2.contourArea(c) <= max_area_px]
    if not candidates:
        raise RuntimeError(f"no HSV blob in [{min_area_px}, {max_area_px}] px found")

    contour = max(candidates, key=cv2.contourArea)
    area_px = float(cv2.contourArea(contour))
    bbox = cv2.boundingRect(contour)
    M = cv2.moments(contour)
    if M["m00"] == 0:
        raise RuntimeError("degenerate contour (zero moment)")
    u = M["m10"] / M["m00"]
    v = M["m01"] / M["m00"]
    return float(u), float(v), {"area_px": area_px, "bbox": bbox, "n_candidates": len(candidates)}
