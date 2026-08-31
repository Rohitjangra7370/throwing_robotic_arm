"""
The live view. Drawing only -- no capture, no camera, no Tk, so it can be
tested on synthetic arrays.

Layered deliberately: per-frame detections are drawn even when no track has been
fitted yet, so a detection failure is visible AS IT HAPPENS rather than only
surfacing later as a refusal from measure_landing.
"""
from __future__ import annotations

import cv2
import numpy as np

_GREEN, _CYAN, _MAGENTA, _WHITE = (0, 255, 0), (255, 255, 0), (255, 0, 255), (255, 255, 255)


def render_overlay(ir1, ir2, detections=None, path_px=None, landing=None, status=""):
    """
    (left IR, right IR) -> one side-by-side BGR frame with the overlay drawn.

    `detections` is (left_list, right_list) of (u, v, area_px). Any detection with
    non-finite u, v, or area is skipped silently.
    `path_px` is an (N, 2) array of the triangulated track projected into the
    LEFT image. Any row with non-finite coordinates is filtered out before drawing.
    `landing` is (x, y, sigma) in base-frame metres.

    If ir1 and ir2 have different heights, the shorter is zero-padded at the bottom
    to match, ensuring a valid side-by-side canvas.
    """
    left = cv2.cvtColor(np.asarray(ir1), cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(np.asarray(ir2), cv2.COLOR_GRAY2BGR)

    # Handle mismatched heights by padding the shorter image
    if left.shape[0] != right.shape[0]:
        max_height = max(left.shape[0], right.shape[0])
        if left.shape[0] < max_height:
            pad_height = max_height - left.shape[0]
            left = cv2.copyMakeBorder(left, 0, pad_height, 0, 0, cv2.BORDER_CONSTANT, value=0)
        if right.shape[0] < max_height:
            pad_height = max_height - right.shape[0]
            right = cv2.copyMakeBorder(right, 0, pad_height, 0, 0, cv2.BORDER_CONSTANT, value=0)

    if detections is not None:
        # Compute radius cap based on image diagonal
        max_r = int(np.sqrt(left.shape[0]**2 + left.shape[1]**2) / 2)
        for img, dets in zip((left, right), detections):
            for (u, v, area) in dets:
                # Skip detections with non-finite coordinates or area
                if not (np.isfinite(u) and np.isfinite(v) and np.isfinite(area)):
                    continue
                r = int(np.clip(np.sqrt(max(area, 1.0) / np.pi) * 2, 6, max_r))
                cv2.circle(img, (int(round(u)), int(round(v))), r, _GREEN, 2)

    if path_px is not None and len(path_px) >= 2:
        # Filter to finite rows only
        path_array = np.asarray(path_px)
        finite_mask = np.isfinite(path_array).all(axis=1)
        finite_pts = path_array[finite_mask]
        if len(finite_pts) >= 2:
            pts = np.asarray(finite_pts, np.int32).reshape(-1, 1, 2)
            cv2.polylines(left, [pts], False, _CYAN, 2)

    canvas = np.hstack([left, right])

    if landing is not None:
        x, y, sigma = landing
        cv2.putText(canvas, f"landing  x={x:+.3f}  y={y:+.3f}  sigma={sigma*100:.1f}cm",
                    (12, canvas.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    _MAGENTA, 2, cv2.LINE_AA)
    if status:
        cv2.putText(canvas, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    _WHITE, 2, cv2.LINE_AA)
    return canvas
