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

    `detections` is (left_list, right_list) of (u, v, area_px).
    `path_px` is an (N, 2) array of the triangulated track projected into the
    LEFT image. `landing` is (x, y, sigma) in base-frame metres.
    """
    left = cv2.cvtColor(np.asarray(ir1), cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(np.asarray(ir2), cv2.COLOR_GRAY2BGR)

    if detections is not None:
        for img, dets in zip((left, right), detections):
            for (u, v, area) in dets:
                r = int(max(6, np.sqrt(max(area, 1.0) / np.pi) * 2))
                cv2.circle(img, (int(round(u)), int(round(v))), r, _GREEN, 2)

    if path_px is not None and len(path_px) >= 2:
        pts = np.asarray(path_px, np.int32).reshape(-1, 1, 2)
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
