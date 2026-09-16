"""
Where is the target bin? One ArUco marker lying on the floor -> base frame.

WHY THE IR STREAM AND NOT COLOUR
--------------------------------
Two independent reasons, and the second is the one that matters.

1. Availability. `session_camera.IRRecorder` opens `infrared,1` and
   `infrared,2` and holds the D435i for the whole run day. A D435i can be
   opened by exactly one process, so while a session is live the colour stream
   does not exist. IR1 frames, by contrast, are already flowing past
   `CameraThread.latest()`.

2. FRAME CONSISTENCY, which is the real argument. `calib/T_B_C.npz` is written
   by `start_of_day.py` from the D435i's **colour** intrinsics, so strictly it
   is the pose of the colour optical frame. `measure_landing` nonetheless
   applies it to IR1-frame triangulated points, because `stereo.StereoRig`
   works in IR1. On a D435i those two optical frames are ~15 mm apart, so every
   landing this project has measured carries that offset. (Suggestively, the
   lateral bias measured across 17 real landings is +1.44 +- 0.84 cm. Not
   proven -- confirming it needs the colour-to-IR1 extrinsic read off the
   device -- but it is the right order and the right axis.)

   Reading the MARKER in IR1 too makes that offset **cancel exactly**: bin and
   ball are then expressed in the same (possibly shifted) frame, and "did the
   ball land on the marker" is answered correctly regardless. Detecting the
   marker in colour would instead subtract two differently-shifted positions
   and bake the offset straight into the aiming error. So: IR1, deliberately,
   even on the day the colour stream is free.

WHY RAY-PLANE AND NOT solvePnP
------------------------------
The marker lies flat on a floor whose height is already known and independently
gated (`start_of_day.py`'s FLOOR gate). Intersecting each detected corner's ray
with z = z_floor uses that outside fact; PnP would instead infer range from the
marker's apparent size, which is the quantity most corrupted by a mis-scaled
printout -- the exact failure `make_aruco_printable.py`'s ruler exists to catch.

That leaves the apparent size free to serve as a CHECK rather than an input:
`side_error_m` compares the side length the four ray-plane corners imply
against the printed size. It is the same argument as
`start_of_day.py`'s SCALE gate, and the same reason reprojection error cannot
validate a calibration -- a wrong scale hides inside a pose but cannot hide
inside a length measured against a known plane.

MEASURED, on 2026-09-11, by compositing markers into a real IR background from
throws/throw_020.npz (real emitter speckle, real ink levels sampled from the
ChArUco board in frame) at the real 239 px/m floor scale: 12/12 detection at
every size from 80 mm (18.8 px) to 300 mm, centre recovered to 1.6-1.9 mm
independent of size. Real printed markers have softer edges than a synthetic
composite, so treat 80 mm as the floor and prefer 150 mm+ if you are printing
one anyway.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from perception.ray_plane import intersect_plane, pixel_ray
from perception.trajectory import Z_FLOOR_BASE

__all__ = ["ARUCO_DICT", "FloorMarker", "detect_floor_markers",
           "detect_bin_marker", "MAX_SIDE_ERROR_FRAC"]

ARUCO_DICT = cv2.aruco.DICT_4X4_50   # matches make_aruco_targets.py

# How far the recovered side length may differ from the printed one before the
# reading is refused, as a fraction. 15% at a 80 mm marker is 12 mm, comfortably
# above the ~2 mm the synthetic sweep showed and well below the 33% a "fit to
# page" printout would produce.
MAX_SIDE_ERROR_FRAC = 0.15


@dataclass(frozen=True)
class FloorMarker:
    """One marker resolved onto the floor plane, in base coordinates."""
    marker_id: int
    centre_xy: np.ndarray      # (2,) base frame
    corners_xy: np.ndarray     # (4, 2) base frame, detector order
    side_m: float              # mean of the four recovered edge lengths
    side_error_m: float        # recovered - printed (nan when size unknown)
    pixel_side: float          # apparent size, for a "is it big enough" read
    centre_px: np.ndarray      # (2,) where it was in the image

    @property
    def x(self):
        return float(self.centre_xy[0])

    @property
    def y(self):
        return float(self.centre_xy[1])


def detect_floor_markers(img, intr, R_bc, t_bc, z_floor=Z_FLOOR_BASE,
                         marker_size_m=None, aruco_dict=ARUCO_DICT,
                         refine=True):
    """
    Grayscale image -> every ArUco marker in it, placed on the floor plane.

    `intr` must be the intrinsics of the stream `img` came from, and
    `R_bc`/`t_bc` the camera pose that goes with it -- pass the same T_B_C the
    landing measurement uses, and read the module docstring before deciding to
    pass anything else.

    Returns a list of FloorMarker, possibly empty. Empty is a normal answer
    (the bin may simply not be in view) and is NOT an error, matching
    `ball_track.detect_candidates`.
    """
    img = np.asarray(img)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    params = cv2.aruco.DetectorParameters()
    if refine:
        # Corner accuracy is the whole output here, and the default is none.
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(aruco_dict), params)
    corners, ids, _ = detector.detectMarkers(img)
    if ids is None or len(ids) == 0:
        return []

    R_bc = np.asarray(R_bc, float)
    t_bc = np.asarray(t_bc, float)
    out = []
    for quad, mid in zip(corners, ids.ravel()):
        px = np.asarray(quad, float).reshape(4, 2)
        o_b, d_b = pixel_ray(px[:, 0], px[:, 1], intr, R_bc, t_bc)
        pts = np.atleast_2d(intersect_plane(o_b, d_b, float(z_floor)))
        if pts.shape != (4, 3) or not np.all(np.isfinite(pts)):
            # A corner whose ray never meets the floor cannot be placed. Drop
            # the marker rather than average a NaN into the centre.
            continue
        xy = pts[:, :2]
        sides = [float(np.linalg.norm(xy[i] - xy[(i + 1) % 4])) for i in range(4)]
        side = float(np.mean(sides))
        out.append(FloorMarker(
            marker_id=int(mid),
            centre_xy=xy.mean(axis=0),
            corners_xy=xy,
            side_m=side,
            side_error_m=(float("nan") if marker_size_m is None
                          else side - float(marker_size_m)),
            pixel_side=float(np.mean([np.linalg.norm(px[i] - px[(i + 1) % 4])
                                      for i in range(4)])),
            centre_px=px.mean(axis=0),
        ))
    return out


def detect_bin_marker(img, intr, R_bc, t_bc, z_floor=Z_FLOOR_BASE,
                      marker_size_m=None, marker_id=None,
                      exclude_ids=(), aruco_dict=ARUCO_DICT,
                      max_side_error_frac=MAX_SIDE_ERROR_FRAC):
    """
    The one marker that is the bin. Raises rather than guessing.

    `marker_id` pins which tag is the bin; without it exactly one marker must
    be in view after `exclude_ids` is applied. `exclude_ids` exists because the
    ChArUco calibration board is GLUED to this floor and permanently in the
    overhead FOV (HARDWARE_RUNBOOK.md) -- its own tags share this dictionary,
    and one of them silently becoming "the bin" is precisely the kind of
    plausible wrong answer this project refuses to produce. In practice the
    board's 26 mm tags are ~6 px here and do not detect at all, but that is a
    property of the current mount, not a guarantee.

    The SCALE check runs only when `marker_size_m` is given, and is worth
    giving: it is the one thing that catches a mis-scaled printout, a wrong
    `z_floor`, or a stale extrinsic, none of which the detection itself can see.
    """
    found = [m for m in detect_floor_markers(img, intr, R_bc, t_bc, z_floor,
                                             marker_size_m, aruco_dict)
             if m.marker_id not in set(exclude_ids)]
    if marker_id is not None:
        found = [m for m in found if m.marker_id == int(marker_id)]
        if not found:
            raise RuntimeError(
                f"marker id {marker_id} is not in view. Detected: "
                f"{sorted({m.marker_id for m in detect_floor_markers(img, intr, R_bc, t_bc, z_floor)})}"
                f" -- move the bin into frame, or pass the id that is actually on it.")
    if not found:
        raise RuntimeError(
            "no ArUco marker in view -- the bin is outside the camera's "
            "field of view, or the marker is not lying flat and readable.")
    if len(found) > 1:
        raise RuntimeError(
            f"{len(found)} markers in view (ids "
            f"{sorted(m.marker_id for m in found)}) and no --marker_id given. "
            f"Refusing to guess which one is the bin.")

    m = found[0]
    if marker_size_m is not None:
        lim = float(marker_size_m) * max_side_error_frac
        if abs(m.side_error_m) > lim:
            raise RuntimeError(
                f"marker measures {m.side_m * 1000:.0f} mm on the floor plane "
                f"against a printed {float(marker_size_m) * 1000:.0f} mm "
                f"({m.side_error_m * 1000:+.0f} mm, limit +-{lim * 1000:.0f}). "
                f"Something outside the detector is wrong: a rescaled printout "
                f"('fit to page'), a wrong --marker_size, a floor height that "
                f"is not {z_floor:+.3f}, or a stale extrinsic. The position "
                f"would be wrong by about the same fraction -- refusing it.")
    return m
