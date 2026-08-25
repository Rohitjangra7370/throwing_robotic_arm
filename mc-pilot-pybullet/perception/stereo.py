"""
Rectified stereo triangulation for the D435i's two IR imagers.

WHY THIS IS NOT THE DEPTH THE REPO REJECTS
------------------------------------------
`ray_plane.py`, `HARDWARE_SETUP.md` and `HANDOFF.md` item 5 all reject
RealSense's block-matching DEPTH MAP as a position source: ~2% of range = 2-4 cm
at 1-2 m, the same size as the landing error being measured. That objection
stands. It is an objection to matching textureless surfaces -- and it is even
worse for a small round airborne ball, where the depth map returns 0.0 as often
as it returns a noisy value.

This module never touches the depth map. It triangulates two INDEPENDENTLY
DETECTED, high-contrast blob centroids off the factory baseline. Different
operation, different error model: ~0.7 mm laterally per frame at Z = 2 m.

WHY THERE IS NO RECTIFICATION STEP
----------------------------------
Measured off the unit: the IR1 -> IR2 extrinsic is a pure translation of
(-49.9448, 0, 0) mm with EXACTLY identity rotation, and every distortion
coefficient is zero. The pair is already rectified in hardware, so disparity is
purely horizontal and depth is one division. Do not add a rectification step;
adding one would only introduce interpolation error.

FRAME. All 3D points here are in the IR1 (LEFT) optical frame: +X right,
+Y down, +Z forward. Converting to the base frame is the caller's job and needs
T_B_C, which this module deliberately does not know about.
"""

from __future__ import annotations

import numpy as np

from perception.ray_plane import Intrinsics

__all__ = ["D435I_IR_BASELINE_M", "StereoRig", "pair_candidates"]

# Measured 2026-08-25: get_extrinsics_to() from infrared,1 to infrared,2 returns
# translation (-49.9448, 0, 0) mm. p_2 = p_1 + t, so the IR2 ORIGIN sits at
# -t = +49.9448 mm along IR1's +X axis: IR1 is the LEFT camera and disparity
# u1 - u2 is positive for any point in front of the pair.
D435I_IR_BASELINE_M = 0.0499448


class StereoRig:
    """A rectified pair: shared intrinsics, one horizontal baseline."""

    def __init__(self, intr: Intrinsics, baseline_m: float):
        if any(intr.coeffs):
            raise ValueError("StereoRig assumes a rectified, undistorted pair; "
                             "these intrinsics carry non-zero distortion")
        self.intr = intr
        self.baseline_m = float(baseline_m)

    def project(self, p_cam):
        """
        IR1-frame point(s) -> (u1, v1, u2, v2).

        Shape contract: (3,) in -> four scalars out; (N, 3) in -> four (N,)
        arrays out.
        """
        p = np.asarray(p_cam, float)
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        if np.any(z <= 0):
            raise RuntimeError("point at or behind the image plane (z <= 0); "
                               "it has no projection")
        i = self.intr
        u1 = i.fx * x / z + i.ppx
        v1 = i.fy * y / z + i.ppy
        u2 = i.fx * (x - self.baseline_m) / z + i.ppx
        v2 = v1 if np.ndim(v1) == 0 else v1.copy()
        return u1, v1, u2, v2

    def triangulate(self, u1, v1, u2, v2):
        """
        Matched pixel pair(s) -> IR1-frame point(s).

        Uses u1 - u2 for depth and (u1, v1) for the bearing. `v2` is accepted so
        callers pass a full match, and is used only to check the pair really does
        lie on a common row -- a mismatched pair that slipped through pairing
        must raise here rather than produce a confident wrong depth.
        """
        u1 = np.asarray(u1, float); v1 = np.asarray(v1, float)
        u2 = np.asarray(u2, float); v2 = np.asarray(v2, float)
        d = u1 - u2
        if np.any(d <= 0):
            raise RuntimeError(
                f"non-positive disparity (min {np.min(d):.3f} px) -- either the "
                f"left/right images are swapped or the pairing is wrong; there "
                f"is no depth for this pair")
        if np.any(np.abs(v1 - v2) > 3.0):
            raise RuntimeError(
                f"pair is not on a common row (max |v1-v2| = "
                f"{np.max(np.abs(v1 - v2)):.2f} px > 3.0) -- the pair is "
                f"rectified, so this is a mis-match, not noise")
        i = self.intr
        z = i.fx * self.baseline_m / d
        x = (u1 - i.ppx) * z / i.fx
        y = (v1 - i.ppy) * z / i.fy
        return np.stack([x, y, z], axis=-1)
