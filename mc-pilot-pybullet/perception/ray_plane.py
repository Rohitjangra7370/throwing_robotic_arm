"""
Pixel -> base-frame position by ray-plane intersection.

WHY NOT DEPTH
-------------
The D435i's stereo depth error is ~2% of range: 2-4 cm at 1-2 m. Our landing
error is ~2.9 cm. Using depth as the position source would mean measuring a
quantity with an instrument whose noise is the same size as the quantity. The
sim `depth_camera.py` back-projects depth only because sim depth is exact; that
method does not transfer.

Instead: the landing surface is a plane of KNOWN height in the base frame, so a
pixel plus calibrated intrinsics and extrinsics determines a position exactly,
with no dependence on stereo depth at all. Depth stays useful as a segmentation
gate and a gross sanity check -- never as the position.

THE BALL-RADIUS CORRECTION
--------------------------
A camera sees the ball's silhouette, whose centroid is the ball's CENTRE. A
ball resting on the plane has its centre one radius above that plane. Naively
intersecting against the support plane therefore reports a point pushed
radially away from the camera by ~r*tan(theta), where theta is the off-axis
angle.

For an overhead mount at 1.9 m and a target 0.8 m off-axis, that is
0.0327 * (0.8/1.9) = 1.4 cm of pure systematic -- half the error budget, and it
would look exactly like a policy bias. So `ball_center_on_plane` intersects
against z = z_plane + radius and returns the centre's (x, y), which IS the
contact point's (x, y) for a sphere.

Everything here is pure geometry: no camera, no ROS, fully unit-testable, and
it is the piece that must be right before a single landing is believed.
"""

from __future__ import annotations

import numpy as np

__all__ = ["Intrinsics", "pixel_ray", "intersect_plane", "ball_center_on_plane",
           "D435I_COLOR_1280x720", "D435I_COLOR_1920x1080"]


class Intrinsics:
    """Pinhole intrinsics. `coeffs` is Brown-Conrady (k1,k2,p1,p2,k3)."""

    def __init__(self, fx, fy, ppx, ppy, width=None, height=None, coeffs=(0.,) * 5):
        self.fx, self.fy = float(fx), float(fy)
        self.ppx, self.ppy = float(ppx), float(ppy)
        self.width, self.height = width, height
        self.coeffs = tuple(float(c) for c in coeffs)

    @property
    def K(self):
        return np.array([[self.fx, 0.0, self.ppx],
                         [0.0, self.fy, self.ppy],
                         [0.0, 0.0, 1.0]])

    def hfov_deg(self):
        return 2.0 * np.degrees(np.arctan(self.width / (2.0 * self.fx)))

    def vfov_deg(self):
        return 2.0 * np.degrees(np.arctan(self.height / (2.0 * self.fy)))


# Read off the lab D435i (SN 349522070924, fw 5.17.0.10) over control transfers
# on 2026-08-07. Factory-rectified colour: all distortion coefficients are zero,
# so undistortion is a no-op for this stream -- do NOT assume that holds if the
# stream resolution changes.
D435I_COLOR_1280x720 = Intrinsics(fx=910.79, fy=910.15, ppx=654.06, ppy=370.69,
                                  width=1280, height=720, coeffs=(0., 0., 0., 0., 0.))

# 1080p is the unit's best colour mode on the USB3 link and the one to use for
# landing position: ~1.46 mm/px from a 2.0 m overhead mount. Intrinsics are
# per-resolution -- fx and ppx scale with width, which is why this cannot be
# derived from the 720p entry by eye. FOV is identical (70.2 x 43.2 deg), as it
# must be for the same sensor.
D435I_COLOR_1920x1080 = Intrinsics(fx=1366.19, fy=1365.22, ppx=981.10, ppy=556.04,
                                   width=1920, height=1080, coeffs=(0., 0., 0., 0., 0.))


def _undistort(x, y, coeffs, iters=5):
    """Invert Brown-Conrady on normalised coords. No-op when coeffs are zero."""
    if not any(coeffs):
        return x, y
    k1, k2, p1, p2, k3 = coeffs
    xu, yu = x, y
    for _ in range(iters):
        r2 = xu * xu + yu * yu
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        dx = 2.0 * p1 * xu * yu + p2 * (r2 + 2.0 * xu * xu)
        dy = p1 * (r2 + 2.0 * yu * yu) + 2.0 * p2 * xu * yu
        xu = (x - dx) / radial
        yu = (y - dy) / radial
    return xu, yu


def pixel_ray(u, v, intr: Intrinsics, R=None, t=None):
    """
    (u, v) -> unit ray in the base frame, plus its origin.

    R, t are the camera pose in base coordinates: p_B = R @ p_C + t. With
    R = I and t = 0 the ray is returned in the optical frame (+Z forward,
    +X right, +Y down), which is what the unit tests exercise.

    Shape contract: scalar (u, v) -> direction of shape (3,); array-like
    (u, v) of length N -> (N, 3). `intersect_plane` accepts either.
    """
    x = (np.asarray(u, float) - intr.ppx) / intr.fx
    y = (np.asarray(v, float) - intr.ppy) / intr.fy
    x, y = _undistort(x, y, intr.coeffs)
    d_c = np.stack([x, y, np.ones_like(x)], axis=-1)
    d_c = d_c / np.linalg.norm(d_c, axis=-1, keepdims=True)
    if R is None:
        R = np.eye(3)
    d_b = d_c @ np.asarray(R, float).T
    o_b = np.zeros(3) if t is None else np.asarray(t, float)
    return o_b, d_b


def intersect_plane(o_b, d_b, z_plane):
    """
    Intersect ray(s) with the horizontal plane z = z_plane, in the base frame.

    Returns NaN where the ray is parallel to the plane or points away from it --
    silently returning a huge number would put a fake landing in the dataset.
    """
    d_b = np.atleast_2d(np.asarray(d_b, float))
    o_b = np.asarray(o_b, float)
    dz = d_b[:, 2]
    lam = np.where(np.abs(dz) < 1e-9, np.nan, (float(z_plane) - o_b[2]) / dz)
    lam = np.where(lam > 0, lam, np.nan)
    p = o_b[None, :] + lam[:, None] * d_b
    return p[0] if p.shape[0] == 1 else p


def ball_center_on_plane(u, v, intr, R, t, z_plane=0.0, ball_radius=0.0327):
    """
    Pixel of a ball's silhouette centroid -> its contact point (x, y) on the plane.

    Intersects against z = z_plane + ball_radius, because that is where the
    ball's CENTRE lies when it rests on the plane. See the module docstring:
    skipping this is a ~1.4 cm systematic at the edge of our landing zone.
    """
    o_b, d_b = pixel_ray(u, v, intr, R, t)
    p = intersect_plane(o_b, d_b, float(z_plane) + float(ball_radius))
    return np.asarray(p)[..., :2]
