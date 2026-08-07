"""
Ray-plane geometry tests. Pure math -- no camera, no arm.

This is the only vision code that can be fully validated before the camera is
mounted, and it is the code every landing measurement will flow through, so it
gets validated now rather than on run day.
"""
import numpy as np
import pytest

from perception.ray_plane import (D435I_COLOR_1280x720, Intrinsics,
                                  ball_center_on_plane, intersect_plane, pixel_ray)

INTR = D435I_COLOR_1280x720
BALL_R = 0.0327


def _overhead(height):
    """Camera at (0,0,height) looking straight down, x_img->+x_B, y_img->-y_B."""
    R = np.array([[1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0],
                  [0.0, 0.0, -1.0]])
    return R, np.array([0.0, 0.0, float(height)])


def test_lab_intrinsics_match_the_device():
    """Read off the lab D435i over control transfers; guards a silent edit."""
    assert (INTR.fx, INTR.fy) == (910.79, 910.15)
    assert (INTR.ppx, INTR.ppy) == (654.06, 370.69)
    assert INTR.hfov_deg() == pytest.approx(70.2, abs=0.1)
    assert INTR.vfov_deg() == pytest.approx(43.2, abs=0.1)
    assert not any(INTR.coeffs), "colour stream is factory-rectified"


def test_principal_ray_is_straight_down_and_hits_directly_below():
    R, t = _overhead(1.9)
    o, d = pixel_ray(INTR.ppx, INTR.ppy, INTR, R, t)
    assert d.shape == (3,), "scalar pixel in -> single ray out"
    assert np.allclose(d, [0, 0, -1], atol=1e-9)
    p = intersect_plane(o, d, 0.0)
    assert np.allclose(p[:2], [0, 0], atol=1e-9)
    assert p[2] == pytest.approx(0.0)


def test_ray_away_from_plane_returns_nan_not_a_fake_landing():
    """A ray pointing up must not silently produce a landing point."""
    R, t = _overhead(1.9)
    o, d = pixel_ray(INTR.ppx, INTR.ppy, INTR, np.eye(3), t)  # +Z is up here
    p = intersect_plane(o, d, 0.0)
    assert np.all(np.isnan(np.asarray(p)[:2]))


def test_round_trip_known_point_through_the_camera():
    """Project a known base-frame point to a pixel, then recover it."""
    R, t = _overhead(1.9)
    p_b = np.array([0.72, -0.15, 0.0])
    p_c = R.T @ (p_b - t)                      # into optical frame
    u = INTR.fx * p_c[0] / p_c[2] + INTR.ppx
    v = INTR.fy * p_c[1] / p_c[2] + INTR.ppy
    o, d = pixel_ray(u, v, INTR, R, t)
    assert np.allclose(intersect_plane(o, d, 0.0)[:2], p_b[:2], atol=1e-9)


def test_ball_radius_correction_is_worth_over_a_centimetre():
    """
    The systematic this whole module exists to avoid.

    A ball resting at 0.80 m off-axis under a 1.9 m overhead mount: intersecting
    the SUPPORT plane instead of the ball-centre plane pushes the estimate
    radially outward by ~r*(offset/height). That is a bias, not noise -- it would
    read as a policy that consistently overthrows.
    """
    R, t = _overhead(1.9)
    true_xy = np.array([0.80, 0.0])
    centre = np.array([true_xy[0], true_xy[1], BALL_R])     # ball centre in B
    p_c = R.T @ (centre - t)
    u = INTR.fx * p_c[0] / p_c[2] + INTR.ppx
    v = INTR.fy * p_c[1] / p_c[2] + INTR.ppy

    good = ball_center_on_plane(u, v, INTR, R, t, z_plane=0.0, ball_radius=BALL_R)
    assert np.allclose(good, true_xy, atol=1e-9), "corrected path must be exact"

    o, d = pixel_ray(u, v, INTR, R, t)
    naive = intersect_plane(o, d, 0.0)[:2]
    err = np.linalg.norm(naive - true_xy)
    assert err > 0.012, f"expected >1.2 cm naive bias, got {err*100:.2f} cm"
    assert naive[0] > true_xy[0], "bias must push radially OUTWARD"


def test_bias_vanishes_directly_under_an_overhead_camera():
    """Sanity on the mechanism: the error is off-axis only, so it is a bias
    that grows across the workspace rather than a constant offset."""
    R, t = _overhead(1.9)
    centre = np.array([0.0, 0.0, BALL_R])
    p_c = R.T @ (centre - t)
    u = INTR.fx * p_c[0] / p_c[2] + INTR.ppx
    v = INTR.fy * p_c[1] / p_c[2] + INTR.ppy
    o, d = pixel_ray(u, v, INTR, R, t)
    assert np.allclose(intersect_plane(o, d, 0.0)[:2], [0, 0], atol=1e-9)


def test_distortion_inversion_round_trips_when_coeffs_are_nonzero():
    """Guards the undistort path, which is a no-op on this unit but must be
    correct if a distorted stream is ever used."""
    intr = Intrinsics(fx=600, fy=600, ppx=320, ppy=240, width=640, height=480,
                      coeffs=(0.1, -0.05, 0.001, -0.002, 0.01))
    xn, yn = 0.21, -0.13
    r2 = xn * xn + yn * yn
    k1, k2, p1, p2, k3 = intr.coeffs
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
    yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    o, d = pixel_ray(xd * intr.fx + intr.ppx, yd * intr.fy + intr.ppy, intr)
    assert np.allclose(d[:2] / d[2], [xn, yn], atol=1e-6)


def test_1080p_intrinsics_agree_with_720p_on_field_of_view():
    """
    Intrinsics are per-resolution, so both are read from the device rather than
    scaled by hand -- but they describe one sensor, so the FOV must match. A
    mismatch means one of them was mistyped, which would silently bias every
    landing measured at that resolution.
    """
    from perception.ray_plane import D435I_COLOR_1920x1080 as HD
    assert HD.hfov_deg() == pytest.approx(INTR.hfov_deg(), abs=0.15)
    assert HD.vfov_deg() == pytest.approx(INTR.vfov_deg(), abs=0.15)
    # principal point should sit near centre for both
    for i in (INTR, HD):
        assert abs(i.ppx - i.width / 2) < 0.05 * i.width
        assert abs(i.ppy - i.height / 2) < 0.05 * i.height


def test_1080p_round_trips_a_known_landing():
    from perception.ray_plane import D435I_COLOR_1920x1080 as HD
    R, t = _overhead(2.0)
    true_xy = np.array([0.66, 0.21])
    centre = np.array([true_xy[0], true_xy[1], BALL_R])
    p_c = R.T @ (centre - t)
    u = HD.fx * p_c[0] / p_c[2] + HD.ppx
    v = HD.fy * p_c[1] / p_c[2] + HD.ppy
    got = ball_center_on_plane(u, v, HD, R, t, z_plane=0.0, ball_radius=BALL_R)
    assert np.allclose(got, true_xy, atol=1e-9)
