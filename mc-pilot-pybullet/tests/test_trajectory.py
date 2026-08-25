"""
Ballistic fitting and the impact solve. Pure math -- no camera, no arm.
"""
import numpy as np
import pytest

from perception.trajectory import (BALL_RADIUS, G_BASE, Z_FLOOR_BASE,
                                   ballistic_position, fit_two_points,
                                   solve_impact, solve_impact_time)


def test_gravity_and_geometry_constants():
    assert np.allclose(G_BASE, [0.0, 0.0, -9.81])
    assert Z_FLOOR_BASE == -0.433, "base frame: base at 0, floor at -base_height"
    assert BALL_RADIUS == 0.0327


def test_position_at_a_hand_computed_time():
    p0 = np.array([0.0, 0.0, 1.0])
    v0 = np.array([2.0, 0.0, 0.0])
    p = ballistic_position(p0, v0, 0.45152)
    assert p[0] == pytest.approx(0.90304, abs=1e-5)
    assert p[2] == pytest.approx(0.0, abs=1e-4)


def test_impact_time_matches_the_closed_form():
    """Drop from z=1.0 with no vertical velocity: t = sqrt(2*1.0/9.81)."""
    t = solve_impact_time(np.array([0.0, 0.0, 1.0]), np.zeros(3), 0.0)
    assert t == pytest.approx(np.sqrt(2.0 / 9.81), rel=1e-12)


def test_impact_takes_the_DESCENDING_root_not_the_first_crossing():
    """
    Rising through the target plane then falling back gives two positive roots.
    Taking the smaller one reports a 'landing' while the ball is still going UP.
    """
    p0 = np.array([0.0, 0.0, 1.0])
    v0 = np.array([1.0, 0.0, 3.0])
    t = solve_impact_time(p0, v0, 1.3)
    assert t == pytest.approx(0.48568, abs=1e-4)
    assert ballistic_position(p0, v0, t)[2] == pytest.approx(1.3, abs=1e-9)
    assert (v0[2] + G_BASE[2] * t) < 0, "must be descending at impact"


def test_no_real_crossing_raises_rather_than_returning_nonsense():
    with pytest.raises(RuntimeError, match="never reaches"):
        solve_impact_time(np.array([0.0, 0.0, 1.0]), np.zeros(3), 1.5)


def test_impact_offsets_by_one_ball_radius():
    """
    A resting sphere's CENTRE is one radius above the floor, and for a sphere
    the centre's (x, y) IS the contact (x, y). This is the same convention as
    ray_plane.ball_center_on_plane -- if the two disagree, the cross-check in
    tests/test_landing_pipeline.py measures nothing.
    """
    p0 = np.array([0.0, 0.0, 1.0])
    v0 = np.array([2.0, 0.0, 0.0])
    x, y, t = solve_impact(p0, v0, z_floor=0.0, ball_radius=BALL_RADIUS)
    assert ballistic_position(p0, v0, t)[2] == pytest.approx(BALL_RADIUS, abs=1e-9)
    x_naive, _, _ = solve_impact(p0, v0, z_floor=0.0, ball_radius=0.0)
    assert x < x_naive, "the radius offset must shorten the flight, not lengthen it"


def test_two_point_fit_recovers_the_generating_trajectory():
    p0 = np.array([0.035, 0.0, 1.137])
    v0 = np.array([1.6218, 0.0, 0.1419])
    ta, tb = 0.12, 0.47
    got_p0, got_v0 = fit_two_points(ta, ballistic_position(p0, v0, ta),
                                    tb, ballistic_position(p0, v0, tb))
    assert np.allclose(got_p0, p0, atol=1e-12)
    assert np.allclose(got_v0, v0, atol=1e-12)


def test_two_point_fit_refuses_a_degenerate_sample():
    p = np.array([0.0, 0.0, 1.0])
    with pytest.raises(RuntimeError, match="separated"):
        fit_two_points(0.20, p, 0.2001, p)
