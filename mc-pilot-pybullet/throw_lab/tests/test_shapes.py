"""Unit tests for the normalized shape library (no PyBullet needed)."""

import numpy as np
import pytest

from throw_lab import shapes as sh


VEL_SHAPES = [
    sh.ConstAccel(),
    sh.MinJerkVel(),
    sh.TrapAccel(0.1),
    sh.TrapAccel(0.25),
    sh.TrapAccel(0.5),
    sh.Plateau(sh.TrapAccel(0.25), 0.15),
    sh.Plateau(sh.MinJerkVel(), 0.20),
]

REST_SHAPES = [sh.CubicRest(), sh.MinJerkRest(), sh.TrapVelRest(0.25), sh.TrapVelRest(0.5)]


@pytest.mark.parametrize("shape", VEL_SHAPES, ids=lambda s: s.name)
def test_velocity_shape_endpoints(shape):
    assert shape.s(0.0) == pytest.approx(0.0, abs=1e-12)
    assert shape.s(1.0) == pytest.approx(1.0, abs=1e-12)
    assert shape.Sint(0.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("shape", VEL_SHAPES, ids=lambda s: s.name)
def test_velocity_shape_monotone(shape):
    """Peak |qd| during the throw phase must be exactly qd_release.

    This is what makes the "|qd| <= qd_max holds by construction" guarantee in
    the planner true.  A non-monotone shape would overshoot the joint velocity
    limit mid-phase without any check catching it.
    """
    t = np.linspace(0.0, 1.0, 2001)
    s = shape.s(t)
    assert np.all(np.diff(s) >= -1e-12)
    assert s.max() <= 1.0 + 1e-12


@pytest.mark.parametrize("shape", VEL_SHAPES, ids=lambda s: s.name)
def test_velocity_shape_derivative_consistency(shape):
    """ds/dds/Sint must be the actual derivatives/integral of s."""
    t = np.linspace(1e-4, 1.0 - 1e-4, 4001)
    num_ds = np.gradient(shape.s(t), t)
    assert np.allclose(num_ds, shape.ds(t), atol=2e-2)
    # Sint' = s
    num_s = np.gradient(shape.Sint(t), t)
    assert np.allclose(num_s, shape.s(t), atol=2e-3)


@pytest.mark.parametrize("shape", VEL_SHAPES, ids=lambda s: s.name)
def test_velocity_shape_S1_matches_integral(shape):
    t = np.linspace(0.0, 1.0, 200001)
    assert shape.Sint(1.0) == pytest.approx(shape.S1, abs=1e-9)
    assert np.trapz(shape.s(t), t) == pytest.approx(shape.S1, abs=1e-6)


@pytest.mark.parametrize("shape", VEL_SHAPES, ids=lambda s: s.name)
def test_velocity_shape_advertised_peaks(shape):
    t = np.linspace(0.0, 1.0, 200001)
    assert np.max(np.abs(shape.ds(t))) == pytest.approx(shape.peak_ds, rel=1e-3)
    if np.isfinite(shape.peak_dds):
        assert np.max(np.abs(shape.dds(t))) <= shape.peak_dds * (1 + 1e-6)


def test_symmetric_shapes_all_integrate_to_half():
    """The load-bearing fact: the cock-back distance is shape-independent.

    Every symmetric shape gives dq = qd_release * T / 2, so const_accel,
    min_jerk and trap_accel can be compared at the SAME windup pose and the
    SAME duration -- only the acceleration distribution differs.
    """
    for shape in (sh.ConstAccel(), sh.MinJerkVel(), sh.TrapAccel(0.25), sh.TrapAccel(0.5)):
        assert shape.S1 == pytest.approx(0.5, abs=1e-12)


def test_release_acceleration_is_the_timing_sensitivity():
    """ds(1) is the first-order dv/dt_release sensitivity, in units qd_e/T."""
    assert sh.ConstAccel().ds_at_release == pytest.approx(1.0)
    assert sh.MinJerkVel().ds_at_release == pytest.approx(0.0)
    assert sh.TrapAccel(0.25).ds_at_release == pytest.approx(0.0)
    for shape in VEL_SHAPES:
        assert shape.ds(1.0) == pytest.approx(shape.ds_at_release, abs=1e-9)


def test_const_accel_reproduces_shipped_cubic_to_velocity():
    """ConstAccel + dq = qd_e*T/2 IS `arm_controller._cubic_to_velocity`.

    The shipped `monotonic_windup` path cocks back by exactly qd_e*local/2, at
    which point the cubic's a3 coefficient vanishes identically.  If this test
    ever fails, the lab's "baseline" is no longer the shipped baseline.
    """
    rng = np.random.default_rng(0)
    n = 7
    qd_e = rng.normal(size=n)
    T = 0.83
    q_end = rng.normal(size=n)
    q_start = q_end - qd_e * T / 2.0          # the monotonic_windup cock

    dq = q_end - q_start
    a0 = q_start
    a1 = np.zeros(n)
    a3 = (qd_e * T - 2.0 * dq) / T**3
    a2 = (3.0 * dq - qd_e * T) / T**2
    assert np.allclose(a3, 0.0, atol=1e-12)   # the degeneracy itself

    shape = sh.ConstAccel()
    for t in np.linspace(0.0, T, 41):
        tau = t / T
        q_cub = a0 + a1 * t + a2 * t**2 + a3 * t**3
        qd_cub = a1 + 2.0 * a2 * t + 3.0 * a3 * t**2
        qdd_cub = 2.0 * a2 + 6.0 * a3 * t
        q_lab = q_start + qd_e * T * shape.Sint(tau)
        qd_lab = qd_e * shape.s(tau)
        qdd_lab = qd_e * shape.ds(tau) / T
        assert np.allclose(q_cub, q_lab, atol=1e-12)
        assert np.allclose(qd_cub, qd_lab, atol=1e-12)
        assert np.allclose(qdd_cub, qdd_lab, atol=1e-12)


def test_cubic_rest_shape_reproduces_shipped_rest_to_rest():
    rng = np.random.default_rng(1)
    n = 7
    q0 = rng.normal(size=n)
    q1 = rng.normal(size=n)
    T = 0.61
    dq = q1 - q0
    a2 = 3.0 * dq / T**2
    a3 = -2.0 * dq / T**3
    shape = sh.CubicRest()
    for t in np.linspace(0.0, T, 41):
        tau = t / T
        q_cub = q0 + a2 * t**2 + a3 * t**3
        qd_cub = 2.0 * a2 * t + 3.0 * a3 * t**2
        assert np.allclose(q_cub, q0 + dq * shape.Vint(tau), atol=1e-12)
        assert np.allclose(qd_cub, dq * shape.v(tau) / T, atol=1e-12)


@pytest.mark.parametrize("shape", REST_SHAPES, ids=lambda s: s.name)
def test_rest_shape_unit_integral_and_endpoints(shape):
    t = np.linspace(0.0, 1.0, 200001)
    assert shape.v(0.0) == pytest.approx(0.0, abs=1e-12)
    assert shape.v(1.0) == pytest.approx(0.0, abs=1e-12)
    assert shape.Vint(0.0) == pytest.approx(0.0, abs=1e-12)
    assert shape.Vint(1.0) == pytest.approx(1.0, abs=1e-9)
    assert np.trapz(shape.v(t), t) == pytest.approx(1.0, abs=1e-6)
    assert np.max(np.abs(shape.v(t))) == pytest.approx(shape.peak_v, rel=1e-3)


@pytest.mark.parametrize("shape", REST_SHAPES, ids=lambda s: s.name)
def test_rest_shape_derivative_consistency(shape):
    t = np.linspace(1e-4, 1.0 - 1e-4, 4001)
    assert np.allclose(np.gradient(shape.Vint(t), t), shape.v(t), atol=2e-3)
    assert np.allclose(np.gradient(shape.v(t), t), shape.dv(t), atol=5e-2)


def test_quintic_bc_hits_all_six_conditions():
    rng = np.random.default_rng(2)
    n = 7
    q0, qd0, qdd0 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    qT, qdT, qddT = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    T = 1.37
    c = sh.quintic_bc(q0, qd0, qdd0, qT, qdT, qddT, T)
    q, qd, qdd, _ = sh.eval_quintic(c, 0.0)
    assert np.allclose(q, q0) and np.allclose(qd, qd0) and np.allclose(qdd, qdd0)
    q, qd, qdd, _ = sh.eval_quintic(c, T)
    assert np.allclose(q, qT) and np.allclose(qd, qdT) and np.allclose(qdd, qddT)


def test_quintic_derivatives_are_consistent():
    rng = np.random.default_rng(3)
    n = 3
    c = sh.quintic_bc(
        rng.normal(size=n), rng.normal(size=n), rng.normal(size=n),
        rng.normal(size=n), rng.normal(size=n), rng.normal(size=n), 1.0,
    )
    h = 1e-6
    for t in (0.2, 0.5, 0.9):
        q0, qd0, qdd0, qddd0 = sh.eval_quintic(c, t)
        qp = sh.eval_quintic(c, t + h)
        qm = sh.eval_quintic(c, t - h)
        assert np.allclose((qp[0] - qm[0]) / (2 * h), qd0, atol=1e-5)
        assert np.allclose((qp[1] - qm[1]) / (2 * h), qdd0, atol=1e-4)
        assert np.allclose((qp[2] - qm[2]) / (2 * h), qddd0, atol=1e-3)


def test_plateau_shifts_displacement_not_velocity():
    base = sh.TrapAccel(0.25)
    pl = sh.Plateau(base, 0.2)
    assert pl.s(1.0) == pytest.approx(1.0)
    assert pl.S1 == pytest.approx(base.S1 * 0.8 + 0.2)
    # zero acceleration everywhere inside the plateau -> release timing is free
    for tau in (0.85, 0.9, 0.95, 1.0):
        assert pl.ds(tau) == pytest.approx(0.0, abs=1e-12)


def test_registry_round_trip():
    assert sh.get_velocity_shape("const_accel").name == "const_accel"
    assert sh.get_velocity_shape("trap_accel", beta=0.3).peak_ds == pytest.approx(
        1.0 / 0.7
    )
    assert sh.get_rest_shape("min_jerk").peak_v == pytest.approx(1.875)
    with pytest.raises(ValueError):
        sh.get_velocity_shape("nope")
