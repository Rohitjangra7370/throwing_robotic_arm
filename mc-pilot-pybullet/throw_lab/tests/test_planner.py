"""Planner / harness / optimizer regressions (needs PyBullet, ~20 s)."""

import numpy as np
import pytest

from simulation_class.release_solver import OptimizedReleaseSolver
from throw_lab import dynopt, shapes as sh
from throw_lab.harness import BaselinePlanAdapter, ThrowHarness
from throw_lab.planner import LabThrowPlanner

BALL_MASS = 0.0577
SHAPES = ["const_accel", "min_jerk", "trap_accel"]


@pytest.fixture(scope="module")
def lab():
    table = list(np.load("throw_pose_table.npy", allow_pickle=True))
    with ThrowHarness(robot_name="kinova_gen3_dyn", ball_mass=BALL_MASS) as H:
        solver = OptimizedReleaseSolver(opt_posture_table=table)
        planner = LabThrowPlanner(H.arm, payload_mass=BALL_MASS)
        rp, q_rel, qd_rel, v_rel = solver.solve(
            H.arm, np.array([1.5, 0.0, 0.0]), target_xy=np.array([0.70, 0.0])
        )
        yield H, planner, np.asarray(q_rel), np.asarray(qd_rel), np.asarray(rp)


@pytest.mark.parametrize("name", SHAPES)
def test_plan_hits_the_release_state_exactly(lab, name):
    """The whole point of the release solver is that q/qd at t_r are EXACT.

    If a shape family ever fails to land on them, the pose table's carefully
    searched hardware-valid release state has been silently replaced by
    whatever the trajectory happened to do.
    """
    _, planner, q_rel, qd_rel, _ = lab
    plan = planner.plan(q_rel, qd_rel, throw_shape=name)
    q, qd, _, _ = plan.eval(plan.t_r)
    assert np.allclose(q, q_rel, atol=1e-9)
    assert np.allclose(qd, qd_rel, atol=1e-9)


@pytest.mark.parametrize("name", SHAPES)
def test_throw_phase_never_exceeds_joint_velocity_limits(lab, name):
    """Peak |qd| in the throw phase is qd_release by construction."""
    _, planner, q_rel, qd_rel, _ = lab
    plan = planner.plan(q_rel, qd_rel, throw_shape=name)
    qd_max = np.asarray(planner.arm._qd_max)
    for t in np.linspace(plan.t_w, plan.t_r, 400):
        _, qd, _, _ = plan.eval(t)
        assert np.all(np.abs(qd) <= np.abs(qd_rel) + 1e-9)
        assert np.all(np.abs(qd) <= qd_max + 1e-9)


@pytest.mark.parametrize("name", SHAPES)
def test_brake_phase_never_exceeds_release_velocity(lab, name):
    """The reason the follow-through is split into brake + return.

    The shipped single-cubic follow-through has no a-priori velocity bound and
    measured 5.6x qd_max at its nominal duration; a time-reversed shape ramp
    can only ever decrease |qd| from |qd_release|.
    """
    _, planner, q_rel, qd_rel, _ = lab
    plan = planner.plan(q_rel, qd_rel, throw_shape=name)
    for t in np.linspace(plan.t_r, plan.t_b, 400):
        _, qd, _, _ = plan.eval(t)
        assert np.all(np.abs(qd) <= np.abs(qd_rel) + 1e-9)


@pytest.mark.parametrize("name", ["min_jerk", "trap_accel"])
def test_smooth_shapes_have_no_acceleration_step_at_any_join(lab, name):
    """C2 across every phase boundary -- the property a cubic cannot have.

    `accel_step` is a central difference over +-1e-6 s, so a genuinely
    continuous acceleration still shows a residual of jerk * 2e-6 (tens of
    microrad/s^2 here).  A real step is O(1) rad/s^2 -- const_accel measures
    >1 at the same joins -- so 1e-3 separates them by three orders of magnitude.
    """
    _, planner, q_rel, qd_rel, _ = lab
    plan = planner.plan(q_rel, qd_rel, throw_shape=name, windup_shape="min_jerk")
    assert ThrowHarness.accel_step(plan) < 1e-3


def test_const_accel_does_have_an_acceleration_step(lab):
    """Guard against the smoothness test passing vacuously."""
    _, planner, q_rel, qd_rel, _ = lab
    plan = planner.plan(q_rel, qd_rel, throw_shape="const_accel",
                        windup_shape="cubic")
    assert ThrowHarness.accel_step(plan) > 1.0


@pytest.mark.parametrize("name", SHAPES)
def test_release_instant_lands_on_the_control_grid(lab, name):
    """`release_step = int(t_r / dt)` floors, so an unsnapped t_r loses up to a
    full 20 ms step -- and at 4.18 s / 0.02 s it loses one to float error alone.
    """
    _, planner, q_rel, qd_rel, _ = lab
    plan = planner.plan(q_rel, qd_rel, throw_shape=name)
    k = plan.t_r / 0.02
    assert abs(k - round(k)) < 1e-6


@pytest.mark.parametrize("name", SHAPES)
def test_every_phase_passes_its_own_feasibility_check(lab, name):
    _, planner, q_rel, qd_rel, _ = lab
    plan = planner.plan(q_rel, qd_rel, throw_shape=name)
    for phase, chk in plan.checks.items():
        assert chk.ok, f"{name}/{phase}: {chk}"


def test_lab_and_shipped_planners_target_the_same_release_state(lab):
    """Apples-to-apples guard: both planners must aim at the same q/qd.

    If this drifts, every profile comparison in `bench.py` becomes a comparison
    of two different throws rather than two trajectories to the same throw.
    """
    H, planner, q_rel, qd_rel, rp = lab
    v_cmd = H.jacobian(q_rel)[0] @ qd_rel
    coeffs, qr, qdr, _ = H.arm.plan_throw(
        v_cmd, rp, 0.5, 1.6, 2.6,
        q_release_override=q_rel, qd_release_override=qd_rel,
        monotonic_windup=True,
    )
    base = BaselinePlanAdapter(H.arm, coeffs, qr, qdr)
    q_b, qd_b, _, _ = base.eval(base.t_r)
    plan = planner.plan(q_rel, qd_rel, throw_shape="const_accel")
    q_l, qd_l, _, _ = plan.eval(plan.t_r)
    assert np.allclose(q_b, q_l, atol=1e-9)
    assert np.allclose(qd_b, qd_l, atol=1e-9)


def test_harness_releases_at_the_planned_step_and_place(lab):
    H, planner, q_rel, qd_rel, _ = lab
    plan = planner.plan(q_rel, qd_rel, throw_shape="trap_accel")
    res = H.run(plan)
    assert res.released_step == int(plan.t_r / H.dt + 1e-9)
    assert res.release_pos_err < 0.01          # 1 cm
    assert 0.9 < res.speed_ratio < 1.1
    assert np.all(np.isfinite(res.land_xy))


def test_harness_release_bias_shifts_the_release_step(lab):
    H, planner, q_rel, qd_rel, _ = lab
    plan = planner.plan(q_rel, qd_rel, throw_shape="trap_accel")
    base = H.run(plan).released_step
    assert H.run(plan, release_bias_steps=2).released_step == base + 2


def test_zero_release_acceleration_shapes_are_timing_insensitive(lab):
    """The headline hardware property, asserted rather than just measured.

    The real Gen3 releases through a gripper with 67.9 +- 6.4 ms latency on a
    25 ms command quantum, so the release instant is uncertain by roughly one
    50 Hz control step.  A profile still accelerating at release converts that
    directly into a release-speed error; one arriving with qdd = 0 does not.
    """
    H, planner, q_rel, qd_rel, _ = lab
    out = {}
    for name in ("const_accel", "trap_accel"):
        plan = planner.plan(q_rel, qd_rel, throw_shape=name)
        hi = H.run(plan, release_bias_steps=1).speed_release
        lo = H.run(plan, release_bias_steps=-1).speed_release
        out[name] = 0.5 * abs(hi - lo)
    assert out["trap_accel"] < 0.25 * out["const_accel"]


def test_min_feasible_duration_respects_the_bandwidth_floor(lab):
    """Regression: the bisection used to bracket down to 0 whenever the floor
    itself was already feasible, returning dt_throw = 0.019 s -- one control
    step -- whose throw left the ball at 0.3% of planned speed."""
    _, planner, q_rel, qd_rel, _ = lab
    shape = sh.get_velocity_shape("trap_accel", beta=0.25)
    frac = np.linspace(0.0, 0.5, len(qd_rel))
    for floor in (0.4, 0.8):
        dur, chk = dynopt.min_feasible_duration(
            planner.chk, q_rel, qd_rel, frac, shape, lo=floor, n_samples=40
        )
        assert dur is not None
        assert dur >= floor - 1e-9


def test_optimize_allocation_never_returns_worse_than_its_start(lab):
    _, planner, q_rel, qd_rel, _ = lab
    shape = sh.get_velocity_shape("trap_accel", beta=0.25)
    alloc = dynopt.optimize_allocation(
        planner.chk, q_rel, qd_rel, shape, n_samples=32, maxiter=6, min_dur=0.4
    )
    assert alloc["dt_throw"] <= alloc["frac0_dt"] + 1e-9
    assert alloc["check"].ok


def test_feasibility_payload_term_raises_torque(lab):
    """The shipped `_throw_peak_torque_ratio` calls `inverse_dynamics`, which
    sees the arm URDF only -- but `step()` adds J^T m (a_ee - g) to the command
    and clips the SUM.  The shipped check is optimistic by exactly this term."""
    _, planner, q_rel, qd_rel, _ = lab
    from throw_lab.feasibility import FeasibilityChecker

    bare = FeasibilityChecker(planner.arm, payload_mass=0.0)
    laden = FeasibilityChecker(planner.arm, payload_mass=BALL_MASS)
    qdd = np.full(len(qd_rel), 1.0)
    t_bare = bare.torque(q_rel, qd_rel, qdd)
    t_laden = laden.torque(q_rel, qd_rel, qdd)
    assert not np.allclose(t_bare, t_laden)
    assert np.max(np.abs(t_laden - t_bare)) > 1e-3
