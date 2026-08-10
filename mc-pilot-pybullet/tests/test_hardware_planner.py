"""
Regression tests for the HARDWARE planning path.

The defect these exist to prevent: `run_hardware_throw.py` used to plan its own
throw with IK + pinv and `profile.timing`, producing a completely different
(weaker, near-horizontal, differently-timed) motion from the one the simulator
was trained and validated on -- while printing "PRECHECK: PASS" the whole way.
Nothing in the suite caught it because nothing compared the two planners.

Test 1 is the important one: sim and hardware must plan the SAME throw.
"""
import os
import pickle as pkl

import numpy as np
import pybullet as p
import pybullet_data
import pytest
import torch

import run_hardware_throw as H
from robot_arm.arm_controller import ArmController
from robot_arm.kinova_hardware import HardwareThrowExecutor, SafetyLimits
from robot_arm.robot_profiles import get_robot_profile
from simulation_class.model_pybullet import PyBulletThrowingSystem

CKPT = "results_kinetic_chain_gen3/1"
TABLE = "throw_pose_table.npy"
ROBOT = "kinova_gen3_dyn"

pytestmark = pytest.mark.skipif(
    not (os.path.exists(os.path.join(CKPT, "config_log.pkl")) and os.path.exists(TABLE)),
    reason="trained overhead checkpoint / pose table not present",
)


class _FixedPolicy:
    """Stands in for the trained RBF so the comparison isolates the planner."""

    def __init__(self, speed):
        self.speed = speed

    def __call__(self, s, t=0):
        return torch.tensor([[self.speed]], dtype=torch.float64)


@pytest.fixture(scope="module")
def cfg():
    with open(os.path.join(CKPT, "config_log.pkl"), "rb") as f:
        return pkl.load(f)


@pytest.fixture(scope="module")
def table():
    return list(np.load(TABLE, allow_pickle=True))


def _sim_plan(cfg, table, target_xy, speed):
    """What PyBulletThrowingSystem._simulate_pybullet would build."""
    profile = get_robot_profile(ROBOT)
    cid = p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
        arm = ArmController(
            cid, pybullet_data.getDataPath() + "/" + profile.urdf_rel_path,
            robot_name=ROBOT,
        )
        arm.reset()
        sysm = PyBulletThrowingSystem(
            mass=cfg["ball_mass"], radius=cfg["ball_radius"],
            launch_angle_deg=table[0]["elev_deg"],
            t_w=cfg["T_W"], t_r=cfg["T_R"], robot_name=cfg["robot_name"],
            opt_posture_table=table, opt_launch_deg=table[0]["elev_deg"],
        )
        sysm._cur_target_xy = np.asarray(target_xy, float)
        v0 = sysm._speed_to_velocity(
            speed, np.asarray(cfg["release_pos"], float), np.asarray(target_xy, float)
        )
        rel, q_ovr, qd_ovr, v_cmd = sysm._optimized_release(arm, v0)
        t_arm = max(1.20, profile.timing[2], cfg["T"] + cfg["T_R"])
        coeffs, _, _, v_ach = arm.plan_throw(
            v_cmd, rel, cfg["T_W"], cfg["T_R"], t_arm,
            q_release_override=q_ovr, qd_release_override=qd_ovr,
            monotonic_windup=True,
        )
        return rel, q_ovr, qd_ovr, v_ach, coeffs
    finally:
        p.disconnect(cid)


def _hw_plan(cfg, target_xy, speed):
    arm, profile, cid = H.build_arm(ROBOT)
    try:
        coeffs, q, qd, v_ach, _, _, rel = H.plan_throw_for_target(
            arm, profile, cfg, _FixedPolicy(speed), target_xy, opt_pose=TABLE
        )
        return rel, q, qd, v_ach, coeffs
    finally:
        p.disconnect(cid)


@pytest.mark.parametrize(
    "target", [(0.72, 0.00), (0.72, 0.05), (0.65, -0.20), (0.78, 0.25), (0.62, 0.30)]
)
def test_hardware_planner_matches_sim(cfg, table, target):
    """The arm must execute the throw the simulator was validated on."""
    speed = 1.43
    rel_s, q_s, qd_s, v_s, co_s = _sim_plan(cfg, table, target, speed)
    rel_h, q_h, qd_h, v_h, co_h = _hw_plan(cfg, target, speed)

    assert np.allclose(rel_s, rel_h, atol=1e-12), "release position differs"
    assert np.allclose(q_s, q_h, atol=1e-12), "release joint configuration differs"
    assert np.allclose(qd_s, qd_h, atol=1e-12), "release joint velocity differs"
    assert np.allclose(v_s, v_h, atol=1e-9), "achieved release velocity differs"
    for key in ("t_w", "t_r", "T"):
        assert co_s[key] == pytest.approx(co_h[key], abs=1e-12), f"{key} differs"
    for key in ("windup", "throw", "follow"):
        assert np.allclose(co_s[key], co_h[key], atol=1e-12), f"{key} cubic differs"


def test_hardware_plan_is_the_overhead_throw(cfg, table):
    """
    Guards the specific regression: an IK+pinv fallback plans a LOW throw with
    every joint moving. The overhead throw releases high, above the base, with
    velocity only on the pitch joints (1, 3, 5).
    """
    rel, q, qd, v_ach, _ = _hw_plan(cfg, (0.72, 0.0), 1.43)
    assert rel[2] > 1.0, f"release height {rel[2]:.3f} m is not an overhead release"
    assert np.hypot(rel[0], rel[1]) < 0.20, "release should sit near the base axis"
    roll_idx = [0, 2, 4, 6]
    assert np.allclose(np.asarray(qd)[roll_idx], 0.0, atol=1e-9), (
        "roll/twist joints must be frozen during the throw (corkscrew regression)"
    )
    assert np.any(np.abs(np.asarray(qd)[[1, 3, 5]]) > 0.1), "pitch joints carry no velocity"


def test_hardware_uses_trained_timings_not_profile_defaults(cfg):
    """
    `profile.timing` (0.4/0.8) disagrees with what this policy trained on
    (T_W=0.5, T_R=1.6). Planning with the profile default roughly doubles the
    commanded peak joint velocity.
    """
    profile = get_robot_profile(ROBOT)
    assert profile.timing[0] != cfg["T_W"] or profile.timing[1] != cfg["T_R"], (
        "fixture no longer exercises the mismatch; pick a config that differs"
    )
    _, _, _, _, coeffs = _hw_plan(cfg, (0.72, 0.0), 1.43)
    # plan_throw time-stretches the WINDUP for torque feasibility but leaves the
    # throw phase itself alone, so its duration is the invariant that identifies
    # which timings were used: trained 1.6-0.5 = 1.1 s, profile default
    # 0.8-0.4 = 0.4 s. A 0.4 s throw phase means the profile default leaked back in.
    throw_duration = coeffs["t_r"] - coeffs["t_w"]
    assert throw_duration == pytest.approx(cfg["T_R"] - cfg["T_W"], rel=1e-9)
    assert throw_duration != pytest.approx(profile.timing[1] - profile.timing[0])


def _limits(profile, **over):
    qd_max = np.array(profile.qd_max, float)
    q_soft = 6.10 * np.ones(len(qd_max))
    kw = dict(
        qd_max=qd_max, q_soft_lo=-q_soft, q_soft_hi=q_soft,
        speed_scale=1.0, control_hz=1000.0, max_traj_seconds=180.0,
        tau_max=np.array(profile.tau_max, float),
    )
    kw.update(over)
    return SafetyLimits(**kw)


def test_precheck_passes_on_the_real_plan(cfg):
    """
    Guards TWO bugs that between them made this check meaningless, both found by
    cross-checking the model against the connected arm (2026-08-07):

    1. build_arm() never enabled gravity, so the torque check saw inertial terms
       only and reported 22% of limit where the truth was 94.2%.
    2. The shipped gen3.urdf declares its three camera frames as empty
       self-closing tags, so PyBullet gave each 1 kg and baked 3 kg of phantom
       mass into the wrist -- inflating torque ~2.3x the other way.

    The two errors pointed in opposite directions and partially cancelled, which
    is why nothing looked wrong. With both fixed the plan sits at 41.4%, and the
    model agrees with the arm's own torque sensors to within 17-27% instead of
    being 2.1-2.4x off.
    """
    arm, profile, cid = H.build_arm(ROBOT)
    try:
        coeffs, _, _, _, _, _, rel = H.plan_throw_for_target(
            arm, profile, cfg, _FixedPolicy(1.43), (0.72, 0.0), opt_pose=TABLE
        )
        ex = HardwareThrowExecutor(_limits(profile), dry_run=True)
        ok, report = ex.precheck(coeffs, arm)
        assert ok, report
        assert "peak |tau|" in report, "torque must actually be reported"
        # and the torque must be REAL -- gravity on, phantom mass off
        tau = np.asarray(arm.inverse_dynamics(
            np.array(profile.q_neutral, float) + np.array([0, 1.2, 0, 1.2, 0, 0, 0]),
            np.zeros(7), np.zeros(7)), float)
        assert np.max(np.abs(tau)) > 5.0, "gravity appears to be off again"
    finally:
        p.disconnect(cid)

def test_precheck_fails_when_velocity_would_be_clamped(cfg):
    """
    Clamping keeps the ARM safe but silently slows the THROW -- the ball lands
    short with nothing in the logs. Must fail closed, not silently correct.
    """
    arm, profile, cid = H.build_arm(ROBOT)
    try:
        coeffs, _, _, _, _, _, _ = H.plan_throw_for_target(
            arm, profile, cfg, _FixedPolicy(1.43), (0.72, 0.0), opt_pose=TABLE
        )
        tight = _limits(profile, qd_max=0.01 * np.array(profile.qd_max, float))
        ex = HardwareThrowExecutor(tight, dry_run=True)
        ok, report = ex.precheck(coeffs, arm)
        assert not ok, "precheck accepted a plan that needs velocity clamping"
        assert "CLAMPING" in report
    finally:
        p.disconnect(cid)


def test_precheck_fails_on_torque_violation(cfg):
    arm, profile, cid = H.build_arm(ROBOT)
    try:
        coeffs, _, _, _, _, _, _ = H.plan_throw_for_target(
            arm, profile, cfg, _FixedPolicy(1.43), (0.72, 0.0), opt_pose=TABLE
        )
        weak = _limits(profile, tau_max=0.05 * np.array(profile.tau_max, float))
        ex = HardwareThrowExecutor(weak, dry_run=True)
        ok, report = ex.precheck(coeffs, arm)
        assert not ok, "precheck accepted a torque-infeasible plan"
        assert "TORQUE" in report
    finally:
        p.disconnect(cid)


def test_release_box_derived_from_table_contains_the_release(cfg, table):
    """
    The old hardcoded box (z <= 0.9 m) excluded the overhead release at
    z ~ 1.137 m, so `throw` refused every valid plan.
    """
    arm, profile, cid = H.build_arm(ROBOT)
    try:
        box = H.release_box_from_table(arm, table)
        limits = _limits(profile, release_box_lo=box[0], release_box_hi=box[1])
        ex = HardwareThrowExecutor(limits, dry_run=True)
        _, _, _, _, _, _, rel = H.plan_throw_for_target(
            arm, profile, cfg, _FixedPolicy(1.43), (0.72, 0.0), opt_pose=TABLE
        )
        assert ex.check_release_pos(rel), f"release {rel} outside derived box {box}"
        assert not ex.check_release_pos(rel + np.array([0.0, 0.0, 0.5])), (
            "derived box is too loose to catch a displaced release"
        )
    finally:
        p.disconnect(cid)


def test_kortex_api_imports_on_this_python():
    """
    kortex_api 2.6.0 pins protobuf 3.5.1, which uses collections.MutableMapping
    -- removed in Python 3.10. Without the shim in kinova_hardware, connecting
    to the real arm dies at import time, at the bench, with the arm powered.
    """
    pytest.importorskip("kortex_api", reason="kortex_api not installed")
    from robot_arm.kinova_hardware import _patch_collections_abc
    _patch_collections_abc()
    from kortex_api.autogen.messages import Base_pb2, Session_pb2
    from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
    from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
    from kortex_api.SessionManager import SessionManager
    for obj, name in (
        (Session_pb2, "CreateSessionInfo"), (Base_pb2, "JointSpeeds"),
        (Base_pb2, "GripperCommand"), (Base_pb2, "GRIPPER_POSITION"),
        (BaseClient, "SendJointSpeedsCommand"), (BaseClient, "SendGripperCommand"),
        (BaseCyclicClient, "RefreshFeedback"),
        (SessionManager, "CreateSession"), (SessionManager, "CloseSession"),
    ):
        assert hasattr(obj, name), f"{name} missing from installed kortex_api"


# --------------------------------------------------------------------------- #
# Homing / joint-angle convention.
#
# These encode a MEASUREMENT, not a guess: the readback below is the real one
# from the lab Gen3 (L53K, SN WO545410-1) on 2026-08-07, taken by
# hw_readonly_check.py with the arm powered and stationary. Kortex reported
# every joint in [0, 360) -- including LIMITED joints, one of which came back
# at 247.37 deg (4.318 rad) against a +-2.57 rad limit.
# --------------------------------------------------------------------------- #

MEASURED_POS_DEG = np.array([355.19, 26.37, 182.60, 247.37, 0.62, 52.21, 89.07])
# physical angles the arm was actually at, i.e. MEASURED_POS_DEG wrapped
MEASURED_Q_RAD = np.array([-0.084, 0.460, -3.096, -1.966, 0.011, 0.911, 1.555])


class _StubBackend:
    """Minimal backend replaying a fixed joint readback. Records all commands."""

    def __init__(self, q_read):
        self.q_read = np.asarray(q_read, dtype=float)
        self.velocity_commands = []

    def connect(self): pass
    def disconnect(self): pass
    def stop(self): self.velocity_commands.append(np.zeros_like(self.q_read))
    def send_gripper(self, pos): pass
    def read_joint_state(self): return self.q_read.copy(), np.zeros_like(self.q_read)
    def send_joint_velocities(self, qd): self.velocity_commands.append(np.asarray(qd, float))


class _StubArm:
    """Carries only the joint ranges `home()` needs (real Gen3 URDF values)."""
    _q_lo = np.array([-6.28, -2.24, -6.28, -2.57, -6.28, -2.09, -6.28])
    _q_hi = -_q_lo


def _executor_with_readback(q_read):
    profile = get_robot_profile(ROBOT)
    ex = HardwareThrowExecutor(H.make_limits(profile, 1.0), dry_run=True)
    ex.dry_run = False                      # take the real read_joint_state path
    ex.backend = _StubBackend(q_read)
    return ex, np.asarray(profile.q_neutral, dtype=float)


def test_kortex_degrees_wrap_into_the_urdf_range():
    """
    read_joint_state() must wrap [0,360) reporting into (-pi, pi].

    Without this, joint 3 arrives as 4.318 rad against a +-2.57 rad limit -- a
    value no downstream check can interpret, and one that made the naive P-servo
    drive the joint the long way for the whole homing window.
    """
    q = np.arctan2(np.sin(np.deg2rad(MEASURED_POS_DEG)),
                   np.cos(np.deg2rad(MEASURED_POS_DEG)))
    assert np.allclose(q, MEASURED_Q_RAD, atol=1e-3)
    assert np.all(np.abs(q) <= np.pi + 1e-9)
    arm = _StubArm()
    assert np.all(q >= arm._q_lo) and np.all(q <= arm._q_hi), (
        "wrapped readback must lie inside every joint's URDF range"
    )


def test_home_refuses_an_unwrapped_readback():
    """The un-wrapped [0,360) reading must be refused, not serviced."""
    ex, q_neutral = _executor_with_readback(np.deg2rad(MEASURED_POS_DEG))
    with pytest.raises(RuntimeError, match="WRAP/UNIT"):
        ex.home(_StubArm(), q_neutral, duration=0.05)
    assert all(np.allclose(v, 0.0) for v in ex.backend.velocity_commands), (
        "a refused homing must not have commanded any motion"
    )


def test_homing_error_takes_the_shortest_path_on_continuous_joints():
    """
    Joint 2 is continuous and sat at -177.4 deg with neutral at -2.3 deg. The
    direct difference (+175.1 deg) is already the shortest path here; the guard
    is that a continuous joint's error can never exceed pi.
    """
    err, cont = HardwareThrowExecutor._homing_error(
        MEASURED_Q_RAD, get_robot_profile(ROBOT).q_neutral,
        _StubArm._q_lo, _StubArm._q_hi)
    assert list(np.where(cont)[0]) == [0, 2, 4, 6]
    assert np.all(np.abs(err[cont]) <= np.pi + 1e-9)
    assert err[2] == pytest.approx(3.056, abs=1e-3)


def test_homing_error_is_direct_on_limited_joints():
    """
    Joint 3 is LIMITED (+-2.57) and needs a legal 3.426 rad sweep through zero.
    Wrapping that to -2.857 would drive it into its own limit -- the exact
    inverse of the continuous-joint fix, which is why one pi threshold for all
    joints was wrong.
    """
    err, cont = HardwareThrowExecutor._homing_error(
        MEASURED_Q_RAD, get_robot_profile(ROBOT).q_neutral,
        _StubArm._q_lo, _StubArm._q_hi)
    assert not cont[3]
    assert err[3] == pytest.approx(3.426, abs=1e-3)
    assert err[3] > np.pi, "regression: limited-joint error was wrapped"


def test_home_extends_a_duration_too_short_to_arrive(capsys):
    """
    3.426 rad at the 0.349 rad/s cap needs ~9.8 s. The 4.0 s default would stop
    the arm part-way and leave an undefined pose as the start of a throw.
    """
    ex, q_neutral = _executor_with_readback(MEASURED_Q_RAD)
    ex.limits.control_hz = 200.0            # keep the test quick
    ex.home(_StubArm(), q_neutral, duration=0.02)
    out = capsys.readouterr().out
    assert "extending duration" in out
    assert ex.backend.velocity_commands, "expected homing to command motion"


def test_home_runs_normally_when_already_near_neutral():
    """The guards must not fire on an ordinary short homing move."""
    profile = get_robot_profile(ROBOT)
    ex, q_neutral = _executor_with_readback(np.asarray(profile.q_neutral, float))
    ex.home(_StubArm(), q_neutral, duration=0.05)
    assert all(np.allclose(v, 0.0) for v in ex.backend.velocity_commands), (
        "already at neutral -> no motion needed"
    )


def test_control_rate_is_clamped_to_the_high_level_ceiling():
    """
    Kinova's own driver docs: "The base high level commands are treated every
    25 ms inside the robot. High level control cannot be achieved at a rate
    faster than 40 Hz for now."

    This executor is high-level (Base.SendJointSpeedsCommand, arm in
    SINGLE_LEVEL_SERVOING -- read back from the lab arm). The old 1000.0 default
    was justified by Kinova's 1 kHz figure, which belongs to LOW_LEVEL_SERVOING.
    The consequence was not a hazard but a false measurement: the executor logged
    "1000 Hz achieved" while the arm consumed 40 commands a second.
    """
    from robot_arm.kinova_hardware import HIGH_LEVEL_MAX_HZ
    assert HIGH_LEVEL_MAX_HZ == 40.0
    profile = get_robot_profile(ROBOT)
    assert H.make_limits(profile, 1.0).control_hz == HIGH_LEVEL_MAX_HZ
    lim = SafetyLimits(qd_max=np.array(profile.qd_max, float),
                       q_soft_lo=-6.1 * np.ones(7), q_soft_hi=6.1 * np.ones(7),
                       control_hz=1000.0)
    assert lim.control_hz == HIGH_LEVEL_MAX_HZ, "1 kHz must be clamped, not honoured"


def test_precheck_reports_release_quantisation_as_a_landing_error():
    """
    25 ms of release quantisation at 1.5 m/s is ~3.7 cm of undershoot -- larger
    than the entire 2.89 cm sim accuracy, and NOT reducible by looping faster.
    It has to appear in the precheck, not in a footnote.
    """
    cfg_path = os.path.join(CKPT, "config_log.pkl")
    with open(cfg_path, "rb") as f:
        cfg = pkl.load(f)
    arm, profile, cid = H.build_arm(ROBOT)
    try:
        pol = _FixedPolicy(1.5)
        coeffs, _, _, v_ach, _, _, _ = H.plan_throw_for_target(
            arm, profile, cfg, pol, (0.75, 0.05), opt_pose=TABLE)
        ex = HardwareThrowExecutor(H.make_limits(profile, 1.0), dry_run=True)
        _, report = ex.precheck(coeffs, arm, release_speed=np.linalg.norm(v_ach))
        assert "release quantisation" in report
        assert "25.0 ms" in report
        assert "cm of undershoot" in report
    finally:
        p.disconnect(cid)


def test_dry_run_backend_supports_the_realtime_feedback_api():
    """
    The 1 kHz UDP feedback path must be exercisable with no arm, or the latency
    tool can only ever be tested on hardware.

    Commands are capped at 40 Hz but FEEDBACK is not -- measured 1122 Hz on the
    lab arm (p99 gap 1.5 ms). That asymmetry is the only reason gripper release
    latency, now the dominant error term, is measurable without a high-speed
    camera.
    """
    from robot_arm.kinova_hardware import _DryRunBackend
    be = _DryRunBackend(7)
    be.connect()
    be.open_realtime_feedback()
    be.send_gripper(1.0)
    pos, vel = be.read_gripper()
    assert pos == 100.0 and vel == 0.0, "dry-run gripper readback must be exact"
    be.send_gripper(0.0)
    assert be.read_gripper()[0] == 0.0
    be.close_realtime_feedback()
    be.disconnect()


def test_gripper_lead_compensates_measured_latency_in_wall_clock():
    """
    The gripper OPEN command must fire early enough that the FINGERS move at
    t_r, and the lead is a wall-clock delay -- so in trajectory time it scales
    with speed_scale.

    Getting the scaling backwards is the dangerous direction: dividing instead
    of multiplying would over-lead the 0.15 rehearsal by 1/0.15 = 6.7x and drop
    the ball 0.45 s before the swing even reaches release.
    """
    from robot_arm.kinova_hardware import GRIPPER_RELEASE_LATENCY_S
    assert GRIPPER_RELEASE_LATENCY_S == pytest.approx(0.0679, abs=1e-4)
    profile = get_robot_profile(ROBOT)
    for scale in (1.0, 0.15):
        lim = H.make_limits(profile, scale)
        assert lim.gripper_lead_s == pytest.approx(GRIPPER_RELEASE_LATENCY_S)
        t_r = 4.928
        s_fire = t_r - lim.gripper_lead_s * scale
        # the WALL time between firing and the intended release is the latency,
        # independent of speed_scale -- that is the whole point
        wall_lead = (t_r - s_fire) / scale
        assert wall_lead == pytest.approx(GRIPPER_RELEASE_LATENCY_S, abs=1e-9)


def test_uncompensated_gripper_latency_is_the_dominant_error():
    """
    Records why the compensation exists: uncompensated, the measured 67.9 ms is
    10.2 cm at the 1.498 m/s release speed -- 3.5x the 2.89 cm sim accuracy, and
    bigger than the 25 ms command-quantisation term it sits on top of.
    """
    from robot_arm.kinova_hardware import GRIPPER_RELEASE_LATENCY_S, HIGH_LEVEL_MAX_HZ
    v = 1.498
    gripper_cm = GRIPPER_RELEASE_LATENCY_S * v * 100
    quant_cm = (1.0 / HIGH_LEVEL_MAX_HZ) * v * 100
    assert gripper_cm == pytest.approx(10.2, abs=0.2)
    assert quant_cm == pytest.approx(3.7, abs=0.2)
    assert gripper_cm > quant_cm, "gripper latency dominates quantisation"
    residual_cm = 0.0064 * v * 100      # measured 6.4 ms jitter, uncompensable
    assert residual_cm < 2.89, "compensated residual must fit inside sim accuracy"


def test_gripper_lead_can_be_disabled_for_an_uncompensated_baseline():
    profile = get_robot_profile(ROBOT)
    lim = H.make_limits(profile, 1.0)
    lim.gripper_lead_s = 0.0
    assert lim.gripper_lead_s == 0.0


def test_readback_guard_separates_wrap_errors_from_out_of_model_poses():
    """
    The two failure causes need different fixes, so they must not share a
    message. The first version of this guard called BOTH a "WRAP/UNIT mismatch",
    which would have sent someone editing correct conversion code to chase a
    pose problem.
    """
    guard = HardwareThrowExecutor._assert_readback_sane
    lo, hi = _StubArm._q_lo, _StubArm._q_hi

    # (a) past pi -> can only be unwrapped [0,360) reporting
    q = np.zeros(7); q[3] = 4.318          # the real 2026-08-07 reading, unwrapped
    with pytest.raises(RuntimeError, match="WRAP/UNIT"):
        guard(q, lo, hi)

    # (b) well-formed angle, outside the model -> NOT a units bug
    q = np.zeros(7); q[3] = -2.656         # the real reading, correctly wrapped
    assert abs(q[3]) <= np.pi, "fixture must be a well-formed angle"
    with pytest.raises(RuntimeError, match="OUTSIDE THE KINEMATIC MODEL") as e:
        guard(q, lo, hi)
    assert "WRAP/UNIT" not in str(e.value), "must not blame the unit conversion"
    assert "jog it back" in str(e.value)

    # a normal pose passes
    guard(np.array([-0.084, 0.460, -3.096, -1.966, 0.011, 0.911, 1.555]), lo, hi)


def test_soft_envelope_uses_real_per_joint_limits_not_a_flat_value():
    """
    The flat +-6.10 envelope was 2.9x too loose on joint 5 (real range +-2.09),
    so the precheck could have passed a trajectory driving a limited joint far
    past its stop. The shipped throw peaks at 57% of the real ranges, so this
    was latent -- but this check exists to catch a BAD plan, not the good one.
    """
    arm, profile, cid = H.build_arm(ROBOT)
    try:
        lim = H.make_limits(profile, 1.0, arm=arm)
        for j in (1, 3, 5):                       # the limited joints
            assert lim.q_soft_hi[j] < 2.6, f"joint {j} envelope still too loose"
            assert lim.q_soft_hi[j] < arm._q_hi[j], "envelope must inset the URDF limit"
            assert lim.q_soft_lo[j] > arm._q_lo[j]
        for j in (0, 2, 4, 6):                    # continuous joints stay wide
            assert lim.q_soft_hi[j] > 6.0
        # without an arm the flat fallback is kept (no trajectory is checked there)
        assert H.make_limits(profile, 1.0).q_soft_hi[5] == pytest.approx(6.10)
    finally:
        p.disconnect(cid)


def test_shipped_throw_still_passes_the_tightened_envelope():
    """The tightened envelope must not reject the plan we intend to run."""
    with open(os.path.join(CKPT, "config_log.pkl"), "rb") as f:
        cfg = pkl.load(f)
    arm, profile, cid = H.build_arm(ROBOT)
    try:
        coeffs, _, _, v_ach, _, _, _ = H.plan_throw_for_target(
            arm, profile, cfg, _FixedPolicy(1.5), (0.75, 0.05), opt_pose=TABLE)
        ex = HardwareThrowExecutor(H.make_limits(profile, 1.0, arm=arm), dry_run=True)
        ok, report = ex.precheck(coeffs, arm, release_speed=np.linalg.norm(v_ach))
        # POSITION is what this test is about. Torque is separately blocked by
        # the phantom-mass issue (see test_precheck_now_refuses_the_shipped_plan
        # _on_torque), so assert no POSITION violation rather than overall pass.
        assert "outside" not in report, (
            f"tightened envelope rejected the throw on JOINT POSITION:\n{report}")
    finally:
        p.disconnect(cid)


def test_hardware_client_has_gravity_enabled():
    """
    The bug this prevents was silent and shipped: build_arm() never called
    setGravity, so PyBullet's client defaulted to zero gravity and
    calculateInverseDynamics returned INERTIAL torque only -- no error, just
    plausible-looking numbers gating motion on a real arm.

    Measured cost on the shipped throw: precheck reported 8.6 Nm (22% of limit)
    where the same trajectory with gravity needs 36.7 Nm (94.2%). A 4.3x
    under-report on the worst joint, and the difference between "PASS" and
    "refuse to move".

    Asserted behaviourally rather than by inspecting the call: a stationary arm
    in a non-singular pose MUST need non-zero torque to hold itself up.
    """
    arm, profile, cid = H.build_arm(ROBOT)
    try:
        grav = p.getPhysicsEngineParameters(physicsClientId=cid)["gravityAccelerationZ"]
        assert grav == pytest.approx(-9.81), f"hardware client gravity is {grav}"
        # shoulder out horizontally -> large, unambiguous holding torque
        q = np.zeros(7); q[1] = 1.2; q[3] = 1.2
        tau = np.asarray(arm.inverse_dynamics(q, np.zeros(7), np.zeros(7)), float)
        assert np.max(np.abs(tau)) > 5.0, (
            f"static holding torque {np.round(tau,2)} is implausibly small -- "
            "gravity is almost certainly off")
    finally:
        p.disconnect(cid)


def test_sim_and_hardware_clients_agree_on_gravity():
    """
    Sim and hardware must score the SAME trajectory identically. They already
    share the release solver and the planner; a physics-parameter difference
    between the two clients reintroduces the divergence by the back door.
    """
    arm, profile, cid = H.build_arm(ROBOT)
    try:
        hw = p.getPhysicsEngineParameters(physicsClientId=cid)["gravityAccelerationZ"]
    finally:
        p.disconnect(cid)
    sim_cid = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -9.81, physicsClientId=sim_cid)   # as model_pybullet.py:192
        sim = p.getPhysicsEngineParameters(physicsClientId=sim_cid)["gravityAccelerationZ"]
    finally:
        p.disconnect(sim_cid)
    assert hw == pytest.approx(sim), f"hardware {hw} vs sim {sim}"
