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
    arm, profile, cid = H.build_arm(ROBOT)
    try:
        coeffs, _, _, _, _, _, rel = H.plan_throw_for_target(
            arm, profile, cfg, _FixedPolicy(1.43), (0.72, 0.0), opt_pose=TABLE
        )
        ex = HardwareThrowExecutor(_limits(profile), dry_run=True)
        ok, report = ex.precheck(coeffs, arm)
        assert ok, report
        assert "peak |tau|" in report, "torque must actually be reported"
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
