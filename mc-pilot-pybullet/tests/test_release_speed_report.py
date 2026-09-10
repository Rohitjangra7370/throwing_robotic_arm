"""
The planner must report the speed it is actually throwing at.

THE DEFECT THESE EXIST TO PREVENT
---------------------------------
`ArmController.plan_throw` computed its returned `v_achieved` from a Jacobian
taken at `localPosition=[0, 0, 0]` -- the bare wrist flange -- while the whole
TCP-offset track (2026-08-27) moved the actual throw point 12 cm out to the
Robotiq 2F-85's tool centre. The wrist is rotating at release, so a point
offset from it picks up an independent `omega x r` term: measured 1.264x on
the shipped `throw_pose_table_tcp.npy` release state.

So `run_hardware_throw.py plan` printed

    policy release speed: 1.424 m/s
    v achievable (after qd clip): |v|=1.126

for a throw that leaves the hand at exactly 1.424 m/s. The label made it read
as a 21% actuator shortfall; there was none, and the joints sat at 0.72 of
qd_max with nothing clipped.

It never changed what the arm executed -- the streamed trajectory comes from
`qd_release`, which was always right -- but `v_achieved` feeds
`HardwareThrowExecutor.precheck(release_speed=...)`, so the command-quantisation
error budget was under-reported by the same 26% (2.8 cm printed against 3.6 cm
real), and any measured-vs-planned release-speed comparison would have been
made against the wrong baseline.

Nothing caught it because `test_dynamic_release.py` allows 50% deviation and
26% fits inside that.
"""
import os
import pickle as pkl
import re

import numpy as np
import pybullet as p
import pytest

import run_hardware_throw as H

CKPT = "results_kinetic_chain_gen3_tcp/1"
TABLE = "throw_pose_table_tcp.npy"
ROBOT = "kinova_gen3_dyn"
TOOL_Z = 0.12

pytestmark = pytest.mark.skipif(
    not (os.path.exists(os.path.join(CKPT, "config_log.pkl")) and os.path.exists(TABLE)),
    reason="TCP-offset checkpoint / pose table not present",
)


@pytest.fixture(scope="module")
def rig():
    arm, profile, cid = H.build_arm(ROBOT)
    pol, cfg = H.load_policy(CKPT, None)
    yield arm, profile, cfg, pol
    p.disconnect(cid)


def _tcp_speed(arm, profile, q, qd, offset_z):
    """|v| at ee_link origin + (0,0,offset_z), straight from PyBullet."""
    n = p.getNumJoints(arm._arm_id, physicsClientId=arm._cid)
    mov = [j for j in range(n)
           if p.getJointInfo(arm._arm_id, j, physicsClientId=arm._cid)[2] != p.JOINT_FIXED]
    cols = [mov.index(j) for j in profile.joint_ids]
    qf = [0.0] * len(mov)
    for li, jid in enumerate(profile.joint_ids):
        qf[mov.index(jid)] = float(q[li])
    jl, _ = p.calculateJacobian(
        arm._arm_id, profile.ee_link, [0.0, 0.0, offset_z], qf,
        [0.0] * len(mov), [0.0] * len(mov), physicsClientId=arm._cid)
    return float(np.linalg.norm(np.array(jl)[:, cols] @ np.asarray(qd, float)))


@pytest.mark.parametrize("target", [(0.67, 0.0), (0.70, 0.0), (0.74, 0.05)])
def test_plan_reports_the_speed_it_actually_throws(rig, target):
    """|v_ach| must equal the policy's commanded release speed, not 0.79x it."""
    arm, profile, cfg, pol = rig
    _, q_rel, qd_rel, v_ach, speed, _, _ = H.plan_throw_for_target(
        arm, profile, cfg, pol, list(target), opt_pose=TABLE, tool_offset_z=TOOL_Z)
    assert np.linalg.norm(v_ach) == pytest.approx(speed, rel=1e-9), (
        "planner reports a release speed that is not the one it commands"
    )
    assert np.linalg.norm(v_ach) == pytest.approx(
        _tcp_speed(arm, profile, q_rel, qd_rel, TOOL_Z), rel=1e-9)


def test_reported_speed_is_the_tcp_not_the_flange(rig):
    """The two differ by the omega x r term; the report must be the TCP one."""
    arm, profile, cfg, pol = rig
    _, q_rel, qd_rel, v_ach, _, _, _ = H.plan_throw_for_target(
        arm, profile, cfg, pol, [0.70, 0.0], opt_pose=TABLE, tool_offset_z=TOOL_Z)
    flange = _tcp_speed(arm, profile, q_rel, qd_rel, 0.0)
    tcp = _tcp_speed(arm, profile, q_rel, qd_rel, TOOL_Z)
    # the effect is real and large on this release state -- if it ever stops
    # being, this test is no longer testing anything
    assert tcp / flange > 1.2
    assert np.linalg.norm(v_ach) == pytest.approx(tcp, rel=1e-9)
    assert np.linalg.norm(v_ach) != pytest.approx(flange, rel=1e-3)


def test_zero_tool_offset_is_bit_identical(rig):
    """
    Default / explicit-zero offset must reproduce the flange behaviour exactly,
    so every legacy checkpoint, table and published number is untouched.
    """
    arm, profile, cfg, pol = rig
    q = np.array([-3.106, -0.436, 0.0, 1.012, 0.0, -0.279, 0.0])
    qd = np.array([0.0, -0.682, 0.0, -0.96, 0.0, -0.84, 0.0])
    v_cmd = np.array([1.374, -0.049, 0.368])
    rel = np.array([0.0, 0.0, 1.14])
    kw = dict(t_w=0.5, t_r=1.6, T=3.2,
              q_release_override=q, qd_release_override=qd, monotonic_windup=True)
    _, _, _, v_default = arm.plan_throw(v_cmd, rel, **kw)
    _, _, _, v_zero = arm.plan_throw(v_cmd, rel, tool_offset=[0.0, 0.0, 0.0], **kw)
    assert np.array_equal(v_default, v_zero)
    assert np.linalg.norm(v_default) == pytest.approx(
        _tcp_speed(arm, profile, q, qd, 0.0), rel=1e-9)


def test_precheck_quantisation_budget_uses_the_real_release_speed(rig):
    """
    The budget printed by precheck scales with release speed, so the flange
    number under-reported it by the same 26%. Guard the consumer, not just the
    producer.
    """
    arm, profile, cfg, pol = rig
    coeffs, _, _, v_ach, speed, _, rel = H.plan_throw_for_target(
        arm, profile, cfg, pol, [0.70, 0.0], opt_pose=TABLE, tool_offset_z=TOOL_Z)
    table = H.load_pose_table(cfg, TABLE)
    box = H.release_box_from_table(arm, table, tool_offset=[0.0, 0.0, TOOL_Z])
    from robot_arm.kinova_hardware import HardwareThrowExecutor
    limits = H.make_limits(profile, 1.0, release_box=box, arm=arm)
    ok, report = HardwareThrowExecutor(limits, dry_run=True).precheck(
        coeffs, arm, release_speed=float(np.linalg.norm(v_ach)))
    assert ok
    # 25 ms at 40 Hz * the true release speed, in cm
    expect_cm = 0.025 * speed * 100.0
    m = re.search(r"([0-9.]+)\s*cm of undershoot", report)
    assert m, f"no quantisation figure in precheck report:\n{report}"
    assert float(m.group(1)) == pytest.approx(expect_cm, abs=0.05)
