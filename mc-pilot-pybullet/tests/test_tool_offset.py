"""
Regression tests for the gripper TCP-offset fix.

The defect: `OptimizedReleaseSolver` and `find_throw_pose.py`'s search have
always solved release position/velocity at `ee_link` (the bare wrist flange,
zero-length tool). The real Robotiq 2F-85 firmware reports
`tool_transform = (0, 0, 0.12) m` -- the ball actually leaves 12 cm further
out, rigidly attached to the flange. Uncorrected this is 0.39 m/s / 12 cm of
release error (measured 2026-08-22, see CLAUDE.md) -- bigger than every other
error term in the budget combined.

These tests hold the fix to the same standard as `test_hardware_planner.py`:
default (zero offset) behavior must be provably unchanged (existing tests are
the regression guard for that), and a nonzero offset must provably move the
solved release state to the TCP, not leave it at the flange.
"""
import numpy as np
import pybullet as p
import pybullet_data
import pytest

import run_hardware_throw as H
import find_throw_pose as FTP
from simulation_class.release_solver import OptimizedReleaseSolver
from simulation_class.model_pybullet import PyBulletThrowingSystem

ROBOT = "kinova_gen3_dyn"
TOOL_OFFSET = np.array([0.0, 0.0, 0.12])


def _tcp_world(arm, q_full, tool_offset):
    """Independent oracle: TCP world position via FK link pose, NOT via the
    solver under test."""
    for j in range(arm._n_dofs):
        p.resetJointState(arm._arm_id, j, q_full[j], physicsClientId=arm._cid)
    ls = p.getLinkState(arm._arm_id, arm._ee_link, computeForwardKinematics=True,
                        physicsClientId=arm._cid)
    link_pos, link_orn = ls[4], ls[5]
    world_pos, _ = p.multiplyTransforms(link_pos, link_orn, tool_offset.tolist(),
                                        [0, 0, 0, 1], physicsClientId=arm._cid)
    return np.array(world_pos)


def _tcp_jacobian(arm, q_full, tool_offset):
    """Independent oracle: linear Jacobian AT the TCP point, via PyBullet's own
    localPosition argument -- NOT via the solver under test."""
    jl, _ = p.calculateJacobian(
        arm._arm_id, arm._ee_link, tool_offset.tolist(), q_full.tolist(),
        [0.0] * arm._n_dofs, [0.0] * arm._n_dofs, physicsClientId=arm._cid,
    )
    return np.array(jl)[:, arm._dof_ids]


@pytest.fixture
def arm():
    a, profile, cid = H.build_arm(ROBOT)
    yield a
    p.disconnect(cid)


def _table_entry(profile):
    q = np.array(profile.q_neutral, dtype=float)
    return [{
        "azimuth_deg": 0.0, "elev_deg": 45.0,
        "q": q.tolist(), "qd": [0.0, 0.3, 0.0, -0.2, 0.0, 0.4, 0.0],
        "v_dir": [1.0, 0.0, 0.3], "speed": 1.5,
        "rotation_built": True,
    }]


def test_default_tool_offset_matches_flange_position(arm):
    """No tool_offset given -> release_pos is the bare flange point (pre-fix
    behavior, must not silently change)."""
    from robot_arm.robot_profiles import get_robot_profile
    profile = get_robot_profile(ROBOT)
    table = _table_entry(profile)
    solver = OptimizedReleaseSolver(opt_posture_table=table)
    rel, q_rel, qd_rel, v = solver.solve(arm, np.array([1.0, 0.0, 1.0]), target_xy=(1.0, 0.0))

    q_full = arm._ik_q_neutral.copy()
    for li, dof in enumerate(arm._dof_ids):
        q_full[dof] = q_rel[li]
    expected = _tcp_world(arm, q_full, np.zeros(3))
    assert np.allclose(rel, expected, atol=1e-9)


def test_tool_offset_moves_release_pos_to_the_tcp(arm):
    """With tool_offset=(0,0,0.12), release_pos must be the TCP point, not the
    flange -- the 12 cm this whole fix exists to correct."""
    from robot_arm.robot_profiles import get_robot_profile
    profile = get_robot_profile(ROBOT)
    table = _table_entry(profile)
    solver = OptimizedReleaseSolver(opt_posture_table=table, tool_offset=TOOL_OFFSET)
    rel, q_rel, qd_rel, v = solver.solve(arm, np.array([1.0, 0.0, 1.0]), target_xy=(1.0, 0.0))

    q_full = arm._ik_q_neutral.copy()
    for li, dof in enumerate(arm._dof_ids):
        q_full[dof] = q_rel[li]
    expected_tcp = _tcp_world(arm, q_full, TOOL_OFFSET)
    expected_flange = _tcp_world(arm, q_full, np.zeros(3))

    assert np.allclose(rel, expected_tcp, atol=1e-9)
    assert not np.allclose(rel, expected_flange, atol=1e-3), (
        "release_pos did not move off the flange -- offset was not applied"
    )
    assert np.linalg.norm(rel - expected_flange) > 0.05, (
        "release_pos moved by less than the 12 cm this fix corrects for"
    )


def test_legacy_posture_lp_solves_velocity_at_the_tcp_not_the_flange(arm):
    """The general LP path (legacy single-posture mode) must size q_dot so the
    TCP -- not the flange -- achieves the commanded speed along the launch
    direction. This is the actual physical fix: the ball leaves from the TCP."""
    from robot_arm.robot_profiles import get_robot_profile
    profile = get_robot_profile(ROBOT)
    q_posture = np.array(profile.q_neutral, dtype=float)
    # Deliberately tiny: below whatever the LP's achievable v_max turns out to
    # be at this posture/direction (whatever that is), so `scale = speed/v_max
    # < 1` and the achieved velocity is exactly `d * speed_cmd` by
    # construction -- isolating "does the LP solve at the TCP" from "how much
    # headroom does this particular posture happen to have".
    speed_cmd = 0.005
    v_cmd = np.array([speed_cmd, 0.0, 0.0])

    solver = OptimizedReleaseSolver(opt_posture=q_posture, opt_launch_deg=0.0,
                                    tool_offset=TOOL_OFFSET)
    rel, q_rel, qd_rel, v_release = solver.solve(arm, v_cmd, target_xy=None)

    q_full = arm._ik_q_neutral.copy()
    for li, dof in enumerate(arm._dof_ids):
        q_full[dof] = q_rel[li]
    J_tcp = _tcp_jacobian(arm, q_full, TOOL_OFFSET)
    v_tcp_achieved = J_tcp @ np.asarray(qd_rel, dtype=float)

    assert np.linalg.norm(v_tcp_achieved) == pytest.approx(
        np.linalg.norm(v_release), abs=1e-6
    ), "solved qd does not deliver the commanded speed at the TCP"

    J_flange = _tcp_jacobian(arm, q_full, np.zeros(3))
    v_flange_achieved = J_flange @ np.asarray(qd_rel, dtype=float)
    assert not np.allclose(v_flange_achieved, v_tcp_achieved, atol=1e-6), (
        "TCP and flange velocities coincide -- offset had no effect on the LP"
    )


# --------------------------------------------------------------------------- #
# find_throw_pose.py's search-time FK/Jacobian helper (_fkj) -- must apply the
# same tool_offset, since the search optimizes achievable TCP velocity, not
# flange velocity, when building a table for hardware use.
# --------------------------------------------------------------------------- #
@pytest.fixture
def ftp_arm():
    """A bare loaded URDF body id, matching how find_throw_pose.py's own
    search()/search_release_state() connect (module-level p.connect, not
    ArmController)."""
    cid = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=cid)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    FTP.set_robot(ROBOT)
    arm_id = p.loadURDF(pybullet_data.getDataPath() + "/" + FTP.URDF_REL_PATH,
                        useFixedBase=True, physicsClientId=cid)
    FTP._set_n_full(arm_id)
    yield arm_id
    FTP.set_tool_offset(np.zeros(3))  # module-global: never leak into other tests
    p.disconnect(cid)


def test_fkj_default_offset_is_zero(ftp_arm):
    q = np.array(FTP.QN, dtype=float)
    pos, J = FTP._fkj(ftp_arm, q)
    ls = p.getLinkState(ftp_arm, FTP.EE, computeForwardKinematics=True)
    assert np.allclose(pos, np.array(ls[4]), atol=1e-9)


def test_set_tool_offset_moves_fkj_position_and_jacobian(ftp_arm):
    q = np.array(FTP.QN, dtype=float)
    pos0, J0 = FTP._fkj(ftp_arm, q)
    FTP.set_tool_offset(TOOL_OFFSET)
    pos1, J1 = FTP._fkj(ftp_arm, q)

    ls = p.getLinkState(ftp_arm, FTP.EE, computeForwardKinematics=True)
    expected_pos, _ = p.multiplyTransforms(ls[4], ls[5], TOOL_OFFSET.tolist(),
                                           [0, 0, 0, 1])
    assert np.allclose(pos1, np.array(expected_pos), atol=1e-9)
    assert np.linalg.norm(pos1 - pos0) > 0.05, (
        "offset FK did not move by the 12 cm this fix corrects for"
    )
    assert not np.allclose(J0, J1, atol=1e-9), (
        "Jacobian unchanged by tool_offset -- localPosition was not threaded through"
    )


# --------------------------------------------------------------------------- #
# run_hardware_throw.py: --tool_offset_z must thread into the solver, AND
# refuse (fail closed) a mismatch against a table's own "tool_offset" stamp --
# the R2/floor_z precedent this project already established for exactly this
# class of silent-wrong-throw mistake.
# --------------------------------------------------------------------------- #
def test_plan_throw_for_target_refuses_a_tool_offset_table_mismatch(arm, tmp_path):
    from robot_arm.robot_profiles import get_robot_profile
    profile = get_robot_profile(ROBOT)
    table = _table_entry(profile)
    table[0]["tool_offset"] = [0.0, 0.0, 0.12]
    table_path = tmp_path / "mismatched_table.npy"
    np.save(table_path, np.array(table, dtype=object))

    cfg = {"release_pos": [0.0, 0.0, 1.1], "T_W": 0.5, "T_R": 1.6, "T": 2.0}
    pol = lambda s, t=0: __import__("torch").tensor([[1.5]], dtype=__import__("torch").float64)

    with pytest.raises(RuntimeError, match="tool_offset"):
        H.plan_throw_for_target(arm, profile, cfg, pol, (0.7, 0.0),
                                opt_pose=str(table_path), tool_offset_z=0.0)


def test_plan_throw_for_target_accepts_a_matching_tool_offset(arm, tmp_path):
    from robot_arm.robot_profiles import get_robot_profile
    profile = get_robot_profile(ROBOT)
    table = _table_entry(profile)
    table[0]["tool_offset"] = [0.0, 0.0, 0.12]
    table_path = tmp_path / "matched_table.npy"
    np.save(table_path, np.array(table, dtype=object))

    cfg = {"release_pos": [0.0, 0.0, 1.1], "T_W": 0.5, "T_R": 1.6, "T": 2.0}
    pol = lambda s, t=0: __import__("torch").tensor([[1.5]], dtype=__import__("torch").float64)

    coeffs, q, qd, v_ach, speed, v_cmd, rel = H.plan_throw_for_target(
        arm, profile, cfg, pol, (0.7, 0.0), opt_pose=str(table_path), tool_offset_z=0.12
    )
    assert rel[2] > 0  # ran to completion, produced a release state


# --------------------------------------------------------------------------- #
# release_box_from_table -- the safety-box bound used by both `plan`'s
# "release pos in safe box" line and `throw`'s fail-closed check MUST be
# derived from the same TCP point solve() now reports, or a correctly-solved
# release gets refused as "outside the box" it was never actually compared
# against. Reproduced live: target (0.75, 0.05) against the TCP-offset table
# PASSED precheck but FAILED the (flange-derived) box check purely because
# the box itself was never moved to match.
# --------------------------------------------------------------------------- #
def test_release_box_from_table_default_offset_matches_flange(arm):
    from robot_arm.robot_profiles import get_robot_profile
    profile = get_robot_profile(ROBOT)
    table = _table_entry(profile)
    lo0, hi0 = H.release_box_from_table(arm, table)
    lo1, hi1 = H.release_box_from_table(arm, table, tool_offset=np.zeros(3))
    assert np.allclose(lo0, lo1) and np.allclose(hi0, hi1)


def test_release_box_from_table_moves_with_tool_offset(arm):
    from robot_arm.robot_profiles import get_robot_profile
    profile = get_robot_profile(ROBOT)
    table = _table_entry(profile)
    lo_flange, hi_flange = H.release_box_from_table(arm, table)
    lo_tcp, hi_tcp = H.release_box_from_table(arm, table, tool_offset=TOOL_OFFSET)
    # margin is identical (0.10) either way, so the box SIZE is unchanged --
    # only its center should shift, by roughly the offset magnitude in z.
    assert (hi_tcp[2] - lo_tcp[2]) == pytest.approx(hi_flange[2] - lo_flange[2], abs=1e-9)
    center_shift = ((lo_tcp + hi_tcp) / 2) - ((lo_flange + hi_flange) / 2)
    # Direction depends on this table entry's own orientation (q_neutral here
    # is not a real release posture) -- the physically meaningful claim is
    # magnitude: the box must move by ~the offset length, not stay put.
    assert np.linalg.norm(center_shift) == pytest.approx(0.12, abs=1e-6), (
        f"box center moved by {np.linalg.norm(center_shift):.4f}, not the "
        "0.12 m tool_offset length -- still flange-anchored or wrongly scaled"
    )


def test_plan_throw_release_pos_falls_inside_its_own_matching_box(arm):
    """The actual regression this fix exists for: a release solved WITH
    tool_offset must land inside a box ALSO derived WITH that same offset --
    not refused as 'outside' a box that was silently still the flange's."""
    from robot_arm.robot_profiles import get_robot_profile
    profile = get_robot_profile(ROBOT)
    table = _table_entry(profile)
    solver = OptimizedReleaseSolver(opt_posture_table=table, tool_offset=TOOL_OFFSET)
    rel, _, _, _ = solver.solve(arm, np.array([1.0, 0.0, 1.0]), target_xy=(1.0, 0.0))
    lo, hi = H.release_box_from_table(arm, table, tool_offset=TOOL_OFFSET)
    assert np.all(rel >= lo) and np.all(rel <= hi), (
        f"release {rel} outside its own matching box [{lo}, {hi}]"
    )


# --------------------------------------------------------------------------- #
# arm_controller.py::attach_ball -- currently-invisible local-frame bug. It
# computes the weld offset as a WORLD-frame delta (ball_pos - ee_pos) but
# passes it straight to createConstraint's parentFramePosition, which PyBullet
# interprets in the PARENT LINK's LOCAL frame. Harmless today because every
# call site spawns the ball exactly AT ee_pos (offset always exactly zero,
# and R(q)@0 == 0 regardless of R) -- becomes a real bug the moment a
# nonzero, rigidly-rotating TCP offset is introduced (training with the ball
# welded 12 cm out at the gripper TCP).
# --------------------------------------------------------------------------- #
def test_attach_ball_offset_is_expressed_in_the_link_local_frame(arm):
    """A ball spawned at a known LOCAL offset from ee_link must be welded
    with that same local offset -- not the world-frame delta, which only
    coincides with the local offset when the link happens to be at identity
    orientation (never true for this arm's q_neutral)."""
    ee_pos, _, ee_orn, _ = arm.ee_state()
    local_offset = np.array([0.03, -0.02, 0.12])  # not axis-aligned: catches a rotation bug
    world_pos, _ = p.multiplyTransforms(
        ee_pos.tolist(), ee_orn.tolist(), local_offset.tolist(), [0, 0, 0, 1],
        physicsClientId=arm._cid,
    )
    ball_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.0327, physicsClientId=arm._cid)
    ball_id = p.createMultiBody(baseMass=0.0577, baseCollisionShapeIndex=ball_col,
                                basePosition=world_pos, physicsClientId=arm._cid)
    arm.attach_ball(ball_id)
    info = p.getConstraintInfo(arm._grip_id, physicsClientId=arm._cid)
    got_local = np.array(info[6])  # jointPivotInParent
    assert np.allclose(got_local, local_offset, atol=1e-6), (
        f"parentFramePosition {got_local} is not the local offset {local_offset} -- "
        "world-frame delta was passed where a local-frame offset was required"
    )


# --------------------------------------------------------------------------- #
# PyBulletThrowingSystem: the ball must actually WELD at the TCP (not the
# flange) when tool_offset is given, and the "dynamic" release path -- which
# reads the ball's REAL PyBullet-computed velocity after being rigidly
# dragged, not a hand formula -- must show the omega x r_offset speed boost
# for free, simply because the physics is now correct.
# --------------------------------------------------------------------------- #
def _rollout_release_speed(tool_offset):
    from robot_arm.robot_profiles import get_robot_profile
    profile = get_robot_profile(ROBOT)
    q_posture = np.array(profile.q_neutral, dtype=float)
    sysm = PyBulletThrowingSystem(
        mass=0.0577, radius=0.0327, robot_name=ROBOT,
        opt_posture=q_posture, opt_launch_deg=0.0,
        tool_offset=tool_offset,
    )
    v_cmd = np.array([0.002, 0.0, 0.0])  # tiny: guarantees scale<1 (below LP v_max)
    sysm._simulate_pybullet(np.array([0.0, 0.0, 1.0]), v_cmd, T=1.0, dt=0.02)
    assert sysm.last_release_info is not None, "dynamic release path did not run"
    return sysm.last_release_info["v_release"]


def test_tool_offset_is_stored_and_passed_to_the_release_solver():
    sysm = PyBulletThrowingSystem(robot_name=ROBOT, tool_offset=TOOL_OFFSET)
    assert np.allclose(sysm.release_solver.tool_offset, TOOL_OFFSET)


def test_dynamic_release_velocity_shows_the_omega_cross_r_boost_with_real_physics():
    """No manual omega x r formula anywhere in this test or in
    model_pybullet.py -- the boost must come from PyBullet actually simulating
    a ball rigidly welded 12 cm out on a rotating wrist and reading back its
    true velocity when released."""
    v_flange = _rollout_release_speed(None)
    v_tcp = _rollout_release_speed(TOOL_OFFSET)
    assert np.linalg.norm(v_tcp) > np.linalg.norm(v_flange), (
        f"TCP release speed {np.linalg.norm(v_tcp):.4f} does not exceed flange "
        f"release speed {np.linalg.norm(v_flange):.4f} -- offset ball weld had no effect"
    )
