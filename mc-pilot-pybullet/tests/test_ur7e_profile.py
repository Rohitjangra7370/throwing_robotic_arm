"""
UR7e (second hardware arm) model + the de-hardcoded release-LP freeze set.

Two things are pinned here.

1. The vendored model itself. `scripts/install_ur7e_urdf.sh` builds
   pybullet_data/ur7e/ from the official UR ROS 2 description tag 4.3.1, and
   that output lives outside this repo. These tests are what notices if it is
   missing, stale, or rebuilt from a different tag whose link indexing moved.

2. The freeze set. The release LP pins qd=0 on the base azimuth joint and every
   roll/twist joint, and that set used to be the literal constant (0,2,4,6) in
   two places (`release_solver.py`, `find_throw_pose.py`). That constant is the
   7-DoF alternating roll-pitch-roll layout of the Gen3 and Panda -- it is not a
   law. A UR is pan / three parallel pitches / wrist2 / tool-roll, i.e.
   (0,4,5), and on any 6-DoF arm the old constant additionally addressed index
   6, which in the LP is the SPEED SLACK variable rather than a joint: freezing
   it pins the release speed to exactly zero and reports success.
"""

import os
import sys

import numpy as np
import pybullet as p
import pybullet_data
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from robot_arm.arm_controller import ArmController              # noqa: E402
from robot_arm.robot_profiles import (get_robot_profile,        # noqa: E402
                                      roll_indices)
from robot_arm.urdf_fixup import massless_links                 # noqa: E402
from simulation_class.release_solver import OptimizedReleaseSolver  # noqa: E402


URDF = os.path.join(pybullet_data.getDataPath(), "ur7e/ur7e.urdf")
needs_model = pytest.mark.skipif(
    not os.path.exists(URDF),
    reason="UR7e model not vendored -- run scripts/install_ur7e_urdf.sh",
)


# --------------------------------------------------------------------------- #
# roll_indices: pure, no PyBullet
# --------------------------------------------------------------------------- #
def test_legacy_arms_keep_the_exact_historical_freeze_set():
    """Every 7-DoF profile must still freeze (0,2,4,6) bit-identically.

    This is the no-regression guard for the whole change: making the set
    profile-driven must not move a single existing checkpoint or pose table.
    """
    for name in ("kinova_gen3", "kinova_gen3_dyn", "franka_panda",
                 "franka_panda_dyn", "kuka_iiwa"):
        assert roll_indices(get_robot_profile(name)) == (0, 2, 4, 6), name


def test_ur7e_freezes_pan_wrist2_and_toolroll():
    for name in ("ur7e", "ur7e_dyn"):
        assert roll_indices(get_robot_profile(name)) == (0, 4, 5), name


def test_fallback_never_addresses_the_lp_speed_slack():
    """The legacy constant is trimmed to the arm's DoF count.

    xarm6 declares no roll_idx, so it takes the fallback. Untrimmed, index 6
    would land on the LP's slack variable and silently zero the release speed.
    """
    prof = get_robot_profile("xarm6")
    idx = roll_indices(prof)
    assert idx == (0, 2, 4)
    assert max(idx) < len(prof.joint_ids)


def test_every_profile_declares_an_in_range_freeze_set():
    from robot_arm.robot_profiles import available_robot_names
    for name in available_robot_names():
        prof = get_robot_profile(name)
        idx = roll_indices(prof)
        assert all(0 <= i < len(prof.joint_ids) for i in idx), (name, idx)


# --------------------------------------------------------------------------- #
# The vendored model
# --------------------------------------------------------------------------- #
@needs_model
def test_urdf_indexing_matches_the_profile():
    """joint_ids / ee_link are positions in the URDF, so a rebuild that
    reorders links must fail loudly rather than plan against the wrong body."""
    cid = p.connect(p.DIRECT)
    try:
        rid = p.loadURDF(URDF, useFixedBase=True, physicsClientId=cid)
        n = p.getNumJoints(rid, physicsClientId=cid)
        rev = [j for j in range(n)
               if p.getJointInfo(rid, j, physicsClientId=cid)[2] == p.JOINT_REVOLUTE]
        names = [p.getJointInfo(rid, j, physicsClientId=cid)[1].decode() for j in rev]
        tool0 = [j for j in range(n)
                 if p.getJointInfo(rid, j, physicsClientId=cid)[12] == b"tool0"]

        prof = get_robot_profile("ur7e_dyn")
        assert tuple(rev) == prof.joint_ids
        assert names == ["shoulder_pan_joint", "shoulder_lift_joint",
                         "elbow_joint", "wrist_1_joint", "wrist_2_joint",
                         "wrist_3_joint"]
        assert tool0 == [prof.ee_link]
    finally:
        p.disconnect(cid)


@needs_model
def test_bodyless_links_are_repaired_away():
    """UR declares six bodyless links, three of them at the wrist; PyBullet
    would give each 1 kg. Raw load is 26.7 kg against 21.7 kg of real links --
    the same defect that inflated Gen3 gravity torque by 2.1-2.4x."""
    assert set(massless_links(open(URDF).read())) == {
        "world", "base_link", "ft_frame", "base", "flange", "tool0"}

    cid = p.connect(p.DIRECT)
    try:
        # ArmController routes through repair_massless_links internally.
        arm = ArmController(cid, URDF, robot_name="ur7e_dyn")
        n = p.getNumJoints(arm._arm_id, physicsClientId=cid)
        mass = sum(p.getDynamicsInfo(arm._arm_id, j, physicsClientId=cid)[0]
                   for j in range(n))
        assert mass == pytest.approx(21.700, abs=1e-3)
        for link in (8, 9, 10):                       # ft_frame, flange, tool0
            assert p.getDynamicsInfo(arm._arm_id, link,
                                     physicsClientId=cid)[0] == 0.0
    finally:
        p.disconnect(cid)


@needs_model
def test_torque_mode_zero_pad_shortcut_is_valid():
    """UR's actuated joints are the URDF's only non-fixed joints, so they must
    occupy dof positions 0..5 contiguously -- ArmController's torque branch
    refuses anything else."""
    cid = p.connect(p.DIRECT)
    try:
        arm = ArmController(cid, URDF, robot_name="ur7e_dyn")
        assert arm._dof_ids == [0, 1, 2, 3, 4, 5]
        assert arm._n_dofs == 6
    finally:
        p.disconnect(cid)


@needs_model
def test_declared_limits_match_the_ur7e_tech_sheet():
    prof = get_robot_profile("ur7e_dyn")
    # 180 deg/s on all six -- independently confirmed against the UR7e tech
    # sheet, not just inherited from the ur5e-derived yaml.
    assert np.allclose(prof.qd_max, np.deg2rad(180.0), atol=1e-4)
    # UR5e-derived and PROVISIONAL (see the profile notes) -- pinned so a
    # silent edit to a measured value has to be deliberate.
    assert prof.tau_max == (150.0, 150.0, 150.0, 28.0, 28.0, 28.0)


# --------------------------------------------------------------------------- #
# The LP, with the real arm
# --------------------------------------------------------------------------- #
@needs_model
def test_release_lp_freezes_exactly_the_declared_joints():
    """Regression for the corkscrew bug, ported to UR geometry: only the three
    pitch joints may carry velocity, and the commanded direction must be met
    exactly (the LP is direction-CONSTRAINED, not a sign trick)."""
    prof = get_robot_profile("ur7e_dyn")
    cid = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -9.81, physicsClientId=cid)
        arm = ArmController(cid, URDF, base_position=(0, 0, 0.433),
                            robot_name="ur7e_dyn")
        solver = OptimizedReleaseSolver(opt_posture=np.array(prof.q_neutral),
                                        opt_launch_deg=70.0,
                                        roll_idx=roll_indices(prof))
        speed, elev = 2.0, np.deg2rad(70.0)
        v_cmd = np.array([speed * np.cos(elev), 0.0, speed * np.sin(elev)])
        _, _, qd_release, v = solver.solve(arm, v_cmd,
                                           target_xy=np.array([1.5, 0.0]))

        frozen = list(roll_indices(prof))
        assert np.all(qd_release[frozen] == 0.0)
        assert np.any(np.abs(qd_release[[1, 2, 3]]) > 1e-6)
        assert np.linalg.norm(v) == pytest.approx(speed, abs=1e-9)
        assert np.abs(v[1]) < 1e-9                     # no out-of-plane drift
    finally:
        p.disconnect(cid)


@needs_model
def test_wrong_freeze_set_would_zero_the_speed():
    """Why the trim matters, demonstrated rather than asserted in a comment.

    Handing the 6-DoF UR the legacy 7-DoF constant pins bounds[6] -- the speed
    slack -- to zero. The LP still succeeds; it just returns a 0 m/s throw.
    """
    prof = get_robot_profile("ur7e_dyn")
    cid = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -9.81, physicsClientId=cid)
        arm = ArmController(cid, URDF, base_position=(0, 0, 0.433),
                            robot_name="ur7e_dyn")
        solver = OptimizedReleaseSolver(opt_posture=np.array(prof.q_neutral),
                                        opt_launch_deg=70.0,
                                        roll_idx=(0, 2, 4, 6))
        elev = np.deg2rad(70.0)
        v_cmd = np.array([2.0 * np.cos(elev), 0.0, 2.0 * np.sin(elev)])
        with pytest.raises(ValueError, match="out of range"):
            solver.solve(arm, v_cmd, target_xy=np.array([1.5, 0.0]))
    finally:
        p.disconnect(cid)


@needs_model
def test_sim_and_hardware_ask_for_the_same_freeze_set():
    """`release_solver.py` is shared by sim and hardware and the two must be
    constructed identically -- the same rule test_hardware_planner.py enforces
    for the rest of the release state."""
    import inspect

    import run_hardware_throw
    from simulation_class import model_pybullet

    sim_src = inspect.getsource(model_pybullet.PyBulletThrowingSystem.__init__)
    hw_src = inspect.getsource(run_hardware_throw.plan_throw_for_target)
    assert "roll_idx=roll_indices(" in sim_src
    assert "roll_idx=roll_indices(" in hw_src


# --------------------------------------------------------------------------- #
# Cartesian TCP-speed ceiling
# --------------------------------------------------------------------------- #
def test_only_ur7e_declares_a_tcp_speed_ceiling():
    """None everywhere else, so the release LP is bit-identical for every arm
    that existed before this field."""
    from robot_arm.robot_profiles import available_robot_names
    for name in available_robot_names():
        want = 4.0 if name.startswith("ur7e") else None
        assert get_robot_profile(name).v_tcp_max == want, name


@needs_model
def test_tcp_ceiling_actually_caps_the_release_lp():
    """The UR7e's joint-velocity LP solves ABOVE its own rated Cartesian speed.

    Regression for a whole pose table of unusable entries: the first UR7e
    overhead search returned 23/23 entries at 4.48-4.65 m/s against a rated
    4.0 m/s, every one of which the controller would clamp or refuse -- so the
    arm would execute a different throw than the one trained, the failure mode
    release_solver.py's docstring exists to prevent. Never binds on the Gen3
    (LP max 2.07 m/s), which is why it went unnoticed until a second arm.
    """
    prof = get_robot_profile("ur7e_dyn")
    cid = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -9.81, physicsClientId=cid)
        arm = ArmController(cid, URDF, base_position=(0, 0, 0.433),
                            robot_name="ur7e_dyn")
        # The fast release posture the overhead search actually selects --
        # q_neutral is not near the velocity ceiling, so it cannot show a cap.
        q = np.array([0.0, -1.7453, -0.2967, -1.8151, 0.0, 0.0])
        elev = np.deg2rad(25.0)
        # Ask for far more speed than the arm is rated for; the solver clamps
        # to whatever the LP can reach, so the cap has to live in the LP.
        v_cmd = np.array([50.0 * np.cos(elev), 0.0, 50.0 * np.sin(elev)])

        uncapped = OptimizedReleaseSolver(opt_posture=q, opt_launch_deg=25.0,
                                          roll_idx=roll_indices(prof))
        capped = OptimizedReleaseSolver(opt_posture=q, opt_launch_deg=25.0,
                                        roll_idx=roll_indices(prof),
                                        v_tcp_max=prof.v_tcp_max)
        def achieved(solver):
            # solve() returns the COMMANDED velocity, which is just d*speed
            # whatever the arm can do. What the TCP actually reaches is
            # J(q_release) @ qd_release, so measure that.
            _, q_rel, qd_rel, _ = solver.solve(arm, v_cmd,
                                               target_xy=np.array([2.0, 0.0]))
            q_full = arm._ik_q_neutral.copy()
            for li, dof in enumerate(arm._dof_ids):
                q_full[dof] = q_rel[li]
            jl, _ = p.calculateJacobian(arm._arm_id, arm._ee_link, [0.0, 0.0, 0.0],
                                        q_full.tolist(), [0.0] * arm._n_dofs,
                                        [0.0] * arm._n_dofs, physicsClientId=cid)
            return float(np.linalg.norm(np.array(jl)[:, arm._dof_ids] @ qd_rel))

        assert achieved(uncapped) > prof.v_tcp_max
        assert achieved(capped) == pytest.approx(prof.v_tcp_max, abs=1e-6)
    finally:
        p.disconnect(cid)


@needs_model
def test_find_throw_pose_carries_the_ceiling_too():
    """aimed_speed() is the search's own LP -- a separate implementation from
    release_solver's, so it needs the cap independently or the table and the
    executor disagree about what speed is reachable."""
    import find_throw_pose as ftp
    ftp.set_robot("ur7e_dyn")
    try:
        assert ftp.V_TCP_MAX == 4.0
        cid = p.connect(p.DIRECT)
        try:
            p.setGravity(0, 0, -9.81, physicsClientId=cid)
            p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                      physicsClientId=cid)
            arm = ftp.load_arm()
            ftp._set_n_full(arm)
            pitch = ftp.pitch_indices()
            q = ftp.sagittal_q(pitch, -1.7453, -0.2967, -1.8151)
            _, J = ftp._fkj(arm, q)
            d = [np.cos(np.deg2rad(25)), 0.0, np.sin(np.deg2rad(25))]
            s_un, _ = ftp.aimed_speed(J, d, ftp.QD, v_tcp_max=None)
            s_cap, _ = ftp.aimed_speed(J, d, ftp.QD)      # profile default
            assert s_un > 4.0
            assert s_cap == pytest.approx(4.0, abs=1e-9)
        finally:
            p.disconnect(cid)
    finally:
        ftp.set_robot("kinova_gen3_dyn")


def test_gen3_lp_is_unaffected_by_the_new_ceiling():
    """The Gen3 declares no ceiling, so its LP bound must still be unbounded --
    the guarantee that every existing table and checkpoint is untouched."""
    import find_throw_pose as ftp
    ftp.set_robot("kinova_gen3_dyn")
    assert ftp.V_TCP_MAX is None
    assert get_robot_profile("kinova_gen3_dyn").v_tcp_max is None


# --------------------------------------------------------------------------- #
# Base-rotation sign
# --------------------------------------------------------------------------- #
@needs_model
def test_base_rotation_sign_is_measured_per_arm():
    """`build_table_by_rotation` aims a throw purely by turning the base, and
    whether that means adding or subtracting the azimuth from q[0] is a URDF
    property, not a constant.

    Regression for a bug that produced no error at all: the Gen3's -1 was
    hardcoded in both the table builder and the release solver. On the UR7e
    (+1) the aim inverted -- landing y moved OPPOSITE the target y, and the
    landing error went from 11 cm on-axis to 147 cm at +-30 deg azimuth.
    """
    import find_throw_pose as ftp
    import pybullet_data as _pd

    expected = {"kinova_gen3_dyn": -1.0, "franka_panda_dyn": +1.0,
                "kuka_iiwa": +1.0, "ur7e_dyn": +1.0}
    try:
        for robot, want in expected.items():
            ftp.set_robot(robot)
            cid = p.connect(p.DIRECT)
            try:
                p.setAdditionalSearchPath(_pd.getDataPath(), physicsClientId=cid)
                arm = ftp.load_arm()
                ftp._set_n_full(arm)
                pitch = ftp.pitch_indices()
                q = ftp.sagittal_q(pitch, -0.6, 0.8, -1.2)
                assert ftp.base_rotation_sign(arm, q) == want, robot
            finally:
                p.disconnect(cid)
    finally:
        ftp.set_robot("kinova_gen3_dyn")


def test_unstamped_table_keeps_the_legacy_sign():
    """Every pose table produced before the stamp existed is Gen3 (verified:
    all 23 on-disk pose-table checkpoints are kinova_gen3_dyn), so an absent
    stamp must still mean -1 or existing checkpoints would change."""
    import inspect

    from simulation_class import release_solver
    src = inspect.getsource(release_solver.OptimizedReleaseSolver.solve)
    assert 'near.get("base_sign", -1.0)' in src


@needs_model
def test_table_builder_refuses_to_guess_the_sign():
    """Fail closed: an unstamped e0 must raise rather than silently assume."""
    import find_throw_pose as ftp
    with pytest.raises(ValueError, match="base_sign"):
        ftp.build_table_by_rotation({"range": 1.0, "q": np.zeros(7),
                                     "qd": np.zeros(7), "elev_deg": 15.0,
                                     "speed": 1.0, "azimuth_deg": 0.0})


@needs_model
def test_solver_release_pos_is_fk_of_its_own_q_release():
    """`release_pos` must be forward kinematics of the `q_release` returned
    beside it, offset by exactly the tool offset -- nothing else.

    Regression for a dof-index-used-as-joint-index bug in `release_solver`:
    `q_full` is indexed by DOF but was fed to `resetJointState`, which takes a
    JOINT index. The two coincide only when the actuated joints are the URDF's
    first N, which holds for Gen3/Panda/KUKA and NOT for a UR (arm joints 2..7,
    dofs 0..5). On the UR7e it posed two FIXED joints, left wrist_2 and wrist_3
    at zero, and reported a release point 14 cm from where the arm would
    actually be -- silently, with the planner still reporting time_scale=1.0
    and clip_scale=1.0.
    """
    import find_throw_pose  # noqa: F401  (ensures the table path is importable)

    table_path = os.path.join(os.path.dirname(__file__), "..",
                              "throw_pose_table_ur7e.npy")
    if not os.path.exists(table_path):
        pytest.skip("UR7e pose table not built")
    table = list(np.load(table_path, allow_pickle=True))
    prof = get_robot_profile("ur7e_dyn")

    cid = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -9.81, physicsClientId=cid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
        arm = ArmController(cid, URDF, robot_name="ur7e_dyn")
        arm.reset()
        solver = OptimizedReleaseSolver(
            opt_posture_table=table,
            opt_launch_deg=float(table[0]["elev_deg"]),
            roll_idx=roll_indices(prof), v_tcp_max=prof.v_tcp_max)
        elev = np.deg2rad(float(table[0]["elev_deg"]))
        for azimuth_target in ([1.0, 0.0], [1.0, 0.5], [1.0, -0.5]):
            v_cmd = np.array([2.0 * np.cos(elev), 0.0, 2.0 * np.sin(elev)])
            rel, q_rel, _, _ = solver.solve(arm, v_cmd,
                                            target_xy=np.array(azimuth_target))
            for li, j in enumerate(arm._joint_ids):
                p.resetJointState(arm._arm_id, j, q_rel[li], physicsClientId=cid)
            ee, _, _, _ = arm.ee_state()
            # tool_offset is zero for this arm, so FK must land on it to
            # PyBullet's own float32 FK precision (~1e-8); the bug this pins
            # was 0.14 m, seven orders of magnitude away.
            assert np.linalg.norm(ee - rel) < 1e-6, azimuth_target
    finally:
        p.disconnect(cid)
