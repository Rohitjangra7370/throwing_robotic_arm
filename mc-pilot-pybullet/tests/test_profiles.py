import numpy as np

from robot_arm.robot_profiles import get_robot_profile, profile_to_dict


def test_kinova_gen3_dyn_profile_exists():
    prof = get_robot_profile("kinova_gen3_dyn")
    assert prof.control_mode == "torque"
    assert prof.tau_max == (39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0)
    assert prof.kp == (400.0,) * 7
    assert prof.kd == (60.0,) * 7
    # kinematics identical to the kinematic kinova_gen3 profile
    base = get_robot_profile("kinova_gen3")
    assert prof.urdf_rel_path == base.urdf_rel_path
    assert prof.q_neutral == base.q_neutral
    assert prof.qd_max == base.qd_max
    assert prof.default_release_pos == base.default_release_pos
    assert prof.speed_bounds == base.speed_bounds
    assert prof.timing == base.timing


def test_existing_profiles_unchanged():
    base = get_robot_profile("kinova_gen3")
    assert base.control_mode == "kinematic"
    assert base.tau_max is None
    kuka = get_robot_profile("kuka_iiwa")
    assert kuka.tau_max is None and kuka.kp is None and kuka.kd is None


def test_profile_to_dict_includes_torque_fields():
    d = profile_to_dict(get_robot_profile("kinova_gen3_dyn"))
    assert d["tau_max"] == [39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0]
    assert d["control_mode"] == "torque"
    d2 = profile_to_dict(get_robot_profile("kuka_iiwa"))
    assert d2["tau_max"] is None
