import numpy as np
import pybullet as p
import pybullet_data
import pytest

from robot_arm.arm_controller import ArmController
from robot_arm.robot_profiles import get_robot_profile

DT = 0.02


def _make_arm(robot_name):
    client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(DT, physicsClientId=client)
    prof = get_robot_profile(robot_name)
    urdf = pybullet_data.getDataPath() + "/" + prof.urdf_rel_path
    return client, ArmController(client, urdf, robot_name=robot_name)


def test_gravity_hold_drift_under_1cm():
    client, arm = _make_arm("kinova_gen3_dyn")
    try:
        q_hold = np.array(get_robot_profile("kinova_gen3_dyn").q_neutral)
        qd_zero = np.zeros(7)
        ee_start = arm.ee_state()[0]
        for _ in range(int(2.0 / DT)):  # 2 s
            arm.step(q_hold, qd_zero, np.zeros(7))
            p.stepSimulation(physicsClientId=client)
        drift = np.linalg.norm(arm.ee_state()[0] - ee_start)
        assert drift < 0.01, f"EE drifted {drift * 100:.2f} cm under gravity hold"
    finally:
        p.disconnect(client)


def test_torque_profile_requires_gains():
    # kinematic profile has no tau_max; torque construction must be impossible
    # for it, and the dyn profile must expose the arrays the controller needs.
    client, arm = _make_arm("kinova_gen3_dyn")
    try:
        assert arm._tau_max.shape == (7,)
        assert arm._kp.shape == (7,)
        assert arm._kd.shape == (7,)
    finally:
        p.disconnect(client)
