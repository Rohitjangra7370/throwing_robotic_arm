import numpy as np
import pybullet as p
import pybullet_data
import pytest

from robot_arm.arm_controller import ArmController
from robot_arm.robot_profiles import get_robot_profile


@pytest.fixture
def arm():
    client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(0.02, physicsClientId=client)
    prof = get_robot_profile("kinova_gen3_dyn")
    urdf = pybullet_data.getDataPath() + "/" + prof.urdf_rel_path
    a = ArmController(client, urdf, robot_name="kinova_gen3_dyn")
    yield a
    p.disconnect(client)


def test_get_setpoint_accel_matches_finite_difference(arm):
    prof = get_robot_profile("kinova_gen3_dyn")
    t_w, t_r, T = prof.timing
    v_cmd = np.array([0.5, 0.0, 0.35])
    coeffs, _, _, _ = arm.plan_throw(v_cmd, np.array(prof.default_release_pos), t_w, t_r, T)

    eps = 1e-5
    for t in [0.1, 0.5 * (t_w + t_r), t_r - 0.05]:
        q, qd, qdd = arm.get_setpoint(coeffs, t, with_accel=True)
        _, qd_lo = arm.get_setpoint(coeffs, t - eps)
        _, qd_hi = arm.get_setpoint(coeffs, t + eps)
        qdd_fd = (qd_hi - qd_lo) / (2 * eps)
        np.testing.assert_allclose(qdd, qdd_fd, atol=1e-4)


def test_get_setpoint_two_tuple_unchanged(arm):
    prof = get_robot_profile("kinova_gen3_dyn")
    t_w, t_r, T = prof.timing
    coeffs, _, _, _ = arm.plan_throw(
        np.array([0.5, 0.0, 0.35]), np.array(prof.default_release_pos), t_w, t_r, T
    )
    out = arm.get_setpoint(coeffs, 0.3)
    assert len(out) == 2
