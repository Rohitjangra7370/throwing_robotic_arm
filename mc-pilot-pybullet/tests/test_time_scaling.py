import numpy as np
import pybullet as p
import pybullet_data
import pytest

from robot_arm.arm_controller import ArmController, _eval_cubic
from robot_arm.robot_profiles import get_robot_profile


@pytest.fixture
def arm():
    client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(0.02, physicsClientId=client)
    prof = get_robot_profile("kinova_gen3_dyn")
    urdf = pybullet_data.getDataPath() + "/" + prof.urdf_rel_path
    a = ArmController(client, urdf, robot_name="kinova_gen3_dyn")
    yield a, prof
    p.disconnect(client)


def _plan(arm, prof, speed):
    t_w, t_r, T = prof.timing
    alpha = np.deg2rad(35.0)
    v_cmd = np.array([speed * np.cos(alpha), 0.0, speed * np.sin(alpha)])
    return arm.plan_throw(v_cmd, np.array(prof.default_release_pos), t_w, t_r, T)


def test_planned_throw_is_torque_feasible_at_max_speed(arm):
    controller, prof = arm
    coeffs, _, _, _ = _plan(controller, prof, 1.0)  # stress speed, beyond speed_bounds
    assert coeffs["time_scale"] >= 1.0
    # Re-check demanded torque over the (possibly stretched) throw phase.
    dt_throw = coeffs["t_r"] - coeffs["t_w"]
    for tau_t in np.linspace(0.0, dt_throw, 50):
        q, qd, qdd = _eval_cubic(coeffs["throw"], tau_t, with_accel=True)
        torque = np.array(
            p.calculateInverseDynamics(
                controller.arm_id, q.tolist(), qd.tolist(), qdd.tolist(),
                physicsClientId=controller._cid,
            )
        )
        assert np.all(np.abs(torque) <= controller._tau_max + 1e-9)


def test_time_scale_recorded_and_follow_duration_preserved(arm):
    controller, prof = arm
    t_w, t_r, T = prof.timing
    coeffs, _, _, _ = _plan(controller, prof, 1.0)
    assert coeffs["t_r"] >= t_r  # never shrinks
    np.testing.assert_allclose(coeffs["T"] - coeffs["t_r"], T - t_r)
