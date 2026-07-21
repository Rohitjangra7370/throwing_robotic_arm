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
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    prof = get_robot_profile("kinova_gen3_dyn")
    urdf = pybullet_data.getDataPath() + "/" + prof.urdf_rel_path
    a = ArmController(client, urdf, robot_name="kinova_gen3_dyn")
    yield a, prof
    p.disconnect(client)


def _peak_qd(coeffs, phase, dt):
    peak = 0.0
    for t in np.linspace(0.0, dt, 120):
        _, qd, _ = _eval_cubic(coeffs[phase], t, with_accel=True)
        peak = max(peak, float(np.max(np.abs(qd))))
    return peak


def test_monotonic_windup_keeps_whole_stroke_under_qd_max(arm):
    controller, prof = arm
    qd_max = np.array(prof.qd_max)
    q_release = np.array(prof.q_neutral) + np.array([0.0, 0.6, -0.5, 0.7, 0.3, -0.4, 0.2])
    qd_release = np.array([0.0, 1.0, -0.1, 1.0, 0.9, 0.8, -0.9])  # <= qd_max, base 0
    v_cmd = np.array([0.6, 0.0, 0.3])
    coeffs, _, _, _ = controller.plan_throw(
        v_cmd, np.array(prof.default_release_pos), 0.5, 1.6, 3.0,
        q_release_override=q_release, qd_release_override=qd_release,
        monotonic_windup=True,
    )
    dt_throw = coeffs["t_r"] - coeffs["t_w"]
    dt_windup = coeffs["t_w"]
    # throw phase never exceeds qd_release (linear ramp) -> <= qd_max
    assert _peak_qd(coeffs, "throw", dt_throw) <= np.max(qd_max) + 1e-6
    # windup phase also within qd_max
    assert _peak_qd(coeffs, "windup", dt_windup) <= np.max(qd_max) + 1e-6
