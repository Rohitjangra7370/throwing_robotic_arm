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
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    prof = get_robot_profile("kinova_gen3_dyn")
    urdf = pybullet_data.getDataPath() + "/" + prof.urdf_rel_path
    a = ArmController(client, urdf, robot_name="kinova_gen3_dyn")
    yield a, prof
    p.disconnect(client)


def _peak_qd(controller, coeffs, t_lo, t_hi):
    """Peak |qd| over [t_lo, t_hi] using the real API (handles stagger clipping)."""
    peak = 0.0
    for t in np.linspace(t_lo, t_hi, 200):
        _, qd = controller.get_setpoint(coeffs, t, with_accel=False)
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
    # throw phase never exceeds qd_release (staggered linear ramps) -> <= qd_max
    assert _peak_qd(controller, coeffs, coeffs["t_w"], coeffs["t_r"]) <= np.max(qd_max) + 1e-6
    # windup phase also within qd_max
    assert _peak_qd(controller, coeffs, 0.0, coeffs["t_w"]) <= np.max(qd_max) + 1e-6


def test_proximal_joint_finishes_ramp_before_distal_joint(arm):
    """Kinetic-chain check: joint 1 (proximal) reaches its release velocity
    earlier than joint 6 (distal), the whip-like sequencing real throws use."""
    controller, prof = arm
    q_release = np.array(prof.q_neutral) + np.array([0.0, 0.6, -0.5, 0.7, 0.3, -0.4, 0.2])
    qd_release = np.array([0.0, 1.0, -0.1, 1.0, 0.9, 0.8, -0.9])
    v_cmd = np.array([0.6, 0.0, 0.3])
    coeffs, _, _, _ = controller.plan_throw(
        v_cmd, np.array(prof.default_release_pos), 0.5, 1.6, 3.0,
        q_release_override=q_release, qd_release_override=qd_release,
        monotonic_windup=True,
    )
    t_w, t_r = coeffs["t_w"], coeffs["t_r"]
    mid = t_w + 0.6 * (t_r - t_w)   # 60% through the throw phase
    _, qd_mid = controller.get_setpoint(coeffs, mid, with_accel=False)
    progress_1 = abs(qd_mid[1]) / abs(qd_release[1])   # proximal: fraction of its own ramp done
    progress_6 = abs(qd_mid[6]) / abs(qd_release[6])   # distal
    # proximal joint (starts ramping immediately) is further through its own
    # ramp than the distal joint (stays cocked longer, snaps late) -- the
    # whip-like proximal-to-distal cascade.
    assert progress_1 > progress_6 + 0.2
