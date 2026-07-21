import numpy as np
import pybullet as p
import pybullet_data
import pytest
from robot_arm.arm_controller import ArmController
from robot_arm.robot_profiles import get_robot_profile
from simulation_class.model_pybullet import PyBulletThrowingSystem


class _NoWind:
    def reset(self): pass
    def __call__(self, t): return np.zeros(3)


@pytest.fixture
def setup():
    client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(0.02, physicsClientId=client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    prof = get_robot_profile("kinova_gen3_dyn")
    urdf = pybullet_data.getDataPath() + "/" + prof.urdf_rel_path
    arm = ArmController(client, urdf, robot_name="kinova_gen3_dyn")
    q_pose = np.array([-0.15, 1.0, -1.0, 1.2, 0.5, -0.5, 0.3])  # synthetic posture
    sysm = PyBulletThrowingSystem(robot_name="kinova_gen3_dyn", opt_posture=q_pose,
                                  opt_launch_deg=25.0, wind_model=_NoWind(),
                                  t_w=0.5, t_r=1.6)
    yield sysm, arm
    p.disconnect(client)


def test_optimized_release_pins_all_roll_joint_velocities(setup):
    sysm, arm = setup
    sysm._cur_target_xy = np.array([0.6, 0.3])       # off-axis target
    v_cmd = np.array([0.5, 0.25, 0.3])
    _, _, qd_release, _ = sysm._optimized_release(arm, v_cmd)
    for i in (0, 2, 4, 6):                            # base + all roll/twist joints
        assert abs(qd_release[i]) < 1e-9, f"joint {i} (roll/twist) should be held still"
    assert np.all(np.abs(qd_release) <= np.array(arm._qd_max) + 1e-9)


def test_optimized_release_only_wraps_base_not_bounded_joints():
    """Regression: the neutral-nearest 2pi wrap used to apply to EVERY joint,
    which could push a valid bounded joint (e.g. elbow at -100deg, well within
    its +-147deg limit) out to +260deg -- physically unreachable, silently
    corrupting the release pose (verified: release speed collapsed to ~0
    regardless of commanded speed). Only the base (continuous rotation) may
    wrap; every other joint must stay exactly as the table specifies it."""
    client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    prof = get_robot_profile("kinova_gen3_dyn")
    arm = ArmController(client, pybullet_data.getDataPath() + "/" + prof.urdf_rel_path,
                        robot_name="kinova_gen3_dyn")
    q_pose = np.deg2rad([0.0, 90.0, 15.0, -100.0, 0.0, 90.0, 0.0])   # elbow far from neutral
    sysm = PyBulletThrowingSystem(robot_name="kinova_gen3_dyn",
                                  opt_posture_table=[{"q": q_pose, "elev_deg": 0.0,
                                                      "azimuth_deg": 0.0}],
                                  wind_model=_NoWind(), t_w=0.5, t_r=1.2)
    sysm._cur_target_xy = np.array([0.6, 0.0])
    _, q_release, _, _ = sysm._optimized_release(arm, np.array([0.2, 0.0, 0.0]))
    assert abs(np.degrees(q_release[3]) - (-100.0)) < 1e-6, (
        f"elbow got wrapped to {np.degrees(q_release[3]):.1f}deg, should stay at -100deg"
    )
    p.disconnect(client)
