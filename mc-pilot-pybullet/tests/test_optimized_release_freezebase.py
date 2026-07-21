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


def test_optimized_release_pins_base_velocity(setup):
    sysm, arm = setup
    sysm._cur_target_xy = np.array([0.6, 0.3])       # off-axis target
    v_cmd = np.array([0.5, 0.25, 0.3])
    _, _, qd_release, _ = sysm._optimized_release(arm, v_cmd)
    assert abs(qd_release[0]) < 1e-9                  # base joint held still
    assert np.all(np.abs(qd_release) <= np.array(arm._qd_max) + 1e-9)
