import os
import numpy as np
import pytest
from simulation_class.model_pybullet import PyBulletThrowingSystem

TABLE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "throw_pose_table.npy")


class _NoWind:
    def reset(self): pass
    def __call__(self, t): return np.zeros(3)


class _ConstPolicy:
    def __init__(self, s): self.s = s
    def __call__(self, s0, t): return np.array([self.s])


@pytest.fixture
def sysm():
    table = list(np.load(TABLE_PATH, allow_pickle=True))
    s = PyBulletThrowingSystem(robot_name="kinova_gen3_dyn",
                               opt_posture_table=table,
                               wind_model=_NoWind(), t_w=0.5, t_r=1.6)
    return s


@pytest.mark.parametrize("speed,tgt", [(0.6, (0.6, 0.0)), (0.6, (0.6, 0.3)),
                                       (0.6, (0.6, -0.3)), (0.6, (0.5, 0.35))])
def test_rollout_aims_and_does_not_blow_up(sysm, speed, tgt):
    s0 = np.concatenate([[0.3, 0.0, 0.5], np.zeros(3), np.array(tgt)])
    pos, vel, wind = sysm.rollout(s0, _ConstPolicy(speed), 2.0, 0.02, 0.0)
    info = sysm.last_release_info
    rel_speed = np.linalg.norm(info["v_release"])
    assert rel_speed < 3.0                                   # no runaway (was 57 m/s)
    # release speed tracks the command (kinetic-chain throw actually reaches it)
    assert abs(rel_speed - speed) < 0.2
    rel = pos[0][:3]
    land = pos[-1][:3]
    land_az = np.degrees(np.arctan2(land[1] - rel[1], land[0] - rel[0]))
    tgt_az = np.degrees(np.arctan2(tgt[1], tgt[0]))
    assert abs(land_az - tgt_az) < 4.0                       # aims at the target


def test_setpoint_velocity_within_qd_max_whole_trajectory(sysm):
    import pybullet as p
    import pybullet_data
    from robot_arm.arm_controller import ArmController, _eval_cubic
    from robot_arm.robot_profiles import get_robot_profile
    client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    prof = get_robot_profile("kinova_gen3_dyn")
    arm = ArmController(client, pybullet_data.getDataPath() + "/" + prof.urdf_rel_path,
                        robot_name="kinova_gen3_dyn")
    qd_max = np.array(prof.qd_max)
    for tgt in [(0.6, 0.0), (0.6, 0.3), (0.6, -0.3)]:
        sysm._cur_target_xy = np.array(tgt)
        v_cmd = np.array([0.4, 0.0, 0.3])                    # only |v_cmd| (=0.5) matters
        rel, q_ovr, qd_ovr, _ = sysm._optimized_release(arm, v_cmd)
        coeffs, _, _, _ = arm.plan_throw(v_cmd, rel, 0.5, 1.6, 3.0,
                                         q_release_override=q_ovr,
                                         qd_release_override=qd_ovr, monotonic_windup=True)
        for phase, dur in [("windup", coeffs["t_w"]),
                           ("throw", coeffs["t_r"] - coeffs["t_w"])]:
            for t in np.linspace(0.0, dur, 150):
                _, qd, _ = _eval_cubic(coeffs[phase], t, with_accel=True)
                assert np.all(np.abs(qd) <= qd_max + 1e-6), f"{phase} exceeds qd_max at {tgt}"
    p.disconnect(client)
