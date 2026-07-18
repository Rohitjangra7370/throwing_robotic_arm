import numpy as np
import pytest

from simulation_class.model_pybullet import PyBulletThrowingSystem
from robot_arm.noise_models import VelocityBiasNoise


def test_rollout_dynamic_release_info():
    sys_ = PyBulletThrowingSystem(robot_name="kinova_gen3_dyn", t_w=0.40, t_r=0.80)
    policy = lambda s, t: np.array([0.6])
    s0 = np.array([0.55, 0.0, 0.45, 0.0, 0.0, 0.0, 0.77, 0.0])
    noisy, inputs, clean = sys_.rollout(s0, policy, T=2.0, dt=0.02, noise=0.0)

    assert np.all(np.isfinite(clean))
    assert clean.shape[1] == 8
    assert clean[-1, 2] < 0.05  # ball landed (interpolated to plane)

    info = sys_.last_release_info
    for key in ("v_cmd", "v_planned", "v_release", "release_pos_err",
                "clip_scale", "time_scale"):
        assert key in info, key
    assert info["time_scale"] >= 1.0
    assert info["release_pos_err"] < 0.10
    # first recorded velocity row must be the ACTUAL ball velocity
    np.testing.assert_allclose(clean[0, 3:6], info["v_release"], atol=1e-9)


def test_arm_noise_rejected_in_torque_mode():
    with pytest.raises(ValueError):
        PyBulletThrowingSystem(
            robot_name="kinova_gen3_dyn", arm_noise=VelocityBiasNoise(0.01)
        )
