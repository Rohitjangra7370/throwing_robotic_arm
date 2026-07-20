import numpy as np
import torch

from policy_learning.MC_PILCO import MC_PILOT


def test_rotate_trajectory_preserves_z_and_vz():
    state = np.array([
        [1.0, 0.0, 0.45, 0.5, 0.0, 0.3, 0.7, 0.0],
        [1.1, 0.1, 0.44, 0.5, 0.0, 0.2, 0.7, 0.0],
    ])
    rotated = MC_PILOT._rotate_trajectory(state, np.deg2rad(37.0))
    np.testing.assert_allclose(rotated[:, 2], state[:, 2])   # z unchanged
    np.testing.assert_allclose(rotated[:, 5], state[:, 5])   # vz unchanged


def test_rotate_trajectory_preserves_xy_and_vxvy_magnitude():
    state = np.array([[1.0, 0.5, 0.45, 0.3, -0.4, 0.2, 0.7, 0.0]])
    for angle_deg in [15, 90, 180, 271]:
        rotated = MC_PILOT._rotate_trajectory(state, np.deg2rad(angle_deg))
        pos_norm_before = np.linalg.norm(state[0, 0:2])
        pos_norm_after = np.linalg.norm(rotated[0, 0:2])
        vel_norm_before = np.linalg.norm(state[0, 3:5])
        vel_norm_after = np.linalg.norm(rotated[0, 3:5])
        assert abs(pos_norm_before - pos_norm_after) < 1e-10
        assert abs(vel_norm_before - vel_norm_after) < 1e-10


def test_rotate_trajectory_90deg_matches_hand_computed():
    # rotating (x,y)=(1,0) by 90deg (CCW) should give (0,1)
    state = np.array([[1.0, 0.0, 0.45, 2.0, 0.0, 0.3, 0.7, 0.0]])
    rotated = MC_PILOT._rotate_trajectory(state, np.deg2rad(90.0))
    np.testing.assert_allclose(rotated[0, 0:2], [0.0, 1.0], atol=1e-10)
    np.testing.assert_allclose(rotated[0, 3:5], [0.0, 2.0], atol=1e-10)


def test_na_augmentation_calls_add_data_na_plus_one_times():
    """
    Integration check against the real kinova pipeline: get_data_from_system
    must call model_learning.add_data exactly (1 + Na) times per real
    trajectory -- once with the real data, once per augmented rotation. This
    checks the WIRING directly (via a spy on add_data), not the downstream
    SOD sparse-approximation's stored sample count, which deduplicates
    near-similar points and so isn't a simple multiple of the raw call count.
    """
    from robot_arm.robot_profiles import get_robot_profile
    from simulation_class.model_pybullet import PyBulletThrowingSystem
    import model_learning.Model_learning as ML
    import policy_learning.Policy as Policy
    import policy_learning.Cost_function as Cost_function

    profile = get_robot_profile("kinova_gen3")
    release_pos = np.array(profile.default_release_pos, dtype=float)
    t_w, t_r, _ = profile.timing
    dtype = torch.float64
    device = torch.device("cpu")

    def make_mc_pilot(Na):
        throwing_system = PyBulletThrowingSystem(
            mass=0.0577, radius=0.0327, launch_angle_deg=35.0,
            t_w=t_w, t_r=t_r, robot_name="kinova_gen3",
        )
        init_dict_RBF = {
            "active_dims": np.arange(0, 6), "lengthscales_init": np.ones(6),
            "flg_train_lengthscales": True, "lambda_init": np.ones(1),
            "flg_train_lambda": False, "sigma_n_init": 1 * np.ones(1),
            "flg_train_sigma_n": True, "sigma_n_num": None,
            "dtype": dtype, "device": device,
        }
        model_learning_par = {
            "num_gp": 3, "T_sampling": 0.02, "approximation_mode": "SOD",
            "approximation_dict": {"SOD_threshold_mode": "relative",
                                   "SOD_threshold": 0.5, "flg_SOD_permutation": False},
            "init_dict_list": [init_dict_RBF] * 3, "dtype": dtype, "device": device,
        }
        rand_exploration_policy_par = {
            "full_state_dim": 8, "u_max": 0.6, "u_min": 0.3,
            "n_strata": 3, "dtype": dtype, "device": device,
        }
        control_policy_par = {
            "full_state_dim": 8, "target_dim": 2, "num_basis": 10, "u_max": 0.6,
            "lengthscales_init": np.array([0.01, 0.01]),
            "centers_init": np.random.uniform(0.6, 0.7, (10, 2)),
            "weight_init": 0.1 * (np.random.rand(1, 10) - 0.5),
            "flg_drop": True, "dtype": dtype, "device": device,
        }
        cost_function_par = {
            "position_indices": [0, 1], "target_indices": [6, 7],
            "lengthscale": 0.5, "dtype": dtype, "device": device,
        }

        def sample_target():
            return release_pos[:2] + np.array([0.13, 0.0])

        return MC_PILOT(
            target_sampler=sample_target, release_position=release_pos,
            throwing_system=throwing_system, T_sampling=0.02, state_dim=8, input_dim=1,
            f_model_learning=ML.Ballistic_Model_learning_RBF, model_learning_par=model_learning_par,
            f_rand_exploration_policy=Policy.Stratified_Throwing_Exploration,
            rand_exploration_policy_par=rand_exploration_policy_par,
            f_control_policy=Policy.Throwing_Policy, control_policy_par=control_policy_par,
            f_cost_function=Cost_function.Throwing_Cost, cost_function_par=cost_function_par,
            std_meas_noise=1e-3 * np.ones(8), log_path=None, dtype=dtype, device=device,
            Na=Na,
        )

    def make_spy(mc):
        calls = []
        real_add_data = mc.model_learning.add_data

        def spy_add_data(new_state_samples, new_input_samples):
            calls.append(new_state_samples.copy())
            return real_add_data(new_state_samples, new_input_samples)

        mc.model_learning.add_data = spy_add_data
        return calls

    np.random.seed(0)
    mc0 = make_mc_pilot(Na=0)
    calls0 = make_spy(mc0)
    mc0.get_data_from_system(initial_state=None, T_exploration=0.6, trial_index=0, flg_exploration=True)
    assert len(calls0) == 1, f"Na=0 should call add_data once, got {len(calls0)}"

    np.random.seed(0)
    mc2 = make_mc_pilot(Na=2)
    calls2 = make_spy(mc2)
    mc2.get_data_from_system(initial_state=None, T_exploration=0.6, trial_index=0, flg_exploration=True)
    assert len(calls2) == 3, f"Na=2 should call add_data 3 times (1 real + 2 augmented), got {len(calls2)}"

    # the two augmented calls must differ from the real one (actually rotated)
    assert not np.allclose(calls2[1], calls2[0])
    assert not np.allclose(calls2[2], calls2[0])
    real = calls2[0]
    for augmented in (calls2[1], calls2[2]):
        # z/vz (cols 2, 5) preserved exactly (rotation is about the vertical axis)
        np.testing.assert_allclose(augmented[:, [2, 5]], real[:, [2, 5]])
        # xy position and xy velocity magnitudes preserved row-by-row (pure rotation)
        np.testing.assert_allclose(
            np.linalg.norm(augmented[:, 0:2], axis=1),
            np.linalg.norm(real[:, 0:2], axis=1), atol=1e-9,
        )
        np.testing.assert_allclose(
            np.linalg.norm(augmented[:, 3:5], axis=1),
            np.linalg.norm(real[:, 3:5], axis=1), atol=1e-9,
        )
