"""
MC_PILOT_Wind — MC-PILCO adapted for wind-disturbed throwing.

Extends MC_PILOT with:
  1. Wind-aware data collection (10-D state with wind dims)
  2. Wind-aware particle propagation in apply_policy()
  3. Wind sampling per particle during optimization

Two modes controlled by wind_aware flag:
  - wind_aware=False: 8-D state, blind GP (baseline comparison)
  - wind_aware=True:  10-D state, wind-conditioned GP + policy
"""

import sys
sys.path.append("..")

import copy
import pickle as pkl
import time

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

import simulation_class.model as model


# ──────────────────────────────────────────────────────────────────────
# Import the full MC_PILCO base hierarchy from the baseline module.
# We exec() the entire base file to get MC_PILCO and MC_PILOT classes
# in our namespace, then extend MC_PILOT below.
# ──────────────────────────────────────────────────────────────────────

# Read and exec the base MC_PILCO module to get all classes
import importlib.util
import os

_base_path = os.path.join(os.path.dirname(__file__), "..", "..", "mc-pilot",
                          "policy_learning", "MC_PILCO.py")
_base_path = os.path.abspath(_base_path)
_spec = importlib.util.spec_from_file_location("mc_pilco_base", _base_path)
_base_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base_mod)

MC_PILCO = _base_mod.MC_PILCO
MC_PILOT = _base_mod.MC_PILOT
MC_PILCO4PMS = _base_mod.MC_PILCO4PMS
MC_PILCO_Experiment = _base_mod.MC_PILCO_Experiment


class MC_PILOT_Wind(MC_PILOT):
    """
    MC-PILOT extended for time-varying wind experiments.

    Key differences from MC_PILOT:
    1. Accepts a wind_model and wind_aware flag
    2. get_data_from_system() records wind trajectory in state arrays
    3. apply_policy() samples wind per particle and includes it in
       the augmented state for GP propagation
    4. Works with both 8-D (blind) and 10-D (aware) state layouts

    Parameters
    ----------
    wind_model : WindModel instance
        The wind model used during simulation.
    wind_aware : bool
        If True, state is 10-D with wind; GP and policy receive wind.
        If False, state is 8-D; wind affects physics but GP/policy are blind.
    wind_sampler : callable or None
        For wind_aware mode: returns a [2] wind sample for each particle
        during apply_policy(). If None, uses the wind_model's current output.
    """

    def __init__(
        self,
        target_sampler,
        release_position,
        throwing_system,
        wind_model,
        wind_aware,
        T_sampling,
        state_dim,
        input_dim,
        f_model_learning,
        model_learning_par,
        f_rand_exploration_policy,
        rand_exploration_policy_par,
        f_control_policy,
        control_policy_par,
        f_cost_function,
        cost_function_par,
        std_meas_noise=None,
        log_path=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
        wind_sampler=None,
    ):
        super(MC_PILOT_Wind, self).__init__(
            target_sampler=target_sampler,
            release_position=release_position,
            throwing_system=throwing_system,
            T_sampling=T_sampling,
            state_dim=state_dim,
            input_dim=input_dim,
            f_model_learning=f_model_learning,
            model_learning_par=model_learning_par,
            f_rand_exploration_policy=f_rand_exploration_policy,
            rand_exploration_policy_par=rand_exploration_policy_par,
            f_control_policy=f_control_policy,
            control_policy_par=control_policy_par,
            f_cost_function=f_cost_function,
            cost_function_par=cost_function_par,
            std_meas_noise=std_meas_noise,
            log_path=log_path,
            dtype=dtype,
            device=device,
        )
        self.wind_model = wind_model
        self.wind_aware = wind_aware
        self.wind_sampler = wind_sampler

    def get_data_from_system(self, initial_state, T_exploration, trial_index,
                             flg_exploration=False):
        """
        Override: sample random target, build augmented s0, run
        WindThrowingSystem rollout. The system handles wind internally
        and returns state arrays with wind dims if wind_aware=True.
        """
        if flg_exploration:
            current_policy = self.rand_exploration_policy
        else:
            current_policy = self.control_policy

        s0 = self._make_augmented_s0()

        # If wind_aware, extend s0 with initial wind measurement
        if self.wind_aware:
            self.wind_model.reset()
            w0 = self.wind_model(0.0)
            s0 = np.concatenate([s0, w0[:2]])

        state_samples, input_samples, noiseless_samples = self.system.rollout(
            s0=s0,
            policy=current_policy.get_np_policy(),
            T=T_exploration,
            dt=self.T_sampling,
            noise=self.std_meas_noise,
        )
        self.state_samples_history.append(state_samples)
        self.input_samples_history.append(input_samples)
        self.noiseless_states_history.append(noiseless_samples)
        self.num_data_collection += 1
        self.model_learning.add_data(
            new_state_samples=state_samples,
            new_input_samples=input_samples,
        )

    def apply_policy(
        self,
        particles_initial_state_mean,
        particles_initial_state_var,
        flg_particles_init_uniform,
        particles_init_up_bound,
        particles_init_low_bound,
        flg_particles_init_multi_gauss,
        num_particles,
        T_control,
        p_dropout=0.0,
    ):
        """
        Override: embed policy speed into initial velocity, optionally
        include wind in particle state for wind-aware mode.

        Gradient chain (preserved):
          cost ← p_final ← GP propagation ← states[0][:,3:6] (v3d)
               ← speed*cos/sin ← speed = policy(state) ← params  ✓
        """
        states_sequence_list = []
        inputs_sequence_list = []

        # --- release position (no grad needed) ---
        ball_pos = torch.tensor(
            self.release_position, dtype=self.dtype, device=self.device
        ).unsqueeze(0).expand(num_particles, -1)                      # [M, 3]

        # --- diverse random targets per particle ---
        targets = torch.tensor(
            np.array([self.target_sampler() for _ in range(num_particles)]),
            dtype=self.dtype, device=self.device,
        )                                                              # [M, 2]

        # --- wind per particle (wind-aware mode only) ---
        if self.wind_aware:
            if self.wind_sampler is not None:
                wind_samples = torch.tensor(
                    np.array([self.wind_sampler() for _ in range(num_particles)]),
                    dtype=self.dtype, device=self.device,
                )                                                      # [M, 2]
            else:
                # Use current wind model output (same wind for all particles)
                self.wind_model.reset()
                w = self.wind_model(0.0)[:2]
                wind_samples = torch.tensor(
                    w, dtype=self.dtype, device=self.device
                ).unsqueeze(0).expand(num_particles, -1)               # [M, 2]

        # --- build policy input ---
        zero_vel = torch.zeros(num_particles, 3, dtype=self.dtype,
                               device=self.device)
        if self.wind_aware:
            # 10-D: [ball_pos, zero_vel, targets, wind]
            policy_input = torch.cat([ball_pos, zero_vel, targets,
                                      wind_samples], dim=1)            # [M, 10]
        else:
            # 8-D: [ball_pos, zero_vel, targets]
            policy_input = torch.cat([ball_pos, zero_vel, targets],
                                      dim=1)                           # [M, 8]

        # --- call policy at t=0 for release speed (with grad) ---
        speed = self.control_policy(policy_input, t=0,
                                     p_dropout=p_dropout)              # [M, 1]

        # --- convert speed → 3-D velocity (torch ops for gradient) ---
        alpha = torch.tensor(
            self.system.launch_angle, dtype=self.dtype, device=self.device
        )
        dx = targets[:, 0:1] - ball_pos[:, 0:1]
        dy = targets[:, 1:2] - ball_pos[:, 1:2]
        azimuth = torch.atan2(dy, dx)                                  # [M, 1]
        vx = speed * torch.cos(alpha) * torch.cos(azimuth)
        vy = speed * torch.cos(alpha) * torch.sin(azimuth)
        vz = speed * torch.sin(alpha)
        v3d = torch.cat([vx, vy, vz], dim=1)                          # [M, 3]

        # --- initial particle state ---
        if self.wind_aware:
            init_particles = torch.cat([ball_pos, v3d, targets,
                                         wind_samples], dim=1)         # [M, 10]
        else:
            init_particles = torch.cat([ball_pos, v3d, targets],
                                         dim=1)                        # [M, 8]

        states_sequence_list.append(init_particles)
        inputs_sequence_list.append(speed)

        # --- GP propagation for t=1..T ---
        zero_input = torch.zeros(num_particles, 1, dtype=self.dtype,
                                  device=self.device)
        landed = torch.zeros(num_particles, dtype=torch.bool,
                              device=self.device)

        for t in range(1, int(T_control)):
            next_particles, _, _ = self.model_learning.get_next_state(
                current_state=states_sequence_list[t - 1],
                current_input=zero_input,
            )
            prev = states_sequence_list[t - 1]
            frozen = torch.where(landed.unsqueeze(1), prev, next_particles)
            landed = landed | (next_particles[:, 2] <= 0.0)
            states_sequence_list.append(frozen)
            inputs_sequence_list.append(zero_input)

        return torch.stack(states_sequence_list), torch.stack(inputs_sequence_list)

    def load_model_from_log(self, num_trial, num_explorations=10, folder="results_tmp/1/"):
        """
        Load model of trial: num_trial from log file inside 'folder'.
        Fixed to account for num_explorations > 1.
        """
        log_file_path = folder + "log.pkl"
        print("\nLoading model from: " + log_file_path)

        log_dict = pkl.load(open(log_file_path, "rb"))
        self.log_dict = log_dict

        self.log_dict["cost_trial_list"] = self.log_dict["cost_trial_list"][0:num_trial]
        self.log_dict["parameters_trial_list"] = self.log_dict["parameters_trial_list"][0:num_trial]
        self.log_dict["particles_states_list"] = self.log_dict["particles_states_list"][0:num_trial]
        self.log_dict["particles_inputs_list"] = self.log_dict["particles_inputs_list"][0:num_trial]

        # Load all explorations plus the completed control trials
        num_to_load = num_explorations + num_trial
        for j in range(num_to_load):
            print("\nGet data from trajectory: " + str(j) + "/" + str(num_to_load))
            state_samples = log_dict["state_samples_history"][j]
            input_samples = log_dict["input_samples_history"][j]
            noiseless_state_samples = log_dict["noiseless_states_history"][j]
            # add samples history
            self.state_samples_history.append(state_samples)
            self.input_samples_history.append(input_samples)
            self.noiseless_states_history.append(noiseless_state_samples)
            self.num_data_collection += 1
            # add data to model_learning object
            self.model_learning.add_data(new_state_samples=state_samples, new_input_samples=input_samples)

        # The last completed trial index for GP is num_explorations + num_trial - 2
        trial_index = num_explorations + num_trial - 2

        # load gp models of trial: trial_index
        print("\nGet parameters for GP trial index:", trial_index)
        self.model_learning.gp_inputs = log_dict["gp_inputs_" + str(trial_index)]
        self.model_learning.gp_output_list = log_dict["gp_output_list_" + str(trial_index)]
        for k in range(self.model_learning.num_gp):
            self.model_learning.gp_list[k].load_state_dict(log_dict["parameters_gp_" + str(trial_index)][k])

        # pre-train gp models
        for k in range(self.model_learning.num_gp):
            self.model_learning.pretrain_gp(k)

