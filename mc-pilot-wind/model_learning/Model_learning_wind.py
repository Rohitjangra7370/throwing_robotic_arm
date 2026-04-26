"""
Wind-aware GP model for MC-PILOT ballistic dynamics.

Extends Ballistic_Model_learning_RBF to optionally include measured wind
(wx, wy) as additional GP input.  This lets the GP condition its velocity
predictions on the current wind, which is necessary for time-varying wind
scenarios (gusts, turbulence).

Two modes:
  1. Wind-blind (standard):  GP input = [x,y,z, vx,vy,vz]  (6-D)
     State layout: [x,y,z, vx,vy,vz, Px,Py]  (8-D)

  2. Wind-aware (new):  GP input = [x,y,z, vx,vy,vz, wx,wy]  (8-D)
     State layout: [x,y,z, vx,vy,vz, Px,Py, wx,wy]  (10-D)
"""

import torch
from torch.distributions.normal import Normal

import model_learning.Model_learning as ML


class WindAware_Ballistic_Model_learning_RBF(ML.Speed_Model_learning_RBF_angle_state):
    """
    GP model for ball free-flight dynamics with wind input.

    Augmented state layout:
        s = [x, y, z, vx, vy, vz, Px, Py, wx, wy]   (10-D)

    GP setup:
        num_gp   = 3          (one per: delta_vx, delta_vy, delta_vz)
        GP input  = [x, y, z, vx, vy, vz, wx, wy]  (8-D)
        GP output = [delta_vx, delta_vy, delta_vz]

    The target dims (6:8) and wind dims (8:10) are carried forward
    unchanged in get_next_state_from_gp_output().
    """

    BALL_STATE_DIM = 6   # dims 0:6 (position + velocity)
    TARGET_DIM     = 2   # dims 6:8 (Px, Py)
    WIND_DIM       = 2   # dims 8:10 (wx, wy)
    GP_INPUT_DIM   = 8   # ball_state(6) + wind(2)

    def __init__(
        self,
        num_gp,
        init_dict_list,
        T_sampling,
        approximation_mode=None,
        approximation_dict=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
        flg_norm=False,
    ):
        # vel_indeces and not_vel_indeces refer to ball-state dims only
        super(WindAware_Ballistic_Model_learning_RBF, self).__init__(
            num_gp=num_gp,
            init_dict_list=init_dict_list,
            T_sampling=T_sampling,
            angle_indeces=[],
            not_angle_indeces=[],
            vel_indeces=[3, 4, 5],
            not_vel_indeces=[0, 1, 2],
            approximation_mode=approximation_mode,
            approximation_dict=approximation_dict,
            dtype=dtype,
            device=device,
            flg_norm=flg_norm,
        )

    def data_to_gp_input(self, states, inputs):
        """
        GP input = [ball_state(6), wind(2)] = dims 0:6 + 8:10.
        Strips target (dims 6:8) and control input from GP input.
        """
        ball_state = states[:, 0:self.BALL_STATE_DIM]        # [N, 6]
        wind_state = states[:, self.BALL_STATE_DIM + self.TARGET_DIM:]  # [N, 2]
        return torch.cat([ball_state, wind_state], dim=1)    # [N, 8]

    def get_next_state_from_gp_output(
        self, current_state, current_input,
        gp_output_mean_list, gp_output_var_list,
        particle_pred=True
    ):
        """
        Integrate velocity changes; carry target AND wind dims forward unchanged.
        """
        delta_vel_mean = torch.cat(gp_output_mean_list, 1)
        delta_vel_var  = torch.cat(gp_output_var_list, 1)

        next_states = torch.zeros(current_state.shape, dtype=self.dtype, device=self.device)

        if particle_pred:
            dist = Normal(delta_vel_mean, torch.sqrt(delta_vel_var))
            delta_speed_sample = dist.rsample()
        else:
            delta_speed_sample = delta_vel_mean

        # Update velocity (dims 3:6)
        next_states[:, self.vel_indeces] = (
            current_state[:, self.vel_indeces] + delta_speed_sample
        )
        # Update position (dims 0:3) using velocity verlet
        next_states[:, self.not_vel_indeces] = (
            current_state[:, self.not_vel_indeces]
            + self.T_sampling * current_state[:, self.vel_indeces]
            + self.T_sampling / 2 * delta_speed_sample
        )
        # Carry target forward unchanged (dims 6:8)
        next_states[:, self.BALL_STATE_DIM:self.BALL_STATE_DIM + self.TARGET_DIM] = (
            current_state[:, self.BALL_STATE_DIM:self.BALL_STATE_DIM + self.TARGET_DIM]
        )
        # Carry wind forward unchanged (dims 8:10)
        next_states[:, self.BALL_STATE_DIM + self.TARGET_DIM:] = (
            current_state[:, self.BALL_STATE_DIM + self.TARGET_DIM:]
        )

        return next_states, delta_vel_mean, delta_vel_var
