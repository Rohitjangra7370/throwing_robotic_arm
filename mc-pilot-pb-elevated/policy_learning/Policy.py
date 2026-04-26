# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Authors: 	Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
         	Fabio Amadio (fabioamadio93@gmail.com)
MERL contact:	Diego Romeres (romeres@merl.com)
"""
import numpy as np
import torch
from scipy import signal


class Policy(torch.nn.Module):
    """
    Superclass of policy objects
    """

    def __init__(
        self, state_dim, input_dim, flg_squash=False, u_max=1, dtype=torch.float64, device=torch.device("cpu")
    ):
        super(Policy, self).__init__()
        # model parameters
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.dtype = dtype
        self.device = device
        # set squashing function
        if flg_squash:
            self.f_squash = lambda x: self.squashing(x, u_max)
        else:
            # assign the identity function
            self.f_squash = lambda x: x

    def forward(self, states, t=None, p_dropout=0.0):
        raise NotImplementedError()

    def forward_np(self, state, t=None):
        """
        Numpy implementation of the policy
        """
        input_tc = self(states=torch.tensor(state, dtype=self.dtype, device=self.device), t=t)
        return input_tc.detach().cpu().numpy()

    def to(self, device):
        """
        Move the model parameters to 'device'
        """
        super(Policy, self).to(device)
        self.device = device

    def squashing(self, u, u_max):
        """
        Squash the inputs inside (-u_max, +u_max)
        """
        if np.isscalar(u_max):
            return u_max * torch.tanh(u / u_max)
        else:
            u_max = torch.tensor(u_max, dtype=self.dtype, device=self.device)
            return u_max * torch.tanh(u / u_max)

    def get_np_policy(self):
        """
        Returns a function handle to the numpy version of the policy
        """
        f = lambda state, t: self.forward_np(state, t)

        return f

    def reinit(self, scaling=1):
        raise NotImplementedError()


class Random_exploration(Policy):
    """
    Random control action, uniform dist. (-u_max, +u_max)
    """

    def __init__(
        self, state_dim, input_dim, flg_squash=True, u_max=1.0, dtype=torch.float64, device=torch.device("cpu")
    ):

        super(Random_exploration, self).__init__(
            state_dim=state_dim, input_dim=input_dim, flg_squash=flg_squash, u_max=u_max, dtype=dtype, device=device
        )
        self.u_max = u_max

    def forward(self, states, t):
        # returns random control action
        rand_u = self.u_max * (2 * np.random.rand(self.input_dim) - 1).reshape([-1, self.input_dim])
        return torch.tensor(rand_u, dtype=self.dtype, device=self.device)


class Sum_of_sinusoids(Policy):
    """
    Exploration policy: sum of 'num_sin' sinusoids with random amplitudes and frequencies
    """

    def __init__(
        self,
        state_dim,
        input_dim,
        num_sin,
        omega_min,
        omega_max,
        amplitude_min,
        amplitude_max,
        flg_squash=False,
        u_max=1,
        dtype=torch.float64,
        device=torch.device("cpu"),
    ):
        super(Sum_of_sinusoids, self).__init__(
            state_dim=state_dim, input_dim=input_dim, flg_squash=flg_squash, u_max=u_max, dtype=dtype, device=device
        )
        self.num_sin = num_sin
        amplitude_min = np.array(amplitude_min)
        amplitude_max = np.array(amplitude_max)
        # generate random parameters
        self.amplitudes = torch.nn.Parameter(
            torch.tensor(
                amplitude_min + (amplitude_max - amplitude_min) * np.random.rand(num_sin, input_dim),
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=False,
        )
        self.omega = torch.nn.Parameter(
            torch.tensor(
                np.random.choice([-1, 1], [num_sin, input_dim])
                * (omega_min + (omega_max - omega_min) * np.random.rand(num_sin, input_dim)),
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=False,
        )
        self.phases = torch.nn.Parameter(
            torch.tensor(
                np.random.choice([-1, 1], [num_sin, input_dim]) * (np.pi * (np.random.rand(num_sin, input_dim) - 0.5)),
                dtype=self.dtype,
                device=self.device,
            ),
            requires_grad=False,
        )

    def forward(self, states, t):
        # returns the sinusoid values at time t
        return self.f_squash(
            torch.sum(self.amplitudes * (torch.sin(self.omega * t + self.phases)), dim=0).reshape([-1, self.input_dim])
        )


class Sum_of_gaussians(Policy):
    """
    Control policy: sum of 'num_basis' gaussians
    """

    def __init__(
        self,
        state_dim,
        input_dim,
        num_basis,
        flg_train_lengthscales=True,
        lengthscales_init=None,
        flg_train_centers=True,
        centers_init=None,
        centers_init_min=-1,
        centers_init_max=1,
        weight_init=None,
        flg_train_weight=True,
        flg_bias=False,
        bias_init=None,
        flg_train_bias=False,
        flg_squash=False,
        u_max=1,
        scale_factor=None,
        flg_drop=True,
        dtype=torch.float64,
        device=torch.device("cpu"),
    ):
        super(Sum_of_gaussians, self).__init__(
            state_dim=state_dim, input_dim=input_dim, flg_squash=flg_squash, u_max=u_max, dtype=dtype, device=device
        )
        # set number of gaussian basis functions
        self.num_basis = num_basis
        # get initial log lengthscales
        if lengthscales_init is None:
            lengthscales_init = np.ones(state_dim)
        self.log_lengthscales = torch.nn.Parameter(
            torch.tensor(np.log(lengthscales_init), dtype=self.dtype, device=self.device).reshape([1, -1]),
            requires_grad=flg_train_lengthscales,
        )
        # get initial centers
        if centers_init is None:
            centers_init = centers_init_min * np.ones([num_basis, state_dim]) + (
                centers_init_max - centers_init_min
            ) * np.random.rand(num_basis, state_dim)
        self.centers = torch.nn.Parameter(
            torch.tensor(centers_init, dtype=self.dtype, device=self.device), requires_grad=flg_train_centers
        )
        # initilize the linear ouput layer
        self.f_linear = torch.nn.Linear(in_features=num_basis, out_features=input_dim, bias=flg_bias)
        # check weight initialization
        if not (weight_init is None):
            self.f_linear.weight.data = torch.tensor(weight_init, dtype=dtype, device=device)
        else:
            self.f_linear.weight.data = torch.tensor(np.ones([input_dim, num_basis]), dtype=dtype, device=device)

        self.f_linear.weight.requires_grad = flg_train_weight
        # check bias initialization
        if flg_bias:
            self.f_linear.bias.requires_grad = flg_train_bias
            if not (bias_init is None):
                self.f_linear.bias.data = torch.tensor(bias_init)
        # set type and device
        self.f_linear.type(self.dtype)
        self.f_linear.to(self.device)

        if scale_factor is None:
            scale_factor = np.ones(state_dim)
        self.scale_factor = torch.tensor(scale_factor, dtype=self.dtype, device=self.device).reshape([1, -1])

        # set dropout
        if flg_drop == True:
            self.f_drop = torch.nn.functional.dropout
        else:
            self.f_drop = lambda x, p: x

    def reinit(self, lenghtscales_par, centers_par, weight_par):
        self.log_lengthscales.data = torch.tensor(
            np.log(lenghtscales_par), dtype=self.dtype, device=self.device
        ).reshape([1, -1])
        self.centers.data = (
            torch.tensor(centers_par, dtype=self.dtype, device=self.device)
            * 2
            * (torch.rand(self.num_basis, self.state_dim, dtype=self.dtype, device=self.device) - 0.5)
        )
        self.f_linear.weight.data = weight_par * (
            torch.rand(self.input_dim, self.num_basis, dtype=self.dtype, device=self.device) - 0.5
        )

    def forward(self, states, t=None, p_dropout=0.0):
        """
        Returns a linear combination of gaussian functions
        with input given by the the distances between that state
        and the vector of centers of the gaussian functions
        """
        # get the lengthscales from log
        lengthscales = torch.exp(self.log_lengthscales)
        # unsqueeze states
        states = states.reshape([-1, self.state_dim]).unsqueeze(1)
        states = states / self.scale_factor
        # normalize states and centers
        norm_states = states / lengthscales
        norm_centers = self.centers / lengthscales
        # get the square distance
        dist = torch.sum(norm_states**2, dim=2, keepdim=True)
        dist = dist + torch.sum(norm_centers**2, dim=1, keepdim=True).transpose(0, 1)
        dist -= 2 * torch.matmul(norm_states, norm_centers.transpose(dim0=0, dim1=1))
        # apply exp and get output
        exp_dist_dropped = self.f_drop(torch.exp(-dist), p_dropout)
        inputs = self.f_linear(exp_dist_dropped).reshape([-1, self.input_dim])

        # returns the constrained control action
        return self.f_squash(inputs)


class Sum_of_gaussians_with_angles(Sum_of_gaussians):
    """
    Extends sum of gaussians policy. Angle indices are mapped in cos and sin before computing the policy
    """

    def __init__(
        self,
        state_dim,
        input_dim,
        num_basis,
        angle_indices,
        non_angle_indices,
        flg_train_lengthscales=True,
        lengthscales_init=None,
        flg_train_centers=True,
        centers_init=None,
        centers_init_min=-1,
        centers_init_max=1,
        weight_init=None,
        flg_train_weight=True,
        flg_bias=False,
        bias_init=None,
        flg_train_bias=False,
        flg_squash=False,
        u_max=1,
        flg_drop=True,
        dtype=torch.float64,
        device=torch.device("cpu"),
    ):
        self.angle_indices = angle_indices
        self.non_angle_indices = non_angle_indices
        self.num_angle_indices = angle_indices.size
        self.num_non_angle_indices = non_angle_indices.size
        super(Sum_of_gaussians_with_angles, self).__init__(
            state_dim=state_dim + self.num_angle_indices,
            input_dim=input_dim,
            num_basis=num_basis,
            flg_train_lengthscales=flg_train_lengthscales,
            lengthscales_init=lengthscales_init,
            flg_train_centers=flg_train_centers,
            centers_init=centers_init,
            centers_init_min=centers_init_min,
            centers_init_max=centers_init_max,
            weight_init=weight_init,
            flg_train_weight=flg_train_weight,
            flg_bias=flg_bias,
            bias_init=bias_init,
            flg_train_bias=flg_train_bias,
            flg_squash=flg_squash,
            u_max=u_max,
            flg_drop=flg_drop,
            dtype=dtype,
            device=device,
        )

    def forward(self, states, t=None, p_dropout=0.0):
        # build a state with non angle features and cos,sin of angle features
        states = states.reshape([-1, self.state_dim - self.num_angle_indices])
        new_state = torch.cat(
            [
                states[:, self.non_angle_indices],
                torch.cos(states[:, self.angle_indices]),
                torch.sin(states[:, self.angle_indices]),
            ],
            1,
        )
        # call the forward method of the superclass
        return super().forward(new_state, t=t, p_dropout=p_dropout)


class Sum_of_gaussians_with_target_trajectory(Sum_of_gaussians):
    """
    Use sum of gaussians policy with a reference target trajectory. It considers and extended state: [state, target-state]
    """

    def __init__(
        self,
        state_dim,
        input_dim,
        num_basis,
        target_traj,
        flg_train_lengthscales=True,
        lengthscales_init=None,
        flg_train_centers=True,
        centers_init=None,
        centers_init_min=-1,
        centers_init_max=1,
        weight_init=None,
        flg_train_weight=True,
        flg_bias=False,
        bias_init=None,
        flg_train_bias=False,
        flg_squash=False,
        u_max=1,
        flg_drop=True,
        dtype=torch.float64,
        device=torch.device("cpu"),
    ):
        super(Sum_of_gaussians_with_target_trajectory, self).__init__(
            state_dim=state_dim,
            input_dim=input_dim,
            num_basis=num_basis,
            flg_train_lengthscales=flg_train_lengthscales,
            lengthscales_init=lengthscales_init,
            flg_train_centers=flg_train_centers,
            centers_init=centers_init,
            centers_init_min=centers_init_min,
            centers_init_max=centers_init_max,
            weight_init=weight_init,
            flg_train_weight=flg_train_weight,
            flg_bias=flg_bias,
            bias_init=bias_init,
            flg_train_bias=flg_train_bias,
            flg_squash=flg_squash,
            u_max=u_max,
            flg_drop=flg_drop,
            dtype=dtype,
            device=device,
        )
        self.target_traj = torch.tensor(target_traj, dtype=self.dtype, device=self.device)

    def forward(self, states, t=None, p_dropout=0.0):
        if states.dim() == 1:
            # single state
            target = self.target_traj[t, :]
            policy_in = torch.cat((states, target - states), 0)
            # policy_in = target-states
        elif states.dim() == 2:
            # particles batch
            target = self.target_traj[t, :]
            particle_targets = target.repeat(1, states.shape[0]).view(states.shape)
            policy_in = torch.cat((states, particle_targets - states), 1)
            # policy_in = particle_targets-states
        u = super().forward(policy_in, t=t, p_dropout=p_dropout)

        return u


class PD_controller(Policy):
    """
    PD controller (N.B. it takes error as input)
    """

    def __init__(
        self,
        state_dim,
        input_dim,
        sqrt_Kp_gains,
        sqrt_Kd_gains,
        target_traj=None,
        flg_squash=True,
        u_max=1.0,
        flg_trainable=False,
        dtype=torch.float64,
        device=torch.device("cpu"),
    ):

        super(PD_controller, self).__init__(
            state_dim=state_dim, input_dim=input_dim, flg_squash=flg_squash, u_max=u_max, dtype=dtype, device=device
        )

        self.target_traj = target_traj
        self.sqrt_Kp_gains = torch.nn.Parameter(
            torch.tensor(sqrt_Kp_gains, dtype=self.dtype, device=self.device), requires_grad=flg_trainable
        )
        self.sqrt_Kd_gains = torch.nn.Parameter(
            torch.tensor(sqrt_Kd_gains, dtype=self.dtype, device=self.device), requires_grad=flg_trainable
        )

    def forward(self, states, t, p_dropout=0.0):
        """
        in states we pass actually the error: target-states
        """
        states = states.reshape([-1, self.state_dim])
        target = self.target_traj[t, :]

        particle_targets = target.repeat(1, states.shape[0]).view(states.shape)
        err = particle_targets - states
        Kp_gains = self.sqrt_Kp_gains**2
        Kd_gains = self.sqrt_Kd_gains**2
        inputs = Kp_gains * err[:, 0 : int(self.state_dim / 2)] + Kd_gains * err[:, int(self.state_dim / 2) :]
        return self.f_squash(inputs)


class Throwing_Policy(Sum_of_gaussians):
    """
    Single-shot RBF policy for the MC-PILOT throwing task (Eq. 21).

    The policy is called at every timestep by MC-PILCO's apply_policy() loop,
    but only produces a non-zero output at t=0 (release). For t>0 it returns
    zeros, meaning no control input during free flight.

    Augmented state layout:
        s = [x, y, z, vx, vy, vz, Px, Py]   (8-D)
    The policy extracts target (Px, Py) from the last 2 dims and feeds
    only those to the RBF network.

    Output: scalar release speed in [0, u_max] using the paper's
    squashing: (u_max/2) * (tanh(raw) + 1)

    Parameters
    ----------
    full_state_dim : int   total augmented state dim (8)
    target_dim     : int   target position dim (2 for ground plane)
    num_basis      : int   number of RBF basis functions (250 from Table 1)
    u_max          : float max release speed in m/s (3.5 from Table 1)
    """

    def __init__(
        self,
        full_state_dim,
        target_dim,
        num_basis,
        u_max,
        lengthscales_init=None,
        centers_init=None,
        weight_init=None,
        flg_drop=True,
        dtype=torch.float64,
        device=torch.device("cpu"),
    ):
        self.full_state_dim = full_state_dim
        self.target_dim = target_dim
        self.u_max = u_max

        # RBF operates on target P only (2-D input, 1-D output for speed)
        super(Throwing_Policy, self).__init__(
            state_dim=target_dim,
            input_dim=1,
            num_basis=num_basis,
            flg_squash=False,   # custom squash applied below
            u_max=u_max,
            lengthscales_init=lengthscales_init,
            centers_init=centers_init,
            weight_init=weight_init,
            flg_drop=flg_drop,
            dtype=dtype,
            device=device,
        )

    def forward(self, states, t=None, p_dropout=0.0):
        """
        states : [batch, full_state_dim] augmented state
        t      : timestep index (int or None)
        Returns: [batch, 1] release speed (zeros for t > 0)
        """
        batch = states.shape[0] if states.dim() == 2 else 1
        if t is not None and t > 0:
            return torch.zeros(batch, 1, dtype=self.dtype, device=self.device)

        # Extract target position P from augmented state
        states_2d = states.reshape(-1, self.full_state_dim)
        P = states_2d[:, -self.target_dim:]   # [batch, target_dim]

        # RBF output (raw, before squashing)
        raw = super(Throwing_Policy, self).forward(P, t=t, p_dropout=p_dropout)

        # Paper's squashing: (uM/2)*(tanh(raw) + 1)  →  speed in [0, uM]
        speed = (self.u_max / 2.0) * (torch.tanh(raw) + 1.0)
        return speed

    def reinit(self, lenghtscales_par, centers_par, weight_par):
        """Delegate reinit to parent (operates on target_dim-space)."""
        super(Throwing_Policy, self).reinit(
            lenghtscales_par=lenghtscales_par,
            centers_par=centers_par,
            weight_par=weight_par,
        )


class Random_Throwing_Exploration(Policy):
    """
    Random exploration policy for the throwing task.

    At t=0: returns a random release speed uniformly sampled from [0, u_max].
    At t>0: returns zeros (no control during free flight).
    """

    def __init__(
        self,
        full_state_dim,
        u_max,
        dtype=torch.float64,
        device=torch.device("cpu"),
    ):
        super(Random_Throwing_Exploration, self).__init__(
            state_dim=full_state_dim,
            input_dim=1,
            flg_squash=False,
            u_max=u_max,
            dtype=dtype,
            device=device,
        )
        self.u_max = u_max

    def forward(self, states, t=None, p_dropout=0.0):
        batch = states.shape[0] if states.dim() == 2 else 1
        if t is not None and t > 0:
            return torch.zeros(batch, 1, dtype=self.dtype, device=self.device)
        speed = self.u_max * torch.rand(batch, 1, dtype=self.dtype, device=self.device)
        return speed


class Stratified_Throwing_Exploration(Policy):
    """
    Stratified exploration policy for the throwing task.

    Divides [u_min, u_max] into n_strata equal bands. Each successive throw at
    t=0 samples speed uniformly from the next band, cycling through all bands
    in order. This guarantees the GP training data covers the full useful speed
    range even with a small number of exploration throws.

    u_min defaults to 0. Set u_min > 0 when the target range requires only the
    upper portion of [0, u_max] (e.g. z=0.0m targets cluster near max range).

    At t>0: returns zeros (no control during free flight).
    """

    def __init__(
        self,
        full_state_dim,
        u_max,
        n_strata,
        u_min=0.0,
        dtype=torch.float64,
        device=torch.device("cpu"),
    ):
        super(Stratified_Throwing_Exploration, self).__init__(
            state_dim=full_state_dim,
            input_dim=1,
            flg_squash=False,
            u_max=u_max,
            dtype=dtype,
            device=device,
        )
        self.u_max    = u_max
        self.u_min    = u_min
        self.n_strata = n_strata
        self._throw_idx = 0   # increments once per throw (at t=0)

    def forward(self, states, t=None, p_dropout=0.0):
        batch = states.shape[0] if states.dim() == 2 else 1
        if t is not None and t > 0:
            return torch.zeros(batch, 1, dtype=self.dtype, device=self.device)
        stratum = self._throw_idx % self.n_strata
        band = (self.u_max - self.u_min) / self.n_strata
        lo = self.u_min + stratum       * band
        hi = self.u_min + (stratum + 1) * band
        speed = lo + (hi - lo) * torch.rand(1, dtype=self.dtype, device=self.device)
        self._throw_idx += 1
        return speed.expand(batch, 1)

    def reinit(self, **kwargs):
        self._throw_idx = 0


class Baseline_Throwing_Exploration(Policy):
    """
    Baseline exploration policy for the throwing task (Eq. 13 of MC-PILOT paper).

    Computes the analytically correct release speed from ballistic equations,
    assuming point-mass projectile with no drag:

        π(P) = sqrt(g · d² / (2 · cos²(α) · (d · tan(α) - z_P + z_rel)))

    This gives the GP data from throws that actually reach the target region,
    improving model coverage across the full target space.
    """

    def __init__(
        self,
        full_state_dim,
        u_max,
        launch_angle,
        release_height,
        dtype=torch.float64,
        device=torch.device("cpu"),
    ):
        super(Baseline_Throwing_Exploration, self).__init__(
            state_dim=full_state_dim,
            input_dim=1,
            flg_squash=False,
            u_max=u_max,
            dtype=dtype,
            device=device,
        )
        self.u_max = u_max
        self.launch_angle = launch_angle
        self.release_height = release_height
        self._g = 9.81

    def forward(self, states, t=None, p_dropout=0.0):
        batch = states.shape[0] if states.dim() == 2 else 1
        states_2d = states.view(batch, -1)
        if t is not None and t > 0:
            return torch.zeros(batch, 1, dtype=self.dtype, device=self.device)

        # Extract target P from augmented state (last 2 dims)
        Px = states_2d[:, -2]
        Py = states_2d[:, -1]
        # Release position (first 3 dims)
        x0 = states_2d[:, 0]
        y0 = states_2d[:, 1]

        dx = Px - x0
        dy = Py - y0
        d = torch.sqrt(dx ** 2 + dy ** 2)

        alpha = self.launch_angle
        z_rel = self.release_height
        tan_a = np.tan(alpha)
        cos_a = np.cos(alpha)

        # Eq. 13: v = sqrt(g * d^2 / (2 * cos^2(a) * (d*tan(a) - z_P + z_rel)))
        # z_P = 0 for ground targets
        denom = d * tan_a + z_rel
        # Clamp denominator to avoid division by zero or negative (infeasible targets)
        denom = torch.clamp(denom, min=0.01)
        speed_sq = self._g * d ** 2 / (2.0 * cos_a ** 2 * denom)
        speed = torch.sqrt(speed_sq)
        # Clamp to [0, u_max]
        speed = torch.clamp(speed, min=0.0, max=self.u_max)

        return speed.unsqueeze(1)
