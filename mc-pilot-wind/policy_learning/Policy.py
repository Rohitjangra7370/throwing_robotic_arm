# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Policy classes for mc-pilot-wind.

Includes:
  - All baseline policies from mc-pilot (imported via inheritance)
  - WindAware_Throwing_Policy: conditions speed on both target AND wind
  - Stratified_Throwing_Exploration: re-exported for convenience
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


class Throwing_Policy(Sum_of_gaussians):
    """
    Single-shot RBF policy for the MC-PILOT throwing task (Eq. 21).

    Extracts target (Px, Py) from EXPLICIT indices [6, 7] in the
    augmented state — works for both 8-D and 10-D layouts.

    Output: scalar release speed in [0, u_max]
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
        self.target_indices = [6, 7]  # explicit indices for target

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
        batch = states.shape[0] if states.dim() == 2 else 1
        if t is not None and t > 0:
            return torch.zeros(batch, 1, dtype=self.dtype, device=self.device)

        # Extract target via explicit indices (works for 8-D and 10-D)
        states_2d = states.reshape(-1, self.full_state_dim)
        P = states_2d[:, self.target_indices]   # [batch, 2]

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


class WindAware_Throwing_Policy(Sum_of_gaussians):
    """
    Wind-conditioned RBF policy for the MC-PILOT wind experiment.

    Extracts BOTH target (Px, Py) AND wind (wx, wy) from the 10-D
    augmented state and feeds [Px, Py, wx, wy] (4-D) to the RBF network.

    This lets the policy output **different speeds for the same target
    under different wind conditions** — essential for compensation.

    Augmented state layout (10-D):
        s = [x, y, z, vx, vy, vz, Px, Py, wx, wy]

    RBF input: [Px, Py, wx, wy]  (4-D)
    Output:    scalar release speed in [0, u_max]
    """

    def __init__(
        self,
        full_state_dim,
        target_dim,
        wind_dim,
        num_basis,
        u_max,
        lengthscales_init=None,
        centers_init=None,
        weight_init=None,
        flg_drop=True,
        dtype=torch.float64,
        device=torch.device("cpu"),
    ):
        self.full_state_dim = full_state_dim   # 10
        self.target_dim = target_dim           # 2
        self.wind_dim = wind_dim               # 2
        self.u_max = u_max
        self.target_indices = [6, 7]
        self.wind_indices = [8, 9]

        rbf_input_dim = target_dim + wind_dim  # 4

        super(WindAware_Throwing_Policy, self).__init__(
            state_dim=rbf_input_dim,
            input_dim=1,
            num_basis=num_basis,
            flg_squash=False,
            u_max=u_max,
            lengthscales_init=lengthscales_init,
            centers_init=centers_init,
            weight_init=weight_init,
            flg_drop=flg_drop,
            dtype=dtype,
            device=device,
        )

    def forward(self, states, t=None, p_dropout=0.0):
        batch = states.shape[0] if states.dim() == 2 else 1
        if t is not None and t > 0:
            return torch.zeros(batch, 1, dtype=self.dtype, device=self.device)

        states_2d = states.reshape(-1, self.full_state_dim)
        P = states_2d[:, self.target_indices]    # [batch, 2]
        W = states_2d[:, self.wind_indices]      # [batch, 2]
        rbf_input = torch.cat([P, W], dim=1)     # [batch, 4]

        raw = super(WindAware_Throwing_Policy, self).forward(rbf_input, t=t, p_dropout=p_dropout)

        speed = (self.u_max / 2.0) * (torch.tanh(raw) + 1.0)
        return speed

    def reinit(self, lenghtscales_par, centers_par, weight_par):
        super(WindAware_Throwing_Policy, self).reinit(
            lenghtscales_par=lenghtscales_par,
            centers_par=centers_par,
            weight_par=weight_par,
        )


class Stratified_Throwing_Exploration(Policy):
    """
    Stratified exploration policy for the throwing task.
    Divides [u_min, u_max] into n_strata equal bands.
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
        self._throw_idx = 0

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


class Random_Throwing_Exploration(Policy):
    """
    Random exploration policy for the throwing task.
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
