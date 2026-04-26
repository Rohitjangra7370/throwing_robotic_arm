# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Authors: 	Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
         	Fabio Amadio (fabioamadio93@gmail.com)
MERL contact:	Diego Romeres (romeres@merl.com)
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.integrate import odeint


# ---------------------------------------------------------------------------
# Ball physical constants (tennis ball defaults)
# ---------------------------------------------------------------------------
_RHO = 1.225        # air density kg/m^3
_MU  = 1.81e-5      # dynamic viscosity kg/(m·s)
_G   = 9.81         # gravitational acceleration m/s^2


def _cd_sphere(v_norm, radius):
    """Drag coefficient for a smooth sphere via Reynolds number."""
    Re = v_norm * 2.0 * radius / _MU * _RHO
    if Re < 1e-6:
        return 0.47
    elif Re < 2e5:
        return 0.47
    else:
        return 0.2


def _ball_accel(pos, vel, mass, radius, wind):
    """Compute ball acceleration: gravity + drag + wind (Eq. 35 from MC-PILOT paper)."""
    v_norm = np.linalg.norm(vel)
    A = np.pi * radius ** 2
    cd = _cd_sphere(v_norm, radius)
    # Aerodynamic drag on ball relative to air
    v_rel = vel - wind
    v_rel_norm = np.linalg.norm(v_rel)
    F_drag = -0.5 * _RHO * cd * A * v_rel_norm * v_rel
    F_grav = np.array([0.0, 0.0, -mass * _G])
    return (F_drag + F_grav) / mass


class Model:
    """
    Dynamic System simulation
    """

    def __init__(self, fcn):
        """
        fcn: function defining the dynamics dx/dt = fcn(x, t, policy)
        x: state (it can be an array)
        t: time instant
        policy: function u = policy(x)
        """
        self.fcn = fcn  # ODE of system dynamics

    def rollout(self, s0, policy, T, dt, noise):
        """
        Generate a rollout of length T (s)  with control inputs computed by 'policy' and applied with a sampling time 'dt'.
        'noise' defines the standard deviation of a Gaussian measurement noise.
            s0: initial state
            policy: policy function
            T: length rollout (s)
            dt: sampling time (s)
        """
        state_dim = len(s0)
        time = np.linspace(0, T, int(T / dt) + 1)
        num_samples = len(time)

        # get first input
        u0 = np.array(policy(s0, 0.0))

        num_inputs = u0.size
        # init variables
        inputs = np.zeros([num_samples, num_inputs])
        states = np.zeros([num_samples, state_dim])
        noisy_states = np.zeros([num_samples, state_dim])
        states[0, :] = s0
        noisy_states[0, :] = s0 + np.random.randn(state_dim) * noise

        for i, t in enumerate(time[:-1]):
            # get input
            u = np.array(policy(noisy_states[i, :], t)).flatten()
            inputs[i, :] = u
            # get state
            u_arg = u.item() if u.size == 1 else u
            odeint_out = odeint(self.fcn, states[i, :], [t, t + dt], args=(u_arg,))
            states[i + 1, :] = odeint_out[1]
            noisy_states[i + 1, :] = odeint_out[1] + np.random.randn(state_dim) * noise

        # last u (only to have the same number of input and state samples)
        inputs[-1, :] = np.array(policy(noisy_states[-1, :], T))

        return noisy_states, inputs, states


class PMS_Model:
    """
    Partially Measurable System simulation
    """

    def __init__(self, fcn, filtering_dict):
        """
        fcn: function defining the dynamics dx/dt = fcn(x, t, policy)
        x: state (it can be an array)
        t: time instant
        policy: function u = policy(x)
        In filtering dict are passed the parameters of the online filter
        """
        self.fcn = fcn
        self.filtering_dict = filtering_dict

    def rollout(self, s0, policy, T, dt, noise, vel_indeces, pos_indeces):
        """
        Generate a rollout of length T (s) for the system defined by 'fcn' with
        control inputs computed by 'policy' and applied with a sampling time 'dt'.
        'noise' defines the standard deviation of a Gaussian measurement noise.
        In this implementation we simulate the interaction with a real mechanical system where
        velocities cannot be measured, but only inferred from the positions.
            s0: initial state
            policy: policy function
            T: length rollout (s)
            dt: sampling time (s)

        """
        state_dim = len(s0)
        time = np.linspace(0, T, int(T / dt) + 1)
        num_samples = len(time)
        # get input size
        num_inputs = np.array(policy(s0, 0.0)).size
        # allocate the space
        inputs = np.zeros([num_samples, num_inputs])
        states = np.zeros([num_samples, state_dim])
        noisy_states = np.zeros([num_samples, state_dim])
        meas_states = np.zeros([num_samples, state_dim])
        # initialize state vectors
        states[0, :] = s0
        noisy_states[0, :] = s0
        meas_states[0, :] = np.copy(noisy_states[0, :])

        # init low-pass filter
        b, a = signal.butter(1, self.filtering_dict["fc"])

        for i, t in enumerate(time[:-1]):
            # get input
            u = np.array(policy(meas_states[i, :], t)).flatten()
            inputs[i, :] = u
            # get state
            u_arg = u.item() if u.size == 1 else u
            odeint_out = odeint(self.fcn, states[i, :], [t, t + dt], args=(u_arg,))
            states[i + 1, :] = odeint_out[1]
            noisy_states[i + 1, :] = odeint_out[1] + np.random.randn(state_dim) * noise

            # positions are measured directly
            meas_states[i + 1, pos_indeces] = noisy_states[i + 1, pos_indeces]
            # velocities are estimated online by causal numerical differentiation ...
            noisy_states[i + 1, vel_indeces] = (meas_states[i + 1, pos_indeces] - meas_states[i, pos_indeces]) / dt
            # ... and low-pass filtered
            meas_states[i + 1, vel_indeces] = (
                b[0] * noisy_states[i + 1, vel_indeces]
                + b[1] * noisy_states[i, vel_indeces]
                - a[1] * meas_states[i, vel_indeces]
            ) / a[0]

        # last u (only to have the same number of input and state samples)
        inputs[-1, :] = np.array(policy(meas_states[-1, :], T))

        return meas_states, inputs, states, noisy_states


class ThrowingSystem:
    """
    Single-shot ball throw simulator (replaces ODE-based Model for MC-PILOT).

    State layout (augmented):
        s = [x, y, z, vx, vy, vz, Px, Py]   (8-D)
            0:3  ball position  (release point on first call, then evolves)
            3:6  ball velocity  (set by policy at t=0)
            6:8  target x,y on ground plane (constant throughout rollout)

    The policy is called ONCE at t=0 to produce a scalar release *speed*.
    Direction is derived from target geometry:
        azimuth  φ = atan2(Py - z0_y, Px - z0_x)
        elevation α = fixed launch angle (default 45° for maximum range,
                      clipped to ensure ball can reach target)
    Then: vx = speed*cos(α)*cos(φ), vy = speed*cos(α)*sin(φ), vz = speed*sin(α)

    For t > 0, the policy returns 0 (no control during free flight).

    rollout() returns the same 3-tuple as Model.rollout() so it plugs in
    to MC-PILCO's get_data_from_system() without modification.
    """

    def __init__(self, mass=0.0577, radius=0.0327, launch_angle_deg=45.0,
                 wind=None):
        """
        mass            : ball mass in kg
        radius          : ball radius in m
        launch_angle_deg: fixed elevation angle in degrees (paper fixes α≈0
                          for horizontal throw; 45° maximises range)
        wind            : optional np.array [wx, wy, wz] m/s (Exploration 2)
        """
        self.mass = mass
        self.radius = radius
        self.launch_angle = np.deg2rad(launch_angle_deg)
        self.wind = np.zeros(3) if wind is None else np.array(wind, dtype=float)

    def _speed_to_velocity(self, speed, release_pos, target_xy):
        """Convert scalar speed + target geometry to 3-D release velocity."""
        dx = target_xy[0] - release_pos[0]
        dy = target_xy[1] - release_pos[1]
        azimuth = np.arctan2(dy, dx)
        alpha = self.launch_angle
        vx = speed * np.cos(alpha) * np.cos(azimuth)
        vy = speed * np.cos(alpha) * np.sin(azimuth)
        vz = speed * np.sin(alpha)
        return np.array([vx, vy, vz])

    def _simulate(self, release_pos, release_vel, T, dt):
        """
        Euler integration of ballistic flight.
        Returns arrays of shape [num_steps, 6] for (pos, vel) at each timestep.
        Simulation terminates at T seconds or when z <= 0 (landing).
        """
        num_steps = int(T / dt) + 1
        pos = np.zeros((num_steps, 3))
        vel = np.zeros((num_steps, 3))
        pos[0] = release_pos.copy()
        vel[0] = release_vel.copy()

        last = num_steps - 1
        for i in range(num_steps - 1):
            accel = _ball_accel(pos[i], vel[i], self.mass, self.radius, self.wind)
            vel[i + 1] = vel[i] + dt * accel
            pos[i + 1] = pos[i] + dt * vel[i] + 0.5 * dt * dt * accel
            # landing detection: z crosses ground plane (z=0)
            if pos[i + 1, 2] <= 0.0 and i > 0:
                # linear interpolation to find exact landing position
                frac = pos[i, 2] / (pos[i, 2] - pos[i + 1, 2])
                pos[i + 1] = pos[i] + frac * (pos[i + 1] - pos[i])
                pos[i + 1, 2] = 0.0
                vel[i + 1] = vel[i] + frac * dt * accel
                last = i + 1
                break

        return pos[:last + 1], vel[:last + 1]

    def rollout(self, s0, policy, T, dt, noise):
        """
        Simulate one throw.  Matches Model.rollout() signature exactly.

        s0     : augmented initial state [x,y,z, 0,0,0, Px,Py]  (8-D)
        policy : callable(state, t) -> scalar speed (only used at t=0)
        T      : total simulation horizon (s)
        dt     : timestep (s)
        noise  : measurement noise std (scalar or array of length state_dim)

        Returns
        -------
        noisy_states : [num_steps, 8]  ball trajectory + target (with noise)
        inputs       : [num_steps, 1]  scalar speed at t=0, zeros after
        clean_states : [num_steps, 8]  noiseless trajectory
        """
        state_dim = len(s0)
        release_pos = np.array(s0[0:3], dtype=float)
        target_xy   = np.array(s0[6:8], dtype=float)

        # --- call policy once at t=0 to get release speed ---
        u0 = np.array(policy(s0, 0.0)).flatten()
        speed = float(u0[0])

        release_vel = self._speed_to_velocity(speed, release_pos, target_xy)

        # --- simulate free flight ---
        pos_traj, vel_traj = self._simulate(release_pos, release_vel, T, dt)
        n = len(pos_traj)

        # --- build state arrays ---
        target_col = np.tile(target_xy, (n, 1))       # [n, 2] constant
        clean_ball = np.hstack([pos_traj, vel_traj])   # [n, 6]
        clean_states = np.hstack([clean_ball, target_col])  # [n, 8]

        noise_arr = np.ones(state_dim) * noise if np.isscalar(noise) else np.array(noise)
        noisy_states = clean_states + np.random.randn(n, state_dim) * noise_arr

        # inputs: speed at t=0, zeros after
        inputs = np.zeros((n, 1))
        inputs[0, 0] = speed

        return noisy_states, inputs, clean_states
