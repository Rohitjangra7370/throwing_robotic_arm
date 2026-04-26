"""
WindThrowingSystem — ballistic simulator with time-varying wind.

Extends the baseline ThrowingSystem by evaluating the wind model at each
Euler integration step, so wind can change during a single throw (gusts,
turbulence).  For constant wind, this is equivalent to the static-wind
baseline.

Returns either:
  - 8-D state [x,y,z, vx,vy,vz, Px,Py]        (wind_aware=False)
  - 10-D state [x,y,z, vx,vy,vz, Px,Py, wx,wy] (wind_aware=True)
"""

import numpy as np

# ──────────────────────────────────────────────────────────────────────
# Physics constants (identical to mc-pilot/simulation_class/model.py)
# ──────────────────────────────────────────────────────────────────────
_G   = 9.81
_RHO = 1.225     # air density (kg/m^3)
_MU  = 1.81e-5   # dynamic viscosity of air (Pa·s)


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
    """Compute ball acceleration: gravity + drag (Eq. 35 from MC-PILOT paper)."""
    A = np.pi * radius ** 2
    v_rel = vel - wind
    v_rel_norm = np.linalg.norm(v_rel)
    cd = _cd_sphere(v_rel_norm, radius)
    F_drag = -0.5 * _RHO * cd * A * v_rel_norm * v_rel
    F_grav = np.array([0.0, 0.0, -mass * _G])
    return (F_drag + F_grav) / mass


class WindThrowingSystem:
    """
    Ballistic throwing simulator with time-varying wind.

    Drop-in replacement for ThrowingSystem with the addition of:
      - A WindModel instance that is queried at each integration step
      - Optional 10-D state output including per-step wind measurements

    Parameters
    ----------
    mass : float
        Ball mass (kg).  Default: 0.0577 (tennis ball).
    radius : float
        Ball radius (m).  Default: 0.0327 (tennis ball).
    launch_angle_deg : float
        Fixed elevation angle of the throw (degrees).
    wind_model : WindModel
        Wind model instance (ConstantWind, GustWind, TurbulentWind).
    wind_aware : bool
        If True, state arrays are 10-D with wind appended.
        If False, state arrays are standard 8-D.
    """

    def __init__(self, mass=0.0577, radius=0.0327, launch_angle_deg=35.0,
                 wind_model=None, wind_aware=False):
        self.mass = mass
        self.radius = radius
        self.launch_angle = np.deg2rad(launch_angle_deg)
        self.wind_model = wind_model
        self.wind_aware = wind_aware

        # Import the calm model as fallback
        if self.wind_model is None:
            from simulation_class.wind_models import WindModel
            self.wind_model = WindModel()  # zero wind

    # ──────────────────────────────────────────────────────────────────
    # Simulation core
    # ──────────────────────────────────────────────────────────────────

    def _simulate(self, release_pos, release_vel, T, dt):
        """
        Euler integration of ballistic flight with time-varying wind.

        Returns:
            pos:  (num_steps, 3)  position trajectory
            vel:  (num_steps, 3)  velocity trajectory
            wind: (num_steps, 2)  wind [wx, wy] at each step
        """
        num_steps = int(T / dt) + 1
        pos  = np.zeros((num_steps, 3))
        vel  = np.zeros((num_steps, 3))
        wind = np.zeros((num_steps, 2))  # record wx, wy per step

        pos[0] = release_pos.copy()
        vel[0] = release_vel.copy()
        w0 = self.wind_model(0.0)
        wind[0] = w0[:2]

        last = num_steps - 1
        for i in range(num_steps - 1):
            t = i * dt
            w = self.wind_model(t)
            wind[i] = w[:2]  # record 2D wind

            accel = _ball_accel(pos[i], vel[i], self.mass, self.radius, w)
            vel[i + 1] = vel[i] + dt * accel
            pos[i + 1] = pos[i] + dt * vel[i] + 0.5 * dt * dt * accel

            # Landing detection: z crosses ground plane (z=0)
            if pos[i + 1, 2] <= 0.0 and i > 0:
                frac = pos[i, 2] / (pos[i, 2] - pos[i + 1, 2])
                pos[i + 1] = pos[i] + frac * (pos[i + 1] - pos[i])
                pos[i + 1, 2] = 0.0
                vel[i + 1] = vel[i] + frac * dt * accel
                # Record wind at landing step
                w_land = self.wind_model(t + frac * dt)
                wind[i + 1] = w_land[:2]
                last = i + 1
                break

        # Fill remaining wind entries for the last valid step
        if last < num_steps - 1:
            wind[last] = wind[max(0, last - 1)]

        return pos[:last + 1], vel[:last + 1], wind[:last + 1]

    # ──────────────────────────────────────────────────────────────────
    # Rollout interface (matches ThrowingSystem API)
    # ──────────────────────────────────────────────────────────────────

    def rollout(self, s0, policy, T, dt, noise):
        """
        Execute one throw episode.

        Parameters
        ----------
        s0 : np.ndarray
            Initial augmented state.
            Wind-blind: [x,y,z, vx,vy,vz, Px,Py] (8-D)
            Wind-aware: [x,y,z, vx,vy,vz, Px,Py, wx,wy] (10-D)
        policy : callable(state, t) -> action
            Control policy (called at t=0 only).
        T : float
            Simulation horizon (seconds).
        dt : float
            Timestep (seconds).
        noise : np.ndarray
            Measurement noise std dev per state dimension.

        Returns
        -------
        noisy_states : np.ndarray (N, state_dim)
        inputs : np.ndarray (N, 1)
        noiseless_states : np.ndarray (N, state_dim)
        """
        # Reset wind model for this episode
        self.wind_model.reset()

        # Extract from augmented state
        release_pos = s0[:3].copy()
        target = s0[6:8].copy()

        # Call policy at t=0 to get release speed
        speed = policy(s0, 0)
        if hasattr(speed, '__len__'):
            speed = float(speed[0])
        else:
            speed = float(speed)

        # Convert speed to 3D velocity
        dx = target[0] - release_pos[0]
        dy = target[1] - release_pos[1]
        azimuth = np.arctan2(dy, dx)
        release_vel = np.array([
            speed * np.cos(self.launch_angle) * np.cos(azimuth),
            speed * np.cos(self.launch_angle) * np.sin(azimuth),
            speed * np.sin(self.launch_angle),
        ])

        # Run simulation with time-varying wind
        pos_traj, vel_traj, wind_traj = self._simulate(
            release_pos, release_vel, T, dt
        )
        N = pos_traj.shape[0]

        # Build augmented state arrays
        if self.wind_aware:
            # 10-D: [x,y,z, vx,vy,vz, Px,Py, wx,wy]
            state_dim = 10
        else:
            # 8-D: [x,y,z, vx,vy,vz, Px,Py]
            state_dim = 8

        noiseless = np.zeros((N, state_dim))
        noiseless[:, 0:3] = pos_traj
        noiseless[:, 3:6] = vel_traj
        noiseless[:, 6:8] = target  # constant across timesteps
        if self.wind_aware:
            noiseless[:, 8:10] = wind_traj  # per-step wind measurement

        # Add measurement noise
        noise_used = noise[:state_dim] if len(noise) >= state_dim else noise
        noisy = noiseless.copy()
        noisy += noise_used * np.random.randn(N, state_dim)
        # Don't add noise to target dims or wind dims (they are known/observed)
        noisy[:, 6:8] = noiseless[:, 6:8]
        if self.wind_aware:
            noisy[:, 8:10] = noiseless[:, 8:10]

        # Input array: speed at t=0, zero otherwise
        inputs = np.zeros((N, 1))
        inputs[0, 0] = speed

        return noisy, inputs, noiseless
