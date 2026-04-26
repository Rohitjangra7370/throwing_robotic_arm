"""
PyBulletThrowingSystem — drop-in replacement for ThrowingSystem.

Same constructor signature, same rollout() return format, same 8-D state layout.
KUKA iiwa7 arm physically executes the throw in a fresh PyBullet DIRECT world
each rollout.  Ball free-flight uses the paper's Eq. 35 drag (via _ball_accel
from model.py), NOT PyBullet's built-in damping, so results are directly
comparable to the numpy sim.

Key design choices:
  - Ball velocity at release is set explicitly to v_cmd + dv_noise via
    resetBaseVelocity.  This decouples ball-physics accuracy from arm-tracking
    quality and gives a clean, calibration-free match with the noise distribution
    used in MC_PILOT.apply_policy().
  - Arm motion is retained (cosmetic + enables GUI visualisation).
  - p.DIRECT mode during training; swap to p.GUI in demo_pybullet_gui.py.
"""

import numpy as np
import pybullet as p
import pybullet_data

from robot_arm.arm_controller import ArmController
from simulation_class.model import _ball_accel


# Timing defaults for arm phases (seconds)
_T_W   = 0.30   # windup phase end
_T_R   = 0.60   # release time
_T_ARM = 1.20   # total arm trajectory time (follow-through)


class PyBulletThrowingSystem:
    """
    Drop-in replacement for ThrowingSystem.

    Parameters
    ----------
    mass, radius, launch_angle_deg, wind
        Same as ThrowingSystem.
    arm_noise : ArmNoise or None
        Noise model applied at release.  VelocityBiasNoise recommended.
        When None, ball velocity exactly equals v_cmd (zero noise).
    t_w, t_r : float
        Phase boundary times for the arm trajectory (seconds).
    gui_mode : bool
        If True, connect with p.GUI instead of p.DIRECT.  Only for demo use —
        training must use DIRECT.
    """

    def __init__(
        self,
        mass=0.0577,
        radius=0.0327,
        launch_angle_deg=35.0,
        wind=None,
        arm_noise=None,
        t_w=_T_W,
        t_r=_T_R,
        gui_mode=False,
    ):
        self.mass = mass
        self.radius = radius
        self.launch_angle = np.deg2rad(launch_angle_deg)   # must match ThrowingSystem API
        self.wind = np.zeros(3) if wind is None else np.array(wind, dtype=float)
        self.arm_noise = arm_noise
        self.t_w = t_w
        self.t_r = t_r
        self._gui_mode = gui_mode
        self._urdf_path = pybullet_data.getDataPath() + "/kuka_iiwa/model.urdf"
        self._plane_urdf = "plane.urdf"

    # ------------------------------------------------------------------
    # Public API (matches ThrowingSystem)
    # ------------------------------------------------------------------

    def rollout(self, s0, policy, T, dt, noise):
        """
        Simulate one throw in PyBullet.

        Parameters  (identical to ThrowingSystem.rollout)
        ----------
        s0     : augmented initial state [x,y,z, 0,0,0, Px,Py]  (8-D numpy)
        policy : callable(state, t) -> scalar speed  (only called at t=0)
        T      : simulation horizon for FREE FLIGHT (s); arm uses _T_ARM >= T
        dt     : timestep (s) — must match Ts in the test script
        noise  : measurement noise std (scalar or 8-D array)

        Returns
        -------
        noisy_states : [n, 8]
        inputs       : [n, 1]
        clean_states : [n, 8]
        """
        release_pos = np.array(s0[0:3], dtype=float)
        target_xy   = np.array(s0[6:8], dtype=float)
        state_dim   = len(s0)

        # Call policy once at t=0 to get release speed (scalar)
        u0    = np.array(policy(s0, 0.0)).flatten()
        speed = float(u0[0])
        v_cmd = self._speed_to_velocity(speed, release_pos, target_xy)

        # (release vel is computed later via arm_noise.pybullet_release_vel)

        # Run PyBullet simulation
        pos_traj, vel_traj = self._simulate_pybullet(
            release_pos, v_cmd, T, dt
        )
        n = len(pos_traj)

        # Build state arrays (same layout as ThrowingSystem)
        target_col   = np.tile(target_xy, (n, 1))
        clean_ball   = np.hstack([pos_traj, vel_traj])
        clean_states = np.hstack([clean_ball, target_col])

        noise_arr    = np.ones(state_dim) * noise if np.isscalar(noise) else np.array(noise)
        noisy_states = clean_states + np.random.randn(n, state_dim) * noise_arr

        inputs       = np.zeros((n, 1))
        inputs[0, 0] = speed

        return noisy_states, inputs, clean_states

    # ------------------------------------------------------------------
    # Internal: speed → velocity  (identical to ThrowingSystem)
    # ------------------------------------------------------------------

    def _speed_to_velocity(self, speed, release_pos, target_xy):
        dx      = target_xy[0] - release_pos[0]
        dy      = target_xy[1] - release_pos[1]
        azimuth = np.arctan2(dy, dx)
        alpha   = self.launch_angle
        return np.array([
            speed * np.cos(alpha) * np.cos(azimuth),
            speed * np.cos(alpha) * np.sin(azimuth),
            speed * np.sin(alpha),
        ])

    # ------------------------------------------------------------------
    # PyBullet simulation
    # ------------------------------------------------------------------

    def _simulate_pybullet(self, release_pos, v_cmd, T, dt):
        """
        Returns (pos_traj, vel_traj) as numpy arrays of shape [n, 3].
        Arm moves through the throw; ball velocity is set explicitly at release.
        After release, Eq. 35 drag is applied manually each step.
        """
        mode   = p.GUI if self._gui_mode else p.DIRECT
        client = p.connect(mode)
        p.setGravity(0, 0, -9.81, physicsClientId=client)
        p.setTimeStep(dt, physicsClientId=client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
        p.loadURDF(self._plane_urdf, physicsClientId=client)

        # Arm
        arm = ArmController(client, self._urdf_path)
        arm.reset()

        # Ball — start at EE position so the constraint is zero-offset
        ee_pos_init, _, _, _ = arm.ee_state()
        ball_col = p.createCollisionShape(
            p.GEOM_SPHERE, radius=self.radius, physicsClientId=client
        )
        ball_vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=self.radius,
            rgbaColor=[1, 1, 0, 1], physicsClientId=client,
        )
        ball_id = p.createMultiBody(
            baseMass=self.mass,
            baseCollisionShapeIndex=ball_col,
            baseVisualShapeIndex=ball_vis,
            basePosition=ee_pos_init.tolist(),
            physicsClientId=client,
        )
        p.changeDynamics(
            ball_id, -1, linearDamping=0.0, angularDamping=0.0,
            physicsClientId=client,
        )

        # Grip and plan throw
        arm.attach_ball(ball_id)
        T_arm    = max(_T_ARM, T + self.t_r)
        coeffs, _, _, _ = arm.plan_throw(v_cmd, release_pos, self.t_w, self.t_r, T_arm)

        # Timing jitter (if noise model supports it)
        release_offset = 0
        if self.arm_noise is not None:
            release_offset = self.arm_noise.sample_release_offset()
        release_step = int(self.t_r / dt) + release_offset

        pos_traj = []
        vel_traj = []
        released = False

        total_steps = int((self.t_r + T) / dt) + 100  # enough for arm + free flight

        for step in range(total_steps):
            t = step * dt

            if not released:
                q_t, qd_t = arm.get_setpoint(coeffs, t)
                arm.step(q_t, qd_t)

                if step >= release_step:
                    _, ee_vel, _, _ = arm.ee_state()
                    if self.arm_noise is not None:
                        # Noise model decides what velocity the ball gets:
                        # - VelocityBiasNoise / VelocitySlipNoise: perturb v_cmd
                        # - ReleaseTimingJitter: use actual ee_vel (arm is mid-decel)
                        actual_release_vel = self.arm_noise.pybullet_release_vel(v_cmd, ee_vel)
                        arm.release_ball(ball_id, set_vel=None)   # removes constraint
                        p.resetBaseVelocity(
                            ball_id,
                            linearVelocity=actual_release_vel.tolist(),
                            angularVelocity=[0.0, 0.0, 0.0],
                            physicsClientId=client,
                        )
                    else:
                        # Zero noise: set ball to v_cmd exactly
                        actual_release_vel = arm.release_ball(ball_id, set_vel=v_cmd)
                    released = True
                    # Record release state as first trajectory point
                    ball_pos, _ = p.getBasePositionAndOrientation(
                        ball_id, physicsClientId=client
                    )
                    pos_traj.append(np.array(ball_pos))
                    vel_traj.append(actual_release_vel.copy())
            else:
                # Free flight with Eq. 35 drag injected manually
                ball_pos, _ = p.getBasePositionAndOrientation(
                    ball_id, physicsClientId=client
                )
                ball_vel, _ = p.getBaseVelocity(ball_id, physicsClientId=client)
                pos = np.array(ball_pos)
                vel = np.array(ball_vel)

                a_total = _ball_accel(pos, vel, self.mass, self.radius, self.wind)
                a_drag  = a_total - np.array([0.0, 0.0, -9.81])
                F_drag  = self.mass * a_drag
                p.applyExternalForce(
                    ball_id, -1, F_drag.tolist(), [0, 0, 0],
                    p.WORLD_FRAME, physicsClientId=client,
                )

                pos_traj.append(pos.copy())
                vel_traj.append(vel.copy())

                # Landing: ball center reaches ground (z ≈ radius when touching)
                if pos[2] <= self.radius + 0.005 and len(pos_traj) > 2:
                    # Linear interpolation to z=0 (matches numpy sim)
                    prev_pos = pos_traj[-2]
                    if prev_pos[2] > 0:
                        frac = prev_pos[2] / (prev_pos[2] - pos[2])
                        land_pos = prev_pos + frac * (pos - prev_pos)
                        land_pos[2] = 0.0
                        land_vel = vel_traj[-2] + frac * (vel - vel_traj[-2])
                        pos_traj[-1] = land_pos
                        vel_traj[-1] = land_vel
                    break

            p.stepSimulation(physicsClientId=client)

        p.disconnect(client)

        if not pos_traj:
            # Fallback: shouldn't happen but avoid crash
            pos_traj = [release_pos.copy()]
            vel_traj = [v_cmd.copy()]

        return np.array(pos_traj), np.array(vel_traj)
