"""
PyBulletThrowingSystem - drop-in replacement for ThrowingSystem.

Same constructor signature, same rollout() return format, same 8-D state
layout. A supported robot arm physically executes the throw in a fresh PyBullet
DIRECT world each rollout. Ball free-flight uses the paper's Eq. 35 drag (via
_ball_accel from model.py), not PyBullet's built-in damping, so results remain
comparable to the numpy simulator.
"""

import numpy as np
import pybullet as p
import pybullet_data

from robot_arm.arm_controller import ArmController
from robot_arm.robot_profiles import get_robot_profile
from simulation_class.model import _ball_accel


_T_W = 0.30
_T_R = 0.60
_T_ARM = 1.20


class PyBulletThrowingSystem:
    def __init__(
        self,
        mass=0.0577,
        radius=0.0327,
        launch_angle_deg=35.0,
        wind_model=None,
        wind_aware=False,
        arm_noise=None,
        t_w=_T_W,
        t_r=_T_R,
        gui_mode=False,
        robot_name="kuka_iiwa",
    ):
        self.mass = mass
        self.radius = radius
        self.launch_angle = np.deg2rad(launch_angle_deg)   # must match ThrowingSystem API
        self.wind_model = wind_model
        self.wind_aware = wind_aware
        self.arm_noise = arm_noise
        self.t_w = t_w
        self.t_r = t_r
        self._gui_mode = gui_mode
        self.robot_name = robot_name
        self._profile = get_robot_profile(robot_name)
        self._urdf_path = pybullet_data.getDataPath() + "/" + self._profile.urdf_rel_path
        self._plane_urdf = "plane.urdf"
        
        # Import calm model as fallback
        if self.wind_model is None:
            from simulation_class.wind_models import WindModel
            self.wind_model = WindModel()

    def rollout(self, s0, policy, T, dt, noise):
        """
        Simulate one throw in PyBullet.

        Parameters match ThrowingSystem.rollout.
        """
        release_pos = np.array(s0[0:3], dtype=float)
        target_xy = np.array(s0[6:8], dtype=float)
        state_dim = len(s0)

        u0 = np.array(policy(s0, 0.0)).flatten()
        speed = float(u0[0])
        v_cmd = self._speed_to_velocity(speed, release_pos, target_xy)

        # (release vel is computed later via arm_noise.pybullet_release_vel)

        # Reset wind model for this episode
        self.wind_model.reset()

        # Run PyBullet simulation
        pos_traj, vel_traj, wind_traj = self._simulate_pybullet(
            release_pos, v_cmd, T, dt
        )
        n = len(pos_traj)
        
        # Determine state dimensionality
        if self.wind_aware:
            state_dim = 10
        else:
            state_dim = 8

        # Build state arrays
        target_col   = np.tile(target_xy, (n, 1))
        if self.wind_aware:
            clean_states = np.hstack([pos_traj, vel_traj, target_col, wind_traj])
        else:
            clean_states = np.hstack([pos_traj, vel_traj, target_col])

        # Add measurement noise to ball state (not target or wind)
        noise_arr = np.ones(state_dim) * noise if np.isscalar(noise) else np.array(noise)
        noise_used = noise_arr[:state_dim]
        noisy_states = clean_states.copy()
        noisy_states += noise_used * np.random.randn(n, state_dim)
        noisy_states[:, 6:8] = clean_states[:, 6:8]
        if self.wind_aware:
            noisy_states[:, 8:10] = clean_states[:, 8:10]

        inputs = np.zeros((n, 1))
        inputs[0, 0] = speed
        return noisy_states, inputs, clean_states

    def _speed_to_velocity(self, speed, release_pos, target_xy):
        dx = target_xy[0] - release_pos[0]
        dy = target_xy[1] - release_pos[1]
        azimuth = np.arctan2(dy, dx)
        alpha = self.launch_angle
        return np.array(
            [
                speed * np.cos(alpha) * np.cos(azimuth),
                speed * np.cos(alpha) * np.sin(azimuth),
                speed * np.sin(alpha),
            ]
        )

    def _simulate_pybullet(self, release_pos, v_cmd, T, dt):
        """
        Returns (pos_traj, vel_traj, wind_traj).
        Arm moves through the throw; ball velocity is set explicitly at release.
        After release, Eq. 35 drag is applied manually each step.
        """
        mode = p.GUI if self._gui_mode else p.DIRECT
        client = p.connect(mode)
        p.setGravity(0, 0, -9.81, physicsClientId=client)
        p.setTimeStep(dt, physicsClientId=client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
        p.loadURDF(self._plane_urdf, physicsClientId=client)

        arm = ArmController(client, self._urdf_path, robot_name=self.robot_name)
        arm.reset()

        ee_pos_init, _, _, _ = arm.ee_state()
        ball_col = p.createCollisionShape(
            p.GEOM_SPHERE, radius=self.radius, physicsClientId=client
        )
        ball_vis = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self.radius,
            rgbaColor=[1, 1, 0, 1],
            physicsClientId=client,
        )
        ball_id = p.createMultiBody(
            baseMass=self.mass,
            baseCollisionShapeIndex=ball_col,
            baseVisualShapeIndex=ball_vis,
            basePosition=ee_pos_init.tolist(),
            physicsClientId=client,
        )
        p.changeDynamics(
            ball_id,
            -1,
            linearDamping=0.0,
            angularDamping=0.0,
            physicsClientId=client,
        )

        arm.attach_ball(ball_id)
        profile_t_arm = self._profile.timing[2]
        t_arm = max(_T_ARM, profile_t_arm, T + self.t_r)
        coeffs, _, _, _ = arm.plan_throw(v_cmd, release_pos, self.t_w, self.t_r, t_arm)

        release_offset = 0
        if self.arm_noise is not None:
            release_offset = self.arm_noise.sample_release_offset()
        release_step = int(self.t_r / dt) + release_offset
        pos_traj = []
        vel_traj = []
        wind_traj = []
        released = False
        total_steps = int((self.t_r + T) / dt) + 100

        for step in range(total_steps):
            t = step * dt
            if not released:
                q_t, qd_t = arm.get_setpoint(coeffs, t)
                arm.step(q_t, qd_t)

                if step >= release_step:
                    _, ee_vel, _, _ = arm.ee_state()
                    if self.arm_noise is not None:
                        actual_release_vel = self.arm_noise.pybullet_release_vel(v_cmd, ee_vel)
                        arm.release_ball(ball_id, set_vel=None)
                        p.resetBaseVelocity(
                            ball_id,
                            linearVelocity=actual_release_vel.tolist(),
                            angularVelocity=[0.0, 0.0, 0.0],
                            physicsClientId=client,
                        )
                    else:
                        actual_release_vel = arm.release_ball(ball_id, set_vel=v_cmd)

                    released = True
                    ball_pos, _ = p.getBasePositionAndOrientation(ball_id, physicsClientId=client)
                    pos_traj.append(np.array(ball_pos))
                    vel_traj.append(actual_release_vel.copy())
                    
                    # Record wind at release
                    w0 = self.wind_model(0.0)
                    wind_traj.append(w0[:2])
            else:
                ball_pos, _ = p.getBasePositionAndOrientation(ball_id, physicsClientId=client)
                ball_vel, _ = p.getBaseVelocity(ball_id, physicsClientId=client)
                pos = np.array(ball_pos)
                vel = np.array(ball_vel)

                t_free = len(pos_traj) * dt
                w = self.wind_model(t_free)

                a_total = _ball_accel(pos, vel, self.mass, self.radius, w)
                a_drag  = a_total - np.array([0.0, 0.0, -9.81])
                f_drag  = self.mass * a_drag
                p.applyExternalForce(
                    ball_id,
                    -1,
                    f_drag.tolist(),
                    [0, 0, 0],
                    p.WORLD_FRAME,
                    physicsClientId=client,
                )

                pos_traj.append(pos.copy())
                vel_traj.append(vel.copy())
                wind_traj.append(w[:2])

                if pos[2] <= self.radius + 0.005 and len(pos_traj) > 2:
                    prev_pos = pos_traj[-2]
                    if prev_pos[2] > 0:
                        frac = prev_pos[2] / (prev_pos[2] - pos[2])
                        land_pos = prev_pos + frac * (pos - prev_pos)
                        land_pos[2] = 0.0
                        land_vel = vel_traj[-2] + frac * (vel - vel_traj[-2])
                        pos_traj[-1] = land_pos
                        vel_traj[-1] = land_vel
                        
                        w_land = self.wind_model((len(pos_traj) - 2) * dt + frac * dt)
                        wind_traj[-1] = w_land[:2]
                    break

            p.stepSimulation(physicsClientId=client)

        p.disconnect(client)

        if not pos_traj:
            pos_traj = [release_pos.copy()]
            vel_traj = [v_cmd.copy()]
            wind_traj = [np.zeros(2)]

        return np.array(pos_traj), np.array(vel_traj), np.array(wind_traj)
