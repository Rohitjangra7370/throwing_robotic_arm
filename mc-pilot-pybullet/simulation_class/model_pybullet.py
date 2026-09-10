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
from robot_arm.robot_profiles import get_robot_profile, roll_indices
from simulation_class.model import _ball_accel
from simulation_class.release_solver import OptimizedReleaseSolver


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
        target_height=0.0,
        opt_posture=None,
        opt_launch_deg=43.0,
        opt_posture_table=None,
        base_height=0.0,
        tool_offset=None,
    ):
        # Optimized-release mode: throw from a hardware-valid frozen-base posture
        # (base = azimuth only, qd[0]=0; shoulder/elbow sweep the vertical plane),
        # scaled to the commanded speed, instead of IK+pinv.
        #   * opt_posture_table: an azimuth->posture TABLE (from find_throw_pose.py).
        #     _optimized_release looks up the nearest-azimuth posture per target,
        #     because rotating a single posture's base does NOT aim off-axis on this
        #     arm (J(base+az) != Rz*J(base)). This is the correct mode.
        #   * opt_posture: legacy single fixed posture (base-rotated at runtime);
        #     kept only for the standalone single-shot scripts.
        # The release solver is SHARED WITH HARDWARE (see release_solver.py).
        # Never inline this logic back into the sim system: run_hardware_throw.py
        # has to plan the identical release, and a second copy will drift.
        # Rigid TCP offset (link frame, e.g. (0,0,0.12) for a modeled Robotiq
        # 2F-85). Zero by default: every existing checkpoint/table/test was
        # validated with the ball welded at ee_link itself. Threaded to BOTH
        # the release solver (so the commanded release state targets the
        # TCP) and the ball's own weld point below (so what physically flies
        # is the same point the solver targeted) -- letting them disagree
        # would silently reintroduce the exact bug this parameter exists to
        # fix.
        self.tool_offset = (np.zeros(3) if tool_offset is None
                            else np.array(tool_offset, dtype=float))
        # Which joints the release LP freezes is arm-dependent (the Gen3/Panda
        # 7-DoF (0,2,4,6) is NOT a general law -- a UR is pan / three parallel
        # pitches / wrist2 / tool-roll). Read it from the profile here and pass
        # the SAME set on the hardware side (run_hardware_throw.py), or the two
        # plan different throws with no error anywhere.
        self.release_solver = OptimizedReleaseSolver(
            opt_posture=opt_posture,
            opt_launch_deg=opt_launch_deg,
            opt_posture_table=opt_posture_table,
            tool_offset=self.tool_offset,
            roll_idx=roll_indices(get_robot_profile(robot_name)),
            v_tcp_max=get_robot_profile(robot_name).v_tcp_max,
        )
        self._opt_posture = self.release_solver.posture
        self._opt_table = self.release_solver.table
        self._opt_launch = self.release_solver.launch
        self.mass = mass
        self.radius = radius
        self.launch_angle = np.deg2rad(launch_angle_deg)   # must match ThrowingSystem API
        self.wind_model = wind_model
        self.wind_aware = wind_aware
        self.arm_noise = arm_noise
        self.t_w = t_w
        self.t_r = t_r
        # Landing plane height (m). 0.0 = ground; >0 = elevated target (basket on
        # a platform). The trajectory is cut at the descending crossing of this
        # plane and the landing point interpolated onto it.
        self.target_height = float(target_height)
        # Height of the arm's base plate above the floor (m). The floor is a real
        # collision plane at world z=0, so an arm on a plate is modelled by
        # RAISING THE BASE, never by pushing target_height negative: the ball
        # would hit the plane at z=0 before crossing a negative target_height,
        # the descending-crossing test would never fire, and the rollout returns
        # with no landing detected and no error (see rollout's landing test and
        # the empty-trajectory fallback). World frame: floor at 0, base at
        # +base_height. Base frame (hardware, pose tables): base at 0, floor at
        # -base_height. One offset, converted in exactly one place.
        self.base_height = float(base_height)
        self._check_target_height(self.target_height)
        self._gui_mode = gui_mode
        self.robot_name = robot_name
        self._profile = get_robot_profile(robot_name)
        self._dynamic_release = self._profile.control_mode == "torque"
        if self._dynamic_release and self.arm_noise is not None:
            raise ValueError(
                "arm_noise is not supported with torque-mode profiles: the "
                "tracking error IS the noise being measured. Use the kinematic "
                "profile + TrackingErrorNoise for noise-aware training."
            )
        self.last_release_info = None
        self._urdf_path = pybullet_data.getDataPath() + "/" + self._profile.urdf_rel_path
        self._plane_urdf = "plane.urdf"
        
        # Import calm model as fallback
        if self.wind_model is None:
            from simulation_class.wind_models import WindModel
            self.wind_model = WindModel()

    @staticmethod
    def _check_target_height(h):
        """A landing plane below the floor cannot be detected, so refuse it.

        The floor (`plane.urdf`) is a real collision plane at world z=0. The
        landing test fires only on a DESCENDING crossing of `target_height`, so
        for h < 0 the ball rests on the floor without ever crossing, the loop
        runs out, and the rollout returns a trajectory whose last point is
        wherever the ball rolled to a stop -- a plausible number, silently
        wrong. Model an arm on a plate with `base_height` instead.
        """
        if float(h) < 0.0:
            raise ValueError(
                f"target_height={float(h):.3f} m is below the floor and cannot "
                "be detected as a landing (the ball hits plane.urdf at z=0 "
                "first and no descending crossing ever occurs). To throw from a "
                "raised base down to the floor, pass base_height=<plate height> "
                "and keep target_height=0.0."
            )

    def rollout(self, s0, policy, T, dt, noise):
        """
        Simulate one throw in PyBullet.

        Parameters match ThrowingSystem.rollout.
        """
        release_pos = np.array(s0[0:3], dtype=float)
        target_xy = np.array(s0[6:8], dtype=float)
        target_full = np.array(s0[6:], dtype=float)  # (Px,Py) or (Px,Py,Ph)
        state_dim = len(s0)
        # Height-conditioned task (9-D state): landing plane comes from the
        # per-episode target height rather than the fixed constructor value.
        if not self.wind_aware and len(target_full) >= 3:
            self._check_target_height(target_full[2])
            self.target_height = float(target_full[2])

        u0 = np.array(policy(s0, 0.0)).flatten()
        speed = float(u0[0])
        v_cmd = self._speed_to_velocity(speed, release_pos, target_xy)
        self._cur_target_xy = target_xy   # used by optimized-posture azimuth (base rotation)

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
            state_dim = 6 + len(target_full)   # 8 for (Px,Py), 9 for (Px,Py,Ph)

        # Build state arrays
        target_col   = np.tile(target_xy if self.wind_aware else target_full, (n, 1))
        if self.wind_aware:
            clean_states = np.hstack([pos_traj, vel_traj, target_col, wind_traj])
        else:
            clean_states = np.hstack([pos_traj, vel_traj, target_col])

        # Add measurement noise to ball state (not target or wind)
        noise_arr = np.ones(state_dim) * noise if np.isscalar(noise) else np.array(noise)
        noise_used = noise_arr[:state_dim]
        noisy_states = clean_states.copy()
        noisy_states += noise_used * np.random.randn(n, state_dim)
        noisy_states[:, 6:] = clean_states[:, 6:]   # targets (and Ph/wind) stay noise-free
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

    @property
    def _opt_mode(self):
        """True when running in optimized-posture release mode (table or legacy)."""
        return self.release_solver.active

    def _lookup_posture(self, azimuth):
        """Nearest-azimuth table entry -> (posture q (7,), launch elevation rad)."""
        return self.release_solver.lookup_posture(azimuth)

    def _optimized_release(self, arm, v_cmd):
        """
        Delegate to the shared solver (`simulation_class/release_solver.py`), which
        hardware planning calls too. Kept as a thin method so existing callers and
        tests keep working.
        """
        return self.release_solver.solve(
            arm, v_cmd, target_xy=getattr(self, "_cur_target_xy", None)
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
        # Sub-step the physics for the aimed real-dynamics throw: torque PD control
        # (kp=400) is unstable at the GP sampling dt (0.02) on the aggressive throw
        # trajectory, but stable at ~0.005. Step control+physics fine, RECORD at dt so the
        # GP still sees Ts-spaced samples. nsub=1 (unchanged) for every other mode.
        nsub = 1   # sub-stepping disabled: it did not resolve the opt_pose control
                   # divergence and introduced a free-flight blowup. Left as 1 (no-op)
                   # pending a proper fix of the aimed-throw training integration.
        dt_phys = dt / nsub
        p.setTimeStep(dt_phys, physicsClientId=client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
        p.loadURDF(self._plane_urdf, physicsClientId=client)

        arm = ArmController(client, self._urdf_path, robot_name=self.robot_name,
                            base_position=(0.0, 0.0, self.base_height))
        arm.reset()

        ee_pos_init, _, ee_orn_init, _ = arm.ee_state()
        # Ball spawns at the TCP (ee_link origin + tool_offset rotated into
        # world frame), not the bare flange -- so attach_ball's own local-
        # frame computation (see arm_controller.py) welds it exactly there.
        # Reduces to ee_pos_init unchanged at the zero default.
        ball_spawn_pos, _ = p.multiplyTransforms(
            ee_pos_init.tolist(), ee_orn_init.tolist(), self.tool_offset.tolist(),
            [0, 0, 0, 1], physicsClientId=client,
        )
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
            basePosition=list(ball_spawn_pos),
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

        q_ovr = qd_ovr = None
        if self._opt_mode:
            release_pos, q_ovr, qd_ovr, v_cmd = self._optimized_release(arm, v_cmd)

        coeffs, _, _, v_planned = arm.plan_throw(
            v_cmd, release_pos, self.t_w, self.t_r, t_arm,
            q_release_override=q_ovr, qd_release_override=qd_ovr,
            monotonic_windup=self._opt_mode,
            # Same throw point the ball is welded at (see attach_ball) and the
            # same one run_hardware_throw.py reports -- the sim and hardware
            # planners must not disagree about where the ball leaves from.
            tool_offset=self.tool_offset,
        )
        t_r_actual = coeffs["t_r"]  # torque mode may have stretched the throw

        release_offset = 0
        if self.arm_noise is not None:
            release_offset = self.arm_noise.sample_release_offset()
        release_step = int(t_r_actual / dt_phys) + release_offset * nsub
        pos_traj = []
        vel_traj = []
        wind_traj = []
        released = False
        total_steps = int((t_r_actual + T) / dt_phys) + 100 * nsub
        speed_norm = np.linalg.norm(v_cmd)
        release_dir = v_cmd / speed_norm if speed_norm > 1e-9 else np.zeros(3)

        for step in range(total_steps):
            t = step * dt_phys
            if not released:
                q_t, qd_t, qdd_t = arm.get_setpoint(coeffs, t, with_accel=True)
                arm.step(q_t, qd_t, qdd_t)

                if step >= release_step:
                    ee_pos, ee_vel, _, _ = arm.ee_state()
                    safe_release_pos = ee_pos + release_dir * (1.25 * self.radius)
                    use_safe_release = self._profile.use_safe_release
                    # Ball-arm collision stays disabled after release: the arm's
                    # follow-through otherwise strikes the just-released ball,
                    # injecting contact impulses into the first flight transitions
                    # (observed as |dvz| up to 8x gravity at z~=0.5 in GP data,
                    # degrading the model as such points accumulate).
                    if self._dynamic_release:
                        ee_pos_rel = ee_pos.copy()
                        actual_release_vel = arm.release_ball(
                            ball_id,
                            dynamic=True,
                            keep_collision_disabled=True,
                        )
                        self.last_release_info = {
                            "v_cmd": v_cmd.copy(),
                            "v_planned": np.array(v_planned, dtype=float),
                            "v_release": actual_release_vel.copy(),
                            "release_pos_err": float(
                                np.linalg.norm(ee_pos_rel - release_pos)
                            ),
                            "clip_scale": float(coeffs["clip_scale"]),
                            "time_scale": float(coeffs["time_scale"]),
                        }
                    elif self.arm_noise is not None:
                        actual_release_vel = self.arm_noise.pybullet_release_vel(v_cmd, ee_vel)
                        arm.release_ball(
                            ball_id,
                            set_vel=None,
                            release_pos=safe_release_pos if use_safe_release else None,
                            keep_collision_disabled=True,
                        )
                        p.resetBaseVelocity(
                            ball_id,
                            linearVelocity=actual_release_vel.tolist(),
                            angularVelocity=[0.0, 0.0, 0.0],
                            physicsClientId=client,
                        )
                    else:
                        actual_release_vel = arm.release_ball(
                            ball_id,
                            set_vel=v_cmd,
                            release_pos=safe_release_pos if use_safe_release else None,
                            keep_collision_disabled=True,
                        )

                    released = True
                    ball_pos, _ = p.getBasePositionAndOrientation(ball_id, physicsClientId=client)
                    pos_traj.append(np.array(ball_pos))
                    vel_traj.append(actual_release_vel.copy())
                    
                    # Record wind at release
                    w0 = self.wind_model(0.0)
                    wind_traj.append(w0[:2])
            else:
                # Keep commanding the arm through the planned follow-through.
                # Without this, torque-mode joints get ZERO command after
                # release -> gravity free-falls the arm (a real physical
                # collapse in the simulated world, and what a real arm would
                # do too if the controller stopped) -- the torque-validated
                # follow-through in coeffs["follow"] existed but was never
                # executed. get_setpoint clamps past T, so the arm settles
                # and holds at q_follow (neutral).
                q_t, qd_t, qdd_t = arm.get_setpoint(coeffs, t, with_accel=True)
                arm.step(q_t, qd_t, qdd_t)
                ball_pos, _ = p.getBasePositionAndOrientation(ball_id, physicsClientId=client)
                ball_vel, _ = p.getBaseVelocity(ball_id, physicsClientId=client)
                pos = np.array(ball_pos)
                vel = np.array(ball_vel)

                t_free = len(pos_traj) * dt
                w = self.wind_model(t_free)

                a_total = _ball_accel(pos, vel, self.mass, self.radius, w)
                a_drag  = a_total - np.array([0.0, 0.0, -9.81])
                f_drag  = self.mass * a_drag
                p.applyExternalForce(       # drag applied EVERY fine step (accurate)
                    ball_id,
                    -1,
                    f_drag.tolist(),
                    [0, 0, 0],
                    p.WORLD_FRAME,
                    physicsClientId=client,
                )

                # RECORD only at the GP sampling rate (every nsub fine steps)
                if (step - release_step) % nsub != 0:
                    p.stepSimulation(physicsClientId=client)
                    hook = getattr(self, "frame_hook", None)
                    if hook is not None:
                        hook(client)
                    continue

                pos_traj.append(pos.copy())
                vel_traj.append(vel.copy())
                wind_traj.append(w[:2])

                h = self.target_height
                if (pos[2] <= h + self.radius + 0.005 and vel[2] < 0.0
                        and len(pos_traj) > 2):
                    prev_pos = pos_traj[-2]
                    if prev_pos[2] > h:
                        frac = (prev_pos[2] - h) / (prev_pos[2] - pos[2])
                        land_pos = prev_pos + frac * (pos - prev_pos)
                        land_pos[2] = h
                        land_vel = vel_traj[-2] + frac * (vel - vel_traj[-2])
                        pos_traj[-1] = land_pos
                        vel_traj[-1] = land_vel

                        w_land = self.wind_model((len(pos_traj) - 2) * dt + frac * dt)
                        wind_traj[-1] = w_land[:2]
                    break

            p.stepSimulation(physicsClientId=client)

            # Optional visualization hook (set `system.frame_hook = fn` before
            # rollout); receives the client id each physics step. No effect on
            # training when unset.
            hook = getattr(self, "frame_hook", None)
            if hook is not None:
                hook(client)

        p.disconnect(client)

        if not pos_traj:
            pos_traj = [release_pos.copy()]
            vel_traj = [v_cmd.copy()]
            wind_traj = [np.zeros(2)]

        return np.array(pos_traj), np.array(vel_traj), np.array(wind_traj)
