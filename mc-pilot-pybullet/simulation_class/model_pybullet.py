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
        target_height=0.0,
        opt_posture=None,
        opt_launch_deg=43.0,
        opt_posture_table=None,
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
        self._opt_posture = None if opt_posture is None else np.array(opt_posture, dtype=float)
        self._opt_table = list(opt_posture_table) if opt_posture_table is not None else None
        self._opt_launch = np.deg2rad(opt_launch_deg)
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
        return self._opt_table is not None or self._opt_posture is not None

    def _lookup_posture(self, azimuth):
        """Nearest-azimuth table entry -> (posture q (7,), launch elevation rad)."""
        azs = np.array([e["azimuth_deg"] for e in self._opt_table])
        i = int(np.argmin(np.abs(azs - np.degrees(azimuth))))
        e = self._opt_table[i]
        return np.array(e["q"], dtype=float), np.deg2rad(float(e["elev_deg"]))

    def _optimized_release(self, arm, v_cmd):
        """
        AIMED frozen-base release: pick a hardware-valid posture for the target azimuth
        (nearest table entry; base = azimuth, held still), then solve the DIRECTION-
        CONSTRAINED joint velocities (qd[0]=0) so the EE velocity points EXACTLY along the
        launch direction d, scaled to the commanded speed.
        Returns (release_pos, q_release, qd_release, v_dir) for plan_throw overrides.
        """
        speed = float(np.linalg.norm(v_cmd))
        tgt = getattr(self, "_cur_target_xy", None)
        azimuth = float(np.arctan2(tgt[1], tgt[0])) if tgt is not None else 0.0
        rotation_built = (self._opt_table is not None
                          and bool(self._opt_table[0].get("rotation_built", False)))
        if rotation_built and tgt is not None:
            # TURRET-AIMING CORRECTION. The ball flies from the RELEASE POINT,
            # not the origin -- and the release point sits at a fixed polar
            # offset (radius r_off, azimuth alpha_off relative to the aim
            # direction) that rotates rigidly with the posture. Aiming the
            # VELOCITY at the target's origin-azimuth misses whenever
            # r_off*sin(alpha_off) is non-negligible next to |T| (measured:
            # the fast posture has alpha_off = -51.7 deg, r_off = 0.54 m ->
            # ~32 deg aim error at 0.8 m targets = the whole 37 cm accuracy
            # collapse; the old posture's -4.4 deg offset was silently
            # absorbed by training). Solve the offset-turret equation for the
            # aim heading phi such that the horizontal ray from the ACTUAL
            # release point along phi passes through the target:
            #   |T| * sin(phi - beta) = -r_off * sin(alpha_off)
            #   => phi = beta + arcsin(-r_off*sin(alpha_off) / |T|)
            if not hasattr(self, "_opt_polar"):
                # (r_off, alpha_off) are rotation-invariant posture constants:
                # measure once from any entry via FK at its stored q.
                e0 = self._opt_table[0]
                q_probe = np.array(e0["q"], dtype=float)
                qf = arm._ik_q_neutral.copy()
                for li, dof in enumerate(arm._dof_ids):
                    qf[dof] = q_probe[li]
                for j in range(arm._n_dofs):
                    p.resetJointState(arm._arm_id, j, qf[j], physicsClientId=arm._cid)
                rp = np.array(p.getLinkState(
                    arm._arm_id, arm._ee_link, computeForwardKinematics=True,
                    physicsClientId=arm._cid)[4])
                for li, joint_id in enumerate(arm._joint_ids):
                    p.resetJointState(arm._arm_id, joint_id, arm._q_neutral[li],
                                      0.0, physicsClientId=arm._cid)
                aim0 = np.deg2rad(float(e0["azimuth_deg"]))
                self._opt_polar = (float(np.hypot(rp[0], rp[1])),
                                   float(np.arctan2(rp[1], rp[0]) - aim0))
            r_off, alpha_off = self._opt_polar
            t_norm = float(np.hypot(tgt[0], tgt[1]))
            s_arg = -r_off * np.sin(alpha_off) / max(t_norm, 1e-9)
            azimuth = azimuth + float(np.arcsin(np.clip(s_arg, -1.0, 1.0)))
        if self._opt_table is not None:
            # Nearest-azimuth posture; rotate the base so the posture aims at
            # the (turret-corrected) heading.
            q_release, elev = self._lookup_posture(azimuth)
            if rotation_built:
                # Entries store q[0] = q0[0] - az_entry (VERIFIED sign: this
                # URDF's base joint measures opposite the world-z rotation
                # sense, |v_rot - Rz v0| ~ 1e-6). Recover q0[0] and rotate to
                # the exact corrected heading.
                near = min(self._opt_table,
                           key=lambda e: abs(np.deg2rad(e["azimuth_deg"]) - azimuth))
                q0_base = float(near["q"][0]) + np.deg2rad(float(near["azimuth_deg"]))
                q_release[0] = q0_base - azimuth
                if "v_dir" in near:
                    # EXACT-qd path (overhead/sagittal tables): with the roll
                    # joints pinned the pitch axes are parallel, so the LP's
                    # exact-direction constraint below is degenerate-
                    # infeasible (achievable velocities span one PLANE, and a
                    # generic d is off-plane by the URDF's y-offsets). No LP:
                    # rotate the stored exact velocity direction along with
                    # the posture and scale the stored qd linearly -- J qd
                    # scales exactly, direction preserved (rotation
                    # invariance verified 1e-6).
                    q_ref0 = (arm._q_neutral[0] if hasattr(arm, "_q_neutral")
                              else 0.0)
                    q_release[0] = q_release[0] + 2.0 * np.pi * np.round(
                        (q_ref0 - q_release[0]) / (2.0 * np.pi))
                    dpsi = azimuth - np.deg2rad(float(near["azimuth_deg"]))
                    ca, sa = np.cos(dpsi), np.sin(dpsi)
                    Rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
                    v_dir = Rz @ np.asarray(near["v_dir"], dtype=float)
                    s_max = float(near["speed"])
                    scale = min(1.0, speed / s_max) if s_max > 1e-9 else 0.0
                    qd_release = np.asarray(near["qd"], dtype=float) * scale
                    q_full = arm._ik_q_neutral.copy()
                    for li, dof in enumerate(arm._dof_ids):
                        q_full[dof] = q_release[li]
                    for j in range(arm._n_dofs):
                        p.resetJointState(arm._arm_id, j, q_full[j],
                                          physicsClientId=arm._cid)
                    release_pos = np.array(p.getLinkState(
                        arm._arm_id, arm._ee_link, computeForwardKinematics=True,
                        physicsClientId=arm._cid)[4])
                    for li, joint_id in enumerate(arm._joint_ids):
                        p.resetJointState(arm._arm_id, joint_id,
                                          arm._q_neutral[li], 0.0,
                                          physicsClientId=arm._cid)
                    return release_pos, q_release, qd_release, v_dir * min(speed, s_max)
            else:
                q_release[0] = azimuth
        else:
            # Legacy single-posture mode (base-rotated); NOTE this does not aim
            # off-axis on this arm -- use the table for training. Kept for scripts.
            q_release = self._opt_posture.copy()
            q_release[0] = self._opt_posture[0] + azimuth
            elev = self._opt_launch
        # Wrap ONLY the base joint (idx0) to the value nearest neutral: it's the sole
        # joint with continuous/infinite rotation range (Kinova spec), so azimuth can
        # legitimately need +-2pi correction. The other 6 joints have hard mechanical
        # limits (e.g. elbow +-147deg) and are ALREADY within range by construction
        # (the pose search respects joint limits) -- wrapping them blindly is wrong
        # and dangerous: verified it can push a valid -100deg elbow target to +260deg,
        # far outside the physical limit, silently corrupting the release pose (the
        # arm can't reach it, release velocity collapses to ~0 regardless of command).
        q_ref = arm._q_neutral if hasattr(arm, "_q_neutral") else np.zeros_like(q_release)
        q_release[0] = q_release[0] + 2.0 * np.pi * np.round((q_ref[0] - q_release[0]) / (2.0 * np.pi))
        d = np.array([                                          # launch direction
            np.cos(elev) * np.cos(azimuth),
            np.cos(elev) * np.sin(azimuth),
            np.sin(elev),
        ])
        # Jacobian at q_release
        q_full = arm._ik_q_neutral.copy()
        for li, dof in enumerate(arm._dof_ids):
            q_full[dof] = q_release[li]
        jl, _ = p.calculateJacobian(
            arm._arm_id, arm._ee_link, [0, 0, 0], q_full.tolist(),
            [0.0] * arm._n_dofs, [0.0] * arm._n_dofs, physicsClientId=arm._cid,
        )
        J = np.array(jl)[:, arm._dof_ids]
        # Direction-constrained aimed q̇: maximize s s.t. J q̇ = s·d, |q̇ᵢ| ≤ qd_max.
        # Forces the EE velocity to lie EXACTLY along d (aimable), unlike the sign trick.
        # Only PITCH joints (shoulder=1, elbow=3, wrist=5) carry velocity -- their axis
        # is perpendicular to the swing plane. ROLL/TWIST joints (base=0, shoulder-
        # roll=2, wrist-roll1=4, wrist-roll2=6) rotate about an axis roughly ALONG the
        # connecting link; letting the LP spin them produces a corkscrew motion, not a
        # throw (verified visually). They're frozen at qd=0 -- static setup only, same
        # as the base's azimuth role -- matching find_throw_pose.py's _ROLL_IDX.
        from scipy.optimize import linprog
        nq = len(arm._qd_max)
        c = np.zeros(nq + 1); c[-1] = -1.0
        A_eq = np.hstack([J, -d.reshape(3, 1)])
        bnds = [(-arm._qd_max[i], arm._qd_max[i]) for i in range(nq)] + [(0, None)]
        for i in (0, 2, 4, 6):
            bnds[i] = (0.0, 0.0)
        lp = linprog(c, A_eq=A_eq, b_eq=np.zeros(3), bounds=bnds, method="highs")
        if lp.success:
            v_max = float(lp.x[-1]); qd_opt = lp.x[:nq]
        else:
            v_max, qd_opt = 0.0, np.zeros(nq)
        scale = min(1.0, speed / v_max) if v_max > 1e-9 else 0.0
        qd_release = qd_opt * scale
        # release position = FK at q_release
        for j in range(arm._n_dofs):
            p.resetJointState(arm._arm_id, j, q_full[j], physicsClientId=arm._cid)
        release_pos = np.array(
            p.getLinkState(arm._arm_id, arm._ee_link, computeForwardKinematics=True,
                           physicsClientId=arm._cid)[4]
        )
        # Restore the arm to neutral: the FK teleport above is a QUERY, not the
        # start of the motion. The throw trajectory begins at q_neutral (windup),
        # so leaving the arm at the contorted release pose gives the torque-PD
        # controller a ~3 rad startup error at step 0 -> torque saturates ->
        # joints run away to PyBullet's 100 rad/s clamp (verified). The standalone
        # aimed-throw scripts reset to neutral here; _simulate_pybullet did not.
        for local_i, joint_id in enumerate(arm._joint_ids):
            p.resetJointState(arm._arm_id, joint_id, arm._q_neutral[local_i], 0.0,
                              physicsClientId=arm._cid)
        return release_pos, q_release, qd_release, d * speed

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

        q_ovr = qd_ovr = None
        if self._opt_mode:
            release_pos, q_ovr, qd_ovr, v_cmd = self._optimized_release(arm, v_cmd)

        coeffs, _, _, v_planned = arm.plan_throw(
            v_cmd, release_pos, self.t_w, self.t_r, t_arm,
            q_release_override=q_ovr, qd_release_override=qd_ovr,
            monotonic_windup=self._opt_mode,
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
