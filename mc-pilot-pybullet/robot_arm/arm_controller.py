"""
ArmController - profile-driven throw planner for mc-pilot-pybullet.

Handles:
  - Loading a supported robot arm URDF into an existing PyBullet client
  - Computing IK + Jacobian pseudoinverse to find joint config and joint
    velocities that produce a desired EE velocity at a given release point
  - Generating a 3-phase piecewise-cubic joint trajectory:
      neutral -> windup  [0, t_w]      (rest-to-rest)
      windup  -> release [t_w, t_r]    (reaches qd_release at t_r)
      release -> rest    [t_r, T]      (follow-through, cosmetic)
  - Commanding joints via PyBullet POSITION_CONTROL each sim step
  - Gripping / releasing the ball via a JOINT_FIXED constraint
"""

import numpy as np
import pybullet as p

from robot_arm.robot_profiles import get_robot_profile


class ArmController:
    def __init__(
        self,
        client_id,
        urdf_path,
        base_position=(0, 0, 0),
        q_neutral=None,
        robot_name="kuka_iiwa",
    ):
        """
        Parameters
        ----------
        client_id : int
            PyBullet physics client returned by p.connect(...).
        urdf_path : str
            Absolute path to the robot URDF.
        base_position : (3,)
            Where to mount the arm base in world frame.
        q_neutral : (n,) or None
            Neutral actuated-joint configuration; if omitted, use the robot profile.
        robot_name : str
            Supported profile name.
        """
        self._profile = get_robot_profile(robot_name)
        self._cid = client_id
        self._arm_id = p.loadURDF(
            urdf_path,
            basePosition=base_position,
            useFixedBase=True,
            physicsClientId=client_id,
        )
        self._n_joints = p.getNumJoints(self._arm_id, physicsClientId=client_id)
        self._joint_ids = list(self._profile.joint_ids)
        self._ee_link = self._profile.ee_link
        self._joint_id_to_dof_id = {}
        dof_counter = 0
        for joint_id in range(self._n_joints):
            ji = p.getJointInfo(self._arm_id, joint_id, physicsClientId=client_id)
            if ji[2] != p.JOINT_FIXED:
                self._joint_id_to_dof_id[joint_id] = dof_counter
                dof_counter += 1
        self._dof_ids = [self._joint_id_to_dof_id[j] for j in self._joint_ids]
        self._n_dofs = dof_counter

        self._q_lo = np.zeros(len(self._joint_ids))
        self._q_hi = np.zeros(len(self._joint_ids))
        self._max_forces = np.zeros(len(self._joint_ids))
        for local_i, joint_id in enumerate(self._joint_ids):
            ji = p.getJointInfo(self._arm_id, joint_id, physicsClientId=client_id)
            self._q_lo[local_i] = ji[8]
            self._q_hi[local_i] = ji[9]
            self._max_forces[local_i] = max(float(ji[10]), 1.0)

        self._ik_q_lo = np.zeros(self._n_dofs)
        self._ik_q_hi = np.zeros(self._n_dofs)
        self._ik_q_neutral = np.zeros(self._n_dofs)
        for local_i, joint_id in enumerate(self._joint_ids):
            dof_id = self._dof_ids[local_i]
            self._ik_q_lo[dof_id] = self._q_lo[local_i]
            self._ik_q_hi[dof_id] = self._q_hi[local_i]

        self._qd_max = np.array(self._profile.qd_max, dtype=float)
        self._position_gain = float(self._profile.position_gain)
        self._velocity_gain = float(self._profile.velocity_gain)
        self._force_scale = float(self._profile.force_scale)
        self._control_mode = str(self._profile.control_mode)
        self._tau_max = None
        self._kp = None
        self._kd = None
        if self._control_mode == "torque":
            if (
                self._profile.tau_max is None
                or self._profile.kp is None
                or self._profile.kd is None
            ):
                raise ValueError(
                    f"Profile '{self._profile.name}' uses torque mode but lacks "
                    "tau_max/kp/kd."
                )
            if self._n_dofs != len(self._joint_ids):
                raise ValueError(
                    "Torque mode requires every DOF to be actuated "
                    f"({self._n_dofs} DOFs vs {len(self._joint_ids)} actuated)."
                )
            self._tau_max = np.array(self._profile.tau_max, dtype=float)
            self._kp = np.array(self._profile.kp, dtype=float)
            self._kd = np.array(self._profile.kd, dtype=float)
            # Disable PyBullet's default velocity motors so TORQUE_CONTROL acts.
            p.setJointMotorControlArray(
                self._arm_id,
                self._joint_ids,
                controlMode=p.VELOCITY_CONTROL,
                forces=[0.0] * len(self._joint_ids),
                physicsClientId=client_id,
            )
        if q_neutral is None:
            self._q_neutral = np.array(self._profile.q_neutral, dtype=float)
        else:
            self._q_neutral = np.array(q_neutral, dtype=float)
        for local_i, dof_id in enumerate(self._dof_ids):
            self._ik_q_neutral[dof_id] = self._q_neutral[local_i]

        self._grip_id = None
        self._attached_ball_id = None
        self.reset()

    def reset(self):
        """Reset actuated joints to q_neutral with zero velocity."""
        for local_i, joint_id in enumerate(self._joint_ids):
            p.resetJointState(
                self._arm_id,
                joint_id,
                targetValue=self._q_neutral[local_i],
                targetVelocity=0.0,
                physicsClientId=self._cid,
            )
        self._grip_id = None
        self._attached_ball_id = None

    def attach_ball(self, ball_id):
        """Weld ball to the end-effector via a fixed constraint."""
        ee_pos = self.ee_state()[0]
        ball_pos, _ = p.getBasePositionAndOrientation(ball_id, physicsClientId=self._cid)
        offset = np.array(ball_pos) - np.array(ee_pos)
        self._set_ball_collision_with_arm(ball_id, enable=False)
        self._grip_id = p.createConstraint(
            parentBodyUniqueId=self._arm_id,
            parentLinkIndex=self._ee_link,
            childBodyUniqueId=ball_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=offset.tolist(),
            childFramePosition=[0, 0, 0],
            physicsClientId=self._cid,
        )
        self._attached_ball_id = ball_id
        return self._grip_id

    def plan_throw(self, v_cmd, release_pos, t_w=0.3, t_r=0.6, T=1.0):
        """
        Plan a 3-phase piecewise-cubic throw trajectory.

        Returns
        -------
        coeffs : dict
            Cubic coefficients for the windup, throw, and follow-through phases.
        q_release : (n,)
            Joint configuration at release.
        qd_release : (n,)
            Joint velocity at release after clipping against qd_max.
        v_achieved : (3,)
            Actual EE velocity achievable after clipping.
        """
        v_cmd = np.array(v_cmd, dtype=float)
        release_pos = np.array(release_pos, dtype=float)

        q_release = np.array(
            p.calculateInverseKinematics(
                self._arm_id,
                self._ee_link,
                targetPosition=release_pos.tolist(),
                restPoses=self._ik_q_neutral.tolist(),
                lowerLimits=self._ik_q_lo.tolist(),
                upperLimits=self._ik_q_hi.tolist(),
                jointRanges=(self._ik_q_hi - self._ik_q_lo).tolist(),
                maxNumIterations=200,
                residualThreshold=1e-4,
                physicsClientId=self._cid,
            )
        )
        q_release = q_release[self._dof_ids]

        q_release_full = self._ik_q_neutral.copy()
        for local_i, dof_id in enumerate(self._dof_ids):
            q_release_full[dof_id] = q_release[local_i]

        j_lin_raw, _ = p.calculateJacobian(
            self._arm_id,
            self._ee_link,
            localPosition=[0, 0, 0],
            objPositions=q_release_full.tolist(),
            objVelocities=[0.0] * self._n_dofs,
            objAccelerations=[0.0] * self._n_dofs,
            physicsClientId=self._cid,
        )
        j_lin = np.array(j_lin_raw)[:, self._dof_ids]
        qd_release = np.linalg.pinv(j_lin) @ v_cmd

        ratio = np.abs(qd_release) / self._qd_max
        clip_scale = 1.0
        if ratio.max() > 1.0:
            clip_scale = float(ratio.max())
            qd_release = qd_release / clip_scale
        v_achieved = j_lin @ qd_release

        q_windup = self._q_neutral + (q_release - self._q_neutral) * (-0.5)
        q_windup = np.clip(q_windup, self._q_lo, self._q_hi)
        q_follow = self._q_neutral.copy()

        dt_throw = t_r - t_w
        follow_dur = T - t_r
        time_scale = 1.0
        throw_coeffs = _cubic_to_velocity(q_windup, q_release, qd_release, dt_throw)

        if self._control_mode == "torque":
            for _ in range(6):
                ratio, worst_tau = self._throw_peak_torque_ratio(throw_coeffs, dt_throw)
                if ratio <= 1.0:
                    break
                dt_throw *= 1.2
                time_scale *= 1.2
                throw_coeffs = _cubic_to_velocity(
                    q_windup, q_release, qd_release, dt_throw
                )
            else:
                ratio, worst_tau = self._throw_peak_torque_ratio(throw_coeffs, dt_throw)
                if ratio > 1.0:
                    report = ", ".join(
                        f"j{j}: {abs(t):.1f}/{m:.1f} Nm"
                        for j, (t, m) in enumerate(zip(worst_tau, self._tau_max))
                    )
                    raise RuntimeError(
                        f"Throw infeasible after 6 time-scaling iterations "
                        f"(peak ratio {ratio:.2f}): {report}"
                    )

        t_r_actual = t_w + dt_throw
        T_actual = t_r_actual + follow_dur

        coeffs = {
            "windup": _cubic_rest_to_rest(self._q_neutral, q_windup, t_w),
            "throw": throw_coeffs,
            "follow": _cubic_from_velocity(q_release, qd_release, q_follow, follow_dur),
            "t_w": t_w,
            "t_r": t_r_actual,
            "T": T_actual,
            "clip_scale": clip_scale,
            "time_scale": time_scale,
        }
        return coeffs, q_release, qd_release, v_achieved

    def _throw_peak_torque_ratio(self, throw_coeffs, dt_throw, n_samples=50):
        """Max over the throw phase of max_j |tau_j| / tau_max_j."""
        worst = 0.0
        worst_tau = None
        for tau_t in np.linspace(0.0, dt_throw, n_samples):
            q, qd, qdd = _eval_cubic(throw_coeffs, tau_t, with_accel=True)
            torque = np.array(
                p.calculateInverseDynamics(
                    self._arm_id,
                    q.tolist(),
                    qd.tolist(),
                    qdd.tolist(),
                    physicsClientId=self._cid,
                )
            )
            ratio = float(np.max(np.abs(torque) / self._tau_max))
            if ratio > worst:
                worst = ratio
                worst_tau = torque
        return worst, worst_tau

    def get_setpoint(self, coeffs, t, with_accel=False):
        """Evaluate the piecewise cubic at time t."""
        t_w = coeffs["t_w"]
        t_r = coeffs["t_r"]
        T = coeffs["T"]
        if t <= t_w:
            return _eval_cubic(coeffs["windup"], t, with_accel)
        if t <= t_r:
            return _eval_cubic(coeffs["throw"], t - t_w, with_accel)
        return _eval_cubic(coeffs["follow"], min(t - t_r, T - t_r), with_accel)

    def step(self, q_target, qd_target, qdd_target=None):
        """Command actuated joints for one sim step (mode set by profile)."""
        if self._control_mode == "kinematic":
            for local_i, joint_id in enumerate(self._joint_ids):
                p.resetJointState(
                    self._arm_id,
                    joint_id,
                    targetValue=float(q_target[local_i]),
                    targetVelocity=float(qd_target[local_i]),
                    physicsClientId=self._cid,
                )
            return

        if self._control_mode == "torque":
            if qdd_target is None:
                qdd_target = np.zeros(len(self._joint_ids))
            states = p.getJointStates(
                self._arm_id, self._joint_ids, physicsClientId=self._cid
            )
            q_meas = np.array([s[0] for s in states])
            qd_meas = np.array([s[1] for s in states])
            e = np.asarray(q_target, dtype=float) - q_meas
            ed = np.asarray(qd_target, dtype=float) - qd_meas
            qdd_cmd = np.asarray(qdd_target, dtype=float) + self._kp * e + self._kd * ed
            # Inverse dynamics for the ARM ALONE: the gripped ball and its
            # constraint forces are deliberately unmodeled (source of the
            # tracking error this study measures).
            tau = np.array(
                p.calculateInverseDynamics(
                    self._arm_id,
                    q_meas.tolist(),
                    qd_meas.tolist(),
                    qdd_cmd.tolist(),
                    physicsClientId=self._cid,
                )
            )
            if not np.all(np.isfinite(tau)):
                raise RuntimeError(
                    f"Non-finite torque command: tau={tau}, q={q_meas}, qd={qd_meas}"
                )
            tau = np.clip(tau, -self._tau_max, self._tau_max)
            p.setJointMotorControlArray(
                self._arm_id,
                self._joint_ids,
                controlMode=p.TORQUE_CONTROL,
                forces=tau.tolist(),
                physicsClientId=self._cid,
            )
            return

        p.setJointMotorControlArray(
            self._arm_id,
            self._joint_ids,
            controlMode=p.POSITION_CONTROL,
            targetPositions=q_target.tolist(),
            targetVelocities=qd_target.tolist(),
            positionGains=[self._position_gain] * len(self._joint_ids),
            velocityGains=[self._velocity_gain] * len(self._joint_ids),
            forces=(self._force_scale * self._max_forces).tolist(),
            physicsClientId=self._cid,
        )

    def release_ball(
        self,
        ball_id,
        set_vel=None,
        dv_noise=None,
        release_pos=None,
        keep_collision_disabled=False,
    ):
        """
        Remove the grip constraint and optionally override the ball velocity.
        """
        if self._grip_id is not None:
            p.removeConstraint(self._grip_id, physicsClientId=self._cid)
            self._grip_id = None
        if self._attached_ball_id is not None:
            if not keep_collision_disabled:
                self._set_ball_collision_with_arm(self._attached_ball_id, enable=True)
            self._attached_ball_id = None

        if release_pos is not None:
            p.resetBasePositionAndOrientation(
                ball_id,
                posObj=np.array(release_pos, dtype=float).tolist(),
                ornObj=[0.0, 0.0, 0.0, 1.0],
                physicsClientId=self._cid,
            )

        if set_vel is not None:
            release_vel = np.array(set_vel, dtype=float)
        else:
            _, ee_vel, _, _ = self.ee_state()
            release_vel = np.array(ee_vel)

        if dv_noise is not None:
            release_vel = release_vel + np.array(dv_noise)

        p.resetBaseVelocity(
            ball_id,
            linearVelocity=release_vel.tolist(),
            angularVelocity=[0.0, 0.0, 0.0],
            physicsClientId=self._cid,
        )
        return release_vel

    def ee_state(self):
        """Query end-effector state in world frame."""
        ls = p.getLinkState(
            self._arm_id,
            self._ee_link,
            computeLinkVelocity=1,
            computeForwardKinematics=1,
            physicsClientId=self._cid,
        )
        pos = np.array(ls[0])
        orient = np.array(ls[1])
        lin_vel = np.array(ls[6])
        ang_vel = np.array(ls[7])
        return pos, lin_vel, orient, ang_vel

    @property
    def arm_id(self):
        return self._arm_id

    @property
    def joint_ids(self):
        return tuple(self._joint_ids)

    @property
    def robot_name(self):
        return self._profile.name

    def _set_ball_collision_with_arm(self, ball_id, enable):
        enable_flag = 1 if enable else 0
        p.setCollisionFilterPair(
            self._arm_id,
            ball_id,
            -1,
            -1,
            enable_flag,
            physicsClientId=self._cid,
        )
        for joint_id in range(self._n_joints):
            p.setCollisionFilterPair(
                self._arm_id,
                ball_id,
                joint_id,
                -1,
                enable_flag,
                physicsClientId=self._cid,
            )


def _cubic_rest_to_rest(q_start, q_end, dt):
    dq = q_end - q_start
    a0 = q_start
    a1 = np.zeros_like(q_start)
    a2 = 3.0 * dq / dt**2
    a3 = -2.0 * dq / dt**3
    return np.stack([a0, a1, a2, a3], axis=1)


def _cubic_to_velocity(q_start, q_end, qd_end, dt):
    dq = q_end - q_start
    a0 = q_start
    a1 = np.zeros_like(q_start)
    a3 = (qd_end * dt - 2.0 * dq) / dt**3
    a2 = (3.0 * dq - qd_end * dt) / dt**2
    return np.stack([a0, a1, a2, a3], axis=1)


def _cubic_from_velocity(q_start, qd_start, q_end, dt):
    a0 = q_start
    a1 = qd_start
    a3 = (2.0 * (q_start - q_end) + qd_start * dt) / dt**3
    a2 = (-qd_start - 3.0 * a3 * dt**2) / (2.0 * dt)
    return np.stack([a0, a1, a2, a3], axis=1)


def _eval_cubic(coeffs, tau, with_accel=False):
    a0, a1, a2, a3 = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2], coeffs[:, 3]
    q = a0 + a1 * tau + a2 * tau**2 + a3 * tau**3
    qd = a1 + 2.0 * a2 * tau + 3.0 * a3 * tau**2
    if not with_accel:
        return q, qd
    qdd = 2.0 * a2 + 6.0 * a3 * tau
    return q, qd, qdd
