"""
ArmController — KUKA iiwa7 throw planner for mc-pilot-pybullet.

Handles:
  - Loading the iiwa7 URDF into an existing PyBullet client
  - Computing IK + Jacobian pseudoinverse to find joint config and joint
    velocities that produce a desired EE velocity at a given release point
  - Generating a 3-phase piecewise-cubic joint trajectory:
      neutral → windup  [0, t_w]      (rest-to-rest)
      windup  → release [t_w, t_r]    (reaches qd_release at t_r)
      release → rest    [t_r, T]      (follow-through, cosmetic)
  - Commanding joints via PyBullet POSITION_CONTROL each sim step
  - Gripping / releasing the ball via a JOINT_FIXED constraint
"""

import numpy as np
import pybullet as p


# KUKA iiwa7 actual velocity limits (rad/s) per joint
_IIWA_QD_MAX = np.array([1.71, 1.71, 1.75, 2.27, 2.44, 3.14, 3.14])


class ArmController:
    def __init__(self, client_id, urdf_path, base_position=(0, 0, 0), q_neutral=None,
                 vel_limit_multiplier=1.0):
        """
        Parameters
        ----------
        client_id   : int — PyBullet physics client returned by p.connect(...)
        urdf_path   : str — absolute path to kuka_iiwa/model.urdf
        base_position : (3,) — where to mount the arm base in world frame
        q_neutral   : (7,) — neutral joint configuration; defaults to forward pose
        vel_limit_multiplier : float — scale factor on joint velocity limits.
            Set >1.0 to allow faster throws (e.g. 1.5 unlocks ~3.5 m/s EE speed).
            Only relevant for simulated robots; never set >1.0 on real hardware.
        """
        self._cid = client_id
        self._arm_id = p.loadURDF(
            urdf_path,
            basePosition=base_position,
            useFixedBase=True,
            physicsClientId=client_id,
        )
        self._n_joints = p.getNumJoints(self._arm_id, physicsClientId=client_id)
        self._ee_link = self._n_joints - 1   # flange is the last link

        # Read joint limits from URDF
        self._q_lo = np.zeros(self._n_joints)
        self._q_hi = np.zeros(self._n_joints)
        for i in range(self._n_joints):
            ji = p.getJointInfo(self._arm_id, i, physicsClientId=client_id)
            self._q_lo[i] = ji[8]
            self._q_hi[i] = ji[9]

        # Velocity limits — scaled to allow faster simulated throws if needed
        self._qd_max = _IIWA_QD_MAX * vel_limit_multiplier

        if q_neutral is None:
            # Arm angled forward and slightly down: EE at ~(0.69, 0, 0.71)
            self._q_neutral = np.array([0.0, 0.5, 0.0, -1.0, 0.0, 0.5, 0.0])
        else:
            self._q_neutral = np.array(q_neutral, dtype=float)

        self._grip_id = None
        self.reset()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def reset(self):
        """Reset all joints to q_neutral with zero velocity."""
        for i in range(self._n_joints):
            p.resetJointState(
                self._arm_id, i,
                targetValue=self._q_neutral[i],
                targetVelocity=0.0,
                physicsClientId=self._cid,
            )
        self._grip_id = None

    def attach_ball(self, ball_id):
        """
        Weld ball to EE via a fixed constraint.
        Returns the constraint id (stored internally; also returned for logging).
        """
        ee_pos = self.ee_state()[0]
        ball_pos, _ = p.getBasePositionAndOrientation(ball_id, physicsClientId=self._cid)
        offset = np.array(ball_pos) - np.array(ee_pos)
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
        return self._grip_id

    # ------------------------------------------------------------------
    # Throw planning
    # ------------------------------------------------------------------

    def plan_throw(self, v_cmd, release_pos, t_w=0.3, t_r=0.6, T=1.0):
        """
        Plan a 3-phase piecewise-cubic throw trajectory.

        Parameters
        ----------
        v_cmd       : (3,) desired EE linear velocity at release instant (m/s)
        release_pos : (3,) world-frame position for ball release
        t_w, t_r, T : phase boundary times (s)

        Returns
        -------
        coeffs      : dict with keys 'windup', 'throw', 'follow' — each a
                      (n_joints, 4) array of [a0, a1, a2, a3] cubic coefficients
        q_release   : (7,) joint config at release
        qd_release  : (7,) joint velocity at release
        v_achieved  : (3,) actual EE velocity achievable (may differ if limits hit)
        """
        v_cmd = np.array(v_cmd, dtype=float)
        release_pos = np.array(release_pos, dtype=float)

        # ------ IK: find joint config that places EE at release_pos ------
        q_release = np.array(p.calculateInverseKinematics(
            self._arm_id,
            self._ee_link,
            targetPosition=release_pos.tolist(),
            restPoses=self._q_neutral.tolist(),
            lowerLimits=self._q_lo.tolist(),
            upperLimits=self._q_hi.tolist(),
            jointRanges=(self._q_hi - self._q_lo).tolist(),
            maxNumIterations=200,
            residualThreshold=1e-4,
            physicsClientId=self._cid,
        ))

        # ------ Jacobian: qd_release = pinv(J_lin) @ v_cmd ------
        J_lin_raw, _ = p.calculateJacobian(
            self._arm_id,
            self._ee_link,
            localPosition=[0, 0, 0],
            objPositions=q_release.tolist(),
            objVelocities=[0.0] * self._n_joints,
            objAccelerations=[0.0] * self._n_joints,
            physicsClientId=self._cid,
        )
        J_lin = np.array(J_lin_raw)   # (3, 7)
        qd_release = np.linalg.pinv(J_lin) @ v_cmd

        # Feasibility: scale down uniformly if any joint exceeds its limit
        ratio = np.abs(qd_release) / self._qd_max
        if ratio.max() > 1.0:
            qd_release = qd_release / ratio.max()
        v_achieved = J_lin @ qd_release

        # ------ Windup pose: mirror neutral past q_release by 50% ------
        q_windup = self._q_neutral + (q_release - self._q_neutral) * (-0.5)
        q_windup = np.clip(q_windup, self._q_lo, self._q_hi)

        # ------ Follow-through pose: neutral (cosmetic) ------
        q_follow = self._q_neutral.copy()

        # ------ Compute cubic coefficients for each phase ------
        coeffs = {
            'windup':  _cubic_rest_to_rest(self._q_neutral, q_windup, t_w),
            'throw':   _cubic_to_velocity(q_windup, q_release, qd_release, t_r - t_w),
            'follow':  _cubic_from_velocity(q_release, qd_release, q_follow, T - t_r),
            't_w': t_w, 't_r': t_r, 'T': T,
        }
        return coeffs, q_release, qd_release, v_achieved

    def get_setpoint(self, coeffs, t):
        """
        Evaluate the piecewise cubic at time t (seconds from throw start).
        Returns (q_target, qd_target) — both (7,) arrays.
        """
        t_w = coeffs['t_w']
        t_r = coeffs['t_r']
        T   = coeffs['T']
        if t <= t_w:
            return _eval_cubic(coeffs['windup'], t)
        elif t <= t_r:
            return _eval_cubic(coeffs['throw'], t - t_w)
        else:
            return _eval_cubic(coeffs['follow'], min(t - t_r, T - t_r))

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def step(self, q_target, qd_target):
        """Command joints via POSITION_CONTROL for one sim step."""
        p.setJointMotorControlArray(
            self._arm_id,
            list(range(self._n_joints)),
            controlMode=p.POSITION_CONTROL,
            targetPositions=q_target.tolist(),
            targetVelocities=qd_target.tolist(),
            positionGains=[2.0] * self._n_joints,
            velocityGains=[1.0] * self._n_joints,
            physicsClientId=self._cid,
        )

    def release_ball(self, ball_id, set_vel=None, dv_noise=None):
        """
        Remove the grip constraint (ball is now in free flight).

        Parameters
        ----------
        set_vel  : (3,) or None — if given, override ball velocity to this vector
                   (+ dv_noise if also provided). Use this in PyBulletThrowingSystem
                   to set v_cmd as the base velocity, decoupling ball physics from
                   arm-tracking quality.
        dv_noise : (3,) or None — additive noise on top of set_vel (or ee_vel if
                   set_vel is None).
        """
        if self._grip_id is not None:
            p.removeConstraint(self._grip_id, physicsClientId=self._cid)
            self._grip_id = None

        if set_vel is not None:
            release_vel = np.array(set_vel, dtype=float)
        else:
            # Use actual EE velocity (good for the standalone visual demo)
            _, ee_vel, _, _ = self.ee_state()
            release_vel = np.array(ee_vel)

        if dv_noise is not None:
            release_vel = release_vel + np.array(dv_noise)

        # Override ball velocity (clears any constraint-induced angular velocity)
        p.resetBaseVelocity(
            ball_id,
            linearVelocity=release_vel.tolist(),
            angularVelocity=[0.0, 0.0, 0.0],
            physicsClientId=self._cid,
        )
        return release_vel

    def ee_state(self):
        """
        Query end-effector state in world frame.
        Returns (pos, lin_vel, orient_quat, ang_vel).
        """
        ls = p.getLinkState(
            self._arm_id, self._ee_link,
            computeLinkVelocity=1,
            computeForwardKinematics=1,
            physicsClientId=self._cid,
        )
        pos     = np.array(ls[0])    # world position
        orient  = np.array(ls[1])    # world orientation quaternion
        lin_vel = np.array(ls[6])    # world linear velocity
        ang_vel = np.array(ls[7])    # world angular velocity
        return pos, lin_vel, orient, ang_vel

    @property
    def arm_id(self):
        return self._arm_id


# ------------------------------------------------------------------
# Cubic polynomial helpers
# ------------------------------------------------------------------

def _cubic_rest_to_rest(q_start, q_end, dt):
    """
    Cubic coefficients for a rest-to-rest move over [0, dt].
    BCs: q(0)=q_start, qd(0)=0, q(dt)=q_end, qd(dt)=0.
    Shape: (n, 4) — columns [a0, a1, a2, a3].
    """
    dq = q_end - q_start
    a0 = q_start
    a1 = np.zeros_like(q_start)
    a2 =  3.0 * dq / dt**2
    a3 = -2.0 * dq / dt**3
    return np.stack([a0, a1, a2, a3], axis=1)


def _cubic_to_velocity(q_start, q_end, qd_end, dt):
    """
    Cubic from q_start (at rest) to (q_end, qd_end) over [0, dt].
    BCs: q(0)=q_start, qd(0)=0, q(dt)=q_end, qd(dt)=qd_end.
    """
    dq = q_end - q_start
    a0 = q_start
    a1 = np.zeros_like(q_start)
    a3 = (qd_end * dt - 2.0 * dq) / dt**3
    a2 = (3.0 * dq - qd_end * dt) / dt**2
    return np.stack([a0, a1, a2, a3], axis=1)


def _cubic_from_velocity(q_start, qd_start, q_end, dt):
    """
    Cubic from (q_start, qd_start) to q_end (at rest) over [0, dt].
    BCs: q(0)=q_start, qd(0)=qd_start, q(dt)=q_end, qd(dt)=0.
    """
    dq = q_end - q_start
    a0 = q_start
    a1 = qd_start
    a3 = (2.0 * (q_start - q_end) + qd_start * dt) / dt**3
    a2 = (3.0 * (q_end - q_start) - 2.0 * qd_start * dt - 3.0 * a3 * dt**2 * (dt / 3.0)) / dt**2
    # Recompute correctly from the two linear equations:
    #   a2*dt² + a3*dt³ = dq - qd_start*dt
    #   2*a2*dt + 3*a3*dt² = -qd_start
    # => a3 = (2*(q_start - q_end) + qd_start*dt) / dt³  (already above)
    # => a2 = (-qd_start - 3*a3*dt²) / (2*dt)
    a3 = (2.0 * (q_start - q_end) + qd_start * dt) / dt**3
    a2 = (-qd_start - 3.0 * a3 * dt**2) / (2.0 * dt)
    return np.stack([a0, a1, a2, a3], axis=1)


def _eval_cubic(coeffs, tau):
    """
    Evaluate piecewise cubic at local time tau.
    coeffs : (n, 4) — columns [a0, a1, a2, a3]
    Returns (q, qd) — each (n,).
    """
    a0, a1, a2, a3 = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2], coeffs[:, 3]
    q  = a0 + a1 * tau + a2 * tau**2 + a3 * tau**3
    qd = a1 + 2.0 * a2 * tau + 3.0 * a3 * tau**2
    return q, qd
