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

_GRAVITY = np.array([0.0, 0.0, -9.81])  # matches p.setGravity(...) used throughout


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
        self._windup_delta = (
            np.array(self._profile.windup_delta, dtype=float)
            if self._profile.windup_delta is not None
            else None
        )
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
            # calculateInverseDynamics needs a FULL-length q/qd/qdd vector
            # (every non-fixed joint, e.g. Panda's 2 gripper-finger prismatic
            # joints beyond the 7 controlled arm joints), not just the
            # controlled ones. step()'s torque branch pads with zeros at the
            # PASSIVE dof positions -- exact only if those positions are a
            # contiguous trailing block (verified for franka_panda_dyn:
            # joint_ids map to dof 0..6, fingers land at dof 7,8). Any profile
            # violating that ordering needs real index-scatter handling, not
            # this zero-pad shortcut.
            if self._dof_ids != list(range(len(self._joint_ids))):
                raise ValueError(
                    f"Torque mode's zero-pad shortcut requires the controlled "
                    f"joints to occupy the FIRST {len(self._joint_ids)} dof "
                    f"positions contiguously; got dof_ids={self._dof_ids}."
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
        self._payload_mass = None
        self.reset()

    def reset(self):
        """Reset actuated joints to q_neutral with zero velocity."""
        self._payload_mass = None
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
        if self._control_mode == "torque":
            # calculateInverseDynamics only sees the arm's own URDF mass; a
            # rigidly gripped payload otherwise goes uncompensated in the
            # feedforward, producing a real tracking lag. Inflating the EE
            # link's own mass (tried first) does NOT work: end_effector_link
            # is a massless (0 kg / 0 inertia) tool-tip frame fixed-jointed to
            # bracelet_link, and neither location's mass responds correctly to
            # changeDynamics through calculateInverseDynamics for this URDF —
            # empirically needed ~20x the real ball mass to close the gap.
            # The correct fix (added mass acts at the EE, not at either link's
            # own origin) is an explicit Jacobian-transpose point-mass term,
            # applied in step(). A real Kortex controller would be configured
            # with this same known payload mass for its own dynamics model.
            self._payload_mass = float(
                p.getDynamicsInfo(ball_id, -1, physicsClientId=self._cid)[0]
            )
        return self._grip_id

    def plan_throw(self, v_cmd, release_pos, t_w=0.3, t_r=0.6, T=1.0,
                   q_release_override=None, qd_release_override=None,
                   monotonic_windup=False):
        """
        Plan a 3-phase piecewise-cubic throw trajectory.

        Overrides (opt-in; default None keeps the original IK+pinv behaviour):
          q_release_override  : explicit release joint configuration (n,) -- skip IK.
          qd_release_override : explicit release joint velocity (n,) -- skip pinv, so the
                                arm can use VELOCITY-LIMIT-OPTIMAL joint scheduling
                                (qd = qd_max*sign(d.J)) instead of the min-norm pinv.

        Returns
        -------
        coeffs, q_release, qd_release (post-clip), v_achieved (post-clip EE velocity)
        """
        v_cmd = np.array(v_cmd, dtype=float)
        release_pos = np.array(release_pos, dtype=float)

        if q_release_override is not None:
            q_release = np.array(q_release_override, dtype=float)
        else:
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
        if qd_release_override is not None:
            qd_release = np.array(qd_release_override, dtype=float)
        else:
            qd_release = np.linalg.pinv(j_lin) @ v_cmd

        ratio = np.abs(qd_release) / self._qd_max
        clip_scale = 1.0
        if ratio.max() > 1.0:
            clip_scale = float(ratio.max())
            qd_release = qd_release / clip_scale
        v_achieved = j_lin @ qd_release

        dt_windup = t_w
        dt_throw = t_r - t_w
        follow_dur = T - t_r
        windup_time_scale = 1.0
        time_scale = 1.0

        # Proximal-to-distal (kinetic-chain) stagger: joint index IS the physical
        # base->wrist chain order for this URDF, so joint i's throw-phase ramp
        # starts at stagger_frac[i]*dt_throw and ends at dt_throw (release) for
        # EVERY joint -- proximal joints (small i) get the whole window and move
        # gradually; distal joints (large i) stay cocked and ramp late/sharp, the
        # whip-like sequencing real throws use (Senoo & Ishikawa 2008; proximal-
        # to-distal / "summation of speed" principle). Each joint's own peak
        # velocity is still exactly its qd_release_i <= qd_max_i (monotonic single
        # ramp per joint), so the whole-trajectory qd<=qd_max guarantee is
        # unaffected by the stagger.
        stagger_frac = (np.linspace(0.0, 0.5, self._n_dofs) if monotonic_windup
                        else np.zeros(self._n_dofs))

        def _windup_pose_and_time(dt_throw_local, dt_windup_local):
            if monotonic_windup:
                local_dur = dt_throw_local * (1.0 - stagger_frac)
                # Cock back by half each joint's OWN local window so its throw-
                # phase ramp is linear 0 -> qd_release (peak = qd_release <= qd_max).
                qw = q_release - qd_release * (local_dur / 2.0)
                qw = np.clip(qw, self._q_lo, self._q_hi)
                # Grow the neutral->windup time so its rest-to-rest cubic peak
                # (1.5*|dq|/t) stays within qd_max.
                span = np.max(np.abs(qw - self._q_neutral))
                min_tw = 1.5 * span / float(np.min(self._qd_max))
                return qw, max(dt_windup_local, min_tw), local_dur
            if self._windup_delta is not None:
                # Explicit cocked-back pose, independent of q_release (needed when
                # q_release == q_neutral, which makes the formula below degenerate --
                # see kinova_gen3's profile notes).
                qw = self._q_neutral + self._windup_delta
            else:
                qw = self._q_neutral + (q_release - self._q_neutral) * (-0.5)
            return np.clip(qw, self._q_lo, self._q_hi), dt_windup_local, None

        q_windup, dt_windup, local_dur = _windup_pose_and_time(dt_throw, dt_windup)
        q_follow = self._q_neutral.copy()
        windup_coeffs = _cubic_rest_to_rest(self._q_neutral, q_windup, dt_windup)
        throw_coeffs = _cubic_to_velocity(
            q_windup, q_release, qd_release, local_dur if monotonic_windup else dt_throw
        )

        if self._control_mode == "torque":
            for _ in range(6):
                ratio, worst_tau = self._throw_peak_torque_ratio(windup_coeffs, dt_windup)
                if ratio <= 1.0:
                    break
                dt_windup *= 1.2
                windup_time_scale *= 1.2
                windup_coeffs = _cubic_rest_to_rest(self._q_neutral, q_windup, dt_windup)
            else:
                ratio, worst_tau = self._throw_peak_torque_ratio(windup_coeffs, dt_windup)
                if ratio > 1.0:
                    report = ", ".join(
                        f"j{j}: {abs(t):.1f}/{m:.1f} Nm"
                        for j, (t, m) in enumerate(zip(worst_tau, self._tau_max))
                    )
                    raise RuntimeError(
                        f"Windup infeasible after 6 time-scaling iterations "
                        f"(peak ratio {ratio:.2f}): {report}"
                    )

            stagger_start = stagger_frac * dt_throw if monotonic_windup else None
            for _ in range(6):
                ratio, worst_tau = self._throw_peak_torque_ratio(
                    throw_coeffs, dt_throw, stagger_start=stagger_start
                )
                if ratio <= 1.0:
                    break
                dt_throw *= 1.2
                time_scale *= 1.2
                stagger_start = stagger_frac * dt_throw if monotonic_windup else None
                if monotonic_windup:
                    # Stretched throw -> larger cock; keep the ramp linear and
                    # rebuild the windup so it still starts from the cocked pose.
                    q_windup, dt_windup, local_dur = _windup_pose_and_time(
                        dt_throw, dt_windup
                    )
                    windup_coeffs = _cubic_rest_to_rest(
                        self._q_neutral, q_windup, dt_windup
                    )
                throw_coeffs = _cubic_to_velocity(
                    q_windup, q_release, qd_release,
                    local_dur if monotonic_windup else dt_throw,
                )
            else:
                ratio, worst_tau = self._throw_peak_torque_ratio(
                    throw_coeffs, dt_throw, stagger_start=stagger_start
                )
                if ratio > 1.0:
                    report = ", ".join(
                        f"j{j}: {abs(t):.1f}/{m:.1f} Nm"
                        for j, (t, m) in enumerate(zip(worst_tau, self._tau_max))
                    )
                    raise RuntimeError(
                        f"Throw infeasible after 6 time-scaling iterations "
                        f"(peak ratio {ratio:.2f}): {report}"
                    )

        t_w_actual = dt_windup
        t_r_actual = t_w_actual + dt_throw

        follow_coeffs = _cubic_from_velocity(q_release, qd_release, q_follow, follow_dur)
        if self._control_mode == "torque":
            # windup and throw both stretch their duration until torque-
            # feasible (or raise); follow (post-release decel back to
            # neutral) had NEITHER -- follow_dur was fixed at the ORIGINAL
            # nominal (T - t_r), never grown even when windup/throw needed a
            # lot of stretching to become feasible. Starting from a high
            # release velocity and covering a large q_release -> q_neutral
            # distance in that now-too-short fixed window forced extreme
            # mid-phase velocities to make up the distance in time (measured:
            # 556% of qd_max, 3.2x tau_max for an extended overhead release)
            # -- not a rendering artifact, a trajectory the real arm cannot
            # execute. Same time-scaling pattern as windup/throw.
            # Follow-through must satisfy BOTH velocity and torque, and the
            # feasible window is NARROW and non-monotonic, so scan candidate
            # durations instead of growing one guess:
            #   * peak |qd| falls monotonically with duration (the cubic has
            #     less distance to cover per second) -- too SHORT violates it
            #     (measured 5.6x qd_max at the nominal 0.6s).
            #   * peak |tau| bottoms out mid-range then RISES again (accel
            #     term shrinks, but the arm then spends longer coasting
            #     through high-gravity-torque configurations) -- too LONG
            #     violates it (1.17x at 10s).
            # For the shipped overhead release only dt ~3.5s satisfies both
            # (qd 0.96, tau 0.93); a torque-only growth loop stopped at 1.79s
            # where torque passed (0.95) but velocity was still 1.88x limit.
            # Unlike windup/throw -- whose monotonic ramps hold qd <= qd_max
            # by construction -- nothing bounds the follow phase's velocity
            # a priori, so it must be checked explicitly.
            # Candidate durations are ABSOLUTE, not multiples of the caller's
            # nominal follow window: that window is (T - t_r), so a caller
            # passing a longer horizon produced a coarse grid that stepped
            # straight over the feasible band (measured: T=2.2 found a
            # solution, T=2.0 scanned 2.0s upward and reported infeasible for
            # the same release state). The caller's own value is kept as a
            # candidate so the nominal timing still wins when it is feasible.
            # Only ever GROW the caller's window -- shortening it would make
            # the recovery more aggressive than asked for and silently change
            # the trajectory's timing contract.
            base_follow = follow_dur
            cand_durs = [base_follow] + [d for d in (0.6, 0.9, 1.2, 1.5, 1.8, 2.2,
                                                    2.6, 3.0, 3.6, 4.4, 5.5, 7.0)
                                         if d > base_follow]
            best = None
            for dur in cand_durs:
                cand = _cubic_from_velocity(q_release, qd_release, q_follow, dur)
                tau_ratio, worst_tau = self._throw_peak_torque_ratio(cand, dur)
                qd_ratio = self._peak_qd_ratio(cand, dur)
                if best is None or max(tau_ratio, qd_ratio) < best[0]:
                    best = (max(tau_ratio, qd_ratio), worst_tau, qd_ratio, tau_ratio)
                if tau_ratio <= 1.0 and qd_ratio <= 1.0:
                    follow_dur, follow_coeffs = dur, cand
                    break
            else:
                _, worst_tau, qd_ratio, tau_ratio = best
                report = ", ".join(
                    f"j{j}: {abs(t):.1f}/{m:.1f} Nm"
                    for j, (t, m) in enumerate(zip(worst_tau, self._tau_max))
                )
                raise RuntimeError(
                    f"Follow-through infeasible over the scanned durations "
                    f"(best: torque {tau_ratio:.2f}x, velocity {qd_ratio:.2f}x "
                    f"of limit): {report}"
                )

        T_actual = t_r_actual + follow_dur

        coeffs = {
            "windup": windup_coeffs,
            "throw": throw_coeffs,
            "follow": follow_coeffs,
            "t_w": t_w_actual,
            "windup_time_scale": windup_time_scale,
            "t_r": t_r_actual,
            "T": T_actual,
            "clip_scale": clip_scale,
            "time_scale": time_scale,
            "stagger_frac": stagger_frac if monotonic_windup else None,
        }
        return coeffs, q_release, qd_release, v_achieved

    def _peak_qd_ratio(self, coeffs, dur, n_samples=60):
        """Max over a phase of max_j |qd_j| / qd_max_j. The windup and throw
        phases bound this by construction (monotonic ramps peaking at
        qd_release <= qd_max); the follow phase does NOT, so it needs an
        explicit check."""
        worst = 0.0
        for t in np.linspace(0.0, dur, n_samples):
            _, qd, _ = _eval_cubic(coeffs, t, with_accel=True)
            worst = max(worst, float(np.max(np.abs(qd) / self._qd_max)))
        return worst

    def _throw_peak_torque_ratio(self, throw_coeffs, dt_throw, n_samples=50, stagger_start=None):
        """Max over the throw phase of max_j |tau_j| / tau_max_j."""
        worst = 0.0
        worst_tau = None
        for tau_t in np.linspace(0.0, dt_throw, n_samples):
            tau_eval = (tau_t if stagger_start is None
                       else np.clip(tau_t - stagger_start, 0.0, dt_throw - stagger_start))
            q, qd, qdd = _eval_cubic(throw_coeffs, tau_eval, with_accel=True)
            n_pad = self._n_dofs - len(self._joint_ids)
            q_full = np.concatenate([q, np.zeros(n_pad)]) if n_pad else q
            qd_full = np.concatenate([qd, np.zeros(n_pad)]) if n_pad else qd
            qdd_full = np.concatenate([qdd, np.zeros(n_pad)]) if n_pad else qdd
            torque = np.array(
                p.calculateInverseDynamics(
                    self._arm_id,
                    q_full.tolist(),
                    qd_full.tolist(),
                    qdd_full.tolist(),
                    physicsClientId=self._cid,
                )
            )[: len(self._joint_ids)]
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
            dt_throw = t_r - t_w
            stagger_frac = coeffs.get("stagger_frac")
            if stagger_frac is not None:
                start_i = stagger_frac * dt_throw
                tau = np.clip((t - t_w) - start_i, 0.0, dt_throw - start_i)
            else:
                tau = t - t_w
            return _eval_cubic(coeffs["throw"], tau, with_accel)
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
            # calculateInverseDynamics needs a FULL-length vector (every non-
            # fixed joint, e.g. Panda's 2 passive gripper-finger joints beyond
            # our N controlled ones) -- zero-pad the tail; exact because the
            # constructor already checked our dofs are the leading block.
            n_pad = self._n_dofs - len(self._joint_ids)
            q_full = np.concatenate([q_meas, np.zeros(n_pad)]) if n_pad else q_meas
            qd_full = np.concatenate([qd_meas, np.zeros(n_pad)]) if n_pad else qd_meas
            qdd_full = np.concatenate([qdd_cmd, np.zeros(n_pad)]) if n_pad else qdd_cmd
            # Inverse dynamics for the arm's own URDF mass only.
            tau = np.array(
                p.calculateInverseDynamics(
                    self._arm_id,
                    q_full.tolist(),
                    qd_full.tolist(),
                    qdd_full.tolist(),
                    physicsClientId=self._cid,
                )
            )
            tau = tau[: len(self._joint_ids)]  # drop the padded passive-dof entries
            if self._payload_mass is not None:
                # Rigidly gripped payload (the ball) is a separate PyBullet
                # body coupled via a runtime constraint, invisible to
                # calculateInverseDynamics no matter which arm link's URDF
                # mass is inflated (verified empirically: doesn't propagate
                # correctly through this URDF's fixed end-effector joint).
                # Add its dynamic contribution analytically instead: a point
                # mass rigidly attached at the EE contributes generalized
                # force J^T * m * (a_ee - g), where a_ee = J * qdd_cmd is the
                # commanded EE linear acceleration (Jacobian time-derivative /
                # Coriolis term omitted as a first-order approximation).
                q_meas_full = self._ik_q_neutral.copy()
                for local_i, dof_id in enumerate(self._dof_ids):
                    q_meas_full[dof_id] = q_meas[local_i]
                j_lin_raw, _ = p.calculateJacobian(
                    self._arm_id,
                    self._ee_link,
                    localPosition=[0, 0, 0],
                    objPositions=q_meas_full.tolist(),
                    objVelocities=[0.0] * self._n_dofs,
                    objAccelerations=[0.0] * self._n_dofs,
                    physicsClientId=self._cid,
                )
                j_lin = np.array(j_lin_raw)[:, self._dof_ids]
                a_ee = j_lin @ qdd_cmd
                f_payload = self._payload_mass * (a_ee - _GRAVITY)
                tau = tau + j_lin.T @ f_payload
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
        dynamic=False,
    ):
        """
        Remove the grip constraint and optionally override the ball velocity.

        dynamic=True: the ball keeps whatever velocity the physics engine gave
        it while dragged by the constraint — no resetBaseVelocity, no
        repositioning; set_vel/dv_noise/release_pos are ignored.
        """
        if self._grip_id is not None:
            p.removeConstraint(self._grip_id, physicsClientId=self._cid)
            self._grip_id = None
        if self._attached_ball_id is not None:
            if not keep_collision_disabled:
                self._set_ball_collision_with_arm(self._attached_ball_id, enable=True)
            self._attached_ball_id = None
            self._payload_mass = None

        if dynamic:
            ball_vel, _ = p.getBaseVelocity(ball_id, physicsClientId=self._cid)
            return np.array(ball_vel)

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
