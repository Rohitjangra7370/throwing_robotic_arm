"""
Optimized-posture release solver -- SHARED BY SIMULATION AND HARDWARE.

This module owns the single most safety- and accuracy-critical computation in
the project: given a trained policy's commanded release speed and a target,
produce the joint configuration, joint velocities and release point that the
throw trajectory is built from.

It lives on its own, outside `PyBulletThrowingSystem`, for one reason: the real
Kinova Gen3 must execute the SAME release the simulator was trained and
validated on. When this logic was a private method of the sim system, the
hardware entry point (`run_hardware_throw.py`) could not reach it and instead
re-derived a release with plain IK + `pinv` -- a completely different, much
weaker, near-horizontal throw. Anything that plans a throw, in sim or on
hardware, must call `OptimizedReleaseSolver.solve()`. Do not re-implement it,
and do not copy it: every historical bug in this project (corkscrew motion from
un-frozen roll joints, the +-2pi wrap that silently zeroed release speed, the
turret-aiming offset, the landing-distance formula) has lived in exactly these
hundred lines, and a second copy guarantees the two paths drift apart.

The physics rationale for each step is kept inline, verbatim from where this
code originated in `model_pybullet.py`.
"""

import numpy as np
import pybullet as p


class OptimizedReleaseSolver:
    """
    AIMED frozen-base release.

    Two modes, mirroring `find_throw_pose.py`'s output formats:

      * ``table``   -- an azimuth->posture TABLE. The correct mode. Looks up the
        nearest-azimuth posture per target, because rotating a single posture's
        base does NOT aim off-axis on this arm (J(base+az) != Rz*J(base)).
        Tables carrying a ``v_dir`` key additionally use the EXACT-qd path (see
        `solve`), which is what the overhead/sagittal throw uses.
      * ``posture`` -- legacy single fixed posture, base-rotated at runtime.
        Kept only for the standalone single-shot scripts.

    ``active`` is False when neither is configured, i.e. the caller should fall
    back to the ordinary IK + pinv release.
    """

    def __init__(self, opt_posture=None, opt_launch_deg=43.0, opt_posture_table=None):
        self.posture = None if opt_posture is None else np.array(opt_posture, dtype=float)
        self.table = list(opt_posture_table) if opt_posture_table is not None else None
        self.launch = np.deg2rad(opt_launch_deg)
        # (r_off, alpha_off) turret constants, measured lazily via FK on first use.
        self._opt_polar = None

    @property
    def active(self):
        """True when running in optimized-posture release mode (table or legacy)."""
        return self.table is not None or self.posture is not None

    def lookup_posture(self, azimuth):
        """Nearest-azimuth table entry -> (posture q (7,), launch elevation rad)."""
        azs = np.array([e["azimuth_deg"] for e in self.table])
        i = int(np.argmin(np.abs(azs - np.degrees(azimuth))))
        e = self.table[i]
        return np.array(e["q"], dtype=float), np.deg2rad(float(e["elev_deg"]))

    def solve(self, arm, v_cmd, target_xy=None):
        """
        Pick a hardware-valid posture for the target azimuth (nearest table entry;
        base = azimuth, held still), then solve the DIRECTION-CONSTRAINED joint
        velocities (qd[0]=0) so the EE velocity points EXACTLY along the launch
        direction d, scaled to the commanded speed.

        Parameters
        ----------
        arm : ArmController
            Used for FK/Jacobian queries only. Left restored to ``q_neutral`` on
            return -- the FK teleports below are QUERIES, not the start of motion.
        v_cmd : (3,)
            Commanded EE velocity; only its magnitude is used for the table path.
        target_xy : (2,) or None
            Target in the base frame, for azimuth aiming. None => azimuth 0.

        Returns
        -------
        (release_pos, q_release, qd_release, v_release)
            Feed the middle two to ``plan_throw`` as ``q_release_override`` /
            ``qd_release_override``.
        """
        speed = float(np.linalg.norm(v_cmd))
        tgt = None if target_xy is None else np.asarray(target_xy, dtype=float)
        azimuth = float(np.arctan2(tgt[1], tgt[0])) if tgt is not None else 0.0
        rotation_built = (self.table is not None
                          and bool(self.table[0].get("rotation_built", False)))
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
            if self._opt_polar is None:
                # (r_off, alpha_off) are rotation-invariant posture constants:
                # measure once from any entry via FK at its stored q.
                e0 = self.table[0]
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
        if self.table is not None:
            # Nearest-azimuth posture; rotate the base so the posture aims at
            # the (turret-corrected) heading.
            q_release, elev = self.lookup_posture(azimuth)
            if rotation_built:
                # Entries store q[0] = q0[0] - az_entry (VERIFIED sign: this
                # URDF's base joint measures opposite the world-z rotation
                # sense, |v_rot - Rz v0| ~ 1e-6). Recover q0[0] and rotate to
                # the exact corrected heading.
                near = min(self.table,
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
            q_release = self.posture.copy()
            q_release[0] = self.posture[0] + azimuth
            elev = self.launch
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
