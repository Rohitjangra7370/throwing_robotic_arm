"""
Closed-loop throw harness -- executes a plan through the REAL torque controller
and measures what actually came out.

This is the only honest way to compare trajectory profiles.  `plan_throw`'s
feasibility numbers and the GP's "final trial cost" both describe a BELIEF; the
quantity that decides whether a throw lands is the end-effector velocity the
arm physically had at the instant the ball left it.  The shipped pipeline
already measures that (`PyBulletThrowingSystem.last_release_info["v_release"]`)
and the shipped answer is 90.3% of commanded speed, averaged over the 225
throws in `results_tracking_error/tracking_error.npz`.  That 9.7% is the
headroom this lab is trying to recover.

The world built here is deliberately identical to
`PyBulletThrowingSystem._simulate_pybullet`:
  * same URDF via the same `ArmController` (so the same massless-link repair,
    the same computed-torque + gravity-comp + Jacobian-transpose payload term)
  * same dt (0.02 s -> 50 Hz control AND physics; nsub is 1 in the shipped code)
  * same rigid `JOINT_FIXED` grip, released by removing the constraint and
    letting the ball keep the velocity physics gave it (`dynamic=True`)
  * same paper Eq. 35 free-flight drag via `simulation_class.model._ball_accel`
  * same descending-crossing landing test and linear interpolation onto the
    landing plane

The differences are all measurement, never physics: the harness records the
tracking error and the commanded/saturated torque each step, and it can shift
the release step to measure release-timing sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pybullet as p
import pybullet_data

from robot_arm.arm_controller import ArmController
from robot_arm.robot_profiles import get_robot_profile
from simulation_class.model import _ball_accel

_GRAVITY = np.array([0.0, 0.0, -9.81])


@dataclass
class ThrowResult:
    label: str
    v_planned: np.ndarray        # J(q_release) @ qd_release, the plan's own target
    v_release: np.ndarray        # what the ball actually left with
    omega_release: np.ndarray    # EE angular velocity at release (TCP-offset driver)
    release_pos: np.ndarray
    release_pos_planned: np.ndarray
    land_xy: np.ndarray
    flight_time: float
    t_r: float
    dt_throw: float
    peak_q_err: float
    peak_qd_err: float
    q_err_at_release: np.ndarray
    tau_sat_frac: float          # fraction of joint-steps clipped at tau_max
    peak_tau_ratio: float
    peak_jerk: float
    accel_step: float            # max |delta qdd| across a phase join [rad/s^2]
    released_step: int
    extra: dict = field(default_factory=dict)

    @property
    def speed_planned(self):
        return float(np.linalg.norm(self.v_planned))

    @property
    def speed_release(self):
        return float(np.linalg.norm(self.v_release))

    @property
    def speed_ratio(self):
        s = self.speed_planned
        return float(self.speed_release / s) if s > 1e-9 else 0.0

    @property
    def dir_err_deg(self):
        a, b = self.v_planned, self.v_release
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return float("nan")
        return float(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1.0, 1.0))))

    @property
    def release_pos_err(self):
        return float(np.linalg.norm(self.release_pos - self.release_pos_planned))

    def row(self):
        return (
            f"{self.label:<24} "
            f"v {self.speed_release:5.3f}/{self.speed_planned:5.3f} "
            f"({100 * self.speed_ratio:5.1f}%) "
            f"dir {self.dir_err_deg:5.2f}deg  "
            f"|w| {np.linalg.norm(self.omega_release):5.3f}  "
            f"posErr {100 * self.release_pos_err:5.2f}cm  "
            f"qErr {1000 * self.peak_q_err:6.1f}mrad  "
            f"sat {100 * self.tau_sat_frac:4.1f}%  "
            f"t_r {self.t_r:5.2f}s"
        )


class ThrowHarness:
    """One persistent PyBullet world; many throws.

    Use as a context manager.  `self.arm` is a real `ArmController` and is the
    object both the release solver and the feasibility checker should be handed,
    so planning and execution see bit-identical kinematics.
    """

    def __init__(
        self,
        robot_name="kinova_gen3_dyn",
        base_height=0.0,
        ball_mass=0.0577,
        ball_radius=0.0327,
        dt=0.02,
        target_height=0.0,
        gui=False,
        max_flight=3.0,
    ):
        self.robot_name = robot_name
        self.profile = get_robot_profile(robot_name)
        self.base_height = float(base_height)
        self.ball_mass = float(ball_mass)
        self.ball_radius = float(ball_radius)
        self.dt = float(dt)
        self.target_height = float(target_height)
        self.gui = bool(gui)
        self.max_flight = float(max_flight)
        self.client = None
        self.arm = None
        self.ball_id = None

    # -- lifecycle --------------------------------------------------------
    def __enter__(self):
        self.client = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.dt, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        urdf = pybullet_data.getDataPath() + "/" + self.profile.urdf_rel_path
        self.arm = ArmController(
            self.client,
            urdf,
            robot_name=self.robot_name,
            base_position=(0.0, 0.0, self.base_height),
        )
        self.arm.reset()
        ee0, _, _, _ = self.arm.ee_state()
        col = p.createCollisionShape(
            p.GEOM_SPHERE, radius=self.ball_radius, physicsClientId=self.client
        )
        vis = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self.ball_radius,
            rgbaColor=[1, 1, 0, 1],
            physicsClientId=self.client,
        )
        self.ball_id = p.createMultiBody(
            baseMass=self.ball_mass,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=ee0.tolist(),
            physicsClientId=self.client,
        )
        p.changeDynamics(
            self.ball_id, -1, linearDamping=0.0, angularDamping=0.0,
            physicsClientId=self.client,
        )
        return self

    def __exit__(self, *exc):
        if self.client is not None:
            p.disconnect(self.client)
            self.client = None
        return False

    # -- helpers ----------------------------------------------------------
    def _reset_for_throw(self):
        self.arm.reset()
        ee0, _, _, _ = self.arm.ee_state()
        p.resetBasePositionAndOrientation(
            self.ball_id, ee0.tolist(), [0, 0, 0, 1], physicsClientId=self.client
        )
        p.resetBaseVelocity(
            self.ball_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client
        )
        self.arm.attach_ball(self.ball_id)

    def jacobian(self, q):
        arm = self.arm
        q_full = arm._ik_q_neutral.copy()
        for local_i, dof_id in enumerate(arm._dof_ids):
            q_full[dof_id] = q[local_i]
        jl, ja = p.calculateJacobian(
            arm._arm_id,
            arm._ee_link,
            [0, 0, 0],
            q_full.tolist(),
            [0.0] * arm._n_dofs,
            [0.0] * arm._n_dofs,
            physicsClientId=arm._cid,
        )
        return np.array(jl)[:, arm._dof_ids], np.array(ja)[:, arm._dof_ids]

    def fk(self, q):
        arm = self.arm
        q_full = arm._ik_q_neutral.copy()
        for local_i, dof_id in enumerate(arm._dof_ids):
            q_full[dof_id] = q[local_i]
        saved = [
            p.getJointState(arm._arm_id, j, physicsClientId=arm._cid)[:2]
            for j in arm._joint_ids
        ]
        for j in range(arm._n_dofs):
            p.resetJointState(arm._arm_id, j, q_full[j], physicsClientId=arm._cid)
        pos = np.array(
            p.getLinkState(
                arm._arm_id, arm._ee_link, computeForwardKinematics=True,
                physicsClientId=arm._cid,
            )[4]
        )
        for local_i, j in enumerate(arm._joint_ids):
            p.resetJointState(
                arm._arm_id, j, saved[local_i][0], saved[local_i][1],
                physicsClientId=arm._cid,
            )
        return pos

    def _tau_command(self, q_t, qd_t, qdd_t):
        """Replicate `ArmController.step`'s torque command, read-only.

        Needed because PyBullet does not report back what a TORQUE_CONTROL
        command was clipped to, and the CLIP is exactly the quantity of
        interest: a saturated joint stops tracking, and that is the mechanism
        by which an over-aggressive profile loses release speed.
        """
        arm = self.arm
        states = p.getJointStates(arm._arm_id, arm._joint_ids, physicsClientId=arm._cid)
        q_meas = np.array([s[0] for s in states])
        qd_meas = np.array([s[1] for s in states])
        e = np.asarray(q_t, dtype=float) - q_meas
        ed = np.asarray(qd_t, dtype=float) - qd_meas
        qdd_cmd = np.asarray(qdd_t, dtype=float) + arm._kp * e + arm._kd * ed
        tau = arm.inverse_dynamics(q_meas, qd_meas, qdd_cmd)
        if arm._payload_mass is not None:
            j_lin, _ = self.jacobian(q_meas)
            a_ee = j_lin @ qdd_cmd
            tau = tau + j_lin.T @ (arm._payload_mass * (a_ee - _GRAVITY))
        return tau, q_meas, qd_meas, e, ed

    def free_flight(self, pos, vel):
        """Ball-only flight from (pos, vel) to the landing plane.

        Used to compute the IDEAL landing for a plan -- where the ball would go
        if the arm tracked its own plan perfectly.  Runs the identical
        integrator, drag model, timestep and descending-crossing test as `run`,
        so the difference between the two is pure execution error with no
        integrator mismatch folded in.  The arm is parked at neutral and its
        collision with the ball left disabled.
        """
        pos = np.asarray(pos, dtype=float)
        vel = np.asarray(vel, dtype=float)
        self.arm.reset()
        self.arm._set_ball_collision_with_arm(self.ball_id, enable=False)
        p.resetBasePositionAndOrientation(
            self.ball_id, pos.tolist(), [0, 0, 0, 1], physicsClientId=self.client
        )
        p.resetBaseVelocity(
            self.ball_id, vel.tolist(), [0, 0, 0], physicsClientId=self.client
        )
        prev = None
        h = self.target_height
        n = int(self.max_flight / self.dt) + 10
        for step in range(n):
            cur, _ = p.getBasePositionAndOrientation(
                self.ball_id, physicsClientId=self.client
            )
            v, _ = p.getBaseVelocity(self.ball_id, physicsClientId=self.client)
            cur = np.array(cur)
            v = np.array(v)
            a_total = _ball_accel(cur, v, self.ball_mass, self.ball_radius, np.zeros(3))
            p.applyExternalForce(
                self.ball_id, -1, (self.ball_mass * (a_total - _GRAVITY)).tolist(),
                [0, 0, 0], p.WORLD_FRAME, physicsClientId=self.client,
            )
            if (
                cur[2] <= h + self.ball_radius + 0.005
                and v[2] < 0.0
                and prev is not None
                and prev[2] > h
                and step > 2
            ):
                frac = (prev[2] - h) / max(prev[2] - cur[2], 1e-12)
                land = prev + frac * (cur - prev)
                return land[:2], (step - 1 + frac) * self.dt
            prev = cur.copy()
            p.stepSimulation(physicsClientId=self.client)
        return np.array([np.nan, np.nan]), float("nan")

    @staticmethod
    def accel_step(plan, eps=1e-6):
        """Largest acceleration DISCONTINUITY across the plan's phase joins.

        The honest jerk metric for a piecewise-cubic plan.  Inside a cubic
        segment jerk is a small constant (6*a3), so a max-|qddd| number makes
        the shipped profile look smooth; the physically meaningful violation is
        the STEP in commanded acceleration at t=0, t_w and t_r, which is an
        unbounded jerk impulse the arm answers with a torque spike.
        """
        joins = getattr(plan, "join_times", (0.0, plan.t_w, plan.t_r))
        worst = 0.0
        for tj in joins:
            before = (np.zeros_like(plan.eval(0.0)[2]) if tj <= eps
                      else plan.eval(tj - eps)[2])
            after = plan.eval(tj + eps)[2]
            worst = max(worst, float(np.max(np.abs(after - before))))
        return worst

    # -- the run ----------------------------------------------------------
    def run(self, plan, label=None, release_bias_steps=0, record_traj=False):
        """Execute `plan` (anything with `.eval(t)`, `.t_w`, `.t_r`, `.T`).

        `release_bias_steps` shifts the release by whole control steps, which is
        how release-timing sensitivity is measured: on hardware the gripper
        latency is 67.9 +- 6.4 ms and the command quantum is 25 ms, so +-1 step
        at 50 Hz (20 ms) is the right unit.
        """
        if self.client is None:
            raise RuntimeError("use ThrowHarness as a context manager")
        arm = self.arm
        dt = self.dt
        self._reset_for_throw()

        j_lin, j_ang = self.jacobian(plan.q_release)
        v_planned = j_lin @ plan.qd_release
        release_pos_planned = self.fk(plan.q_release)

        # The shipped executor uses `int(t_r_actual / dt_phys)`
        # (model_pybullet.py:283).  That floors -- so it already releases up to
        # one full control step EARLY -- and it is float-fragile on top: with
        # t_r = 4.18 and dt = 0.02, t_r/dt evaluates to 208.99999999999997 and
        # int() drops a further whole 20 ms step.  Measured cost of that single
        # lost step at this release state: 2.5 deg of direction error and
        # 2.6 cm of landing error, because the shipped profile is still at full
        # acceleration when the ball leaves.  The epsilon here removes the
        # float artefact so profile rows differ by profile, not by rounding
        # luck; the systematic floor is kept, because that IS what the shipped
        # executor does.
        release_step = int(plan.t_r / dt + 1e-9) + int(release_bias_steps)
        total_steps = int((plan.t_r + self.max_flight) / dt) + 100

        released = False
        peak_q_err = 0.0
        peak_qd_err = 0.0
        q_err_at_release = np.zeros(len(arm._joint_ids))
        sat_hits = 0
        sat_slots = 0
        peak_tau_ratio = 0.0
        peak_jerk = 0.0
        v_release = np.zeros(3)
        omega_release = np.zeros(3)
        release_pos = release_pos_planned.copy()
        released_at = -1
        land_xy = np.array([np.nan, np.nan])
        flight_time = float("nan")
        traj = [] if record_traj else None
        prev_pos = None

        for step in range(total_steps):
            t = step * dt
            q_t, qd_t, qdd_t, qddd_t = plan.eval(t)
            if not released:
                peak_jerk = max(peak_jerk, float(np.max(np.abs(qddd_t))))

            tau, q_meas, qd_meas, e, ed = self._tau_command(q_t, qd_t, qdd_t)
            r = np.abs(tau) / arm._tau_max
            peak_tau_ratio = max(peak_tau_ratio, float(r.max()))
            sat_hits += int(np.sum(r > 1.0))
            sat_slots += len(r)
            if not released:
                peak_q_err = max(peak_q_err, float(np.max(np.abs(e))))
                peak_qd_err = max(peak_qd_err, float(np.max(np.abs(ed))))

            arm.step(q_t, qd_t, qdd_t)

            if not released and step >= release_step:
                ee_pos, ee_vel, _, ee_ang = arm.ee_state()
                q_err_at_release = e.copy()
                release_pos = ee_pos.copy()
                omega_release = np.array(ee_ang)
                v_release = arm.release_ball(
                    self.ball_id, dynamic=True, keep_collision_disabled=True
                )
                released = True
                released_at = step
                p.stepSimulation(physicsClientId=self.client)
                continue

            if released:
                pos, _ = p.getBasePositionAndOrientation(
                    self.ball_id, physicsClientId=self.client
                )
                vel, _ = p.getBaseVelocity(self.ball_id, physicsClientId=self.client)
                pos = np.array(pos)
                vel = np.array(vel)
                if traj is not None:
                    traj.append(np.concatenate([pos, vel]))
                a_total = _ball_accel(pos, vel, self.ball_mass, self.ball_radius,
                                      np.zeros(3))
                f_drag = self.ball_mass * (a_total - _GRAVITY)
                p.applyExternalForce(
                    self.ball_id, -1, f_drag.tolist(), [0, 0, 0], p.WORLD_FRAME,
                    physicsClientId=self.client,
                )
                h = self.target_height
                if (
                    pos[2] <= h + self.ball_radius + 0.005
                    and vel[2] < 0.0
                    and step > released_at + 2
                    and prev_pos is not None
                    and prev_pos[2] > h
                ):
                    frac = (prev_pos[2] - h) / max(prev_pos[2] - pos[2], 1e-12)
                    land = prev_pos + frac * (pos - prev_pos)
                    land_xy = land[:2]
                    flight_time = (step - released_at - 1 + frac) * dt
                    break
                prev_pos = pos.copy()

            p.stepSimulation(physicsClientId=self.client)

        return ThrowResult(
            label=label or getattr(plan, "meta", {}).get("label", "?"),
            v_planned=v_planned,
            v_release=np.asarray(v_release, dtype=float),
            omega_release=omega_release,
            release_pos=release_pos,
            release_pos_planned=release_pos_planned,
            land_xy=land_xy,
            flight_time=flight_time,
            t_r=float(plan.t_r),
            dt_throw=float(plan.t_r - plan.t_w),
            peak_q_err=peak_q_err,
            peak_qd_err=peak_qd_err,
            q_err_at_release=q_err_at_release,
            tau_sat_frac=(sat_hits / max(sat_slots, 1)),
            peak_tau_ratio=peak_tau_ratio,
            peak_jerk=peak_jerk,
            accel_step=self.accel_step(plan),
            released_step=released_at,
            extra={"traj": np.array(traj) if traj else None,
                   "release_bias_steps": int(release_bias_steps)},
        )


class BaselinePlanAdapter:
    """Wraps `ArmController.plan_throw`'s coeff dict in the `LabPlan` interface.

    Lets the shipped planner be run through the identical harness, so the
    comparison is between TRAJECTORIES, not between two different executors.
    Jerk is reported as the analytic value of the piecewise cubic (`6*a3`,
    constant within a segment); the step discontinuities at the phase joins are
    genuinely unbounded and are reported separately by `bench.py`.
    """

    def __init__(self, arm, coeffs, q_release, qd_release, label="baseline_cubic"):
        self.arm = arm
        self.coeffs = coeffs
        self.q_release = np.asarray(q_release, dtype=float)
        self.qd_release = np.asarray(qd_release, dtype=float)
        self.t_w = float(coeffs["t_w"])
        self.t_r = float(coeffs["t_r"])
        self.T = float(coeffs["T"])
        self.join_times = (0.0, self.t_w, self.t_r)
        self.meta = {"label": label}
        self.checks = {}

    def eval(self, t):
        q, qd, qdd = self.arm.get_setpoint(self.coeffs, t, with_accel=True)
        seg = (
            "windup" if t <= self.t_w else ("throw" if t <= self.t_r else "follow")
        )
        qddd = 6.0 * self.coeffs[seg][:, 3]
        return q, qd, qdd, qddd
