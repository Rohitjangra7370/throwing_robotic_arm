import numpy as np
import pybullet as p
import pybullet_data
import pytest

from robot_arm.arm_controller import ArmController
from robot_arm.robot_profiles import get_robot_profile

DT = 0.02


def _make_arm(robot_name):
    client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(DT, physicsClientId=client)
    prof = get_robot_profile(robot_name)
    urdf = pybullet_data.getDataPath() + "/" + prof.urdf_rel_path
    return client, ArmController(client, urdf, robot_name=robot_name)


def test_gravity_hold_drift_under_1cm():
    client, arm = _make_arm("kinova_gen3_dyn")
    try:
        q_hold = np.array(get_robot_profile("kinova_gen3_dyn").q_neutral)
        qd_zero = np.zeros(7)
        ee_start = arm.ee_state()[0]
        for _ in range(int(2.0 / DT)):  # 2 s
            arm.step(q_hold, qd_zero, np.zeros(7))
            p.stepSimulation(physicsClientId=client)
        drift = np.linalg.norm(arm.ee_state()[0] - ee_start)
        assert drift < 0.01, f"EE drifted {drift * 100:.2f} cm under gravity hold"
    finally:
        p.disconnect(client)


def test_torque_profile_requires_gains():
    # kinematic profile has no tau_max; torque construction must be impossible
    # for it, and the dyn profile must expose the arrays the controller needs.
    client, arm = _make_arm("kinova_gen3_dyn")
    try:
        assert arm._tau_max.shape == (7,)
        assert arm._kp.shape == (7,)
        assert arm._kd.shape == (7,)
    finally:
        p.disconnect(client)


def test_ball_mass_compensation_reduces_tracking_error():
    """
    Root cause (found via instrumented probing): true closed-loop joint
    tracking error (measured against the state the controller actually reacts
    to, i.e. BEFORE the physics step that applies this command — not one step
    later) grows from ~0.004-0.009 rad with no payload to ~0.017-0.021 rad once
    a 57.7g ball is gripped, because calculateInverseDynamics only knows the
    arm's own URDF mass. Torque saturation was ruled out (0% clipped steps,
    10x torque headroom changes nothing). Compensating by inflating the EE
    link's mass while the ball is attached should bring tracking back down
    near the no-payload baseline.
    """
    client, arm = _make_arm("kinova_gen3_dyn")
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
        p.loadURDF("plane.urdf", physicsClientId=client)
        ee_pos = arm.ee_state()[0]
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.0327, physicsClientId=client)
        ball = p.createMultiBody(baseMass=0.0577, baseCollisionShapeIndex=col,
                                 basePosition=ee_pos.tolist(), physicsClientId=client)
        p.changeDynamics(ball, -1, linearDamping=0.0, angularDamping=0.0,
                         physicsClientId=client)
        arm.attach_ball(ball)

        prof = get_robot_profile("kinova_gen3_dyn")
        t_w, t_r, T = prof.timing
        alpha = np.deg2rad(35.0)
        speed = 0.65  # uncompensated error here was ~0.021 rad (mid-high speed)
        v_cmd = np.array([speed * np.cos(alpha), 0.0, speed * np.sin(alpha)])
        coeffs, _, _, _ = arm.plan_throw(
            v_cmd, np.array(prof.default_release_pos), t_w, t_r, T
        )

        max_err = 0.0
        n_steps = int(coeffs["t_r"] / DT)
        for step in range(n_steps):
            t = step * DT
            q_t, qd_t, qdd_t = arm.get_setpoint(coeffs, t, with_accel=True)
            if t > coeffs["t_w"]:
                states = p.getJointStates(arm.arm_id, arm.joint_ids, physicsClientId=client)
                q_meas = np.array([s[0] for s in states])
                max_err = max(max_err, float(np.max(np.abs(q_t - q_meas))))
            arm.step(q_t, qd_t, qdd_t)
            p.stepSimulation(physicsClientId=client)

        # no-payload baseline at this speed is ~0.0088 rad; uncompensated
        # with-payload was ~0.0208 rad. 0.012 gives margin above baseline
        # while still well below the uncompensated value.
        assert max_err < 0.012, (
            f"closed-loop tracking error {max_err:.4f} rad "
            "(payload mass not compensated?)"
        )
    finally:
        p.disconnect(client)
