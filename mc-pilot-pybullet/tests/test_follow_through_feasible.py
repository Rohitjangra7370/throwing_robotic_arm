import os

import numpy as np
import pybullet as p
import pybullet_data
import pytest
from robot_arm.arm_controller import ArmController, _eval_cubic
from robot_arm.robot_profiles import get_robot_profile


@pytest.fixture
def arm():
    client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    prof = get_robot_profile("kinova_gen3_dyn")
    urdf = pybullet_data.getDataPath() + "/" + prof.urdf_rel_path
    a = ArmController(client, urdf, robot_name="kinova_gen3_dyn")
    yield a, prof, client
    p.disconnect(client)


def test_follow_through_infeasible_release_raises_not_silently_collapses(arm):
    """windup and throw both have real time-scaling+feasibility loops that
    stretch duration (and raise if still infeasible after 6 tries) -- follow
    (the post-release decel back to neutral) had NEITHER: follow_dur was
    fixed at the ORIGINAL nominal (T - t_r), never adjusted even when windup/
    throw get stretched a lot. For this extended overhead release (far from
    q_neutral, released at near-qd_max), the follow phase peaked at 556% of
    qd_max and 3.2x tau_max at the original duration -- a trajectory the real
    arm cannot execute (looks like a violent "collapse" in sim; would fault
    or be clamped on real hardware). Measured directly: even the BEST
    duration (searched dt=0.6..6.0s) only gets this candidate down to ~1.17x
    tau_max -- peak ratio vs duration is NOT monotonic here (bottoms out
    then rises again), so this is a structural infeasibility of this
    specific release state's straight-line recovery path, not a
    time-scaling problem. Correct behavior: raise loudly, not produce an
    infeasible trajectory silently."""
    controller, prof, client = arm
    q_release = np.array([-3.14159169, -0.52359878, 0.0, 0.66322512, 0.0, 0.41887902, 0.0])
    qd_release = np.array([0.0, -1.3963, 0.0, -1.3963, 0.0, -1.03565721, 0.0]) * (1.85 / 1.929)
    with pytest.raises(RuntimeError, match="Follow-through infeasible"):
        controller.plan_throw(
            np.array([1.0, 0.0, 0.0]), np.array(prof.default_release_pos), 0.5, 1.6, 2.2,
            q_release_override=q_release, qd_release_override=qd_release,
            monotonic_windup=True,
        )


def test_shipped_table_entry_whole_trajectory_within_velocity_and_torque(arm):
    """End-to-end guarantee on what actually ships: for the real az=0 entry of
    throw_pose_table.npy at a real commanded speed, EVERY phase (windup,
    throw, follow) must stay within BOTH qd_max and tau_max. Velocity is the
    part that was missed: windup/throw bound it by construction, follow does
    not, and a torque-only follow check accepted a duration whose peak joint
    velocity was 1.88x qd_max (torque was fine at 0.95x) -- caught only when
    the envelope was plotted for a report."""
    controller, prof, client = arm
    table_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "throw_pose_table.npy")
    table = list(np.load(table_path, allow_pickle=True))
    e0 = min(table, key=lambda e: abs(e["azimuth_deg"]))
    q_release = np.array(e0["q"])
    qd_release = np.array(e0["qd"]) * (1.49 / float(e0["speed"]))
    coeffs, _, _, _ = controller.plan_throw(
        np.array([1.0, 0.0, 0.0]), np.array(prof.default_release_pos), 0.5, 1.6, 2.2,
        q_release_override=q_release, qd_release_override=qd_release,
        monotonic_windup=True,
    )
    qd_max = np.array(prof.qd_max)
    tau_max = np.array(prof.tau_max)
    for t in np.linspace(0.0, coeffs["T"], 300):
        q, qd, qdd = controller.get_setpoint(coeffs, t, with_accel=True)
        assert np.all(np.abs(qd) <= qd_max + 1e-6), (
            f"joint velocity exceeds qd_max at t={t:.3f} "
            f"({np.max(np.abs(qd) / qd_max):.2f}x)")
        tau = np.array(p.calculateInverseDynamics(
            controller._arm_id, list(q), list(qd), list(qdd), physicsClientId=client))
        assert np.all(np.abs(tau) <= tau_max + 1e-6), (
            f"joint torque exceeds tau_max at t={t:.3f} "
            f"({np.max(np.abs(tau) / tau_max):.2f}x)")


def test_rollout_keeps_commanding_arm_after_release():
    """_simulate_pybullet's post-release branch handled ONLY the ball -- the
    arm was never stepped again, so in torque mode its joints got zero
    command and gravity free-fell the whole arm (the 'collapse' visible in
    every rendered video; a real arm would do the same if its controller
    stopped). The torque-validated follow-through existed in the plan but
    was never executed. Regression: during a real rollout, every arm joint
    velocity after release must stay near the tracked-trajectory envelope
    (free-fall blows far past qd_max within a few hundred ms)."""
    import pybullet_data
    from simulation_class.model_pybullet import PyBulletThrowingSystem

    table_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "throw_pose_table.npy")
    table = list(np.load(table_path, allow_pickle=True))

    class _NoWind:
        def reset(self): pass
        def __call__(self, t): return np.zeros(3)

    class _ConstPolicy:
        def __call__(self, s0, t): return np.array([1.4])

    sysm = PyBulletThrowingSystem(robot_name="kinova_gen3_dyn",
                                  opt_posture_table=table,
                                  opt_launch_deg=float(table[0]["elev_deg"]),
                                  wind_model=_NoWind(), t_w=0.5, t_r=1.6)
    qd_worst = {"v": 0.0}

    def hook(cid):
        for b in range(p.getNumBodies(physicsClientId=cid)):
            bid = p.getBodyUniqueId(b, physicsClientId=cid)
            if p.getNumJoints(bid, physicsClientId=cid) >= 7:
                js = p.getJointStates(bid, list(range(7)), physicsClientId=cid)
                qd = np.max(np.abs([s[1] for s in js]))
                qd_worst["v"] = max(qd_worst["v"], float(qd))

    sysm.frame_hook = hook
    s0 = np.concatenate([[0.0, 0.0, 1.1], np.zeros(3), [0.68, 0.0]])
    sysm.rollout(s0, _ConstPolicy(), T=0.6, dt=0.02, noise=0.0)
    # tracked joints stay near qd_max (1.4 rad/s); free-fall blows far past it
    assert qd_worst["v"] < 2.5, (
        f"arm joint velocity hit {qd_worst['v']:.1f} rad/s during rollout -- "
        "consistent with post-release free-fall, not tracked follow-through")


def test_follow_through_feasible_release_stays_within_limits(arm):
    """Sanity check the mechanism doesn't break/over-reject the normal case:
    a release state close to neutral (small displacement, moderate release
    velocity -- representative of the earlier, accurate narrow-grid throws
    this session validated) must plan successfully with the whole follow
    phase within qd_max and tau_max."""
    controller, prof, client = arm
    q_neutral = np.array(prof.q_neutral)
    q_release = q_neutral + np.array([0.0, 0.15, 0.0, -0.2, 0.0, 0.1, 0.0])
    qd_release = np.array([0.0, 0.3, 0.0, -0.25, 0.0, 0.2, 0.0])
    coeffs, _, _, _ = controller.plan_throw(
        np.array([0.5, 0.0, 0.2]), np.array(prof.default_release_pos), 0.5, 1.6, 2.2,
        q_release_override=q_release, qd_release_override=qd_release,
        monotonic_windup=True,
    )
    follow_dur = coeffs["T"] - coeffs["t_r"]
    qd_max = np.array(prof.qd_max)
    tau_max = np.array(prof.tau_max)
    for t in np.linspace(0.0, follow_dur, 40):
        q, qd, qdd = _eval_cubic(coeffs["follow"], t, with_accel=True)
        assert np.all(np.abs(qd) <= qd_max + 1e-6), f"follow exceeds qd_max at t={t}"
        tau = np.array(p.calculateInverseDynamics(
            controller._arm_id, list(q), list(qd), list(qdd), physicsClientId=client))
        assert np.all(np.abs(tau) <= tau_max + 1e-6), f"follow exceeds tau_max at t={t}"
