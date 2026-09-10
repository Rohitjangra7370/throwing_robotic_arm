"""
The throw must record the joint VELOCITY it actually reached at release.

WHY THIS EXISTS
---------------
`rehearse_or_throw`'s drift `track` recorded `(s, q_planned, q_measured)` and
nothing else. `read_joint_state()` returns `(q, qd)` off the same 1 kHz UDP
frame the position came from -- the velocity was read and then dropped on the
floor, every tick, of every throw ever executed on this arm.

The consequence is that the single most load-bearing sim-to-real claim in this
project -- "the arm releases the ball at the speed the policy asked for" -- had
never been measured, and could not have been from the logs. Open-loop velocity
streaming has no feedback term, so a shortfall would not correct itself and
nothing downstream would report it: `last_exec_stats` carried POSITION drift
only, and position drift at release (0.017-0.032 rad, ~1-2 cm of landing error)
is a much smaller quantity than a release-speed error would be.

`measured_release_velocity` turns the recorded joint velocity into the TCP
velocity through the same Jacobian-at-the-tool-offset the planner uses, so
planned and measured are the same kind of number and can be subtracted.

The round-trip test is the important one: feeding the PLANNED joint state
through the measurement path must reproduce the planner's own reported
`v_ach` exactly. Without that, a deviation measured on real data could just as
easily be a bug in the measurement as a real sim-to-real gap.
"""
import numpy as np
import pybullet as p
import pytest

from robot_arm.kinova_hardware import HardwareThrowExecutor, SafetyLimits
from robot_arm.robot_profiles import get_robot_profile
from robot_arm.tcp_velocity import measured_release_velocity, tcp_velocity

import run_hardware_throw as H

ROBOT = "kinova_gen3_dyn"
TOOL_Z = 0.12


class _FakeArm:
    """Minimal stand-in for ArmController inside the streaming loop."""

    def __init__(self, n=7):
        self.n = n

    def get_setpoint(self, coeffs, s, with_accel=False):
        q = np.full(self.n, 0.1 * s)
        qd = np.full(self.n, 0.2)
        return (q, qd, np.zeros(self.n)) if with_accel else (q, qd)


def _limits(profile):
    return SafetyLimits(
        qd_max=np.array(profile.qd_max, float),
        q_soft_lo=-6.10 * np.ones(len(profile.qd_max)),
        q_soft_hi=6.10 * np.ones(len(profile.qd_max)),
        speed_scale=1.0,
    )


def test_track_records_velocity_alongside_position():
    profile = get_robot_profile(ROBOT)
    ex = HardwareThrowExecutor(_limits(profile), dry_run=True)
    coeffs = {"t_w": 0.02, "t_r": 0.04, "T": 0.06}
    track = []
    ex.rehearse_or_throw(coeffs, _FakeArm(), verbose=False, track=track)
    assert track, "nothing tracked"
    for row in track:
        assert len(row) == 8, (
            "track rows must carry (s, q_planned, q_meas, qd_planned, qd_meas, "
            "wall, gripper_pct, gripper_vel)"
        )
    s, q_pl, q_me, qd_pl, qd_me, wall, gp, gv = track[0]
    assert np.shape(qd_pl) == np.shape(q_pl)
    assert np.shape(qd_me) == np.shape(q_me)
    assert np.all(np.asarray(qd_pl) == 0.2)      # from _FakeArm
    assert np.isfinite(wall)
    # Gripper position rides on the SAME feedback frame as the joint state.
    # Without it there is no way to tell whether the fingers actually finished
    # opening during a throw -- they can freeze part-open the moment joint-speed
    # streaming resumes after GRIPPER_RELEASE_PAUSE_S, and every throw before
    # 2026-09-09 was blind to that.
    assert np.isfinite(gp) and np.isfinite(gv)


def test_gripper_column_records_the_release():
    """The tracked gripper position must actually change when it is opened."""
    profile = get_robot_profile(ROBOT)
    ex = HardwareThrowExecutor(_limits(profile), dry_run=True)
    ex.backend.gripper_pos = 1.0                 # start closed
    coeffs = {"t_w": 0.02, "t_r": 0.04, "T": 0.10}
    track = []
    ex.rehearse_or_throw(coeffs, _FakeArm(), verbose=False, track=track)
    grip = np.array([r[6] for r in track])
    assert grip[0] > 90.0, "should start closed"
    assert grip[-1] < 10.0, "gripper OPEN at release must show up in the trace"


def test_exec_stats_carries_measured_release_velocity_error():
    profile = get_robot_profile(ROBOT)
    ex = HardwareThrowExecutor(_limits(profile), dry_run=True)
    coeffs = {"t_w": 0.02, "t_r": 0.04, "T": 0.06}
    track = []
    ex.rehearse_or_throw(coeffs, _FakeArm(), verbose=False, track=track)
    st = ex.last_exec_stats
    for k in ("qd_err_at_release_rad_s", "qd_max_frac_at_release"):
        assert k in st, f"{k} missing from last_exec_stats"
    # dry-run feedback is all zeros against a planned 0.2 rad/s
    assert st["qd_err_at_release_rad_s"] == pytest.approx(0.2, abs=1e-9)


def test_tcp_velocity_round_trips_the_planner():
    """
    Planned joint state through the measurement path == the planner's own v_ach.
    If this drifts, every measured-vs-planned number is uninterpretable.
    """
    arm, profile, cid = H.build_arm(ROBOT)
    try:
        q = np.array([-3.106, -0.436, 0.0, 1.012, 0.0, -0.279, 0.0])
        qd = np.array([0.0, -0.682, 0.0, -0.96, 0.0, -0.84, 0.0])
        v_cmd = np.array([1.374, -0.049, 0.368])
        _, q_rel, qd_rel, v_ach = arm.plan_throw(
            v_cmd, np.array([0.0, 0.0, 1.14]), t_w=0.5, t_r=1.6, T=3.2,
            q_release_override=q, qd_release_override=qd,
            monotonic_windup=True, tool_offset=[0.0, 0.0, TOOL_Z])
        v_meas = tcp_velocity(arm, profile, q_rel, qd_rel, [0.0, 0.0, TOOL_Z])
        assert np.allclose(v_meas, v_ach, rtol=1e-12, atol=1e-12)
    finally:
        p.disconnect(cid)


def test_tcp_velocity_zero_offset_is_the_flange():
    arm, profile, cid = H.build_arm(ROBOT)
    try:
        q = np.array([-3.106, -0.436, 0.0, 1.012, 0.0, -0.279, 0.0])
        qd = np.array([0.0, -0.682, 0.0, -0.96, 0.0, -0.84, 0.0])
        vf = tcp_velocity(arm, profile, q, qd, [0.0, 0.0, 0.0])
        vt = tcp_velocity(arm, profile, q, qd, [0.0, 0.0, TOOL_Z])
        assert np.linalg.norm(vf) == pytest.approx(1.1262, abs=1e-3)
        assert np.linalg.norm(vt) == pytest.approx(1.4238, abs=1e-3)
    finally:
        p.disconnect(cid)


def test_measured_release_velocity_picks_the_release_sample():
    """It must read the sample nearest trajectory time t_r, not the last one."""
    arm, profile, cid = H.build_arm(ROBOT)
    try:
        q = np.array([-3.106, -0.436, 0.0, 1.012, 0.0, -0.279, 0.0])
        qd = np.array([0.0, -0.682, 0.0, -0.96, 0.0, -0.84, 0.0])
        track = [
            (0.0, q, q, qd, np.zeros(7), 0.0),          # far from t_r
            (1.6, q, q, qd, qd, 1.0),                   # the release sample
            (3.2, q, q, qd, np.zeros(7), 2.0),          # after release
        ]
        out = measured_release_velocity(arm, profile, track, t_r=1.6,
                                        tool_offset=[0.0, 0.0, TOOL_Z])
        assert out["s_used"] == pytest.approx(1.6)
        assert np.linalg.norm(out["v_measured"]) == pytest.approx(1.4238, abs=1e-3)
        assert np.linalg.norm(out["v_planned"]) == pytest.approx(1.4238, abs=1e-3)
        assert out["speed_ratio"] == pytest.approx(1.0, abs=1e-9)
    finally:
        p.disconnect(cid)


def test_measured_release_velocity_reports_a_shortfall():
    arm, profile, cid = H.build_arm(ROBOT)
    try:
        q = np.array([-3.106, -0.436, 0.0, 1.012, 0.0, -0.279, 0.0])
        qd = np.array([0.0, -0.682, 0.0, -0.96, 0.0, -0.84, 0.0])
        track = [(1.6, q, q, qd, 0.8 * qd, 1.0)]       # arm 20% slow
        out = measured_release_velocity(arm, profile, track, t_r=1.6,
                                        tool_offset=[0.0, 0.0, TOOL_Z])
        assert out["speed_ratio"] == pytest.approx(0.8, rel=1e-9)
    finally:
        p.disconnect(cid)


def test_measured_release_velocity_refuses_an_empty_track():
    arm, profile, cid = H.build_arm(ROBOT)
    try:
        assert measured_release_velocity(arm, profile, [], t_r=1.6,
                                         tool_offset=[0, 0, TOOL_Z]) is None
        # a legacy 3-tuple track carries no velocity and must be refused, not
        # silently treated as zero velocity (which would read as a total stall)
        assert measured_release_velocity(
            arm, profile, [(1.6, np.zeros(7), np.zeros(7))], t_r=1.6,
            tool_offset=[0, 0, TOOL_Z]) is None
    finally:
        p.disconnect(cid)


# --------------------------------------------------------------------------- #
# measure_gripper_latency.py --loaded refusal band.
#
# The first version of this guard was `pos < GRASP_THRESHOLD_PCT -> refuse`,
# which is wrong at the open end: a fully open 2F-85 reads ~0.87%, so it
# refused the ONE state it is actually safe to close from and would have made
# --loaded unusable. The hazardous state is stalled PART-WAY.
# --------------------------------------------------------------------------- #
def test_holding_something_only_flags_a_partway_stall():
    from measure_gripper_latency import holding_something

    assert not holding_something(0.87)    # fully open, safe to close
    assert not holding_something(0.0)
    assert not holding_something(4.9)     # still open enough
    assert holding_something(61.4)        # measured: this tennis ball
    assert holding_something(58.08)       # measured: the previous ball
    assert not holding_something(99.13)   # closed on nothing, no object
    assert not holding_something(100.0)


def test_release_pause_is_long_enough_to_free_the_ball():
    """
    The pause is the ONLY window in which the gripper can move -- joint-speed
    streaming blocks SendGripperCommand outright, measured 2026-09-09 as a hard
    freeze (max |gripper velocity| = 0.00 for 3.4 s after the stream resumed).

    So the pause must cover: onset + the finger travel needed to free the ball.
    Numbers are measured on this rig at speed_scale 1.0.
    """
    from robot_arm.kinova_hardware import GRIPPER_RELEASE_PAUSE_S

    ONSET_S = 0.025          # fingers first move, measured from pause start
    RATE_PCT_PER_S = 137.0   # open rate during the pause
    GRASP_PCT = 32.89        # tennis ball, encompassing grip
    ESCAPE_PCT = 8.33        # ball caged until at least here
    needed = ONSET_S + (GRASP_PCT - ESCAPE_PCT) / RATE_PCT_PER_S
    assert needed == pytest.approx(0.204, abs=0.005)
    assert GRIPPER_RELEASE_PAUSE_S > needed, (
        f"pause {GRIPPER_RELEASE_PAUSE_S}s does not even reach the ball's "
        f"escape point at {needed:.3f}s -- this is the 4 ms margin that made "
        f"the ball dribble off the gripper body on 2026-09-09"
    )
    # and with real margin, not another 4 ms
    assert GRIPPER_RELEASE_PAUSE_S >= needed + 0.10
