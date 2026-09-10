"""
Turn a recorded joint state into the TCP velocity, the same way the planner does.

This is the read-side counterpart to `ArmController.plan_throw`'s `tool_offset`.
Both must take the Jacobian at the SAME point, or "planned vs measured" is
comparing a tool-centre velocity against a flange velocity and the 1.26x
difference reads as a sim-to-real gap that isn't there (that is exactly the bug
tests/test_release_speed_report.py exists to prevent, in the other direction).

Never hand-roll `v = v_flange + omega x r` here. PyBullet's own
`calculateJacobian(..., localPosition=tool_offset, ...)` is the same call
`OptimizedReleaseSolver` and `find_throw_pose.py` already use; a second
derivation of the same geometry is how this repo has historically grown silent
disagreements between the planner and everything else.
"""
import numpy as np
import pybullet as p


def _jacobian_columns(arm, profile):
    """Map profile joint ids onto calculateJacobian's movable-joint ordering."""
    n = p.getNumJoints(arm._arm_id, physicsClientId=arm._cid)
    movable = [j for j in range(n)
               if p.getJointInfo(arm._arm_id, j,
                                 physicsClientId=arm._cid)[2] != p.JOINT_FIXED]
    return movable, [movable.index(j) for j in profile.joint_ids]


def tcp_velocity(arm, profile, q, qd, tool_offset=None):
    """
    Linear velocity (3,) of `ee_link` origin + `tool_offset` at joint state
    (q, qd). `tool_offset` is in ee_link's LOCAL frame; None/zeros = the bare
    flange, which is what every pre-2026-09-09 number in this repo means.
    """
    q = np.asarray(q, float)
    qd = np.asarray(qd, float)
    lp = ([0.0, 0.0, 0.0] if tool_offset is None
          else [float(v) for v in np.asarray(tool_offset, float).ravel()])
    movable, cols = _jacobian_columns(arm, profile)
    q_full = [0.0] * len(movable)
    for local_i, jid in enumerate(profile.joint_ids):
        q_full[movable.index(jid)] = float(q[local_i])
    j_lin, _ = p.calculateJacobian(
        arm._arm_id, profile.ee_link, lp, q_full,
        [0.0] * len(movable), [0.0] * len(movable), physicsClientId=arm._cid)
    return np.array(j_lin)[:, cols] @ qd[: len(cols)]


def measured_release_velocity(arm, profile, track, t_r, tool_offset=None):
    """
    Planned vs actually-measured TCP velocity at release, from a drift track.

    `track` rows are `(s, q_planned, q_measured, qd_planned, qd_measured, wall)`
    as recorded by `HardwareThrowExecutor.rehearse_or_throw`. A legacy 3-element
    row carries no velocity at all and is REFUSED (returns None) rather than read
    as zero velocity -- a missing field must never be able to masquerade as a
    measurement of a stalled arm.

    The sample nearest trajectory time `t_r` is used, not the last one: the
    trajectory continues through follow-through, where the arm is decelerating.

    Returns None if there is nothing usable, else a dict with `v_planned`,
    `v_measured`, both speeds, `speed_ratio` (measured/planned), the angle
    between them, and which sample was used.
    """
    rows = [r for r in track if len(r) >= 6]
    if not rows:
        return None
    s_all = np.array([float(r[0]) for r in rows])
    i = int(np.argmin(np.abs(s_all - float(t_r))))
    s, _q_pl, q_me, qd_pl, qd_me, wall = rows[i][:6]

    # Planned velocity is evaluated at the PLANNED joint angles and the measured
    # one at the MEASURED angles: position drift moves the Jacobian too, and
    # attributing that to the velocity command would hide it.
    v_pl = tcp_velocity(arm, profile, rows[i][1], qd_pl, tool_offset)
    v_me = tcp_velocity(arm, profile, q_me, qd_me, tool_offset)
    n_pl = float(np.linalg.norm(v_pl))
    n_me = float(np.linalg.norm(v_me))
    if n_pl > 1e-9 and n_me > 1e-9:
        cosang = float(np.clip(np.dot(v_pl, v_me) / (n_pl * n_me), -1.0, 1.0))
        ang = float(np.degrees(np.arccos(cosang)))
    else:
        ang = float("nan")
    return {
        "s_used": float(s),
        "wall_used": float(wall),
        "v_planned": v_pl,
        "v_measured": v_me,
        "speed_planned": n_pl,
        "speed_measured": n_me,
        "speed_ratio": (n_me / n_pl) if n_pl > 1e-9 else float("nan"),
        "direction_error_deg": ang,
        "qd_planned": np.asarray(qd_pl, float),
        "qd_measured": np.asarray(qd_me, float),
    }
