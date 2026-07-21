import numpy as np
from find_throw_pose import aimed_speed, windup_within_limits

QD = np.array([1.3963, 1.3963, 1.3963, 1.3963, 1.2218, 1.2218, 1.2218])


def test_aimed_speed_freezes_base_and_respects_limits():
    # base joint (col 0) drives +y; joint 4 (col 3) drives +x. Throw dir is +x.
    J = np.zeros((3, 7))
    J[0, 3] = 1.0   # qd[3] -> +x EE velocity
    J[1, 0] = 2.0   # qd[0] (base) -> +y EE velocity (a pure sideways source)
    d = np.array([1.0, 0.0, 0.0])
    s, qd = aimed_speed(J, d, QD, freeze_base=True)
    assert qd is not None
    assert abs(qd[0]) < 1e-9                      # base pinned to zero
    assert np.all(np.abs(qd) <= QD + 1e-9)        # every joint within qd_max
    assert abs(s - 1.3963) < 1e-3                 # s == qd[3] cap, base unused


def test_windup_within_limits_flags_violation():
    q_rel = np.array([0.0, 1.0, 0.0, -0.5, 0.0, 0.2, 0.0])
    qd_rel = QD.copy()
    tight_lo, tight_hi = -np.full(7, 0.6), np.full(7, 0.6)
    assert windup_within_limits(q_rel, qd_rel, 1.1, tight_lo, tight_hi) is False
    wide_lo, wide_hi = -np.full(7, 10.0), np.full(7, 10.0)
    assert windup_within_limits(q_rel, qd_rel, 1.1, wide_lo, wide_hi) is True
