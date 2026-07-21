import numpy as np
import pybullet as p
import pybullet_data
from find_throw_pose import (aimed_speed, windup_within_limits, is_natural_posture,
                             windup_path_feasible, static_feasible, QN)

QD = np.array([1.3963, 1.3963, 1.3963, 1.3963, 1.2218, 1.2218, 1.2218])
ROLL_IDX = (0, 2, 4, 6)   # base, shoulder-roll, wrist-roll1, wrist-roll2
PITCH_IDX = (1, 3, 5)     # shoulder, elbow, wrist -- perpendicular to the swing plane


def test_aimed_speed_freezes_all_roll_joints_and_respects_limits():
    # roll joints (0,2,4,6) each drive +y (a twist/sideways source); pitch
    # joint 3 drives +x. Throw dir is +x -- only the pitch joint should be used.
    J = np.zeros((3, 7))
    J[0, 3] = 1.0                          # pitch joint (idx3, elbow) -> +x
    for i in ROLL_IDX:
        J[1, i] = 2.0                      # every roll joint -> +y (sideways)
    d = np.array([1.0, 0.0, 0.0])
    s, qd = aimed_speed(J, d, QD, freeze_roll=True)
    assert qd is not None
    for i in ROLL_IDX:
        assert abs(qd[i]) < 1e-9            # every roll/twist joint pinned to zero
    assert np.all(np.abs(qd) <= QD + 1e-9)  # every joint within qd_max
    assert abs(s - 1.3963) < 1e-3           # s == pitch joint's own cap


def test_windup_within_limits_flags_violation():
    q_rel = np.array([0.0, 1.0, 0.0, -0.5, 0.0, 0.2, 0.0])
    qd_rel = QD.copy()
    tight_lo, tight_hi = -np.full(7, 0.6), np.full(7, 0.6)
    assert windup_within_limits(q_rel, qd_rel, 1.1, tight_lo, tight_hi) is False
    wide_lo, wide_hi = -np.full(7, 10.0), np.full(7, 10.0)
    assert windup_within_limits(q_rel, qd_rel, 1.1, wide_lo, wide_hi) is True


def test_is_natural_posture_requires_forward_reach_and_caps_elevation():
    az = 0.0
    assert is_natural_posture(np.array([0.6, 0.0, 0.4]), az)        # forward, ~34deg elev
    assert not is_natural_posture(np.array([0.05, 0.0, 1.1]), az)   # ~stacked at base
    assert not is_natural_posture(np.array([0.2, 0.0, 1.0]), az)    # flagpole, ~79deg elev


def test_search_builds_azimuth_table_with_base_at_azimuth_and_zero_roll_velocity():
    from find_throw_pose import search
    table = search(azimuth_deg_grid=[0.0])
    assert len(table) == 1
    for e in table:
        # base joint angle equals the entry azimuth; every roll joint velocity is zero
        assert abs(np.degrees(e["q"][0]) - e["azimuth_deg"]) < 1e-6
        for i in ROLL_IDX:
            assert abs(e["qd"][i]) < 1e-9
        # every release joint velocity within qd_max, and a real throw
        assert np.all(np.abs(e["qd"]) <= QD + 1e-9)
        # windup-path feasibility narrows the candidate set (real throws are
        # smaller than a naive release-only-feasible search would find), but
        # every remaining candidate must still be a genuine, non-trivial throw
        assert e["speed"] > 0.1


def test_windup_path_feasible_rejects_mid_swing_gravity_violation():
    """A release pose can be statically feasible while the neutral->windup
    swing still passes through a worse intermediate gravity configuration --
    verified: shoulder sweeping neutral ~22deg -> cocked 90deg peaks ~39Nm
    mid-swing even though both endpoints are individually fine. This must be
    rejected, not just checked at the two endpoints."""
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    arm = p.loadURDF(pybullet_data.getDataPath() + "/kinova_gen3/gen3.urdf", useFixedBase=True)
    # release pose with shoulder far from neutral (idx1: 90deg vs QN's ~22deg)
    q_release = np.deg2rad([0.0, 90.0, 15.0, -100.0, 0.0, 90.0, 0.0])
    qd_release = np.array([0.0, 0.0, 0.0, 1.0965, 0.0, -0.7335, 0.0])
    assert static_feasible(arm, q_release)          # release endpoint alone looks fine
    assert not windup_path_feasible(arm, q_release, qd_release, t_throw=0.9)
    p.disconnect(cid)


def test_windup_path_feasible_accepts_small_swing():
    """A release pose close to neutral should pass -- sanity check that the
    filter isn't universally rejecting everything."""
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    arm = p.loadURDF(pybullet_data.getDataPath() + "/kinova_gen3/gen3.urdf", useFixedBase=True)
    q_release = QN.copy()
    qd_release = np.zeros(7)
    assert windup_path_feasible(arm, q_release, qd_release, t_throw=0.9)
    p.disconnect(cid)
