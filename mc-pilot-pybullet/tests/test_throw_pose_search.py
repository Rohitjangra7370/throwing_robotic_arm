import numpy as np
import pybullet as p
import pybullet_data
from find_throw_pose import (aimed_speed, windup_within_limits, is_natural_posture,
                             windup_path_feasible, release_dynamics_feasible,
                             throw_ramp_feasible, follow_through_feasible,
                             static_feasible, QN, TAU_MAX)

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
    import pybullet as p
    import pybullet_data
    from find_throw_pose import search, _fkj

    table = search(azimuth_deg_grid=[0.0])
    assert len(table) == 1

    # A stale/corrupted table entry can carry a qd that satisfies the LP's own
    # bookkeeping but no longer projects along its OWN labeled aim direction
    # once re-checked against a fresh Jacobian for its saved q (this exact
    # failure mode shipped once: an az=0/elev=0 entry whose qd pointed 71%
    # sideways). Recompute J independently from the saved q/azimuth and verify
    # J @ qd is still aligned with the entry's own labeled direction.
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    arm = p.loadURDF(pybullet_data.getDataPath() + "/kinova_gen3/gen3.urdf", useFixedBase=True)

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

        az = np.deg2rad(e["azimuth_deg"])
        elev = np.deg2rad(e["elev_deg"])
        d_expected = np.array([np.cos(elev) * np.cos(az), np.cos(elev) * np.sin(az),
                               np.sin(elev)])
        _, J_fresh = _fkj(arm, np.array(e["q"]))
        v = J_fresh @ np.array(e["qd"])
        assert abs(np.linalg.norm(v) - e["speed"]) < 1e-3          # speed matches
        cos_align = np.dot(v, d_expected) / np.linalg.norm(v)
        assert cos_align > 0.999                                    # >2.5deg off = fail

        # windup_path_feasible only checks STATIC torque along the neutral->
        # windup path (near-zero velocity); it says nothing about whether the
        # arm can actually SUSTAIN qd_release once it gets there. A training
        # run crashed on exactly this: a candidate statically fine but needing
        # 39.5/39.0 Nm at its own release velocity (Coriolis/centripetal,
        # which does NOT shrink when the throw ramp is stretched -- verified
        # unmoved across 6 stretch iterations). Every accepted table entry
        # must also pass the dynamic check at its own release state.
        assert release_dynamics_feasible(arm, np.array(e["q"]), np.array(e["qd"]))

    p.disconnect(cid)


def test_throw_ramp_feasible_catches_mid_ramp_torque_release_check_misses():
    """release_dynamics_feasible only checks the TERMINAL instant (qdd=0) of
    the throw ramp. Two real training crashes (peak torque ratio 1.02, then
    1.01 of tau_max) happened at MID-ramp points it never samples -- this
    fixture is the exact overhead release state that crashed a real
    training run despite passing release_dynamics_feasible. throw_ramp_
    feasible must catch what the release-only check misses."""
    cid = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    arm = p.loadURDF(pybullet_data.getDataPath() + "/kinova_gen3/gen3.urdf", useFixedBase=True)
    q = np.array([-3.141585, 0.087266, 0.0, 0.837758, 0.0, -0.279253, 0.0])
    qd = np.array([0.0, -1.3963, 0.0, -1.386713, 0.0, -1.2218, 0.0])
    assert release_dynamics_feasible(arm, q, qd)      # the misleading pass
    assert not throw_ramp_feasible(arm, q, qd, 1.1)   # the real, correct catch
    p.disconnect(cid)


def test_rotation_built_table_aims_every_azimuth_and_is_uniform():
    """build_table_by_rotation propagates ONE az=0 posture across the wedge by
    base-rotation (TossingBot / MC-PILOT Eq.5 formulation). Two invariants:
    every entry's qd must aim along its own labeled azimuth/elevation when
    re-checked against a fresh Jacobian (the sign of the base rotation is a
    real trap -- this URDF's base measures opposite world-z, verified 1e-6),
    and speed/elevation must be exactly uniform (the whole point: no
    azimuth discontinuities)."""
    from find_throw_pose import (build_table_by_rotation, _fkj, aimed_speed,
                                 base_rotation_sign)
    cid = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    arm = p.loadURDF(pybullet_data.getDataPath() + "/kinova_gen3/gen3.urdf", useFixedBase=True)

    q0 = np.deg2rad([0.0, 72.0, -60.0, -100.0, 0.0, 30.0, 0.0])
    _, J0 = _fkj(arm, q0)
    elev = np.deg2rad(15.0)
    d0 = np.array([np.cos(elev), 0.0, np.sin(elev)])
    s0, qd0 = aimed_speed(J0, d0, QD, freeze_roll=True)
    assert qd0 is not None and s0 > 0.3          # the fast wide-grid posture
    # The base-rotation sign is MEASURED, never assumed from the axis field --
    # build_table_by_rotation refuses an unstamped e0 rather than guessing.
    # Gen3 is -1 (its base measures opposite world-z); a UR is +1, and getting
    # it backwards aims the throw the wrong way round without any error.
    bs = base_rotation_sign(arm, q0)
    assert bs == -1.0
    e0 = {"range": 0.239, "q": q0, "qd": qd0, "elev_deg": 15.0,
          "speed": s0, "azimuth_deg": 0.0, "base_sign": bs}

    table = build_table_by_rotation(e0)
    assert len(table) == 23                       # full -33..+33 wedge, no gaps
    for e in table:
        assert abs(e["speed"] - s0) < 1e-12       # exactly uniform
        assert abs(e["elev_deg"] - 15.0) < 1e-12
        az = np.deg2rad(e["azimuth_deg"])
        d_lab = np.array([np.cos(elev) * np.cos(az), np.cos(elev) * np.sin(az),
                          np.sin(elev)])
        _, J = _fkj(arm, np.array(e["q"]))
        v = J @ np.array(e["qd"])
        assert abs(np.linalg.norm(v) - s0) < 1e-4
        cos_align = np.dot(v, d_lab) / np.linalg.norm(v)
        assert cos_align > 0.9999                 # aims at its own label
    p.disconnect(cid)


def test_follow_through_feasible_rejects_release_state_that_broke_hardware_recovery():
    """No search filter validated whether the arm can RECOVER after release
    (decelerate back to neutral) -- windup/throw/Coriolis-at-release were all
    checked, follow-through never was. A candidate that passed every other
    filter needed 1.15x tau_max to decelerate post-release (verified via
    ArmController.plan_throw's own runtime check raising RuntimeError on
    this exact state). The search's own filter must catch this BEFORE it
    ships in a table, not rely on discovering it at training/render time."""
    cid = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    arm = p.loadURDF(pybullet_data.getDataPath() + "/kinova_gen3/gen3.urdf", useFixedBase=True)
    q = np.array([-3.14159169, -0.52359878, 0.0, 0.66322512, 0.0, 0.41887902, 0.0])
    qd = np.array([0.0, -1.3963, 0.0, -1.3963, 0.0, -1.03565721, 0.0]) * (1.79 / 1.929)
    assert not follow_through_feasible(arm, q, qd)
    p.disconnect(cid)


def test_release_dynamics_feasible_catches_coriolis_violation_static_check_misses():
    """A posture/velocity combo that is statically fine (qd=0) can still be
    dynamically infeasible at its actual release velocity -- this is the
    exact case that crashed a training run (peak ratio 1.01, unmoved by
    stretching dt_throw). Regression fixture: an azimuth-wedge-edge posture
    from a real search run, static_feasible=True, release_dynamics_feasible
    at its own qd_release=False."""
    cid = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    arm = p.loadURDF(pybullet_data.getDataPath() + "/kinova_gen3/gen3.urdf", useFixedBase=True)
    q = np.array([-0.575959, 1.047198, -1.047198, -0.488692, 0.0, 1.308997, 0.0])
    qd = np.array([0.0, 0.175986, 0.0, -0.415818, 0.0, -1.2218, 0.0])
    assert static_feasible(arm, q)                             # zero-velocity check passes
    assert not release_dynamics_feasible(arm, q, qd)            # at-velocity check catches it
    p.disconnect(cid)


def test_windup_path_feasible_rejects_mid_swing_gravity_violation():
    """A release pose can be statically feasible while the neutral->windup
    swing still passes through a worse intermediate gravity configuration --
    verified: shoulder sweeping neutral ~22deg -> cocked 90deg peaks ~39Nm
    mid-swing even though both endpoints are individually fine. This must be
    rejected, not just checked at the two endpoints."""
    cid = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
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
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    arm = p.loadURDF(pybullet_data.getDataPath() + "/kinova_gen3/gen3.urdf", useFixedBase=True)
    q_release = QN.copy()
    qd_release = np.zeros(7)
    assert windup_path_feasible(arm, q_release, qd_release, t_throw=0.9)
    p.disconnect(cid)
