"""Arm-on-a-plate geometry: the base is raised, the floor is not lowered.

The real Gen3 is bolted to a 0.433 m plate and throws to the floor. Two ways to
express that, one of which is silently wrong:

  * base_height=0.433, target_height=0.0   -- correct. Floor stays the real
    collision plane at world z=0; the arm sits above it.
  * base_height=0.0, target_height=-0.433  -- WRONG, and wrong without an error.
    The ball hits plane.urdf at z=0 before it can cross a landing plane below
    it, the descending-crossing test never fires, and the rollout returns with
    the ball's resting position as if it were a landing.

These tests pin both, plus the pose search's landing plane, which was hardcoded
to base-frame z=0 (i.e. floor level with the base) for every table shipped
before 2026-08-12.
"""
import os
import sys

import numpy as np
import pybullet as p
import pybullet_data
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot_arm.arm_controller import ArmController          # noqa: E402
from robot_arm.robot_profiles import get_robot_profile       # noqa: E402
from simulation_class.model_pybullet import PyBulletThrowingSystem  # noqa: E402

PLATE = 0.433


def test_negative_target_height_is_refused_at_construction():
    with pytest.raises(ValueError, match="below the floor"):
        PyBulletThrowingSystem(robot_name="kinova_gen3_dyn", target_height=-PLATE)


def test_negative_target_height_is_refused_per_episode():
    """The 9-D height-conditioned task sets target_height from the state."""
    sys_ = PyBulletThrowingSystem(robot_name="kinova_gen3_dyn")
    s0 = np.zeros(9)
    s0[8] = -PLATE                      # per-episode landing plane
    with pytest.raises(ValueError, match="below the floor"):
        sys_.rollout(s0, lambda s, t: np.array([1.0]), T=1.0, dt=0.02, noise=None)


def test_base_height_defaults_to_zero():
    """Every result before 2026-08-12 was base-at-floor; the default must hold."""
    assert PyBulletThrowingSystem(robot_name="kinova_gen3_dyn").base_height == 0.0


def test_base_height_raises_the_whole_arm_by_exactly_that_much():
    """FK at the same joint angles must shift by the plate height, nothing else."""
    prof = get_robot_profile("kinova_gen3_dyn")
    urdf = pybullet_data.getDataPath() + "/" + prof.urdf_rel_path
    q = np.array(prof.q_neutral, dtype=float)

    ee = []
    for z in (0.0, PLATE):
        cid = p.connect(p.DIRECT)
        arm = ArmController(cid, urdf, robot_name="kinova_gen3_dyn",
                            base_position=(0.0, 0.0, z))
        arm.reset()
        for j, jid in enumerate(prof.joint_ids):
            p.resetJointState(arm._arm_id, jid, float(q[j]), physicsClientId=cid)
        pos, _, _, _ = arm.ee_state()
        ee.append(np.asarray(pos, dtype=float))
        p.disconnect(cid)

    delta = ee[1] - ee[0]
    assert np.allclose(delta[:2], 0.0, atol=1e-9), "raising the base moved x/y"
    assert delta[2] == pytest.approx(PLATE, abs=1e-9)


def test_ball_lands_on_the_floor_not_on_the_base_plane():
    """With the base raised, the recorded landing is the real floor at z=0."""
    sys_ = PyBulletThrowingSystem(robot_name="kinova_gen3_dyn",
                                  base_height=PLATE, target_height=0.0)
    s0 = np.zeros(8)
    s0[0:3] = [0.0, 0.0, 1.0 + PLATE]     # release above the raised base
    s0[6:8] = [0.7, 0.0]
    pos, _, _ = sys_.rollout(s0, lambda s, t: np.array([1.4]), T=1.0, dt=0.02,
                             noise=0.0)
    assert pos[-1][2] == pytest.approx(0.0, abs=1e-6)


def test_pose_search_landing_plane_is_parameterised():
    """A lower floor must give a longer throw; the default must not move."""
    import find_throw_pose as ftp

    pos = np.array([0.035, 0.0, 1.137])           # the shipped release locus
    vel = np.array([1.628, 0.0, 0.0])

    rng_level, land_level = ftp.ballistic_range(pos, vel, floor_z=0.0)
    rng_plate, land_plate = ftp.ballistic_range(pos, vel, floor_z=-PLATE)

    assert land_level[2] <= 0.0
    assert land_plate[2] <= -PLATE
    assert rng_plate > rng_level, "dropping the floor must lengthen the flight"
    # Default (no argument) must reproduce the pre-2026-08-12 behaviour exactly.
    assert ftp.ballistic_range(pos, vel)[0] == pytest.approx(rng_level, abs=1e-12)


def test_set_floor_z_is_honoured_and_restorable():
    import find_throw_pose as ftp

    pos = np.array([0.035, 0.0, 1.137])
    vel = np.array([1.628, 0.0, 0.0])
    baseline = ftp.ballistic_range(pos, vel)[0]
    try:
        ftp.set_floor_z(-PLATE)
        assert ftp.ballistic_range(pos, vel)[0] > baseline
    finally:
        ftp.set_floor_z(0.0)
    assert ftp.ballistic_range(pos, vel)[0] == pytest.approx(baseline, abs=1e-12)
