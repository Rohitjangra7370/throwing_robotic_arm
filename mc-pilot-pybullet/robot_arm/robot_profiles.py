"""
Robot profiles for mc-pilot-pybullet.

This module centralizes the robot-dependent assumptions that used to be baked
into the KUKA-only arm controller:
  - URDF path inside pybullet_data
  - actuated joint indices used for planning
  - end-effector link index used for IK / Jacobians / ball attachment
  - neutral configuration used as the throw start pose
  - joint velocity limits used to detect velocity clipping
  - default release position and timing hints for quick arm comparisons
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np


@dataclass(frozen=True)
class RobotProfile:
    name: str
    urdf_rel_path: str
    joint_ids: tuple[int, ...]
    ee_link: int
    q_neutral: tuple[float, ...]
    qd_max: tuple[float, ...]
    default_release_pos: tuple[float, float, float]
    speed_bounds: tuple[float, float]
    timing: tuple[float, float, float]
    position_gain: float = 2.0
    velocity_gain: float = 1.0
    force_scale: float = 1.5
    control_mode: str = "position"
    use_safe_release: bool = False
    notes: str = ""


_PROFILES: Dict[str, RobotProfile] = {
    "kuka_iiwa": RobotProfile(
        name="kuka_iiwa",
        urdf_rel_path="kuka_iiwa/model.urdf",
        joint_ids=(0, 1, 2, 3, 4, 5, 6),
        ee_link=6,
        q_neutral=(0.0, 0.5, 0.0, -1.0, 0.0, 0.5, 0.0),
        qd_max=(1.71, 1.71, 1.75, 2.27, 2.44, 3.14, 3.14),
        default_release_pos=(0.50, 0.00, 0.50),
        speed_bounds=(0.6, 2.5),
        timing=(0.30, 0.60, 1.20),
        position_gain=2.0,
        velocity_gain=1.0,
        force_scale=1.5,
        control_mode="position",
        use_safe_release=False,
        notes="Original mc-pilot-pybullet baseline arm.",
    ),
    "franka_panda": RobotProfile(
        name="franka_panda",
        urdf_rel_path="franka_panda/panda.urdf",
        joint_ids=(0, 1, 2, 3, 4, 5, 6),
        ee_link=8,
        q_neutral=(0.0, -0.30, 0.0, -2.20, 0.0, 2.00, 0.80),
        qd_max=(2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61),
        default_release_pos=(0.45, 0.00, 0.40),
        speed_bounds=(0.5, 2.2),
        timing=(0.30, 0.60, 1.20),
        position_gain=0.6,
        velocity_gain=0.3,
        force_scale=0.5,
        control_mode="position",
        use_safe_release=True,
        notes="7-DoF Franka arm; EE link is the hand reached through fixed joints after joint 7.",
    ),
    "xarm6": RobotProfile(
        name="xarm6",
        urdf_rel_path="xarm/xarm6_robot.urdf",
        joint_ids=(1, 2, 3, 4, 5, 6),
        ee_link=6,
        q_neutral=(0.0, -0.60, -1.20, 0.0, 1.80, 0.0),
        qd_max=(3.14, 3.14, 3.14, 3.14, 3.14, 3.14),
        default_release_pos=(0.42, 0.00, 0.32),
        speed_bounds=(0.4, 2.0),
        timing=(0.25, 0.50, 1.00),
        position_gain=1.5,
        velocity_gain=0.7,
        force_scale=1.0,
        control_mode="kinematic",
        use_safe_release=True,
        notes="6-DoF xArm; index 0 is a fixed world joint so the actuated chain starts at joint 1.",
    ),
}


def available_robot_names() -> List[str]:
    return sorted(_PROFILES.keys())


def get_robot_profile(name: str) -> RobotProfile:
    key = name.lower()
    if key not in _PROFILES:
        raise ValueError(
            f"Unknown robot '{name}'. Available: {', '.join(available_robot_names())}"
        )
    return _PROFILES[key]


def profile_to_dict(profile: RobotProfile) -> dict:
    return {
        "name": profile.name,
        "urdf_rel_path": profile.urdf_rel_path,
        "joint_ids": list(profile.joint_ids),
        "ee_link": profile.ee_link,
        "q_neutral": list(profile.q_neutral),
        "qd_max": list(profile.qd_max),
        "default_release_pos": list(profile.default_release_pos),
        "speed_bounds": list(profile.speed_bounds),
        "timing": list(profile.timing),
        "position_gain": profile.position_gain,
        "velocity_gain": profile.velocity_gain,
        "force_scale": profile.force_scale,
        "control_mode": profile.control_mode,
        "use_safe_release": profile.use_safe_release,
        "notes": profile.notes,
    }


def qd_max_norm(profile: RobotProfile) -> float:
    return float(np.linalg.norm(np.asarray(profile.qd_max, dtype=float)))


def iter_profiles(names: Iterable[str] | None = None) -> List[RobotProfile]:
    if names is None:
        return [get_robot_profile(name) for name in available_robot_names()]
    return [get_robot_profile(name) for name in names]
