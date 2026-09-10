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
    tau_max: tuple[float, ...] | None = None
    kp: tuple[float, ...] | None = None
    kd: tuple[float, ...] | None = None
    windup_delta: tuple[float, ...] | None = None
    # Indices (into joint_ids) of joints frozen at qd=0 during the throw: the
    # base's azimuth joint plus every roll/twist joint, whose axis runs roughly
    # ALONG its own link so an LP that spins it makes a corkscrew, not a throw.
    # None keeps the historical 7-DoF alternating roll-pitch-roll set (0,2,4,6),
    # trimmed to this arm's DoF count -- see `roll_indices()`. Arms that are not
    # alternating roll-pitch (any UR: pan / three parallel pitches / wrist2 /
    # tool-roll) MUST state it explicitly.
    roll_idx: tuple[int, ...] | None = None
    # Manufacturer-rated maximum Cartesian TCP speed (m/s), or None when the
    # arm publishes no such limit. This is a SEPARATE ceiling from qd_max: the
    # release LP maximizes end-effector speed subject to joint-velocity bounds
    # alone, and on an arm whose Cartesian rating binds first, every solution it
    # returns is over the manufacturer's limit and would be clamped or refused
    # by the controller. Never bound on the Gen3 (LP max 2.07 m/s, no published
    # Cartesian rating that binds); binds on EVERY UR7e table entry (LP 4.65 m/s
    # against a rated 4.0). None leaves the LP exactly as it was.
    v_tcp_max: float | None = None
    # Follow-through-safe release-speed ceiling (m/s), or None when the arm can
    # recover from its pose table's full kinematic max. A table entry is chosen
    # for RANGE; being able to stop afterwards is a separate question, and on
    # the Gen3 only 76.8% of the table max is recoverable-from. Measured by
    # bisecting find_throw_pose.follow_through_feasible() on the shipped table
    # (Gen3: 1.591 m/s against a 2.070 table max, which is where the 1.60 in
    # eval_adapted_height.py came from; UR7e: 100% of its 4.0, no cap needed).
    # Re-measure per arm AND per table -- it is a property of both.
    safe_u_cap: float | None = None
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
    "franka_panda_dyn": RobotProfile(
        name="franka_panda_dyn",
        urdf_rel_path="franka_panda/panda.urdf",
        joint_ids=(0, 1, 2, 3, 4, 5, 6),
        ee_link=8,
        q_neutral=(0.0, -0.30, 0.0, -2.20, 0.0, 2.00, 0.80),
        qd_max=(2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61),
        default_release_pos=(0.45, 0.00, 0.40),
        speed_bounds=(0.5, 2.2),
        timing=(0.30, 0.60, 1.20),
        control_mode="torque",
        use_safe_release=False,
        tau_max=(87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0),
        kp=(400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0),
        kd=(60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0),
        notes=(
            "Franka Panda under computed-torque control, kinetic-chain (opt_pose) "
            "throw mode -- same measured-dynamic-release mechanism as "
            "kinova_gen3_dyn. tau_max: published Franka Emika limits, 87 Nm "
            "joints 1-4, 12 Nm joints 5-7 (franka_ros joint_limits.yaml). "
            "Joint-axis check (numeric, at q_neutral): joints 2,4,6 (idx1,3,5) "
            "are exactly perpendicular (90deg) to the base->EE vector -- pitch, "
            "carry throw velocity; joints 1,3,5,7 (idx0,2,4,6) are 0-69deg -- "
            "roll/twist, frozen during the throw. Same alternating roll-pitch "
            "structure as Kinova Gen3."
        ),
    ),
    "kinova_gen3": RobotProfile(
        name="kinova_gen3",
        urdf_rel_path="kinova_gen3/gen3.urdf",
        joint_ids=(0, 1, 2, 3, 4, 5, 6),
        ee_link=7,
        q_neutral=(-0.01, 0.382, -0.04, 1.46, -0.012, 0.731, 0.0),
        qd_max=(1.3963, 1.3963, 1.3963, 1.3963, 1.2218, 1.2218, 1.2218),
        default_release_pos=(0.55, 0.00, 0.45),
        speed_bounds=(0.3, 0.6),
        timing=(0.40, 0.80, 1.60),
        position_gain=0.6,
        velocity_gain=0.3,
        force_scale=0.5,
        control_mode="kinematic",
        use_safe_release=True,
        # q_release computed by plan_throw's IK for default_release_pos comes out
        # EXACTLY equal to q_neutral (default_release_pos was defined as wherever
        # q_neutral's own forward kinematics already places the EE) -- so the
        # normal windup formula q_neutral + (q_release-q_neutral)*(-0.5) collapses
        # to zero for ANY multiplier, and the arm never visibly winds up (verified:
        # kuka swings 44deg, xarm6 38deg, franka 8.6deg, kinova 0.0deg exactly).
        # windup_delta gives an explicit, independent "cocked back" pose so a real
        # swing exists without touching q_neutral or default_release_pos (both are
        # load-bearing for the speed-ceiling/target-range calibration in this repo).
        safe_u_cap=1.60,
        windup_delta=(0.0, -0.20, 0.0, -0.25, 0.0, 0.0, 0.0),
        notes=(
            "Kinova Gen3 7-DoF (lab hardware target). URDF: official ros_kortex "
            "GEN3-7DOF-VISION V12, meshes vendored into pybullet_data/kinova_gen3. "
            "Joint velocity limits from URDF (1.396/1.222 rad/s) cap achievable "
            "EE speed; measured (via plan_throw's qd_max clip_scale) at zero "
            "clipping across the full +-30-deg azimuth range, the ceiling is "
            "~0.61 m/s, not the earlier ~1.0 m/s estimate (that number was "
            "measured on-axis only, at 0-deg azimuth, where the Jacobian is "
            "more favorable than at the +-30-deg extremes)."
        ),
    ),
    "kinova_gen3_dyn": RobotProfile(
        name="kinova_gen3_dyn",
        urdf_rel_path="kinova_gen3/gen3.urdf",
        joint_ids=(0, 1, 2, 3, 4, 5, 6),
        ee_link=7,
        q_neutral=(-0.01, 0.382, -0.04, 1.46, -0.012, 0.731, 0.0),
        qd_max=(1.3963, 1.3963, 1.3963, 1.3963, 1.2218, 1.2218, 1.2218),
        default_release_pos=(0.55, 0.00, 0.45),
        speed_bounds=(0.3, 0.6),
        timing=(0.40, 0.80, 1.60),
        control_mode="torque",
        use_safe_release=False,
        tau_max=(39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0),
        kp=(400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0),
        kd=(60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0),
        safe_u_cap=1.60,
        windup_delta=(0.0, -0.20, 0.0, -0.25, 0.0, 0.0, 0.0),
        notes=(
            "Kinova Gen3 under computed-torque control (velocity-from-dynamics "
            "study). Same kinematics as kinova_gen3; release velocity comes from "
            "tracked arm motion, not resetBaseVelocity. tau_max: 39 Nm large "
            "actuators (joints 1-4), 9 Nm wrists (5-7). windup_delta: see "
            "kinova_gen3's notes -- q_release equals q_neutral here too."
        ),
    ),
    # --- Synthetic torque-headroom sweep (ICRA paper ablation, Sec. results-ablation) ---
    # Copies of kinova_gen3_dyn with ONLY tau_max scaled, everything else identical
    # (URDF, kinematics, q_neutral, windup_delta, kp/kd). Isolates torque headroom as
    # the sole variable so the feasibility-cascade ablation can show a dose-response
    # curve instead of resting on two arms that also differ in kinematics/DOF layout.
    # k=1.0 reproduces kinova_gen3_dyn's real 39/9 Nm; k=3.33 lands near the Panda's
    # real 87/12 Nm (used as an external cross-check, not a sweep endpoint).
    "kinova_gen3_dyn_tau0.50": RobotProfile(
        name="kinova_gen3_dyn_tau0.50",
        urdf_rel_path="kinova_gen3/gen3.urdf",
        joint_ids=(0, 1, 2, 3, 4, 5, 6),
        ee_link=7,
        q_neutral=(-0.01, 0.382, -0.04, 1.46, -0.012, 0.731, 0.0),
        qd_max=(1.3963, 1.3963, 1.3963, 1.3963, 1.2218, 1.2218, 1.2218),
        default_release_pos=(0.55, 0.00, 0.45),
        speed_bounds=(0.3, 0.6),
        timing=(0.40, 0.80, 1.60),
        control_mode="torque",
        use_safe_release=False,
        tau_max=(19.5, 19.5, 19.5, 19.5, 4.5, 4.5, 4.5),
        kp=(400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0),
        kd=(60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0),
        windup_delta=(0.0, -0.20, 0.0, -0.25, 0.0, 0.0, 0.0),
        notes="Torque sweep k=0.50x kinova_gen3_dyn (39/9 -> 19.5/4.5 Nm). See kinova_gen3_dyn_tau1.00 for sweep rationale.",
    ),
    "kinova_gen3_dyn_tau0.75": RobotProfile(
        name="kinova_gen3_dyn_tau0.75",
        urdf_rel_path="kinova_gen3/gen3.urdf",
        joint_ids=(0, 1, 2, 3, 4, 5, 6),
        ee_link=7,
        q_neutral=(-0.01, 0.382, -0.04, 1.46, -0.012, 0.731, 0.0),
        qd_max=(1.3963, 1.3963, 1.3963, 1.3963, 1.2218, 1.2218, 1.2218),
        default_release_pos=(0.55, 0.00, 0.45),
        speed_bounds=(0.3, 0.6),
        timing=(0.40, 0.80, 1.60),
        control_mode="torque",
        use_safe_release=False,
        tau_max=(29.25, 29.25, 29.25, 29.25, 6.75, 6.75, 6.75),
        kp=(400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0),
        kd=(60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0),
        windup_delta=(0.0, -0.20, 0.0, -0.25, 0.0, 0.0, 0.0),
        notes="Torque sweep k=0.75x kinova_gen3_dyn (39/9 -> 29.25/6.75 Nm). See kinova_gen3_dyn_tau1.00 for sweep rationale.",
    ),
    "kinova_gen3_dyn_tau1.00": RobotProfile(
        name="kinova_gen3_dyn_tau1.00",
        urdf_rel_path="kinova_gen3/gen3.urdf",
        joint_ids=(0, 1, 2, 3, 4, 5, 6),
        ee_link=7,
        q_neutral=(-0.01, 0.382, -0.04, 1.46, -0.012, 0.731, 0.0),
        qd_max=(1.3963, 1.3963, 1.3963, 1.3963, 1.2218, 1.2218, 1.2218),
        default_release_pos=(0.55, 0.00, 0.45),
        speed_bounds=(0.3, 0.6),
        timing=(0.40, 0.80, 1.60),
        control_mode="torque",
        use_safe_release=False,
        tau_max=(39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0),
        kp=(400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0),
        kd=(60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0),
        windup_delta=(0.0, -0.20, 0.0, -0.25, 0.0, 0.0, 0.0),
        notes=(
            "Torque sweep k=1.00x: bit-identical to kinova_gen3_dyn's real 39/9 Nm, "
            "kept as a separate entry so the sweep results file set is self-contained "
            "and doesn't rely on cross-referencing the production profile. Sweep exists "
            "to isolate torque headroom as the causal variable behind the "
            "feasibility-cascade ablation (Table ablation): kinova_gen3_dyn vs. "
            "franka_panda_dyn differ in torque AND kinematics/DOF layout, so a "
            "two-arm table can't rule out kinematics as the real driver. Every entry "
            "in this sweep shares kinova_gen3_dyn's exact URDF/kinematics/q_neutral/ "
            "windup_delta/kp/kd -- only tau_max varies."
        ),
    ),
    "kinova_gen3_dyn_tau1.50": RobotProfile(
        name="kinova_gen3_dyn_tau1.50",
        urdf_rel_path="kinova_gen3/gen3.urdf",
        joint_ids=(0, 1, 2, 3, 4, 5, 6),
        ee_link=7,
        q_neutral=(-0.01, 0.382, -0.04, 1.46, -0.012, 0.731, 0.0),
        qd_max=(1.3963, 1.3963, 1.3963, 1.3963, 1.2218, 1.2218, 1.2218),
        default_release_pos=(0.55, 0.00, 0.45),
        speed_bounds=(0.3, 0.6),
        timing=(0.40, 0.80, 1.60),
        control_mode="torque",
        use_safe_release=False,
        tau_max=(58.5, 58.5, 58.5, 58.5, 13.5, 13.5, 13.5),
        kp=(400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0),
        kd=(60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0),
        windup_delta=(0.0, -0.20, 0.0, -0.25, 0.0, 0.0, 0.0),
        notes="Torque sweep k=1.50x kinova_gen3_dyn (39/9 -> 58.5/13.5 Nm). See kinova_gen3_dyn_tau1.00 for sweep rationale.",
    ),
    "kinova_gen3_dyn_tau2.25": RobotProfile(
        name="kinova_gen3_dyn_tau2.25",
        urdf_rel_path="kinova_gen3/gen3.urdf",
        joint_ids=(0, 1, 2, 3, 4, 5, 6),
        ee_link=7,
        q_neutral=(-0.01, 0.382, -0.04, 1.46, -0.012, 0.731, 0.0),
        qd_max=(1.3963, 1.3963, 1.3963, 1.3963, 1.2218, 1.2218, 1.2218),
        default_release_pos=(0.55, 0.00, 0.45),
        speed_bounds=(0.3, 0.6),
        timing=(0.40, 0.80, 1.60),
        control_mode="torque",
        use_safe_release=False,
        tau_max=(87.75, 87.75, 87.75, 87.75, 20.25, 20.25, 20.25),
        kp=(400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0),
        kd=(60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0),
        windup_delta=(0.0, -0.20, 0.0, -0.25, 0.0, 0.0, 0.0),
        notes="Torque sweep k=2.25x kinova_gen3_dyn (39/9 -> 87.75/20.25 Nm). See kinova_gen3_dyn_tau1.00 for sweep rationale.",
    ),
    "kinova_gen3_dyn_tau3.33": RobotProfile(
        name="kinova_gen3_dyn_tau3.33",
        urdf_rel_path="kinova_gen3/gen3.urdf",
        joint_ids=(0, 1, 2, 3, 4, 5, 6),
        ee_link=7,
        q_neutral=(-0.01, 0.382, -0.04, 1.46, -0.012, 0.731, 0.0),
        qd_max=(1.3963, 1.3963, 1.3963, 1.3963, 1.2218, 1.2218, 1.2218),
        default_release_pos=(0.55, 0.00, 0.45),
        speed_bounds=(0.3, 0.6),
        timing=(0.40, 0.80, 1.60),
        control_mode="torque",
        use_safe_release=False,
        tau_max=(129.87, 129.87, 129.87, 129.87, 29.97, 29.97, 29.97),
        kp=(400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0),
        kd=(60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0),
        windup_delta=(0.0, -0.20, 0.0, -0.25, 0.0, 0.0, 0.0),
        notes=(
            "Torque sweep k=3.33x kinova_gen3_dyn (39/9 -> 129.87/29.97 Nm, "
            "~= franka_panda_dyn's real 87/12 Nm ratio-matched). Sweep endpoint used "
            "as external cross-check against the real Panda's 0%-rejection result, "
            "not as a physically meaningful arm. See kinova_gen3_dyn_tau1.00 for "
            "sweep rationale."
        ),
    ),
    # --- Universal Robots UR7e (second hardware target, added 2026-09-06) ---
    # URDF: official UniversalRobots/Universal_Robots_ROS2_Description tag 4.3.1,
    # config/ur7e, xacro-expanded and vendored to pybullet_data/ur7e/ by
    # scripts/install_ur7e_urdf.sh (that script is the provenance record -- read
    # it before trusting any number below).
    #
    # WHAT IS VERIFIED AND WHAT IS NOT, stated up front because this arm's
    # upstream description is partly inherited from the UR5e:
    #   VERIFIED  kinematics  -- UR ship ONE shared "UR5e/UR7e" JT file and one
    #             shared working-area PDF; same 850 mm reach, 20.6 kg, D151 mm
    #             footprint. The identical link geometry is real, not a stub.
    #   VERIFIED  qd_max 180 deg/s = 3.1416 rad/s on all six -- matches the UR7e
    #             tech sheet, not just the UR5e-sourced yaml.
    #   PROVISIONAL  tau_max 150/150/150/28/28/28 Nm -- these are the UR5e's
    #             published values, carried over verbatim by upstream (the yaml
    #             header cites the UR5e manual). UR's public max-joint-torque
    #             article has no UR7e row. The UR7e lifts 7.5 kg vs the UR5e's 5,
    #             so the true limits are very likely HIGHER, i.e. this is
    #             conservative and fails closed -- but it is not measured.
    #             Measure via RTDE actual_current / target_moment before
    #             reporting any torque-headroom result for this arm.
    #   PROVISIONAL  link masses/inertias -- byte-identical to ur5e's upstream.
    #   PROVISIONAL  kp/kd, timing, windup_delta, speed_bounds -- see below.
    #
    # Joint structure is NOT the Gen3/Panda alternating roll-pitch-roll layout,
    # so roll_idx is stated explicitly. Classified numerically off the PyBullet
    # Jacobian at a candidate release pose (convention-free -- axis-vs-link-vector
    # angles were tried first and gave the wrong answer through a frame-convention
    # slip, the Jacobian columns did not):
    #     idx 0 shoulder_pan   |Jv| 0.436, ALL of it out-of-plane -> azimuth, FREEZE
    #     idx 1 shoulder_lift  |Jv| 0.440, out-of-plane 0.000      -> PITCH
    #     idx 2 elbow          |Jv| 0.364, out-of-plane 0.000      -> PITCH
    #     idx 3 wrist_1        |Jv| 0.141, out-of-plane 0.000      -> PITCH
    #     idx 4 wrist_2        |Jv| 0.100, ALL of it out-of-plane  -> FREEZE
    #     idx 5 wrist_3        |Jv| 0.000 EXACTLY                  -> tool roll, FREEZE
    # Three velocity-carrying joints, same count as the Gen3's (1,3,5), so the
    # release LP's structure is unchanged. wrist_3 is the exact analogue of the
    # Gen3's joint 7 that `--wrist_roll_offset_deg` exploits: its axis IS the
    # tool axis, so it is provably free for finger clearance -- and stays free
    # under a TCP offset only while that offset is purely axial (0,0,L).
    #
    # ee_link=10 is `tool0`, whose local +z runs out along the tool axis (checked
    # in PyBullet: flange/ft_frame/tool0 are co-located but differently rotated,
    # and only tool0 gives the z-out convention `tool_offset=(0,0,L)` assumes).
    #
    # NOTE the y=0.1333 m wrist offset in default_release_pos: unlike the Gen3,
    # the UR's throw plane does NOT pass through the base axis. Train with
    # --flight_targets (sample by flight distance from the release point, not
    # polar-from-origin) -- that flag exists for exactly this geometry.
    "ur7e": RobotProfile(
        name="ur7e",
        urdf_rel_path="ur7e/ur7e.urdf",
        joint_ids=(2, 3, 4, 5, 6, 7),
        ee_link=10,
        q_neutral=(0.0, -0.5236, 0.8727, -1.4835, -1.5708, 0.0),
        qd_max=(3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416),
        default_release_pos=(0.7849, 0.1333, 0.1084),
        speed_bounds=(0.5, 2.5),
        timing=(0.40, 0.80, 1.60),
        position_gain=0.6,
        velocity_gain=0.3,
        force_scale=0.5,
        control_mode="kinematic",
        use_safe_release=True,
        roll_idx=(0, 4, 5),
        v_tcp_max=4.0,
        windup_delta=(0.0, -0.35, -0.30, -0.25, 0.0, 0.0),
        notes=(
            "UR7e 6-DoF, KINEMATIC mode -- ball velocity is assigned, the arm is "
            "cosmetic. Comparison/plotting only, NEVER a physical throw. Use "
            "ur7e_dyn for anything real. joint_ids (2..7) skip the two leading "
            "FIXED joints (base_joint, base_link-base_link_inertia) that the UR "
            "URDF puts before shoulder_pan. q_neutral is the LP-optimal forward "
            "release posture found by a coarse pose sweep, and "
            "default_release_pos is its own tool0 FK, so plan_throw's IK returns "
            "q_neutral exactly -- which is why windup_delta is given explicitly "
            "(same collapse-to-zero-windup problem as the Gen3; see its notes)."
        ),
    ),
    "ur7e_dyn": RobotProfile(
        name="ur7e_dyn",
        urdf_rel_path="ur7e/ur7e.urdf",
        joint_ids=(2, 3, 4, 5, 6, 7),
        ee_link=10,
        q_neutral=(0.0, -0.5236, 0.8727, -1.4835, -1.5708, 0.0),
        qd_max=(3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416),
        default_release_pos=(0.7849, 0.1333, 0.1084),
        speed_bounds=(0.5, 2.5),
        timing=(0.40, 0.80, 1.60),
        control_mode="torque",
        use_safe_release=False,
        tau_max=(150.0, 150.0, 150.0, 28.0, 28.0, 28.0),
        kp=(400.0, 400.0, 400.0, 80.0, 80.0, 80.0),
        kd=(60.0, 60.0, 60.0, 12.0, 12.0, 12.0),
        roll_idx=(0, 4, 5),
        v_tcp_max=4.0,
        windup_delta=(0.0, -0.35, -0.30, -0.25, 0.0, 0.0),
        notes=(
            "UR7e under computed-torque control -- the real throw mode, the "
            "ur7e counterpart of kinova_gen3_dyn. "
            "SPEED: the direction-constrained release LP over a forward-release "
            "pose sweep gives 4.157 m/s unaimed, which EXCEEDS the UR7e tech "
            "sheet's 4 m/s max TCP speed -- so this arm is TCP-speed-limited, not "
            "joint-velocity-limited, the opposite of the Gen3. Capped at 4 m/s "
            "that is 1.93x the Gen3's 2.07 m/s and 2.14x its range (2.00 m vs "
            "0.9355 m off a 0.433 m plate). Every trained target band, cost "
            "lengthscale and RBF lengthscale init from the Gen3 is therefore "
            "invalid here, and the lab needs ~2 m of clear floor. speed_bounds "
            "below is a deliberately conservative provisional range; the trainer's "
            "--uM/--uMin override it and find_throw_pose.py fixes the real "
            "achievable envelope. "
            "kp/kd come from an actual 2-D gain sweep (2026-09-06), not a guess. "
            "The first guess -- scaling the Gen3's 400/60 by the tau_max ratio to "
            "1500/150 -- was UNSTABLE: measured release speed reached 9.5 m/s "
            "against a commanded 4.0, i.e. the controller was injecting energy, "
            "not tracking. kd, not kp, was the culprit: every kd >= 160 blows up "
            "at every kp tried (100-800), while kp is almost flat over 200-800 "
            "once kd <= 80. Values below keep the Gen3's proven kd/kp = 0.15 "
            "damping ratio and scale the wrists by the tau_max ratio (28/150). "
            "Re-sweep if the timing or the pose table changes. "
            "timing/windup_delta are carried over from the Gen3 unvalidated -- "
            "check all three trajectory phases through the real planner "
            "(feasibility != realization) before running anything."
        ),
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


_LEGACY_ROLL_IDX = (0, 2, 4, 6)


def roll_indices(profile: RobotProfile) -> tuple[int, ...]:
    """
    Joints frozen at qd=0 during the throw, as indices into `joint_ids`.

    Falls back to the historical 7-DoF alternating roll-pitch-roll set when a
    profile does not declare one, TRIMMED to the arm's DoF count. The trim is
    not cosmetic: on a 6-DoF arm the bare constant addresses index 6, which in
    the release LP is the speed slack variable, so freezing it would pin the
    release speed to exactly zero with no error raised.
    """
    if profile.roll_idx is not None:
        idx = tuple(int(i) for i in profile.roll_idx)
    else:
        idx = tuple(i for i in _LEGACY_ROLL_IDX if i < len(profile.joint_ids))
    n = len(profile.joint_ids)
    bad = [i for i in idx if not (0 <= i < n)]
    if bad:
        raise ValueError(
            f"{profile.name}: roll_idx {bad} out of range for {n} actuated joints"
        )
    return idx


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
        "tau_max": list(profile.tau_max) if profile.tau_max is not None else None,
        "kp": list(profile.kp) if profile.kp is not None else None,
        "kd": list(profile.kd) if profile.kd is not None else None,
        "windup_delta": list(profile.windup_delta) if profile.windup_delta is not None else None,
        "roll_idx": list(roll_indices(profile)),
        "v_tcp_max": profile.v_tcp_max,
        "safe_u_cap": profile.safe_u_cap,
        "notes": profile.notes,
    }


def qd_max_norm(profile: RobotProfile) -> float:
    return float(np.linalg.norm(np.asarray(profile.qd_max, dtype=float)))


def iter_profiles(names: Iterable[str] | None = None) -> List[RobotProfile]:
    if names is None:
        return [get_robot_profile(name) for name in available_robot_names()]
    return [get_robot_profile(name) for name in names]
