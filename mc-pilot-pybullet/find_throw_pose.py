"""
Hardware-valid aimed-throw pose search on the kinetic-chain principle.

  * Base joint j1 (vertical z-axis) sets AZIMUTH only and is held STILL during
    the throw: qd[0] = 0. It is NOT a speed source.
  * ONLY joints whose rotation axis is PERPENDICULAR to the swing plane carry
    throw velocity. For this URDF that is the three PITCH joints -- shoulder
    (joint_2, idx1), elbow (joint_4, idx3), wrist (joint_6, idx5). The ROLL/
    TWIST joints -- base (joint_1, idx0), shoulder-roll (joint_3, idx2),
    wrist-roll1 (joint_5, idx4), wrist-roll2 (joint_7, idx6) -- rotate about an
    axis roughly ALONG the connecting link, not perpendicular to any useful
    swing plane; letting the LP spin them produces a corkscrew/twisted motion,
    not a throw. They are frozen at qd=0 during the throw and used only for
    static SETUP (which vertical plane the swing happens in). Their STATIC
    angle (esp. shoulder-roll, idx2) is still a free search parameter: with
    ALL of them pinned to literal zero angle too, the three pitch axes become
    exactly parallel and the achievable velocity direction collapses to a
    single line (verified: s~0 for every elevation > a few degrees). A modest
    static roll offset breaks that degeneracy.
  * Release-instant joint velocities are <= qd_max by LP construction; the
    monotonic windup (handled in plan_throw) keeps the whole stroke <= qd_max.
  * Postures are filtered by REAL static torque feasibility (gravity alone,
    zero velocity, via calculateInverseDynamics against the actual Kinova
    Gen3 actuator limits -- 39 Nm large joints, 9 Nm wrists, confirmed against
    Kinova's own published URDF, Kinovarobotics/ros_kortex) -- not a proxy
    metric. A posture that can't even be held still under gravity is rejected
    outright, regardless of how much throw speed the LP finds for it.

Rotating the base does NOT cleanly rotate this arm's throw geometry (verified:
J(base+az) != Rz*J(base)), so a single posture aimed at az=0 collapses off-axis.
We therefore search a FRESH posture PER AZIMUTH over the training wedge and
save an azimuth->posture TABLE. Poses whose windup cock leaves joint limits
are rejected.
"""
import argparse

import numpy as np
import pybullet as p
import pybullet_data
from scipy.optimize import linprog
from simulation_class.model import _ball_accel
from robot_arm.robot_profiles import get_robot_profile
from robot_arm.arm_controller import _cubic_to_velocity, _cubic_from_velocity, _eval_cubic

# Defaults preserve the original Kinova Gen3 behavior; override via --robot to
# target a different arm (e.g. franka_panda_dyn) using the same algorithm --
# only the robot-specific numbers below change, the physics/search logic
# (aimed_speed, static_feasible, search_azimuth, ...) is robot-agnostic.
_ROBOT_NAME = "kinova_gen3_dyn"
_PROFILE = get_robot_profile(_ROBOT_NAME)
QD = np.array(_PROFILE.qd_max, dtype=float)
QN = np.array(_PROFILE.q_neutral, dtype=float)
N, EE, MASS, RAD = len(_PROFILE.joint_ids), _PROFILE.ee_link, 0.0577, 0.0327
URDF_REL_PATH = _PROFILE.urdf_rel_path

# Real actuator torque limits (published by the manufacturer -- Kinova
# ros_kortex gen3_macro.xacro for Gen3, franka_ros joint_limits.yaml for
# Panda), set per-robot via set_robot() below.
TAU_MAX = np.array(_PROFILE.tau_max, dtype=float)

# Roll/twist joints: base + every other joint going out the chain (idx 0,2,4,6
# for a 7-DOF alternating roll-pitch-roll arm). Frozen at qd=0 during the
# throw -- see module docstring. Verified numerically for both Gen3 and Panda
# (joint-axis vs base->EE-vector angle: pitch joints 1,3,5 sit at ~90deg,
# roll joints 0,2,4,6 sit far from 90deg) -- not just assumed from the name.
_ROLL_IDX = (0, 2, 4, 6)

# Total non-fixed DOF count of the loaded URDF, set by _set_n_full() once the
# body exists. Some arms (Panda) carry extra PASSIVE dofs beyond the N=7
# controlled joints -- e.g. 2 gripper-finger prismatic joints -- and PyBullet's
# calculateInverseDynamics/calculateJacobian require FULL-length q/qd/qdd
# vectors sized to ALL non-fixed joints, not just the ones we command. Our
# N controlled joints are always the URDF's first N non-fixed joints (verified
# for both Gen3 -- N_FULL==N, no extras -- and Panda -- fingers come after,
# at higher joint indices, so they land at DOF positions N.. in the full
# vector), so padding with zeros at the tail is exact, not an approximation.
N_FULL = None


def _set_n_full(arm):
    global N_FULL
    n_full = 0
    for j in range(p.getNumJoints(arm)):
        if p.getJointInfo(arm, j)[2] != p.JOINT_FIXED:
            n_full += 1
    N_FULL = n_full


def _pad(q):
    """Zero-pad an N-length joint vector to the URDF's full non-fixed DOF
    count (no-op when the arm has no passive extra joints, e.g. Gen3)."""
    if N_FULL is None or N_FULL == N:
        return list(q)
    full = [0.0] * N_FULL
    full[:N] = list(q)
    return full


def set_robot(robot_name):
    """Switch the module-level robot-specific constants (QD, QN, N, EE,
    TAU_MAX, URDF_REL_PATH) to a different profile. Call before search()."""
    global _ROBOT_NAME, _PROFILE, QD, QN, N, EE, TAU_MAX, URDF_REL_PATH, N_FULL
    _ROBOT_NAME = robot_name
    _PROFILE = get_robot_profile(robot_name)
    QD = np.array(_PROFILE.qd_max, dtype=float)
    QN = np.array(_PROFILE.q_neutral, dtype=float)
    N, EE = len(_PROFILE.joint_ids), _PROFILE.ee_link
    TAU_MAX = np.array(_PROFILE.tau_max, dtype=float)
    URDF_REL_PATH = _PROFILE.urdf_rel_path
    N_FULL = None


def aimed_speed(J, d, qd_max, freeze_roll=True):
    """max s s.t. J qd = s d_hat, |qd_i|<=qd_max, optionally qd[roll]=0 for
    every roll/twist joint (base + joints 3,5,7) so only the pitch joints
    (perpendicular to the swing plane) carry throw velocity."""
    d = np.asarray(d, dtype=float)
    d = d / np.linalg.norm(d)
    n = len(qd_max)
    c = np.zeros(n + 1)
    c[-1] = -1.0
    A_eq = np.hstack([np.asarray(J, dtype=float), -d.reshape(3, 1)])
    bounds = [(-qd_max[i], qd_max[i]) for i in range(n)] + [(0, None)]
    if freeze_roll:
        for i in _ROLL_IDX:
            bounds[i] = (0.0, 0.0)
    r = linprog(c, A_eq=A_eq, b_eq=np.zeros(3), bounds=bounds, method="highs")
    if not r.success:
        return 0.0, None
    return float(r.x[-1]), r.x[:n]


def windup_within_limits(q_release, qd_release, t_throw, lo, hi):
    """True iff the monotonic-windup cock stays inside the joint limits."""
    q_windup = np.asarray(q_release) - np.asarray(qd_release) * (t_throw / 2.0)
    return bool(np.all(q_windup >= lo - 1e-9) and np.all(q_windup <= hi + 1e-9))


def ballistic_range(pos, vel, mass=MASS, radius=RAD):
    x = np.asarray(pos, dtype=float).copy()
    v = np.asarray(vel, dtype=float).copy()
    dt = 0.002
    for _ in range(5000):
        v = v + _ball_accel(x, v, mass, radius, np.zeros(3)) * dt
        x = x + v * dt
        if x[2] <= 0 and v[2] < 0:
            break
    return float(np.hypot(x[0] - pos[0], x[1] - pos[1])), x


def _fkj(arm, q):
    for j in range(N):
        p.resetJointState(arm, j, q[j])
    pos = np.array(p.getLinkState(arm, EE, computeForwardKinematics=True)[4])
    q_full = _pad(q)
    jl, _ = p.calculateJacobian(arm, EE, [0, 0, 0], q_full, [0.] * len(q_full), [0.] * len(q_full))
    return pos, np.array(jl)[:, :N]


def static_feasible(arm, q, tau_max=None):
    """True iff the posture can be held still under gravity alone within the
    real actuator torque limits (calculateInverseDynamics, zero vel/accel)."""
    if tau_max is None:
        tau_max = TAU_MAX
    q_full = _pad(q)
    tau = np.array(p.calculateInverseDynamics(arm, q_full, [0.] * len(q_full), [0.] * len(q_full)))
    return bool(np.all(np.abs(tau[:N]) <= tau_max))


# A loose absolute forward-reach floor alone is not enough: range-maximization
# still finds near-vertical "flagpole" shapes that just barely clear it (verified:
# a shape at 78deg arm elevation, reach 0.204m, cleared a 0.20m floor). Cap the
# ARM'S OWN elevation angle directly -- the angle of the release point above
# horizontal, as seen from the base -- not just a minimum distance.
_MIN_FORWARD_REACH = 0.25
_MAX_ARM_ELEV_DEG = 50.0
_MIN_ARM_ELEV_DEG = 5.0


def is_natural_posture(pos, azimuth):
    """Reject postures that are too close to vertical (flagpole) or reach
    behind/beside the target instead of toward it."""
    forward = pos[0] * np.cos(azimuth) + pos[1] * np.sin(azimuth)
    if forward < _MIN_FORWARD_REACH:
        return False
    reach = np.hypot(pos[0], pos[1])
    if reach < 1e-6:
        return False
    arm_elev = np.degrees(np.arctan2(pos[2], reach))
    return _MIN_ARM_ELEV_DEG <= arm_elev <= _MAX_ARM_ELEV_DEG


_WINDUP_TAU_MARGIN = 0.90  # leave headroom for the real dynamic trajectory


def windup_path_feasible(arm, q_release, qd_release, t_throw, tau_max=None,
                         margin=_WINDUP_TAU_MARGIN, n_samples=5):
    """A candidate's RELEASE pose can be statically feasible while the
    NEUTRAL->WINDUP swing still passes through a worse intermediate gravity
    configuration (verified: e.g. shoulder sweeping neutral 22deg -> cocked
    90deg peaks ~39Nm mid-swing even though both endpoints are fine). Sample
    the straight-line neutral->windup path and require every point within
    margin*tau_max -- a real check, not just the two endpoints."""
    if tau_max is None:
        tau_max = TAU_MAX
    q_windup = np.asarray(q_release) - np.asarray(qd_release) * (t_throw / 2.0)
    q_windup = np.clip(q_windup, -np.pi, np.pi)
    for frac in np.linspace(0.0, 1.0, n_samples):
        q_mid = QN + frac * (q_windup - QN)
        q_full = _pad(q_mid)
        tau = np.array(p.calculateInverseDynamics(arm, q_full, [0.] * len(q_full), [0.] * len(q_full)))
        if np.any(np.abs(tau[:N]) > margin * tau_max):
            return False
    return True


_RELEASE_DYN_MARGIN = 0.99  # deliberately tighter than windup's 0.90: this
# checks a specific, well-understood failure (Coriolis torque exactly at
# release velocity), and the real crashes measured this session were at
# ratio >= 1.00 (true limit), not below it -- a known-good, already-trained
# candidate sits at 0.986, so 0.90 here would reject empirically-safe cases
# for no real reason. 0.99 rejects the actual crash regime, admits the rest.


def release_dynamics_feasible(arm, q_release, qd_release, tau_max=None,
                              margin=_RELEASE_DYN_MARGIN):
    """windup_path_feasible only samples the NEUTRAL->WINDUP path at near-zero
    velocity (static torque). It says nothing about whether the arm can
    actually SUSTAIN qd_release at q_release -- Coriolis/centripetal torque
    depends on velocity squared and does NOT shrink when the throw ramp is
    stretched (verified: a training run stretched dt_throw x2.99 over 6
    iterations and a release state stayed pinned at 39.5/39.0 Nm, unmoved --
    the offending term is velocity-dependent, not acceleration-dependent, so
    ramp duration can't fix it). Check inverse dynamics AT the release
    state directly (qdd=0, i.e. "just reached qd_release, no residual
    accel" -- the dominant term at the end of a cubic_to_velocity ramp)."""
    if tau_max is None:
        tau_max = TAU_MAX
    q_full = _pad(q_release)
    qd_full = _pad(qd_release)
    tau = np.array(p.calculateInverseDynamics(
        arm, q_full, qd_full, [0.] * len(q_full)))
    return bool(np.all(np.abs(tau[:N]) <= margin * tau_max))


_THROW_RAMP_MARGIN = 0.75


def throw_ramp_feasible(arm, q_release, qd_release, t_throw, tau_max=None,
                        margin=_THROW_RAMP_MARGIN, ball_mass=None, n_samples=40):
    """The REAL check, not an approximation: release_dynamics_feasible only
    evaluates torque at the terminal instant (qdd=0). Two separate real
    training crashes (peak ratio 1.02, then 1.01) happened at mid-ramp points
    this missed -- the cubic_to_velocity throw trajectory has nonzero qdd
    everywhere except possibly one instant, and inertial (M(q)*qdd) torque
    adds to the Coriolis term measured at the endpoint. Build the EXACT same
    cubic plan_throw uses (q_windup -> q_release, ending at qd_release) and
    sample q/qd/qdd across it with the real inverse-dynamics call, mirroring
    ArmController._throw_peak_torque_ratio exactly so a candidate that passes
    here is guaranteed to pass at runtime (mod ball wrench added by hand,
    since payload compensation happens in ArmController.step, not
    calculateInverseDynamics)."""
    if tau_max is None:
        tau_max = TAU_MAX
    if ball_mass is None:
        ball_mass = MASS
    q_release = np.asarray(q_release, dtype=float)
    qd_release = np.asarray(qd_release, dtype=float)
    q_windup = q_release - qd_release * (t_throw / 2.0)
    coeffs = _cubic_to_velocity(q_windup, q_release, qd_release, t_throw)
    for t in np.linspace(0.0, t_throw, n_samples):
        q, qd, qdd = _eval_cubic(coeffs, t, with_accel=True)
        q_full, qd_full, qdd_full = _pad(q), _pad(qd), _pad(qdd)
        tau = np.array(p.calculateInverseDynamics(arm, q_full, qd_full, qdd_full))[:N]
        # approximate ball-holding wrench (arm carries the ball for the
        # whole windup+throw, released only at the end) via the release
        # posture's Jacobian -- consistent with the payload-compensation
        # term ArmController.step adds for the real controller.
        _, J = _fkj(arm, q)
        tau_ball = J.T @ (ball_mass * np.array([0.0, 0.0, 9.81]))
        if np.any(np.abs(tau + tau_ball) > margin * tau_max):
            return False
    return True


_FOLLOW_MARGIN = 0.85
_FOLLOW_DUR_CANDIDATES = np.array([0.6, 0.9, 1.3, 1.8, 2.2, 2.6, 3.2, 4.0])


def follow_through_feasible(arm, q_release, qd_release, tau_max=None,
                            margin=_FOLLOW_MARGIN, q_end=None):
    """Every earlier torque check in this module validates windup and throw
    -- nothing validated whether the arm can actually RECOVER after release
    (decelerate from qd_release, at whatever extreme posture the search
    picked for aim/speed, back to neutral). Real training used exactly such
    a posture and plan_throw's own runtime check (mirrored here) flagged its
    follow-through at 1.15-3.2x tau_max -- a trajectory that would fault
    real hardware, not a rendering artifact ("collapses as soon as it lets
    go of the ball" was the correct visual read of an infeasible commanded
    motion). The mirrored ArmController check found peak-torque-ratio vs.
    duration is NOT monotonic (bottoms out then rises again) -- growing a
    single duration can walk past the true minimum. Sample a spread of
    candidate durations directly instead of growing one guess; feasible if
    ANY of them clears margin."""
    if tau_max is None:
        tau_max = TAU_MAX
    if q_end is None:
        q_end = QN
    q_release = np.asarray(q_release, dtype=float)
    qd_release = np.asarray(qd_release, dtype=float)
    for dur in _FOLLOW_DUR_CANDIDATES:
        coeffs = _cubic_from_velocity(q_release, qd_release, q_end, dur)
        worst = 0.0
        for t in np.linspace(0.0, dur, 30):
            q, qd, qdd = _eval_cubic(coeffs, t, with_accel=True)
            q_full, qd_full, qdd_full = _pad(q), _pad(qd), _pad(qdd)
            tau = np.array(p.calculateInverseDynamics(arm, q_full, qd_full, qdd_full))[:N]
            worst = max(worst, float(np.max(np.abs(tau) / tau_max)))
        if worst <= margin:
            return True
    return False


_MAX_JOINT_DELTA_DEG = 25.0  # continuity cap between adjacent-azimuth postures


def search_azimuth(arm, lo, hi, azimuth, t_throw=1.1, seed_q=None,
                   max_joint_delta_deg=_MAX_JOINT_DELTA_DEG, wide=False):
    """Best posture for one azimuth: base = azimuth (fixed), shoulder-roll
    (idx2) varies as a static setup parameter (breaks the pitch-axis-collapse
    degeneracy), shoulder/elbow/wrist PITCH vary and carry all throw velocity.
    Filtered by real static torque feasibility, not a proxy metric.

    seed_q: previous azimuth's winning posture. Independent per-azimuth
    argmax-range search picks whatever discrete grid point wins locally, with
    no relation to its neighbors -- verified this produces speed cliffs (e.g.
    0.45 m/s at one azimuth, 0.15 m/s three degrees over) that break
    _optimized_release's nearest-azimuth interpolation and wrecked training
    accuracy (56cm mean error, was 4.5cm). When seed_q is given, restrict
    acceptance to candidates within max_joint_delta_deg of seed_q (per
    searched joint) so the table varies smoothly azimuth-to-azimuth; only
    fall back to the single closest-by-distance candidate if none clear that
    bar, to still prefer continuity over a stranded gap in the table.

    wide: use the wider/finer grid (+-60/12/12/15deg) that holds ~2x speed
    headroom (0.66 vs 0.25 m/s at az=0). ONLY safe for a single-azimuth
    search whose winner is then propagated by base-rotation (see
    build_table_by_rotation) -- running it independently per azimuth produces
    the speed-cliff problem described above. The dynamic Coriolis-at-release
    check is enabled in this mode (its violations were only ever observed on
    wide-grid candidates)."""
    if wide:
        grid_roll = np.deg2rad(np.arange(-60, 61, 20))
        grid2 = np.deg2rad(np.arange(-120, 121, 12))
        grid4 = np.deg2rad(np.arange(-160, -9, 12))
        grid6 = np.deg2rad(np.arange(-120, 121, 15))
    else:
        grid_roll = np.deg2rad(np.arange(-40, 41, 20))    # shoulder-roll, static only
        grid2 = np.deg2rad(np.arange(-90, 91, 15))        # shoulder pitch
        grid4 = np.deg2rad(np.arange(-150, -9, 15))       # elbow pitch
        grid6 = np.deg2rad(np.arange(-90, 91, 20))        # wrist pitch
    # Pure range-maximization degenerates to a near-flat (elev~0) push for a
    # torque-starved arm: release height dominates achievable speed, so the
    # range-optimal angle collapses toward horizontal (verified: monotonic
    # 6.8cm@0deg -> 2.6cm@70deg for this arm). That's technically a launch but
    # reads as "placing the ball", not throwing. Floor elevation at 15deg so
    # every accepted candidate is a recognizable upward-arcing throw, even at
    # a real range cost.
    elevs = np.deg2rad(np.arange(15, 71, 10))
    max_delta = np.deg2rad(max_joint_delta_deg)
    candidates = []
    for j3 in grid_roll:
        for s2 in grid2:
            for s4 in grid4:
                for s6 in grid6:
                    q = np.array([azimuth, s2, j3, s4, 0.0, s6, 0.0])
                    if not static_feasible(arm, q):
                        continue
                    pos, J = _fkj(arm, q)
                    if pos[2] < 0.15:
                        continue
                    if not is_natural_posture(pos, azimuth):
                        continue
                    for elev in elevs:
                        d = np.array([np.cos(elev) * np.cos(azimuth),
                                      np.cos(elev) * np.sin(azimuth),
                                      np.sin(elev)])
                        s, qd = aimed_speed(J, d, QD, freeze_roll=True)
                        if qd is None or s < 1e-3:
                            continue
                        if not windup_within_limits(q, qd, t_throw, lo, hi):
                            continue
                        if not windup_path_feasible(arm, q, qd, t_throw):
                            continue
                        # Coriolis-at-release check: only needed on the wide
                        # grid (its violations were only ever observed there);
                        # on the narrow grid it starved azimuths (10/23) for
                        # no accuracy benefit on an already-validated grid.
                        if wide and not release_dynamics_feasible(arm, q, qd):
                            continue
                        rng, _ = ballistic_range(pos, s * d)
                        candidates.append({
                            "range": rng, "q": q.copy(), "qd": qd.copy(),
                            "elev_deg": float(np.degrees(elev)), "speed": s,
                            "azimuth_deg": float(np.degrees(azimuth)),
                        })
    if not candidates:
        return None
    if seed_q is None:
        return max(candidates, key=lambda c: c["range"])

    def joint_dist(c):
        return float(np.max(np.abs(c["q"][(1, 2, 3, 5),] - seed_q[(1, 2, 3, 5),])))

    within = [c for c in candidates if joint_dist(c) <= max_delta]
    if within:
        return max(within, key=lambda c: c["range"])
    return min(candidates, key=joint_dist)


def build_table_by_rotation(e0, azimuth_deg_grid=None):
    """Propagate ONE verified az=0 posture across the whole azimuth wedge by
    base-rotation. This is the TossingBot / MC-PILOT (Eq. 5) formulation: a
    fixed release state in the arm's sagittal plane, aimed purely by base
    azimuth, with the policy controlling only speed.

    Verified numerically (1e-6): adding -az to the base joint angle rotates
    the whole downstream chain (and thus J qd) by Rz(+az) -- the MINUS is
    because this URDF's base joint measures opposite the world-z rotation
    sense (checked empirically, not assumed from the axis field). Gravity is
    z-symmetric, so every torque-feasibility property of the az=0 entry
    (static, windup-path, Coriolis-at-release) is preserved exactly at every
    azimuth -- also spot-verified, not assumed.

    By construction the table is perfectly uniform in speed/elevation/
    geometry -- the azimuth-discontinuity failure mode of independent
    per-azimuth search (37cm training error) cannot occur."""
    if azimuth_deg_grid is None:
        azimuth_deg_grid = np.arange(-33.0, 33.1, 3.0)
    az0 = np.deg2rad(float(e0.get("azimuth_deg", 0.0)))   # e0's OWN heading
    table = []
    for az_deg in azimuth_deg_grid:
        az = np.deg2rad(az_deg) - az0     # rotation relative to e0's heading
        q = np.array(e0["q"], dtype=float).copy()
        q[0] -= az
        entry = {
            "range": e0["range"], "q": q, "qd": np.array(e0["qd"]).copy(),
            "elev_deg": e0["elev_deg"], "speed": e0["speed"],
            "azimuth_deg": float(az_deg),
            # marks the base-angle convention (q[0] = q0[0] - az, verified
            # sign) so _optimized_release can recover q0 and aim exactly
            "rotation_built": True,
        }
        if "v_dir" in e0:
            # exact release-velocity direction, rotated with the posture --
            # lets the runtime skip the LP entirely (qd scales linearly).
            ca, sa = np.cos(az), np.sin(az)
            Rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
            entry["v_dir"] = Rz @ np.asarray(e0["v_dir"], dtype=float)
        table.append(entry)
    return table


def search_release_state(t_throw=1.1):
    """Release-state-first overhead search (design doc 2026-07-22): maximize
    LANDING DISTANCE FROM BASE over sagittal postures (roll joints = 0) with
    ONLY hardware constraints -- joint limits, qd_max, 80% torque incl. ball
    wrench at the actual release velocity, windup-within-limits, windup-path
    torque. No elevation caps, no reach floors: those cosmetic filters are
    what strangled the old pipeline to 0.66 m/s when 1.89 m/s is feasible.

    With rolls pinned the pitch axes are parallel, so achievable velocities
    span ONE plane -- slightly tilted off x-z by the URDF's y-offsets. The LP
    direction must lie IN that plane (aim the plane, don't fight it): d is
    built from the plane basis at each posture."""
    cid = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    arm = p.loadURDF(pybullet_data.getDataPath() + "/" + URDF_REL_PATH,
                     useFixedBase=True)
    _set_n_full(arm)
    lo, hi = [], []
    for j in range(N):
        ji = p.getJointInfo(arm, j)
        l, h = ji[8], ji[9]
        if l >= h:
            l, h = -np.pi, np.pi
        lo.append(l)
        hi.append(h)
    lo, hi = np.array(lo), np.array(hi)

    g2 = np.deg2rad(np.arange(-120, 121, 5))
    g4 = np.deg2rad(np.arange(-147, 148, 5))
    g6 = np.deg2rad(np.arange(-120, 121, 8))
    best = None
    for s2 in g2:
        for s4 in g4:
            for s6 in g6:
                q = np.array([0.0, s2, 0.0, s4, 0.0, s6, 0.0])
                pos, J = _fkj(arm, q)
                if pos[2] < 0.15:
                    continue
                u1, u3 = J[:, 1], J[:, 3]
                nvec = np.cross(u1, u3)
                nn = np.linalg.norm(nvec)
                if nn < 1e-8:
                    continue
                nvec /= nn
                h_ip = np.cross(nvec, [0.0, 0.0, 1.0])
                hn = np.linalg.norm(h_ip)
                if hn < 1e-9:
                    continue
                h_ip /= hn
                v_ip = np.cross(nvec, h_ip)
                if v_ip[2] < 0:
                    v_ip = -v_ip
                for elev_deg in range(5, 46, 5):
                    th = np.deg2rad(elev_deg)
                    if abs(v_ip[2]) < np.sin(th):
                        continue
                    spsi = np.sin(th) / v_ip[2]
                    cpsi = np.sqrt(max(0.0, 1.0 - spsi * spsi))
                    for sgn in (1.0, -1.0):
                        d = sgn * cpsi * h_ip + spsi * v_ip
                        s, qd = aimed_speed(J, d, QD, freeze_roll=True)
                        if qd is None or s < 0.5:
                            continue
                        if not windup_within_limits(q, qd, t_throw, lo, hi):
                            continue
                        if not windup_path_feasible(arm, q, qd, t_throw):
                            continue
                        # Real full-ramp check (samples q/qd/qdd across the
                        # actual cubic_to_velocity trajectory, mirroring
                        # ArmController._throw_peak_torque_ratio exactly) --
                        # NOT the release-only qdd=0 approximation, which
                        # missed two real training crashes (1.02x, then
                        # 1.01x of limit) at points margin-tuning couldn't
                        # reliably predict since it wasn't sampling them.
                        if not throw_ramp_feasible(arm, q, qd, t_throw):
                            continue
                        # Windup and throw feasible does NOT mean the arm can
                        # recover afterward -- validated separately (real
                        # crash found: a candidate that passed every check
                        # above still needed 44.8/39.0 Nm, 1.15x limit, to
                        # decelerate back to neutral post-release).
                        if not follow_through_feasible(arm, q, qd):
                            continue
                        # ballistic_range's landing_pos IS the real landing
                        # point -- distance from base is hypot of THAT, not
                        # hypot(release_pos) + range (wrong whenever release
                        # point and throw direction aren't collinear through
                        # the origin, which they aren't here: verified this
                        # bug inflated every "beyond reach" number reported
                        # this session by ~1.6x, e.g. claimed 1.07m when the
                        # real landing distance from base was 0.67m).
                        rng, landing_pos = ballistic_range(pos, s * d)
                        land = float(np.hypot(landing_pos[0], landing_pos[1]))
                        if best is None or land > best["land"]:
                            best = {"land": land, "range": rng, "q": q.copy(),
                                    "qd": np.array(qd), "speed": s,
                                    "elev_deg": float(elev_deg),
                                    "v_dir": d.copy(), "azimuth_deg": 0.0,
                                    "release_pos": pos.copy()}
    p.disconnect(cid)
    if best is None:
        return []
    # label azimuth by the actual horizontal heading of v_dir so the table's
    # azimuth_deg means "where the ball actually flies" for entry 0
    best["azimuth_deg"] = float(np.degrees(np.arctan2(best["v_dir"][1],
                                                      best["v_dir"][0])))
    print(f"release state: land={best['land']:.2f} m  speed={best['speed']:.2f} m/s  "
          f"elev={best['elev_deg']:.0f}  release={np.round(best['release_pos'],3)}  "
          f"plane_az={best['azimuth_deg']:.1f} deg")
    return build_table_by_rotation(best)


def search(azimuth_deg_grid=None, t_throw=1.1):
    """Azimuth->posture table over the training wedge. Returns list of dicts."""
    if azimuth_deg_grid is None:
        azimuth_deg_grid = np.arange(-33.0, 33.1, 3.0)
    cid = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    arm = p.loadURDF(pybullet_data.getDataPath() + "/" + URDF_REL_PATH,
                     useFixedBase=True)
    _set_n_full(arm)
    lo, hi = [], []
    for j in range(N):
        ji = p.getJointInfo(arm, j)
        l, h = ji[8], ji[9]
        if l >= h:
            l, h = -np.pi, np.pi
        lo.append(l)
        hi.append(h)
    lo, hi = np.array(lo), np.array(hi)

    # Independent per-azimuth search (no seed_q continuity chaining -- see
    # search_azimuth's docstring/comments: that machinery is needed for the
    # wider grid, not this one, and stacking it here starved azimuths for no
    # accuracy benefit).
    table = []
    for az_deg in azimuth_deg_grid:
        b = search_azimuth(arm, lo, hi, np.deg2rad(az_deg), t_throw)
        if b is not None:
            table.append(b)
    p.disconnect(cid)
    return table


def search_and_rotate(t_throw=1.1):
    """Single wide-grid search at az=0 (max speed, all feasibility filters
    incl. Coriolis-at-release), then propagate by base-rotation. See
    build_table_by_rotation for why this is both the fastest AND the smooth/
    accurate way to cover the wedge."""
    cid = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    arm = p.loadURDF(pybullet_data.getDataPath() + "/" + URDF_REL_PATH,
                     useFixedBase=True)
    _set_n_full(arm)
    lo, hi = [], []
    for j in range(N):
        ji = p.getJointInfo(arm, j)
        l, h = ji[8], ji[9]
        if l >= h:
            l, h = -np.pi, np.pi
        lo.append(l)
        hi.append(h)
    lo, hi = np.array(lo), np.array(hi)
    e0 = search_azimuth(arm, lo, hi, 0.0, t_throw, wide=True)
    p.disconnect(cid)
    if e0 is None:
        return []
    return build_table_by_rotation(e0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="kinova_gen3_dyn")
    ap.add_argument("--out", default=None)
    ap.add_argument("--t_throw", type=float, default=1.1)
    ap.add_argument("--mode", choices=["overhead", "rotate", "independent"],
                    default="overhead",
                    help=("overhead (default): release-state-first search -- max "
                          "landing distance from base, sagittal overhead whip, "
                          "1.89 m/s class, beyond-reach targets (design doc "
                          "2026-07-22). rotate: legacy wide-grid range-max az=0 "
                          "search propagated by rotation. independent: legacy "
                          "narrow per-azimuth search."))
    args = ap.parse_args()
    if args.robot != _ROBOT_NAME:
        set_robot(args.robot)
    out_path = args.out or ("throw_pose_table.npy" if args.robot == "kinova_gen3_dyn"
                            else f"{args.robot}_throw_pose_table.npy")

    if args.mode == "overhead":
        table = search_release_state(t_throw=args.t_throw)
    elif args.mode == "rotate":
        table = search_and_rotate(t_throw=args.t_throw)
    else:
        table = search(t_throw=args.t_throw)
    np.save(out_path, np.array(table, dtype=object))
    print(f"saved {out_path}: {len(table)} azimuth entries")
    for e in table:
        roll_qd = np.max(np.abs(e["qd"][list(_ROLL_IDX)]))
        print(f"  az={e['azimuth_deg']:+6.1f}  speed={e['speed']:.2f} m/s  "
              f"range={e['range']*100:5.1f} cm  elev={e['elev_deg']:.0f}  "
              f"max|qd|/qdmax={np.max(np.abs(e['qd'])/QD):.2f}  "
              f"roll_qd(should be 0)={roll_qd:.4f}  j3_static={np.degrees(e['q'][2]):+.0f}")
