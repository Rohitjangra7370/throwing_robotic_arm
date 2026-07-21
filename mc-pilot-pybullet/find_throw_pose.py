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
import numpy as np
import pybullet as p
import pybullet_data
from scipy.optimize import linprog
from simulation_class.model import _ball_accel

QD = np.array([1.3963, 1.3963, 1.3963, 1.3963, 1.2218, 1.2218, 1.2218])
QN = np.array([-0.01, 0.382, -0.04, 1.46, -0.012, 0.731, 0.0])
N, EE, MASS, RAD = 7, 7, 0.0577, 0.0327

# Real Kinova Gen3 actuator limits (Kinovarobotics/ros_kortex gen3_macro.xacro):
# large actuators (joints 1-4) effort=39 Nm, small actuators (joints 5-7) effort=9 Nm.
TAU_MAX = np.array([39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0])

# Roll/twist joints: base (idx0), shoulder-roll (idx2), wrist-roll1 (idx4),
# wrist-roll2 (idx6). Frozen at qd=0 during the throw -- see module docstring.
_ROLL_IDX = (0, 2, 4, 6)


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
    jl, _ = p.calculateJacobian(arm, EE, [0, 0, 0], q.tolist(), [0.] * N, [0.] * N)
    return pos, np.array(jl)


def static_feasible(arm, q, tau_max=TAU_MAX):
    """True iff the posture can be held still under gravity alone within the
    real actuator torque limits (calculateInverseDynamics, zero vel/accel)."""
    tau = np.array(p.calculateInverseDynamics(arm, q.tolist(), [0.] * N, [0.] * N))
    return bool(np.all(np.abs(tau) <= tau_max))


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


_WINDUP_TAU_MARGIN = 0.85  # leave headroom for the real dynamic trajectory


def windup_path_feasible(arm, q_release, qd_release, t_throw, tau_max=TAU_MAX,
                         margin=_WINDUP_TAU_MARGIN, n_samples=5):
    """A candidate's RELEASE pose can be statically feasible while the
    NEUTRAL->WINDUP swing still passes through a worse intermediate gravity
    configuration (verified: e.g. shoulder sweeping neutral 22deg -> cocked
    90deg peaks ~39Nm mid-swing even though both endpoints are fine). Sample
    the straight-line neutral->windup path and require every point within
    margin*tau_max -- a real check, not just the two endpoints."""
    q_windup = np.asarray(q_release) - np.asarray(qd_release) * (t_throw / 2.0)
    q_windup = np.clip(q_windup, -np.pi, np.pi)
    for frac in np.linspace(0.0, 1.0, n_samples):
        q_mid = QN + frac * (q_windup - QN)
        tau = np.array(p.calculateInverseDynamics(arm, q_mid.tolist(), [0.] * N, [0.] * N))
        if np.any(np.abs(tau) > margin * tau_max):
            return False
    return True


def search_azimuth(arm, lo, hi, azimuth, t_throw=1.1):
    """Best posture for one azimuth: base = azimuth (fixed), shoulder-roll
    (idx2) varies as a static setup parameter (breaks the pitch-axis-collapse
    degeneracy), shoulder/elbow/wrist PITCH vary and carry all throw velocity.
    Filtered by real static torque feasibility, not a proxy metric."""
    grid_roll = np.deg2rad(np.arange(-40, 41, 20))    # shoulder-roll, static only
    grid2 = np.deg2rad(np.arange(-90, 91, 15))        # shoulder pitch
    grid4 = np.deg2rad(np.arange(-150, -9, 15))       # elbow pitch
    grid6 = np.deg2rad(np.arange(-90, 91, 20))        # wrist pitch
    elevs = np.deg2rad(np.arange(0, 71, 10))
    best = None
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
                        rng, _ = ballistic_range(pos, s * d)
                        if best is None or rng > best["range"]:
                            best = {"range": rng, "q": q.copy(), "qd": qd.copy(),
                                    "elev_deg": float(np.degrees(elev)), "speed": s,
                                    "azimuth_deg": float(np.degrees(azimuth))}
    return best


def search(azimuth_deg_grid=None, t_throw=1.1):
    """Azimuth->posture table over the training wedge. Returns list of dicts."""
    if azimuth_deg_grid is None:
        azimuth_deg_grid = np.arange(-33.0, 33.1, 3.0)
    cid = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    arm = p.loadURDF(pybullet_data.getDataPath() + "/kinova_gen3/gen3.urdf",
                     useFixedBase=True)
    lo, hi = [], []
    for j in range(N):
        ji = p.getJointInfo(arm, j)
        l, h = ji[8], ji[9]
        if l >= h:
            l, h = -np.pi, np.pi
        lo.append(l)
        hi.append(h)
    lo, hi = np.array(lo), np.array(hi)

    table = []
    for az_deg in azimuth_deg_grid:
        b = search_azimuth(arm, lo, hi, np.deg2rad(az_deg), t_throw)
        if b is not None:
            table.append(b)
    p.disconnect(cid)
    return table


if __name__ == "__main__":
    table = search()
    np.save("throw_pose_table.npy", np.array(table, dtype=object))
    print(f"saved throw_pose_table.npy: {len(table)} azimuth entries")
    for e in table:
        roll_qd = np.max(np.abs(e["qd"][list(_ROLL_IDX)]))
        print(f"  az={e['azimuth_deg']:+6.1f}  speed={e['speed']:.2f} m/s  "
              f"range={e['range']*100:5.1f} cm  elev={e['elev_deg']:.0f}  "
              f"max|qd|/qdmax={np.max(np.abs(e['qd'])/QD):.2f}  "
              f"roll_qd(should be 0)={roll_qd:.4f}  j3_static={np.degrees(e['q'][2]):+.0f}")
