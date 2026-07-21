"""
Hardware-valid aimed-throw pose search on the kinetic-chain principle.

  * Base joint j1 (vertical z-axis) sets AZIMUTH only and is held STILL during
    the throw: qd[0] = 0. It is NOT a speed source (grounding shows it adds ~0%).
  * Shoulder/elbow/wrist sweep the vertical plane; the release velocity VECTOR
    points exactly along the launch direction d (aimable).
  * Release-instant joint velocities are <= qd_max by LP construction; the
    monotonic windup (handled in plan_throw) keeps the whole stroke <= qd_max.

Search extended forward/up postures (base=0; azimuth applied at runtime), score
by drag ballistic range, and REJECT poses whose windup cock leaves joint limits.
"""
import numpy as np
import pybullet as p
import pybullet_data
from scipy.optimize import linprog
from simulation_class.model import _ball_accel

QD = np.array([1.3963, 1.3963, 1.3963, 1.3963, 1.2218, 1.2218, 1.2218])
QN = np.array([-0.01, 0.382, -0.04, 1.46, -0.012, 0.731, 0.0])
N, EE, MASS, RAD = 7, 7, 0.0577, 0.0327


def aimed_speed(J, d, qd_max, freeze_base=True):
    """max s s.t. J qd = s d_hat, |qd_i|<=qd_max, optional qd[0]=0."""
    d = np.asarray(d, dtype=float)
    d = d / np.linalg.norm(d)
    n = len(qd_max)
    c = np.zeros(n + 1)
    c[-1] = -1.0
    A_eq = np.hstack([np.asarray(J, dtype=float), -d.reshape(3, 1)])
    bounds = [(-qd_max[i], qd_max[i]) for i in range(n)] + [(0, None)]
    if freeze_base:
        bounds[0] = (0.0, 0.0)
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


def search(t_throw=1.1):
    cid = p.connect(p.DIRECT)
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

    grid = {
        2: np.deg2rad(np.arange(-70, 71, 8)),    # shoulder
        4: np.deg2rad(np.arange(-125, 1, 8)),    # elbow
        6: np.deg2rad(np.arange(-90, 91, 12)),   # wrist
    }
    best = None
    for s2 in grid[2]:
        for s4 in grid[4]:
            for s6 in grid[6]:
                q = np.array([0.0, s2, 0.0, s4, 0.0, s6, 0.0])
                pos, J = _fkj(arm, q)
                if pos[2] < 0.15:
                    continue
                for elev in np.deg2rad(np.arange(20, 56, 5)):
                    d = np.array([np.cos(elev), 0.0, np.sin(elev)])
                    s, qd = aimed_speed(J, d, QD, freeze_base=True)
                    if qd is None:
                        continue
                    if not windup_within_limits(q, qd, t_throw, lo, hi):
                        continue
                    rng, _ = ballistic_range(pos, s * d)
                    if best is None or rng > best["range"]:
                        best = {"range": rng, "q": q.copy(), "qd": qd.copy(),
                                "elev_deg": float(np.degrees(elev)), "speed": s}
    p.disconnect(cid)
    return best


if __name__ == "__main__":
    b = search()
    pose = {"q": b["q"], "qd": b["qd"], "elev_deg": b["elev_deg"], "speed": b["speed"]}
    np.save("throw_pose.npy", pose)
    print(f"saved throw_pose.npy: speed={b['speed']:.3f} m/s  "
          f"range={b['range']*100:.1f} cm  elev={b['elev_deg']:.0f}deg")
    print(f"  q  = {np.round(b['q'], 3)}")
    print(f"  qd = {np.round(b['qd'], 3)}  |qd|/qd_max = {np.round(np.abs(b['qd'])/QD, 2)}")
    print(f"  base qd[0] = {b['qd'][0]:.4f}  (must be 0)")
