# Kinetic-Chain Throw Pose Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Gen3 aimed-throw pose on the kinetic-chain principle (base = azimuth only, `qd[0]=0`; shoulder/elbow sweep the vertical plane) so the entire windup→throw motion stays within `qd_max` and is executable on real hardware.

**Architecture:** A frozen-base aimed LP finds an extended posture whose release-instant joint velocities are all ≤ `qd_max`. A monotonic windup — cocking the joints back by `qd_release·(t_throw/2)` — makes the throw a linear velocity ramp `0→qd_release`, so no joint ever exceeds `qd_max` mid-stroke. This feeds the already-fixed `_optimized_release`/`_simulate_pybullet` path.

**Tech Stack:** Python 3.11, NumPy, SciPy (`linprog`), PyBullet (DIRECT), pytest.

## Global Constraints

- All work in `mc-pilot-pybullet/`; run `python`/`pytest` from that directory (relative imports assume CWD = variant root).
- Do NOT modify `MC-PILCO/` (vendored upstream).
- Kinova Gen3 hard joint-velocity limits (`qd_max`): `[1.3963, 1.3963, 1.3963, 1.3963, 1.2218, 1.2218, 1.2218]` rad/s — verbatim from `robot_arm/arm_controller.py` `_IIWA_QD_MAX`/profile.
- Neutral pose `q_neutral` (kinova): `[-0.01, 0.382, -0.04, 1.46, -0.012, 0.731, 0.0]`.
- The neutral-reset fix in `_optimized_release` (`model_pybullet.py`) is already applied — do not remove it.
- Commit each task scoped to its own files (`git add <exact paths>`), never `git add -A` — the working tree carries unrelated uncommitted changes.

---

### Task 1: Frozen-base pose search (`find_throw_pose.py` rewrite)

**Files:**
- Modify (rewrite): `mc-pilot-pybullet/find_throw_pose.py`
- Test: `mc-pilot-pybullet/tests/test_throw_pose_search.py`

**Interfaces:**
- Produces:
  - `aimed_speed(J, d, qd_max, freeze_base=True) -> (float, np.ndarray | None)` — LP `max s s.t. J·qd = s·d̂, |qd_i|≤qd_max`; when `freeze_base`, also `qd[0]=0`. Returns `(s, qd)` or `(0.0, None)`.
  - `windup_within_limits(q_release, qd_release, t_throw, lo, hi) -> bool` — True iff `q_release − qd_release·(t_throw/2)` is inside `[lo, hi]` element-wise.
  - `ballistic_range(pos, vel, mass, radius) -> (float, np.ndarray)` — drag ballistic horizontal range and landing point (uses `simulation_class.model._ball_accel`).
  - `search(t_throw=1.1) -> dict` — grid search; returns best `{q, qd, elev_deg, speed}`.
  - `__main__` — runs `search()`, saves `throw_pose.npy`, prints summary.

- [ ] **Step 1: Write the failing test**

Create `mc-pilot-pybullet/tests/test_throw_pose_search.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mc-pilot-pybullet && python -m pytest tests/test_throw_pose_search.py -v`
Expected: FAIL — `ImportError: cannot import name 'aimed_speed'` (current `find_throw_pose.py` runs a search at import and has no such function).

- [ ] **Step 3: Rewrite `find_throw_pose.py`**

Replace the whole file with importable functions (search runs only under `__main__`):

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mc-pilot-pybullet && python -m pytest tests/test_throw_pose_search.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Regenerate the pose artifact and eyeball it**

Run: `cd mc-pilot-pybullet && python find_throw_pose.py 2>&1 | grep -v b3Warning | tr '\r' '\n'`
Expected: prints `base qd[0] = 0.0000`, `|qd|/qd_max` all ≤ 1.0, speed ≈ 1.5–2.1 m/s, range ≈ 70–110 cm.

- [ ] **Step 6: Commit**

```bash
cd mc-pilot-pybullet
git add find_throw_pose.py tests/test_throw_pose_search.py throw_pose.npy
git commit -m "Frozen-base aimed pose search (kinetic-chain principle)"
```

---

### Task 2: Freeze the base in `_optimized_release`'s LP

**Files:**
- Modify: `mc-pilot-pybullet/simulation_class/model_pybullet.py` (`_optimized_release`, the `linprog` bounds)
- Test: `mc-pilot-pybullet/tests/test_optimized_release_freezebase.py`

**Interfaces:**
- Consumes: `PyBulletThrowingSystem._optimized_release(arm, v_cmd)` (existing, returns `(release_pos, q_release, qd_release, v_dir)`).
- Produces: same signature; new invariant `qd_release[0] == 0`.

- [ ] **Step 1: Write the failing test**

Create `mc-pilot-pybullet/tests/test_optimized_release_freezebase.py`:

```python
import numpy as np
import pybullet as p
import pybullet_data
import pytest
from robot_arm.arm_controller import ArmController
from robot_arm.robot_profiles import get_robot_profile
from simulation_class.model_pybullet import PyBulletThrowingSystem


class _NoWind:
    def reset(self): pass
    def __call__(self, t): return np.zeros(3)


@pytest.fixture
def setup():
    client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(0.02, physicsClientId=client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    prof = get_robot_profile("kinova_gen3_dyn")
    urdf = pybullet_data.getDataPath() + "/" + prof.urdf_rel_path
    arm = ArmController(client, urdf, robot_name="kinova_gen3_dyn")
    q_pose = np.array([-0.15, 1.0, -1.0, 1.2, 0.5, -0.5, 0.3])  # synthetic posture
    sysm = PyBulletThrowingSystem(robot_name="kinova_gen3_dyn", opt_posture=q_pose,
                                  opt_launch_deg=25.0, wind_model=_NoWind(),
                                  t_w=0.5, t_r=1.6)
    yield sysm, arm
    p.disconnect(client)


def test_optimized_release_pins_base_velocity(setup):
    sysm, arm = setup
    sysm._cur_target_xy = np.array([0.6, 0.3])       # off-axis target
    v_cmd = np.array([0.5, 0.25, 0.3])
    _, _, qd_release, _ = sysm._optimized_release(arm, v_cmd)
    assert abs(qd_release[0]) < 1e-9                  # base joint held still
    assert np.all(np.abs(qd_release) <= np.array(arm._qd_max) + 1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mc-pilot-pybullet && python -m pytest tests/test_optimized_release_freezebase.py -v`
Expected: FAIL — `assert abs(qd_release[0]) < 1e-9` fails (current LP lets the base saturate).

- [ ] **Step 3: Add the frozen-base bound**

In `simulation_class/model_pybullet.py`, inside `_optimized_release`, find the LP bounds line:

```python
        bnds = [(-arm._qd_max[i], arm._qd_max[i]) for i in range(nq)] + [(0, None)]
```

Replace with:

```python
        bnds = [(-arm._qd_max[i], arm._qd_max[i]) for i in range(nq)] + [(0, None)]
        bnds[0] = (0.0, 0.0)   # base joint = azimuth only, held still: qd[0] = 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mc-pilot-pybullet && python -m pytest tests/test_optimized_release_freezebase.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
cd mc-pilot-pybullet
git add simulation_class/model_pybullet.py tests/test_optimized_release_freezebase.py
git commit -m "Pin base velocity to zero in _optimized_release LP (aim by azimuth only)"
```

---

### Task 3: Monotonic windup in `plan_throw`

**Files:**
- Modify: `mc-pilot-pybullet/robot_arm/arm_controller.py` (`plan_throw`)
- Test: `mc-pilot-pybullet/tests/test_monotonic_windup.py`

**Interfaces:**
- Consumes: `plan_throw(v_cmd, release_pos, t_w, t_r, T, q_release_override=None, qd_release_override=None)` (existing).
- Produces: new keyword `monotonic_windup=False`. When True, `q_windup = q_release − qd_release·(t_throw/2)` (recomputed whenever the torque loop stretches `t_throw`), and `t_windup` is grown so the neutral→windup peak velocity `1.5·max|q_windup−q_neutral|/t_windup ≤ min(qd_max)`. The throw phase is then the linear ramp `0→qd_release`.

- [ ] **Step 1: Write the failing test**

Create `mc-pilot-pybullet/tests/test_monotonic_windup.py`:

```python
import numpy as np
import pybullet as p
import pybullet_data
import pytest
from robot_arm.arm_controller import ArmController, _eval_cubic
from robot_arm.robot_profiles import get_robot_profile


@pytest.fixture
def arm():
    client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(0.02, physicsClientId=client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    prof = get_robot_profile("kinova_gen3_dyn")
    urdf = pybullet_data.getDataPath() + "/" + prof.urdf_rel_path
    a = ArmController(client, urdf, robot_name="kinova_gen3_dyn")
    yield a, prof
    p.disconnect(client)


def _peak_qd(coeffs, phase, dt):
    peak = 0.0
    for t in np.linspace(0.0, dt, 120):
        _, qd, _ = _eval_cubic(coeffs[phase], t, with_accel=True)
        peak = max(peak, float(np.max(np.abs(qd))))
    return peak


def test_monotonic_windup_keeps_whole_stroke_under_qd_max(arm):
    controller, prof = arm
    qd_max = np.array(prof.qd_max)
    q_release = np.array(prof.q_neutral) + np.array([0.0, 0.6, -0.5, 0.7, 0.3, -0.4, 0.2])
    qd_release = np.array([0.0, 1.0, -0.1, 1.0, 0.9, 0.8, -0.9])  # <= qd_max, base 0
    v_cmd = np.array([0.6, 0.0, 0.3])
    coeffs, _, _, _ = controller.plan_throw(
        v_cmd, np.array(prof.default_release_pos), 0.5, 1.6, 3.0,
        q_release_override=q_release, qd_release_override=qd_release,
        monotonic_windup=True,
    )
    dt_throw = coeffs["t_r"] - coeffs["t_w"]
    dt_windup = coeffs["t_w"]
    # throw phase never exceeds qd_release (linear ramp) -> <= qd_max
    assert _peak_qd(coeffs, "throw", dt_throw) <= np.max(qd_max) + 1e-6
    # windup phase also within qd_max
    assert _peak_qd(coeffs, "windup", dt_windup) <= np.max(qd_max) + 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mc-pilot-pybullet && python -m pytest tests/test_monotonic_windup.py -v`
Expected: FAIL — `plan_throw() got an unexpected keyword argument 'monotonic_windup'`.

- [ ] **Step 3: Implement `monotonic_windup` in `plan_throw`**

In `robot_arm/arm_controller.py`, change the `plan_throw` signature:

```python
    def plan_throw(self, v_cmd, release_pos, t_w=0.3, t_r=0.6, T=1.0,
                   q_release_override=None, qd_release_override=None,
                   monotonic_windup=False):
```

Replace the whole block that currently spans from the `if self._windup_delta is
not None:` windup-pose selection through the first `throw_coeffs = ...` line
(current lines ~249–265) with the following. This keeps correct ordering
(`dt_throw` is defined before the windup cock uses it):

```python
        dt_windup = t_w
        dt_throw = t_r - t_w
        follow_dur = T - t_r
        windup_time_scale = 1.0
        time_scale = 1.0

        def _windup_pose_and_time(dt_throw_local, dt_windup_local):
            if monotonic_windup:
                # Cock back by half the ballistic so the throw is a linear
                # velocity ramp 0 -> qd_release (peak = qd_release <= qd_max).
                qw = q_release - qd_release * (dt_throw_local / 2.0)
                qw = np.clip(qw, self._q_lo, self._q_hi)
                # Grow the neutral->windup time so its rest-to-rest cubic peak
                # (1.5*|dq|/t) stays within qd_max.
                span = np.max(np.abs(qw - self._q_neutral))
                min_tw = 1.5 * span / float(np.min(self._qd_max))
                return qw, max(dt_windup_local, min_tw)
            if self._windup_delta is not None:
                qw = self._q_neutral + self._windup_delta
            else:
                qw = self._q_neutral + (q_release - self._q_neutral) * (-0.5)
            return np.clip(qw, self._q_lo, self._q_hi), dt_windup_local

        q_windup, dt_windup = _windup_pose_and_time(dt_throw, dt_windup)
        q_follow = self._q_neutral.copy()
        windup_coeffs = _cubic_rest_to_rest(self._q_neutral, q_windup, dt_windup)
        throw_coeffs = _cubic_to_velocity(q_windup, q_release, qd_release, dt_throw)
```

Note: this replaces the original `dt_windup = t_w` / `dt_throw = ...` /
`follow_dur = ...` / `windup_time_scale` / `time_scale` / `windup_coeffs` /
`throw_coeffs` assignments too — they are folded into the block above, so do not
leave duplicates. Then in the throw torque-feasibility loop, recompute the windup
after each stretch. Find:

```python
            for _ in range(6):
                ratio, worst_tau = self._throw_peak_torque_ratio(throw_coeffs, dt_throw)
                if ratio <= 1.0:
                    break
                dt_throw *= 1.2
                time_scale *= 1.2
                throw_coeffs = _cubic_to_velocity(
                    q_windup, q_release, qd_release, dt_throw
                )
```

Replace the loop body's stretch branch so `q_windup` and `dt_windup` track the
stretched `dt_throw`, and rebuild the windup cubic too:

```python
            for _ in range(6):
                ratio, worst_tau = self._throw_peak_torque_ratio(throw_coeffs, dt_throw)
                if ratio <= 1.0:
                    break
                dt_throw *= 1.2
                time_scale *= 1.2
                if monotonic_windup:
                    q_windup, dt_windup = _windup_pose_and_time(dt_throw, dt_windup)
                    windup_coeffs = _cubic_rest_to_rest(
                        self._q_neutral, q_windup, dt_windup
                    )
                throw_coeffs = _cubic_to_velocity(
                    q_windup, q_release, qd_release, dt_throw
                )
```

Note: `windup_coeffs` and `throw_coeffs` are first built a few lines above from
`dt_windup`/`dt_throw`; because `_windup_pose_and_time` may have grown `dt_windup`
before those lines, ensure the initial `windup_coeffs = _cubic_rest_to_rest(...)`
and the windup torque loop use the updated `dt_windup`. (The windup torque loop
already reassigns `dt_windup`; monotonic sizing only raises its floor.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mc-pilot-pybullet && python -m pytest tests/test_monotonic_windup.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Guard against regressions in the existing throw path**

Run: `cd mc-pilot-pybullet && python -m pytest tests/test_time_scaling.py tests/test_setpoint_accel.py -v`
Expected: PASS (existing behaviour with `monotonic_windup=False` unchanged).

- [ ] **Step 6: Commit**

```bash
cd mc-pilot-pybullet
git add robot_arm/arm_controller.py tests/test_monotonic_windup.py
git commit -m "Monotonic kinetic-chain windup in plan_throw (linear 0->qd_release ramp)"
```

---

### Task 4: Wire opt_pose to the monotonic sweep + integration verification

**Files:**
- Modify: `mc-pilot-pybullet/simulation_class/model_pybullet.py` (`_simulate_pybullet`, the `plan_throw` call)
- Test: `mc-pilot-pybullet/tests/test_kinetic_chain_rollout.py`

**Interfaces:**
- Consumes: `plan_throw(..., monotonic_windup=True)` from Task 3; `_optimized_release` frozen base from Task 2; `throw_pose.npy` from Task 1.
- Produces: opt_pose rollouts that don't blow up, aim within ~3°, and hold every setpoint joint velocity ≤ `qd_max` across the whole trajectory.

- [ ] **Step 1: Write the failing test**

Create `mc-pilot-pybullet/tests/test_kinetic_chain_rollout.py`:

```python
import os
import numpy as np
import pytest
from simulation_class.model_pybullet import PyBulletThrowingSystem

POSE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "throw_pose.npy")


class _NoWind:
    def reset(self): pass
    def __call__(self, t): return np.zeros(3)


class _ConstPolicy:
    def __init__(self, s): self.s = s
    def __call__(self, s0, t): return np.array([self.s])


@pytest.fixture
def sysm():
    pose = np.load(POSE_PATH, allow_pickle=True).item()
    s = PyBulletThrowingSystem(robot_name="kinova_gen3_dyn",
                               opt_posture=np.array(pose["q"]),
                               opt_launch_deg=float(pose["elev_deg"]),
                               wind_model=_NoWind(), t_w=0.5, t_r=1.6)
    return s


@pytest.mark.parametrize("speed,tgt", [(0.6, (0.6, 0.0)), (0.6, (0.6, 0.3)),
                                       (0.6, (0.6, -0.3))])
def test_rollout_aims_and_does_not_blow_up(sysm, speed, tgt):
    s0 = np.concatenate([[0.3, 0.0, 0.5], np.zeros(3), np.array(tgt)])
    pos, vel, wind = sysm.rollout(s0, _ConstPolicy(speed), 2.0, 0.02, 0.0)
    info = sysm.last_release_info
    rel_speed = np.linalg.norm(info["v_release"])
    assert rel_speed < 3.0                                   # no runaway (was 57 m/s)
    rel = pos[0][:3]
    land = pos[-1][:3]
    land_az = np.degrees(np.arctan2(land[1] - rel[1], land[0] - rel[0]))
    tgt_az = np.degrees(np.arctan2(tgt[1], tgt[0]))
    assert abs(land_az - tgt_az) < 4.0                       # aims at the target
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mc-pilot-pybullet && python -m pytest tests/test_kinetic_chain_rollout.py -v`
Expected: FAIL — `_simulate_pybullet` still calls `plan_throw` without `monotonic_windup`, so the throw phase overshoots (or release speed is off). (If Task 1's pose happens to pass aiming, the blowup/overshoot assertion still guards it.)

- [ ] **Step 3: Pass `monotonic_windup=True` from `_simulate_pybullet`**

In `simulation_class/model_pybullet.py`, find the opt_pose `plan_throw` call:

```python
        coeffs, _, _, v_planned = arm.plan_throw(
            v_cmd, release_pos, self.t_w, self.t_r, t_arm,
            q_release_override=q_ovr, qd_release_override=qd_ovr,
        )
```

Replace with:

```python
        coeffs, _, _, v_planned = arm.plan_throw(
            v_cmd, release_pos, self.t_w, self.t_r, t_arm,
            q_release_override=q_ovr, qd_release_override=qd_ovr,
            monotonic_windup=self._opt_posture is not None,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mc-pilot-pybullet && python -m pytest tests/test_kinetic_chain_rollout.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Whole-trajectory velocity-limit check (the hardware-valid guarantee)**

Add this test to `tests/test_kinetic_chain_rollout.py`:

```python
def test_setpoint_velocity_within_qd_max_whole_trajectory(sysm):
    import pybullet as p
    import pybullet_data
    from robot_arm.arm_controller import ArmController
    from robot_arm.robot_profiles import get_robot_profile
    pose = np.load(POSE_PATH, allow_pickle=True).item()
    client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    prof = get_robot_profile("kinova_gen3_dyn")
    arm = ArmController(client, pybullet_data.getDataPath() + "/" + prof.urdf_rel_path,
                        robot_name="kinova_gen3_dyn")
    sysm._cur_target_xy = np.array([0.6, 0.3])
    d = np.array([np.cos(np.deg2rad(pose["elev_deg"])) * np.cos(0.46),
                  np.cos(np.deg2rad(pose["elev_deg"])) * np.sin(0.46),
                  np.sin(np.deg2rad(pose["elev_deg"]))])
    rel, q_ovr, qd_ovr, _ = sysm._optimized_release(arm, d * 0.6)
    coeffs, _, _, _ = arm.plan_throw(d * 0.6, rel, 0.5, 1.6, 3.0,
                                     q_release_override=q_ovr,
                                     qd_release_override=qd_ovr, monotonic_windup=True)
    from robot_arm.arm_controller import _eval_cubic
    qd_max = np.array(prof.qd_max)
    for phase, dur in [("windup", coeffs["t_w"]),
                       ("throw", coeffs["t_r"] - coeffs["t_w"])]:
        for t in np.linspace(0.0, dur, 150):
            _, qd, _ = _eval_cubic(coeffs[phase], t, with_accel=True)
            assert np.all(np.abs(qd) <= qd_max + 1e-6), f"{phase} exceeds qd_max"
    p.disconnect(client)
```

Run: `cd mc-pilot-pybullet && python -m pytest tests/test_kinetic_chain_rollout.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Full opt_pose test-suite regression + summary**

Run: `cd mc-pilot-pybullet && python -m pytest tests/ -q 2>&1 | tail -5`
Expected: all tests pass (new 4 + existing suite green).

- [ ] **Step 7: Commit**

```bash
cd mc-pilot-pybullet
git add simulation_class/model_pybullet.py tests/test_kinetic_chain_rollout.py
git commit -m "Wire opt_pose rollout to monotonic windup; verify aim + whole-trajectory qd<=qd_max"
```

---

## Notes for the implementer

- PyBullet prints `b3Warning[...]` without trailing newlines; pipe stdout through `tr '\r' '\n'` and `grep -v b3Warning` when reading script output.
- `throw_pose.npy` lives at `mc-pilot-pybullet/throw_pose.npy` and is loaded by `allow_pickle=True).item()`.
- If Task 1's search returns `None` (no pose passes the windup-limit rejection), widen `grid`/`elev` ranges before relaxing the `qd_max` guarantee — the guarantee is the deliverable.
- The old base-using behaviour of `_optimized_release` is only reachable via opt_pose mode; no other rollout path calls it, so Task 2's change has no cross-mode impact.
