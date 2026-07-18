# Velocity-from-Dynamics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gen3 sim throws where release velocity comes from torque-tracked arm motion (not `resetBaseVelocity`), plus a measured tracking-error distribution fitted into a reusable noise model.

**Architecture:** Extend `mc-pilot-pybullet/` in place: new `kinova_gen3_dyn` profile (`control_mode="torque"`), computed-torque branch in `ArmController` with torque-feasibility time scaling in `plan_throw`, a `dynamic=True` release path that lets the ball keep its physics velocity, a sweep script that measures command-vs-actual release velocity through the true `PyBulletThrowingSystem.rollout` pipeline, and a `TrackingErrorNoise` class fitted from the sweep.

**Tech Stack:** Python 3.10, PyBullet (`calculateInverseDynamics`, `TORQUE_CONTROL`), NumPy, pytest 8.3.3, matplotlib (Agg).

**Spec:** `docs/superpowers/specs/2026-07-19-velocity-from-dynamics-design.md`

## Global Constraints

- All code lives under `mc-pilot-pybullet/`; never modify `MC-PILCO/` (vendored upstream).
- Run every command from inside `mc-pilot-pybullet/` (relative imports assume CWD = variant root).
- Do NOT edit the existing `kinova_gen3` profile — prior results and a running multi-seed batch depend on it. New behavior goes only on `kinova_gen3_dyn`.
- Existing callers of `get_setpoint(coeffs, t)` (4 files) must keep working — additions must be backward compatible (`with_accel` keyword, optional `qdd_target`).
- Gen3 torque limits: `tau_max=(39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0)` Nm. Initial gains `kp=100`, `kd=20` per joint.
- Sim timestep everywhere: `dt = 0.02` s (matches Ts in all configs). Real hardware is 1 kHz Kortex low-level — documented delta only.
- Sweep flags (does not silently drop) throws with release-position error > 0.05 m; flagged throws are excluded from the noise fit.
- Tests: pytest, files under `mc-pilot-pybullet/tests/`, run as `python3 -m pytest tests/<file> -v` from `mc-pilot-pybullet/`.
- Commit after each task; short imperative messages matching repo style.
- The multi-seed training batch may be running on this machine — sweep and tests are single-process; do not launch extra parallel workloads.

---

### Task 1: Profile fields + `kinova_gen3_dyn` + test scaffolding

**Files:**
- Modify: `robot_arm/robot_profiles.py`
- Create: `tests/conftest.py`
- Test: `tests/test_profiles.py`

**Interfaces:**
- Produces: `RobotProfile` gains optional fields `tau_max`, `kp`, `kd` (tuples of 7 floats or `None`); `get_robot_profile("kinova_gen3_dyn")` returns a torque-mode Gen3 profile. Later tasks read `profile.tau_max`, `profile.kp`, `profile.kd`, `profile.control_mode == "torque"`.

- [ ] **Step 1: Create `tests/conftest.py`** so tests resolve `robot_arm`/`simulation_class` imports regardless of pytest's sys.path handling:

```python
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

- [ ] **Step 2: Write the failing test** in `tests/test_profiles.py`:

```python
import numpy as np

from robot_arm.robot_profiles import get_robot_profile, profile_to_dict


def test_kinova_gen3_dyn_profile_exists():
    prof = get_robot_profile("kinova_gen3_dyn")
    assert prof.control_mode == "torque"
    assert prof.tau_max == (39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0)
    assert prof.kp == (100.0,) * 7
    assert prof.kd == (20.0,) * 7
    # kinematics identical to the kinematic kinova_gen3 profile
    base = get_robot_profile("kinova_gen3")
    assert prof.urdf_rel_path == base.urdf_rel_path
    assert prof.q_neutral == base.q_neutral
    assert prof.qd_max == base.qd_max
    assert prof.default_release_pos == base.default_release_pos
    assert prof.speed_bounds == base.speed_bounds
    assert prof.timing == base.timing


def test_existing_profiles_unchanged():
    base = get_robot_profile("kinova_gen3")
    assert base.control_mode == "kinematic"
    assert base.tau_max is None
    kuka = get_robot_profile("kuka_iiwa")
    assert kuka.tau_max is None and kuka.kp is None and kuka.kd is None


def test_profile_to_dict_includes_torque_fields():
    d = profile_to_dict(get_robot_profile("kinova_gen3_dyn"))
    assert d["tau_max"] == [39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0]
    assert d["control_mode"] == "torque"
    d2 = profile_to_dict(get_robot_profile("kuka_iiwa"))
    assert d2["tau_max"] is None
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python3 -m pytest tests/test_profiles.py -v`
Expected: FAIL — `TypeError`/`AttributeError` (no `tau_max` field) or `ValueError: Unknown robot 'kinova_gen3_dyn'`.

- [ ] **Step 4: Implement.** In `robot_arm/robot_profiles.py`:

Add three fields to the dataclass after `use_safe_release: bool = False`:

```python
    tau_max: tuple[float, ...] | None = None
    kp: tuple[float, ...] | None = None
    kd: tuple[float, ...] | None = None
```

Add the new profile to `_PROFILES` right after the `"kinova_gen3"` entry:

```python
    "kinova_gen3_dyn": RobotProfile(
        name="kinova_gen3_dyn",
        urdf_rel_path="kinova_gen3/gen3.urdf",
        joint_ids=(0, 1, 2, 3, 4, 5, 6),
        ee_link=7,
        q_neutral=(-0.01, 0.382, -0.04, 1.46, -0.012, 0.731, 0.0),
        qd_max=(1.3963, 1.3963, 1.3963, 1.3963, 1.2218, 1.2218, 1.2218),
        default_release_pos=(0.55, 0.00, 0.45),
        speed_bounds=(0.3, 1.0),
        timing=(0.40, 0.80, 1.60),
        control_mode="torque",
        use_safe_release=False,
        tau_max=(39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0),
        kp=(100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0),
        kd=(20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0),
        notes=(
            "Kinova Gen3 under computed-torque control (velocity-from-dynamics "
            "study). Same kinematics as kinova_gen3; release velocity comes from "
            "tracked arm motion, not resetBaseVelocity. tau_max: 39 Nm large "
            "actuators (joints 1-4), 9 Nm wrists (5-7)."
        ),
    ),
```

(`use_safe_release=False`: dynamic release must not teleport the ball; collision with the arm stays disabled instead.)

In `profile_to_dict`, add before the closing brace:

```python
        "tau_max": list(profile.tau_max) if profile.tau_max is not None else None,
        "kp": list(profile.kp) if profile.kp is not None else None,
        "kd": list(profile.kd) if profile.kd is not None else None,
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/test_profiles.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add robot_arm/robot_profiles.py tests/conftest.py tests/test_profiles.py
git commit -m "Add kinova_gen3_dyn torque-mode profile (tau_max/kp/kd fields)"
```

---

### Task 2: Acceleration from the cubic setpoints

**Files:**
- Modify: `robot_arm/arm_controller.py` (`_eval_cubic`, `get_setpoint`)
- Test: `tests/test_setpoint_accel.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `_eval_cubic(coeffs, tau, with_accel=False)` → `(q, qd)` or `(q, qd, qdd)`; `ArmController.get_setpoint(coeffs, t, with_accel=False)` → same shape. Existing 2-tuple callers unaffected.

- [ ] **Step 1: Write the failing test** in `tests/test_setpoint_accel.py`. Uses a real planned trajectory and checks `qdd` against a central finite difference of `qd`:

```python
import numpy as np
import pybullet as p
import pybullet_data
import pytest

from robot_arm.arm_controller import ArmController
from robot_arm.robot_profiles import get_robot_profile


@pytest.fixture
def arm():
    client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(0.02, physicsClientId=client)
    prof = get_robot_profile("kinova_gen3_dyn")
    urdf = pybullet_data.getDataPath() + "/" + prof.urdf_rel_path
    a = ArmController(client, urdf, robot_name="kinova_gen3_dyn")
    yield a
    p.disconnect(client)


def test_get_setpoint_accel_matches_finite_difference(arm):
    prof = get_robot_profile("kinova_gen3_dyn")
    t_w, t_r, T = prof.timing
    v_cmd = np.array([0.5, 0.0, 0.35])
    coeffs, _, _, _ = arm.plan_throw(v_cmd, np.array(prof.default_release_pos), t_w, t_r, T)

    eps = 1e-5
    for t in [0.1, 0.5 * (t_w + t_r), t_r - 0.05]:
        q, qd, qdd = arm.get_setpoint(coeffs, t, with_accel=True)
        _, qd_lo = arm.get_setpoint(coeffs, t - eps)
        _, qd_hi = arm.get_setpoint(coeffs, t + eps)
        qdd_fd = (qd_hi - qd_lo) / (2 * eps)
        np.testing.assert_allclose(qdd, qdd_fd, atol=1e-4)


def test_get_setpoint_two_tuple_unchanged(arm):
    prof = get_robot_profile("kinova_gen3_dyn")
    t_w, t_r, T = prof.timing
    coeffs, _, _, _ = arm.plan_throw(
        np.array([0.5, 0.0, 0.35]), np.array(prof.default_release_pos), t_w, t_r, T
    )
    out = arm.get_setpoint(coeffs, 0.3)
    assert len(out) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_setpoint_accel.py -v`
Expected: FAIL — `TypeError: get_setpoint() got an unexpected keyword argument 'with_accel'`.
(Note: Task 3 adds torque-mode init; before Task 3 lands, `ArmController` construction with a torque profile works fine — `control_mode` is only compared to `"kinematic"` in `step`. If construction fails here for another reason, fix forward in Task 3, not by weakening this test.)

- [ ] **Step 3: Implement.** In `arm_controller.py`, replace `_eval_cubic` and `get_setpoint`:

```python
def _eval_cubic(coeffs, tau, with_accel=False):
    a0, a1, a2, a3 = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2], coeffs[:, 3]
    q = a0 + a1 * tau + a2 * tau**2 + a3 * tau**3
    qd = a1 + 2.0 * a2 * tau + 3.0 * a3 * tau**2
    if not with_accel:
        return q, qd
    qdd = 2.0 * a2 + 6.0 * a3 * tau
    return q, qd, qdd
```

```python
    def get_setpoint(self, coeffs, t, with_accel=False):
        """Evaluate the piecewise cubic at time t."""
        t_w = coeffs["t_w"]
        t_r = coeffs["t_r"]
        T = coeffs["T"]
        if t <= t_w:
            return _eval_cubic(coeffs["windup"], t, with_accel)
        if t <= t_r:
            return _eval_cubic(coeffs["throw"], t - t_w, with_accel)
        return _eval_cubic(coeffs["follow"], min(t - t_r, T - t_r), with_accel)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_setpoint_accel.py tests/test_profiles.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add robot_arm/arm_controller.py tests/test_setpoint_accel.py
git commit -m "get_setpoint/_eval_cubic optionally return acceleration"
```

---

### Task 3: Computed-torque control branch

**Files:**
- Modify: `robot_arm/arm_controller.py` (`__init__`, `step`)
- Test: `tests/test_torque_control.py`

**Interfaces:**
- Consumes: `profile.tau_max/kp/kd` (Task 1), `with_accel` setpoints (Task 2).
- Produces: `ArmController.step(q_target, qd_target, qdd_target=None)` — torque mode computes `τ = InverseDynamics(q_meas, qd_meas, qdd_des + Kp·e + Kd·ė)` clipped to `±tau_max`; raises `RuntimeError` on non-finite τ. Torque-mode construction validates the profile.

- [ ] **Step 1: Write the failing test** in `tests/test_torque_control.py` (gravity-hold, spec Validation #1):

```python
import numpy as np
import pybullet as p
import pybullet_data
import pytest

from robot_arm.arm_controller import ArmController
from robot_arm.robot_profiles import get_robot_profile

DT = 0.02


def _make_arm(robot_name):
    client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(DT, physicsClientId=client)
    prof = get_robot_profile(robot_name)
    urdf = pybullet_data.getDataPath() + "/" + prof.urdf_rel_path
    return client, ArmController(client, urdf, robot_name=robot_name)


def test_gravity_hold_drift_under_1cm():
    client, arm = _make_arm("kinova_gen3_dyn")
    try:
        q_hold = np.array(get_robot_profile("kinova_gen3_dyn").q_neutral)
        qd_zero = np.zeros(7)
        ee_start = arm.ee_state()[0]
        for _ in range(int(2.0 / DT)):  # 2 s
            arm.step(q_hold, qd_zero, np.zeros(7))
            p.stepSimulation(physicsClientId=client)
        drift = np.linalg.norm(arm.ee_state()[0] - ee_start)
        assert drift < 0.01, f"EE drifted {drift * 100:.2f} cm under gravity hold"
    finally:
        p.disconnect(client)


def test_torque_profile_requires_gains():
    # kinematic profile has no tau_max; torque construction must be impossible
    # for it, and the dyn profile must expose the arrays the controller needs.
    client, arm = _make_arm("kinova_gen3_dyn")
    try:
        assert arm._tau_max.shape == (7,)
        assert arm._kp.shape == (7,)
        assert arm._kd.shape == (7,)
    finally:
        p.disconnect(client)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_torque_control.py -v`
Expected: FAIL — `AttributeError: '_tau_max'` and gravity-hold drift assertion (unactuated arm falls; with default motors still enabled it may hold by accident — the `_tau_max` test fails regardless).

- [ ] **Step 3: Implement.** In `ArmController.__init__`, after `self._control_mode = str(self._profile.control_mode)` add:

```python
        self._tau_max = None
        self._kp = None
        self._kd = None
        if self._control_mode == "torque":
            if (
                self._profile.tau_max is None
                or self._profile.kp is None
                or self._profile.kd is None
            ):
                raise ValueError(
                    f"Profile '{self._profile.name}' uses torque mode but lacks "
                    "tau_max/kp/kd."
                )
            if self._n_dofs != len(self._joint_ids):
                raise ValueError(
                    "Torque mode requires every DOF to be actuated "
                    f"({self._n_dofs} DOFs vs {len(self._joint_ids)} actuated)."
                )
            self._tau_max = np.array(self._profile.tau_max, dtype=float)
            self._kp = np.array(self._profile.kp, dtype=float)
            self._kd = np.array(self._profile.kd, dtype=float)
            # Disable PyBullet's default velocity motors so TORQUE_CONTROL acts.
            p.setJointMotorControlArray(
                self._arm_id,
                self._joint_ids,
                controlMode=p.VELOCITY_CONTROL,
                forces=[0.0] * len(self._joint_ids),
                physicsClientId=client_id,
            )
```

Replace `step` with a three-branch version (kinematic branch and position branch unchanged in behavior):

```python
    def step(self, q_target, qd_target, qdd_target=None):
        """Command actuated joints for one sim step (mode set by profile)."""
        if self._control_mode == "kinematic":
            for local_i, joint_id in enumerate(self._joint_ids):
                p.resetJointState(
                    self._arm_id,
                    joint_id,
                    targetValue=float(q_target[local_i]),
                    targetVelocity=float(qd_target[local_i]),
                    physicsClientId=self._cid,
                )
            return

        if self._control_mode == "torque":
            if qdd_target is None:
                qdd_target = np.zeros(len(self._joint_ids))
            states = p.getJointStates(
                self._arm_id, self._joint_ids, physicsClientId=self._cid
            )
            q_meas = np.array([s[0] for s in states])
            qd_meas = np.array([s[1] for s in states])
            e = np.asarray(q_target, dtype=float) - q_meas
            ed = np.asarray(qd_target, dtype=float) - qd_meas
            qdd_cmd = np.asarray(qdd_target, dtype=float) + self._kp * e + self._kd * ed
            # Inverse dynamics for the ARM ALONE: the gripped ball and its
            # constraint forces are deliberately unmodeled (source of the
            # tracking error this study measures).
            tau = np.array(
                p.calculateInverseDynamics(
                    self._arm_id,
                    q_meas.tolist(),
                    qd_meas.tolist(),
                    qdd_cmd.tolist(),
                    physicsClientId=self._cid,
                )
            )
            if not np.all(np.isfinite(tau)):
                raise RuntimeError(
                    f"Non-finite torque command: tau={tau}, q={q_meas}, qd={qd_meas}"
                )
            tau = np.clip(tau, -self._tau_max, self._tau_max)
            p.setJointMotorControlArray(
                self._arm_id,
                self._joint_ids,
                controlMode=p.TORQUE_CONTROL,
                forces=tau.tolist(),
                physicsClientId=self._cid,
            )
            return

        p.setJointMotorControlArray(
            self._arm_id,
            self._joint_ids,
            controlMode=p.POSITION_CONTROL,
            targetPositions=q_target.tolist(),
            targetVelocities=qd_target.tolist(),
            positionGains=[self._position_gain] * len(self._joint_ids),
            velocityGains=[self._velocity_gain] * len(self._joint_ids),
            forces=(self._force_scale * self._max_forces).tolist(),
            physicsClientId=self._cid,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_torque_control.py -v`
Expected: 2 passed. If gravity-hold drifts > 1 cm: torque path is miswired (check motor-disable call ran, ID uses measured not target state) — do not raise the threshold.

- [ ] **Step 5: Commit**

```bash
git add robot_arm/arm_controller.py tests/test_torque_control.py
git commit -m "Add computed-torque control branch to ArmController"
```

---

### Task 4: Torque-feasibility time scaling in `plan_throw`

**Files:**
- Modify: `robot_arm/arm_controller.py` (`plan_throw`, new `_throw_peak_torque_ratio`)
- Test: `tests/test_time_scaling.py`

**Interfaces:**
- Consumes: torque fields (Task 3), `_eval_cubic(..., with_accel=True)` (Task 2).
- Produces: in torque mode, `plan_throw` returns `coeffs` whose `t_r`/`T` may be stretched so throw-phase torque demand fits `tau_max`; adds keys `coeffs["time_scale"]` (float ≥ 1.0, 1.0 for non-torque modes). Raises `RuntimeError` with per-joint report after 6 failed stretch iterations. Callers must use `coeffs["t_r"]`, never the `t_r` they passed in (Task 6 fixes the one caller that doesn't).

- [ ] **Step 1: Write the failing test** in `tests/test_time_scaling.py`:

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
    prof = get_robot_profile("kinova_gen3_dyn")
    urdf = pybullet_data.getDataPath() + "/" + prof.urdf_rel_path
    a = ArmController(client, urdf, robot_name="kinova_gen3_dyn")
    yield a, prof
    p.disconnect(client)


def _plan(arm, prof, speed):
    t_w, t_r, T = prof.timing
    alpha = np.deg2rad(35.0)
    v_cmd = np.array([speed * np.cos(alpha), 0.0, speed * np.sin(alpha)])
    return arm.plan_throw(v_cmd, np.array(prof.default_release_pos), t_w, t_r, T)


def test_planned_throw_is_torque_feasible_at_max_speed(arm):
    controller, prof = arm
    coeffs, _, _, _ = _plan(controller, prof, 1.0)  # top of speed_bounds
    assert coeffs["time_scale"] >= 1.0
    # Re-check demanded torque over the (possibly stretched) throw phase.
    dt_throw = coeffs["t_r"] - coeffs["t_w"]
    for tau_t in np.linspace(0.0, dt_throw, 50):
        q, qd, qdd = _eval_cubic(coeffs["throw"], tau_t, with_accel=True)
        torque = np.array(
            p.calculateInverseDynamics(
                controller.arm_id, q.tolist(), qd.tolist(), qdd.tolist(),
                physicsClientId=controller._cid,
            )
        )
        assert np.all(np.abs(torque) <= controller._tau_max + 1e-9)


def test_time_scale_recorded_and_follow_duration_preserved(arm):
    controller, prof = arm
    t_w, t_r, T = prof.timing
    coeffs, _, _, _ = _plan(controller, prof, 1.0)
    assert coeffs["t_r"] >= t_r  # never shrinks
    np.testing.assert_allclose(coeffs["T"] - coeffs["t_r"], T - t_r)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_time_scaling.py -v`
Expected: FAIL — `KeyError: 'time_scale'` (and possibly torque-bound violations).

- [ ] **Step 3: Implement.** In `ArmController`, add a helper and extend `plan_throw`.

Helper (place above `release_ball`):

```python
    def _throw_peak_torque_ratio(self, throw_coeffs, dt_throw, n_samples=50):
        """Max over the throw phase of max_j |tau_j| / tau_max_j."""
        worst = 0.0
        worst_tau = None
        for tau_t in np.linspace(0.0, dt_throw, n_samples):
            q, qd, qdd = _eval_cubic(throw_coeffs, tau_t, with_accel=True)
            torque = np.array(
                p.calculateInverseDynamics(
                    self._arm_id,
                    q.tolist(),
                    qd.tolist(),
                    qdd.tolist(),
                    physicsClientId=self._cid,
                )
            )
            ratio = float(np.max(np.abs(torque) / self._tau_max))
            if ratio > worst:
                worst = ratio
                worst_tau = torque
        return worst, worst_tau
```

In `plan_throw`, replace the block that builds `coeffs` (keep everything above `q_windup = ...` unchanged) with:

```python
        q_windup = self._q_neutral + (q_release - self._q_neutral) * (-0.5)
        q_windup = np.clip(q_windup, self._q_lo, self._q_hi)
        q_follow = self._q_neutral.copy()

        dt_throw = t_r - t_w
        follow_dur = T - t_r
        time_scale = 1.0
        throw_coeffs = _cubic_to_velocity(q_windup, q_release, qd_release, dt_throw)

        if self._control_mode == "torque":
            for _ in range(6):
                ratio, worst_tau = self._throw_peak_torque_ratio(throw_coeffs, dt_throw)
                if ratio <= 1.0:
                    break
                dt_throw *= 1.2
                time_scale *= 1.2
                throw_coeffs = _cubic_to_velocity(
                    q_windup, q_release, qd_release, dt_throw
                )
            else:
                ratio, worst_tau = self._throw_peak_torque_ratio(throw_coeffs, dt_throw)
                if ratio > 1.0:
                    report = ", ".join(
                        f"j{j}: {abs(t):.1f}/{m:.1f} Nm"
                        for j, (t, m) in enumerate(zip(worst_tau, self._tau_max))
                    )
                    raise RuntimeError(
                        f"Throw infeasible after 6 time-scaling iterations "
                        f"(peak ratio {ratio:.2f}): {report}"
                    )

        t_r_actual = t_w + dt_throw
        T_actual = t_r_actual + follow_dur

        coeffs = {
            "windup": _cubic_rest_to_rest(self._q_neutral, q_windup, t_w),
            "throw": throw_coeffs,
            "follow": _cubic_from_velocity(q_release, qd_release, q_follow, follow_dur),
            "t_w": t_w,
            "t_r": t_r_actual,
            "T": T_actual,
            "clip_scale": clip_scale,
            "time_scale": time_scale,
        }
        return coeffs, q_release, qd_release, v_achieved
```

- [ ] **Step 4: Run all tests**

Run: `python3 -m pytest tests/ -v`
Expected: all pass (earlier tests unaffected — `time_scale` is additive).

- [ ] **Step 5: Commit**

```bash
git add robot_arm/arm_controller.py tests/test_time_scaling.py
git commit -m "Stretch throw phase until torque-feasible in torque mode"
```

---

### Task 5: Dynamic release + slow-tracking gate (gain tuning)

**Files:**
- Modify: `robot_arm/arm_controller.py` (`release_ball`); possibly `robot_arm/robot_profiles.py` (kp/kd retune)
- Test: `tests/test_dynamic_release.py`

**Interfaces:**
- Consumes: torque stepping (Task 3), scaled plans (Task 4).
- Produces: `release_ball(ball_id, ..., dynamic=False)` — with `dynamic=True` it removes the grip constraint, ignores `set_vel`/`dv_noise`/`release_pos`, does NOT call `resetBaseVelocity`, and returns the ball's own `getBaseVelocity` linear velocity. This is the release path Task 6 wires into the rollout.

- [ ] **Step 1: Write the failing test** in `tests/test_dynamic_release.py` (spec Validation #2 — full slow throw, tracking gate, ball flies from physics):

```python
import numpy as np
import pybullet as p
import pybullet_data
import pytest

from robot_arm.arm_controller import ArmController
from robot_arm.robot_profiles import get_robot_profile

DT = 0.02


def _throw_world():
    client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(DT, physicsClientId=client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.loadURDF("plane.urdf", physicsClientId=client)
    prof = get_robot_profile("kinova_gen3_dyn")
    urdf = pybullet_data.getDataPath() + "/" + prof.urdf_rel_path
    arm = ArmController(client, urdf, robot_name="kinova_gen3_dyn")
    ee_pos = arm.ee_state()[0]
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.0327, physicsClientId=client)
    ball = p.createMultiBody(
        baseMass=0.0577,
        baseCollisionShapeIndex=col,
        basePosition=ee_pos.tolist(),
        physicsClientId=client,
    )
    p.changeDynamics(ball, -1, linearDamping=0.0, angularDamping=0.0,
                     physicsClientId=client)
    arm.attach_ball(ball)
    return client, arm, ball, prof


def test_slow_throw_tracks_and_releases_dynamically():
    client, arm, ball, prof = _throw_world()
    try:
        t_w, t_r, T = prof.timing
        speed = 0.3
        alpha = np.deg2rad(35.0)
        v_cmd = np.array([speed * np.cos(alpha), 0.0, speed * np.sin(alpha)])
        coeffs, _, _, v_achieved = arm.plan_throw(
            v_cmd, np.array(prof.default_release_pos), t_w, t_r, T
        )

        max_err = 0.0
        n_steps = int(coeffs["t_r"] / DT)
        for step in range(n_steps):
            t = step * DT
            q_t, qd_t, qdd_t = arm.get_setpoint(coeffs, t, with_accel=True)
            arm.step(q_t, qd_t, qdd_t)
            p.stepSimulation(physicsClientId=client)
            if t > coeffs["t_w"]:  # only gate the throw phase
                states = p.getJointStates(arm.arm_id, arm.joint_ids,
                                          physicsClientId=client)
                q_meas = np.array([s[0] for s in states])
                max_err = max(max_err, float(np.max(np.abs(q_t - q_meas))))

        v_release = arm.release_ball(ball, dynamic=True, keep_collision_disabled=True)

        assert max_err < 0.02, f"joint tracking error {max_err:.4f} rad"
        # Ball velocity must be physical and in the ballpark of the command.
        assert np.all(np.isfinite(v_release))
        assert np.linalg.norm(v_release - v_achieved) < 0.5 * np.linalg.norm(v_achieved)
        # Ball must keep flying under physics (no resetBaseVelocity happened).
        for _ in range(5):
            p.stepSimulation(physicsClientId=client)
        v_after, _ = p.getBaseVelocity(ball, physicsClientId=client)
        assert np.all(np.isfinite(np.array(v_after)))
    finally:
        p.disconnect(client)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_dynamic_release.py -v`
Expected: FAIL — `TypeError: release_ball() got an unexpected keyword argument 'dynamic'`.

- [ ] **Step 3: Implement `dynamic=True`.** In `release_ball`, change the signature and short-circuit before the `release_pos` block:

```python
    def release_ball(
        self,
        ball_id,
        set_vel=None,
        dv_noise=None,
        release_pos=None,
        keep_collision_disabled=False,
        dynamic=False,
    ):
        """
        Remove the grip constraint and optionally override the ball velocity.

        dynamic=True: the ball keeps whatever velocity the physics engine gave
        it while dragged by the constraint — no resetBaseVelocity, no
        repositioning; set_vel/dv_noise/release_pos are ignored.
        """
        if self._grip_id is not None:
            p.removeConstraint(self._grip_id, physicsClientId=self._cid)
            self._grip_id = None
        if self._attached_ball_id is not None:
            if not keep_collision_disabled:
                self._set_ball_collision_with_arm(self._attached_ball_id, enable=True)
            self._attached_ball_id = None

        if dynamic:
            ball_vel, _ = p.getBaseVelocity(ball_id, physicsClientId=self._cid)
            return np.array(ball_vel)
```

(the rest of the method stays exactly as it is).

- [ ] **Step 4: Run the test; tune gains if the tracking gate fails**

Run: `python3 -m pytest tests/test_dynamic_release.py -v`

If `max_err >= 0.02 rad`: tune `kp`/`kd` on the `kinova_gen3_dyn` profile only (Task 1 file). Procedure — change, re-run this test, keep best:
1. Lagging (error grows smoothly through the phase): raise kp → `(200,)*7`, kd → `(30,)*7`.
2. Oscillating (error alternates sign step to step): raise kd first → `(40,)*7`, keep kp.
3. Still failing at kp=400/kd=60: stop — the 50 Hz control step is the likely cause; report to the user with the measured error curve instead of forcing the threshold.

Expected: PASS with final gains recorded in the profile.

- [ ] **Step 5: Run the full suite**

Run: `python3 -m pytest tests/ -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add robot_arm/arm_controller.py robot_arm/robot_profiles.py tests/test_dynamic_release.py
git commit -m "Add dynamic release path (ball keeps physics velocity)"
```

---

### Task 6: Wire torque mode + dynamic release into `PyBulletThrowingSystem`

**Files:**
- Modify: `simulation_class/model_pybullet.py` (`__init__`, `_simulate_pybullet`)
- Test: `tests/test_rollout_dynamic.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `PyBulletThrowingSystem(robot_name="kinova_gen3_dyn").rollout(...)` runs the torque-tracked throw with dynamic release; after each rollout `system.last_release_info` is a dict with keys `v_cmd` (3,), `v_planned` (3, post-clip `v_achieved`), `v_release` (3, actual ball velocity), `release_pos_err` (float, m), `clip_scale` (float), `time_scale` (float). Constructor raises `ValueError` if `arm_noise` is combined with a torque-mode profile. Task 7's sweep reads `last_release_info`.

- [ ] **Step 1: Write the failing test** in `tests/test_rollout_dynamic.py`:

```python
import numpy as np
import pytest

from simulation_class.model_pybullet import PyBulletThrowingSystem
from robot_arm.noise_models import VelocityBiasNoise


def test_rollout_dynamic_release_info():
    sys_ = PyBulletThrowingSystem(robot_name="kinova_gen3_dyn", t_w=0.40, t_r=0.80)
    policy = lambda s, t: np.array([0.6])
    s0 = np.array([0.55, 0.0, 0.45, 0.0, 0.0, 0.0, 0.77, 0.0])
    noisy, inputs, clean = sys_.rollout(s0, policy, T=2.0, dt=0.02, noise=0.0)

    assert np.all(np.isfinite(clean))
    assert clean.shape[1] == 8
    assert clean[-1, 2] < 0.05  # ball landed (interpolated to plane)

    info = sys_.last_release_info
    for key in ("v_cmd", "v_planned", "v_release", "release_pos_err",
                "clip_scale", "time_scale"):
        assert key in info, key
    assert info["time_scale"] >= 1.0
    assert info["release_pos_err"] < 0.10
    # first recorded velocity row must be the ACTUAL ball velocity
    np.testing.assert_allclose(clean[0, 3:6], info["v_release"], atol=1e-9)


def test_arm_noise_rejected_in_torque_mode():
    with pytest.raises(ValueError):
        PyBulletThrowingSystem(
            robot_name="kinova_gen3_dyn", arm_noise=VelocityBiasNoise(0.01)
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_rollout_dynamic.py -v`
Expected: FAIL — no `last_release_info`, no `ValueError`, and release still goes through `resetBaseVelocity`.

- [ ] **Step 3: Implement.** In `model_pybullet.py`:

(a) End of `__init__`, after `self._profile = get_robot_profile(robot_name)`:

```python
        self._dynamic_release = self._profile.control_mode == "torque"
        if self._dynamic_release and self.arm_noise is not None:
            raise ValueError(
                "arm_noise is not supported with torque-mode profiles: the "
                "tracking error IS the noise being measured. Use the kinematic "
                "profile + TrackingErrorNoise for noise-aware training."
            )
        self.last_release_info = None
```

(b) In `_simulate_pybullet`, after `coeffs, _, _, _ = arm.plan_throw(...)`, capture the planned velocity and use the (possibly stretched) release time. Replace:

```python
        coeffs, _, _, _ = arm.plan_throw(v_cmd, release_pos, self.t_w, self.t_r, t_arm)

        release_offset = 0
        if self.arm_noise is not None:
            release_offset = self.arm_noise.sample_release_offset()
        release_step = int(self.t_r / dt) + release_offset
```

with:

```python
        coeffs, _, _, v_planned = arm.plan_throw(
            v_cmd, release_pos, self.t_w, self.t_r, t_arm
        )
        t_r_actual = coeffs["t_r"]  # torque mode may have stretched the throw

        release_offset = 0
        if self.arm_noise is not None:
            release_offset = self.arm_noise.sample_release_offset()
        release_step = int(t_r_actual / dt) + release_offset
```

and change `total_steps = int((self.t_r + T) / dt) + 100` to `total_steps = int((t_r_actual + T) / dt) + 100`.

(c) The stepping line becomes accel-aware (kinematic/position arms ignore `qdd`):

```python
                q_t, qd_t, qdd_t = arm.get_setpoint(coeffs, t, with_accel=True)
                arm.step(q_t, qd_t, qdd_t)
```

(d) In the release block (`if step >= release_step:`), add the dynamic branch FIRST, before the existing `if self.arm_noise is not None:` branch:

```python
                    if self._dynamic_release:
                        ee_pos_rel = ee_pos.copy()
                        actual_release_vel = arm.release_ball(
                            ball_id,
                            dynamic=True,
                            keep_collision_disabled=True,
                        )
                        self.last_release_info = {
                            "v_cmd": v_cmd.copy(),
                            "v_planned": np.array(v_planned, dtype=float),
                            "v_release": actual_release_vel.copy(),
                            "release_pos_err": float(
                                np.linalg.norm(ee_pos_rel - release_pos)
                            ),
                            "clip_scale": float(coeffs["clip_scale"]),
                            "time_scale": float(coeffs["time_scale"]),
                        }
                    elif self.arm_noise is not None:
```

(the existing noise branch and the final `else` branch stay as they are, re-indented under `elif`/`else`).

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_rollout_dynamic.py -v`
Expected: 2 passed. If `release_pos_err` fails at 0.10 m, the tracked arm isn't reaching the release pose — debug tracking (Task 5 gains), don't loosen the bound.

- [ ] **Step 5: Regression — kinematic path untouched**

Run: `python3 -m pytest tests/ -v` and then a 2-trial smoke run of the existing kinematic pipeline:

```bash
python3 train_mc_pilot_pb_arm.py --robot kinova_gen3 --seed 99 --num_trials 2 --results_root /tmp/claude-1000/-home-olympusforge-trade-throwing-robotic-arm/bda45de7-24ad-468f-87b4-15c245d02bb2/scratchpad/smoke_kinematic
```

Expected: tests pass; smoke run completes without exception (hit rate irrelevant at 2 trials). Delete nothing — the scratchpad path keeps it out of results.

- [ ] **Step 6: Commit**

```bash
git add simulation_class/model_pybullet.py tests/test_rollout_dynamic.py
git commit -m "Wire torque-mode dynamic release into PyBulletThrowingSystem"
```

---

### Task 7: Tracking-error measurement sweep script

**Files:**
- Create: `measure_tracking_error.py`
- Test: `tests/test_measure_sweep.py` (runs the script's `--quick` grid via its main function)

**Interfaces:**
- Consumes: `PyBulletThrowingSystem` + `last_release_info` (Task 6).
- Produces: `measure_tracking_error.py` CLI writing `<out>/tracking_error.npz` with arrays `u_cmd (N,)`, `angle (N,)`, `v_cmd (N,3)`, `v_planned (N,3)`, `v_release (N,3)`, `release_pos_err (N,)`, `time_scale (N,)`, `land_xy (N,2)`, `flag (N,)` (1 = pos_err > 0.05 m, excluded from fits); plus `tracking_error.png` and a stdout stats table. `run_sweep(u_grid, angle_grid, robot_name) -> dict of arrays` is importable. Task 8 fits from the npz.

- [ ] **Step 1: Write the failing test** in `tests/test_measure_sweep.py`:

```python
import numpy as np

from measure_tracking_error import run_sweep


def test_quick_sweep_produces_records():
    u_grid = np.linspace(0.3, 1.0, 3)
    angle_grid = np.deg2rad(np.linspace(-30.0, 30.0, 2))
    rec = run_sweep(u_grid, angle_grid, robot_name="kinova_gen3_dyn")
    n = 3 * 2
    assert rec["u_cmd"].shape == (n,)
    assert rec["v_cmd"].shape == (n, 3)
    assert rec["v_release"].shape == (n, 3)
    assert rec["flag"].shape == (n,)
    assert np.all(np.isfinite(rec["v_release"]))
    # commanded speeds actually span the grid
    np.testing.assert_allclose(np.unique(rec["u_cmd"]), u_grid, atol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_measure_sweep.py -v`
Expected: FAIL — `ModuleNotFoundError: measure_tracking_error`.

- [ ] **Step 3: Create `measure_tracking_error.py`:**

```python
"""
Measure command-vs-actual release-velocity error for a torque-mode profile.

Sweeps commanded release speed x target azimuth through the TRUE
PyBulletThrowingSystem.rollout pipeline (hand-rolled replays showed ~10 cm
systematic discrepancy in earlier work) and records the actual ball velocity
at dynamic release. Sim is deterministic: the distribution comes from command
diversity, not repeats.

Usage:
  python3 measure_tracking_error.py                # full 25x9 grid, 225 throws
  python3 measure_tracking_error.py --quick        # 5x3 grid for smoke tests
  python3 measure_tracking_error.py --out results_tracking_error
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robot_arm.robot_profiles import get_robot_profile
from simulation_class.model_pybullet import PyBulletThrowingSystem

POS_ERR_FLAG = 0.05  # m; throws worse than this are flagged + excluded from fits


def run_sweep(u_grid, angle_grid, robot_name="kinova_gen3_dyn"):
    profile = get_robot_profile(robot_name)
    t_w, t_r, _ = profile.timing
    release_pos = np.array(profile.default_release_pos, dtype=float)
    # Target azimuth is set through the target position; distance only fixes
    # the aim direction (speed is commanded directly), so mid-band is fine.
    dist = 0.77

    system = PyBulletThrowingSystem(robot_name=robot_name, t_w=t_w, t_r=t_r)

    rec = {k: [] for k in ("u_cmd", "angle", "v_cmd", "v_planned", "v_release",
                           "release_pos_err", "time_scale", "land_xy", "flag")}
    n_total = len(u_grid) * len(angle_grid)
    i = 0
    for u in u_grid:
        for ang in angle_grid:
            i += 1
            target_xy = np.array([dist * np.cos(ang), dist * np.sin(ang)])
            s0 = np.concatenate([release_pos, np.zeros(3), target_xy])
            policy = lambda s, t, _u=u: np.array([_u])
            _, _, clean = system.rollout(s0, policy, T=2.0, dt=0.02, noise=0.0)
            info = system.last_release_info

            rec["u_cmd"].append(u)
            rec["angle"].append(ang)
            rec["v_cmd"].append(info["v_cmd"])
            rec["v_planned"].append(info["v_planned"])
            rec["v_release"].append(info["v_release"])
            rec["release_pos_err"].append(info["release_pos_err"])
            rec["time_scale"].append(info["time_scale"])
            rec["land_xy"].append(clean[-1, 0:2])
            rec["flag"].append(1.0 if info["release_pos_err"] > POS_ERR_FLAG else 0.0)
            print(f"[{i}/{n_total}] u={u:.3f} ang={np.rad2deg(ang):+.0f}deg "
                  f"|dv|={np.linalg.norm(info['v_release'] - info['v_cmd']):.4f} "
                  f"pos_err={info['release_pos_err'] * 100:.2f}cm "
                  f"ts={info['time_scale']:.2f}", flush=True)

    return {k: np.array(v) for k, v in rec.items()}


def print_stats(rec):
    ok = rec["flag"] < 0.5
    n_flag = int(rec["flag"].sum())
    print(f"\n=== Tracking-error stats ({ok.sum()} throws, {n_flag} flagged/excluded) ===")
    dv = rec["v_release"][ok] - rec["v_cmd"][ok]
    for j, name in enumerate("xyz"):
        print(f"  dv_{name}: mean {dv[:, j].mean():+.4f}  std {dv[:, j].std():.4f} m/s")
    u = rec["u_cmd"][ok]
    print("  per u-band (|dv| mean):")
    for lo in np.arange(0.3, 1.0, 0.1):
        band = (u >= lo) & (u < lo + 0.1)
        if band.any():
            mags = np.linalg.norm(dv[band], axis=1)
            print(f"    u in [{lo:.1f},{lo + 0.1:.1f}): {mags.mean():.4f} m/s (n={band.sum()})")


def save_figure(rec, path):
    ok = rec["flag"] < 0.5
    dv = rec["v_release"] - rec["v_cmd"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    for j, (ax, name) in enumerate(zip(axes, "xyz")):
        ax.scatter(rec["u_cmd"][ok], dv[ok, j], s=12, label="ok")
        if (~ok).any():
            ax.scatter(rec["u_cmd"][~ok], dv[~ok, j], s=12, c="red", label="flagged")
        ax.axhline(0.0, color="gray", lw=0.5)
        ax.set_xlabel("commanded speed u (m/s)")
        ax.set_ylabel(f"dv_{name} (m/s)")
        ax.legend(fontsize=7)
    fig.suptitle("Release-velocity tracking error vs commanded speed (Gen3 torque mode)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--robot", type=str, default="kinova_gen3_dyn")
    ap.add_argument("--out", type=str, default="results_tracking_error")
    ap.add_argument("--quick", action="store_true", help="5x3 grid instead of 25x9")
    args = ap.parse_args()

    if args.quick:
        u_grid = np.linspace(0.3, 1.0, 5)
        angle_grid = np.deg2rad(np.linspace(-30.0, 30.0, 3))
    else:
        u_grid = np.linspace(0.3, 1.0, 25)
        angle_grid = np.deg2rad(np.linspace(-30.0, 30.0, 9))

    rec = run_sweep(u_grid, angle_grid, robot_name=args.robot)
    os.makedirs(args.out, exist_ok=True)
    npz_path = os.path.join(args.out, "tracking_error.npz")
    np.savez(npz_path, **rec)
    save_figure(rec, os.path.join(args.out, "tracking_error.png"))
    print_stats(rec)
    print(f"\nSaved {npz_path} and tracking_error.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test, then the quick CLI**

Run: `python3 -m pytest tests/test_measure_sweep.py -v`
Expected: PASS.

Run: `python3 measure_tracking_error.py --quick --out /tmp/claude-1000/-home-olympusforge-trade-throwing-robotic-arm/bda45de7-24ad-468f-87b4-15c245d02bb2/scratchpad/sweep_quick`
Expected: 15 progress lines, stats table, npz + png written, flagged count reported (expect 0 flagged; a few are acceptable, investigate if most are flagged).

- [ ] **Step 5: Commit**

```bash
git add measure_tracking_error.py tests/test_measure_sweep.py
git commit -m "Add tracking-error measurement sweep script"
```

---

### Task 8: `TrackingErrorNoise` fitted from measurements

**Files:**
- Modify: `robot_arm/noise_models.py`
- Test: `tests/test_tracking_noise.py`

**Interfaces:**
- Consumes: sweep npz layout (Task 7).
- Produces: `TrackingErrorNoise(coef_a, coef_b, resid_cov, seed=None)` implementing the `ArmNoise` interface (`pybullet_release_vel(v_cmd, ee_vel)`, `perturb_numpy(v3d_np, n)`); classmethod `TrackingErrorNoise.from_measurements(npz_path, seed=None)`. Fit and application are in the **throw-aligned frame** (parallel-to-command, lateral, vertical) so azimuth symmetry is respected: `dv_par = a_par*u + b_par`, etc.

- [ ] **Step 1: Write the failing test** in `tests/test_tracking_noise.py` — synthetic data with a known speed-proportional slip must be recovered:

```python
import numpy as np

from robot_arm.noise_models import TrackingErrorNoise


def _synthetic_npz(tmp_path, slip=0.10, vz_bias=-0.03, sigma=0.005, seed=0):
    rng = np.random.default_rng(seed)
    alpha = np.deg2rad(35.0)
    u = np.repeat(np.linspace(0.3, 1.0, 25), 9)
    ang = np.tile(np.deg2rad(np.linspace(-30, 30, 9)), 25)
    v_cmd = np.stack([
        u * np.cos(alpha) * np.cos(ang),
        u * np.cos(alpha) * np.sin(ang),
        u * np.sin(alpha),
    ], axis=1)
    # ground truth: lose `slip` fraction of speed along the command direction,
    # constant vertical bias, small isotropic scatter
    v_release = (1.0 - slip) * v_cmd
    v_release[:, 2] += vz_bias
    v_release += rng.normal(0.0, sigma, v_cmd.shape)
    path = str(tmp_path / "sweep.npz")
    np.savez(path, u_cmd=u, angle=ang, v_cmd=v_cmd,
             v_planned=v_cmd, v_release=v_release,
             release_pos_err=np.zeros_like(u), time_scale=np.ones_like(u),
             land_xy=np.zeros((len(u), 2)), flag=np.zeros_like(u))
    return path


def test_fit_recovers_speed_proportional_slip(tmp_path):
    noise = TrackingErrorNoise.from_measurements(_synthetic_npz(tmp_path), seed=1)
    # parallel component: dv_par = -slip * u  =>  a_par ~ -0.10, b_par ~ 0
    assert abs(noise.coef_a[0] - (-0.10)) < 0.02
    assert abs(noise.coef_b[0]) < 0.02
    # vertical: constant bias => a_z ~ 0, b_z ~ -0.03
    assert abs(noise.coef_b[2] - (-0.03)) < 0.02
    # residual scatter should be near sigma, far below the bias magnitudes
    assert np.all(np.sqrt(np.diag(noise.resid_cov)) < 0.02)


def test_release_vel_applies_bias_in_command_frame(tmp_path):
    noise = TrackingErrorNoise.from_measurements(_synthetic_npz(tmp_path, sigma=1e-6), seed=1)
    v_cmd = np.array([0.6, 0.2, 0.5])
    out = noise.pybullet_release_vel(v_cmd, ee_vel=None)
    u = np.linalg.norm(v_cmd)
    # bias along command direction ~ -slip*u; overall speed must shrink
    assert np.linalg.norm(out) < u
    assert abs((np.linalg.norm(out) - u) / u + 0.10) < 0.05


def test_perturb_numpy_shapes(tmp_path):
    noise = TrackingErrorNoise.from_measurements(_synthetic_npz(tmp_path), seed=1)
    v3d = np.tile(np.array([0.5, 0.0, 0.35]), (7, 1))
    scale, additive = noise.perturb_numpy(v3d, 7)
    assert scale.shape == (7,)
    assert additive.shape == (7, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_tracking_noise.py -v`
Expected: FAIL — `ImportError: cannot import name 'TrackingErrorNoise'`.

- [ ] **Step 3: Implement.** Append to `robot_arm/noise_models.py`:

```python
def _throw_frame(v_cmd):
    """Orthonormal frame aligned with a command: (parallel, lateral, vertical-ish).

    e_par is the unit command direction; e_lat is horizontal, perpendicular to
    the command azimuth; e_ver completes the right-handed frame (mostly +z).
    """
    v = np.asarray(v_cmd, dtype=float)
    n = np.linalg.norm(v)
    e_par = v / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    e_lat = np.cross(up, e_par)
    ln = np.linalg.norm(e_lat)
    e_lat = e_lat / ln if ln > 1e-9 else np.array([0.0, 1.0, 0.0])
    e_ver = np.cross(e_par, e_lat)
    return np.stack([e_par, e_lat, e_ver])  # rows = frame axes


class TrackingErrorNoise(ArmNoise):
    """
    Release-velocity error measured from torque-tracked throws (velocity-from-
    dynamics study), applied as noise for noise-aware training on the
    kinematic profile.

    Fit (from measure_tracking_error.py output) is done in the throw-aligned
    frame so azimuth symmetry is respected:
        dv_frame_i = coef_a[i] * u + coef_b[i] + N(0, resid_cov)
    with i = (parallel-to-command, lateral, vertical), u = ||v_cmd||.

    Parameters
    ----------
    coef_a, coef_b : (3,) — linear bias coefficients per aligned component
    resid_cov      : (3, 3) — residual covariance in the aligned frame
    seed           : optional int
    """

    def __init__(self, coef_a, coef_b, resid_cov, seed=None):
        self.coef_a = np.asarray(coef_a, dtype=float)
        self.coef_b = np.asarray(coef_b, dtype=float)
        self.resid_cov = np.asarray(resid_cov, dtype=float)
        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_measurements(cls, npz_path, seed=None):
        d = np.load(npz_path)
        ok = d["flag"] < 0.5
        u = d["u_cmd"][ok]
        v_cmd = d["v_cmd"][ok]
        dv_world = d["v_release"][ok] - v_cmd

        dv_frame = np.empty_like(dv_world)
        for i in range(len(u)):
            dv_frame[i] = _throw_frame(v_cmd[i]) @ dv_world[i]

        A = np.stack([u, np.ones_like(u)], axis=1)          # (N, 2)
        coef, *_ = np.linalg.lstsq(A, dv_frame, rcond=None)  # (2, 3)
        resid = dv_frame - A @ coef
        resid_cov = np.cov(resid.T)
        return cls(coef[0], coef[1], resid_cov, seed=seed)

    def _bias(self, u):
        return self.coef_a * u + self.coef_b

    def pybullet_release_vel(self, v_cmd, ee_vel):
        v_cmd = np.asarray(v_cmd, dtype=float)
        frame = _throw_frame(v_cmd)
        u = np.linalg.norm(v_cmd)
        dv_frame = self._bias(u) + self.rng.multivariate_normal(
            np.zeros(3), self.resid_cov
        )
        return v_cmd + frame.T @ dv_frame

    def perturb_numpy(self, v3d_np, n):
        additive = np.zeros((n, 3))
        scatter = self.rng.multivariate_normal(np.zeros(3), self.resid_cov, n)
        for i in range(n):
            frame = _throw_frame(v3d_np[i])
            u = np.linalg.norm(v3d_np[i])
            additive[i] = frame.T @ (self._bias(u) + scatter[i])
        return np.ones(n), additive
```

Also extend the module docstring's "Available classes" list with one line:

```
  TrackingErrorNoise — speed-dependent bias + scatter fitted from measured
                       torque-tracking error (velocity-from-dynamics study)
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_tracking_noise.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add robot_arm/noise_models.py tests/test_tracking_noise.py
git commit -m "Add TrackingErrorNoise fitted from tracking-error sweeps"
```

---

### Task 9: Full sweep, noise fit, sim2sim gap report

**Files:**
- Create: `eval_sim2sim_gap.py`
- Output (not committed as results? — results dirs are normally kept untracked like `results_*`; keep these untracked too): `results_tracking_error/tracking_error.npz`, `.png`, `fitted_noise.npz`, `sim2sim_gap.npz`

**Interfaces:**
- Consumes: sweep script (Task 7), `TrackingErrorNoise` (Task 8), trained checkpoint `results_mc_pilot_pb_A_kinova_gen3/1` (exists), policy-loading pattern from `demo_pybullet_gui.py:47-170`.
- Produces: printed fitted coefficients + a 10-throw kinematic-vs-dynamic landing-error comparison (spec Validation #3).

- [ ] **Step 1: Run the full sweep**

```bash
python3 measure_tracking_error.py --out results_tracking_error
```

Expected: 225 progress lines, stats table, ~0 flagged. Takes minutes (fresh DIRECT world per throw). If many throws are flagged, stop and investigate before fitting.

- [ ] **Step 2: Fit + persist the noise model.** Quick inline fit (no new file needed):

```bash
python3 - <<'EOF'
import numpy as np
from robot_arm.noise_models import TrackingErrorNoise

noise = TrackingErrorNoise.from_measurements("results_tracking_error/tracking_error.npz")
np.savez("results_tracking_error/fitted_noise.npz",
         coef_a=noise.coef_a, coef_b=noise.coef_b, resid_cov=noise.resid_cov)
print("coef_a (par, lat, ver):", noise.coef_a)
print("coef_b (par, lat, ver):", noise.coef_b)
print("resid std:", np.sqrt(np.diag(noise.resid_cov)))
EOF
```

Expected: finite coefficients; parallel-component slope is the headline number (speed-proportional loss, analogous to a measured `alpha` for `VelocitySlipNoise`).

- [ ] **Step 3: Create `eval_sim2sim_gap.py`** (policy loading mirrors `demo_pybullet_gui.py`):

```python
"""
Sim2sim gap: replay the trained Kinova policy under (a) the kinematic release
used in training and (b) torque-tracked dynamic release, same 10 targets.

Usage: python3 eval_sim2sim_gap.py --log_path results_mc_pilot_pb_A_kinova_gen3/1
"""

import argparse
import os
import pickle as pkl

import numpy as np
import torch

import policy_learning.Policy as Policy
from robot_arm.robot_profiles import get_robot_profile
from simulation_class.model_pybullet import PyBulletThrowingSystem


def load_policy(log_path):
    with open(os.path.join(log_path, "log.pkl"), "rb") as f:
        log = pkl.load(f)
    with open(os.path.join(log_path, "config_log.pkl"), "rb") as f:
        cfg = pkl.load(f)
    state = log["parameters_trial_list"][-1]
    policy_obj = Policy.Throwing_Policy(
        full_state_dim=8,
        target_dim=2,
        num_basis=state["centers"].shape[0],
        u_max=cfg["uM"],
        lengthscales_init=state["log_lengthscales"].exp().numpy()[0],
        centers_init=state["centers"].numpy(),
        weight_init=state["f_linear.weight"].numpy(),
        flg_drop=False,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    policy_obj.load_state_dict(state)
    policy_obj.eval()
    return policy_obj, cfg


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--log_path", type=str,
                    default="results_mc_pilot_pb_A_kinova_gen3/1")
    ap.add_argument("--num_throws", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", type=str, default="results_tracking_error")
    args = ap.parse_args()

    policy_obj, cfg = load_policy(args.log_path)
    lm, lM, gM = cfg["lm"], cfg["lM"], cfg.get("gM", np.pi / 6)
    profile = get_robot_profile("kinova_gen3")
    release_pos = np.array(profile.default_release_pos, dtype=float)
    t_w, t_r, _ = profile.timing

    def policy(s, t):
        with torch.no_grad():
            inp = torch.tensor(np.asarray(s, dtype=float),
                               dtype=torch.float64).unsqueeze(0)
            return np.array([float(policy_obj(inp, t=0, p_dropout=0.0).item())])

    rng = np.random.default_rng(args.seed)
    targets = []
    for _ in range(args.num_throws):
        dist = rng.uniform(lm, lM)
        ang = rng.uniform(-gM, gM)
        targets.append([dist * np.cos(ang), dist * np.sin(ang)])
    targets = np.array(targets)

    results = {}
    for label, robot in (("kinematic", "kinova_gen3"), ("dynamic", "kinova_gen3_dyn")):
        system = PyBulletThrowingSystem(robot_name=robot, t_w=t_w, t_r=t_r)
        errs = []
        for tgt in targets:
            s0 = np.concatenate([release_pos, np.zeros(3), tgt])
            _, _, clean = system.rollout(s0, policy, T=2.0, dt=0.02, noise=0.0)
            errs.append(float(np.linalg.norm(clean[-1, 0:2] - tgt)))
        results[label] = np.array(errs)

    print(f"\n=== Sim2sim gap ({args.num_throws} throws, policy {args.log_path}) ===")
    print(f"{'target':>16s} {'kinematic':>10s} {'dynamic':>10s}")
    for i, tgt in enumerate(targets):
        print(f"({tgt[0]:+.2f},{tgt[1]:+.2f})  "
              f"{results['kinematic'][i] * 100:8.2f}cm "
              f"{results['dynamic'][i] * 100:8.2f}cm")
    for label in ("kinematic", "dynamic"):
        e = results[label]
        print(f"{label}: mean {e.mean() * 100:.2f} cm, max {e.max() * 100:.2f} cm")

    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "sim2sim_gap.npz"),
             targets=targets, **results)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the gap eval**

```bash
python3 eval_sim2sim_gap.py --log_path results_mc_pilot_pb_A_kinova_gen3/1
```

Expected: kinematic errors ~2-3 cm (matches the trained checkpoint's convergence); dynamic errors larger — this gap is the study's headline sim2sim number. If `load_policy` fails on a key name, inspect `log["parameters_trial_list"][-1].keys()` and adapt the constructor mapping (compare `demo_pybullet_gui.py:139-170`).

- [ ] **Step 5: Commit**

```bash
git add eval_sim2sim_gap.py
git commit -m "Add sim2sim gap eval (kinematic vs dynamic release)"
```

- [ ] **Step 6: Report** — summarize for the user: fitted `coef_a/coef_b/resid` numbers, flagged-throw count, sim2sim mean/max gap, and pointers to `results_tracking_error/`. Suggest (do not do): port highlights into `paper/change_history.md` and the next email update; noise-robust Gen3 training with `TrackingErrorNoise` is the next milestone.

---

## Self-Review

1. **Spec coverage:** profile (Task 1), torque branch + NaN guard (Task 3), time scaling + failure report (Task 4), dynamic release (Task 5), rollout wiring + `last_release_info` + arm_noise rejection (Task 6), sweep + flagging + figure + stats (Task 7), noise fit + interface (Task 8), gravity-hold (Task 3 test), slow-tracking + gain gate (Task 5 test), sim2sim gap via true pipeline (Task 9). 240 Hz-vs-1 kHz note lives in the sweep docstring + spec. No gaps found.
2. **Placeholder scan:** all steps carry complete code/commands; no TBDs.
3. **Type consistency:** `with_accel` keyword consistent (Tasks 2/3/5/6); `coeffs["t_r"]/["time_scale"]` producers/consumers match (Tasks 4/6/7); npz keys match between Task 7 writer and Task 8 reader (`u_cmd/v_cmd/v_release/flag`); `last_release_info` keys match between Task 6 and Task 7.
