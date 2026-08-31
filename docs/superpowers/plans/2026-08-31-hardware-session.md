# Hardware Throw Session Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One application that runs the start-of-day checks, shows a live annotated view of the thrown ball and its computed landing point, guides an operator through N real throws, and then updates the MC-PILOT model on that real data.

**Architecture:** Single process. Tk dashboard on the main thread; a camera thread that owns the D435i, keeps a dual-IR ring buffer and draws an OpenCV overlay window; a worker thread for blocking arm calls. All new algorithmic work lives in `hardware_learning.py` (pure functions, no GUI, no hardware) so it is testable; `hardware_session.py` is the app that wires it to the existing scripts.

**Tech Stack:** Python 3.10 (`/usr/bin/python3`), numpy, torch 2.9, OpenCV 4.12, pyrealsense2, Tkinter, pytest, kortex_api 2.6.0.post3.

**Spec:** `docs/superpowers/specs/2026-08-31-hardware-session-design.md`

## Global Constraints

- **Always `/usr/bin/python3`.** Bare `python3` resolves to a Conda 3.14 with no cv2/torch/pyrealsense2. Never bare `pip`/`pip3` (Blender snap's 3.13) — use `python3 -m pip`. Never bare `pytest` — use `python3 -m pytest`.
- **Run everything from inside `mc-pilot-pybullet/`.** Imports resolve via `sys.path.append("..")`.
- **One implementation of anything shared.** Import and call `start_of_day.py`, `pickup_and_lift.py`, `run_hardware_throw.py`, `run_closed_loop_throws.py`, `measure_landing.py`, `perception/*`. Never copy their logic. Same rule that governs `simulation_class/release_solver.py` and `perception/wrist_chain.py`.
- **Never resample the fitted parabola into the GP.** `fit_ballistic` is gravity-only; feeding its output back teaches the GP `Δv = g·dt`, its own assumption. Ingest raw RANSAC-inlier triangulated points only.
- **Never overwrite a trained checkpoint.** Policy re-optimization writes a new directory.
- **The confirm checkbox unchecks itself after every run.** Re-affirmed per throw, never once per session.
- **Speed-scale ladder is enforced in code:** 0.15 → 0.30 → 0.60 → 1.00, each requiring a clean logged run at the step below.
- **Ball constants:** tennis ball, `ball_mass=0.0577`, `ball_radius=0.0327`. Matches the trained checkpoint.
- **Frames:** base frame has the base at 0 and the floor at `-base_height` = **−0.433 m**.
- **Trained target band:** `[0.68, 0.74] × [-0.25, +0.25]`.
- **Checkpoint under test:** `results_kinetic_chain_gen3_tcp/1` with `--opt_pose throw_pose_table_tcp.npy --tool_offset_z 0.12`, `u_cap 2.00`.
- **Exact array shapes the model expects** (verified against `PyBulletThrowingSystem.rollout`): `state_samples` is `(n, 8)` = `[x, y, z, vx, vy, vz, Px, Py]`; `input_samples` is `(n, 1)` with the release speed at `[0, 0]` and zeros elsewhere. `T_sampling = Ts = 0.02`.

---

### Task 1: Target spread and the speed-scale escalation gate

The two pure decisions the session makes on the operator's behalf. Both are policy, not plumbing, so they are tested first and separately.

**Files:**
- Create: `mc-pilot-pybullet/hardware_learning.py`
- Test: `mc-pilot-pybullet/tests/test_hardware_learning.py`

**Interfaces:**
- Consumes: nothing
- Produces: `propose_targets(n, band=((0.68, 0.74), (-0.25, 0.25)), seed=0) -> np.ndarray (n, 2)`; `next_allowed_scale(logged_scales, ladder=(0.15, 0.30, 0.60, 1.00)) -> float`; `scale_allowed(requested, logged_scales, ladder=...) -> tuple[bool, str]`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the pure decision + learning logic behind the hardware session."""
import numpy as np
import pytest

from hardware_learning import next_allowed_scale, propose_targets, scale_allowed

BAND = ((0.68, 0.74), (-0.25, 0.25))


def test_propose_targets_stays_inside_the_trained_band():
    t = propose_targets(10, BAND, seed=0)
    assert t.shape == (10, 2)
    assert np.all(t[:, 0] >= 0.68) and np.all(t[:, 0] <= 0.74)
    assert np.all(t[:, 1] >= -0.25) and np.all(t[:, 1] <= 0.25)


def test_propose_targets_actually_spreads():
    """10 near-identical throws teach the GP almost nothing -- that is the point."""
    t = propose_targets(10, BAND, seed=0)
    assert t[:, 1].max() - t[:, 1].min() > 0.30   # uses most of the y range
    assert t[:, 0].max() - t[:, 0].min() > 0.03   # and both ends of the narrow x range


def test_propose_targets_is_deterministic_for_a_seed():
    assert np.allclose(propose_targets(10, BAND, seed=7), propose_targets(10, BAND, seed=7))


def test_escalation_starts_at_the_bottom_of_the_ladder():
    assert next_allowed_scale([]) == pytest.approx(0.15)
    ok, why = scale_allowed(1.00, [])
    assert not ok and "0.15" in why


def test_escalation_advances_one_rung_per_clean_run():
    assert next_allowed_scale([0.15]) == pytest.approx(0.30)
    assert next_allowed_scale([0.15, 0.30]) == pytest.approx(0.60)
    assert next_allowed_scale([0.15, 0.30, 0.60]) == pytest.approx(1.00)


def test_escalation_refuses_skipping_a_rung():
    ok, why = scale_allowed(0.60, [0.15])
    assert not ok and "0.30" in why


def test_escalation_allows_repeating_or_dropping_back():
    assert scale_allowed(0.15, [0.15, 0.30])[0]
    assert scale_allowed(0.30, [0.15, 0.30])[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd mc-pilot-pybullet && /usr/bin/python3 -m pytest tests/test_hardware_learning.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hardware_learning'`

- [ ] **Step 3: Write the implementation**

```python
"""
The parts of the hardware throw session that are decisions or mathematics
rather than plumbing, kept free of Tk, PyBullet and pyrealsense2 so they can be
tested without a window, an arm, or a camera.

See docs/superpowers/specs/2026-08-31-hardware-session-design.md.
"""
from __future__ import annotations

import numpy as np

SCALE_LADDER = (0.15, 0.30, 0.60, 1.00)
TRAINED_BAND = ((0.68, 0.74), (-0.25, 0.25))


def propose_targets(n, band=TRAINED_BAND, seed=0):
    """
    `n` targets spread across the trained band.

    Stratified, not uniform-random: n random draws routinely cluster, and ten
    throws that land in the same place tell the GP almost nothing it does not
    already believe. y gets the stratification because it is the wide axis;
    x is jittered inside its much narrower range.
    """
    (x_lo, x_hi), (y_lo, y_hi) = band
    rng = np.random.default_rng(seed)
    edges = np.linspace(y_lo, y_hi, n + 1)
    ys = edges[:-1] + rng.uniform(0.0, 1.0, n) * np.diff(edges)
    xs = x_lo + rng.uniform(0.0, 1.0, n) * (x_hi - x_lo)
    order = rng.permutation(n)
    return np.stack([xs[order], ys[order]], axis=1)


def next_allowed_scale(logged_scales, ladder=SCALE_LADDER):
    """Highest rung reachable given the clean runs logged so far."""
    best = -1
    for s in logged_scales:
        for i, rung in enumerate(ladder):
            if abs(s - rung) < 1e-9:
                best = max(best, i)
    return ladder[min(best + 1, len(ladder) - 1)]


def scale_allowed(requested, logged_scales, ladder=SCALE_LADDER):
    """
    (ok, reason). The runbook's escalation ladder, enforced rather than advised:
    every rung needs a clean run at the rung below it first. Repeating a rung or
    dropping back down is always fine.
    """
    ceiling = next_allowed_scale(logged_scales, ladder)
    if requested <= ceiling + 1e-9:
        return True, ""
    return False, (f"speed_scale {requested:g} needs a clean logged run at "
                   f"{ceiling:g} first -- escalate one rung at a time")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd mc-pilot-pybullet && /usr/bin/python3 -m pytest tests/test_hardware_learning.py -q`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add mc-pilot-pybullet/hardware_learning.py mc-pilot-pybullet/tests/test_hardware_learning.py
git commit -m "feat(session): target spread + enforced speed-scale escalation ladder"
```

---

### Task 2: Extend the throw record with the fields vision now provides

`build_throw_record` predates the extrinsic and hardcodes `landing_xy=None`. The session needs the measured release state — that is what makes the release model fittable — plus the refusal reason when a track is rejected.

**Files:**
- Modify: `mc-pilot-pybullet/run_closed_loop_throws.py:146-172`
- Test: `mc-pilot-pybullet/tests/test_closed_loop_launch.py` (append)

**Interfaces:**
- Consumes: nothing
- Produces: `build_throw_record(..., landing_xy, measurement=None, release_in_box=None)` where `measurement` is the dict `measure_landing()` returns or `{"refusal_reason": str}`. New record keys: `sigma_xy_m`, `n_frames`, `n_inliers`, `rms_px`, `measured_p0`, `measured_v0`, `refusal_reason`, `release_in_box`.

- [ ] **Step 1: Write the failing tests**

```python
def test_build_throw_record_carries_the_measured_release_state():
    """measured_v0 vs commanded speed is the whole release model -- it must be logged."""
    import numpy as np
    from run_closed_loop_throws import build_throw_record
    meas = {"x": 0.71, "y": 0.02, "t_impact": 0.51, "sigma_xy_m": 0.018,
            "n_frames": 44, "n_inliers": 40, "rms_px": 0.42,
            "p0": np.array([0.30, 0.0, 0.02]), "v0": np.array([1.39, 0.0, 0.37]),
            "max_mask_frac": 0.03}
    r = build_throw_record(0, [0.71, 0.0], 1.44, 0.15, [0.0] * 7, [0.0] * 7,
                           True, {}, "tennis-01", "throws/throw_000.npz",
                           [0.71, 0.02], measurement=meas, release_in_box=True)
    assert r["landing_xy"] == [0.71, 0.02]
    assert r["measured_v0"] == [1.39, 0.0, 0.37]
    assert r["measured_p0"] == [0.30, 0.0, 0.02]
    assert r["sigma_xy_m"] == 0.018
    assert r["n_inliers"] == 40
    assert r["refusal_reason"] is None
    assert r["release_in_box"] is True


def test_build_throw_record_records_a_refusal_without_a_landing():
    """A refused track is still a logged throw -- it is not silently dropped."""
    from run_closed_loop_throws import build_throw_record
    r = build_throw_record(3, [0.71, 0.0], 1.44, 0.15, [0.0] * 7, [0.0] * 7,
                           True, {}, "tennis-01", "throws/throw_003.npz", None,
                           measurement={"refusal_reason": "inlier fraction 0.49"})
    assert r["landing_xy"] is None
    assert "0.49" in r["refusal_reason"]
    assert r["measured_v0"] is None


def test_build_throw_record_is_backward_compatible():
    """The old 11-positional-arg call must keep working unchanged."""
    from run_closed_loop_throws import build_throw_record
    r = build_throw_record(0, [0.75, 0.05], 1.5, 0.15, [0.0] * 7, [0.0] * 7,
                           True, {}, "b", "c.npz", None)
    assert r["landing_xy"] is None and r["measured_v0"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd mc-pilot-pybullet && /usr/bin/python3 -m pytest tests/test_closed_loop_launch.py -q -k throw_record`
Expected: FAIL — `TypeError: build_throw_record() got an unexpected keyword argument 'measurement'`

- [ ] **Step 3: Write the implementation**

Replace the `return {...}` in `build_throw_record` and extend its signature:

```python
def build_throw_record(throw_index, target, commanded_speed, speed_scale,
                       q_release, qd_release, precheck_ok, exec_stats,
                       ball_id, capture_file, landing_xy,
                       measurement=None, release_in_box=None):
    """
    One line of the dataset HARDWARE_RUNBOOK.md Sec 4 describes: "Record per
    throw (this is the dataset, not a debug log)".

    `measurement` is measure_landing()'s dict when a track was accepted, or
    {"refusal_reason": str} when it was refused. A refused throw is still
    logged: it is evidence about the rig, and dropping it would quietly bias
    the dataset toward the throws that happened to track well.

    `measured_v0` is the ball's ACTUAL release velocity from vision. Commanded
    speed vs measured_v0 is where the sim-to-real gap actually lives at these
    speeds (25 ms quantisation ~ 3 cm, drag ~ 5 mm), so it is a first-class
    field, not a diagnostic.
    """
    m = measurement or {}

    def _vec(key):
        v = m.get(key)
        return None if v is None else [float(x) for x in np.asarray(v).reshape(-1)]

    return {
        "throw_index": int(throw_index),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "target": [float(x) for x in target],
        "commanded_speed": float(commanded_speed),
        "speed_scale": float(speed_scale),
        "q_release": [float(x) for x in q_release],
        "qd_release": [float(x) for x in qd_release],
        "precheck_ok": bool(precheck_ok),
        "release_in_box": None if release_in_box is None else bool(release_in_box),
        "exec_stats": exec_stats,
        "ball_id": ball_id,
        "capture_file": capture_file,
        "landing_xy": landing_xy,
        "sigma_xy_m": m.get("sigma_xy_m"),
        "n_frames": m.get("n_frames"),
        "n_inliers": m.get("n_inliers"),
        "rms_px": m.get("rms_px"),
        "measured_p0": _vec("p0"),
        "measured_v0": _vec("v0"),
        "refusal_reason": m.get("refusal_reason"),
    }
```

Add `import numpy as np` to the file's imports if it is not already present.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd mc-pilot-pybullet && /usr/bin/python3 -m pytest tests/test_closed_loop_launch.py -q`
Expected: all pass, including the pre-existing tests

- [ ] **Step 5: Commit**

```bash
git add mc-pilot-pybullet/run_closed_loop_throws.py mc-pilot-pybullet/tests/test_closed_loop_launch.py
git commit -m "feat(session): log measured release state and refusal reason per throw"
```

---

### Task 3: Resample a real track onto the model's Ts grid

The single most dangerous function in this plan: get it wrong and the GP learns an assumption instead of the world, silently. Written before any GUI exists.

**Files:**
- Modify: `mc-pilot-pybullet/hardware_learning.py`
- Test: `mc-pilot-pybullet/tests/test_hardware_learning.py` (append)

**Interfaces:**
- Consumes: `perception.trajectory.ransac_track` output (`inliers`, `FitResult`), `perception.stereo.StereoRig`
- Produces: `track_to_state_samples(points_base, times, target_xy, commanded_speed, ts=0.02) -> (state_samples (n,8), input_samples (n,1))`

- [ ] **Step 1: Write the failing tests**

```python
def test_track_to_state_samples_has_the_exact_shapes_the_model_expects():
    """(n, 8) = [x,y,z,vx,vy,vz,Px,Py] and (n, 1) with the speed only at t=0 --
    verified against PyBulletThrowingSystem.rollout, not assumed."""
    from hardware_learning import track_to_state_samples
    t = np.arange(0.0, 0.50, 1 / 90.0)
    p0, v0, g = np.array([0.3, 0.0, 0.02]), np.array([1.39, 0.0, 0.37]), np.array([0, 0, -9.81])
    pts = p0 + np.outer(t, v0) + 0.5 * np.outer(t ** 2, g)
    s, u = track_to_state_samples(pts, t, (0.71, 0.02), 1.44, ts=0.02)
    assert s.shape[1] == 8 and u.shape[1] == 1
    assert s.shape[0] == u.shape[0]
    assert u[0, 0] == pytest.approx(1.44)
    assert np.allclose(u[1:, 0], 0.0)
    assert np.allclose(s[:, 6], 0.71) and np.allclose(s[:, 7], 0.02)


def test_track_to_state_samples_recovers_a_known_velocity_profile():
    from hardware_learning import track_to_state_samples
    t = np.arange(0.0, 0.50, 1 / 90.0)
    p0, v0, g = np.array([0.3, 0.0, 0.02]), np.array([1.39, 0.0, 0.37]), np.array([0, 0, -9.81])
    pts = p0 + np.outer(t, v0) + 0.5 * np.outer(t ** 2, g)
    s, _ = track_to_state_samples(pts, t, (0.71, 0.02), 1.44, ts=0.02)
    assert np.allclose(s[0, 0:3], p0, atol=2e-3)
    assert np.allclose(s[0, 3:6], v0, atol=2e-2)
    dt = 0.02
    dv = (s[1:, 3:6] - s[:-1, 3:6]) / dt
    assert np.allclose(dv[:, 2].mean(), -9.81, atol=0.5)


def test_track_to_state_samples_is_sampled_at_ts_not_at_camera_rate():
    """90 fps in, 50 Hz out -- the GP's propagation assumes Ts spacing."""
    from hardware_learning import track_to_state_samples
    t = np.arange(0.0, 0.50, 1 / 90.0)
    pts = np.stack([t * 1.4, t * 0, 0.02 - 4.9 * t ** 2], axis=1)
    s, _ = track_to_state_samples(pts, t, (0.71, 0.0), 1.44, ts=0.02)
    assert 24 <= s.shape[0] <= 26        # 0.50 s / 0.02 s


def test_track_to_state_samples_rejects_a_track_too_short_to_difference():
    from hardware_learning import track_to_state_samples
    t = np.array([0.0, 0.01])
    pts = np.zeros((2, 3))
    with pytest.raises(ValueError, match="too short"):
        track_to_state_samples(pts, t, (0.71, 0.0), 1.44, ts=0.02)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd mc-pilot-pybullet && /usr/bin/python3 -m pytest tests/test_hardware_learning.py -q -k track_to_state`
Expected: FAIL — `ImportError: cannot import name 'track_to_state_samples'`

- [ ] **Step 3: Write the implementation**

Append to `hardware_learning.py`:

```python
TS_DEFAULT = 0.02          # T_sampling, must match the trained checkpoint


def track_to_state_samples(points_base, times, target_xy, commanded_speed,
                           ts=TS_DEFAULT):
    """
    A measured flight -> the (state_samples, input_samples) pair the model's
    `add_data` consumes, shaped exactly as `PyBulletThrowingSystem.rollout`
    returns them: states (n, 8) = [x, y, z, vx, vy, vz, Px, Py], inputs (n, 1)
    with the release speed at [0, 0] and zeros after.

    `points_base` MUST be the raw triangulated RANSAC inliers, never a
    resampled `fit_ballistic` output. The fit is gravity-only, so feeding it
    back would hand the GP `dv = g*dt` -- its own assumption returned as
    evidence. That failure is invisible in the cost curve, which is exactly why
    it gets a test (`test_pure_parabola_teaches_the_gp_nothing`).

    Position is resampled by local linear interpolation onto the Ts grid and
    velocity by central differences of the resampled positions, so the states
    obey the same `p_{t+1} = p_t + Ts*v_t + (Ts/2)*dv` relation the GP's
    propagation assumes (paper Eq. 18).
    """
    p = np.asarray(points_base, float)
    t = np.asarray(times, float).reshape(-1)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"expected (N, 3) points, got {p.shape}")
    if t.size < 4:
        raise ValueError(f"track too short to difference: {t.size} samples")

    t0 = t - t[0]
    grid = np.arange(0.0, t0[-1] + 1e-12, ts)
    if grid.size < 3:
        raise ValueError(f"track too short to difference: spans {t0[-1]:.3f} s at ts={ts}")

    pos = np.stack([np.interp(grid, t0, p[:, k]) for k in range(3)], axis=1)
    vel = np.gradient(pos, ts, axis=0, edge_order=2)

    n = grid.size
    states = np.zeros((n, 8))
    states[:, 0:3] = pos
    states[:, 3:6] = vel
    states[:, 6] = float(target_xy[0])
    states[:, 7] = float(target_xy[1])

    inputs = np.zeros((n, 1))
    inputs[0, 0] = float(commanded_speed)
    return states, inputs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd mc-pilot-pybullet && /usr/bin/python3 -m pytest tests/test_hardware_learning.py -q`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add mc-pilot-pybullet/hardware_learning.py mc-pilot-pybullet/tests/test_hardware_learning.py
git commit -m "feat(session): resample a measured flight onto the model's Ts grid"
```

---

### Task 4: The above-noise verdict

Turns "did the GP learn anything?" from a judgement call into a printed number with a threshold. This is the task that decides whether the session's headline result is honest.

**Files:**
- Modify: `mc-pilot-pybullet/hardware_learning.py`
- Test: `mc-pilot-pybullet/tests/test_hardware_learning.py` (append)

**Interfaces:**
- Consumes: `track_to_state_samples` output
- Produces: `velocity_noise_sigma(pos_sigma_m=0.0206, ts=0.02) -> float`; `deviation_verdict(dv_learned, sigma_v, k=2.0) -> dict` with keys `rms_deviation`, `rms_sigma`, `ratio`, `above_noise`, `text`

- [ ] **Step 1: Write the failing tests**

```python
def test_velocity_noise_sigma_propagates_position_noise_through_differencing():
    """18mm extrinsic (+) 10mm triangulation, differenced over 20ms, is large."""
    from hardware_learning import velocity_noise_sigma
    s = velocity_noise_sigma(pos_sigma_m=0.0206, ts=0.02)
    assert s > 0.5          # m/s -- differencing cm-scale noise at 50 Hz is brutal
    assert velocity_noise_sigma(0.0206, 0.04) < s     # longer baseline, less noise


def test_verdict_is_below_noise_when_the_signal_is_smaller_than_sigma():
    """The expected real-world answer for a tennis ball: drag ~5mm, noise ~18mm."""
    from hardware_learning import deviation_verdict
    v = deviation_verdict(dv_learned=np.full((40, 3), 0.01), sigma_v=0.5)
    assert not v["above_noise"]
    assert "BELOW NOISE" in v["text"]
    assert v["ratio"] < 1.0


def test_verdict_is_above_noise_only_past_the_2x_threshold():
    from hardware_learning import deviation_verdict
    just_under = deviation_verdict(np.full((40, 1), 0.99), sigma_v=0.5, k=2.0)
    just_over = deviation_verdict(np.full((40, 1), 1.01), sigma_v=0.5, k=2.0)
    assert not just_under["above_noise"]
    assert just_over["above_noise"]
    assert "ABOVE NOISE" in just_over["text"]


def test_verdict_text_always_reports_both_numbers_and_the_count():
    """A verdict without its evidence is exactly the kind of number this repo
    has been bitten by before."""
    from hardware_learning import deviation_verdict
    v = deviation_verdict(np.full((37, 3), 0.02), sigma_v=0.5)
    assert "37" in v["text"]
    assert f"{v['rms_deviation']:.4f}" in v["text"]
    assert f"{v['rms_sigma']:.4f}" in v["text"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd mc-pilot-pybullet && /usr/bin/python3 -m pytest tests/test_hardware_learning.py -q -k "verdict or noise_sigma"`
Expected: FAIL — `ImportError: cannot import name 'velocity_noise_sigma'`

- [ ] **Step 3: Write the implementation**

Append to `hardware_learning.py`:

```python
# 18 mm extrinsic repeatability (measured 2026-08-31, 5 solves, static rig)
# combined with ~10 mm stereo triangulation noise at 1.6 m.
POS_SIGMA_M = float(np.hypot(0.018, 0.010))


def velocity_noise_sigma(pos_sigma_m=POS_SIGMA_M, ts=TS_DEFAULT):
    """
    Position noise propagated into a central-difference velocity.

    v_k = (p_{k+1} - p_{k-1}) / (2*ts), so sigma_v = sqrt(2)*sigma_p / (2*ts).
    At 2.06 cm and 50 Hz this is ~0.73 m/s, which is half the release speed --
    the honest reason a per-sample velocity from this rig cannot resolve drag.
    """
    return float(np.sqrt(2.0) * pos_sigma_m / (2.0 * ts))


def deviation_verdict(dv_learned, sigma_v=None, k=2.0):
    """
    Is the non-ballistic correction the GP claims to have found bigger than the
    noise it was fitted through?

    `dv_learned` is the GP's predicted delta-v minus the pure-gravity delta-v,
    i.e. only the part that is not already assumed. ABOVE NOISE requires
    RMS(deviation) > k * RMS(sigma). Both numbers and the sample count go into
    the text, always -- a verdict without its evidence is how a noise-sized
    number becomes a claimed discovery.
    """
    d = np.asarray(dv_learned, float)
    sigma_v = velocity_noise_sigma() if sigma_v is None else float(sigma_v)
    rms_d = float(np.sqrt(np.mean(d ** 2)))
    ratio = rms_d / (k * sigma_v) if sigma_v > 0 else np.inf
    above = rms_d > k * sigma_v
    text = (f"{'ABOVE NOISE' if above else 'BELOW NOISE'}: "
            f"RMS deviation {rms_d:.4f} m/s vs {k:g}x RMS sigma "
            f"{sigma_v:.4f} m/s over {d.shape[0]} samples "
            f"(ratio {ratio:.2f}). "
            + ("The GP found structure the noise cannot explain."
               if above else
               "The GP learned nothing distinguishable from measurement noise -- "
               "expected for a tennis ball at this speed, where drag displaces "
               "~5 mm against ~21 mm of position noise. Report it as such."))
    return {"rms_deviation": rms_d, "rms_sigma": sigma_v, "ratio": ratio,
            "above_noise": bool(above), "n_samples": int(d.shape[0]), "text": text}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd mc-pilot-pybullet && /usr/bin/python3 -m pytest tests/test_hardware_learning.py -q`
Expected: 15 passed

- [ ] **Step 5: Commit**

```bash
git add mc-pilot-pybullet/hardware_learning.py mc-pilot-pybullet/tests/test_hardware_learning.py
git commit -m "feat(session): above/below-noise verdict for the learned flight correction"
```

---

### Task 5: The release model, and the tautology guard

The release discrepancy is the term that is actually measurable. The tautology guard test belongs here because it needs a real GP.

**Files:**
- Modify: `mc-pilot-pybullet/hardware_learning.py`
- Test: `mc-pilot-pybullet/tests/test_hardware_learning.py` (append)

**Interfaces:**
- Consumes: JSONL records from Task 2
- Produces: `fit_release_model(records) -> dict` with keys `gain`, `offset`, `residual_sigma`, `n`, `direction_error_deg`, `text`

- [ ] **Step 1: Write the failing tests**

```python
def test_fit_release_model_recovers_a_known_gain_and_offset():
    from hardware_learning import fit_release_model
    rng = np.random.default_rng(0)
    recs = []
    for c in np.linspace(1.2, 1.6, 12):
        actual = 0.90 * c + 0.05
        recs.append({"commanded_speed": float(c),
                     "measured_v0": [float(actual), 0.0, 0.0],
                     "landing_xy": [0.7, 0.0]})
    out = fit_release_model(recs)
    assert out["gain"] == pytest.approx(0.90, abs=1e-6)
    assert out["offset"] == pytest.approx(0.05, abs=1e-6)
    assert out["n"] == 12


def test_fit_release_model_skips_refused_throws():
    """A throw with no measurement carries no release information."""
    from hardware_learning import fit_release_model
    recs = [{"commanded_speed": 1.4, "measured_v0": [1.31, 0, 0], "landing_xy": [0.7, 0]},
            {"commanded_speed": 1.5, "measured_v0": None, "landing_xy": None},
            {"commanded_speed": 1.6, "measured_v0": [1.49, 0, 0], "landing_xy": [0.7, 0]}]
    assert fit_release_model(recs)["n"] == 2


def test_fit_release_model_refuses_to_fit_too_few_points():
    from hardware_learning import fit_release_model
    recs = [{"commanded_speed": 1.4, "measured_v0": [1.3, 0, 0], "landing_xy": [0.7, 0]}]
    with pytest.raises(ValueError, match="at least 3"):
        fit_release_model(recs)


def test_pure_parabola_teaches_the_gp_nothing():
    """
    THE TAUTOLOGY GUARD. A gravity-only track must produce a learned deviation
    that the verdict calls BELOW NOISE. If someone resamples fit_ballistic's
    output into add_data, this is what should catch it.
    """
    from hardware_learning import deviation_verdict, track_to_state_samples
    t = np.arange(0.0, 0.50, 1 / 90.0)
    p0, v0, g = np.array([0.3, 0.0, 0.02]), np.array([1.39, 0.0, 0.37]), np.array([0, 0, -9.81])
    pts = p0 + np.outer(t, v0) + 0.5 * np.outer(t ** 2, g)
    s, _ = track_to_state_samples(pts, t, (0.71, 0.0), 1.44)

    dv = np.diff(s[:, 3:6], axis=0)
    dv_gravity = np.tile(g * 0.02, (dv.shape[0], 1))
    verdict = deviation_verdict(dv - dv_gravity)
    assert not verdict["above_noise"], (
        "a pure-gravity track produced a deviation the verdict called real -- "
        "the guard against feeding the fitted parabola back has broken")


def test_injected_drag_is_recovered_when_noise_is_set_below_it():
    """The verdict must also be able to say yes, or it is not a test."""
    from hardware_learning import deviation_verdict
    dv = np.full((40, 3), 0.05)
    assert deviation_verdict(dv, sigma_v=0.001)["above_noise"]
    assert not deviation_verdict(dv, sigma_v=1.0)["above_noise"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd mc-pilot-pybullet && /usr/bin/python3 -m pytest tests/test_hardware_learning.py -q -k "release_model or tautology or injected_drag"`
Expected: FAIL — `ImportError: cannot import name 'fit_release_model'`

- [ ] **Step 3: Write the implementation**

Append to `hardware_learning.py`:

```python
def fit_release_model(records):
    """
    Commanded release speed -> measured release speed, over the logged throws.

    This is the term worth fitting. At this speed the flight is ballistic to
    within ~5 mm while the release carries 2.9-3.7 cm of command quantisation
    plus ~1 cm of gripper-latency residual, so the discrepancy between what the
    policy asked for and what the ball actually left with is both large and
    directly observable in `measured_v0`.

    Refused throws (no measurement) are skipped, not imputed.
    """
    cmd, meas, dirs = [], [], []
    for r in records:
        v0 = r.get("measured_v0")
        if v0 is None or r.get("commanded_speed") is None:
            continue
        v = np.asarray(v0, float)
        cmd.append(float(r["commanded_speed"]))
        meas.append(float(np.linalg.norm(v)))
        dirs.append(v / max(np.linalg.norm(v), 1e-12))
    if len(cmd) < 3:
        raise ValueError(f"need at least 3 measured throws to fit a release "
                         f"model, have {len(cmd)}")

    c = np.asarray(cmd)
    m = np.asarray(meas)
    A = np.stack([c, np.ones_like(c)], axis=1)
    (gain, offset), *_ = np.linalg.lstsq(A, m, rcond=None)
    resid = m - (gain * c + offset)
    sigma = float(np.std(resid, ddof=min(2, len(c) - 1)))

    d = np.asarray(dirs)
    mean_dir = d.mean(axis=0)
    mean_dir /= max(np.linalg.norm(mean_dir), 1e-12)
    spread = np.degrees(np.arccos(np.clip(d @ mean_dir, -1, 1))).max()

    return {"gain": float(gain), "offset": float(offset),
            "residual_sigma": sigma, "n": int(len(c)),
            "direction_error_deg": float(spread),
            "text": (f"measured |v0| = {gain:.4f} * commanded + {offset:+.4f} m/s, "
                     f"residual sigma {sigma:.4f} m/s over {len(c)} throws; "
                     f"release direction spread {spread:.2f} deg")}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd mc-pilot-pybullet && /usr/bin/python3 -m pytest tests/test_hardware_learning.py -q`
Expected: 20 passed

- [ ] **Step 5: Commit**

```bash
git add mc-pilot-pybullet/hardware_learning.py mc-pilot-pybullet/tests/test_hardware_learning.py
git commit -m "feat(session): release-discrepancy model + tautology guard test"
```

---

### Task 6: Camera manager — ring buffer and release-window extraction

The threading core. Owning the camera is what lets the session record around the *known* release instant instead of guessing from disparity.

**Files:**
- Create: `mc-pilot-pybullet/session_camera.py`
- Test: `mc-pilot-pybullet/tests/test_session_camera.py`

**Interfaces:**
- Consumes: `perception.ir_capture.IRRecorder`
- Produces: `RingBuffer(seconds, fps)` with `.append(ts, ir1, ir2)`, `.window(t_start, t_end) -> dict` (keys `ts`, `ir1`, `ir2`, matching what `measure_landing.build_observations` expects); `CameraThread(on_frame=None)` with `.start()`, `.stop()`, `.mark_release(t) -> None`, `.pop_event(timeout) -> dict | None`, `.latest() -> tuple | None`

- [ ] **Step 1: Write the failing tests**

The ring buffer is pure and gets real tests; the thread gets a construction/lifecycle test only, since it needs a camera.

```python
"""Ring buffer + camera-thread lifecycle. No camera is opened by these tests."""
import numpy as np
import pytest

from session_camera import RingBuffer


def _frame(v):
    return np.full((8, 8), v, np.uint8)


def test_ring_buffer_keeps_only_its_window():
    rb = RingBuffer(seconds=0.1, fps=100)      # capacity 10
    for k in range(25):
        rb.append(k * 0.01, _frame(k), _frame(k))
    assert len(rb) == 10
    assert rb.window(0.0, 1.0)["ts"][0] == pytest.approx(0.15)


def test_window_extracts_the_requested_span_inclusive():
    rb = RingBuffer(seconds=10.0, fps=100)
    for k in range(100):
        rb.append(k * 0.01, _frame(k), _frame(k))
    w = rb.window(0.20, 0.30)
    assert w["ts"][0] >= 0.20 - 1e-9 and w["ts"][-1] <= 0.30 + 1e-9
    assert w["ir1"].shape[0] == w["ts"].size == w["ir2"].shape[0]
    assert w["ir1"].shape[1:] == (8, 8)


def test_window_returns_stacked_arrays_shaped_for_build_observations():
    """measure_landing.build_observations indexes rec['ir1'][k] -- so ir1 must
    be a stacked (N, H, W) array, not a list."""
    rb = RingBuffer(seconds=10.0, fps=100)
    for k in range(10):
        rb.append(k * 0.01, _frame(k), _frame(k))
    w = rb.window(0.0, 0.09)
    assert isinstance(w["ir1"], np.ndarray) and w["ir1"].ndim == 3


def test_window_raises_when_the_span_holds_no_frames():
    rb = RingBuffer(seconds=10.0, fps=100)
    rb.append(0.0, _frame(1), _frame(1))
    with pytest.raises(ValueError, match="no frames"):
        rb.window(5.0, 6.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd mc-pilot-pybullet && /usr/bin/python3 -m pytest tests/test_session_camera.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'session_camera'`

- [ ] **Step 3: Write the implementation**

```python
"""
The session's camera thread: one owner of the D435i for the whole session.

WHY THE SESSION OWNS THE CAMERA
--------------------------------
The D435i can be opened by exactly one process, which is why
`throw_capture.py` and `run_closed_loop_throws.py` are decoupled through files
today. Owning it here buys correctness, not tidiness: the session ISSUES the
throw, so it knows the release instant exactly and records a window around it,
instead of inferring "that blob was probably a throw" from a disparity gate.
"""
from __future__ import annotations

import collections
import queue
import threading
import time

import numpy as np

PRE_S = 0.45          # kept before release
POST_S = 1.00         # kept after


class RingBuffer:
    """Fixed-duration dual-IR history. Not thread-safe; the owner locks."""

    def __init__(self, seconds=3.0, fps=90):
        self.capacity = max(1, int(round(seconds * fps)))
        self._ts = collections.deque(maxlen=self.capacity)
        self._ir1 = collections.deque(maxlen=self.capacity)
        self._ir2 = collections.deque(maxlen=self.capacity)

    def __len__(self):
        return len(self._ts)

    def append(self, ts, ir1, ir2):
        self._ts.append(float(ts))
        self._ir1.append(ir1)
        self._ir2.append(ir2)

    def window(self, t_start, t_end):
        """
        Inclusive [t_start, t_end] slice, stacked into the dict shape
        measure_landing.build_observations expects (`ts`, `ir1`, `ir2`).
        """
        ts = np.asarray(self._ts, float)
        keep = np.nonzero((ts >= t_start - 1e-9) & (ts <= t_end + 1e-9))[0]
        if keep.size == 0:
            raise ValueError(
                f"no frames in [{t_start:.3f}, {t_end:.3f}] -- buffer holds "
                f"{len(self)} frames spanning "
                f"[{ts[0]:.3f}, {ts[-1]:.3f}]" if len(self) else "buffer is empty")
        ir1 = np.stack([self._ir1[i] for i in keep])
        ir2 = np.stack([self._ir2[i] for i in keep])
        return {"ts": ts[keep], "ir1": ir1, "ir2": ir2}


class CameraThread:
    """
    Continuous dual-IR capture into a ring buffer, plus a single-slot handoff of
    the newest frame for display.

    Single-slot is deliberate: a slow consumer drops frames rather than
    stalling capture, because a stalled capture loses the throw.
    """

    def __init__(self, seconds=3.0, fps=90, width=848, height=480,
                 exposure_us=2000, emitter=True, on_frame=None):
        self.buf = RingBuffer(seconds, fps)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._events = queue.Queue()
        self._latest = None
        self._cfg = dict(width=width, height=height, fps=fps,
                         exposure_us=exposure_us, emitter=emitter)
        self._on_frame = on_frame
        self._thread = None
        self.error = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def latest(self):
        with self._lock:
            return self._latest

    def mark_release(self, t_release, pre=PRE_S, post=POST_S):
        """
        Called by the worker thread the instant the throw fires. The window is
        cut once `post` seconds of frames past the release have actually
        arrived, then pushed to `pop_event`.
        """
        threading.Thread(target=self._cut, args=(t_release, pre, post),
                         daemon=True).start()

    def _cut(self, t_release, pre, post):
        deadline = t_release + post
        while time.time() < deadline + 0.05 and not self._stop.is_set():
            time.sleep(0.01)
        try:
            with self._lock:
                rec = self.buf.window(t_release - pre, t_release + post)
            self._events.put({"t_release": t_release, "rec": rec})
        except ValueError as e:
            self._events.put({"t_release": t_release, "error": str(e)})

    def pop_event(self, timeout=None):
        try:
            return self._events.get(timeout=timeout)
        except queue.Empty:
            return None

    def _run(self):
        from perception.ir_capture import IRRecorder
        try:
            with IRRecorder(width=self._cfg["width"], height=self._cfg["height"],
                            fps=self._cfg["fps"],
                            exposure_us=self._cfg["exposure_us"],
                            emitter=self._cfg["emitter"]) as rec:
                for ts, ir1, ir2 in rec.stream():
                    if self._stop.is_set():
                        break
                    with self._lock:
                        self.buf.append(ts, ir1, ir2)
                        self._latest = (ts, ir1, ir2)
                    if self._on_frame is not None:
                        self._on_frame(ts, ir1, ir2)
        except Exception as e:                     # a camera fault ends the session
            self.error = e
```

> **Note for the implementer:** `IRRecorder` currently exposes `record(seconds)`, not `stream()`. Add a `stream()` generator to `perception/ir_capture.py` that yields `(timestamp, ir1, ir2)` per frame and have `record()` call it, so there is one capture loop rather than two. Do this as part of this task and include it in the commit.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd mc-pilot-pybullet && /usr/bin/python3 -m pytest tests/test_session_camera.py -q`
Expected: 4 passed

- [ ] **Step 5: Verify against the real camera**

Run: `cd mc-pilot-pybullet && /usr/bin/python3 -c "
import time, session_camera
c = session_camera.CameraThread(); c.start(); time.sleep(3)
print('frames buffered:', len(c.buf), 'error:', c.error)
t = time.time(); c.mark_release(t - 0.5)
ev = c.pop_event(timeout=5); c.stop()
print('event frames:', None if ev is None else ev.get('rec', {}).get('ts', []).size, ev.get('error'))
"`
Expected: several hundred frames buffered, `error: None`, event with >50 frames

- [ ] **Step 6: Commit**

```bash
git add mc-pilot-pybullet/session_camera.py mc-pilot-pybullet/tests/test_session_camera.py mc-pilot-pybullet/perception/ir_capture.py
git commit -m "feat(session): camera thread with ring buffer and release-window extraction"
```

---

### Task 7: Live annotated overlay window

**Files:**
- Create: `mc-pilot-pybullet/session_overlay.py`
- Test: `mc-pilot-pybullet/tests/test_session_overlay.py`

**Interfaces:**
- Consumes: `perception.ball_track.detect_candidates`, `perception.stereo.StereoRig`
- Produces: `render_overlay(ir1, ir2, detections=None, path_px=None, landing=None, status="") -> np.ndarray (H, 2W, 3)`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from session_overlay import render_overlay


def test_overlay_returns_a_side_by_side_colour_image():
    ir1 = np.zeros((480, 848), np.uint8)
    ir2 = np.zeros((480, 848), np.uint8)
    out = render_overlay(ir1, ir2)
    assert out.shape == (480, 1696, 3) and out.dtype == np.uint8


def test_overlay_draws_something_where_a_detection_is():
    ir1 = np.zeros((480, 848), np.uint8)
    ir2 = np.zeros((480, 848), np.uint8)
    blank = render_overlay(ir1, ir2)
    marked = render_overlay(ir1, ir2, detections=([(400.0, 200.0, 30.0)], []))
    assert not np.array_equal(blank, marked)
    assert marked[190:215, 385:415].any()


def test_overlay_renders_the_landing_text_when_given_one():
    ir1 = np.zeros((480, 848), np.uint8)
    out = render_overlay(ir1, ir1, landing=(0.712, 0.021, 0.018))
    assert out.any()      # text was drawn somewhere on an otherwise black frame


def test_overlay_survives_an_empty_path():
    ir1 = np.zeros((480, 848), np.uint8)
    assert render_overlay(ir1, ir1, path_px=np.zeros((0, 2))).shape == (480, 1696, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd mc-pilot-pybullet && /usr/bin/python3 -m pytest tests/test_session_overlay.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'session_overlay'`

- [ ] **Step 3: Write the implementation**

```python
"""
The live view. Drawing only -- no capture, no camera, no Tk, so it can be
tested on synthetic arrays.

Layered deliberately: per-frame detections are drawn even when no track has been
fitted yet, so a detection failure is visible AS IT HAPPENS rather than only
surfacing later as a refusal from measure_landing.
"""
from __future__ import annotations

import cv2
import numpy as np

_GREEN, _CYAN, _MAGENTA, _WHITE = (0, 255, 0), (255, 255, 0), (255, 0, 255), (255, 255, 255)


def render_overlay(ir1, ir2, detections=None, path_px=None, landing=None, status=""):
    """
    (left IR, right IR) -> one side-by-side BGR frame with the overlay drawn.

    `detections` is (left_list, right_list) of (u, v, area_px).
    `path_px` is an (N, 2) array of the triangulated track projected into the
    LEFT image. `landing` is (x, y, sigma) in base-frame metres.
    """
    left = cv2.cvtColor(np.asarray(ir1), cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(np.asarray(ir2), cv2.COLOR_GRAY2BGR)

    if detections is not None:
        for img, dets in zip((left, right), detections):
            for (u, v, area) in dets:
                r = int(max(6, np.sqrt(max(area, 1.0) / np.pi) * 2))
                cv2.circle(img, (int(round(u)), int(round(v))), r, _GREEN, 2)

    if path_px is not None and len(path_px) >= 2:
        pts = np.asarray(path_px, np.int32).reshape(-1, 1, 2)
        cv2.polylines(left, [pts], False, _CYAN, 2)

    canvas = np.hstack([left, right])

    if landing is not None:
        x, y, sigma = landing
        cv2.putText(canvas, f"landing  x={x:+.3f}  y={y:+.3f}  sigma={sigma*100:.1f}cm",
                    (12, canvas.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    _MAGENTA, 2, cv2.LINE_AA)
    if status:
        cv2.putText(canvas, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    _WHITE, 2, cv2.LINE_AA)
    return canvas
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd mc-pilot-pybullet && /usr/bin/python3 -m pytest tests/test_session_overlay.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add mc-pilot-pybullet/session_overlay.py mc-pilot-pybullet/tests/test_session_overlay.py
git commit -m "feat(session): live annotated dual-IR overlay renderer"
```

---

### Task 8: The session app — stage 0, throw cycle, dashboard

Wires Tasks 1–7 to the existing scripts. Phase A ends here: after this task the operator can run a real session and collect the dataset, with the MC buttons still disabled.

**Files:**
- Create: `mc-pilot-pybullet/hardware_session.py`
- Test: `mc-pilot-pybullet/tests/test_hardware_session.py`

**Interfaces:**
- Consumes: `hardware_learning.propose_targets/scale_allowed`, `session_camera.CameraThread`, `session_overlay.render_overlay`, `start_of_day` stage functions, `pickup_and_lift.pickup_and_lift`, `run_hardware_throw` planner/executor, `run_closed_loop_throws.build_throw_record/append_log`, `measure_landing.measure_landing`
- Produces: `SessionState` (dataclass) and `ThrowCycle.step_*` methods; `main(argv=None) -> int`

- [ ] **Step 1: Write the failing tests**

Only the state machine is testable without hardware; that is where the safety rules live, so that is what gets tested.

```python
"""State-machine and safety-gate tests. No Tk window, arm, or camera is created."""
import pytest

from hardware_session import SessionState, Stage


def test_session_starts_cold_with_throwing_disabled():
    s = SessionState()
    assert s.stage is Stage.COLD
    assert not s.can_throw()


def test_no_go_from_stage_zero_blocks_throwing():
    s = SessionState()
    s.record_startup(go=False, failures=["FLOOR: -14.0 cm"])
    assert s.stage is Stage.BLOCKED
    assert not s.can_throw()
    assert "FLOOR" in s.blocked_reason


def test_go_then_camera_makes_the_session_ready():
    s = SessionState()
    s.record_startup(go=True, failures=[])
    assert s.stage is Stage.CALIBRATED and not s.can_throw()
    s.camera_ready()
    assert s.stage is Stage.READY and s.can_throw()


def test_confirm_resets_after_every_throw():
    """Re-affirmed per throw, never once per session."""
    s = SessionState()
    s.record_startup(go=True, failures=[])
    s.camera_ready()
    s.confirmed = True
    s.record_throw({"speed_scale": 0.15, "landing_xy": [0.7, 0.0]})
    assert s.confirmed is False


def test_speed_scale_ladder_is_enforced_by_the_session():
    s = SessionState()
    s.record_startup(go=True, failures=[])
    s.camera_ready()
    ok, why = s.check_scale(1.00)
    assert not ok and "0.15" in why
    s.record_throw({"speed_scale": 0.15, "landing_xy": [0.7, 0.0]})
    assert s.check_scale(0.30)[0]
    assert not s.check_scale(0.60)[0]


def test_a_refused_measurement_still_counts_as_a_logged_throw():
    s = SessionState()
    s.record_startup(go=True, failures=[])
    s.camera_ready()
    s.record_throw({"speed_scale": 0.15, "landing_xy": None,
                    "refusal_reason": "inlier fraction 0.49"})
    assert s.n_throws == 1
    assert s.n_measured == 0


def test_model_update_needs_enough_measured_throws():
    s = SessionState(min_throws_for_update=3)
    s.record_startup(go=True, failures=[])
    s.camera_ready()
    for _ in range(2):
        s.record_throw({"speed_scale": 0.15, "landing_xy": [0.7, 0.0]})
    assert not s.can_update_model()
    s.record_throw({"speed_scale": 0.15, "landing_xy": [0.7, 0.0]})
    assert s.can_update_model()


def test_policy_button_is_locked_until_the_model_is_updated():
    s = SessionState(min_throws_for_update=1)
    s.record_startup(go=True, failures=[])
    s.camera_ready()
    s.record_throw({"speed_scale": 0.15, "landing_xy": [0.7, 0.0]})
    assert not s.can_reoptimize_policy()
    s.record_model_update()
    assert s.can_reoptimize_policy()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd mc-pilot-pybullet && /usr/bin/python3 -m pytest tests/test_hardware_session.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hardware_session'`

- [ ] **Step 3: Write the state machine**

Create `hardware_session.py` beginning with the testable core (the Tk app is added in Step 5):

```python
"""
The hardware throw session: cold rig -> calibrated -> N real throws -> a model
update on that data.

Spec: docs/superpowers/specs/2026-08-31-hardware-session-design.md

Stage 0 runs BEFORE the camera thread starts, because calibration needs colour
at 1920x1080 while tracking needs IR at 848x480/90fps. Sequential, so there is
no stream reconfiguration mid-session and start_of_day.py is reused exactly as
it is, opening and closing the camera itself.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Optional

from hardware_learning import scale_allowed


class Stage(enum.Enum):
    COLD = "cold"
    BLOCKED = "blocked"
    CALIBRATED = "calibrated"
    READY = "ready"
    MODEL_UPDATED = "model_updated"


@dataclass
class SessionState:
    """
    Every safety rule the session enforces lives here, and nowhere else, so it
    can be tested without a window or an arm.
    """
    min_throws_for_update: int = 5
    stage: Stage = Stage.COLD
    blocked_reason: str = ""
    confirmed: bool = False
    throws: List[dict] = field(default_factory=list)
    model_updated: bool = False

    @property
    def n_throws(self):
        return len(self.throws)

    @property
    def n_measured(self):
        return sum(1 for t in self.throws if t.get("landing_xy") is not None)

    @property
    def logged_scales(self):
        return [t["speed_scale"] for t in self.throws
                if t.get("landing_xy") is not None]

    def record_startup(self, go, failures):
        self.stage = Stage.CALIBRATED if go else Stage.BLOCKED
        self.blocked_reason = "" if go else "; ".join(failures)

    def camera_ready(self):
        if self.stage is Stage.CALIBRATED:
            self.stage = Stage.READY

    def can_throw(self):
        return self.stage in (Stage.READY, Stage.MODEL_UPDATED)

    def check_scale(self, requested):
        return scale_allowed(requested, self.logged_scales)

    def record_throw(self, record):
        self.throws.append(record)
        self.confirmed = False        # re-affirm every single time

    def can_update_model(self):
        return self.n_measured >= self.min_throws_for_update

    def record_model_update(self):
        self.model_updated = True
        self.stage = Stage.MODEL_UPDATED

    def can_reoptimize_policy(self):
        return self.model_updated
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd mc-pilot-pybullet && /usr/bin/python3 -m pytest tests/test_hardware_session.py -q`
Expected: 8 passed

- [ ] **Step 5: Add the Tk app and the throw cycle**

Append to `hardware_session.py`. The cycle calls existing code; it does not reimplement planning or execution.

```python
def run_stage_zero(args):
    """start_of_day.py's own stages, reused. Returns (go, failures, report)."""
    import start_of_day as sod
    rep = sod.Report()
    ok = sod.stage_env(rep)
    ok &= sod.stage_arm(rep, args)
    calib_ok, _ = sod.stage_calibrate(rep, args) if ok else (False, None)
    ok &= calib_ok
    ok &= sod.stage_throw(rep, args)
    return ok, [m for _, lvl, m in rep.rows if lvl == sod.FAIL], rep


class ThrowCycle:
    """
    One throw, as the sequence of gates HARDWARE_RUNBOOK.md Sec 2 describes.
    Each step returns (ok, message) and refuses to advance past a failure.
    """

    def __init__(self, state, camera, args):
        self.state, self.camera, self.args = state, camera, args

    def step_pickup(self):
        from pickup_and_lift import pickup_and_lift
        grasped, pct = pickup_and_lift(self.args.ip, self.args.robot,
                                       self.args.pickup_pose, self.args.lift_z)
        return grasped, (f"grasped a ball ({pct:.1f}% closed)" if grasped else
                         f"CLOSED ON NOTHING ({pct:.1f}%) -- place a ball and retry")

    def step_plan(self, target, speed_scale):
        """
        Exactly the sequence run_closed_loop_throws.main() uses -- same calls,
        same order, so there is one planning path and not two.

        Two verdicts, deliberately kept apart: precheck can pass while the
        release position is outside the safe box, and the GUI must show both.
        """
        import numpy as np
        import run_hardware_throw as H
        from robot_arm.kinova_hardware import HardwareThrowExecutor

        a = self.args
        arm, profile, cid = H.build_arm(a.robot)
        pol, cfg = H.load_policy(a.log_path, None)
        coeffs, q_rel, qd_rel, v_ach, speed, v_cmd, rel = H.plan_throw_for_target(
            arm, profile, cfg, pol, target,
            opt_pose=a.opt_pose, u_cap=a.u_cap, tool_offset_z=a.tool_offset_z,
            wrist_roll_offset=np.deg2rad(a.wrist_roll_offset_deg))
        table = H.load_pose_table(cfg, a.opt_pose)
        box = H.release_box_from_table(
            arm, table, tool_offset=[0.0, 0.0, a.tool_offset_z]) if table else None
        limits = H.make_limits(profile, speed_scale, release_box=box, arm=arm,
                               positioning_scale=a.positioning_scale)
        ex = HardwareThrowExecutor(limits, dry_run=not a.arm, ip=a.ip)

        release_box_ok = ex.check_release_pos(rel)
        precheck_ok, report = ex.precheck(coeffs, arm,
                                          release_speed=float(np.linalg.norm(v_ach)))
        return {"arm": arm, "profile": profile, "cid": cid, "ex": ex,
                "coeffs": coeffs, "q_rel": q_rel, "qd_rel": qd_rel,
                "speed": speed, "rel": rel, "precheck_ok": precheck_ok,
                "report": report, "release_box_ok": release_box_ok}

    def step_throw_and_measure(self, plan, target, speed_scale, throw_index):
        import time
        import numpy as np
        from measure_landing import measure_landing
        from perception import base_frame

        ex, arm, profile = plan["ex"], plan["arm"], plan["profile"]

        def _on_release():
            self.camera.mark_release(time.time())

        with ex:
            ex.set_gripper(closed=True)
            ex.home(arm, np.array(profile.q_neutral, float), duration=self.args.duration)
            ex.backend.open_realtime_feedback()
            try:
                ex.rehearse_or_throw(plan["coeffs"], arm, track=None,
                                     on_release=_on_release)
            finally:
                ex.backend.close_realtime_feedback()
        exec_stats = dict(getattr(ex, "last_exec_stats", {}) or {})

        event = self.camera.pop_event(timeout=self.args.measure_timeout)
        if event is None or "error" in event:
            return None, {"refusal_reason": (event or {}).get("error", "no capture window")}, exec_stats

        R, t = base_frame.load_extrinsic()
        try:
            meas = measure_landing(event["rec"], R, t, z_floor=-self.args.base_height,
                                   ball_radius=self.args.ball_radius)
            return [float(meas["x"]), float(meas["y"])], meas, exec_stats
        except RuntimeError as e:
            # A refusal means re-throw. Never loosen a threshold to force a number.
            return None, {"refusal_reason": str(e)}, exec_stats
```

> **Implementer note — the one change needed in existing code.** `H.build_arm`, `H.load_policy`, `H.plan_throw_for_target(arm, profile, cfg, pol, target_xy, ...)`, `H.load_pose_table`, `H.release_box_from_table`, `H.make_limits` and `HardwareThrowExecutor.precheck/check_release_pos/rehearse_or_throw` all exist today and are called above with their real signatures (verified against `run_closed_loop_throws.main()`). The only addition is an `on_release=None` keyword on `HardwareThrowExecutor.rehearse_or_throw` (`robot_arm/kinova_hardware.py:917`), invoked at the instant the release-time gripper OPEN is issued — the same point `GRIPPER_RELEASE_LATENCY_S` is applied. Default `None` keeps every existing caller unchanged. Add a regression test asserting the callback fires exactly once during a dry-run throw, and that a `None` callback is a no-op.
>
> Do **not** call `on_release` from inside the 40 Hz control loop's blocking path — the callback only timestamps and hands off to the camera thread, which is why it is safe here, unlike the gripper-confirm call that stalled the loop for 693 ms.

The Tk dashboard is a thin shell over the above, following `closed_loop_gui.py`'s existing shape: entry fields (ip, checkpoint, ball_id, target, speed_scale), a confirm checkbox bound to `state.confirmed` that clears on every `record_throw`, a scrolling log pane, a table of logged throws, and four buttons — **Run start-of-day**, **Pick up & throw**, **Update model** (disabled until `can_update_model()`), **Re-optimize policy** (disabled until `can_reoptimize_policy()`). Blocking work runs on the worker thread; the GUI is updated through `root.after`.

- [ ] **Step 6: Verify the app opens and stage 0 runs**

Run: `cd mc-pilot-pybullet && /usr/bin/python3 hardware_session.py --dry_run`
Expected: window opens; "Run start-of-day" produces the same GO verdict as `start_of_day.py`; throw buttons stay disabled until it passes

- [ ] **Step 7: Commit**

```bash
git add mc-pilot-pybullet/hardware_session.py mc-pilot-pybullet/tests/test_hardware_session.py mc-pilot-pybullet/run_hardware_throw.py
git commit -m "feat(session): guided throw-cycle app with enforced gates"
```

---

### Task 9: Button 1 — update the model on real data and report

**Files:**
- Modify: `mc-pilot-pybullet/hardware_learning.py`, `mc-pilot-pybullet/hardware_session.py`
- Test: `mc-pilot-pybullet/tests/test_hardware_learning.py` (append)

**Interfaces:**
- Consumes: `track_to_state_samples`, `deviation_verdict`, `fit_release_model`, `MC_PILCO_module.MC_PILOT`
- Produces: `ingest_throws(mc, records, recordings_dir, R_bc, t_bc, na=0) -> dict` with keys `n_ingested`, `n_skipped`, `verdict`, `release_model`, `text`

- [ ] **Step 1: Write the failing test**

```python
def test_ingest_throws_skips_refused_records_and_counts_what_it_used():
    """A fake model records what add_data was called with -- no torch needed."""
    from hardware_learning import ingest_throws

    class FakeModel:
        def __init__(self):
            self.calls = []

        def add_data(self, new_state_samples, new_input_samples):
            self.calls.append((new_state_samples.shape, new_input_samples.shape))

    class FakeMC:
        def __init__(self):
            self.model_learning = FakeModel()

    t = np.arange(0.0, 0.50, 1 / 90.0)
    pts = np.stack([0.3 + 1.39 * t, 0 * t, 0.02 + 0.37 * t - 4.905 * t ** 2], axis=1)
    recs = [
        {"commanded_speed": 1.44, "target": [0.71, 0.0], "landing_xy": [0.71, 0.0],
         "measured_v0": [1.39, 0.0, 0.37], "_track": (pts, t)},
        {"commanded_speed": 1.44, "target": [0.71, 0.0], "landing_xy": None,
         "measured_v0": None, "refusal_reason": "inlier fraction 0.49"},
    ]
    mc = FakeMC()
    out = ingest_throws(mc, recs, track_getter=lambda r: r.get("_track"))
    assert out["n_ingested"] == 1 and out["n_skipped"] == 1
    assert mc.model_learning.calls[0][0][1] == 8      # (n, 8) states
    assert mc.model_learning.calls[0][1][1] == 1      # (n, 1) inputs
    assert "BELOW NOISE" in out["verdict"]["text"]    # a clean parabola, as expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mc-pilot-pybullet && /usr/bin/python3 -m pytest tests/test_hardware_learning.py -q -k ingest`
Expected: FAIL — `ImportError: cannot import name 'ingest_throws'`

- [ ] **Step 3: Write the implementation**

Append to `hardware_learning.py`:

```python
G_BASE_VEC = np.array([0.0, 0.0, -9.81])


def ingest_throws(mc, records, track_getter, na=0, ts=TS_DEFAULT):
    """
    Append real flights to the model exactly as the simulated loop does, and
    report what that changed.

    `track_getter(record)` returns `(points_base (N,3), times (N,))` -- the RAW
    RANSAC-inlier triangulated points for that throw. It is a callback so the
    caller owns file loading and this stays testable.

    Rotation augmentation (`na`) mirrors MC_PILOT.get_data_from_system, which
    applies the paper's Na augmentation to every trial it collects.
    """
    n_in = n_skip = 0
    deviations = []
    for r in records:
        track = None if r.get("landing_xy") is None else track_getter(r)
        if track is None:
            n_skip += 1
            continue
        pts, times = track
        states, inputs = track_to_state_samples(
            pts, times, r["target"], r["commanded_speed"], ts=ts)
        mc.model_learning.add_data(new_state_samples=states, new_input_samples=inputs)
        for _ in range(na):
            ang = np.random.uniform(0.0, 2.0 * np.pi)
            c, s = np.cos(ang), np.sin(ang)
            rot = states.copy()
            for sl in (slice(0, 2), slice(3, 5), slice(6, 8)):
                x, y = states[:, sl].T
                rot[:, sl] = np.stack([c * x - s * y, s * x + c * y], axis=1)
            mc.model_learning.add_data(new_state_samples=rot, new_input_samples=inputs)

        dv = np.diff(states[:, 3:6], axis=0)
        deviations.append(dv - np.tile(G_BASE_VEC * ts, (dv.shape[0], 1)))
        n_in += 1

    verdict = deviation_verdict(np.concatenate(deviations)) if deviations else \
        {"text": "no throws ingested", "above_noise": False, "n_samples": 0,
         "rms_deviation": 0.0, "rms_sigma": 0.0, "ratio": 0.0}
    try:
        release = fit_release_model(records)
    except ValueError as e:
        release = {"text": f"release model not fitted: {e}", "n": 0}

    return {"n_ingested": n_in, "n_skipped": n_skip, "verdict": verdict,
            "release_model": release,
            "text": (f"ingested {n_in} throws, skipped {n_skip}\n"
                     f"flight GP: {verdict['text']}\n"
                     f"release:   {release['text']}")}
```

In `hardware_session.py`, the **Update model** button loads the checkpoint with `mc.load_model_from_log(...)` (the pattern `adapt_policy_height.py` already uses), calls `ingest_throws` with a `track_getter` that re-runs `ransac_track` on each throw's stored recording and returns the inlier points, calls `mc.model_learning.reinforce_model()`, then renders `out["text"]` in the report pane and calls `state.record_model_update()`. **It does not touch the policy.**

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd mc-pilot-pybullet && /usr/bin/python3 -m pytest tests/test_hardware_learning.py -q`
Expected: 21 passed

- [ ] **Step 5: Commit**

```bash
git add mc-pilot-pybullet/hardware_learning.py mc-pilot-pybullet/hardware_session.py mc-pilot-pybullet/tests/test_hardware_learning.py
git commit -m "feat(session): ingest real throws into the GP and report against the noise floor"
```

---

### Task 10: Button 2 — re-optimize the policy into a new checkpoint

**Files:**
- Modify: `mc-pilot-pybullet/hardware_session.py`
- Test: `mc-pilot-pybullet/tests/test_hardware_session.py` (append)

**Interfaces:**
- Consumes: `SessionState.can_reoptimize_policy`, `MC_PILCO_module.MC_PILOT.reinforce_policy`
- Produces: `reoptimize_policy(mc, out_dir, opt_steps) -> str` (the new checkpoint path)

- [ ] **Step 1: Write the failing test**

```python
def test_reoptimize_refuses_to_overwrite_an_existing_checkpoint(tmp_path):
    """Never overwrite a trained checkpoint -- it is the only copy."""
    from hardware_session import reoptimize_policy
    existing = tmp_path / "results_kinetic_chain_gen3_tcp" / "1"
    existing.mkdir(parents=True)

    class FakeMC:
        def reinforce_policy(self, **kw):
            return [0.1], None, None, None

    with pytest.raises(FileExistsError, match="would overwrite"):
        reoptimize_policy(FakeMC(), str(existing), opt_steps=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mc-pilot-pybullet && /usr/bin/python3 -m pytest tests/test_hardware_session.py -q -k reoptimize`
Expected: FAIL — `ImportError: cannot import name 'reoptimize_policy'`

- [ ] **Step 3: Write the implementation**

Append to `hardware_session.py`:

```python
def reoptimize_policy(mc, out_dir, opt_steps=200):
    """
    Re-optimize the policy against the updated model, following the pattern
    adapt_policy_height.py already establishes (reuse the GP, re-optimize the
    policy only).

    Writes a NEW checkpoint directory. The result is NOT thrown automatically:
    its release state may differ from the one whose finger clearance the
    operator visually verified, so it goes back through stage 0 and the
    escalation ladder like any other checkpoint.
    """
    import os
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise FileExistsError(
            f"{out_dir} exists and is not empty -- this would overwrite a "
            f"trained checkpoint; pass a new --out_dir")
    os.makedirs(out_dir, exist_ok=True)
    cost_list, *_ = mc.reinforce_policy(num_optimization_steps=opt_steps)
    return out_dir
```

The button is enabled only when `state.can_reoptimize_policy()` is true, and the report pane must print, verbatim: *"New checkpoint written to `<dir>`. It has NOT been validated on hardware — re-run start-of-day against it and restart the escalation ladder at 0.15."*

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd mc-pilot-pybullet && /usr/bin/python3 -m pytest tests/ -q`
Expected: full suite passes (190 existing + ~40 new)

- [ ] **Step 5: Update the docs**

Add a `HARDWARE_RUNBOOK.md` section §0.3 pointing at `hardware_session.py` as the run-day entry point above `start_of_day.py`, and a `CLAUDE.md` bullet under the active-track section covering the two new modules and the "flight is below the noise floor, release is not" finding.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(session): policy re-optimization into a new checkpoint + docs"
```

---

## Self-review

**Spec coverage.** §1 module boundaries → Tasks 1/3/6/7/8. §2 state machine → Task 8. §3 live view → Tasks 6/7. §4 throw cycle → Task 8 (all six gates; escalation in Task 1). §5 data model → Task 2. §6 model update → Tasks 3/4/5/9, including the tautology guard as a test and the formula-defined verdict. §7 policy → Task 10. §8 testing → every listed test appears in a task. §9 out of scope → nothing implements the whiffle ball or auto-reload. §10 risks → camera fault (Task 6 sets `.error`), noise-sized learning (Task 4), drift (stage 0), thread contention (Task 6 single-slot), operator fatigue (Task 8 confirm reset).

**Type consistency.** `state_samples` is `(n, 8)` and `input_samples` `(n, 1)` in Tasks 3, 9 and the Global Constraints. `deviation_verdict` returns the same key set in Tasks 4, 5 and 9. `measurement` dicts flow from `measure_landing` through Task 2's `build_throw_record` into Task 5's `fit_release_model` under the same key names (`p0`, `v0` → `measured_p0`, `measured_v0`).

**Signatures checked against the real code, not assumed.** Task 8's planning sequence was corrected after reading `run_closed_loop_throws.main()`: `plan_throw_for_target` takes `(arm, profile, cfg, pol, target_xy, ...)` — constructed objects, not paths — and there is no `execute_throw`; execution is `HardwareThrowExecutor.rehearse_or_throw(coeffs, arm, ...)` inside a `with ex:` block, wrapped by `open_realtime_feedback()`/`close_realtime_feedback()`.

**Exactly two additive changes to existing code, both flagged inline:** `IRRecorder.stream()` (Task 6) and `rehearse_or_throw(..., on_release=None)` (Task 8). Both default to current behaviour, so every existing caller is unchanged.

**Phasing.** Tasks 1–8 are Phase A and produce a usable session: the operator can calibrate, throw, and collect the dataset with the MC buttons disabled. Tasks 9–10 are Phase B. Stopping after Task 8 leaves working software, which is the point of the split.
