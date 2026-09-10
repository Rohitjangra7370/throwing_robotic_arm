# Ball Tracking and Landing-Point Measurement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure where a thrown tennis ball *first contacts the floor*, offline, to a stated uncertainty, in the arm's base frame.

**Architecture:** Record both D435i infrared imagers at 848x480@90 fps through the throw. Detect the ball independently in each image by differencing against a per-pixel median background. Pair left/right detections by image row (the pair is factory-rectified), triangulate off the 49.9448 mm baseline, fit one drag-free parabola by Gauss-Newton on *pixel* reprojection error with RANSAC association, then solve for the descending floor crossing.

**Tech Stack:** Python 3.10.12 (system interpreter), numpy, OpenCV (`cv2`), `pyrealsense2` 2.58.2, pytest. All already installed. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-25-ball-tracking-design.md` — read it first; this plan argues from it.

## Global Constraints

- **Working directory is `mc-pilot-pybullet/` for every command in this plan.** Imports resolve via that directory being on `sys.path` (`tests/conftest.py` inserts it).
- **Never use bare `pip`/`pip3`** — they resolve to the Blender snap's Python 3.13. Use `python3 -m pip`.
- **Never use bare `pytest`** — ROS system plugins on PATH break it. Use `python3 -m pytest`.
- **Bare `python3` may resolve to a Conda base env with no torch/pybullet.** Verify with `python3 -c "import cv2, numpy, pyrealsense2"`; fall back to `/usr/bin/python3` explicitly if it fails.
- **No new dependencies.** `cv2`, `numpy`, `pyrealsense2` are present; nothing else is needed.
- **Fail loudly, never fabricate.** Every estimator raises `RuntimeError` with a diagnostic rather than returning a plausible-looking wrong number. This is the repo's standing rule (`CLAUDE.md`, the model-belief trap) and `detect_ball_bgsub` already follows it.
- **Measured device constants, exact values — do not round, do not re-derive:**
  - `fx = fy = 426.167`, `ppx = 420.286`, `ppy = 238.505` (IR, 848x480)
  - all five Brown-Conrady coefficients `0.0`
  - baseline `0.0499448 m`, IR1 -> IR2 rotation exactly identity
  - IR2 is 49.9448 mm along IR1's **+X** axis, i.e. IR1 is the **left** camera
- **Geometry constants:** `z_floor = -0.433` in base frame (base at 0, floor at `-base_height`); `ball_radius = 0.0327`; `g = (0, 0, -9.81)` in base frame.
- **Rejection thresholds (spec §6):** minimum 12 inlier frames; RANSAC inlier fraction >= 0.6; RMS reprojection residual <= 1.0 px.
- Tasks 1-8 require **no camera and no arm**. Only Tasks 9-11 touch hardware.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `perception/ray_plane.py` | modify | Add the IR intrinsics constant next to the existing colour ones. Everything else untouched. |
| `perception/stereo.py` | create | `StereoRig`: project a camera-frame point to both images, triangulate back. Left/right candidate pairing. |
| `perception/ball_track.py` | create | Median background; one image -> ball candidates with subpixel centroid. |
| `perception/trajectory.py` | create | Ballistic kinematics, impact solve, two-point exact fit, Gauss-Newton fit, RANSAC association. |
| `perception/ir_capture.py` | create | Dual-IR recorder. Hardware only. |
| `record_throw_ir.py` | create | CLI: capture a throw window to disk. |
| `measure_landing.py` | create | CLI + `measure_landing()`: recording -> first-contact (x, y) in base frame. |
| `tune_ir_exposure.py` | create | CLI: emitter/exposure A/B (spec §9). |
| `tests/test_stereo.py` | create | Round-trip and hand-checked triangulation. |
| `tests/test_ball_track.py` | create | Detection on synthetic images. |
| `tests/test_trajectory.py` | create | Kinematics, impact solve, fitting, RANSAC. |
| `tests/test_landing_pipeline.py` | create | Synthetic end-to-end acceptance test (spec §7.1). |

---

### Task 1: IR stereo rig — intrinsics, projection, triangulation

**Files:**
- Modify: `perception/ray_plane.py` (append a constant after `D435I_COLOR_1920x1080`)
- Create: `perception/stereo.py`
- Test: `tests/test_stereo.py`

**Interfaces:**
- Consumes: `perception.ray_plane.Intrinsics` (existing).
- Produces: `perception.ray_plane.D435I_IR_848x480`; `perception.stereo.D435I_IR_BASELINE_M`, `perception.stereo.StereoRig` with `.intr`, `.baseline_m`, `.project(p_cam) -> (u1, v1, u2, v2)`, `.triangulate(u1, v1, u2, v2) -> ndarray (..., 3)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_stereo.py`:

```python
"""
Stereo IR triangulation. Pure geometry -- no camera.

Every landing measurement flows through this, and a baseline or a disparity
sign error here produces plausible-looking positions rather than an exception,
so it is pinned down before anything else is built on it.
"""
import numpy as np
import pytest

from perception.ray_plane import D435I_IR_848x480
from perception.stereo import D435I_IR_BASELINE_M, StereoRig

RIG = StereoRig(D435I_IR_848x480, D435I_IR_BASELINE_M)


def test_ir_constants_match_the_device():
    """Read off the lab D435i on 2026-08-25; guards a silent edit."""
    intr = D435I_IR_848x480
    assert (intr.fx, intr.fy) == (426.167, 426.167)
    assert (intr.ppx, intr.ppy) == (420.286, 238.505)
    assert (intr.width, intr.height) == (848, 480)
    assert not any(intr.coeffs), "IR stream is factory-rectified"
    assert D435I_IR_BASELINE_M == 0.0499448
    assert intr.hfov_deg() == pytest.approx(89.7, abs=0.1)
    assert intr.vfov_deg() == pytest.approx(58.8, abs=0.1)


def test_disparity_at_two_metres_matches_hand_calculation():
    """d = fx*B/Z = 426.167*0.0499448/2.0 = 10.642 px. A baseline typo shows here."""
    u1, v1, u2, v2 = RIG.project(np.array([0.0, 0.0, 2.0]))
    assert (u1 - u2) == pytest.approx(10.642, abs=0.01)


def test_triangulate_inverts_project_exactly():
    p = np.array([0.30, -0.20, 2.00])
    got = RIG.triangulate(*RIG.project(p))
    assert np.allclose(got, p, atol=1e-9)


def test_rectified_pair_has_identical_rows():
    """IR1->IR2 rotation is exactly identity, so v2 must equal v1."""
    u1, v1, u2, v2 = RIG.project(np.array([0.4, 0.25, 1.7]))
    assert v2 == pytest.approx(v1, abs=1e-12)


def test_ir2_is_to_the_right_so_disparity_is_positive():
    """IR2 sits at +49.9448 mm along IR1's +X. Getting this backwards flips
    the sign of every depth. A negative disparity must not yield a position."""
    u1, v1, u2, v2 = RIG.project(np.array([0.0, 0.0, 2.0]))
    assert u1 > u2
    with pytest.raises(RuntimeError, match="disparity"):
        RIG.triangulate(u1, v1, u1 + 1.0, v2)


def test_batch_shape_contract():
    pts = np.array([[0.0, 0.0, 2.0], [0.3, -0.2, 1.5], [-0.4, 0.1, 2.4]])
    u1, v1, u2, v2 = RIG.project(pts)
    assert u1.shape == (3,)
    got = RIG.triangulate(u1, v1, u2, v2)
    assert got.shape == (3, 3)
    assert np.allclose(got, pts, atol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_stereo.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'perception.stereo'`

- [ ] **Step 3: Add the IR intrinsics constant**

Append to `perception/ray_plane.py`, immediately after `D435I_COLOR_1920x1080`, and add `"D435I_IR_848x480"` to `__all__`:

```python
# Read off the lab D435i (SN 349522070924, fw 5.17.3.10) on 2026-08-25 via
# pyrealsense2 stream profiles. IR, not colour: both IR imagers are global
# shutter (OV9282) while the colour imager (OV2740) is rolling shutter, and the
# IR field of view is much wider (89.7 x 58.8 deg vs 70.2 x 43.2), which is what
# puts the ball's flight in frame from an overhead mount at all.
# fx == fy exactly, and all distortion coefficients are zero -- factory
# rectified, so undistortion is a no-op HERE and must be re-checked if the
# stream resolution ever changes.
D435I_IR_848x480 = Intrinsics(fx=426.167, fy=426.167, ppx=420.286, ppy=238.505,
                              width=848, height=480, coeffs=(0., 0., 0., 0., 0.))
```

- [ ] **Step 4: Write `perception/stereo.py`**

```python
"""
Rectified stereo triangulation for the D435i's two IR imagers.

WHY THIS IS NOT THE DEPTH THE REPO REJECTS
------------------------------------------
`ray_plane.py`, `HARDWARE_SETUP.md` and `HANDOFF.md` item 5 all reject
RealSense's block-matching DEPTH MAP as a position source: ~2% of range = 2-4 cm
at 1-2 m, the same size as the landing error being measured. That objection
stands. It is an objection to matching textureless surfaces -- and it is even
worse for a small round airborne ball, where the depth map returns 0.0 as often
as it returns a noisy value.

This module never touches the depth map. It triangulates two INDEPENDENTLY
DETECTED, high-contrast blob centroids off the factory baseline. Different
operation, different error model: ~0.7 mm laterally per frame at Z = 2 m.

WHY THERE IS NO RECTIFICATION STEP
----------------------------------
Measured off the unit: the IR1 -> IR2 extrinsic is a pure translation of
(-49.9448, 0, 0) mm with EXACTLY identity rotation, and every distortion
coefficient is zero. The pair is already rectified in hardware, so disparity is
purely horizontal and depth is one division. Do not add a rectification step;
adding one would only introduce interpolation error.

FRAME. All 3D points here are in the IR1 (LEFT) optical frame: +X right,
+Y down, +Z forward. Converting to the base frame is the caller's job and needs
T_B_C, which this module deliberately does not know about.
"""

from __future__ import annotations

import numpy as np

from perception.ray_plane import Intrinsics

__all__ = ["D435I_IR_BASELINE_M", "StereoRig", "pair_candidates"]

# Measured 2026-08-25: get_extrinsics_to() from infrared,1 to infrared,2 returns
# translation (-49.9448, 0, 0) mm. p_2 = p_1 + t, so the IR2 ORIGIN sits at
# -t = +49.9448 mm along IR1's +X axis: IR1 is the LEFT camera and disparity
# u1 - u2 is positive for any point in front of the pair.
D435I_IR_BASELINE_M = 0.0499448


class StereoRig:
    """A rectified pair: shared intrinsics, one horizontal baseline."""

    def __init__(self, intr: Intrinsics, baseline_m: float):
        if any(intr.coeffs):
            raise ValueError("StereoRig assumes a rectified, undistorted pair; "
                             "these intrinsics carry non-zero distortion")
        self.intr = intr
        self.baseline_m = float(baseline_m)

    def project(self, p_cam):
        """
        IR1-frame point(s) -> (u1, v1, u2, v2).

        Shape contract: (3,) in -> four scalars out; (N, 3) in -> four (N,)
        arrays out.
        """
        p = np.asarray(p_cam, float)
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        if np.any(z <= 0):
            raise RuntimeError("point at or behind the image plane (z <= 0); "
                               "it has no projection")
        i = self.intr
        u1 = i.fx * x / z + i.ppx
        v1 = i.fy * y / z + i.ppy
        u2 = i.fx * (x - self.baseline_m) / z + i.ppx
        v2 = v1 if np.ndim(v1) == 0 else v1.copy()
        return u1, v1, u2, v2

    def triangulate(self, u1, v1, u2, v2):
        """
        Matched pixel pair(s) -> IR1-frame point(s).

        Uses u1 - u2 for depth and (u1, v1) for the bearing. `v2` is accepted so
        callers pass a full match, and is used only to check the pair really does
        lie on a common row -- a mismatched pair that slipped through pairing
        must raise here rather than produce a confident wrong depth.
        """
        u1 = np.asarray(u1, float); v1 = np.asarray(v1, float)
        u2 = np.asarray(u2, float); v2 = np.asarray(v2, float)
        d = u1 - u2
        if np.any(d <= 0):
            raise RuntimeError(
                f"non-positive disparity (min {np.min(d):.3f} px) -- either the "
                f"left/right images are swapped or the pairing is wrong; there "
                f"is no depth for this pair")
        if np.any(np.abs(v1 - v2) > 3.0):
            raise RuntimeError(
                f"pair is not on a common row (max |v1-v2| = "
                f"{np.max(np.abs(v1 - v2)):.2f} px > 3.0) -- the pair is "
                f"rectified, so this is a mis-match, not noise")
        i = self.intr
        z = i.fx * self.baseline_m / d
        x = (u1 - i.ppx) * z / i.fx
        y = (v1 - i.ppy) * z / i.fy
        return np.stack([x, y, z], axis=-1)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_stereo.py -q`
Expected: PASS, 6 tests.

- [ ] **Step 6: Confirm nothing else regressed**

Run: `python3 -m pytest tests/ -q`
Expected: all pre-existing tests still pass (`ray_plane.py` gained a constant only).

- [ ] **Step 7: Commit**

```bash
git add perception/ray_plane.py perception/stereo.py tests/test_stereo.py
git commit -m "feat(vision): rectified IR stereo rig for the D435i pair"
```

---

### Task 2: Left/right candidate pairing

**Files:**
- Modify: `perception/stereo.py`
- Test: `tests/test_stereo.py`

**Interfaces:**
- Consumes: `StereoRig` (Task 1); `perception.ball_track.Candidate` is *not* required — pairing takes plain `(u, v, area_px)` tuples so it can be tested without the detector.
- Produces: `perception.stereo.pair_candidates(left, right, row_tol_px=3.0, area_ratio_tol=2.5) -> list[tuple[int, int]]` returning `(left_index, right_index)` pairs.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_stereo.py`:

```python
from perception.stereo import pair_candidates


def test_pairing_matches_on_row_and_area():
    left = [(500.0, 200.0, 150.0), (300.0, 400.0, 140.0)]
    right = [(292.0, 400.5, 145.0), (489.0, 199.6, 152.0)]
    assert sorted(pair_candidates(left, right)) == [(0, 1), (1, 0)]


def test_pairing_rejects_a_row_mismatch():
    left = [(500.0, 200.0, 150.0)]
    right = [(489.0, 260.0, 152.0)]
    assert pair_candidates(left, right) == []


def test_pairing_rejects_an_area_mismatch():
    """Same row, but one blob is 10x the other -- not the same object."""
    left = [(500.0, 200.0, 150.0)]
    right = [(489.0, 200.0, 1500.0)]
    assert pair_candidates(left, right) == []


def test_pairing_rejects_negative_disparity():
    """A right-image detection to the RIGHT of its left partner is impossible."""
    left = [(400.0, 200.0, 150.0)]
    right = [(430.0, 200.0, 150.0)]
    assert pair_candidates(left, right) == []


def test_pairing_is_one_to_one_and_prefers_the_closer_row():
    left = [(500.0, 200.0, 150.0)]
    right = [(489.0, 202.9, 150.0), (487.0, 200.1, 150.0)]
    assert pair_candidates(left, right) == [(0, 1)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_stereo.py -q -k pairing`
Expected: FAIL — `ImportError: cannot import name 'pair_candidates'`

- [ ] **Step 3: Implement `pair_candidates`**

Append to `perception/stereo.py`:

```python
def pair_candidates(left, right, row_tol_px=3.0, area_ratio_tol=2.5):
    """
    Match left-image detections to right-image detections.

    `left` and `right` are sequences of (u, v, area_px). Returns a list of
    (left_index, right_index), one-to-one, best-first.

    The pair is rectified, so a true match lies on the same row -- `row_tol_px`
    covers motion blur and centroid noise, not epipolar geometry. Disparity must
    be positive (IR2 is the right camera). Area must agree within
    `area_ratio_tol`, which rejects pairing the ball against a background object
    that happens to share its row.

    Greedy nearest-row assignment is sufficient because there is one ball in
    flight; if the scene ever has two, the RANSAC ballistic association in
    `trajectory.py` is the backstop, not this function.
    """
    scored = []
    for li, (ul, vl, al) in enumerate(left):
        for ri, (ur, vr, ar) in enumerate(right):
            if ul - ur <= 0:
                continue
            drow = abs(vl - vr)
            if drow > row_tol_px:
                continue
            ratio = max(al, ar) / max(min(al, ar), 1e-9)
            if ratio > area_ratio_tol:
                continue
            scored.append((drow, li, ri))

    scored.sort()
    used_l, used_r, out = set(), set(), []
    for _, li, ri in scored:
        if li in used_l or ri in used_r:
            continue
        used_l.add(li); used_r.add(ri)
        out.append((li, ri))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_stereo.py -q`
Expected: PASS, 11 tests.

- [ ] **Step 5: Commit**

```bash
git add perception/stereo.py tests/test_stereo.py
git commit -m "feat(vision): left/right candidate pairing on the rectified IR pair"
```

---

### Task 3: Ball detection — median background and subpixel candidates

**Files:**
- Create: `perception/ball_track.py`
- Test: `tests/test_ball_track.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `perception.ball_track.Candidate` (dataclass with fields `u, v, area_px, radius_px, circularity`), `median_background(frames) -> ndarray`, `detect_candidates(frame, background, diff_thresh=18, min_area_px=20, max_area_px=2000, min_circularity=0.30, morph_kernel=3) -> list[Candidate]`, `frame_diagnostics(frame, background, diff_thresh=18) -> dict` with key `mask_nonzero_frac`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ball_track.py`:

```python
"""
Ball detection on synthetic images. No camera.

Synthetic rather than recorded because the quantity being validated is centroid
BIAS -- a systematic that a real recording cannot bound, since it has no ground
truth. A drawn disc has one exactly.
"""
import cv2
import numpy as np
import pytest

from perception.ball_track import (Candidate, detect_candidates,
                                   frame_diagnostics, median_background)

H, W = 480, 848


def _bg(level=40):
    """Textured background -- a flat one would make detection unrealistically easy."""
    rng = np.random.default_rng(0)
    return (level + rng.integers(0, 12, size=(H, W))).astype(np.uint8)


def _with_ball(bg, u, v, r=7.0, level=220):
    img = bg.copy()
    cv2.circle(img, (int(round(u)), int(round(v))), int(round(r)), int(level), -1)
    return img


def test_median_background_ignores_a_ball_that_moves():
    bg = _bg()
    frames = [_with_ball(bg, 100 + 40 * i, 200) for i in range(9)]
    got = median_background(frames)
    assert np.abs(got.astype(int) - bg.astype(int)).max() <= 1


def test_detects_a_single_ball_with_subpixel_accuracy():
    bg = _bg()
    frame = _with_ball(bg, 423.0, 217.0, r=7.0)
    cands = detect_candidates(frame, bg)
    assert len(cands) == 1
    c = cands[0]
    assert c.u == pytest.approx(423.0, abs=0.3)
    assert c.v == pytest.approx(217.0, abs=0.3)
    assert c.radius_px == pytest.approx(7.0, rel=0.15)
    assert c.circularity > 0.7


def test_returns_empty_when_nothing_changed():
    """No ball must mean no candidates -- never a fabricated one."""
    bg = _bg()
    assert detect_candidates(bg.copy(), bg) == []


def test_rejects_blobs_outside_the_size_gate():
    bg = _bg()
    tiny = _with_ball(bg, 400.0, 200.0, r=1.0)
    huge = _with_ball(bg, 400.0, 200.0, r=60.0)
    assert detect_candidates(tiny, bg) == []
    assert detect_candidates(huge, bg) == []


def test_finds_both_balls_when_two_are_present():
    """Detection does not decide which blob is the ball; that is RANSAC's job."""
    bg = _bg()
    frame = _with_ball(_with_ball(bg, 200.0, 150.0), 600.0, 300.0)
    assert len(detect_candidates(frame, bg)) == 2


def test_frame_diagnostics_flags_lighting_drift():
    """
    Spec section 6 wants the one number that catches a flooded frame -- the same
    role mask_nonzero_frac plays in detect_ball_bgsub. A ball is a few tenths of
    a percent of the frame; a lighting shift or a bumped camera is tens of
    percent, and would otherwise surface as a silent detection failure.
    """
    bg = _bg()
    quiet = frame_diagnostics(_with_ball(bg, 423.0, 217.0), bg)
    assert quiet["mask_nonzero_frac"] < 0.01
    flooded = frame_diagnostics(np.full_like(bg, 200), bg)
    assert flooded["mask_nonzero_frac"] > 0.9


def test_candidate_is_a_plain_tuple_of_floats_for_pairing():
    """pair_candidates() consumes (u, v, area_px); keep them interoperable."""
    bg = _bg()
    c = detect_candidates(_with_ball(bg, 423.0, 217.0), bg)[0]
    assert isinstance(c.as_uv_area(), tuple)
    assert len(c.as_uv_area()) == 3
    assert all(isinstance(x, float) for x in c.as_uv_area())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_ball_track.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'perception.ball_track'`

- [ ] **Step 3: Write `perception/ball_track.py`**

```python
"""
Find the ball in one IR image by differencing against a static background.

WHY NOT YOLO
------------
The ball spans ~12-20 px in the 848x480 IR stream, which sits at the floor of
YOLOv8's stride-8 P3 head. The camera is static and the ball is high contrast
against a differenced background, so a threshold plus a contour finds it in
~1 ms of CPU with no model, no training set and no GPU -- which also matters
because `torch.cuda` is currently unusable on this machine. A YOLO backend
behind this same interface is a reasonable fallback; it is not the primary.

WHY THIS RETURNS EVERY CANDIDATE, NOT "THE BALL"
------------------------------------------------
The arm moves fast in exactly the region the ball starts in, and it is the main
false-positive source. This module deliberately does NOT try to pick the ball
out -- no hand-drawn ROI mask, no "largest blob" heuristic. It reports every
plausible blob and lets the RANSAC ballistic association in `trajectory.py`
reject the arm for free: arm pixels do not fit a parabola with g = 9.81. That
also handles reflections and the second bounce, which no mask would.

Contrast this with `ball_detector.detect_ball_bgsub`, which DOES pick a single
largest blob -- correct there, because it looks at a settled scene with one
changed object, and there is no trajectory available to arbitrate.

SUBPIXEL CENTROID
-----------------
Intensity-weighted over the difference image, not the binary mask. For a
symmetric motion blur the intensity centroid is unbiased at MID-EXPOSURE, which
is why `ir_capture.py` timestamps frames at mid-exposure rather than at arrival.
Getting that pairing wrong is a systematic: at 5.8 m/s an 8.5 ms exposure is
5 cm of travel.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["Candidate", "median_background", "detect_candidates",
           "frame_diagnostics"]


@dataclass(frozen=True)
class Candidate:
    """One plausible ball blob in one image."""
    u: float
    v: float
    area_px: float
    radius_px: float
    circularity: float

    def as_uv_area(self):
        """The triple `stereo.pair_candidates` consumes."""
        return (float(self.u), float(self.v), float(self.area_px))


def median_background(frames):
    """
    Per-pixel median over a stack of frames.

    A ball occupies any given pixel for at most a couple of frames out of the
    ~44 in a throw window, so the temporal median IS the empty scene -- no
    separate background capture, and no drift between a "before" shot and the
    throw. This is strictly more robust than the two-shot approach in
    `ball_detector.py`, which is vulnerable to auto-exposure resettling between
    the two captures.
    """
    stack = np.asarray(frames)
    if stack.ndim != 3:
        raise ValueError(f"expected a stack of 2-D frames, got shape {stack.shape}")
    if stack.shape[0] < 3:
        raise ValueError(f"need >= 3 frames for a median background, got {stack.shape[0]}")
    return np.median(stack, axis=0).astype(np.uint8)


def frame_diagnostics(frame, background, diff_thresh=18):
    """
    The one number that says whether this frame is interpretable at all.

    `mask_nonzero_frac` is the fraction of pixels that changed. A ball is a few
    tenths of a percent. Tens of percent means the lighting shifted, the camera
    was bumped, or auto-exposure resettled -- in which case the detector will
    either find nothing or find garbage, and either way the answer is to fix the
    capture, not to loosen a threshold. Same role, same name, as
    `ball_detector.detect_ball_bgsub`'s diagnostic.
    """
    diff = cv2.absdiff(frame, background)
    mask = diff >= diff_thresh
    return {"mask_nonzero_frac": float(np.count_nonzero(mask)) / mask.size,
            "diff_max": int(diff.max()), "diff_mean": float(diff.mean())}


def detect_candidates(frame, background, diff_thresh=18, min_area_px=20,
                      max_area_px=2000, min_circularity=0.30, morph_kernel=3):
    """
    One image + its background -> every plausible ball blob.

    Parameters
    ----------
    diff_thresh : int
        Absolute per-pixel difference counting as "changed".
    min_area_px, max_area_px : int
        The ball spans ~12-20 px across at 1.5-2.4 m, i.e. ~113-314 px of area.
        The window is deliberately wider than that: motion blur elongates the
        blob, so the gate rejects sensor speckle and whole-frame lighting drift,
        not marginal balls.
    min_circularity : float
        4*pi*area/perimeter^2. LOOSE on purpose -- a blurred ball is an ellipse,
        not a disc, and tightening this is how you silently drop the fastest
        (most informative) frames. It only exists to reject long thin edges.

    Returns a list of Candidate, possibly empty. Returning [] is a valid,
    meaningful answer -- it does NOT raise, because "no ball in this frame" is
    expected for every frame before the ball enters view.
    """
    if frame.shape != background.shape:
        raise ValueError(f"frame shape {frame.shape} != background shape "
                         f"{background.shape} -- the camera must not have moved")

    diff = cv2.absdiff(frame, background)
    mask = (diff >= diff_thresh).astype(np.uint8) * 255
    kernel = np.ones((morph_kernel, morph_kernel), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if not (min_area_px <= area <= max_area_px):
            continue
        perim = float(cv2.arcLength(c, True))
        if perim <= 0:
            continue
        circ = 4.0 * np.pi * area / (perim * perim)
        if circ < min_circularity:
            continue

        blob = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(blob, [c], -1, 255, -1)
        w = diff.astype(np.float64) * (blob > 0)
        total = w.sum()
        if total <= 0:
            continue
        ys, xs = np.nonzero(blob)
        u = float((w[ys, xs] * xs).sum() / total)
        v = float((w[ys, xs] * ys).sum() / total)
        out.append(Candidate(u=u, v=v, area_px=area,
                             radius_px=float(np.sqrt(area / np.pi)),
                             circularity=float(circ)))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_ball_track.py -q`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add perception/ball_track.py tests/test_ball_track.py
git commit -m "feat(vision): median-background ball candidate detection"
```

---

### Task 4: Ballistic kinematics and the impact solve

**Files:**
- Create: `perception/trajectory.py`
- Test: `tests/test_trajectory.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `perception.trajectory.G_BASE` (ndarray `[0, 0, -9.81]`), `Z_FLOOR_BASE = -0.433`, `BALL_RADIUS = 0.0327`, `ballistic_position(p0, v0, t, g=G_BASE) -> ndarray`, `solve_impact_time(p0, v0, z_target, g=G_BASE) -> float`, `solve_impact(p0, v0, z_floor=Z_FLOOR_BASE, ball_radius=BALL_RADIUS, g=G_BASE) -> (x, y, t)`, `fit_two_points(t_a, p_a, t_b, p_b, g=G_BASE) -> (p0, v0)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_trajectory.py`:

```python
"""
Ballistic fitting and the impact solve. Pure math -- no camera, no arm.
"""
import numpy as np
import pytest

from perception.trajectory import (BALL_RADIUS, G_BASE, Z_FLOOR_BASE,
                                   ballistic_position, fit_two_points,
                                   solve_impact, solve_impact_time)


def test_gravity_and_geometry_constants():
    assert np.allclose(G_BASE, [0.0, 0.0, -9.81])
    assert Z_FLOOR_BASE == -0.433, "base frame: base at 0, floor at -base_height"
    assert BALL_RADIUS == 0.0327


def test_position_at_a_hand_computed_time():
    p0 = np.array([0.0, 0.0, 1.0])
    v0 = np.array([2.0, 0.0, 0.0])
    p = ballistic_position(p0, v0, 0.45152)
    assert p[0] == pytest.approx(0.90304, abs=1e-5)
    assert p[2] == pytest.approx(0.0, abs=1e-4)


def test_impact_time_matches_the_closed_form():
    """Drop from z=1.0 with no vertical velocity: t = sqrt(2*1.0/9.81)."""
    t = solve_impact_time(np.array([0.0, 0.0, 1.0]), np.zeros(3), 0.0)
    assert t == pytest.approx(np.sqrt(2.0 / 9.81), rel=1e-12)


def test_impact_takes_the_DESCENDING_root_not_the_first_crossing():
    """
    Rising through the target plane then falling back gives two positive roots.
    Taking the smaller one reports a 'landing' while the ball is still going UP.
    """
    p0 = np.array([0.0, 0.0, 1.0])
    v0 = np.array([1.0, 0.0, 3.0])
    t = solve_impact_time(p0, v0, 1.3)
    assert t == pytest.approx(0.48568, abs=1e-4)
    assert ballistic_position(p0, v0, t)[2] == pytest.approx(1.3, abs=1e-9)
    assert (v0[2] + G_BASE[2] * t) < 0, "must be descending at impact"


def test_no_real_crossing_raises_rather_than_returning_nonsense():
    with pytest.raises(RuntimeError, match="never reaches"):
        solve_impact_time(np.array([0.0, 0.0, 1.0]), np.zeros(3), 1.5)


def test_impact_offsets_by_one_ball_radius():
    """
    A resting sphere's CENTRE is one radius above the floor, and for a sphere
    the centre's (x, y) IS the contact (x, y). This is the same convention as
    ray_plane.ball_center_on_plane -- if the two disagree, the cross-check in
    tests/test_landing_pipeline.py measures nothing.
    """
    p0 = np.array([0.0, 0.0, 1.0])
    v0 = np.array([2.0, 0.0, 0.0])
    x, y, t = solve_impact(p0, v0, z_floor=0.0, ball_radius=BALL_RADIUS)
    assert ballistic_position(p0, v0, t)[2] == pytest.approx(BALL_RADIUS, abs=1e-9)
    x_naive, _, _ = solve_impact(p0, v0, z_floor=0.0, ball_radius=0.0)
    assert x < x_naive, "the radius offset must shorten the flight, not lengthen it"


def test_two_point_fit_recovers_the_generating_trajectory():
    p0 = np.array([0.035, 0.0, 1.137])
    v0 = np.array([1.6218, 0.0, 0.1419])
    ta, tb = 0.12, 0.47
    got_p0, got_v0 = fit_two_points(ta, ballistic_position(p0, v0, ta),
                                    tb, ballistic_position(p0, v0, tb))
    assert np.allclose(got_p0, p0, atol=1e-12)
    assert np.allclose(got_v0, v0, atol=1e-12)


def test_two_point_fit_refuses_a_degenerate_sample():
    p = np.array([0.0, 0.0, 1.0])
    with pytest.raises(RuntimeError, match="separated"):
        fit_two_points(0.20, p, 0.2001, p)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_trajectory.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'perception.trajectory'`

- [ ] **Step 3: Write `perception/trajectory.py`**

```python
"""
Ballistic trajectory fitting and the floor-impact solve.

WHAT THIS MEASURES, AND WHY IT IS NOT THE RESTING POSITION
-----------------------------------------------------------
`ball_detector.py` measures where the ball SETTLES. On a tile floor a ball
thrown at ~1.6 m/s bounces and rolls, so the settled position is not the
first-contact point -- and it is first contact that the policy's landing error
is defined against. The gap is a systematic, and nothing in the static pipeline
bounds it. Fitting the flight and solving for the floor crossing measures first
contact directly.

WHY NO DRAG TERM
----------------
Tennis ball, 57.7 g, 65.4 mm diameter (fixed by the trained checkpoints and
HARDWARE_RUNBOOK.md). Drag at 2 m/s is 0.5*1.2*0.5*3.36e-3*v^2 = 4.0e-3 N
against 0.566 N of weight -- 0.7%. A drag-free parabola is valid HERE.

It is NOT valid for the whiffle ball (0.004 kg / 0.06 m) of the drag-crossover
study, where drag is the entire point. If this is ever pointed at that ball,
`g` alone will not describe the motion and a drag term has to be added to
`ballistic_position` and to the fit residual together.

FRAME. Base frame throughout: base at z = 0, floor at z = -base_height =
-0.433, g = (0, 0, -9.81).
"""

from __future__ import annotations

import numpy as np

__all__ = ["G_BASE", "Z_FLOOR_BASE", "BALL_RADIUS", "ballistic_position",
           "solve_impact_time", "solve_impact", "fit_two_points"]

G_BASE = np.array([0.0, 0.0, -9.81])
Z_FLOOR_BASE = -0.433      # measured base plate; see CLAUDE.md's frame note
BALL_RADIUS = 0.0327       # tennis ball, matches the trained checkpoints


def ballistic_position(p0, v0, t, g=G_BASE):
    """p(t) = p0 + v0*t + 0.5*g*t^2. Scalar t -> (3,); array t -> (N, 3)."""
    p0 = np.asarray(p0, float); v0 = np.asarray(v0, float); g = np.asarray(g, float)
    t = np.asarray(t, float)
    return p0 + v0 * t[..., None] + 0.5 * g * (t[..., None] ** 2)


def solve_impact_time(p0, v0, z_target, g=G_BASE):
    """
    Time of the DESCENDING crossing of the horizontal plane z = z_target.

    A trajectory that rises through the plane and falls back has TWO positive
    roots. The smaller one is the ball on its way up; reporting it would give a
    confident landing point for a ball that has not landed. Always the larger.

    Raises RuntimeError if the plane is never reached.
    """
    p0 = np.asarray(p0, float); v0 = np.asarray(v0, float); g = np.asarray(g, float)
    a = 0.5 * float(g[2])
    b = float(v0[2])
    c = float(p0[2]) - float(z_target)
    if a == 0.0:
        raise RuntimeError("zero vertical acceleration -- not a ballistic arc")
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        raise RuntimeError(
            f"trajectory never reaches z = {z_target:.4f} (discriminant "
            f"{disc:.4e}); apex is z = {float(p0[2]) - b * b / (4.0 * a):.4f}")
    root = np.sqrt(disc)
    # a < 0, so the LARGER time is (-b - root)/(2a). Both roots computed so the
    # failure message can say which ones existed.
    t_desc = (-b - root) / (2.0 * a)
    t_asc = (-b + root) / (2.0 * a)
    if t_desc <= 0.0:
        raise RuntimeError(
            f"no descending crossing at positive time (roots {t_asc:.4f}, "
            f"{t_desc:.4f} s) -- the ball is already past this plane")
    return float(t_desc)


def solve_impact(p0, v0, z_floor=Z_FLOOR_BASE, ball_radius=BALL_RADIUS, g=G_BASE):
    """
    First-contact point on the floor: returns (x, y, t).

    Solves for the ball's CENTRE reaching z = z_floor + ball_radius, because
    that is where the centre is at the instant the sphere touches down. For a
    sphere the centre's (x, y) is the contact (x, y). Identical convention to
    `ray_plane.ball_center_on_plane` -- the two must agree or comparing them
    means nothing.
    """
    t = solve_impact_time(p0, v0, float(z_floor) + float(ball_radius), g=g)
    p = ballistic_position(p0, v0, t, g=g)
    return float(p[0]), float(p[1]), float(t)


def fit_two_points(t_a, p_a, t_b, p_b, g=G_BASE, min_dt=0.02):
    """
    Exact (p0, v0) through two timed 3-D points. Six equations, six unknowns.

    This is RANSAC's minimal sample. `min_dt` rejects a degenerate pair: two
    nearly simultaneous points leave v0 = dp/dt numerically meaningless, and the
    resulting wild trajectory would be scored against the data as if it were a
    real hypothesis.
    """
    g = np.asarray(g, float)
    t_a, t_b = float(t_a), float(t_b)
    if abs(t_b - t_a) < min_dt:
        raise RuntimeError(f"sample points are separated by only "
                           f"{abs(t_b - t_a) * 1e3:.1f} ms (need >= "
                           f"{min_dt * 1e3:.0f} ms) -- degenerate")
    q_a = np.asarray(p_a, float) - 0.5 * g * t_a ** 2
    q_b = np.asarray(p_b, float) - 0.5 * g * t_b ** 2
    v0 = (q_b - q_a) / (t_b - t_a)
    p0 = q_a - v0 * t_a
    return p0, v0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_trajectory.py -q`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add perception/trajectory.py tests/test_trajectory.py
git commit -m "feat(vision): ballistic kinematics and descending-root impact solve"
```

---

### Task 5: Gauss-Newton fit on pixel reprojection error

**Files:**
- Modify: `perception/trajectory.py`
- Test: `tests/test_trajectory.py`

**Interfaces:**
- Consumes: `StereoRig.project` (Task 1), `ballistic_position`, `fit_two_points` (Task 4).
- Produces: `perception.trajectory.FitResult` (dataclass: `p0`, `v0`, `cov` (6x6), `rms_px`, `n_obs`), and `fit_ballistic(obs, rig, R_bc, t_bc, g=G_BASE, max_iter=50, tol=1e-10) -> FitResult`.
- **Observation format, used by every later task:** `obs` is an `(N, 5)` float array with columns `[t, u1, v1, u2, v2]`.
- **Frame convention:** `R_bc`, `t_bc` are the camera pose in base coordinates, `p_base = R_bc @ p_cam + t_bc` — the same convention as `ray_plane.pixel_ray` and the stored `T_B_C`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_trajectory.py`:

```python
from perception.ray_plane import D435I_IR_848x480
from perception.stereo import D435I_IR_BASELINE_M, StereoRig
from perception.trajectory import FitResult, fit_ballistic

RIG = StereoRig(D435I_IR_848x480, D435I_IR_BASELINE_M)

# Overhead mount, camera at base-frame (0.82, 0, 1.767) looking straight down.
# Same R convention as tests/test_ray_plane.py::_overhead.
R_BC = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
T_BC = np.array([0.82, 0.0, 1.767])

TRUE_P0 = np.array([0.035, 0.0, 1.137])
TRUE_V0 = np.array([1.6218, 0.0, 0.1419])


def _synth_obs(times, p0=TRUE_P0, v0=TRUE_V0, noise_px=0.0, seed=0):
    """Project a known base-frame parabola into both IR images."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in times:
        p_b = ballistic_position(p0, v0, t)
        p_c = R_BC.T @ (p_b - T_BC)
        u1, v1, u2, v2 = RIG.project(p_c)
        rows.append([t, u1, v1, u2, v2])
    obs = np.asarray(rows, float)
    if noise_px:
        obs[:, 1:] += rng.normal(0.0, noise_px, size=obs[:, 1:].shape)
    return obs


def test_noiseless_fit_recovers_the_trajectory_exactly():
    obs = _synth_obs(np.linspace(0.13, 0.55, 40))
    fit = fit_ballistic(obs, RIG, R_BC, T_BC)
    assert isinstance(fit, FitResult)
    assert np.allclose(fit.p0, TRUE_P0, atol=1e-6)
    assert np.allclose(fit.v0, TRUE_V0, atol=1e-6)
    assert fit.rms_px < 1e-6
    assert fit.n_obs == 40


def test_landing_error_under_realistic_pixel_noise_beats_the_budget():
    """
    Spec's error budget claims ~4.4 mm total sigma at 0.15 px centroid noise.
    Assert the realised spread over 30 trials is under 1 cm -- comfortably
    inside the budget, but loose enough not to be a flaky test.
    """
    truth = solve_impact(TRUE_P0, TRUE_V0, z_floor=Z_FLOOR_BASE)
    errs = []
    for seed in range(30):
        obs = _synth_obs(np.linspace(0.13, 0.55, 40), noise_px=0.15, seed=seed)
        fit = fit_ballistic(obs, RIG, R_BC, T_BC)
        x, y, _ = solve_impact(fit.p0, fit.v0, z_floor=Z_FLOOR_BASE)
        errs.append(np.hypot(x - truth[0], y - truth[1]))
    assert np.mean(errs) < 0.010, f"mean landing error {np.mean(errs) * 1e3:.1f} mm"


def test_fit_reports_a_usable_covariance():
    obs = _synth_obs(np.linspace(0.13, 0.55, 40), noise_px=0.15, seed=7)
    fit = fit_ballistic(obs, RIG, R_BC, T_BC)
    assert fit.cov.shape == (6, 6)
    assert np.all(np.diag(fit.cov) > 0)
    assert np.allclose(fit.cov, fit.cov.T, atol=1e-12)


def test_fit_refuses_too_few_observations():
    """Spec section 6: fewer than 12 usable frames is a refusal, not a guess."""
    obs = _synth_obs(np.linspace(0.13, 0.55, 8))
    with pytest.raises(RuntimeError, match="12"):
        fit_ballistic(obs, RIG, R_BC, T_BC)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_trajectory.py -q`
Expected: FAIL at collection — `ImportError: cannot import name 'FitResult'`

- [ ] **Step 3: Implement the fit**

Add `from dataclasses import dataclass` to the imports at the TOP of
`perception/trajectory.py`, extend `__all__` with `"FitResult"`, `"fit_ballistic"`,
`"MIN_INLIER_FRAMES"`, `"MAX_RMS_PX"`, then append:

```python
MIN_INLIER_FRAMES = 12     # spec section 6: 2x redundancy over 6 unknowns
MAX_RMS_PX = 1.0           # spec section 6, vs 0.15 px assumed centroid precision


@dataclass
class FitResult:
    """A fitted trajectory and how much to trust it."""
    p0: np.ndarray
    v0: np.ndarray
    cov: np.ndarray        # 6x6 over [p0, v0]
    rms_px: float
    n_obs: int


def _residuals(theta, obs, rig, R_bc, t_bc, g):
    """Stacked (u1, v1, u2, v2) reprojection residuals, 4 per observation."""
    p0, v0 = theta[:3], theta[3:]
    p_b = ballistic_position(p0, v0, obs[:, 0], g=g)
    p_c = (p_b - t_bc) @ R_bc            # == (R_bc.T @ (p_b - t_bc).T).T
    u1, v1, u2, v2 = rig.project(p_c)
    pred = np.stack([u1, v1, u2, v2], axis=-1)
    return (pred - obs[:, 1:]).ravel()


def fit_ballistic(obs, rig, R_bc, t_bc, g=G_BASE, max_iter=50, tol=1e-10):
    """
    Fit (p0, v0) to timed stereo pixel observations by Gauss-Newton.

    `obs` is (N, 5): columns [t, u1, v1, u2, v2]. `R_bc`, `t_bc` are the camera
    pose in base coordinates (p_base = R_bc @ p_cam + t_bc).

    THE OBJECTIVE IS PIXEL ERROR, NOT ERROR AGAINST TRIANGULATED POINTS.
    Pixel noise is the iid quantity. Triangulated points carry correlated,
    strongly range-dependent noise (a fixed 0.2 px of disparity is 4 cm at 2 m
    and 1 cm at 1 m), so least-squares over them silently weights the far,
    noisier frames as heavily as the near ones. Fitting in pixels is the
    statistically correct thing and costs nothing offline.

    The Jacobian is computed by central differences ON PURPOSE. It is 6 columns
    over a few hundred residuals -- microseconds offline -- and this codebase has
    a long history of silent sign errors in hand-derived geometry. A wrong
    analytic Jacobian does not crash; it converges somewhere plausible.
    """
    obs = np.asarray(obs, float)
    if obs.ndim != 2 or obs.shape[1] != 5:
        raise ValueError(f"obs must be (N, 5) [t,u1,v1,u2,v2], got {obs.shape}")
    if obs.shape[0] < MIN_INLIER_FRAMES:
        raise RuntimeError(
            f"only {obs.shape[0]} observations, need >= {MIN_INLIER_FRAMES} "
            f"(2x redundancy over 6 unknowns) -- refusing to fit")

    R_bc = np.asarray(R_bc, float); t_bc = np.asarray(t_bc, float)
    g = np.asarray(g, float)
    order = np.argsort(obs[:, 0])
    obs = obs[order]

    # Initialise from the two most widely separated frames, triangulated.
    p_first = rig.triangulate(*obs[0, 1:])
    p_last = rig.triangulate(*obs[-1, 1:])
    to_base = lambda p_c: R_bc @ p_c + t_bc
    p0, v0 = fit_two_points(obs[0, 0], to_base(p_first),
                            obs[-1, 0], to_base(p_last), g=g)
    theta = np.concatenate([p0, v0])

    r = _residuals(theta, obs, rig, R_bc, t_bc, g)
    for _ in range(max_iter):
        J = np.empty((r.size, 6))
        for k in range(6):
            step = 1e-6 * max(1.0, abs(theta[k]))
            tp = theta.copy(); tp[k] += step
            tm = theta.copy(); tm[k] -= step
            J[:, k] = (_residuals(tp, obs, rig, R_bc, t_bc, g)
                       - _residuals(tm, obs, rig, R_bc, t_bc, g)) / (2.0 * step)
        delta, *_ = np.linalg.lstsq(J, -r, rcond=None)
        theta = theta + delta
        r = _residuals(theta, obs, rig, R_bc, t_bc, g)
        if np.linalg.norm(delta) < tol:
            break

    m = r.size
    dof = m - 6
    if dof <= 0:
        raise RuntimeError(f"{m} residuals cannot constrain 6 parameters")
    sigma2 = float(r @ r) / dof
    JtJ = J.T @ J
    try:
        cov = sigma2 * np.linalg.inv(JtJ)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(f"singular normal equations -- the trajectory is "
                           f"not observable from these frames: {exc}") from exc

    rms_px = float(np.sqrt(float(r @ r) / m))
    return FitResult(p0=theta[:3].copy(), v0=theta[3:].copy(), cov=cov,
                     rms_px=rms_px, n_obs=int(obs.shape[0]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_trajectory.py -q`
Expected: PASS, 12 tests.

- [ ] **Step 5: Commit**

```bash
git add perception/trajectory.py tests/test_trajectory.py
git commit -m "feat(vision): Gauss-Newton ballistic fit on pixel reprojection error"
```

---

### Task 6: RANSAC ballistic association

**Files:**
- Modify: `perception/trajectory.py`
- Test: `tests/test_trajectory.py`

**Interfaces:**
- Consumes: `fit_two_points`, `fit_ballistic`, `StereoRig`.
- Produces: `perception.trajectory.ransac_track(obs, rig, R_bc, t_bc, g=G_BASE, thresh_px=2.0, min_inlier_frac=0.6, iters=300, seed=0) -> (inlier_idx: ndarray, FitResult)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_trajectory.py`:

```python
from perception.trajectory import ransac_track


def _arm_like_outliers(n, seed=3):
    """
    Rows that look like detections but do not lie on ANY g=9.81 parabola --
    what the moving arm, a reflection, or the second bounce produce.
    """
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.13, 0.55, size=n)
    u1 = rng.uniform(100, 700, size=n)
    v1 = rng.uniform(60, 420, size=n)
    return np.stack([t, u1, v1, u1 - rng.uniform(8, 20, size=n), v1], axis=-1)


def test_ransac_rejects_arm_like_outliers_and_recovers_the_ball():
    good = _synth_obs(np.linspace(0.13, 0.55, 40), noise_px=0.15, seed=1)
    obs = np.vstack([good, _arm_like_outliers(12)])
    idx, fit = ransac_track(obs, RIG, R_BC, T_BC)
    assert len(idx) >= 36, f"kept only {len(idx)} of 40 true inliers"
    assert set(idx.tolist()).issubset(set(range(40))), "an outlier was kept"
    assert np.allclose(fit.p0, TRUE_P0, atol=0.02)
    assert np.allclose(fit.v0, TRUE_V0, atol=0.05)


def test_ransac_refuses_when_the_inlier_fraction_is_too_low():
    """Spec section 6: below 0.6 inliers is a refusal, not a best effort."""
    good = _synth_obs(np.linspace(0.13, 0.55, 14), noise_px=0.15, seed=2)
    obs = np.vstack([good, _arm_like_outliers(40)])
    with pytest.raises(RuntimeError, match="inlier"):
        ransac_track(obs, RIG, R_BC, T_BC)


def test_ransac_is_deterministic_for_a_fixed_seed():
    obs = np.vstack([_synth_obs(np.linspace(0.13, 0.55, 40), noise_px=0.15, seed=1),
                     _arm_like_outliers(12)])
    a, _ = ransac_track(obs, RIG, R_BC, T_BC, seed=11)
    b, _ = ransac_track(obs, RIG, R_BC, T_BC, seed=11)
    assert np.array_equal(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_trajectory.py -q -k ransac`
Expected: FAIL — `ImportError: cannot import name 'ransac_track'`

- [ ] **Step 3: Implement `ransac_track`**

Append to `perception/trajectory.py` (add `"ransac_track"` to `__all__`):

```python
def ransac_track(obs, rig, R_bc, t_bc, g=G_BASE, thresh_px=2.0,
                 min_inlier_frac=0.6, iters=300, seed=0):
    """
    Pick out the frames that lie on one ballistic arc, then fit them.

    THIS IS THE ARM REJECTOR. The arm moves fast in exactly the region the ball
    starts in, and it is the dominant false-positive source. Rather than mask it
    out by hand -- which would need re-drawing for every mount and every pose,
    and would silently clip real detections -- this exploits the one property the
    ball has and the arm does not: the ball's positions fit a parabola with
    g = 9.81. Reflections and the second bounce fall out for the same reason.

    Minimal sample is 2 timed 3-D points (6 equations, 6 unknowns). Scoring is
    in pixels, consistent with `fit_ballistic`'s objective.

    Returns (inlier_indices_into_obs, FitResult).
    """
    obs = np.asarray(obs, float)
    if obs.ndim != 2 or obs.shape[1] != 5:
        raise ValueError(f"obs must be (N, 5) [t,u1,v1,u2,v2], got {obs.shape}")
    n = obs.shape[0]
    if n < MIN_INLIER_FRAMES:
        raise RuntimeError(f"only {n} observations, need >= {MIN_INLIER_FRAMES}")

    R_bc = np.asarray(R_bc, float); t_bc = np.asarray(t_bc, float)
    g = np.asarray(g, float)
    rng = np.random.default_rng(seed)

    # Triangulate once. A pair that cannot be triangulated at all (bad disparity,
    # mismatched row) is dropped here rather than poisoning the sampling.
    usable, pts_b = [], []
    for i in range(n):
        try:
            p_c = rig.triangulate(*obs[i, 1:])
        except RuntimeError:
            continue
        usable.append(i)
        pts_b.append(R_bc @ p_c + t_bc)
    usable = np.asarray(usable, int)
    if usable.size < MIN_INLIER_FRAMES:
        raise RuntimeError(
            f"only {usable.size} of {n} observations triangulate at all "
            f"(need >= {MIN_INLIER_FRAMES}) -- check left/right pairing")
    pts_b = np.asarray(pts_b, float)

    best_idx = np.empty(0, dtype=int)
    for _ in range(iters):
        a, b = rng.choice(usable.size, size=2, replace=False)
        try:
            p0, v0 = fit_two_points(obs[usable[a], 0], pts_b[a],
                                    obs[usable[b], 0], pts_b[b], g=g)
        except RuntimeError:
            continue
        try:
            err = np.abs(_residuals(np.concatenate([p0, v0]), obs[usable],
                                    rig, R_bc, t_bc, g)).reshape(-1, 4)
        except RuntimeError:
            # A wild minimal-sample hypothesis can put the arc behind the camera,
            # where project() has no answer. That is a rejected hypothesis, not an
            # error -- drop the sample and keep searching.
            continue
        inl = usable[np.max(err, axis=1) <= thresh_px]
        if inl.size > best_idx.size:
            best_idx = inl

    frac = best_idx.size / float(n)
    if best_idx.size < MIN_INLIER_FRAMES or frac < min_inlier_frac:
        raise RuntimeError(
            f"no ballistic arc found: best inlier consensus {best_idx.size}/{n} "
            f"frames (inlier fraction {frac:.2f}, need >= {min_inlier_frac} and "
            f">= {MIN_INLIER_FRAMES} frames) -- this recording does not contain "
            f"a clean throw")

    fit = fit_ballistic(obs[best_idx], rig, R_bc, t_bc, g=g)
    if fit.rms_px > MAX_RMS_PX:
        raise RuntimeError(
            f"fit RMS {fit.rms_px:.2f} px exceeds {MAX_RMS_PX} px -- the frames "
            f"agree on an arc but not a good one; do not use this landing point")
    return best_idx, fit
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_trajectory.py -q`
Expected: PASS, 15 tests.

- [ ] **Step 5: Commit**

```bash
git add perception/trajectory.py tests/test_trajectory.py
git commit -m "feat(vision): RANSAC ballistic association rejects the arm"
```

---

### Task 7: End-to-end pipeline over synthetic images

**Files:**
- Create: `measure_landing.py` (the `measure_landing()` function only; the CLI is Task 10)
- Test: `tests/test_landing_pipeline.py`

**Interfaces:**
- Consumes: everything from Tasks 1-6.
- Produces: `measure_landing.build_observations(rec, bg1=None, bg2=None, **detect_kw) -> (ndarray (N, 5), max_mask_frac: float)`; `measure_landing.measure_landing(rec, R_bc, t_bc, z_floor=Z_FLOOR_BASE, ball_radius=BALL_RADIUS, seed=0, rig=None, **detect_kw) -> dict` with keys `x`, `y`, `t_impact`, `sigma_xy_m`, `n_frames`, `n_inliers`, `rms_px`, `p0`, `v0`, `max_mask_frac`.
- **Recording format, used by Tasks 8-10:** a dict with `t` `(N,)` float seconds, `ir1` `(N, H, W)` uint8, `ir2` `(N, H, W)` uint8, `meta` dict.

- [ ] **Step 1: Write the failing test**

Create `tests/test_landing_pipeline.py`:

```python
"""
Spec section 7.1 -- synthetic end-to-end. Renders a known parabola into two
IMAGES (not just observations), runs the whole pipeline, and checks the landing
point against ground truth.

This is the test that would catch a detector/pairing/fit integration error that
every unit test passes individually.
"""
import cv2
import numpy as np
import pytest

from measure_landing import build_observations, measure_landing
from perception.ray_plane import D435I_IR_848x480
from perception.stereo import D435I_IR_BASELINE_M, StereoRig
from perception.trajectory import (BALL_RADIUS, Z_FLOOR_BASE,
                                   ballistic_position, solve_impact)

RIG = StereoRig(D435I_IR_848x480, D435I_IR_BASELINE_M)
R_BC = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
T_BC = np.array([0.82, 0.0, 1.767])
TRUE_P0 = np.array([0.035, 0.0, 1.137])
TRUE_V0 = np.array([1.6218, 0.0, 0.1419])
H, W = 480, 848


SHIFT = 4  # cv2 sub-pixel fixed point: 1/16 px


def _disc(img, u, v, r):
    """
    Filled disc at a SUB-PIXEL centre.

    An integer centre would quantise the ground truth to 0.5 px -- 3.3x the
    0.15 px centroid precision the error budget assumes -- so the test would be
    measuring cv2's rounding, not the pipeline.
    """
    k = 1 << SHIFT
    cv2.circle(img, (int(round(u * k)), int(round(v * k))),
               int(round(max(2.0, r) * k)), 225, -1, shift=SHIFT)


def _render(times, seed=0):
    """
    Draw the ball into both IR images at its true projected position.

    Frames where the disc would touch an image edge are SKIPPED. Before
    t ~= 0.120 s the ball is off the left edge of IR2 (u2 = -10.9 px at
    t = 0.10), and a clipped disc has a biased centroid -- which would look
    exactly like a pipeline error.
    """
    rng = np.random.default_rng(seed)
    base = (40 + rng.integers(0, 12, size=(H, W))).astype(np.uint8)
    ts, ir1, ir2 = [], [], []
    for t in times:
        p_c = R_BC.T @ (ballistic_position(TRUE_P0, TRUE_V0, t) - T_BC)
        u1, v1, u2, v2 = RIG.project(p_c)
        r_px = D435I_IR_848x480.fx * BALL_RADIUS / p_c[2]
        inside = all(r_px <= u <= W - 1 - r_px for u in (u1, u2)) and \
                 all(r_px <= v <= H - 1 - r_px for v in (v1, v2))
        if not inside:
            continue
        f1, f2 = base.copy(), base.copy()
        _disc(f1, u1, v1, r_px)
        _disc(f2, u2, v2, r_px)
        ts.append(t); ir1.append(f1); ir2.append(f2)
    if len(ts) < 20:
        raise AssertionError(f"only {len(ts)} frames rendered in view -- "
                             f"the test time window is wrong, not the pipeline")
    return {"t": np.asarray(ts, float), "ir1": np.asarray(ir1),
            "ir2": np.asarray(ir2), "meta": {"synthetic": True}}


# One real throw at 90 fps, from the frame the ball clears IR2's edge to impact.
FLIGHT_TIMES = np.arange(0.12, 0.58, 1.0 / 90.0)


def test_end_to_end_landing_point_matches_truth():
    rec = _render(FLIGHT_TIMES)
    got = measure_landing(rec, R_BC, T_BC)
    x_true, y_true, _ = solve_impact(TRUE_P0, TRUE_V0, z_floor=Z_FLOOR_BASE)
    err = np.hypot(got["x"] - x_true, got["y"] - y_true)
    assert err < 0.02, f"landing error {err * 1e3:.1f} mm"
    assert got["n_inliers"] >= 0.8 * len(rec["t"])
    assert got["rms_px"] < 1.0
    assert got["sigma_xy_m"] > 0


def test_observations_are_built_for_most_rendered_frames():
    rec = _render(FLIGHT_TIMES)
    obs, max_frac = build_observations(rec)
    assert max_frac < 0.01, "a synthetic scene must not look like lighting drift"
    assert obs.shape[1] == 5
    assert obs.shape[0] >= 0.9 * len(rec["t"]), \
        f"only {obs.shape[0]} of {len(rec['t'])} rendered frames paired"


def test_a_recording_with_no_ball_raises_rather_than_inventing_a_landing():
    rng = np.random.default_rng(1)
    flat = (40 + rng.integers(0, 12, size=(20, H, W))).astype(np.uint8)
    rec = {"t": np.linspace(0, 0.2, 20), "ir1": flat, "ir2": flat.copy(), "meta": {}}
    with pytest.raises(RuntimeError):
        measure_landing(rec, R_BC, T_BC)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_landing_pipeline.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'measure_landing'`

- [ ] **Step 3: Write `measure_landing.py`**

```python
"""
Recording -> the ball's first contact point with the floor, in the base frame.

This is the offline half of the vision pipeline: it never touches the camera.
`record_throw_ir.py` captures; this interprets. Keeping them apart means every
recording is a permanent regression fixture -- a changed fitter can be re-run
against a real throw from weeks ago, which matters in a project with this much
history of plausible-looking wrong numbers.

    from measure_landing import measure_landing
    result = measure_landing(rec, R_bc, t_bc)
"""

from __future__ import annotations

import numpy as np

from perception.ball_track import (detect_candidates, frame_diagnostics,
                                   median_background)
from perception.ray_plane import D435I_IR_848x480
from perception.stereo import D435I_IR_BASELINE_M, StereoRig, pair_candidates
from perception.trajectory import (BALL_RADIUS, G_BASE, Z_FLOOR_BASE,
                                   ransac_track, solve_impact)

__all__ = ["build_observations", "measure_landing", "default_rig"]


def default_rig():
    return StereoRig(D435I_IR_848x480, D435I_IR_BASELINE_M)


def build_observations(rec, bg1=None, bg2=None, **detect_kw):
    """
    Recording -> ((N, 5) observation array [t, u1, v1, u2, v2], max_mask_frac).

    Backgrounds default to the per-pixel temporal median of the recording
    itself, which needs no separate empty-scene capture and cannot drift
    relative to the throw.

    Frames with no detection are skipped silently -- every frame before the ball
    enters view is one of those, and it is not an error. Frames with several
    candidates contribute several rows; deciding which is the ball is
    `ransac_track`'s job, not this function's.
    """
    ir1, ir2, ts = rec["ir1"], rec["ir2"], np.asarray(rec["t"], float)
    if len(ir1) != len(ir2) or len(ir1) != len(ts):
        raise ValueError(f"ragged recording: {len(ir1)} ir1, {len(ir2)} ir2, "
                         f"{len(ts)} timestamps")
    if bg1 is None:
        bg1 = median_background(ir1)
    if bg2 is None:
        bg2 = median_background(ir2)

    rows, max_frac = [], 0.0
    for k in range(len(ts)):
        max_frac = max(max_frac,
                       frame_diagnostics(ir1[k], bg1)["mask_nonzero_frac"],
                       frame_diagnostics(ir2[k], bg2)["mask_nonzero_frac"])
        left = detect_candidates(ir1[k], bg1, **detect_kw)
        right = detect_candidates(ir2[k], bg2, **detect_kw)
        if not left or not right:
            continue
        lt = [c.as_uv_area() for c in left]
        rt = [c.as_uv_area() for c in right]
        for li, ri in pair_candidates(lt, rt):
            rows.append([ts[k], lt[li][0], lt[li][1], rt[ri][0], rt[ri][1]])

    if max_frac > 0.15:
        raise RuntimeError(
            f"a frame changed over {max_frac:.0%} of its pixels against the "
            f"background -- the camera was bumped, the lighting shifted, or "
            f"auto-exposure resettled mid-capture. Re-record; do not loosen the "
            f"detector to work around it")
    if not rows:
        raise RuntimeError(
            "no left/right ball candidates paired in any frame -- either the "
            "ball never entered view, the detector thresholds are wrong for "
            "this exposure, or the two streams are misaligned")
    return np.asarray(rows, float), max_frac


def measure_landing(rec, R_bc, t_bc, z_floor=Z_FLOOR_BASE,
                    ball_radius=BALL_RADIUS, seed=0, rig=None, **detect_kw):
    """
    The whole offline pipeline, in base-frame coordinates.

    `R_bc`, `t_bc` are the camera pose in base coordinates
    (p_base = R_bc @ p_cam + t_bc), i.e. the stored T_B_C.

    Returns a dict: x, y, t_impact, sigma_xy_m, n_frames, n_inliers, rms_px,
    p0, v0. Raises RuntimeError at the first stage that cannot honestly proceed.

    `sigma_xy_m` propagates the fit covariance to the landing point by linearising
    solve_impact around the estimate. A landing point WITHOUT a sigma is not
    eligible to become a GP datapoint, which is why this is returned and not
    merely logged.
    """
    rig = rig or default_rig()
    obs, max_frac = build_observations(rec, **detect_kw)
    inliers, fit = ransac_track(obs, rig, R_bc, t_bc, seed=seed)
    x, y, t_imp = solve_impact(fit.p0, fit.v0, z_floor=z_floor,
                               ball_radius=ball_radius)

    # Linearised propagation: d(x,y)/d(theta) by central differences on the same
    # impact solve the answer came from, so the sigma describes THIS estimator.
    theta = np.concatenate([fit.p0, fit.v0])
    Jl = np.zeros((2, 6))
    for k in range(6):
        step = 1e-6 * max(1.0, abs(theta[k]))
        tp, tm = theta.copy(), theta.copy()
        tp[k] += step; tm[k] -= step
        xp, yp, _ = solve_impact(tp[:3], tp[3:], z_floor=z_floor, ball_radius=ball_radius)
        xm, ym, _ = solve_impact(tm[:3], tm[3:], z_floor=z_floor, ball_radius=ball_radius)
        Jl[:, k] = [(xp - xm) / (2 * step), (yp - ym) / (2 * step)]
    cov_xy = Jl @ fit.cov @ Jl.T

    return {"x": x, "y": y, "t_impact": t_imp,
            "sigma_xy_m": float(np.sqrt(max(np.trace(cov_xy), 0.0))),
            "n_frames": int(obs.shape[0]), "n_inliers": int(inliers.size),
            "rms_px": fit.rms_px, "p0": fit.p0, "v0": fit.v0,
            "max_mask_frac": max_frac}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_landing_pipeline.py -q`
Expected: PASS, 3 tests.

- [ ] **Step 5: Run the whole suite**

Run: `python3 -m pytest tests/ -q`
Expected: all pre-existing tests plus the ~32 new ones pass.

- [ ] **Step 6: Commit**

```bash
git add measure_landing.py tests/test_landing_pipeline.py
git commit -m "feat(vision): end-to-end synthetic landing measurement"
```

---

### Task 8: Dual-IR recorder

**Files:**
- Create: `perception/ir_capture.py`
- Test: manual, against the camera (documented in the step)

**Interfaces:**
- Consumes: nothing.
- Produces: `perception.ir_capture.IRRecorder(width=848, height=480, fps=90, exposure_us=2000, gain=None, emitter=True)` with `.record(seconds) -> rec_dict`, plus `save_recording(path, rec)` and `load_recording(path) -> rec_dict`. Recording dict matches Task 7's format.

- [ ] **Step 1: Write `perception/ir_capture.py`**

```python
"""
Record both D435i infrared imagers through a throw.

WHY BOTH IR STREAMS AND NOT COLOUR OR DEPTH
--------------------------------------------
Both IR imagers are global shutter (OV9282); the colour imager (OV2740) is
rolling shutter, which skews a fast ball. The IR field of view is also much
wider (89.7 x 58.8 deg vs 70.2 x 43.2), and that is what puts the flight in
frame from an overhead mount at all. The DEPTH stream is deliberately not
enabled: this project does not use the block-matching depth map as a position
source (see perception/stereo.py).

WHY RECORD-THEN-DUMP
--------------------
848x480 y8 is 407 kB per image, x2 cameras x 90 fps = 73 MB/s. A 2 s window is
146 MB, which sits in RAM comfortably. Writing during capture risks a disk stall
dropping frames in the middle of the flight, and there is no reason to accept
that when the whole window fits in memory.

TIMESTAMPS
----------
Frames are timestamped at MID-EXPOSURE, from the sensor timestamp, not at
arrival. At 5.8 m/s an 8.5 ms exposure is 5 cm of travel, so the choice of time
reference is a systematic error, not noise -- and it must agree with
ball_track.detect_candidates, whose intensity-weighted centroid is unbiased at
mid-exposure.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pyrealsense2 as rs

__all__ = ["IRRecorder", "save_recording", "load_recording"]


class IRRecorder:
    """Dual-IR capture into RAM. One throw per `record()` call."""

    def __init__(self, width=848, height=480, fps=90, exposure_us=2000,
                 gain=None, emitter=True):
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self.exposure_us = int(exposure_us)
        self.gain = gain
        self.emitter = bool(emitter)
        self._pipe = None

    def __enter__(self):
        cfg = rs.config()
        cfg.enable_stream(rs.stream.infrared, 1, self.width, self.height,
                          rs.format.y8, self.fps)
        cfg.enable_stream(rs.stream.infrared, 2, self.width, self.height,
                          rs.format.y8, self.fps)
        self._pipe = rs.pipeline()
        profile = self._pipe.start(cfg)
        sensor = profile.get_device().first_depth_sensor()
        # Manual exposure. Auto-exposure will happily pick 8.5 ms, which is 5 cm
        # of motion blur at impact speed, and will also change between frames --
        # both are fatal to a subpixel centroid.
        sensor.set_option(rs.option.enable_auto_exposure, 0)
        sensor.set_option(rs.option.exposure, float(self.exposure_us))
        if self.gain is not None:
            sensor.set_option(rs.option.gain, float(self.gain))
        if sensor.supports(rs.option.emitter_enabled):
            sensor.set_option(rs.option.emitter_enabled, 1 if self.emitter else 0)
        self._profile = profile
        return self

    def __exit__(self, *exc):
        if self._pipe is not None:
            self._pipe.stop()
            self._pipe = None

    def record(self, seconds):
        """Capture for `seconds`, return the recording dict."""
        if self._pipe is None:
            raise RuntimeError("use IRRecorder as a context manager")
        n_expect = int(np.ceil(seconds * self.fps)) + 8
        ir1 = np.empty((n_expect, self.height, self.width), np.uint8)
        ir2 = np.empty((n_expect, self.height, self.width), np.uint8)
        ts = np.empty(n_expect, float)
        half_exp_s = 0.5 * self.exposure_us * 1e-6

        k, t_end = 0, time.monotonic() + float(seconds)
        while time.monotonic() < t_end and k < n_expect:
            fs = self._pipe.wait_for_frames(2000)
            f1 = fs.get_infrared_frame(1)
            f2 = fs.get_infrared_frame(2)
            if not f1 or not f2:
                continue
            ir1[k] = np.asanyarray(f1.get_data())
            ir2[k] = np.asanyarray(f2.get_data())
            ts[k] = f1.get_timestamp() * 1e-3 + half_exp_s   # ms -> s, mid-exposure
            k += 1

        if k == 0:
            raise RuntimeError("captured zero frames -- is the camera streaming?")
        ts = ts[:k] - ts[0]
        meta = {"width": self.width, "height": self.height, "fps": self.fps,
                "exposure_us": self.exposure_us, "gain": self.gain,
                "emitter": self.emitter, "n_frames": int(k),
                "achieved_fps": float((k - 1) / max(ts[-1], 1e-9)) if k > 1 else 0.0}
        return {"t": ts, "ir1": ir1[:k], "ir2": ir2[:k], "meta": meta}


def save_recording(path, rec):
    """
    Uncompressed .npz on purpose -- savez_compressed spends ~30 s on 146 MB of
    uint8 for a modest saving, and the point of dumping after the throw is to be
    back to ready quickly.
    """
    np.savez(path, t=rec["t"], ir1=rec["ir1"], ir2=rec["ir2"],
             meta=json.dumps(rec["meta"]))


def load_recording(path):
    z = np.load(path, allow_pickle=False)
    return {"t": z["t"], "ir1": z["ir1"], "ir2": z["ir2"],
            "meta": json.loads(str(z["meta"]))}
```

- [ ] **Step 2: Verify the module imports and the round-trip works without a camera**

Run:

```bash
python3 - <<'PY'
import numpy as np, tempfile, os
from perception.ir_capture import save_recording, load_recording
rec = {"t": np.linspace(0, 0.2, 5),
       "ir1": np.zeros((5, 8, 8), np.uint8),
       "ir2": np.ones((5, 8, 8), np.uint8),
       "meta": {"fps": 90, "emitter": True}}
p = os.path.join(tempfile.mkdtemp(), "r.npz")
save_recording(p, rec)
got = load_recording(p)
assert np.array_equal(got["ir2"], rec["ir2"]) and got["meta"]["fps"] == 90
print("round-trip OK")
PY
```

Expected: `round-trip OK`

- [ ] **Step 3: Verify against the real camera**

Run:

```bash
python3 - <<'PY'
from perception.ir_capture import IRRecorder
with IRRecorder(exposure_us=2000) as r:
    rec = r.record(1.0)
print(rec["meta"])
print("shapes", rec["ir1"].shape, rec["ir2"].shape, "t span %.3f s" % rec["t"][-1])
PY
```

Expected: `achieved_fps` close to 90 and both arrays `(N, 480, 848)`. **If `achieved_fps` is far below 90, stop and report it** — a dropped-frame rate changes the error budget and must not be worked around silently.

- [ ] **Step 4: Commit**

```bash
git add perception/ir_capture.py
git commit -m "feat(vision): dual-IR throw recorder with mid-exposure timestamps"
```

---

### Task 9: Exposure and emitter A/B tool

**Files:**
- Create: `tune_ir_exposure.py`

**Interfaces:**
- Consumes: `IRRecorder`, `median_background`, `detect_candidates`.
- Produces: a CLI only. No importable API other tasks depend on.

This resolves spec §9's two deliberately-unresolved questions with measurement instead of argument.

- [ ] **Step 1: Write `tune_ir_exposure.py`**

```python
"""
Settle the two questions spec section 9 deliberately left open: emitter on or
off, and what exposure.

Neither is answerable from a datasheet. The IR projector's dot pattern is static
on the floor, so background subtraction removes it -- and on the BALL it adds
texture, which helps. But it can also saturate. Short exposure cuts motion blur
(2 ms is 2.2 px at the 5.8 m/s impact speed) but may leave the scene too dark
indoors with the emitter off. Ten minutes of A/B beats any amount of reasoning.

Wave the ball through the frame during each capture.

    python3 tune_ir_exposure.py
    python3 tune_ir_exposure.py --exposures 1000 2000 4000 8000
"""
import argparse

import numpy as np

from perception.ball_track import detect_candidates, median_background
from perception.ir_capture import IRRecorder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exposures", type=int, nargs="+", default=[1000, 2000, 4000, 8000])
    ap.add_argument("--seconds", type=float, default=1.5)
    args = ap.parse_args()

    print(f"{'emitter':>8} {'exp_us':>7} {'fps':>6} {'mean_lvl':>9} "
          f"{'sat%':>6} {'frames_with_ball':>17} {'med_area':>9} {'med_circ':>9}")
    for emitter in (True, False):
        for exp in args.exposures:
            with IRRecorder(exposure_us=exp, emitter=emitter) as r:
                rec = r.record(args.seconds)
            ir1 = rec["ir1"]
            bg = median_background(ir1)
            areas, circs, hits = [], [], 0
            for f in ir1:
                cands = detect_candidates(f, bg)
                if cands:
                    hits += 1
                    best = max(cands, key=lambda c: c.circularity)
                    areas.append(best.area_px); circs.append(best.circularity)
            sat = 100.0 * float(np.mean(ir1 >= 250))
            print(f"{str(emitter):>8} {exp:>7} {rec['meta']['achieved_fps']:>6.1f} "
                  f"{ir1.mean():>9.1f} {sat:>6.2f} {hits:>10d}/{len(ir1):<6d} "
                  f"{np.median(areas) if areas else float('nan'):>9.1f} "
                  f"{np.median(circs) if circs else float('nan'):>9.2f}")

    print("\nPick the row with the most frames_with_ball at the SHORTEST exposure,")
    print("with sat% near zero. Record the choice in HARDWARE_RUNBOOK.md.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it against the camera**

Run: `python3 tune_ir_exposure.py`
Expected: a table. Wave the ball through the frame during each of the 8 captures.

- [ ] **Step 3: Record the chosen settings**

Add a line to `HARDWARE_RUNBOOK.md` naming the chosen `exposure_us` and `emitter` setting and the `frames_with_ball` figure that justified it, then change the `IRRecorder` defaults in `perception/ir_capture.py` to match.

- [ ] **Step 4: Commit**

```bash
git add tune_ir_exposure.py perception/ir_capture.py HARDWARE_RUNBOOK.md
git commit -m "feat(vision): IR exposure/emitter A/B, and the settings it chose"
```

---

### Task 10: Capture and measurement CLIs

**Files:**
- Create: `record_throw_ir.py`
- Modify: `measure_landing.py` (add the CLI; the API from Task 7 is unchanged)

**Interfaces:**
- Consumes: everything above.
- Produces: two CLIs. `measure_landing.py` gains `load_extrinsic(path) -> (R_bc, t_bc)`.

- [ ] **Step 1: Write `record_throw_ir.py`**

```python
"""
Capture one throw to disk.

    python3 record_throw_ir.py --out throws/2026-08-25_t01.npz
    python3 record_throw_ir.py --out throws/t02.npz --seconds 2.5 --countdown 5

Start this, then throw. The window only has to CONTAIN the flight -- the
median-background detector needs no separate empty-scene capture, and frames
before the ball appears cost nothing.
"""
import argparse
import os
import time

from perception.ir_capture import IRRecorder, save_recording


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--countdown", type=int, default=3)
    ap.add_argument("--exposure_us", type=int, default=2000)
    ap.add_argument("--no_emitter", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with IRRecorder(exposure_us=args.exposure_us, emitter=not args.no_emitter) as r:
        for k in range(args.countdown, 0, -1):
            print(f"  {k}...", flush=True)
            time.sleep(1.0)
        print("  THROW", flush=True)
        rec = r.record(args.seconds)

    save_recording(args.out, rec)
    m = rec["meta"]
    print(f"saved {args.out}: {m['n_frames']} frames at {m['achieved_fps']:.1f} fps "
          f"(exposure {m['exposure_us']} us, emitter {m['emitter']})")
    if m["achieved_fps"] < 0.9 * m["fps"]:
        print(f"  CHECK: achieved {m['achieved_fps']:.1f} fps against a requested "
              f"{m['fps']} -- dropped frames widen the error budget")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add the CLI to `measure_landing.py`**

Append to `measure_landing.py`:

```python
def load_extrinsic(path):
    """
    Load T_B_C from a .npz with `R` (3x3) and `t` (3,), the convention used
    throughout this project: p_base = R @ p_cam + t.
    """
    z = np.load(path)
    R, t = np.asarray(z["R"], float), np.asarray(z["t"], float)
    if R.shape != (3, 3) or t.shape != (3,):
        raise ValueError(f"expected R (3,3) and t (3,), got {R.shape} and {t.shape}")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
        raise ValueError("R is not orthonormal -- this is not a rotation")
    return R, t


def main():
    import argparse

    from perception.ir_capture import load_recording

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recording", required=True)
    ap.add_argument("--extrinsic", required=True, help=".npz with R (3x3), t (3,)")
    ap.add_argument("--z_floor", type=float, default=Z_FLOOR_BASE)
    ap.add_argument("--ball_radius", type=float, default=BALL_RADIUS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rec = load_recording(args.recording)
    R_bc, t_bc = load_extrinsic(args.extrinsic)
    out = measure_landing(rec, R_bc, t_bc, z_floor=args.z_floor,
                          ball_radius=args.ball_radius, seed=args.seed)

    print(f"landing (base frame): x = {out['x']:+.4f} m   y = {out['y']:+.4f} m")
    print(f"  sigma            : {out['sigma_xy_m'] * 1e3:.1f} mm")
    print(f"  impact at t      : {out['t_impact']:.4f} s")
    print(f"  frames / inliers : {out['n_frames']} / {out['n_inliers']}")
    print(f"  fit RMS          : {out['rms_px']:.3f} px")
    print(f"  max changed-px   : {out['max_mask_frac']:.2%}  (>15% = bumped camera)")
    print(f"  release p0       : {np.array2string(out['p0'], precision=4)}")
    print(f"  release v0       : {np.array2string(out['v0'], precision=4)}  "
          f"|v0| = {np.linalg.norm(out['v0']):.4f} m/s")
    print("\nNOTE: absolute accuracy is bounded by T_B_C, not by the vision. "
          "Confirm the extrinsic is current for the present mount.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify both CLIs run end to end on synthetic data**

Run:

```bash
python3 - <<'PY'
import numpy as np, os, sys, tempfile
sys.path.insert(0, "tests")
import test_landing_pipeline as T
from perception.ir_capture import save_recording
d = tempfile.mkdtemp()
rec = T._render(T.FLIGHT_TIMES)
save_recording(os.path.join(d, "rec.npz"), rec)
np.savez(os.path.join(d, "ext.npz"), R=T.R_BC, t=T.T_BC)
print(d)
PY
```

Then run `measure_landing.py` against the printed directory:

```bash
python3 measure_landing.py --recording <dir>/rec.npz --extrinsic <dir>/ext.npz
```

Expected: `x` near `+0.9667`, `y` near `0.000`, sigma a few mm, inliers >= 36.
(0.9667, not 0.976: `measure_landing` defaults to `ball_radius = 0.0327`, so it solves
for the centre reaching `z_floor + r`, which lands 9.6 mm shorter than a zero-radius solve.)

- [ ] **Step 4: Run the full suite once more**

Run: `python3 -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add record_throw_ir.py measure_landing.py
git commit -m "feat(vision): capture and landing-measurement CLIs"
```

---

### Task 11: Documentation and the real-throw acceptance gate

**Files:**
- Modify: `CLAUDE.md` (the `mc-pilot-pybullet/` bullet list under "The active track")
- Modify: `HARDWARE_RUNBOOK.md`
- Modify: `status_update/HANDOFF.md`

**Interfaces:** none — documentation.

- [ ] **Step 1: Document the pipeline in `CLAUDE.md`**

Add a bullet to the `mc-pilot-pybullet/` section covering: the four new `perception/` modules and two CLIs; that landing is measured as **first contact**, not resting position, and why that differs; that this triangulates IR blob centroids and is **not** the block-matching depth map the repo rejects; and that absolute accuracy is bounded by `T_B_C`, which is still the laptop-rig calibration.

- [ ] **Step 2: Add the run-day procedure to `HARDWARE_RUNBOOK.md`**

Record: the chosen exposure/emitter from Task 9; `record_throw_ir.py` then `measure_landing.py`; and the refusal conditions (fewer than 12 inliers, inlier fraction below 0.6, RMS above 1.0 px) with the instruction that a refusal means **re-throw**, never hand-tune thresholds until it passes.

- [ ] **Step 3: Run the real-throw acceptance gate (spec §7.3)**

With the mount up and `T_B_C` re-measured, record throws where the ball does not bounce far. For each, compare `measure_landing.py`'s first-contact point against the static `ball_detector` + `ray_plane.ball_center_on_plane` resting measurement. Record both numbers and their difference for at least 5 throws.

**Expected:** agreement to within a couple of centimetres on low-bounce throws. A systematic offset in a consistent direction indicates a `T_B_C` error, not a fitter error — the two paths share the extrinsic, so a shared bias cancels in neither. Report the numbers; do not adjust anything to make them agree.

- [ ] **Step 4: Update `status_update/HANDOFF.md`**

New session entry with the REAL vs ASSIGNED vs NOT-WORKING table this repo requires. Be explicit that Tasks 1-7 are validated **synthetically only**, and state exactly which parts have touched the real camera.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md HARDWARE_RUNBOOK.md status_update/HANDOFF.md
git commit -m "docs(vision): ball-tracking pipeline, run-day procedure, acceptance gate"
```

---

## Sequencing note

Tasks 1-7 need **no camera and no arm** and can be done anywhere. Tasks 8-10 need the camera but not the mount and not the arm — it can sit on a table while a ball is thrown past it by hand. Only Task 11 Step 3 needs the overhead mount, a re-measured `T_B_C` and the arm.

This ordering is deliberate: it keeps the mount off the critical path, exactly as spec §8 argues.
