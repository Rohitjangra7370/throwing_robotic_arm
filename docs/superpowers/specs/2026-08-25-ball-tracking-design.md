# Ball tracking and landing-point measurement — design

_2026-08-25. Status: design approved, not yet implemented._

## Goal

Measure **where a thrown tennis ball first contacts the floor**, offline, to a stated
uncertainty, in the arm's base frame — so that landing error can eventually be fed back
into the GP ("close the loop" in `HARDWARE_SETUP.md`).

Explicitly **not** in scope here: real-time output, catching or deflecting, target-bin
detection, the GP feedback wiring itself, and ROS.

## Why this replaces the existing static measurement

`perception/ball_detector.py` finds the ball **after it settles**. On a tile floor a ball
thrown at ~1.6 m/s bounces and rolls, so the resting position is not the first-contact
point, and it is first contact that the policy's landing error is defined against. The
gap is a systematic, not noise, and nothing in the current pipeline bounds it.

Tracking the ball through flight and solving for the floor crossing measures first contact
directly. The static path is kept — as an independent cross-check (§7.3), not as the primary.

## Hardware facts this design rests on

Measured off the lab unit (D435i, SN 349522070924, fw 5.17.3.10) on 2026-08-25, not quoted
from a datasheet:

| Quantity | Value |
|---|---|
| IR streams | `infrared,1` and `infrared,2`, 848x480 y8 @ 90 fps, both available simultaneously |
| IR intrinsics (848x480) | `fx = fy = 426.167`, `ppx = 420.286`, `ppy = 238.505` |
| Distortion | all five Brown-Conrady coefficients **0.0** (factory-rectified; undistortion is a no-op) |
| IR1 -> IR2 extrinsic | `t = (-49.9448, 0, 0) mm`, rotation **exactly identity** |
| IR field of view | **89.7 deg x 58.8 deg** |
| Exposure range | 1 - 165000 us (currently 8500) |
| Emitter | controllable |

Two consequences matter:

1. **The pair is perfectly rectified.** Disparity is purely horizontal, so triangulation is
   `Z = fx * B / d` with no rectification step and no matching algorithm.
2. **The IR FOV is much wider than the colour FOV** (89.7 x 58.8 vs 70.2 x 43.2). This is what
   makes the fixed overhead mount viable: computed against the *colour* FOV the release is out
   of frame and only the last stretch of descent is visible, but against the real IR FOV the
   ball enters frame at t ~ 0.09 s and lands at t = 0.580 s — **~44 frames of flight in view at
   90 fps**.

Both IR imagers are global shutter (OV9282); the colour imager (OV2740) is rolling shutter.
IR is therefore the correct stream for a fast ball regardless of the FOV argument.

## Why not depth, and why this is not the thing the repo already rejected

`perception/ray_plane.py`, `HARDWARE_SETUP.md` and `HANDOFF.md` item 5 all reject
**RealSense's block-matching depth map** as a position source: ~2% of range = 2-4 cm at
1-2 m, the same size as the landing error being measured. That objection stands and is not
being reversed. It is also worse for this specific target than the 2% figure suggests — a
small, round, textureless sphere in mid-air with no background support returns 0.0 as often
as it returns a noisy value.

Triangulating two **independently detected, high-contrast blob centroids** is a different
operation with a different error model. It shares only the word "stereo" with the rejected
method. The depth stream may still be enabled as a coarse segmentation gate, never as the
position.

## Error budget

| Term | Value |
|---|---|
| Centroid precision, ~15 px disc, subpixel | 0.15 px |
| Disparity (two independent centroids) | 0.21 px |
| Per-frame range sigma at Z = 2.0 m | 4.0 cm |
| Per-frame lateral sigma at Z = 2.0 m | 0.70 mm |
| 44 frames -> 6-parameter fit, sqrt(44/6) = 2.71 | range 1.5 cm, lateral 0.26 mm |
| Impact-time error 1.5 cm / 5.55 m/s = 2.7 ms, x v_x = 1.62 m/s | 4.4 mm |
| **Total landing sigma** | **~4.4 mm** |

Against the 2.89 cm sim accuracy this is ~6x headroom. The overhead mount is favourable
because the landing (x,y) being measured lies in the camera's **lateral** direction (0.70 mm
per frame) while the poorly determined **range** direction is height, which only perturbs the
impact *time*. This is the same conclusion `plan_camera_mount.py` reaches via ray-plane
incidence, arrived at independently.

**The dominant error is therefore `T_B_C`, not the vision.** See §8.

## Ball and flight model

Tennis ball, 57.7 g, 65.4 mm diameter, `ball_radius = 0.0327` — fixed by the trained
checkpoints (all three `results_kinetic_chain_gen3/*/config_log.pkl`) and by
`HARDWARE_RUNBOOK.md`. Drag force at 2 m/s is `0.5 * 1.2 * 0.5 * 3.36e-3 * v^2 = 4.0e-3 N`
against `0.566 N` of weight — **0.7%** — so a drag-free parabola is valid.

This does **not** generalise to the whiffle ball (`0.004 kg / 0.06 m`) of the drag-crossover
study. A documented hook for a linear-drag term is left in `trajectory.py`; it is not
implemented.

## 1. Module boundaries

Pure geometry and math live in `perception/` and are unit-testable with no camera; hardware
is a thin shell around them. This follows the existing split (`ray_plane.py` is pure,
`demo_ball_detector.py` is the shell).

| File | Status | Responsibility | Depends on |
|---|---|---|---|
| `perception/ir_capture.py` | new | Dual-IR recorder: configure streams, ring-buffer a throw window, dump to disk. No detection, no math. | `pyrealsense2` |
| `perception/ball_track.py` | new | One image + background -> ball candidates with subpixel centroid, radius, area. ndarray in, list out. | `cv2`, `numpy` |
| `perception/stereo.py` | new | Rectified IR1/IR2 candidate pairs -> camera-frame 3D points. | `numpy` |
| `perception/trajectory.py` | new | RANSAC ballistic association, Gauss-Newton parabola fit, impact solve. | `numpy` |
| `perception/ball_detector.py` | exists | Static settle-and-measure. Reused as the independent cross-check. | — |
| `perception/ray_plane.py` | exists | Reused for the cross-check and for the ball-radius convention. | — |
| `record_throw_ir.py` | new CLI | Capture a throw window to disk. | `ir_capture` |
| `measure_landing.py` | new CLI | Offline: recording -> first-contact (x,y) in base frame. | all of the above |

Each unit answers "what does it do / how is it used / what does it depend on" without
reading its internals, and `stereo.py` + `trajectory.py` are testable against closed-form
truth with no image and no camera.

## 2. Data flow: record then fit

`record_throw_ir.py` ring-buffers into RAM and dumps after the throw. Nothing downstream
touches the camera.

- 848x480 y8 = 407 kB/image, x2 cameras x 90 fps = **73 MB/s**; a 2 s window is 146 MB.
  Trivial in RAM, and buffering avoids a disk stall dropping frames mid-flight.
- Every recording becomes a permanent regression fixture. Re-running a changed fitter
  against a real throw from last week is worth more than the disk space, in a project with
  this much history of plausible-looking wrong numbers.
- Timestamps come from the **sensor** timestamp, offset by **exposure/2** to mid-exposure —
  not frame-arrival time. At 5.8 m/s an 8.5 ms exposure is 5 cm of travel; a wrong time
  reference is a systematic, not noise.

Pipeline:

```
dual IR frames
  -> per-image candidates            (ball_track)
  -> L/R pairing, same row           (stereo)
  -> triangulate                     (stereo)
  -> RANSAC ballistic association    (trajectory)
  -> Gauss-Newton fit                (trajectory)
  -> impact solve                    (trajectory)
  -> (x, y) in base frame            (T_B_C)
```

## 3. Detection

Frame-difference against a rolling-median background. The camera is static and the ball is
a ~15 px high-contrast blob; this runs in ~1 ms on CPU.

YOLO is **not** the primary detector. A 15 px object sits at the floor of YOLOv8's stride-8
P3 head, the classical method is both faster and better conditioned here, and the GPU is
currently unusable anyway (`torch.cuda.is_available()` is False on the RTX 4070 with
`CUDA unknown error`; `tensorrt` is not installed). A YOLO backend behind the same interface
is permitted as a fallback but is not on the critical path.

**The arm is the main false-positive source**, moving fast in exactly the region where the
ball starts. This is *not* handled with a hand-drawn ROI mask. The RANSAC ballistic
association rejects arm pixels for free — they do not fit a parabola with g = 9.81 — and the
same mechanism handles reflections and the second bounce.

Centroid estimate is intensity-weighted over the diff blob. For a symmetric motion blur this
is unbiased at mid-exposure, which is why §2 defines the timestamp that way.

## 4. Stereo pairing and triangulation

The pair is rectified with identity rotation, so a correct L/R match lies on the same image
row. Pair by row agreement (with a tolerance covering blur) plus area agreement. With a
single ball in flight this is unambiguous; the RANSAC stage in §5 is the backstop if it is
not.

`Z = fx * B / d`, then `X = (u - ppx) * Z / fx`, `Y = (v - ppy) * Z / fy`. No undistortion
(coefficients are exactly zero at this resolution — re-check if the stream resolution
changes, per `ray_plane.py`'s standing warning).

## 5. Trajectory fit and impact solve

Unknowns: `p0` (3) and `v0` (3), in base frame. Model `p(t) = p0 + v0*t + 0.5*g*t^2` with
`g = (0, 0, -9.81)`.

**The objective is reprojection error in pixels, in both cameras — not error against the
triangulated 3D points.** Pixel noise is the iid quantity; triangulated points carry
correlated, range-dependent noise, so least-squares over them applies the wrong weighting.
Gauss-Newton, initialised by linear least squares on the triangulated points.

Impact is the **descending** root of `z(t) = z_floor + ball_radius`, with `z_floor = -0.433`
in base frame (base at 0, floor at `-base_height`). This matches
`ray_plane.ball_center_on_plane` exactly: a resting sphere's centre is one radius above the
plane and the centre's (x,y) *is* the contact (x,y). Both paths must use one convention or
the cross-check in §7.3 measures nothing.

The fit reports its covariance. **A landing point without a stated sigma is not eligible to
become a GP datapoint.**

## 6. Failure handling

Fail loudly; never fabricate a plausible number. This follows `detect_ball_bgsub`, which
already raises rather than returning an invented centroid.

Each of these raises, with a diagnostic identifying which one fired:

- fewer than **12** usable frames on the track (against ~44 expected; 12 leaves 2x redundancy over the 6 unknowns)
- RANSAC inlier fraction below **0.6**
- RMS reprojection residual above **1.0 px** (vs the 0.15 px centroid precision assumed in the error budget)
- impact solve with no descending real root

Every stage returns a diagnostics dict in the style of `detect_ball_bgsub`'s
`mask_nonzero_frac` / `area_px` / `n_candidates`, so a bad run is debuggable from the log
rather than by re-throwing.

## 7. Testing

1. **Synthetic end-to-end, no hardware.** Generate a known parabola, project it into both
   cameras through the *real* measured intrinsics and 49.9448 mm baseline, add pixel noise,
   assert the recovered landing point is within tolerance of truth. This is the TDD entry
   point and can be written before the mount exists.
2. **Unit tests per module** — triangulation against hand-computed disparities, impact solve
   against closed-form roots, RANSAC against a track with injected arm-like outliers.
3. **Acceptance gate on real throws.** For throws where the ball does not bounce far, the
   trajectory-predicted first contact must agree with the static `ball_detector` +
   `ray_plane` resting measurement. Two independent paths, one already trusted.

New pytest files: `tests/test_ball_track.py`, `tests/test_stereo.py`,
`tests/test_trajectory.py`, alongside the existing `tests/test_ball_detector.py` and
`tests/test_ray_plane.py`.

## 8. Prerequisite, not blocker: `T_B_C`

The current extrinsic (2026-08-22) belongs to the **laptop-held rig, explicitly not the final
mount**, and its board detection was marginal throughout (16-19 px marker edges against a
35 px "robust" bar). It must be re-measured once the overhead mount is bolted up, via
`calibrate_via_wrist_camera.py` (the adopted FK + shared-marker path).

This does not block implementation. Everything in §1-§7 is testable with the camera on a
table, reporting in **camera frame**; the base-frame transform is the last step and drops in
when the mount is up. Sequencing the work this way means the mount is not on the critical
path.

## 9. Deliberately unresolved, to settle empirically

Two questions that argument will not settle and a ten-minute A/B will:

- **Emitter on or off.** The projector's dot pattern is static on the floor, so background
  subtraction removes it, and on the ball it adds useful texture — but it may saturate.
- **Exposure.** 2 ms gives 2.2 px of blur at the 5.8 m/s impact speed, but may be too dark
  indoors with the emitter off.

## 10. Relationship to the originally proposed architecture

The original proposal was: IR + depth at 90 fps -> TensorRT YOLO -> `rs2_deproject_pixel_to_point`
-> `robot_localization` EKF -> quadratic impact solve.

Kept: the IR-at-90 fps decision (correct, and for the right reason — global shutter), and the
quadratic impact solve.

Changed:

- **Depth deprojection -> stereo triangulation.** Depth-as-position is already rejected in
  this repo with numbers, and block matching fails on this particular target.
- **ROS2 `robot_localization` EKF -> batch least squares.** The repo has zero ROS; the EKF is
  an odometry filter with no gravity process model; and offline, with every frame available at
  once, batch least squares is the optimal estimator and needs no tuning.
- **TensorRT YOLO -> classical frame differencing**, with YOLO retained as an optional backend.
