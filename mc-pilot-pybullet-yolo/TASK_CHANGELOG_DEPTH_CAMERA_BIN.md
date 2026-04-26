# Task Change Log: Depth-Camera Bin Targeting

This document records the code changes made for the request:
- add a depth camera,
- place a bin at the target,
- detect bin coordinates with OpenCV,
- use detected coordinates as the throw target.

It also includes follow-up changes requested during iteration (camera placement, visualization, and hollow bin geometry).

## Files Changed

- `robot_arm/depth_camera.py` (new module, then iteratively refined)
- `demo_pybullet_gui.py` (integration and debugging UX)

## 1) Added New Perception Module

### Change
Created `robot_arm/depth_camera.py` with:
- target-bin spawning helper (`spawn_target_bin`)
- camera class (`ArmMountedDepthCamera`)
- RGB-D capture from PyBullet (`getCameraImage`)
- OpenCV segmentation for the green bin
- depth back-projection from image to world coordinates

### Reasoning
The existing code sampled targets numerically. We needed an explicit perception path so the algorithm can consume visually detected targets instead of ideal sampled values.

## 2) Integrated Vision Into Throw Loop

### Change
Updated `demo_pybullet_gui.py` so each throw can:
- spawn a visible bin at ground-truth target,
- capture camera frame,
- detect bin position,
- feed detected `(x, y)` into policy speed prediction and throw command.

Added CLI controls:
- `--target-source {vision,random}`
- `--save-detections`
- `--show-detections`

### Reasoning
This keeps original behavior available (`random`) while enabling perception-driven targeting (`vision`) without changing the policy architecture.

## 3) Added Failure Handling and Fallback

### Change
If OpenCV fails to detect reliably, code falls back to sampled target and logs the failure.

### Reasoning
Demo should continue running even when a frame is imperfect. This avoids crash/stop behavior during multi-throw runs.

## 4) Added Detection Debug Visualizations

### Change
Added overlays in debug image:
- contour,
- bounding box of detected bin,
- centroid marker,
- projected true-bin marker (in-frame/off-frame),
- numeric text for estimated target.

Added an OpenCV debug window for live inspection.

### Reasoning
You asked to verify camera direction and detection quality. Visual overlays make it obvious whether error comes from camera framing, segmentation, or 3D projection.

## 5) Added Camera Pose Debug in World

### Change
`draw_debug_pose()` now draws:
- camera forward ray,
- up-direction ray,
- camera label in PyBullet world.

### Reasoning
Needed quick spatial sanity-check in 3D scene to verify where the camera is physically mounted and where it points.

## 6) Camera Mount Iterations

### Change History
1. Initial mount followed end-effector (wrist-like behavior).
2. Switched to base mount for stability.
3. Tried arm-link mount for user request to place camera on arm.
4. Final update: moved to **front of base** mount (current), per latest request.

Current mount parameters:
- `local_offset=(0.16, 0.0, 0.62)`
- `workspace_focus=(0.9, 0.0, 0.08)`
- `fov_deg=95.0`

### Reasoning
Each relocation responded to visibility and stability concerns. Final placement follows your explicit instruction: camera on base/front side.

## 7) Corrected Bin-Center Bias in Target Estimate

### Change
Detection computes:
- `surface_target_xy` from depth points on visible bin surface,
- corrected `target_xy` by shifting along camera-to-bin ray:
  `target_xy = surface_target_xy + (2R/pi) * unit_ray`.

### Reasoning
Without correction, estimates were consistently offset toward camera-facing bin surface. This produced systematic targeting bias (~few cm). Center correction reduces that bias.

## 8) Added Explicit Policy-Target Dot

### Change
Added magenta marker for **actual target used by policy**:
- 2D magenta dot + label in debug image (`policy target`)
- 3D magenta sphere in PyBullet at `(target_x, target_y)`.

### Reasoning
You asked to show exactly which point is being used as target. This removes ambiguity between box corners, centroid, surface estimate, and corrected target.

## 9) Changed Bin to Hollow Geometry

### Change
Replaced solid cylinder bin with hollow structure:
- thin circular floor,
- ring of fixed wall segments (`BIN_NUM_WALLS=12`),
- wall and floor thickness parameters.

### Reasoning
A solid cylinder prevented the ball from entering the bin volume. Hollow geometry allows physically plausible “ball goes inside bin” behavior.

## 10) Minor Consistency and UX Updates

### Change
- Updated labels/messages/docs in demo to match mount mode and detection mode.
- Kept color thresholds and bin color aligned for robust segmentation.
- Added/updated debug text for error interpretation (`surface_err`, `vision_err`, `in_view`).

### Reasoning
These reduce confusion while tuning and make logs actionable.

## Current Behavior Summary

- Bin is spawned physically and visually.
- Camera observes scene and detects bin via OpenCV.
- Detected and corrected target coordinates are used for throw command.
- Debug tools show camera aim, true projection, detection box, and final policy target.
- Bin is hollow so ball can enter it.

