# mc-pilot-pybullet-yolo — Study 5: Vision Pipeline (OpenCV + YOLO)

Extends `mc-pilot-pybullet/` with a perception layer so the robot detects the target bin visually before throwing, rather than receiving numeric coordinates.

Two detection approaches:

| Method | How it works |
|--------|-------------|
| **OpenCV** | RGB-D camera → HSV colour threshold → contour centroid → depth back-projection to 3D |
| **YOLOv8** | Synthetic PyBullet renders → fine-tuned YOLOv8n → bounding box centroid + depth → 3D |

Both feed detected `(x, y)` into the existing trained policy. The bin is a hollow 12-wall ring so the ball physically enters it.

Demo video: `../yolo (1).mp4` (project root).

---

## New components (vs mc-pilot-pybullet/)

| File | What it does |
|------|-------------|
| `robot_arm/depth_camera.py` | `DepthCamera` — PyBullet RGB-D image, HSV segmentation, contour detection, bias correction |
| `generate_yolo_target_dataset.py` | Render 1000 synthetic images in PyBullet (800 train / 200 val) |
| `train_yolo_target_detector.py` | Fine-tune YOLOv8n on rendered dataset |
| `demo_pybullet_gui.py` | Full GUI demo: camera → detect → throw → land |

---

## Entry points

```bash
# Step 1 — train the throwing policy (required before demo):
python test_mc_pilot_pb_A.py -seed 1 -num_trials 10

# Step 2 — GUI demo with OpenCV vision (depth camera + colour segmentation):
python demo_pybullet_gui.py --log_path results_mc_pilot_pb_A/1 --num_throws 5

# (Optional) Re-train YOLO detector on fresh synthetic data:
python generate_yolo_target_dataset.py
python train_yolo_target_detector.py

# GUI demo with YOLO detection:
python demo_pybullet_gui.py --log_path results_mc_pilot_pb_A/1 --num_throws 5 --vision yolo
```

---

## File guide

- `test_mc_pilot_pb_A.py` — **train the policy** (identical to mc-pilot-pybullet/ version)
- `demo_pybullet_gui.py` — **visual demo** with vision
- `generate_yolo_target_dataset.py`, `train_yolo_target_detector.py` — YOLO pipeline
- `standalone_throw.py` — single throw without full training loop (quick sanity check)
- `test_mc_pilot_pb_A_noisy.py`, `test_mc_pilot_pb_B.py`, `test_mc_pilot_pb_C.py` — earlier variants carried over (ignore)
- `apply_mcpilco_policy.py`, `log_plot_cartpole.py`, `test_mcpilco_cartpole*.py` — upstream MC-PILCO boilerplate (ignore)
