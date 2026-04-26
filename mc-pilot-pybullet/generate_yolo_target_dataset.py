"""
Generate a synthetic YOLO detection dataset for the PyBullet throwing target.

The generated target matches the current vision test script: a visible red
sphere placed on the ground plane at the sampled throwing target location.

Example:
  python generate_yolo_target_dataset.py ^
      --output_dir datasets/throw_target_yolo ^
      --num_train 800 ^
      --num_val 200 ^
      --class_name throw_target
"""

import argparse
import json
import os
import time

import numpy as np
import pybullet as p
import pybullet_data


def build_arg_parser():
    parser = argparse.ArgumentParser("Generate synthetic YOLO data for the PyBullet throw target")
    parser.add_argument("--output_dir", type=str, required=True, help="dataset output directory")
    parser.add_argument("--num_train", type=int, default=800, help="number of training images")
    parser.add_argument("--num_val", type=int, default=200, help="number of validation images")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--class_name", type=str, default="throw_target", help="YOLO class name")
    parser.add_argument("--camera_width", type=int, default=640, help="image width")
    parser.add_argument("--camera_height", type=int, default=480, help="image height")
    parser.add_argument("--camera_fov", type=float, default=60.0, help="vertical FOV in degrees")
    parser.add_argument("--camera_near", type=float, default=0.05, help="near clipping plane")
    parser.add_argument("--camera_far", type=float, default=4.0, help="far clipping plane")
    parser.add_argument(
        "--camera_eye",
        type=float,
        nargs=3,
        default=[1.20, 0.00, 1.20],
        metavar=("X", "Y", "Z"),
        help="nominal camera eye position",
    )
    parser.add_argument(
        "--camera_target",
        type=float,
        nargs=3,
        default=[0.85, 0.00, 0.05],
        metavar=("X", "Y", "Z"),
        help="nominal camera look-at target",
    )
    parser.add_argument(
        "--camera_up",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 1.0],
        metavar=("X", "Y", "Z"),
        help="camera up vector",
    )
    parser.add_argument("--camera_eye_jitter", type=float, default=0.08, help="uniform xyz jitter on camera eye")
    parser.add_argument(
        "--camera_target_jitter",
        type=float,
        default=0.08,
        help="uniform xyz jitter on camera look-at target",
    )
    parser.add_argument("--lm", type=float, default=0.6, help="minimum target distance")
    parser.add_argument("--lM", type=float, default=1.1, help="maximum target distance")
    parser.add_argument("--gM", type=float, default=float(np.pi / 6), help="maximum target azimuth in radians")
    parser.add_argument("--target_radius", type=float, default=0.06, help="target sphere radius in meters")
    parser.add_argument(
        "--target_rgba",
        type=float,
        nargs=4,
        default=[1.0, 0.0, 0.0, 0.85],
        metavar=("R", "G", "B", "A"),
        help="target sphere color",
    )
    parser.add_argument("--max_distractors", type=int, default=3, help="max random distractor objects per image")
    parser.add_argument("--min_bbox_pixels", type=float, default=10.0, help="discard boxes smaller than this")
    parser.add_argument("--save_preview_every", type=int, default=0, help="save preview PNG every N images; 0 disables")
    return parser


def ensure_pillow():
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ImportError("Pillow is required to save dataset images. Install pillow locally.") from exc
    return Image, ImageDraw


def sample_target(rng, lm, lM, gM):
    dist = rng.uniform(lm, lM)
    angle = rng.uniform(-gM, gM)
    return np.array([dist * np.cos(angle), dist * np.sin(angle), 0.0], dtype=float)


def randomize_camera(args, rng):
    eye = np.asarray(args.camera_eye, dtype=float) + rng.uniform(
        -args.camera_eye_jitter, args.camera_eye_jitter, size=3
    )
    target = np.asarray(args.camera_target, dtype=float) + rng.uniform(
        -args.camera_target_jitter, args.camera_target_jitter, size=3
    )
    up = np.asarray(args.camera_up, dtype=float)
    return eye, target, up


def build_camera_config(width, height, fov_deg, near, far, eye, target, up):
    aspect = float(width) / float(height)
    view_matrix_flat = p.computeViewMatrix(
        cameraEyePosition=eye.tolist(),
        cameraTargetPosition=target.tolist(),
        cameraUpVector=up.tolist(),
    )
    projection_matrix_flat = p.computeProjectionMatrixFOV(
        fov=float(fov_deg),
        aspect=aspect,
        nearVal=float(near),
        farVal=float(far),
    )
    view_matrix = np.array(view_matrix_flat, dtype=float).reshape(4, 4, order="F")
    projection_matrix = np.array(projection_matrix_flat, dtype=float).reshape(4, 4, order="F")
    return {
        "width": int(width),
        "height": int(height),
        "fov_deg": float(fov_deg),
        "near": float(near),
        "far": float(far),
        "eye": eye,
        "target": target,
        "up": up,
        "view_matrix": view_matrix,
        "projection_matrix": projection_matrix,
        "view_matrix_flat": np.asarray(view_matrix_flat, dtype=float),
        "projection_matrix_flat": np.asarray(projection_matrix_flat, dtype=float),
    }


def render_rgb(client_id, camera_cfg):
    _, _, rgba, _, _ = p.getCameraImage(
        width=camera_cfg["width"],
        height=camera_cfg["height"],
        viewMatrix=camera_cfg["view_matrix_flat"].tolist(),
        projectionMatrix=camera_cfg["projection_matrix_flat"].tolist(),
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=client_id,
    )
    rgba = np.reshape(rgba, (camera_cfg["height"], camera_cfg["width"], 4))
    return rgba[:, :, :3].astype(np.uint8)


def project_world_points(points_world, camera_cfg):
    pixels = []
    pv = camera_cfg["projection_matrix"] @ camera_cfg["view_matrix"]
    for point in points_world:
        world_h = np.array([point[0], point[1], point[2], 1.0], dtype=float)
        clip = pv @ world_h
        if abs(clip[3]) < 1e-9:
            continue
        ndc = clip[:3] / clip[3]
        if not np.all(np.isfinite(ndc)):
            continue
        u = ((ndc[0] + 1.0) * 0.5) * camera_cfg["width"]
        v = ((1.0 - ndc[1]) * 0.5) * camera_cfg["height"]
        pixels.append([u, v, ndc[2]])
    return np.asarray(pixels, dtype=float)


def sphere_bbox_pixels(center_world, radius, camera_cfg):
    dirs = []
    for sx in (-1.0, 0.0, 1.0):
        for sy in (-1.0, 0.0, 1.0):
            for sz in (-1.0, 0.0, 1.0):
                vec = np.array([sx, sy, sz], dtype=float)
                norm = np.linalg.norm(vec)
                if norm > 0.0:
                    dirs.append(vec / norm)
    dirs.extend(
        [
            np.array([1.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, -1.0]),
        ]
    )
    samples = [center_world + radius * direction for direction in dirs]
    pixels = project_world_points(samples, camera_cfg)
    if pixels.size == 0:
        return None
    xs = pixels[:, 0]
    ys = pixels[:, 1]
    x1 = float(np.clip(xs.min(), 0.0, camera_cfg["width"] - 1.0))
    x2 = float(np.clip(xs.max(), 0.0, camera_cfg["width"] - 1.0))
    y1 = float(np.clip(ys.min(), 0.0, camera_cfg["height"] - 1.0))
    y2 = float(np.clip(ys.max(), 0.0, camera_cfg["height"] - 1.0))
    return x1, y1, x2, y2


def bbox_to_yolo(x1, y1, x2, y2, width, height):
    xc = ((x1 + x2) * 0.5) / width
    yc = ((y1 + y2) * 0.5) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return xc, yc, bw, bh


def create_target(client_id, target_xyz, radius, rgba):
    visual = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=list(rgba),
        physicsClientId=client_id,
    )
    body = p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=visual,
        basePosition=[float(target_xyz[0]), float(target_xyz[1]), float(radius)],
        physicsClientId=client_id,
    )
    return body


def add_random_distractors(client_id, rng, count):
    body_ids = []
    for _ in range(count):
        shape_kind = rng.choice(["box", "cylinder", "sphere"])
        x = rng.uniform(0.4, 1.3)
        y = rng.uniform(-0.6, 0.6)
        scale = rng.uniform(0.03, 0.09)
        rgba = [float(c) for c in rng.uniform(0.0, 1.0, size=3)] + [0.9]

        if shape_kind == "box":
            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[scale, scale, scale],
                rgbaColor=rgba,
                physicsClientId=client_id,
            )
            body = p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=visual,
                basePosition=[x, y, scale],
                physicsClientId=client_id,
            )
        elif shape_kind == "cylinder":
            visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=scale,
                length=2.0 * scale,
                rgbaColor=rgba,
                physicsClientId=client_id,
            )
            body = p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=visual,
                basePosition=[x, y, scale],
                physicsClientId=client_id,
            )
        else:
            visual = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=scale,
                rgbaColor=rgba,
                physicsClientId=client_id,
            )
            body = p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=visual,
                basePosition=[x, y, scale],
                physicsClientId=client_id,
            )
        body_ids.append(body)
    return body_ids


def ensure_dirs(base_dir):
    for split in ("train", "val"):
        os.makedirs(os.path.join(base_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "labels", split), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "preview", split), exist_ok=True)


def save_preview(preview_path, image_rgb, bbox_xyxy):
    Image, ImageDraw = ensure_pillow()
    image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = bbox_xyxy
    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
    image.save(preview_path)


def write_data_yaml(output_dir, class_name):
    yaml_text = (
        f"path: {os.path.abspath(output_dir)}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "names:\n"
        f"  0: {class_name}\n"
    )
    with open(os.path.join(output_dir, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml_text)


def main():
    args = build_arg_parser().parse_args()
    rng = np.random.default_rng(args.seed)
    ensure_dirs(args.output_dir)
    write_data_yaml(args.output_dir, args.class_name)

    client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    plane_id = p.loadURDF("plane.urdf", physicsClientId=client)
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.85, 0.85, 0.85, 1.0], physicsClientId=client)

    manifest = []
    try:
        split_counts = {"train": int(args.num_train), "val": int(args.num_val)}
        index_global = 0
        start_time = time.time()
        for split, count in split_counts.items():
            for index in range(count):
                index_global += 1
                target_ground = sample_target(rng, args.lm, args.lM, args.gM)
                eye, target, up = randomize_camera(args, rng)
                camera_cfg = build_camera_config(
                    width=args.camera_width,
                    height=args.camera_height,
                    fov_deg=args.camera_fov,
                    near=args.camera_near,
                    far=args.camera_far,
                    eye=eye,
                    target=target,
                    up=up,
                )

                target_body = create_target(
                    client_id=client,
                    target_xyz=target_ground,
                    radius=args.target_radius,
                    rgba=args.target_rgba,
                )
                distractor_bodies = add_random_distractors(client, rng, rng.integers(0, args.max_distractors + 1))

                image_rgb = render_rgb(client, camera_cfg)
                bbox = sphere_bbox_pixels(
                    center_world=np.array([target_ground[0], target_ground[1], args.target_radius], dtype=float),
                    radius=args.target_radius,
                    camera_cfg=camera_cfg,
                )
                if bbox is None:
                    p.removeBody(target_body, physicsClientId=client)
                    for body_id in distractor_bodies:
                        p.removeBody(body_id, physicsClientId=client)
                    continue

                x1, y1, x2, y2 = bbox
                if (x2 - x1) < args.min_bbox_pixels or (y2 - y1) < args.min_bbox_pixels:
                    p.removeBody(target_body, physicsClientId=client)
                    for body_id in distractor_bodies:
                        p.removeBody(body_id, physicsClientId=client)
                    continue

                xc, yc, bw, bh = bbox_to_yolo(x1, y1, x2, y2, args.camera_width, args.camera_height)
                if not (0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                    p.removeBody(target_body, physicsClientId=client)
                    for body_id in distractor_bodies:
                        p.removeBody(body_id, physicsClientId=client)
                    continue

                stem = f"{split}_{index + 1:05d}"
                image_path = os.path.join(args.output_dir, "images", split, stem + ".png")
                label_path = os.path.join(args.output_dir, "labels", split, stem + ".txt")
                Image, _ = ensure_pillow()
                Image.fromarray(image_rgb).save(image_path)
                with open(label_path, "w", encoding="utf-8") as f:
                    f.write(f"0 {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}\n")

                if args.save_preview_every > 0 and ((index + 1) % args.save_preview_every == 0):
                    preview_path = os.path.join(args.output_dir, "preview", split, stem + ".png")
                    save_preview(preview_path, image_rgb, bbox)

                manifest.append(
                    {
                        "split": split,
                        "stem": stem,
                        "image_path": image_path,
                        "label_path": label_path,
                        "target_ground_xyz": target_ground.tolist(),
                        "target_center_xyz": [float(target_ground[0]), float(target_ground[1]), float(args.target_radius)],
                        "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                        "camera_eye": eye.tolist(),
                        "camera_target": target.tolist(),
                        "camera_up": up.tolist(),
                    }
                )

                p.removeBody(target_body, physicsClientId=client)
                for body_id in distractor_bodies:
                    p.removeBody(body_id, physicsClientId=client)

                if index_global % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"Generated {index_global} images in {elapsed:.1f}s")

        with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print("Dataset generation complete.")
        print(f"  Output: {args.output_dir}")
        print(f"  Train images requested: {args.num_train}")
        print(f"  Val images requested:   {args.num_val}")
        print(f"  Class name: {args.class_name}")
        print(f"  Dataset YAML: {os.path.join(args.output_dir, 'data.yaml')}")
    finally:
        if p.isConnected(client):
            p.disconnect(client)


if __name__ == "__main__":
    main()
