"""
Train a YOLO detector for the PyBullet throwing target.

This is a thin local wrapper around Ultralytics so the detector training stays
inside mc-pilot-pybullet and can be paired with the synthetic dataset generator.

Examples:
  python train_yolo_target_detector.py ^
      --dataset_dir datasets/throw_target_yolo ^
      --base_model yolov8n.pt ^
      --epochs 50

  python train_yolo_target_detector.py ^
      --dataset_dir datasets/throw_target_yolo ^
      --base_model C:/models/yolov8n.pt ^
      --epochs 50
"""

import argparse
import os
import subprocess
import sys


def prepare_ultralytics_env():
    config_dir = os.path.abspath(".ultralytics")
    os.makedirs(config_dir, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", config_dir)
    return config_dir


def build_arg_parser():
    parser = argparse.ArgumentParser("Train a YOLO detector for the PyBullet target")
    parser.add_argument("--dataset_dir", type=str, required=True, help="dataset root containing data.yaml")
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="base YOLO model name or local weights path, e.g. yolov8n.pt or C:/models/yolov8n.pt",
    )
    parser.add_argument("--epochs", type=int, default=50, help="number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="training image size")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--device", type=str, default="cpu", help="training device, e.g. cpu or 0")
    parser.add_argument("--workers", type=int, default=0, help="number of data loader workers")
    parser.add_argument("--project", type=str, default="runs_yolo_target", help="Ultralytics project directory")
    parser.add_argument("--name", type=str, default="throw_target_detector", help="Ultralytics run name")
    parser.add_argument("--patience", type=int, default=20, help="early stopping patience")
    parser.add_argument("--cache", action="store_true", help="cache dataset in memory/disk if supported")
    parser.add_argument("--val", action="store_true", help="run validation after training")
    parser.add_argument("--save_json", action="store_true", help="save COCO-style validation JSON when validating")
    return parser


def import_yolo_or_raise():
    prepare_ultralytics_env()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        pip_cmd = f'"{sys.executable}" -m pip install ultralytics'
        site_hint = ""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", "ultralytics"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                site_hint = (
                    "\n`pip show ultralytics` returned package metadata for this interpreter, "
                    "so the import may be failing because of a broken install."
                )
        except Exception:
            site_hint = ""

        raise ImportError(
            "ultralytics is required for detector training, but it could not be imported by the "
            "current Python interpreter.\n"
            f"Current Python: {sys.executable}\n"
            "Install it into this same interpreter with:\n"
            f"  {pip_cmd}\n"
            "If you already installed ultralytics elsewhere, run this script with that same Python executable."
            f"{site_hint}"
        ) from exc
    return YOLO


def main():
    args = build_arg_parser().parse_args()

    data_yaml = os.path.join(args.dataset_dir, "data.yaml")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(
            f"No data.yaml found at {data_yaml}. Run generate_yolo_target_dataset.py first."
        )

    if os.path.splitext(args.base_model)[1].lower() == ".pt" and os.path.isabs(args.base_model):
        if not os.path.exists(args.base_model):
            raise FileNotFoundError(
                f"Base model weights not found at {args.base_model}. "
                "Provide a valid local weights file or a model name such as yolov8n.pt."
            )

    YOLO = import_yolo_or_raise()

    print(f"Dataset:   {os.path.abspath(args.dataset_dir)}")
    print(f"Data YAML: {os.path.abspath(data_yaml)}")
    print(f"Base:      {args.base_model}")
    print(f"Project:   {args.project}")
    print(f"Run name:  {args.name}")

    model = YOLO(args.base_model)
    train_results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        cache=args.cache,
    )

    print("Training complete.")
    save_dir = getattr(train_results, "save_dir", None)
    if save_dir is not None:
        print(f"  Run dir:  {save_dir}")
        print(f"  Best pt:  {os.path.join(str(save_dir), 'weights', 'best.pt')}")
        print(f"  Last pt:  {os.path.join(str(save_dir), 'weights', 'last.pt')}")

    if args.val:
        val_results = model.val(
            data=data_yaml,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=args.project,
            name=args.name + "_val",
            save_json=args.save_json,
        )
        map50 = getattr(getattr(val_results, "box", None), "map50", None)
        if map50 is not None:
            print(f"Validation mAP50: {map50:.4f}")


if __name__ == "__main__":
    main()
