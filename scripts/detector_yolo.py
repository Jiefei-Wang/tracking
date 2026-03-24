"""
Train a YOLOv8 detector on rat keypoint data.

- Derives bounding boxes from DLC keypoints (min/max of visible points + padding)
- Camera4_stitched is the holdout validation set
- Explicit train/test split for all non-ai videos
- Saves best + periodic checkpoints to output/yolo/<datetime>
"""

import os
import sys
import shutil
import yaml
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from modules.label_csv_utils import load_keypoints
from modules.sam_utils import get_coordinates


# ──────────────────────────────────────────────────────────────
# Load train / test / val split from config.yaml
# ──────────────────────────────────────────────────────────────

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml")

with open(CONFIG_PATH, "r") as f:
    _config = yaml.safe_load(f)

TRAIN_VIDEOS = _config["train_videos"]
TEST_VIDEOS = _config["test_videos"]
VAL_VIDEOS = _config["val_videos"]

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

LABELED_DATA_DIR = "input/labeled-data"
VIDEO_FRAMES_DIR = os.path.join("output", "video_frames_for_sam")
BBOX_PAD_FRAC = 0.05  # pad bounding box by 5% of box size
CLASS_NAME = "rat"
MODEL_NAME = "yolov8m"
IMGSZ = 960
EPOCHS = 200

def keypoints_to_bbox(df, row_idx, img_w, img_h, pad_frac=BBOX_PAD_FRAC):
    """Convert keypoint row to a YOLO-format bounding box [cx, cy, w, h] normalized."""
    coords = get_coordinates(df, row_idx)
    if len(coords) < 2:
        return None

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    bw = x_max - x_min
    bh = y_max - y_min
    pad_x = bw * pad_frac
    pad_y = bh * pad_frac

    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(img_w, x_max + pad_x)
    y_max = min(img_h, y_max + pad_y)

    cx = (x_min + x_max) / 2.0 / img_w
    cy = (y_min + y_max) / 2.0 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return [cx, cy, w, h]


def get_frame_filename(frame_idx, video_frames_dir):
    """Find the jpg filename for a given frame index in the video frames dir."""
    candidates = [
        f for f in os.listdir(video_frames_dir)
        if os.path.splitext(f)[-1].lower() in [".jpg", ".jpeg"]
    ]
    idx_to_name = {}
    for c in candidates:
        idx = int(os.path.splitext(c)[0])
        idx_to_name[idx] = c
    return idx_to_name.get(frame_idx)


def prepare_yolo_dataset(output_dir):
    """
    Create YOLO dataset directory structure:
      output_dir/
        images/train/  images/val/  images/test/
        labels/train/  labels/val/  labels/test/
        data.yaml
    """
    splits = {"train": TRAIN_VIDEOS, "val": VAL_VIDEOS, "test": TEST_VIDEOS}

    # Clean output directory before writing
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    for split_name in splits:
        os.makedirs(os.path.join(output_dir, "images", split_name), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split_name), exist_ok=True)

    total_images = 0

    for split_name, video_list in splits.items():
        for video_name in video_list:
            csv_path = os.path.join(LABELED_DATA_DIR, video_name, "CollectedData_rats.csv")
            frames_dir = os.path.join(VIDEO_FRAMES_DIR, video_name)

            if not os.path.isfile(csv_path):
                print(f"  Warning: no CSV for {video_name}, skipping")
                continue
            if not os.path.isdir(frames_dir):
                print(f"  Warning: no frames for {video_name}, skipping")
                continue

            label_df = load_keypoints(csv_path)
            if label_df.empty:
                continue

            # Build frame index -> filename mapping
            all_frames = [
                f for f in os.listdir(frames_dir)
                if os.path.splitext(f)[-1].lower() in [".jpg", ".jpeg"]
            ]
            idx_to_name = {int(os.path.splitext(f)[0]): f for f in all_frames}

            # Get image dimensions from first frame
            sample_name = all_frames[0]
            sample_img = Image.open(os.path.join(frames_dir, sample_name))
            img_w, img_h = sample_img.size

            count = 0
            for row_idx in range(len(label_df)):
                frame_idx = int(label_df["frame"].iloc[row_idx])
                frame_name = idx_to_name.get(frame_idx)
                if frame_name is None:
                    continue

                bbox = keypoints_to_bbox(label_df, row_idx, img_w, img_h)
                if bbox is None:
                    continue

                # Unique filename: videoname_frameXXX.jpg
                safe_video = video_name.replace(" ", "_")
                out_basename = f"{safe_video}_frame{frame_idx:06d}"

                # Copy image
                src_img = os.path.join(frames_dir, frame_name)
                dst_img = os.path.join(output_dir, "images", split_name, out_basename + ".jpg")
                shutil.copy2(src_img, dst_img)

                # Write YOLO label (class 0, cx, cy, w, h)
                dst_lbl = os.path.join(output_dir, "labels", split_name, out_basename + ".txt")
                with open(dst_lbl, "w") as f:
                    f.write(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

                count += 1

            total_images += count
            print(f"  {split_name:5s} | {video_name:35s} | {count} images")

    # Write data.yaml
    data_yaml = {
        "path": os.path.abspath(output_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: CLASS_NAME},
    }
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"\nTotal images prepared: {total_images}")
    print(f"Dataset YAML: {yaml_path}")
    return yaml_path


def train_yolo(data_yaml_path, run_dir, epochs=100, imgsz=640, batch=16):
    """Train YOLOv8 detector with augmentation and checkpoint saving."""
    import torch
    from ultralytics import YOLO

    # Workaround for CUDA pinned memory allocation errors
    torch.cuda.empty_cache()

    # Start from pretrained model (downloaded into input/yolo/ if needed)
    model_dir = os.path.join("input", "yolo")
    os.makedirs(model_dir, exist_ok=True)
    model = YOLO(os.path.join(model_dir, f"{MODEL_NAME}.pt"))

    results = model.train(
        data=os.path.abspath(data_yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=os.path.abspath(run_dir),
        name="train",
        exist_ok=True,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # Checkpointing
        save=True,
        save_period=10,  # save every 10 epochs
        patience=30,     # early stopping patience
        # Other
        workers=1,
        device=0,
        verbose=True,
    )

    return results


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("output", "yolo", f"{MODEL_NAME}_{IMGSZ}_{timestamp}")
    dataset_dir = os.path.join(run_dir, "dataset")

    os.makedirs(run_dir, exist_ok=True)

    print("=" * 60)
    print(f"YOLO Detector Training")
    print(f"Run directory: {run_dir}")
    print(f"Timestamp: {timestamp}")
    print("=" * 60)

    # Print split info
    print(f"\nTrain videos ({len(TRAIN_VIDEOS)}):")
    for v in TRAIN_VIDEOS:
        print(f"  - {v}")
    print(f"\nTest videos ({len(TEST_VIDEOS)}):")
    for v in TEST_VIDEOS:
        print(f"  - {v}")
    print(f"\nValidation (holdout) videos ({len(VAL_VIDEOS)}):")
    for v in VAL_VIDEOS:
        print(f"  - {v}")

    # Step 1: Prepare dataset
    print("\n" + "=" * 60)
    print("Step 1: Preparing YOLO dataset")
    print("=" * 60)
    data_yaml_path = prepare_yolo_dataset(dataset_dir)

    # Step 2: Train
    print("\n" + "=" * 60)
    print("Step 2: Training YOLOv8")
    print("=" * 60)
    results = train_yolo(data_yaml_path, run_dir, epochs=EPOCHS, imgsz=IMGSZ, batch=16)

    # Step 3: Evaluate best model on test set
    print("\n" + "=" * 60)
    print("Step 3: Evaluating best model on test set")
    print("=" * 60)
    from ultralytics import YOLO
    best_model_path = os.path.join(run_dir, "train", "weights", "best.pt")
    best_model = YOLO(best_model_path)
    test_results = best_model.val(
        data=os.path.abspath(data_yaml_path),
        split="test",
        imgsz=IMGSZ,
        batch=16,
        device=0,
        project=os.path.abspath(run_dir),
        name="test_eval",
        exist_ok=True,
    )

    print("\n" + "=" * 60)
    print(f"Training complete! Results saved to: {run_dir}")
    print(f"Best model: {best_model_path}")
    print(f"Test mAP50:    {test_results.box.map50:.4f}")
    print(f"Test mAP50-95: {test_results.box.map:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()