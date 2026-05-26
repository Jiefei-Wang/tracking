# This script generates pose predictions for specific frames based on label JSON files, using a trained RTMPose model and its associated detector. It reads the requested frame indices from the label files, runs the detector and pose model on the corresponding images, and saves the predictions in a structured JSON format for later review and manual correction if needed.
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

# Resolve project path
if "__file__" in globals():
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
else:
    PROJECT_ROOT = Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
keypoints_dir = PROJECT_ROOT / "scripts" / "keypoints"
if str(keypoints_dir) not in sys.path:
    sys.path.insert(0, str(keypoints_dir))

from scripts.keypoints.RTMPose import expand_box_xyxy, resolve_device
from modules.detector_ssdlite_model import load_detector
from modules.keypoint_rtmpose_predict_common import (
    decode_keypoints_with_predictor,
    load_model_from_checkpoint_for_inference,
    load_yaml_file,
    simcc_probabilities,
    visibility_probabilities,
)


# Inputs
model_dir = PROJECT_ROOT / "output" / "RTMPose" / "no_weak_20260328_174401"
checkpoint_path = None  # Default to the best checkpoint
label_dir = PROJECT_ROOT / "input" / "labels"
frames_root = PROJECT_ROOT / "output" / "extracted_frames"
output_root = PROJECT_ROOT / "output" / "predicted_frames"
label_names = None  # Example: ["ai1.json"]
batch_size = 8
device = None  # Defaults to the training run's inference/training device.
detector_device = None  # Defaults to the run config detector device.
overwrite = False


def to_repo_path(path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    path = Path(path_like)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def read_label_frames(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return sorted({int(item["frame_idx"]) for item in payload})


def read_existing_predictions(path: Path) -> list[dict]:
    if not path.is_file() or overwrite:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}, got {type(payload).__name__}")
    return payload


def save_predictions(path: Path, predictions: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_payload = sorted(predictions, key=lambda item: (item["video_name"], int(item["frame_idx"])))
    path.write_text(json.dumps(sorted_payload, indent=2, sort_keys=True), encoding="utf-8")


def load_rgb_image(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


if checkpoint_path is None:
    if model_dir is None:
        raise ValueError("Set model_dir or checkpoint_path before running the cell.")
    model_dir = to_repo_path(model_dir)
    checkpoint_path = model_dir / "checkpoint_best.pt"
else:
    checkpoint_path = to_repo_path(checkpoint_path)
    model_dir = checkpoint_path.parent

if not checkpoint_path.is_file():
    raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

run_config_path = model_dir / "run_config.yaml"
project_config_path = model_dir / "project_config.yaml"
if not run_config_path.is_file():
    raise FileNotFoundError(f"Missing run config: {run_config_path}")
if not project_config_path.is_file():
    raise FileNotFoundError(f"Missing project config: {project_config_path}")

run_bundle = yaml.safe_load(run_config_path.read_text(encoding="utf-8")) or {}
run_args = dict(run_bundle.get("args") or {})
run_cfg = dict(run_bundle.get("config") or {})
project_cfg = load_yaml_file(project_config_path)
bodyparts = list(project_cfg["bodyparts"])
model_name = model_dir.name

device_name = str(device or run_args.get("device") or run_cfg.get("inference", {}).get("device") or "cuda:0")
detector_device_name = str(detector_device or run_args.get("detector_device") or device_name)
device_obj = resolve_device(device_name)
detector_device_obj = resolve_device(detector_device_name)

model, _ = load_model_from_checkpoint_for_inference(
    model_path=model_dir,
    checkpoint=checkpoint_path,
    device=device_obj,
)
model_cfg = dict(model.cfg)

detector_args = argparse.Namespace(detector_checkpoint=to_repo_path(run_args.get("detector_checkpoint")))
if detector_args.detector_checkpoint is None or not detector_args.detector_checkpoint.is_file():
    raise FileNotFoundError(f"Missing detector checkpoint: {detector_args.detector_checkpoint}")
detector = load_detector(detector_args.detector_checkpoint.parent, detector_args.detector_checkpoint, device=detector_device_obj)
detector_score_threshold = float(run_args.get("detector_score_threshold", run_cfg.get("detector_score_threshold", 0.5)))

input_w, input_h = [int(v) for v in model_cfg["model"]["heads"]["bodypart"].get("input_size", [256, 256])]
crop_expand_scale = float(run_args.get("crop_expand_scale", run_cfg.get("crop_expand_scale", 0.15)))
image_mean = np.asarray(model_cfg.get("image_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
image_std = np.asarray(model_cfg.get("image_std", [0.229, 0.224, 0.225]), dtype=np.float32)

label_dir = to_repo_path(label_dir)
frames_root = to_repo_path(frames_root)
output_root = (
    to_repo_path(output_root)
    if output_root is not None
    else (PROJECT_ROOT / "output" / "predicted_frames" / model_name)
)
output_root.mkdir(parents=True, exist_ok=True)

label_paths = sorted(label_dir.glob("*.json"))
if label_names:
    selected = {name if name.endswith(".json") else f"{name}.json" for name in label_names}
    label_paths = [path for path in label_paths if path.name in selected]
if not label_paths:
    raise FileNotFoundError(f"No label JSON files found in {label_dir}")

total_new_predictions = 0
for label_path in label_paths:
    video_name = label_path.stem
    frame_paths = {}
    for ext in ("jpg", "jpeg", "png"):
        for image_path in (frames_root / video_name).glob(f"*.{ext}"):
            try:
                frame_paths[int(image_path.stem)] = image_path
            except ValueError:
                continue

    if not frame_paths:
        print(f"Skipping {video_name}: no frames found under {frames_root / video_name}")
        continue

    requested_frames = read_label_frames(label_path)
    output_path = output_root / f"{video_name}.json"
    existing_predictions = read_existing_predictions(output_path)
    existing_keys = {(str(item.get("video_name")), int(item.get("frame_idx"))) for item in existing_predictions}
    pending_frames = [frame_idx for frame_idx in requested_frames if (video_name, frame_idx) not in existing_keys]

    if not pending_frames:
        print(f"Skipping {video_name}: all {len(requested_frames)} frames already predicted.")
        continue

    batch_items = []
    new_predictions = []
    progress = tqdm(pending_frames, desc=video_name, leave=False)
    for frame_idx in progress:
        image_path = frame_paths.get(frame_idx)
        if image_path is None:
            print(f"Missing image for {video_name} frame {frame_idx}: expected under {frames_root / video_name}")
            continue
        image_rgb = load_rgb_image(image_path)
        batch_items.append({
            "video_name": video_name,
            "frame_idx": frame_idx,
            "image_path": image_path,
            "image_rgb": image_rgb,
        })

        should_run = len(batch_items) >= int(batch_size) or frame_idx == pending_frames[-1]
        if not should_run:
            continue

        images_rgb = [item["image_rgb"] for item in batch_items]
        detections = detector.detect_batch(images_rgb, score_threshold=detector_score_threshold)
        pose_tensors = []
        pose_meta = []
        for item, (det_box, det_score) in zip(batch_items, detections):
            if det_box is None:
                print(f"Detector missed {item['video_name']} frame {item['frame_idx']}")
                continue
            crop_box = expand_box_xyxy(det_box, item["image_rgb"].shape[:2], crop_expand_scale)
            x1, y1, x2, y2 = crop_box
            crop = item["image_rgb"][y1:y2, x1:x2]
            resized = cv2.resize(crop, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
            image_float = resized.astype(np.float32) / 255.0
            image_norm = (image_float - image_mean[None, None, :]) / image_std[None, None, :]
            tensor = torch.from_numpy(np.ascontiguousarray(image_norm.transpose(2, 0, 1))).float()
            pose_tensors.append(tensor)
            pose_meta.append({
                "video_name": item["video_name"],
                "frame_idx": item["frame_idx"],
                "image_width": int(item["image_rgb"].shape[1]),
                "image_height": int(item["image_rgb"].shape[0]),
                "crop_box": [float(v) for v in crop_box],
                "detector_box": [float(v) for v in det_box],
                "detector_score": float(det_score),
            })

        if pose_tensors:
            tensor_batch = torch.stack(pose_tensors, dim=0).to(device_obj)
            outputs = model(tensor_batch)
            points, coord_scores = decode_keypoints_with_predictor(model, outputs)
            visibility_scores = visibility_probabilities(outputs)
            probs_x, probs_y = simcc_probabilities(outputs)

            points = points.detach().cpu().numpy()
            coord_scores = coord_scores.detach().cpu().numpy()
            visibility_scores = visibility_scores.detach().cpu().numpy()
            probs_x = probs_x.detach().cpu().numpy()
            probs_y = probs_y.detach().cpu().numpy()

            for idx, meta in enumerate(pose_meta):
                crop_box = meta["crop_box"]
                crop_w = max(1.0, crop_box[2] - crop_box[0])
                crop_h = max(1.0, crop_box[3] - crop_box[1])
                mapped = points[idx].copy()
                mapped[:, 0] = crop_box[0] + mapped[:, 0] * crop_w / float(input_w)
                mapped[:, 1] = crop_box[1] + mapped[:, 1] * crop_h / float(input_h)
                new_predictions.append({
                    "video_name": meta["video_name"],
                    "frame_idx": int(meta["frame_idx"]),
                    "image_width": int(meta["image_width"]),
                    "image_height": int(meta["image_height"]),
                    "crop_box": meta["crop_box"],
                    "detector_box": meta["detector_box"],
                    "detector_score": float(meta["detector_score"]),
                    "checkpoint_path": str(checkpoint_path),
                    "model_name": model_name,
                    "keypoints": {
                        bodypart: {
                            "x": float(mapped[kp_idx, 0]),
                            "y": float(mapped[kp_idx, 1]),
                            "score": float(coord_scores[idx, kp_idx]),
                            "visibility_score": float(visibility_scores[idx, kp_idx]),
                            "x_distribution": probs_x[idx, kp_idx].astype(float).tolist(),
                            "y_distribution": probs_y[idx, kp_idx].astype(float).tolist(),
                        }
                        for kp_idx, bodypart in enumerate(bodyparts)
                    },
                })

        batch_items = []

    combined_predictions = existing_predictions + new_predictions
    save_predictions(output_path, combined_predictions)
    total_new_predictions += len(new_predictions)
    print(
        f"Saved {len(new_predictions)} new predictions for {video_name} "
        f"({len(combined_predictions)}/{len(requested_frames)} frames written) -> {output_path}"
    )

print(f"Finished. Wrote {total_new_predictions} new frame predictions to {output_root}")
