#!/usr/bin/env python3
"""
HRNet-style rat keypoint training script.

Features implemented in this file:
- split handling driven only by config.yaml
- labeled and mask-only weak samples
- cached single-rat ROI loading
- HRNet-style heatmap model with visibility head
- labeled loss = heatmap + visibility + mask containment
- unlabeled loss = mask containment + anti-collapse entropy
- train / eval / predict / visualize subcommands
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from modules.label_csv_utils import load_keypoints


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_LABELS_ROOT = PROJECT_ROOT / "input" / "labeled-data"
DEFAULT_FRAMES_ROOT = PROJECT_ROOT / "output" / "video_frames_for_sam"
DEFAULT_MASKS_ROOT = PROJECT_ROOT / "output" / "sam2_labels"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "hrnet"
DEFAULT_KEYPOINT_CACHE_ROOT = PROJECT_ROOT / "output" / "keypoint_cache"
DEFAULT_MODEL_CONFIG_PATH = PROJECT_ROOT / "input" / "hrnet.yaml"
DEFAULT_TRAIN_CONFIG_PATH = PROJECT_ROOT / "training_config" / "default.yaml"
DEFAULT_ROI_CACHE_PATH = DEFAULT_KEYPOINT_CACHE_ROOT / "roi_cache.json"

LAMBDA_KPT = 1.0
LAMBDA_VIS = 0.5
LAMBDA_MASK = 0.5
LAMBDA_ENTROPY = 0.05

SPLIT_NAMES = ("train", "val", "test")


@dataclass
class ProjectConfig:
    train_videos: List[str]
    val_videos: List[str]
    test_videos: List[str]
    bodyparts: List[str]
    skeleton: List[List[str]]
    left_right_symmetry: List[List[str]]

    def videos_for_split(self, split: str) -> List[str]:
        if split == "train":
            return self.train_videos
        if split == "val":
            return self.val_videos
        if split == "test":
            return self.test_videos
        raise ValueError(f"Unknown split: {split}")


@dataclass
class LabeledSample:
    split: str
    video_name: str
    frame_idx: int
    image_path: str
    mask_path: str
    keypoints: List[List[float]]
    visibility: List[int]


@dataclass
class WeakSample:
    split: str
    video_name: str
    frame_idx: int
    image_path: str
    mask_path: str


@dataclass
class SplitIndex:
    split: str
    videos: List[str]
    labeled_samples: List[LabeledSample]
    weak_samples: List[WeakSample]
    warnings: List[str]


def load_project_config(config_path: Path) -> ProjectConfig:
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ProjectConfig(
        train_videos=list(raw.get("train_videos", [])),
        val_videos=list(raw.get("val_videos", [])),
        test_videos=list(raw.get("test_videos", [])),
        bodyparts=list(raw.get("bodyparts", [])),
        skeleton=list(raw.get("skeleton", [])),
        left_right_symmetry=list(raw.get("left_right_symmetry", [])),
    )


def load_yaml_file(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in YAML file: {path}")
    return payload


def load_model_registry(model_config_path: Path) -> Dict[str, Dict[str, object]]:
    payload = load_yaml_file(model_config_path)
    registry: Dict[str, Dict[str, object]] = {}
    for model_name, model_config in payload.items():
        if isinstance(model_config, dict):
            registry[str(model_name)] = dict(model_config)
    if not registry:
        raise ValueError(f"No model configurations found in {model_config_path}")
    return registry


def resolve_model_config(args: argparse.Namespace) -> Dict[str, object]:
    registry = load_model_registry(args.model_config)
    if args.model_name not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError(
            f"Unknown model_name '{args.model_name}' in {args.model_config}. Available: {available}"
        )
    model_config = dict(registry[args.model_name])
    model_config["model_name"] = args.model_name
    return model_config


def collect_parser_defaults(parser: argparse.ArgumentParser) -> Dict[str, object]:
    defaults: Dict[str, object] = {}
    for action in parser._actions:
        if not getattr(action, "dest", None) or action.dest == argparse.SUPPRESS:
            continue
        if action.default is not argparse.SUPPRESS:
            defaults[action.dest] = action.default
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                defaults.update(collect_parser_defaults(subparser))
    return defaults


def collect_parser_dests(parser: argparse.ArgumentParser) -> Set[str]:
    dests: Set[str] = set()
    for action in parser._actions:
        if getattr(action, "dest", None) and action.dest != argparse.SUPPRESS:
            dests.add(action.dest)
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                dests.update(collect_parser_dests(subparser))
    return dests


def apply_train_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    train_config_path = getattr(args, "train_config", None)
    if train_config_path is None:
        return args
    if not train_config_path.is_file():
        raise FileNotFoundError(f"Training config not found: {train_config_path}")

    payload = load_yaml_file(train_config_path)
    parser_defaults = collect_parser_defaults(parser)
    parser_dests = collect_parser_dests(parser)
    unknown_keys = sorted(key for key in payload if key not in parser_dests)
    if unknown_keys:
        raise ValueError(
            "Unknown keys in training config "
            f"{train_config_path}: {', '.join(unknown_keys)}"
        )
    path_keys = {
        "config",
        "labels_root",
        "frames_root",
        "masks_root",
        "output_root",
        "model_config",
        "train_config",
        "roi_cache",
    }
    for key, value in payload.items():
        if not hasattr(args, key):
            continue
        current_value = getattr(args, key)
        default_value = parser_defaults.get(key)
        if current_value != default_value:
            continue
        if key in path_keys and value is not None:
            setattr(args, key, Path(value))
        else:
            setattr(args, key, value)
    return args


def summarize_resolved_train_args(args: argparse.Namespace) -> str:
    return (
        "Resolved train config: "
        f"epochs={args.epochs} "
        f"patience={args.patience} "
        f"lr={args.lr} "
        f"weight_decay={args.weight_decay} "
        f"labeled_batch_size={args.labeled_batch_size} "
        f"weak_batch_size={args.weak_batch_size} "
        f"eval_batch_size={args.eval_batch_size} "
        f"workers={args.workers} "
        f"prefetch_factor={args.prefetch_factor} "
        f"pin_memory={args.pin_memory} "
        f"persistent_workers={args.persistent_workers} "
        f"preload_images={args.preload_images} "
        f"preload_masks={args.preload_masks} "
        f"input={args.input_width}x{args.input_height} "
        f"heatmap={args.heatmap_width}x{args.heatmap_height} "
        f"sigma={args.sigma} "
        f"model_name={args.model_name}"
    )


def validate_disjoint_splits(config: ProjectConfig) -> None:
    split_to_videos = {
        "train": set(config.train_videos),
        "val": set(config.val_videos),
        "test": set(config.test_videos),
    }
    overlaps = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        common = sorted(split_to_videos[left] & split_to_videos[right])
        if common:
            overlaps.append((left, right, common))
    if overlaps:
        lines = ["config.yaml contains overlapping videos across splits:"]
        for left, right, common in overlaps:
            lines.append(f"  {left}/{right}: {', '.join(common)}")
        raise ValueError("\n".join(lines))


def write_yaml(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def make_serializable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: make_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_serializable(item) for item in value]
    return value


def write_metrics_text(path: Path, title: str, metrics: Mapping[str, object]) -> None:
    lines = [title]
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.6f}")
        else:
            lines.append(f"{key}: {value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_required_torch_device(device_arg: str) -> torch.device:
    requested = (device_arg or "cuda:0").strip()
    if not requested.startswith("cuda"):
        raise ValueError(f"CUDA is required. Unsupported device setting: '{requested}'")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script, but torch.cuda.is_available() is False.")
    return torch.device(requested)


def load_mask_frame_dict(mask_path: Path) -> Dict[int, np.ndarray]:
    with mask_path.open("rb") as f:
        payload = pickle.load(f)
    mask_map: Dict[int, np.ndarray] = {}
    if not isinstance(payload, Mapping):
        return mask_map
    for key, value in payload.items():
        try:
            frame_idx = int(key)
        except (TypeError, ValueError):
            continue
        if not isinstance(value, Mapping) or not value:
            continue
        first_mask = value[next(iter(value))]
        mask = np.asarray(first_mask).astype(bool)
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.ndim != 2:
            continue
        mask_map[frame_idx] = mask
    return mask_map


def list_frame_files(video_frames_dir: Path) -> Dict[int, str]:
    frame_files: Dict[int, str] = {}
    if not video_frames_dir.is_dir():
        return frame_files
    for name in os.listdir(video_frames_dir):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        try:
            frame_idx = int(stem)
        except ValueError:
            continue
        frame_files[frame_idx] = str(video_frames_dir / name)
    return frame_files


def extract_keypoints_for_row(df, row_idx: int, bodyparts: Sequence[str]) -> Tuple[List[List[float]], List[int]]:
    row = df.iloc[row_idx]
    keypoints: List[List[float]] = []
    visibility: List[int] = []
    for bodypart in bodyparts:
        x_key = f"{bodypart}_x"
        y_key = f"{bodypart}_y"
        x = row.get(x_key, np.nan)
        y = row.get(y_key, np.nan)
        if np.isnan(x) or np.isnan(y):
            keypoints.append([0.0, 0.0])
            visibility.append(0)
        else:
            keypoints.append([float(x), float(y)])
            visibility.append(1)
    return keypoints, visibility


def build_split_index(
    config: ProjectConfig,
    split: str,
    labels_root: Path,
    frames_root: Path,
    masks_root: Path,
) -> SplitIndex:
    videos = config.videos_for_split(split)
    labeled_samples: List[LabeledSample] = []
    weak_samples: List[WeakSample] = []
    warnings: List[str] = []

    for video_name in videos:
        csv_path = labels_root / video_name / "CollectedData_rats.csv"
        frames_dir = frames_root / video_name
        mask_path = masks_root / f"{video_name}.pkl"

        if not frames_dir.is_dir():
            warnings.append(f"{split}:{video_name}: missing frames dir {frames_dir}")
            continue
        if not mask_path.is_file():
            warnings.append(f"{split}:{video_name}: missing SAM2 mask file {mask_path}")
            continue

        frame_files = list_frame_files(frames_dir)
        if not frame_files:
            warnings.append(f"{split}:{video_name}: no frame files found")
            continue

        mask_map = load_mask_frame_dict(mask_path)
        if not mask_map:
            warnings.append(f"{split}:{video_name}: no valid masks found")
            continue

        labeled_frame_indices: Set[int] = set()
        if csv_path.is_file():
            label_df = load_keypoints(str(csv_path))
            for row_idx in range(len(label_df)):
                frame_idx = int(label_df["frame"].iloc[row_idx])
                labeled_frame_indices.add(frame_idx)
                image_path = frame_files.get(frame_idx)
                if image_path is None:
                    warnings.append(f"{split}:{video_name}: labeled frame {frame_idx} missing image")
                    continue
                if frame_idx not in mask_map:
                    warnings.append(f"{split}:{video_name}: labeled frame {frame_idx} missing mask")
                    continue
                keypoints, visibility = extract_keypoints_for_row(label_df, row_idx, config.bodyparts)
                labeled_samples.append(
                    LabeledSample(
                        split=split,
                        video_name=video_name,
                        frame_idx=frame_idx,
                        image_path=image_path,
                        mask_path=str(mask_path),
                        keypoints=keypoints,
                        visibility=visibility,
                    )
                )
        else:
            warnings.append(f"{split}:{video_name}: missing label CSV {csv_path}")

        for frame_idx, image_path in frame_files.items():
            if frame_idx in labeled_frame_indices:
                continue
            if frame_idx not in mask_map:
                continue
            weak_samples.append(
                WeakSample(
                    split=split,
                    video_name=video_name,
                    frame_idx=frame_idx,
                    image_path=image_path,
                    mask_path=str(mask_path),
                )
            )

    return SplitIndex(
        split=split,
        videos=videos,
        labeled_samples=sorted(labeled_samples, key=lambda s: (s.video_name, s.frame_idx)),
        weak_samples=sorted(weak_samples, key=lambda s: (s.video_name, s.frame_idx)),
        warnings=warnings,
    )


def build_all_split_indices(
    config: ProjectConfig,
    labels_root: Path,
    frames_root: Path,
    masks_root: Path,
) -> Dict[str, SplitIndex]:
    return {
        split: build_split_index(config, split, labels_root, frames_root, masks_root)
        for split in SPLIT_NAMES
    }


def print_split_summaries(split_indices: Mapping[str, SplitIndex]) -> None:
    print("=" * 80)
    print("Config-driven dataset splits")
    print("=" * 80)
    for split in SPLIT_NAMES:
        split_index = split_indices[split]
        print(f"\n[{split.upper()}]")
        print(f"Videos ({len(split_index.videos)}):")
        for video_name in split_index.videos:
            print(f"  - {video_name}")
        print(f"Labeled samples: {len(split_index.labeled_samples)}")
        print(f"Weak unlabeled samples: {len(split_index.weak_samples)}")
        print(f"Warnings: {len(split_index.warnings)}")
    print()


def ensure_split_membership(split_indices: Mapping[str, SplitIndex]) -> None:
    expected = {split: set(split_indices[split].videos) for split in SPLIT_NAMES}
    for split in SPLIT_NAMES:
        for sample in split_indices[split].labeled_samples:
            if sample.video_name not in expected[split]:
                raise ValueError(f"Leaked labeled sample {sample.video_name}:{sample.frame_idx}")
        for sample in split_indices[split].weak_samples:
            if sample.video_name not in expected[split]:
                raise ValueError(f"Leaked weak sample {sample.video_name}:{sample.frame_idx}")


def summarize_membership(split_indices: Mapping[str, SplitIndex]) -> Dict[str, Dict[str, object]]:
    summary: Dict[str, Dict[str, object]] = {}
    for split in SPLIT_NAMES:
        split_index = split_indices[split]
        summary[split] = {
            "videos": list(split_index.videos),
            "labeled_sample_count": len(split_index.labeled_samples),
            "weak_sample_count": len(split_index.weak_samples),
            "warnings": list(split_index.warnings),
        }
    return summary


def create_run_dir(output_root: Path, prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def create_run_dir_from_config(output_root: Path, config_path: Path) -> Path:
    return create_run_dir(output_root, config_path.stem)


def format_float_tag(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def build_loss_tag(args: argparse.Namespace) -> str:
    return (
        f"lk{format_float_tag(args.lambda_kpt)}_"
        f"lv{format_float_tag(args.lambda_vis)}_"
        f"lm{format_float_tag(args.lambda_mask)}"
    )


def safe_video_name(video_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", video_name).strip("_")


def sample_cache_key(video_name: str, frame_idx: int) -> str:
    return f"{video_name}:{int(frame_idx)}"


def normalize_roi(roi: Sequence[object]) -> Tuple[int, int, int, int]:
    if len(roi) != 4:
        raise ValueError(f"ROI must contain 4 values, got {roi}")
    x1, y1, x2, y2 = [int(v) for v in roi]
    return x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)


def load_roi_cache(path: Path) -> Dict[Tuple[str, int], Tuple[int, int, int, int]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"ROI cache not found: {path}. Run `python scripts/keypoint_preprocess.py` first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", payload)
    roi_cache: Dict[Tuple[str, int], Tuple[int, int, int, int]] = {}
    if not isinstance(entries, Mapping):
        raise ValueError(f"Invalid ROI cache format in {path}")
    for key, value in entries.items():
        if not isinstance(key, str) or ":" not in key:
            continue
        video_name, frame_text = key.rsplit(":", 1)
        try:
            frame_idx = int(frame_text)
        except ValueError:
            continue
        raw_roi = value.get("augmentation_roi") if isinstance(value, Mapping) else value
        if raw_roi is None and isinstance(value, Mapping):
            raw_roi = value.get("roi")
        if not isinstance(raw_roi, Sequence):
            continue
        roi_cache[(video_name, frame_idx)] = normalize_roi(raw_roi)
    return roi_cache


def load_keypoint_cache_entries(path: Path) -> Dict[Tuple[str, int], Dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"ROI cache not found: {path}. Run `python scripts/keypoint_preprocess.py` first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", payload)
    cache_entries: Dict[Tuple[str, int], Dict[str, object]] = {}
    if not isinstance(entries, Mapping):
        raise ValueError(f"Invalid ROI cache format in {path}")
    for key, value in entries.items():
        if not isinstance(key, str) or ":" not in key or not isinstance(value, Mapping):
            continue
        video_name, frame_text = key.rsplit(":", 1)
        try:
            frame_idx = int(frame_text)
        except ValueError:
            continue
        entry = dict(value)
        if "augmentation_roi" in entry:
            entry["augmentation_roi"] = normalize_roi(entry["augmentation_roi"])
        elif "roi" in entry:
            entry["augmentation_roi"] = normalize_roi(entry["roi"])
        if "object_roi" in entry:
            entry["object_roi"] = normalize_roi(entry["object_roi"])
        cache_entries[(video_name, frame_idx)] = entry
    return cache_entries


def select_videos_for_command(
    split_indices: Mapping[str, SplitIndex],
    split: str,
    video_name: Optional[str],
) -> List[str]:
    split_videos = split_indices[split].videos
    if video_name is None:
        return list(split_videos)
    if video_name not in split_videos:
        raise ValueError(f"Video '{video_name}' is not part of split '{split}' in config.yaml")
    return [video_name]


class PoseSampleStore:
    def __init__(
        self,
        bodyparts: Sequence[str],
        cache_entries: Mapping[Tuple[str, int], Mapping[str, object]],
    ) -> None:
        self.bodyparts = list(bodyparts)
        self.cache_entries = {key: dict(value) for key, value in cache_entries.items()}
        self.image_cache: Dict[str, np.ndarray] = {}
        self.crop_mask_cache: Dict[str, np.ndarray] = {}
        self.mask_cache: Dict[str, Dict[int, np.ndarray]] = {}

    def load_mask(self, mask_path: str, frame_idx: int) -> np.ndarray:
        if mask_path.lower().endswith(".npy"):
            if mask_path not in self.crop_mask_cache:
                self.crop_mask_cache[mask_path] = np.load(mask_path).astype(bool)
            return self.crop_mask_cache[mask_path]
        if mask_path not in self.mask_cache:
            self.mask_cache[mask_path] = load_mask_frame_dict(Path(mask_path))
        return self.mask_cache[mask_path][frame_idx]

    def load_image(self, image_path: str) -> np.ndarray:
        if image_path in self.image_cache:
            return self.image_cache[image_path]
        if image_path.lower().endswith(".npy"):
            image = np.load(image_path)
            self.image_cache[image_path] = image
            return image
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.image_cache[image_path] = image
        return image

    def load_roi(self, video_name: str, frame_idx: int) -> Tuple[int, int, int, int]:
        key = (video_name, frame_idx)
        if key not in self.cache_entries:
            raise KeyError(
                f"Missing ROI cache entry for {video_name}:{frame_idx}. "
                f"Run `python scripts/keypoint_preprocess.py` to regenerate {DEFAULT_ROI_CACHE_PATH}."
            )
        return self.cache_entries[key]["augmentation_roi"]

    def load_prepared_crop(
        self,
        video_name: str,
        frame_idx: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[int, int]]]:
        key = (video_name, frame_idx)
        entry = self.cache_entries.get(key)
        if entry is None:
            return None
        crop_path = entry.get("crop_path")
        crop_mask_path = entry.get("crop_mask_path")
        roi = entry.get("augmentation_roi")
        if not isinstance(crop_path, str) or not isinstance(crop_mask_path, str) or roi is None:
            return None
        image = self.load_image(crop_path)
        mask = self.load_mask(crop_mask_path, frame_idx)
        x1, y1, _, _ = roi
        return image, mask, (x1, y1)

    def preload(
        self,
        samples: Sequence[object],
        preload_images: bool,
        preload_masks: bool,
    ) -> Dict[str, object]:
        image_bytes = 0
        mask_bytes = 0
        prepared_mask_count = 0

        if preload_images:
            for sample in samples:
                key = (sample.video_name, sample.frame_idx)
                entry = self.cache_entries.get(key, {})
                candidate_path = entry.get("crop_path") if isinstance(entry, Mapping) else None
                image_path = candidate_path if isinstance(candidate_path, str) else sample.image_path
                if image_path in self.image_cache:
                    continue
                image = self.load_image(image_path)
                self.image_cache[image_path] = image
                image_bytes += int(image.nbytes)

        if preload_masks:
            for sample in samples:
                key = (sample.video_name, sample.frame_idx)
                entry = self.cache_entries.get(key, {})
                crop_mask_path = entry.get("crop_mask_path") if isinstance(entry, Mapping) else None
                if isinstance(crop_mask_path, str):
                    if crop_mask_path in self.crop_mask_cache:
                        continue
                    mask = self.load_mask(crop_mask_path, sample.frame_idx)
                    mask_bytes += int(mask.nbytes)
                    prepared_mask_count += 1
                    continue
                if sample.mask_path in self.mask_cache:
                    continue
                mask_map = load_mask_frame_dict(Path(sample.mask_path))
                self.mask_cache[sample.mask_path] = mask_map
                mask_bytes += int(sum(mask.nbytes for mask in mask_map.values()))

        return {
            "image_count": len(self.image_cache),
            "prepared_mask_count": prepared_mask_count,
            "mask_file_count": len(self.mask_cache),
            "image_bytes": image_bytes,
            "mask_bytes": mask_bytes,
        }


def resize_keypoints(
    keypoints: np.ndarray,
    visibility: np.ndarray,
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> np.ndarray:
    out = keypoints.copy()
    if src_w <= 0 or src_h <= 0:
        return out
    scale_x = dst_w / float(src_w)
    scale_y = dst_h / float(src_h)
    visible = visibility > 0
    out[visible, 0] *= scale_x
    out[visible, 1] *= scale_y
    return out


def horizontal_flip_keypoints(
    keypoints: np.ndarray,
    visibility: np.ndarray,
    width: int,
    flip_pairs: Sequence[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    out_points = keypoints.copy()
    out_vis = visibility.copy()
    visible = out_vis > 0
    out_points[visible, 0] = (width - 1) - out_points[visible, 0]
    for left_idx, right_idx in flip_pairs:
        out_points[[left_idx, right_idx]] = out_points[[right_idx, left_idx]]
        out_vis[[left_idx, right_idx]] = out_vis[[right_idx, left_idx]]
    return out_points, out_vis


def rotate_sample(
    image: np.ndarray,
    mask: np.ndarray,
    keypoints: np.ndarray,
    visibility: np.ndarray,
    angle_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    center = (0.5 * w, 0.5 * h)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated_image = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    rotated_mask = cv2.warpAffine(
        mask.astype(np.uint8),
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    rotated_points = keypoints.copy()
    visible = visibility > 0
    if visible.any():
        pts = np.concatenate([rotated_points[visible], np.ones((visible.sum(), 1), dtype=np.float32)], axis=1)
        rotated_points[visible] = pts @ matrix.T
    return rotated_image, rotated_mask, rotated_points


def combined_foreground_box(
    mask: np.ndarray,
    keypoints: np.ndarray,
    visibility: np.ndarray,
    image_shape: Tuple[int, int],
    pad_pixels: int = 4,
) -> Optional[Tuple[int, int, int, int]]:
    h, w = image_shape
    boxes: List[Tuple[int, int, int, int]] = []

    ys, xs = np.where(mask > 0)
    if len(xs) > 0 and len(ys) > 0:
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))

    visible = visibility > 0
    if visible.any():
        pts = keypoints[visible]
        boxes.append(
            (
                int(np.floor(pts[:, 0].min())),
                int(np.floor(pts[:, 1].min())),
                int(np.ceil(pts[:, 0].max())) + 1,
                int(np.ceil(pts[:, 1].max())) + 1,
            )
        )

    if not boxes:
        return None

    x1 = max(0, min(box[0] for box in boxes) - pad_pixels)
    y1 = max(0, min(box[1] for box in boxes) - pad_pixels)
    x2 = min(w, max(box[2] for box in boxes) + pad_pixels)
    y2 = min(h, max(box[3] for box in boxes) + pad_pixels)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def tighten_crop_after_augmentation(
    image: np.ndarray,
    mask: np.ndarray,
    keypoints: np.ndarray,
    visibility: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    box = combined_foreground_box(mask, keypoints, visibility, image.shape[:2])
    if box is None:
        return image, mask, keypoints
    x1, y1, x2, y2 = box
    cropped_image = image[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]
    cropped_keypoints = keypoints.copy()
    visible = visibility > 0
    cropped_keypoints[visible, 0] -= x1
    cropped_keypoints[visible, 1] -= y1
    return cropped_image, cropped_mask, cropped_keypoints


def build_gaussian_heatmaps(
    keypoints: np.ndarray,
    visibility: np.ndarray,
    output_h: int,
    output_w: int,
    sigma: float,
    input_w: int,
    input_h: int,
) -> np.ndarray:
    heatmaps = np.zeros((len(keypoints), output_h, output_w), dtype=np.float32)
    tmp_size = int(sigma * 3)
    stride_x = input_w / float(output_w)
    stride_y = input_h / float(output_h)
    for idx, ((x, y), vis) in enumerate(zip(keypoints, visibility)):
        if vis <= 0:
            continue
        mu_x = x / stride_x
        mu_y = y / stride_y
        mu_x_int = int(round(mu_x))
        mu_y_int = int(round(mu_y))
        ul = [mu_x_int - tmp_size, mu_y_int - tmp_size]
        br = [mu_x_int + tmp_size + 1, mu_y_int + tmp_size + 1]
        if ul[0] >= output_w or ul[1] >= output_h or br[0] < 0 or br[1] < 0:
            continue
        size = 2 * tmp_size + 1
        x_coords = np.arange(0, size, 1, np.float32)
        y_coords = x_coords[:, None]
        gaussian = np.exp(-((x_coords - tmp_size) ** 2 + (y_coords - tmp_size) ** 2) / (2 * sigma ** 2))
        g_x = max(0, -ul[0]), min(br[0], output_w) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], output_h) - ul[1]
        img_x = max(0, ul[0]), min(br[0], output_w)
        img_y = max(0, ul[1]), min(br[1], output_h)
        heatmaps[idx, img_y[0]:img_y[1], img_x[0]:img_x[1]] = gaussian[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return heatmaps


class PoseDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[object],
        sample_store: PoseSampleStore,
        config: ProjectConfig,
        input_size: Tuple[int, int],
        heatmap_size: Tuple[int, int],
        sigma: float,
        train_mode: bool,
        is_labeled: bool,
    ) -> None:
        self.samples = list(samples)
        self.sample_store = sample_store
        self.config = config
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.train_mode = train_mode
        self.is_labeled = is_labeled
        name_to_idx = {name: idx for idx, name in enumerate(config.bodyparts)}
        self.flip_pairs = [
            (name_to_idx[left], name_to_idx[right])
            for left, right in config.left_right_symmetry
            if left in name_to_idx and right in name_to_idx
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        roi = self.sample_store.load_roi(sample.video_name, sample.frame_idx)
        prepared = self.sample_store.load_prepared_crop(sample.video_name, sample.frame_idx)
        x1, y1, x2, y2 = roi
        if prepared is not None:
            crop_image, crop_mask, (x1, y1) = prepared
        else:
            image = self.sample_store.load_image(sample.image_path)
            mask = self.sample_store.load_mask(sample.mask_path, sample.frame_idx)
            x2 = max(x2, x1 + 1)
            y2 = max(y2, y1 + 1)
            crop_image = image[y1:y2, x1:x2]
            crop_mask = mask[y1:y2, x1:x2]
            if crop_image.size == 0 or crop_mask.size == 0:
                crop_image = image
                crop_mask = mask
                x1, y1 = 0, 0

        if self.is_labeled:
            keypoints = np.asarray(sample.keypoints, dtype=np.float32).copy()
            visibility = np.asarray(sample.visibility, dtype=np.float32).copy()
            visible = visibility > 0
            keypoints[visible, 0] -= x1
            keypoints[visible, 1] -= y1
        else:
            keypoints = np.zeros((len(self.config.bodyparts), 2), dtype=np.float32)
            visibility = np.zeros((len(self.config.bodyparts),), dtype=np.float32)

        if self.train_mode:
            if random.random() < 0.5:
                crop_image = np.ascontiguousarray(crop_image[:, ::-1])
                crop_mask = np.ascontiguousarray(crop_mask[:, ::-1])
                keypoints, visibility = horizontal_flip_keypoints(
                    keypoints, visibility, crop_image.shape[1], self.flip_pairs
                )
            angle = random.uniform(-20.0, 20.0)
            crop_image, crop_mask, keypoints = rotate_sample(
                crop_image, crop_mask, keypoints, visibility, angle
            )
            crop_image, crop_mask, keypoints = tighten_crop_after_augmentation(
                crop_image, crop_mask, keypoints, visibility
            )
            alpha = random.uniform(0.9, 1.1)
            beta = random.uniform(-10.0, 10.0)
            crop_image = np.clip(crop_image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        src_h, src_w = crop_image.shape[:2]
        dst_w, dst_h = self.input_size
        resized_image = cv2.resize(crop_image, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(crop_mask.astype(np.uint8), (dst_w, dst_h), interpolation=cv2.INTER_NEAREST) > 0
        resized_keypoints = resize_keypoints(keypoints, visibility, src_w, src_h, dst_w, dst_h)

        heatmaps = build_gaussian_heatmaps(
            resized_keypoints,
            visibility,
            self.heatmap_size[1],
            self.heatmap_size[0],
            self.sigma,
            dst_w,
            dst_h,
        )
        mask_small = cv2.resize(
            resized_mask.astype(np.uint8),
            self.heatmap_size,
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)

        image_tensor = torch.from_numpy(resized_image.astype(np.float32).transpose(2, 0, 1) / 255.0)
        heatmap_tensor = torch.from_numpy(heatmaps)
        visibility_tensor = torch.from_numpy(visibility.astype(np.float32))
        mask_tensor = torch.from_numpy(mask_small[None, ...])
        keypoint_tensor = torch.from_numpy(resized_keypoints.astype(np.float32))

        return {
            "image": image_tensor,
            "heatmaps": heatmap_tensor,
            "visibility": visibility_tensor,
            "mask": mask_tensor,
            "keypoints": keypoint_tensor,
            "is_labeled": torch.tensor(1.0 if self.is_labeled else 0.0, dtype=torch.float32),
            "video_name": sample.video_name,
            "frame_idx": torch.tensor(sample.frame_idx, dtype=torch.int64),
        }


class BasicBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


def make_basic_stage(channels: int, num_blocks: int) -> nn.Sequential:
    return nn.Sequential(*[BasicBlock(channels) for _ in range(max(1, num_blocks))])


class MiniHRNet(nn.Module):
    def __init__(self, num_keypoints: int, model_config: Optional[Mapping[str, object]] = None) -> None:
        super().__init__()
        cfg = dict(model_config or {})
        base_channels = int(cfg.get("base_channels", 32))
        branch2_factor = int(cfg.get("branch2_factor", 2))
        branch1_blocks = int(cfg.get("branch1_blocks", 2))
        branch2_blocks = int(cfg.get("branch2_blocks", 2))
        fuse_blocks = int(cfg.get("fuse_blocks", 2))
        vis_hidden_channels = int(cfg.get("vis_hidden_channels", base_channels))
        stem_layers = int(cfg.get("stem_layers", 2))

        stem_modules: List[nn.Module] = []
        in_channels = 3
        for _ in range(max(1, stem_layers)):
            stem_modules.extend(
                [
                    nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = base_channels

        self.stem = nn.Sequential(
            *stem_modules
        )
        self.branch1 = make_basic_stage(base_channels, branch1_blocks)
        branch2_channels = base_channels * branch2_factor
        self.down_to_branch2 = nn.Sequential(
            nn.Conv2d(base_channels, branch2_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(branch2_channels),
            nn.ReLU(inplace=True),
        )
        self.branch2 = make_basic_stage(branch2_channels, branch2_blocks)
        self.up_to_branch1 = nn.Sequential(
            nn.Conv2d(branch2_channels, base_channels, 1, bias=False),
            nn.BatchNorm2d(base_channels),
        )
        self.fuse_high = make_basic_stage(base_channels, fuse_blocks)
        self.head = nn.Conv2d(base_channels, num_keypoints, 1)
        self.vis_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels, vis_hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(vis_hidden_channels, num_keypoints),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        high = self.branch1(x)
        low = self.branch2(self.down_to_branch2(high))
        low_up = F.interpolate(low, size=high.shape[-2:], mode="bilinear", align_corners=False)
        high = self.fuse_high(high + self.up_to_branch1(low_up))
        heatmaps = self.head(high)
        visibility_logits = self.vis_head(high)
        return heatmaps, visibility_logits


def heatmaps_to_probabilities(heatmaps: torch.Tensor) -> torch.Tensor:
    b, k, h, w = heatmaps.shape
    probs = F.softmax(heatmaps.view(b, k, h * w), dim=-1)
    return probs.view(b, k, h, w)


def compute_losses(
    pred_heatmaps: torch.Tensor,
    pred_vis_logits: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    lambda_kpt: float,
    lambda_vis: float,
    lambda_mask: float,
    lambda_entropy: float,
) -> Dict[str, torch.Tensor]:
    gt_heatmaps = batch["heatmaps"]
    gt_visibility = batch["visibility"]
    mask = batch["mask"]
    is_labeled = batch["is_labeled"]

    visible_weights = gt_visibility[:, :, None, None]
    if visible_weights.sum() > 0:
        heatmap_loss = ((pred_heatmaps - gt_heatmaps) ** 2 * visible_weights).sum() / visible_weights.sum()
    else:
        heatmap_loss = pred_heatmaps.sum() * 0.0

    vis_loss = F.binary_cross_entropy_with_logits(pred_vis_logits, gt_visibility)

    probs = heatmaps_to_probabilities(pred_heatmaps)
    outside = 1.0 - mask
    outside_mass = (probs * outside).sum(dim=(-1, -2))
    mask_loss = outside_mass.mean()

    entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=(-1, -2))
    entropy = entropy / math.log(pred_heatmaps.shape[-1] * pred_heatmaps.shape[-2])
    entropy_loss = entropy.mean()

    labeled_mask = is_labeled > 0.5
    unlabeled_mask = ~labeled_mask

    total = pred_heatmaps.sum() * 0.0
    if labeled_mask.any():
        total = total + lambda_kpt * heatmap_loss + lambda_vis * vis_loss + lambda_mask * mask_loss
    if unlabeled_mask.any():
        total = total + lambda_mask * mask_loss + lambda_entropy * entropy_loss

    return {
        "total": total,
        "heatmap": heatmap_loss.detach(),
        "visibility": vis_loss.detach(),
        "mask": mask_loss.detach(),
        "entropy": entropy_loss.detach(),
    }


def decode_keypoints(heatmaps: torch.Tensor, input_size: Tuple[int, int]) -> torch.Tensor:
    b, k, h, w = heatmaps.shape
    flat = heatmaps.view(b, k, h * w)
    idx = flat.argmax(dim=-1)
    xs = (idx % w).float() * (input_size[0] / float(w))
    ys = (idx // w).float() * (input_size[1] / float(h))
    return torch.stack([xs, ys], dim=-1)


def accumulate_pose_metrics(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    visibility: torch.Tensor,
) -> Dict[str, float]:
    visible = visibility > 0.5
    if not visible.any():
        return {
            "visible_count": 0.0,
            "sum_euclidean_error": 0.0,
            "sum_squared_coord_error": 0.0,
            "coord_count": 0.0,
        }

    diffs = pred_points[visible] - gt_points[visible]
    euclidean_error = torch.norm(diffs, dim=-1)
    return {
        "visible_count": float(visible.sum().item()),
        "sum_euclidean_error": float(euclidean_error.sum().item()),
        "sum_squared_coord_error": float((diffs ** 2).sum().item()),
        "coord_count": float(diffs.numel()),
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    input_size: Tuple[int, int],
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_visible = 0
    total_error = 0.0
    total_squared_coord_error = 0.0
    total_coord_count = 0.0
    total_vis_correct = 0.0
    total_vis_count = 0.0

    for batch in dataloader:
        image = batch["image"].to(device)
        heatmaps = batch["heatmaps"].to(device)
        visibility = batch["visibility"].to(device)
        mask = batch["mask"].to(device)
        is_labeled = batch["is_labeled"].to(device)
        keypoints = batch["keypoints"].to(device)
        pred_heatmaps, pred_vis_logits = model(image)
        losses = compute_losses(
            pred_heatmaps,
            pred_vis_logits,
            {
                "heatmaps": heatmaps,
                "visibility": visibility,
                "mask": mask,
                "is_labeled": is_labeled,
            },
            lambda_kpt=1.0,
            lambda_vis=1.0,
            lambda_mask=0.5,
            lambda_entropy=0.05,
        )
        total_loss += float(losses["total"].item())
        total_batches += 1

        pred_points = decode_keypoints(pred_heatmaps, input_size)
        pose_metrics = accumulate_pose_metrics(pred_points, keypoints, visibility)
        total_error += pose_metrics["sum_euclidean_error"]
        total_visible += int(pose_metrics["visible_count"])
        total_squared_coord_error += pose_metrics["sum_squared_coord_error"]
        total_coord_count += pose_metrics["coord_count"]

        pred_vis = (torch.sigmoid(pred_vis_logits) > 0.5).float()
        total_vis_correct += float((pred_vis == visibility).float().sum().item())
        total_vis_count += float(visibility.numel())

    return {
        "loss": total_loss / max(1, total_batches),
        "mean_pixel_error": total_error / max(1, total_visible),
        "rmse_unfiltered": math.sqrt(total_squared_coord_error / max(1.0, total_coord_count)),
        "visibility_accuracy": total_vis_correct / max(1.0, total_vis_count),
    }


def cycle_loader(loader: Optional[DataLoader]) -> Iterable[Optional[Dict[str, torch.Tensor]]]:
    if loader is None:
        while True:
            yield None
    while True:
        for batch in loader:
            yield batch


def maybe_cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024.0


def train_one_epoch(
    model: nn.Module,
    labeled_loader: DataLoader,
    weak_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    lambda_kpt: float,
    lambda_vis: float,
    lambda_mask: float,
    lambda_entropy: float,
    use_amp: bool,
    profile_steps: int = 0,
) -> Dict[str, float]:
    model.train()
    labeled_iter = cycle_loader(labeled_loader)
    weak_iter = cycle_loader(weak_loader)
    steps = max(len(labeled_loader), len(weak_loader) if weak_loader is not None else 0, 1)
    totals = {
        "total": 0.0,
        "heatmap": 0.0,
        "visibility": 0.0,
        "mask": 0.0,
        "entropy": 0.0,
        "mean_pixel_error": 0.0,
        "rmse_unfiltered": 0.0,
        "visibility_accuracy": 0.0,
    }
    total_visible = 0.0
    total_euclidean_error = 0.0
    total_squared_coord_error = 0.0
    total_coord_count = 0.0
    total_vis_correct = 0.0
    total_vis_count = 0.0
    total_data_time = 0.0
    total_h2d_time = 0.0
    total_forward_time = 0.0
    total_backward_time = 0.0
    total_step_time = 0.0

    for step_idx in range(steps):
        step_start = time.perf_counter()
        data_start = time.perf_counter()
        batches = [next(labeled_iter)]
        weak_batch = next(weak_iter)
        if weak_loader is not None and weak_batch is not None:
            batches.append(weak_batch)
        data_time = time.perf_counter() - data_start

        optimizer.zero_grad(set_to_none=True)
        h2d_time = 0.0
        forward_time = 0.0
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            total_tensor = None
            merged_logs = {}
            for batch in batches:
                maybe_cuda_sync(device)
                h2d_start = time.perf_counter()
                image = batch["image"].to(device)
                heatmaps = batch["heatmaps"].to(device)
                visibility = batch["visibility"].to(device)
                mask = batch["mask"].to(device)
                is_labeled = batch["is_labeled"].to(device)
                maybe_cuda_sync(device)
                h2d_time += time.perf_counter() - h2d_start

                maybe_cuda_sync(device)
                forward_start = time.perf_counter()
                pred_heatmaps, pred_vis_logits = model(image)
                losses = compute_losses(
                    pred_heatmaps,
                    pred_vis_logits,
                    {
                        "heatmaps": heatmaps,
                        "visibility": visibility,
                        "mask": mask,
                        "is_labeled": is_labeled,
                    },
                    lambda_kpt=lambda_kpt,
                    lambda_vis=lambda_vis,
                    lambda_mask=lambda_mask,
                    lambda_entropy=lambda_entropy,
                )
                maybe_cuda_sync(device)
                forward_time += time.perf_counter() - forward_start
                total_tensor = losses["total"] if total_tensor is None else total_tensor + losses["total"]
                for key, value in losses.items():
                    merged_logs[key] = merged_logs.get(key, 0.0) + float(value.item())

                if float(is_labeled.max().item()) > 0.5:
                    pred_points = decode_keypoints(pred_heatmaps, (batch["image"].shape[-1], batch["image"].shape[-2]))
                    pose_metrics = accumulate_pose_metrics(pred_points, batch["keypoints"].to(device), visibility)
                    total_euclidean_error += pose_metrics["sum_euclidean_error"]
                    total_visible += pose_metrics["visible_count"]
                    total_squared_coord_error += pose_metrics["sum_squared_coord_error"]
                    total_coord_count += pose_metrics["coord_count"]
                    pred_vis = (torch.sigmoid(pred_vis_logits) > 0.5).float()
                    total_vis_correct += float((pred_vis == visibility).float().sum().item())
                    total_vis_count += float(visibility.numel())

        maybe_cuda_sync(device)
        backward_start = time.perf_counter()
        scaler.scale(total_tensor).backward()
        scaler.step(optimizer)
        scaler.update()
        maybe_cuda_sync(device)
        backward_time = time.perf_counter() - backward_start
        step_time = time.perf_counter() - step_start

        for key in totals:
            if key in {"mean_pixel_error", "rmse_unfiltered", "visibility_accuracy"}:
                continue
            totals[key] += merged_logs.get(key, 0.0)
        total_data_time += data_time
        total_h2d_time += h2d_time
        total_forward_time += forward_time
        total_backward_time += backward_time
        total_step_time += step_time

        if profile_steps > 0 and step_idx < profile_steps:
            print(
                f"[profile] step={step_idx + 1}/{steps} "
                f"data={data_time:.4f}s "
                f"h2d={h2d_time:.4f}s "
                f"forward={forward_time:.4f}s "
                f"backward={backward_time:.4f}s "
                f"total={step_time:.4f}s"
            )

    for key in totals:
        if key in {"mean_pixel_error", "rmse_unfiltered", "visibility_accuracy"}:
            continue
        totals[key] /= float(steps)
    totals["mean_pixel_error"] = total_euclidean_error / max(1.0, total_visible)
    totals["rmse_unfiltered"] = math.sqrt(total_squared_coord_error / max(1.0, total_coord_count))
    totals["visibility_accuracy"] = total_vis_correct / max(1.0, total_vis_count)
    totals["profile_data_time"] = total_data_time / float(steps)
    totals["profile_h2d_time"] = total_h2d_time / float(steps)
    totals["profile_forward_time"] = total_forward_time / float(steps)
    totals["profile_backward_time"] = total_backward_time / float(steps)
    totals["profile_step_time"] = total_step_time / float(steps)
    return totals


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    config: ProjectConfig,
    args: argparse.Namespace,
    model_config: Mapping[str, object],
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "bodyparts": config.bodyparts,
        "args": vars(args),
        "model_config": make_serializable(dict(model_config)),
    }
    torch.save(payload, path)


def load_checkpoint_payload(path: Path) -> Dict[str, object]:
    return torch.load(path, map_location="cpu")


def load_checkpoint(path: Path, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, object]:
    checkpoint = load_checkpoint_payload(path)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


def build_dataloaders(
    args: argparse.Namespace,
    config: ProjectConfig,
    split_indices: Mapping[str, SplitIndex],
    cache_entries: Mapping[Tuple[str, int], Mapping[str, object]],
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    sample_store = PoseSampleStore(config.bodyparts, cache_entries)
    all_samples = [
        *split_indices["train"].labeled_samples,
        *split_indices["train"].weak_samples,
        *split_indices["val"].labeled_samples,
    ]
    preload_stats = sample_store.preload(
        all_samples,
        preload_images=args.preload_images,
        preload_masks=args.preload_masks,
    )
    input_size = (args.input_width, args.input_height)
    heatmap_size = (args.heatmap_width, args.heatmap_height)
    loader_kwargs = {
        "num_workers": args.workers,
        "drop_last": False,
        "pin_memory": args.pin_memory,
    }
    if args.workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_labeled_dataset = PoseDataset(
        split_indices["train"].labeled_samples,
        sample_store,
        config,
        input_size,
        heatmap_size,
        args.sigma,
        train_mode=True,
        is_labeled=True,
    )
    train_weak_dataset = PoseDataset(
        split_indices["train"].weak_samples,
        sample_store,
        config,
        input_size,
        heatmap_size,
        args.sigma,
        train_mode=True,
        is_labeled=False,
    )
    val_dataset = PoseDataset(
        split_indices["val"].labeled_samples,
        sample_store,
        config,
        input_size,
        heatmap_size,
        args.sigma,
        train_mode=False,
        is_labeled=True,
    )

    labeled_loader = DataLoader(
        train_labeled_dataset,
        batch_size=args.labeled_batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    weak_loader = None
    if len(train_weak_dataset) > 0 and args.weak_batch_size > 0:
        weak_loader = DataLoader(
            train_weak_dataset,
            batch_size=args.weak_batch_size,
            shuffle=True,
            **loader_kwargs,
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    print(
        "Preload summary: "
        f"images={preload_stats['image_count']} ({format_bytes(int(preload_stats['image_bytes']))} loaded this run), "
        f"prepared_masks={preload_stats['prepared_mask_count']}, "
        f"mask_files={preload_stats['mask_file_count']} "
        f"({format_bytes(int(preload_stats['mask_bytes']))} loaded this run)"
    )
    return labeled_loader, weak_loader, val_loader


def command_train(
    args: argparse.Namespace,
    config: ProjectConfig,
    split_indices: Mapping[str, SplitIndex],
) -> int:
    ensure_split_membership(split_indices)
    device = resolve_required_torch_device(args.device)
    use_amp = not args.no_amp
    model_config = resolve_model_config(args)
    cache_entries = load_keypoint_cache_entries(args.roi_cache)

    if getattr(args, "train_config", None) is not None:
        run_dir = create_run_dir_from_config(args.output_root, args.train_config)
    else:
        run_dir = create_run_dir(args.output_root, f"train_{build_loss_tag(args)}")
    print(f"Train run directory: {run_dir}")
    print(f"Python: {sys.executable}")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Torch device: {device}")
    print(f"ROI cache: {args.roi_cache}")
    print(f"Model preset: {args.model_name}")
    print(f"Workers: {args.workers}")
    print(f"Pin memory: {args.pin_memory}")
    print(f"Persistent workers: {args.persistent_workers}")
    print(f"Prefetch factor: {args.prefetch_factor}")
    print(f"Preload images: {args.preload_images}")
    print(f"Preload masks: {args.preload_masks}")
    print(summarize_resolved_train_args(args))

    write_yaml(run_dir / "args.yaml", make_serializable(vars(args)))
    write_json(run_dir / "split_summary.json", summarize_membership(split_indices))
    write_yaml(run_dir / "model_config.yaml", make_serializable(model_config))

    if args.dry_run:
        print("Dry run enabled; split summary written and training skipped.")
        return 0

    labeled_loader, weak_loader, val_loader = build_dataloaders(args, config, split_indices, cache_entries)

    model = MiniHRNet(num_keypoints=len(config.bodyparts), model_config=model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_metric = float("inf")
    best_epoch = -1
    patience_counter = 0
    train_log_path = run_dir / "train_log.csv"
    train_text_log_path = run_dir / "train_log.txt"
    with train_log_path.open("w", encoding="utf-8", newline="") as f:
        train_text_log_path.write_text("", encoding="utf-8")
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_total_loss",
                "train_heatmap_loss",
                "train_visibility_loss",
                "train_mask_loss",
                "train_entropy_loss",
                "train_mean_pixel_error",
                "train_rmse_unfiltered",
                "train_visibility_accuracy",
                "val_loss",
                "val_mean_pixel_error",
                "val_rmse_unfiltered",
                "val_visibility_accuracy",
                "lr",
            ]
        )

        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(
                model=model,
                labeled_loader=labeled_loader,
                weak_loader=weak_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                lambda_kpt=args.lambda_kpt,
                lambda_vis=args.lambda_vis,
                lambda_mask=args.lambda_mask,
                lambda_entropy=args.lambda_entropy,
                use_amp=use_amp,
                profile_steps=args.profile_steps,
            )
            val_metrics = evaluate_model(
                model,
                val_loader,
                device=device,
                input_size=(args.input_width, args.input_height),
            )
            lr = optimizer.param_groups[0]["lr"]
            writer.writerow(
                [
                    epoch,
                    train_metrics["total"],
                    train_metrics["heatmap"],
                    train_metrics["visibility"],
                    train_metrics["mask"],
                    train_metrics["entropy"],
                    train_metrics["mean_pixel_error"],
                    train_metrics["rmse_unfiltered"],
                    train_metrics["visibility_accuracy"],
                    val_metrics["loss"],
                    val_metrics["mean_pixel_error"],
                    val_metrics["rmse_unfiltered"],
                    val_metrics["visibility_accuracy"],
                    lr,
                ]
            )
            f.flush()

            log_line = (
                f"Epoch {epoch:03d} | "
                f"train_total={train_metrics['total']:.4f} "
                f"train_rmse={train_metrics['rmse_unfiltered']:.4f} "
                f"train_px={train_metrics['mean_pixel_error']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_rmse={val_metrics['rmse_unfiltered']:.4f} "
                f"val_px={val_metrics['mean_pixel_error']:.4f} "
                f"val_vis_acc={val_metrics['visibility_accuracy']:.3f} "
                f"profile(data={train_metrics['profile_data_time']:.4f}s "
                f"h2d={train_metrics['profile_h2d_time']:.4f}s "
                f"forward={train_metrics['profile_forward_time']:.4f}s "
                f"backward={train_metrics['profile_backward_time']:.4f}s "
                f"step={train_metrics['profile_step_time']:.4f}s)"
            )
            print(log_line)
            with train_text_log_path.open("a", encoding="utf-8") as text_f:
                text_f.write(log_line + "\n")

            write_metrics_text(
                run_dir / "latest_metrics.txt",
                "Latest Training Metrics",
                {
                    "epoch": epoch,
                    "train_total_loss": train_metrics["total"],
                    "train_heatmap_loss": train_metrics["heatmap"],
                    "train_visibility_loss": train_metrics["visibility"],
                    "train_mask_loss": train_metrics["mask"],
                    "train_entropy_loss": train_metrics["entropy"],
                    "train_mean_pixel_error": train_metrics["mean_pixel_error"],
                    "train_rmse_unfiltered": train_metrics["rmse_unfiltered"],
                    "train_visibility_accuracy": train_metrics["visibility_accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_mean_pixel_error": val_metrics["mean_pixel_error"],
                    "val_rmse_unfiltered": val_metrics["rmse_unfiltered"],
                    "val_visibility_accuracy": val_metrics["visibility_accuracy"],
                    "lr": lr,
                },
            )

            save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, best_metric, config, args, model_config)
            if val_metrics["rmse_unfiltered"] < best_metric:
                best_metric = val_metrics["rmse_unfiltered"]
                best_epoch = epoch
                patience_counter = 0
                save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, best_metric, config, args, model_config)
                write_metrics_text(
                    run_dir / "best_metrics.txt",
                    "Best Validation Metrics",
                    {
                        "best_epoch": best_epoch,
                        "best_val_rmse_unfiltered": best_metric,
                        "best_val_mean_pixel_error": val_metrics["mean_pixel_error"],
                        "best_val_visibility_accuracy": val_metrics["visibility_accuracy"],
                    },
                )
            else:
                patience_counter += 1

            scheduler.step()
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break

    print(f"Best validation RMSE (unfiltered): {best_metric:.4f} at epoch {best_epoch}")
    return 0


def build_eval_dataset(
    args: argparse.Namespace,
    config: ProjectConfig,
    samples: Sequence[LabeledSample],
    cache_entries: Mapping[Tuple[str, int], Mapping[str, object]],
) -> DataLoader:
    sample_store = PoseSampleStore(config.bodyparts, cache_entries)
    preload_stats = sample_store.preload(
        samples,
        preload_images=args.preload_images,
        preload_masks=args.preload_masks,
    )
    dataset = PoseDataset(
        samples,
        sample_store,
        config,
        (args.input_width, args.input_height),
        (args.heatmap_width, args.heatmap_height),
        args.sigma,
        train_mode=False,
        is_labeled=True,
    )
    loader_kwargs = {
        "batch_size": args.eval_batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": args.pin_memory,
    }
    if args.workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    print(
        "Preload summary: "
        f"images={preload_stats['image_count']} ({format_bytes(int(preload_stats['image_bytes']))} loaded this run), "
        f"prepared_masks={preload_stats['prepared_mask_count']}, "
        f"mask_files={preload_stats['mask_file_count']} "
        f"({format_bytes(int(preload_stats['mask_bytes']))} loaded this run)"
    )
    return DataLoader(dataset, **loader_kwargs)


def command_eval(
    args: argparse.Namespace,
    config: ProjectConfig,
    split_indices: Mapping[str, SplitIndex],
) -> int:
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for eval")
    device = resolve_required_torch_device(args.device)
    cache_entries = load_keypoint_cache_entries(args.roi_cache)
    split = args.split
    selected_videos = select_videos_for_command(split_indices, split, args.video)
    selected_samples = [
        sample for sample in split_indices[split].labeled_samples if sample.video_name in selected_videos
    ]
    print(f"Evaluation split: {split}")
    print(f"Videos selected ({len(selected_videos)}):")
    for video_name in selected_videos:
        print(f"  - {video_name}")
    print(f"Labeled evaluation samples: {len(selected_samples)}")

    dataloader = build_eval_dataset(args, config, selected_samples, cache_entries)
    checkpoint_payload = load_checkpoint_payload(args.checkpoint)
    model_config = checkpoint_payload.get("model_config") or resolve_model_config(args)
    model = MiniHRNet(num_keypoints=len(config.bodyparts), model_config=model_config).to(device)
    model.load_state_dict(checkpoint_payload["model_state"])
    metrics = evaluate_model(model, dataloader, device, (args.input_width, args.input_height))
    run_dir = create_run_dir(args.output_root, f"eval_{split}")
    print(f"Eval output directory: {run_dir}")
    metrics_payload = {
        "split": split,
        "videos": selected_videos,
        "sample_count": len(selected_samples),
        **metrics,
    }
    print(json.dumps(metrics, indent=2))
    write_json(run_dir / "metrics.json", metrics_payload)
    write_metrics_text(run_dir / "metrics.txt", f"Evaluation Metrics ({split})", metrics_payload)
    write_yaml(run_dir / "model_config.yaml", make_serializable(model_config))
    if args.output_json:
        write_json(args.output_json, metrics_payload)
    return 0


@torch.no_grad()
def run_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    config: ProjectConfig,
    device: torch.device,
    input_size: Tuple[int, int],
) -> List[Dict[str, object]]:
    model.eval()
    predictions: List[Dict[str, object]] = []
    for batch in dataloader:
        image = batch["image"].to(device)
        pred_heatmaps, pred_vis_logits = model(image)
        keypoints = decode_keypoints(pred_heatmaps, input_size).cpu().numpy()
        vis_probs = torch.sigmoid(pred_vis_logits).cpu().numpy()
        frame_indices = batch["frame_idx"].cpu().numpy().tolist()
        video_names = batch["video_name"]
        for idx in range(len(frame_indices)):
            predictions.append(
                {
                    "video_name": video_names[idx],
                    "frame_idx": int(frame_indices[idx]),
                    "keypoints": {
                        bodypart: {
                            "x": float(keypoints[idx, kp_idx, 0]),
                            "y": float(keypoints[idx, kp_idx, 1]),
                            "visibility_prob": float(vis_probs[idx, kp_idx]),
                        }
                        for kp_idx, bodypart in enumerate(config.bodyparts)
                    },
                }
            )
    return predictions


def command_predict(
    args: argparse.Namespace,
    config: ProjectConfig,
    split_indices: Mapping[str, SplitIndex],
) -> int:
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for predict")
    device = resolve_required_torch_device(args.device)
    cache_entries = load_keypoint_cache_entries(args.roi_cache)
    split = args.split
    selected_videos = select_videos_for_command(split_indices, split, args.video)

    labeled_samples = [
        sample for sample in split_indices[split].labeled_samples if sample.video_name in selected_videos
    ]
    if not labeled_samples:
        print("No labeled samples found for predict; nothing to do.")
        return 0

    dataloader = build_eval_dataset(args, config, labeled_samples, cache_entries)
    checkpoint_payload = load_checkpoint_payload(args.checkpoint)
    model_config = checkpoint_payload.get("model_config") or resolve_model_config(args)
    model = MiniHRNet(num_keypoints=len(config.bodyparts), model_config=model_config).to(device)
    model.load_state_dict(checkpoint_payload["model_state"])
    predictions = run_predictions(model, dataloader, config, device, (args.input_width, args.input_height))

    output_path = args.output_json or (args.output_root / f"predictions_{split}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, predictions)
    print(f"Wrote predictions to {output_path}")
    return 0


def overlay_prediction(
    image_path: str,
    prediction: Dict[str, object],
    skeleton: Sequence[Sequence[str]],
    output_path: Path,
) -> None:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return
    kp_map = prediction["keypoints"]
    for bodypart, info in kp_map.items():
        x = int(round(info["x"]))
        y = int(round(info["y"]))
        color = (0, 255, 0) if info["visibility_prob"] >= 0.5 else (0, 165, 255)
        cv2.circle(image, (x, y), 3, color, -1)
        cv2.putText(image, bodypart, (x + 4, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
    for left, right in skeleton:
        if left not in kp_map or right not in kp_map:
            continue
        x1 = int(round(kp_map[left]["x"]))
        y1 = int(round(kp_map[left]["y"]))
        x2 = int(round(kp_map[right]["x"]))
        y2 = int(round(kp_map[right]["y"]))
        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 0), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def command_visualize(
    args: argparse.Namespace,
    config: ProjectConfig,
    split_indices: Mapping[str, SplitIndex],
) -> int:
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for visualize")
    split = args.split
    selected_videos = select_videos_for_command(split_indices, split, args.video)
    device = resolve_required_torch_device(args.device)
    cache_entries = load_keypoint_cache_entries(args.roi_cache)
    labeled_samples = [
        sample for sample in split_indices[split].labeled_samples if sample.video_name in selected_videos
    ]
    dataloader = build_eval_dataset(args, config, labeled_samples[: args.max_visualizations], cache_entries)
    checkpoint_payload = load_checkpoint_payload(args.checkpoint)
    model_config = checkpoint_payload.get("model_config") or resolve_model_config(args)
    model = MiniHRNet(num_keypoints=len(config.bodyparts), model_config=model_config).to(device)
    model.load_state_dict(checkpoint_payload["model_state"])
    predictions = run_predictions(model, dataloader, config, device, (args.input_width, args.input_height))
    output_dir = args.output_root / f"visualize_{split}"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_lookup = {(sample.video_name, sample.frame_idx): sample for sample in labeled_samples}
    for prediction in predictions[: args.max_visualizations]:
        key = (prediction["video_name"], prediction["frame_idx"])
        sample = sample_lookup.get(key)
        if sample is None:
            continue
        out_name = f"{safe_video_name(sample.video_name)}_frame{sample.frame_idx:06d}.jpg"
        overlay_prediction(sample.image_path, prediction, config.skeleton, output_dir / out_name)
    print(f"Wrote visualizations to {output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HRNet rat pose training with config.yaml-driven splits.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--labels-root", type=Path, default=DEFAULT_LABELS_ROOT)
    parser.add_argument("--frames-root", type=Path, default=DEFAULT_FRAMES_ROOT)
    parser.add_argument("--masks-root", type=Path, default=DEFAULT_MASKS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--roi-cache", type=Path, default=DEFAULT_ROI_CACHE_PATH)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG_PATH)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--preload-images", action="store_true")
    parser.add_argument("--preload-masks", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG_PATH)
    train_parser.add_argument("--dry-run", action="store_true")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--patience", type=int, default=5)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--labeled-batch-size", type=int, default=8)
    train_parser.add_argument("--weak-batch-size", type=int, default=4)
    train_parser.add_argument("--eval-batch-size", type=int, default=8)
    train_parser.add_argument("--input-width", type=int, default=256)
    train_parser.add_argument("--input-height", type=int, default=192)
    train_parser.add_argument("--heatmap-width", type=int, default=64)
    train_parser.add_argument("--heatmap-height", type=int, default=48)
    train_parser.add_argument("--sigma", type=float, default=2.0)
    train_parser.add_argument("--lambda-kpt", type=float, default=LAMBDA_KPT)
    train_parser.add_argument("--lambda-vis", type=float, default=LAMBDA_VIS)
    train_parser.add_argument("--lambda-mask", type=float, default=LAMBDA_MASK)
    train_parser.add_argument("--lambda-entropy", type=float, default=LAMBDA_ENTROPY)
    train_parser.add_argument("--model-name", type=str, default="hrnet_w32")
    train_parser.add_argument("--no-amp", action="store_true")
    train_parser.add_argument("--profile-steps", type=int, default=0)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--split", choices=("val", "test"), default="test")
    eval_parser.add_argument("--video", type=str, default=None)
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--output-json", type=Path, default=None)
    eval_parser.add_argument("--eval-batch-size", type=int, default=8)
    eval_parser.add_argument("--input-width", type=int, default=256)
    eval_parser.add_argument("--input-height", type=int, default=192)
    eval_parser.add_argument("--heatmap-width", type=int, default=64)
    eval_parser.add_argument("--heatmap-height", type=int, default=48)
    eval_parser.add_argument("--sigma", type=float, default=2.0)
    eval_parser.add_argument("--model-name", type=str, default="hrnet_w32")

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--split", choices=SPLIT_NAMES, default="test")
    predict_parser.add_argument("--video", type=str, default=None)
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--output-json", type=Path, default=None)
    predict_parser.add_argument("--eval-batch-size", type=int, default=8)
    predict_parser.add_argument("--input-width", type=int, default=256)
    predict_parser.add_argument("--input-height", type=int, default=192)
    predict_parser.add_argument("--heatmap-width", type=int, default=64)
    predict_parser.add_argument("--heatmap-height", type=int, default=48)
    predict_parser.add_argument("--sigma", type=float, default=2.0)
    predict_parser.add_argument("--model-name", type=str, default="hrnet_w32")

    visualize_parser = subparsers.add_parser("visualize")
    visualize_parser.add_argument("--split", choices=SPLIT_NAMES, default="val")
    visualize_parser.add_argument("--video", type=str, default=None)
    visualize_parser.add_argument("--checkpoint", type=Path, required=True)
    visualize_parser.add_argument("--max-visualizations", type=int, default=32)
    visualize_parser.add_argument("--eval-batch-size", type=int, default=8)
    visualize_parser.add_argument("--input-width", type=int, default=256)
    visualize_parser.add_argument("--input-height", type=int, default=192)
    visualize_parser.add_argument("--heatmap-width", type=int, default=64)
    visualize_parser.add_argument("--heatmap-height", type=int, default=48)
    visualize_parser.add_argument("--sigma", type=float, default=2.0)
    visualize_parser.add_argument("--model-name", type=str, default="hrnet_w32")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        args = apply_train_config(args, parser)
    set_seed(args.seed)

    config = load_project_config(args.config)
    validate_disjoint_splits(config)
    split_indices = build_all_split_indices(config, args.labels_root, args.frames_root, args.masks_root)
    print_split_summaries(split_indices)

    if args.command == "train":
        return command_train(args, config, split_indices)
    if args.command == "eval":
        return command_eval(args, config, split_indices)
    if args.command == "predict":
        return command_predict(args, config, split_indices)
    if args.command == "visualize":
        return command_visualize(args, config, split_indices)

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
