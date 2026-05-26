#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from modules.detector_ssdlite_model import SSDLiteDetector, build_ssdlite_model, load_checkpoint, load_detector
from modules.label_csv_utils import load_keypoints
from modules.keypoint_rtmpose_predict_common import load_model_from_checkpoint_for_inference
from modules.keypoint_rtmpose_predict_common import (
    build_pose_model,
    decode_keypoints_with_predictor,
    resolve_detector_model_config,
    simcc_probabilities,
    visibility_probabilities,
)
from modules.keypoint_rtmpose_model import (
    StandaloneRTMPose,
    save_rtmpose_checkpoint,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_PROJECT_CONFIG = PROJECT_ROOT / "config.yaml"
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "input" / "RTMPose" / "model_rtmpose_s.yaml"
DEFAULT_RUN_CONFIG = PROJECT_ROOT / "input" / "RTMPose" / "config.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "RTMPose"
SPLIT_NAMES = ("train", "val", "test")


@dataclass
class ProjectConfig:
    train_videos: list[str]
    val_videos: list[str]
    test_videos: list[str]
    bodyparts: list[str]
    skeleton: list[list[str]]
    left_right_symmetry: list[list[str]]

    def videos_for_split(self, split: str) -> list[str]:
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
    keypoints: list[list[float]]
    visibility: list[int]
    source_name: str = "label"


@dataclass
class WeakSample:
    split: str
    video_name: str
    frame_idx: int
    image_path: str
    mask_path: str
    source_name: str = "sam2"


@dataclass
class SplitIndex:
    split: str
    videos: list[str]
    labeled_samples: list[LabeledSample]
    weak_samples: list[WeakSample]
    warnings: list[str]


def load_yaml_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in YAML file: {path}")
    return payload


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def write_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_serializable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: make_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_serializable(item) for item in value]
    return value


class TeeStream:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def prune_periodic_checkpoints(run_dir: Path, max_keep: int) -> None:
    if max_keep < 0:
        return
    checkpoints = sorted(run_dir.glob("checkpoint_epoch_*.pt"))
    if len(checkpoints) <= max_keep:
        return
    for checkpoint_path in checkpoints[:-max_keep]:
        checkpoint_path.unlink(missing_ok=True)


def resolve_merged_run_config(run_config_path: Path, config_overwrite_path: Path | None) -> dict[str, Any]:
    payload = load_yaml_file(run_config_path)
    if config_overwrite_path is not None:
        payload = deep_merge_dicts(payload, load_yaml_file(config_overwrite_path))
    return payload


def export_config_bundle(
    run_dir: Path,
    *,
    project_config_path: Path,
    resolved_run_args: Mapping[str, Any],
    model_config_path: Path,
    model_name: str,
    run_config_path: Path,
    config_overwrite_path: Path | None,
) -> None:
    write_yaml(run_dir / "project_config.yaml", load_yaml_file(project_config_path))
    model_registry = load_yaml_file(model_config_path)
    resolved_model_cfg = model_registry.get(model_name)
    if not isinstance(resolved_model_cfg, Mapping):
        raise ValueError(f"Unknown model_name={model_name!r} in {model_config_path}")
    model_payload = dict(resolved_model_cfg)
    model_payload["model_name"] = model_name
    write_yaml(run_dir / "model_config.yaml", model_payload)
    write_yaml(
        run_dir / "run_config.yaml",
        {
            "config": resolve_merged_run_config(run_config_path, config_overwrite_path),
            "args": make_serializable(dict(resolved_run_args)),
        },
    )


def deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge_dicts(dict(merged[key]), dict(value))
        else:
            merged[key] = value
    return merged


def collect_parser_defaults(parser: argparse.ArgumentParser) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for action in parser._actions:
        if not getattr(action, "dest", None) or action.dest == argparse.SUPPRESS:
            continue
        if action.default is not argparse.SUPPRESS:
            defaults[action.dest] = action.default
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                defaults.update(collect_parser_defaults(subparser))
    return defaults


def merge_cli_with_yaml(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    payload = load_yaml_file(args.run_config)
    if args.config_overwrite is not None:
        payload = deep_merge_dicts(payload, load_yaml_file(args.config_overwrite))
    merged: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in {"training", "inference"}:
            merged[key] = value
    section = "training" if args.command in {"train", "debug", "prepare"} else "inference"
    if isinstance(payload.get(section), Mapping):
        merged.update(dict(payload[section]))

    defaults = collect_parser_defaults(parser)
    path_keys = {
        "project_config",
        "model_config",
        "run_config",
        "config_overwrite",
        "labels_root",
        "labeled_frames_root",
        "frames_root",
        "masks_root",
        "output_root",
        "checkpoint",
        "video",
        "detector_checkpoint",
        "pretrained_backbone_checkpoint",
        "init_checkpoint",
    }
    for key, value in merged.items():
        if not hasattr(args, key):
            continue
        if key in defaults and getattr(args, key) != defaults[key]:
            continue
        if key in path_keys and value is not None:
            setattr(args, key, resolve_project_path(Path(value)))
        else:
            setattr(args, key, value)
    return args


def load_project_config(config_path: Path) -> ProjectConfig:
    raw = load_yaml_file(config_path)
    return ProjectConfig(
        train_videos=list(raw.get("train_videos", [])),
        val_videos=list(raw.get("val_videos", [])),
        test_videos=list(raw.get("test_videos", [])),
        bodyparts=list(raw.get("bodyparts", [])),
        skeleton=list(raw.get("skeleton", [])),
        left_right_symmetry=list(raw.get("left_right_symmetry", [])),
    )


def validate_disjoint_splits(config: ProjectConfig) -> None:
    split_sets = {
        "train": set(config.train_videos),
        "val": set(config.val_videos),
        "test": set(config.test_videos),
    }
    overlaps: list[str] = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        common = sorted(split_sets[left] & split_sets[right])
        if common:
            overlaps.append(f"{left}/{right}: {', '.join(common)}")
    if overlaps:
        raise ValueError("Overlapping split membership detected:\n" + "\n".join(overlaps))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    requested = (device_name or "cuda:0").strip()
    if not requested.startswith("cuda"):
        raise ValueError(f"CUDA is required for RTMPose training/inference, got {requested!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but torch.cuda.is_available() is False.")
    return torch.device(requested)


def maybe_cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_mask_frame_dict(mask_path: Path) -> dict[int, np.ndarray]:
    with mask_path.open("rb") as f:
        payload = pickle.load(f)
    mask_map: dict[int, np.ndarray] = {}
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
        if mask.ndim == 2:
            mask_map[frame_idx] = mask
    return mask_map


def load_raw_mask_frame_dict(mask_path: Path) -> dict[int, dict[int, np.ndarray]]:
    with mask_path.open("rb") as f:
        payload = pickle.load(f)
    raw_map: dict[int, dict[int, np.ndarray]] = {}
    if not isinstance(payload, Mapping):
        return raw_map
    for key, value in payload.items():
        try:
            frame_idx = int(key)
        except (TypeError, ValueError):
            continue
        if not isinstance(value, Mapping) or not value:
            continue
        frame_masks: dict[int, np.ndarray] = {}
        for obj_id, obj_mask in value.items():
            try:
                obj_key = int(obj_id)
            except (TypeError, ValueError):
                continue
            mask = np.asarray(obj_mask).astype(bool)
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]
            if mask.ndim == 2:
                frame_masks[obj_key] = mask
        if frame_masks:
            raw_map[frame_idx] = frame_masks
    return raw_map


def list_frame_files(video_frames_dir: Path) -> dict[int, str]:
    frame_files: dict[int, str] = {}
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


def mask_bbox_xyxy(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max() + 1)
    y2 = float(ys.max() + 1)
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def bbox_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return float(inter_area / denom)


def extract_keypoints_for_row(df, row_idx: int, bodyparts: Sequence[str]) -> tuple[list[list[float]], list[int]]:
    row = df.iloc[row_idx]
    keypoints: list[list[float]] = []
    visibility: list[int] = []
    for bodypart in bodyparts:
        x = row.get(f"{bodypart}_x", np.nan)
        y = row.get(f"{bodypart}_y", np.nan)
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
    labeled_frames_root: Path,
    weak_frames_root: Path,
    masks_root: Path,
    include_weak: bool = True,
    require_masks: bool = True,
) -> SplitIndex:
    videos = config.videos_for_split(split)
    labeled_samples: list[LabeledSample] = []
    weak_samples: list[WeakSample] = []
    warnings: list[str] = []

    for video_name in videos:
        csv_path = labels_root / video_name / "CollectedData_rats.csv"
        labeled_frames_dir = labeled_frames_root / video_name
        weak_frames_dir = weak_frames_root / video_name
        mask_path = masks_root / f"{video_name}.pkl"
        labeled_frame_files = list_frame_files(labeled_frames_dir)
        weak_frame_files = list_frame_files(weak_frames_dir)
        if not labeled_frame_files:
            warnings.append(f"{split}:{video_name}: missing labeled frames in {labeled_frames_dir}")
            continue
        if include_weak and not weak_frame_files:
            warnings.append(f"{split}:{video_name}: missing weak frames in {weak_frames_dir}")
        mask_map: dict[int, np.ndarray] = {}
        if require_masks or include_weak:
            if not mask_path.is_file():
                warnings.append(f"{split}:{video_name}: missing SAM2 file {mask_path}")
            else:
                mask_map = load_mask_frame_dict(mask_path)
                if not mask_map:
                    warnings.append(f"{split}:{video_name}: empty SAM2 mask map")

        labeled_frame_indices: set[int] = set()
        if csv_path.is_file():
            label_df = load_keypoints(str(csv_path))
            for row_idx in range(len(label_df)):
                frame_idx = int(label_df["frame"].iloc[row_idx])
                labeled_frame_indices.add(frame_idx)
                image_path = labeled_frame_files.get(frame_idx)
                if image_path is None:
                    warnings.append(f"{split}:{video_name}: labeled frame {frame_idx} missing image in {labeled_frames_dir}")
                    continue
                if require_masks and frame_idx not in mask_map:
                    warnings.append(f"{split}:{video_name}: labeled frame {frame_idx} missing mask (kept without mask)")
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
            warnings.append(f"{split}:{video_name}: missing label csv {csv_path}")

        if include_weak:
            for frame_idx, image_path in weak_frame_files.items():
                if frame_idx in labeled_frame_indices:
                    continue
                if require_masks and frame_idx not in mask_map:
                    warnings.append(f"{split}:{video_name}: weak frame {frame_idx} missing mask (kept without mask)")
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
        videos=list(videos),
        labeled_samples=sorted(labeled_samples, key=lambda s: (s.video_name, s.frame_idx)),
        weak_samples=sorted(weak_samples, key=lambda s: (s.video_name, s.frame_idx)),
        warnings=warnings,
    )


def split_labeled_train_val_samples(
    labeled_samples: Sequence[LabeledSample],
    val_fraction: float,
    seed: int,
) -> tuple[list[LabeledSample], list[LabeledSample]]:
    samples = sorted(labeled_samples, key=lambda s: (s.video_name, s.frame_idx))
    if not samples or val_fraction <= 0.0:
        return list(samples), []
    if len(samples) == 1:
        return list(samples), []

    requested = int(round(float(len(samples)) * float(val_fraction)))
    val_count = max(1, min(len(samples) - 1, requested))
    rng = random.Random(int(seed))
    shuffled_indices = list(range(len(samples)))
    rng.shuffle(shuffled_indices)
    val_idx = set(shuffled_indices[:val_count])
    train = [sample for idx, sample in enumerate(samples) if idx not in val_idx]
    val = [sample for idx, sample in enumerate(samples) if idx in val_idx]
    return train, val


def build_all_split_indices(
    config: ProjectConfig,
    labels_root: Path,
    labeled_frames_root: Path,
    weak_frames_root: Path,
    masks_root: Path,
    include_weak: bool = True,
    require_masks: bool = True,
    auto_val_fraction: float = 0.1,
    split_seed: int = 42,
) -> dict[str, SplitIndex]:
    split_indices = {
        split: build_split_index(
            config,
            split,
            labels_root,
            labeled_frames_root,
            weak_frames_root,
            masks_root,
            include_weak=include_weak,
            require_masks=require_masks,
        )
        for split in SPLIT_NAMES
    }
    if not config.val_videos:
        train_labeled, val_labeled = split_labeled_train_val_samples(
            split_indices["train"].labeled_samples,
            val_fraction=float(auto_val_fraction),
            seed=int(split_seed),
        )
        if val_labeled:
            train_index = split_indices["train"]
            val_index = split_indices["val"]
            inferred_val_videos = sorted({sample.video_name for sample in val_labeled})
            split_indices["train"] = SplitIndex(
                split=train_index.split,
                videos=list(train_index.videos),
                labeled_samples=train_labeled,
                weak_samples=list(train_index.weak_samples),
                warnings=list(train_index.warnings)
                + [
                    f"train:auto_val_split seed={int(split_seed)} fraction={float(auto_val_fraction):.4f} "
                    f"moved={len(val_labeled)}"
                ],
            )
            split_indices["val"] = SplitIndex(
                split=val_index.split,
                videos=inferred_val_videos,
                labeled_samples=val_labeled,
                weak_samples=list(val_index.weak_samples),
                warnings=list(val_index.warnings)
                + [
                    f"val:auto_val_split seed={int(split_seed)} fraction={float(auto_val_fraction):.4f} "
                    f"selected={len(val_labeled)}"
                ],
            )
    if include_weak:
        all_weak_samples: list[WeakSample] = []
        for split in SPLIT_NAMES:
            all_weak_samples.extend(split_indices[split].weak_samples)
        split_indices["train"] = SplitIndex(
            split=split_indices["train"].split,
            videos=list(split_indices["train"].videos),
            labeled_samples=list(split_indices["train"].labeled_samples),
            weak_samples=sorted(all_weak_samples, key=lambda s: (s.video_name, s.frame_idx)),
            warnings=list(split_indices["train"].warnings),
        )
        for split in ("val", "test"):
            split_indices[split] = SplitIndex(
                split=split_indices[split].split,
                videos=list(split_indices[split].videos),
                labeled_samples=list(split_indices[split].labeled_samples),
                weak_samples=[],
                warnings=list(split_indices[split].warnings),
            )
    return split_indices

def validate_mutual_exclusion(split_indices: Mapping[str, SplitIndex]) -> None:
    for split, split_index in split_indices.items():
        labeled_keys = {(s.video_name, s.frame_idx) for s in split_index.labeled_samples}
        weak_keys = {(s.video_name, s.frame_idx) for s in split_index.weak_samples}
        overlap = sorted(labeled_keys & weak_keys)
        if overlap:
            raise ValueError(f"Labeled and weak samples overlap in split {split}: {overlap[:5]}")


def summarize_split_indices(split_indices: Mapping[str, SplitIndex]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split in SPLIT_NAMES:
        info = split_indices[split]
        summary[split] = {
            "videos": list(info.videos),
            "labeled_sample_count": len(info.labeled_samples),
            "weak_sample_count": len(info.weak_samples),
            "warnings": list(info.warnings),
        }
    return summary


def print_split_summaries(split_indices: Mapping[str, SplitIndex]) -> None:
    print("Dataset splits")
    for split in SPLIT_NAMES:
        split_index = split_indices[split]
        print(
            f"  {split}: videos={len(split_index.videos)} "
            f"label={len(split_index.labeled_samples)} sam2={len(split_index.weak_samples)} "
            f"warnings={len(split_index.warnings)}"
        )


def resolve_model_config(args: argparse.Namespace) -> dict[str, Any]:
    registry = load_yaml_file(args.model_config)
    model_name = str(args.model_name)
    if model_name not in registry or not isinstance(registry[model_name], Mapping):
        raise ValueError(f"Unknown model_name={model_name!r} in {args.model_config}")
    cfg = dict(registry[model_name])
    cfg["model_name"] = model_name
    return cfg


def build_run_dir(output_root: Path, prefix: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prefix = str(prefix).strip().replace(" ", "_")
    folder = f"{safe_prefix}_{timestamp}" if safe_prefix else timestamp
    run_dir = output_root / folder
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def with_optional_prefix(base_prefix: str, user_prefix: str) -> str:
    left = str(user_prefix or "").strip().replace(" ", "_")
    right = str(base_prefix or "").strip().replace(" ", "_")
    if left and right:
        return f"{left}_{right}"
    if left:
        return left
    return right


def build_prepare_dir(output_root: Path) -> Path:
    return output_root / "prepare"


def cycle_loader(loader: Optional[DataLoader]) -> Iterable[Optional[dict[str, Any]]]:
    if loader is None:
        while True:
            yield None
    while True:
        for batch in loader:
            yield batch


def collate_pose_batch(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in items], dim=0),
        "keypoints": torch.stack([item["keypoints"] for item in items], dim=0),
        "mask": torch.stack([item["mask"] for item in items], dim=0),
        "has_mask": torch.tensor([float(item["has_mask"]) for item in items], dtype=torch.float32),
        "is_labeled": torch.tensor([float(item["is_labeled"]) for item in items], dtype=torch.float32),
        "video_name": [item["video_name"] for item in items],
        "frame_idx": torch.tensor([int(item["frame_idx"]) for item in items], dtype=torch.int64),
        "source_name": [item["source_name"] for item in items],
        "crop_box": torch.from_numpy(np.stack([item["crop_box"] for item in items], axis=0)).float(),
        "orig_size": torch.from_numpy(np.stack([item["orig_size"] for item in items], axis=0)).float(),
        "mask_det_iou": torch.tensor([float(item["mask_det_iou"]) for item in items], dtype=torch.float32),
        "debug_image": torch.stack([item["debug_image"] for item in items], dim=0),
    }


def horizontal_flip_keypoints(
    keypoints: np.ndarray,
    width: int,
    flip_pairs: Sequence[tuple[int, int]],
) -> np.ndarray:
    out = keypoints.copy()
    visible = out[:, 2] > 0
    out[visible, 0] = (width - 1) - out[visible, 0]
    for left_idx, right_idx in flip_pairs:
        out[[left_idx, right_idx]] = out[[right_idx, left_idx]]
    return out


def clamp_visibility_to_window(
    keypoints: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    out = keypoints.copy()
    visible = out[:, 2] > 0
    if not visible.any():
        return out
    inside = (
        (out[:, 0] >= 0.0)
        & (out[:, 0] < float(width))
        & (out[:, 1] >= 0.0)
        & (out[:, 1] < float(height))
    )
    keep = visible & inside
    drop = visible & ~inside
    out[drop, :2] = 0.0
    out[drop, 2] = 0.0
    out[keep, 2] = 1.0
    return out


def rotate_points(keypoints: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    out = keypoints.copy()
    visible = out[:, 2] > 0
    if visible.any():
        pts = np.concatenate([out[visible, :2], np.ones((int(visible.sum()), 1), dtype=np.float32)], axis=1)
        out[visible, :2] = pts @ matrix.T
    return out


def transform_box_xyxy(box: np.ndarray, matrix: np.ndarray, width: int, height: int) -> np.ndarray:
    corners = np.asarray(
        [
            [box[0], box[1]],
            [box[2], box[1]],
            [box[2], box[3]],
            [box[0], box[3]],
        ],
        dtype=np.float32,
    )
    pts = np.concatenate([corners, np.ones((4, 1), dtype=np.float32)], axis=1)
    rotated = pts @ matrix.T
    x_coords = rotated[:, 0]
    y_coords = rotated[:, 1]
    return np.asarray(
        [
            np.clip(x_coords.min(), 0.0, float(width)),
            np.clip(y_coords.min(), 0.0, float(height)),
            np.clip(x_coords.max(), 0.0, float(width)),
            np.clip(y_coords.max(), 0.0, float(height)),
        ],
        dtype=np.float32,
    )


def xyxy_to_xywh(box_xyxy: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    return np.asarray([x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)], dtype=np.float32)


def xywh_to_xyxy(box_xywh: Sequence[float]) -> np.ndarray:
    x, y, w, h = [float(v) for v in box_xywh]
    return np.asarray([x, y, x + max(1.0, w), y + max(1.0, h)], dtype=np.float32)


def bbox_from_visible_keypoints(
    keypoints: np.ndarray,
    image_shape: Sequence[int],
    margin: float = 20.0,
) -> np.ndarray | None:
    visible = keypoints[:, 2] > 0
    if not visible.any():
        return None
    h, w = int(image_shape[0]), int(image_shape[1])
    pts = keypoints[visible, :2]
    x1 = float(max(0.0, np.min(pts[:, 0]) - margin))
    y1 = float(max(0.0, np.min(pts[:, 1]) - margin))
    x2 = float(min(float(w), np.max(pts[:, 0]) + margin))
    y2 = float(min(float(h), np.max(pts[:, 1]) + margin))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def _sample_truncnorm(low: float, high: float, size: Sequence[int]) -> np.ndarray:
    # Lightweight clipped normal sampler to mimic DLC truncnorm behavior without scipy dependency.
    sample = np.random.normal(loc=0.0, scale=0.5, size=size)
    return np.clip(sample, low, high).astype(np.float32)


def apply_random_bbox_transform(
    bbox_xyxy: np.ndarray,
    image_shape: Sequence[int],
    cfg: Mapping[str, Any],
) -> np.ndarray:
    if not cfg or float(cfg.get("p", 1.0)) <= 0.0:
        return bbox_xyxy
    if random.random() > float(cfg.get("p", 1.0)):
        return bbox_xyxy

    shift_factor = float(cfg.get("shift_factor", 0.16))
    shift_prob = float(cfg.get("shift_prob", 0.3))
    scale_factor_cfg = cfg.get("scale_factor", [0.75, 1.25])
    if isinstance(scale_factor_cfg, Sequence) and len(scale_factor_cfg) >= 2:
        scale_low = float(scale_factor_cfg[0])
        scale_high = float(scale_factor_cfg[1])
    else:
        scale_low, scale_high = 0.75, 1.25
    scale_prob = float(cfg.get("scale_prob", 1.0))

    h, w = int(image_shape[0]), int(image_shape[1])
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = x1 + 0.5 * bw
    cy = y1 + 0.5 * bh

    if random.random() < scale_prob:
        sf = _sample_truncnorm(scale_low, scale_high, size=(2,))
        bw *= float(sf[0])
        bh *= float(sf[1])
    if random.random() < shift_prob:
        sh = _sample_truncnorm(-shift_factor, shift_factor, size=(2,))
        cx += float(sh[0]) * bw
        cy += float(sh[1]) * bh

    x1n = max(0.0, cx - 0.5 * bw)
    y1n = max(0.0, cy - 0.5 * bh)
    x2n = min(float(w), cx + 0.5 * bw)
    y2n = min(float(h), cy + 0.5 * bh)
    if x2n <= x1n or y2n <= y1n:
        return bbox_xyxy
    return np.asarray([x1n, y1n, x2n, y2n], dtype=np.float32)


def apply_motion_blur(image_rgb: np.ndarray, p: float = 0.5, kmin: int = 3, kmax: int = 7) -> np.ndarray:
    if random.random() > p:
        return image_rgb
    k = random.randrange(kmin, kmax + 1, 2)
    kernel = np.zeros((k, k), dtype=np.float32)
    if random.random() < 0.5:
        kernel[k // 2, :] = 1.0
    else:
        kernel[:, k // 2] = 1.0
    kernel /= float(np.sum(kernel))
    return cv2.filter2D(image_rgb, -1, kernel)


def apply_gaussian_noise(image_rgb: np.ndarray, noise_std: float = 12.75, p: float = 0.5) -> np.ndarray:
    if random.random() > p:
        return image_rgb
    noise = np.random.normal(loc=0.0, scale=float(noise_std), size=image_rgb.shape).astype(np.float32)
    out = image_rgb.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def top_down_crop_pair(
    image: np.ndarray,
    mask: np.ndarray,
    bbox_xywh: Sequence[float],
    output_size: tuple[int, int],
    margin: int = 0,
    crop_with_context: bool = True,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float], tuple[float, float]]:
    image_h, image_w = image.shape[:2]
    out_w, out_h = int(output_size[0]), int(output_size[1])
    x, y, w, h = [float(v) for v in bbox_xywh]

    cx = x + w / 2.0
    cy = y + h / 2.0
    w += 2.0 * float(margin)
    h += 2.0 * float(margin)

    if crop_with_context:
        input_ratio = w / max(1e-6, h)
        output_ratio = out_w / max(1e-6, out_h)
        if input_ratio > output_ratio:
            h = w / output_ratio
        elif input_ratio < output_ratio:
            w = h * output_ratio

    x1 = int(round(cx - (w / 2.0)))
    y1 = int(round(cy - (h / 2.0)))
    x2 = int(round(cx + (w / 2.0)))
    y2 = int(round(cy + (h / 2.0)))

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - image_w)
    pad_bottom = max(0, y2 - image_h)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_w, x2)
    y2 = min(image_h, y2)

    crop_img = image[y1:y2, x1:x2]
    crop_mask = np.asarray(mask[y1:y2, x1:x2], dtype=np.uint8)
    if pad_left or pad_right or pad_top or pad_bottom:
        crop_img = cv2.copyMakeBorder(
            crop_img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        crop_mask = cv2.copyMakeBorder(
            crop_mask,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )

    pre_h, pre_w = crop_img.shape[:2]
    resized_img = cv2.resize(crop_img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(crop_mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    offset = (float(x1 - pad_left), float(y1 - pad_top))
    scale = (float(pre_w) / float(out_w), float(pre_h) / float(out_h))
    return resized_img, resized_mask, offset, scale


def draw_box_on_image(
    image_rgb: np.ndarray,
    box_xyxy: Sequence[float],
    color: tuple[int, int, int] = (255, 128, 0),
    thickness: int = 2,
) -> np.ndarray:
    canvas = cv2.cvtColor(np.ascontiguousarray(image_rgb.copy()), cv2.COLOR_RGB2BGR)
    x1, y1, x2, y2 = [int(round(float(v))) for v in box_xyxy]
    cv2.rectangle(canvas, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)), color, thickness)
    return canvas


def draw_keypoints_on_image(
    image_rgb: np.ndarray,
    keypoints: np.ndarray,
    bodyparts: Sequence[str],
    skeleton: Sequence[Sequence[str]],
    visibility_cutoff: float = 0.5,
) -> np.ndarray:
    canvas = cv2.cvtColor(np.ascontiguousarray(image_rgb.copy()), cv2.COLOR_RGB2BGR)
    name_to_idx = {name: idx for idx, name in enumerate(bodyparts)}
    for left, right in skeleton:
        if left not in name_to_idx or right not in name_to_idx:
            continue
        li = name_to_idx[left]
        ri = name_to_idx[right]
        if keypoints[li, 2] < visibility_cutoff or keypoints[ri, 2] < visibility_cutoff:
            continue
        p1 = (int(round(keypoints[li, 0])), int(round(keypoints[li, 1])))
        p2 = (int(round(keypoints[ri, 0])), int(round(keypoints[ri, 1])))
        cv2.line(canvas, p1, p2, (255, 255, 0), 1, cv2.LINE_AA)
    for idx, bodypart in enumerate(bodyparts):
        if keypoints[idx, 2] < visibility_cutoff:
            continue
        x = int(round(keypoints[idx, 0]))
        y = int(round(keypoints[idx, 1]))
        cv2.circle(canvas, (x, y), 3, (0, 255, 0), -1)
        cv2.putText(canvas, bodypart, (x + 3, y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
    return canvas


def draw_mask_on_image(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    canvas = cv2.cvtColor(np.ascontiguousarray(image_rgb.copy()), cv2.COLOR_RGB2BGR)
    overlay = np.zeros_like(canvas)
    overlay[mask > 0] = (0, 0, 255)
    return cv2.addWeighted(canvas, 0.75, overlay, 0.25, 0.0)


class RamImageMaskStore:
    def __init__(self) -> None:
        self.image_cache: dict[str, np.ndarray] = {}
        self.mask_cache: dict[str, dict[int, np.ndarray]] = {}
        self.raw_mask_cache: dict[str, dict[int, dict[int, np.ndarray]]] = {}

    def load_image(self, image_path: str) -> np.ndarray:
        if image_path in self.image_cache:
            return self.image_cache[image_path]
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.image_cache[image_path] = image_rgb
        return image_rgb

    def load_mask(
        self,
        mask_path: str,
        frame_idx: int,
        *,
        policy: str = "first",
        bbox_xyxy: np.ndarray | None = None,
    ) -> np.ndarray:
        if policy == "first":
            if mask_path not in self.mask_cache:
                self.mask_cache[mask_path] = load_mask_frame_dict(Path(mask_path))
            mask_map = self.mask_cache[mask_path]
            if frame_idx not in mask_map:
                raise KeyError(f"Missing SAM2 mask for frame {frame_idx} in {mask_path}")
            return mask_map[frame_idx]

        if mask_path not in self.raw_mask_cache:
            self.raw_mask_cache[mask_path] = load_raw_mask_frame_dict(Path(mask_path))
        raw_map = self.raw_mask_cache[mask_path]
        if frame_idx not in raw_map:
            raise KeyError(f"Missing SAM2 mask for frame {frame_idx} in {mask_path}")
        frame_masks = raw_map[frame_idx]
        if not frame_masks:
            raise KeyError(f"Missing SAM2 mask for frame {frame_idx} in {mask_path}")
        if policy == "largest":
            return max(frame_masks.values(), key=lambda m: int(m.sum()))
        if policy == "best_iou":
            if bbox_xyxy is None:
                return max(frame_masks.values(), key=lambda m: int(m.sum()))
            best_mask = None
            best_iou = -1.0
            for candidate in frame_masks.values():
                bbox = mask_bbox_xyxy(candidate)
                if bbox is None:
                    continue
                iou = bbox_iou_xyxy(bbox, bbox_xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_mask = candidate
            if best_mask is not None:
                return best_mask
            return max(frame_masks.values(), key=lambda m: int(m.sum()))
        raise ValueError(f"Unsupported mask select policy: {policy}")

    def preload(self, samples: Sequence[object], preload_images: bool, preload_masks: bool) -> dict[str, Any]:
        image_bytes = 0
        mask_bytes = 0
        if preload_images:
            for sample in samples:
                if sample.image_path in self.image_cache:
                    continue
                image = self.load_image(sample.image_path)
                image_bytes += int(image.nbytes)
        if preload_masks:
            seen_masks: set[str] = set()
            for sample in samples:
                if sample.mask_path in seen_masks:
                    continue
                seen_masks.add(sample.mask_path)
                mask_map = self.mask_cache.get(sample.mask_path)
                if mask_map is None:
                    mask_map = load_mask_frame_dict(Path(sample.mask_path))
                    self.mask_cache[sample.mask_path] = mask_map
                    mask_bytes += int(sum(mask.nbytes for mask in mask_map.values()))
        return {
            "image_count": len(self.image_cache),
            "mask_file_count": len(self.mask_cache),
            "image_bytes": image_bytes,
            "mask_bytes": mask_bytes,
        }


def expand_box_xyxy(box: Sequence[float], image_shape: Sequence[int], scale: float) -> tuple[int, int, int, int]:
    h, w = int(image_shape[0]), int(image_shape[1])
    x1, y1, x2, y2 = [float(v) for v in box]
    box_w = max(1.0, float(x2 - x1))
    box_h = max(1.0, float(y2 - y1))
    pad_x = max(0.0, box_w * float(scale))
    pad_y = max(0.0, box_h * float(scale))
    out = (
        max(0, int(math.floor(x1 - pad_x))),
        max(0, int(math.floor(y1 - pad_y))),
        min(w, int(math.ceil(x2 + pad_x))),
        min(h, int(math.ceil(y2 + pad_y))),
    )
    x1i, y1i, x2i, y2i = out
    return x1i, y1i, max(x1i + 1, x2i), max(y1i + 1, y2i)


def prepare_detector_boxes(
    detector: SSDLiteDetector,
    store: RamImageMaskStore,
    samples: Sequence[object],
    detector_batch_desc: str,
    batch_size: int,
    score_threshold: float,
) -> dict[tuple[str, int], dict[str, Any]]:
    box_map: dict[tuple[str, int], dict[str, Any]] = {}
    unique_samples: list[object] = []
    seen: set[tuple[str, int]] = set()
    for sample in samples:
        key = (sample.video_name, int(sample.frame_idx))
        if key in seen:
            continue
        seen.add(key)
        unique_samples.append(sample)
    batch_size = max(int(batch_size), 1)
    progress = tqdm(total=len(unique_samples), desc=detector_batch_desc)
    for start in range(0, len(unique_samples), batch_size):
        chunk = unique_samples[start:start + batch_size]
        images = [store.load_image(sample.image_path) for sample in chunk]
        results = detector.detect_batch(images, score_threshold=score_threshold)
        for sample, (box, score) in zip(chunk, results):
            if box is None:
                continue
            key = (sample.video_name, int(sample.frame_idx))
            box_map[key] = {"box": box, "score": float(score)}
        progress.update(len(chunk))
    progress.close()
    return box_map


def filter_samples_with_detector_boxes(
    split_indices: Mapping[str, SplitIndex],
    detector_boxes: Mapping[tuple[str, int], dict[str, Any]],
    weak_sample_weight: float,
) -> dict[str, SplitIndex]:
    filtered: dict[str, SplitIndex] = {}
    for split in SPLIT_NAMES:
        split_index = split_indices[split]
        warnings = list(split_index.warnings)
        # Labeled samples use keypoint-derived bboxes (DLC-style), not detector boxes.
        labeled_samples = list(split_index.labeled_samples)
        weak_samples: list[WeakSample] = []
        if split == "train" and weak_sample_weight > 0.0:
            weak_samples = [
                sample for sample in split_index.weak_samples
                if (sample.video_name, sample.frame_idx) in detector_boxes
            ]
            for sample in split_index.weak_samples:
                if (sample.video_name, sample.frame_idx) not in detector_boxes:
                    warnings.append(f"{split}:{sample.video_name}:{sample.frame_idx}: detector box missing")
        filtered[split] = SplitIndex(
            split=split,
            videos=list(split_index.videos),
            labeled_samples=labeled_samples,
            weak_samples=weak_samples,
            warnings=warnings,
        )
    return filtered


class RTMPoseDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[object],
        store: RamImageMaskStore,
        detector_boxes: Mapping[tuple[str, int], dict[str, Any]],
        bodyparts: Sequence[str],
        skeleton: Sequence[Sequence[str]],
        left_right_symmetry: Sequence[Sequence[str]],
        input_size: tuple[int, int],
        image_mean: Sequence[float],
        image_std: Sequence[float],
        crop_expand_scale: float,
        bbox_margin: float,
        train_aug_cfg: Mapping[str, Any],
        crop_cfg: Mapping[str, Any],
        train_mode: bool,
        include_weak: bool,
        use_masks: bool,
        mask_select_policy: str,
        weak_mask_iou_thresh: float,
    ) -> None:
        self.samples = list(samples)
        self.store = store
        self.detector_boxes = detector_boxes
        self.bodyparts = list(bodyparts)
        self.skeleton = [list(pair) for pair in skeleton]
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.image_mean = np.asarray(image_mean, dtype=np.float32)
        self.image_std = np.asarray(image_std, dtype=np.float32)
        self.crop_expand_scale = float(crop_expand_scale)
        self.bbox_margin = float(bbox_margin)
        self.train_aug_cfg = dict(train_aug_cfg)
        self.crop_cfg = dict(crop_cfg)
        self.train_mode = bool(train_mode)
        self.include_weak = bool(include_weak)
        self.use_masks = bool(use_masks)
        self.mask_select_policy = str(mask_select_policy)
        self.weak_mask_iou_thresh = float(weak_mask_iou_thresh)
        name_to_idx = {name: idx for idx, name in enumerate(self.bodyparts)}
        self.flip_pairs = [
            (name_to_idx[left], name_to_idx[right])
            for left, right in left_right_symmetry
            if left in name_to_idx and right in name_to_idx
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = self.store.load_image(sample.image_path)

        keypoints = np.zeros((len(self.bodyparts), 3), dtype=np.float32)
        if hasattr(sample, "keypoints"):
            keypoints[:, :2] = np.asarray(sample.keypoints, dtype=np.float32)
            keypoints[:, 2] = np.asarray(sample.visibility, dtype=np.float32)

        # BBox source policy:
        # - labeled samples: DLC-style keypoint-derived bbox
        # - weak samples: detector bbox
        if hasattr(sample, "keypoints"):
            bbox_xyxy = bbox_from_visible_keypoints(
                keypoints,
                image.shape[:2],
                margin=self.bbox_margin,
            )
            if bbox_xyxy is None:
                raise ValueError(f"No visible keypoints for labeled sample {sample.video_name}:{sample.frame_idx}")
        else:
            det = self.detector_boxes.get((sample.video_name, sample.frame_idx))
            if det is None:
                raise KeyError(f"Missing detector box for weak sample {sample.video_name}:{sample.frame_idx}")
            det_box = np.asarray(det["box"], dtype=np.float32)
            x1, y1, x2, y2 = expand_box_xyxy(det_box, image.shape[:2], self.crop_expand_scale)
            bbox_xyxy = np.asarray([x1, y1, x2, y2], dtype=np.float32)

        mask = None
        mask_det_iou = -1.0
        if self.use_masks:
            try:
                policy = "first"
                bbox_for_policy = None
                if not hasattr(sample, "keypoints"):
                    policy = self.mask_select_policy
                    bbox_for_policy = bbox_xyxy
                mask = self.store.load_mask(
                    sample.mask_path,
                    sample.frame_idx,
                    policy=policy,
                    bbox_xyxy=bbox_for_policy,
                )
            except KeyError:
                mask = None
        if mask is not None and not hasattr(sample, "keypoints"):
            mask_box = mask_bbox_xyxy(mask)
            if mask_box is not None:
                mask_det_iou = bbox_iou_xyxy(mask_box, bbox_xyxy)
                if self.weak_mask_iou_thresh > 0.0 and mask_det_iou < self.weak_mask_iou_thresh:
                    mask = None
                    mask_det_iou = -1.0

        # DLC-like order: full-image augmentation first, then top-down crop.
        if self.train_mode:
            hflip_cfg = self.train_aug_cfg.get("hflip", {})
            if isinstance(hflip_cfg, bool):
                hflip_p = 0.5 if hflip_cfg else 0.0
            elif isinstance(hflip_cfg, Mapping):
                hflip_p = float(hflip_cfg.get("p", 0.5))
            else:
                hflip_p = 0.0
            if random.random() < hflip_p:
                image = np.ascontiguousarray(image[:, ::-1])
                if mask is not None:
                    mask = np.ascontiguousarray(mask[:, ::-1])
                keypoints = horizontal_flip_keypoints(keypoints, image.shape[1], self.flip_pairs)
                bbox_xyxy = np.asarray(
                    [
                        (image.shape[1] - 1) - bbox_xyxy[2],
                        bbox_xyxy[1],
                        (image.shape[1] - 1) - bbox_xyxy[0],
                        bbox_xyxy[3],
                    ],
                    dtype=np.float32,
                )

            affine_cfg = self.train_aug_cfg.get("affine", {})
            affine_p = float(affine_cfg.get("p", 0.5)) if isinstance(affine_cfg, Mapping) else 0.0
            if random.random() < affine_p:
                rot = float(affine_cfg.get("rotation", 30.0))
                angle = random.uniform(-rot, rot)
                scale_cfg = affine_cfg.get("scaling", [1.0, 1.0])
                if isinstance(scale_cfg, Sequence) and len(scale_cfg) >= 2:
                    scale = random.uniform(float(scale_cfg[0]), float(scale_cfg[1]))
                else:
                    scale = 1.0
                trans_cfg = affine_cfg.get("translation", 0)
                if isinstance(trans_cfg, Sequence) and len(trans_cfg) >= 2:
                    tx = random.uniform(float(trans_cfg[0]), float(trans_cfg[1]))
                    ty = random.uniform(float(trans_cfg[0]), float(trans_cfg[1]))
                else:
                    tr = float(trans_cfg)
                    tx = random.uniform(-tr, tr)
                    ty = random.uniform(-tr, tr)
                center = (image.shape[1] * 0.5, image.shape[0] * 0.5)
                matrix = cv2.getRotationMatrix2D(center, angle, scale)
                matrix[0, 2] += tx
                matrix[1, 2] += ty
                image = cv2.warpAffine(
                    image,
                    matrix,
                    (image.shape[1], image.shape[0]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                mask = cv2.warpAffine(
                    mask.astype(np.uint8),
                    matrix,
                    (mask.shape[1], mask.shape[0]),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                ) if mask is not None else None
                keypoints = rotate_points(keypoints, matrix)
                bbox_xyxy = transform_box_xyxy(bbox_xyxy, matrix, image.shape[1], image.shape[0])

            rb_cfg = self.train_aug_cfg.get("random_bbox_transform", {})
            if isinstance(rb_cfg, Mapping):
                bbox_xyxy = apply_random_bbox_transform(bbox_xyxy, image.shape[:2], rb_cfg)

            if bool(self.train_aug_cfg.get("motion_blur", False)):
                image = apply_motion_blur(image, p=0.5)

            noise_val = self.train_aug_cfg.get("gaussian_noise", False)
            if noise_val:
                noise_std = float(noise_val) if isinstance(noise_val, (int, float)) else 12.75
                image = apply_gaussian_noise(image, noise_std=noise_std, p=0.5)

        keypoints = clamp_visibility_to_window(keypoints, image.shape[1], image.shape[0])
        bbox_xyxy[0] = np.clip(bbox_xyxy[0], 0.0, float(image.shape[1] - 1))
        bbox_xyxy[1] = np.clip(bbox_xyxy[1], 0.0, float(image.shape[0] - 1))
        bbox_xyxy[2] = np.clip(bbox_xyxy[2], 1.0, float(image.shape[1]))
        bbox_xyxy[3] = np.clip(bbox_xyxy[3], 1.0, float(image.shape[0]))

        crop_margin = int(self.crop_cfg.get("margin", 0))
        crop_with_context = bool(self.crop_cfg.get("crop_with_context", True))
        bbox_xywh = xyxy_to_xywh(bbox_xyxy)
        dst_w, dst_h = self.input_size
        has_mask = mask is not None
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        resized_image, resized_mask, offsets, scales = top_down_crop_pair(
            image=image,
            mask=mask,
            bbox_xywh=bbox_xywh,
            output_size=(dst_w, dst_h),
            margin=crop_margin,
            crop_with_context=crop_with_context,
        )
        debug_image = resized_image.copy()

        visible = keypoints[:, 2] > 0
        if visible.any():
            keypoints[visible, 0] = (keypoints[visible, 0] - offsets[0]) / scales[0]
            keypoints[visible, 1] = (keypoints[visible, 1] - offsets[1]) / scales[1]
        keypoints = clamp_visibility_to_window(keypoints, dst_w, dst_h)

        roi_box = np.asarray(
            [
                (bbox_xyxy[0] - offsets[0]) / scales[0],
                (bbox_xyxy[1] - offsets[1]) / scales[1],
                (bbox_xyxy[2] - offsets[0]) / scales[0],
                (bbox_xyxy[3] - offsets[1]) / scales[1],
            ],
            dtype=np.float32,
        )
        roi_box[0] = np.clip(roi_box[0], 0.0, float(dst_w))
        roi_box[2] = np.clip(roi_box[2], 0.0, float(dst_w))
        roi_box[1] = np.clip(roi_box[1], 0.0, float(dst_h))
        roi_box[3] = np.clip(roi_box[3], 0.0, float(dst_h))
        crop_box = np.asarray(
            [
                offsets[0],
                offsets[1],
                offsets[0] + scales[0] * float(dst_w),
                offsets[1] + scales[1] * float(dst_h),
            ],
            dtype=np.float32,
        )

        image_float = resized_image.astype(np.float32) / 255.0
        image_norm = (image_float - self.image_mean[None, None, :]) / self.image_std[None, None, :]
        image_tensor = torch.from_numpy(np.ascontiguousarray(image_norm.transpose(2, 0, 1)))
        has_mask_value = 1.0 if has_mask else 0.0

        return {
            "image": image_tensor.float(),
            "keypoints": torch.from_numpy(keypoints.astype(np.float32)),
            "mask": torch.from_numpy(resized_mask[None, ...].astype(np.float32)),
            "has_mask": has_mask_value,
            "is_labeled": 1.0 if hasattr(sample, "keypoints") else 0.0,
            "video_name": sample.video_name,
            "frame_idx": int(sample.frame_idx),
            "source_name": sample.source_name,
            "crop_box": np.asarray(crop_box, dtype=np.float32),
            "roi_box_in_crop": roi_box.astype(np.float32),
            "orig_size": np.asarray([image.shape[1], image.shape[0]], dtype=np.float32),
            "mask_det_iou": float(mask_det_iou),
            "debug_image": torch.from_numpy(np.ascontiguousarray(debug_image.transpose(2, 0, 1).astype(np.float32) / 255.0)),
        }


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_pose_batch,
        "drop_last": drop_last,
    }
    if shuffle:
        kwargs["sampler"] = RandomSampler(dataset)
    if workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def expected_keypoints_from_outputs(outputs: Mapping[str, torch.Tensor], split_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    probs_x, probs_y = simcc_probabilities(outputs)
    x_axis = torch.arange(probs_x.shape[-1], device=probs_x.device, dtype=probs_x.dtype)
    y_axis = torch.arange(probs_y.shape[-1], device=probs_y.device, dtype=probs_y.dtype)
    exp_x = (probs_x * x_axis[None, None, :]).sum(dim=-1) / float(split_ratio)
    exp_y = (probs_y * y_axis[None, None, :]).sum(dim=-1) / float(split_ratio)
    scores = torch.sqrt(probs_x.max(dim=-1).values * probs_y.max(dim=-1).values).clamp_min(0.0)
    return torch.stack([exp_x, exp_y], dim=-1), scores


def expected_points_mask_loss(
    outputs: Mapping[str, torch.Tensor],
    mask: torch.Tensor,
    split_ratio: float,
    visibility: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    probs_x, probs_y = simcc_probabilities(outputs)
    mask_resized = F.interpolate(
        mask.float(),
        size=(probs_y.shape[-1], probs_x.shape[-1]),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1).clamp(0.0, 1.0)
    inside_mass = torch.einsum("bky,byx,bkx->bk", probs_y, mask_resized, probs_x)
    loss = 1.0 - inside_mass
    weight_tensor: Optional[torch.Tensor] = None
    if visibility is not None:
        weight_tensor = visibility.float().clamp_min(0.0)
    if weights is not None:
        weight_tensor = weights.float().clamp_min(0.0) if weight_tensor is None else weight_tensor * weights.float().clamp_min(0.0)
    if weight_tensor is not None:
        denom = weight_tensor.sum().clamp_min(1.0)
        return (loss * weight_tensor).sum() / denom
    return loss.mean()


def ring_masks(mask_resized: torch.Tensor, radius: int) -> tuple[torch.Tensor, torch.Tensor]:
    if radius <= 0:
        return mask_resized, mask_resized
    kernel = 2 * int(radius) + 1
    outer = F.max_pool2d(mask_resized.unsqueeze(1), kernel_size=kernel, stride=1, padding=radius).squeeze(1).clamp(0.0, 1.0)
    inv = (1.0 - mask_resized).unsqueeze(1)
    core = (1.0 - F.max_pool2d(inv, kernel_size=kernel, stride=1, padding=radius)).squeeze(1).clamp(0.0, 1.0)
    return core, outer


def weighted_loss_reduce(loss: torch.Tensor, visibility: Optional[torch.Tensor], weights: Optional[torch.Tensor]) -> torch.Tensor:
    weight_tensor: Optional[torch.Tensor] = None
    if visibility is not None:
        weight_tensor = visibility.float().clamp_min(0.0)
    if weights is not None:
        sample_weights = weights.float().clamp_min(0.0)
        weight_tensor = sample_weights if weight_tensor is None else weight_tensor * sample_weights
    if weight_tensor is None:
        return loss.mean()
    denom = weight_tensor.sum().clamp_min(1.0)
    return (loss * weight_tensor).sum() / denom


def mask_ring_loss(
    outputs: Mapping[str, torch.Tensor],
    mask: torch.Tensor,
    ring_radius: int,
    outside_weight: float,
    mass_floor: float,
    visibility: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    probs_x, probs_y = simcc_probabilities(outputs)
    mask_resized = F.interpolate(
        mask.float(),
        size=(probs_y.shape[-1], probs_x.shape[-1]),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1).clamp(0.0, 1.0)
    core, outer = ring_masks(mask_resized, ring_radius)
    inside_core = torch.einsum("bky,byx,bkx->bk", probs_y, core, probs_x)
    outside_outer = torch.einsum("bky,byx,bkx->bk", probs_y, (1.0 - outer), probs_x)
    loss = (1.0 - inside_core) + float(outside_weight) * outside_outer
    if mass_floor > 0.0:
        loss = loss + torch.relu(float(mass_floor) - inside_core)
    return weighted_loss_reduce(loss, visibility, weights)


def mask_trimmed_outside_loss(
    outputs: Mapping[str, torch.Tensor],
    mask: torch.Tensor,
    outside_weight: float,
    outside_trim: float,
    mass_floor: float,
    visibility: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    probs_x, probs_y = simcc_probabilities(outputs)
    mask_resized = F.interpolate(
        mask.float(),
        size=(probs_y.shape[-1], probs_x.shape[-1]),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1).clamp(0.0, 1.0)
    inside_mass = torch.einsum("bky,byx,bkx->bk", probs_y, mask_resized, probs_x)
    outside_mass = 1.0 - inside_mass
    outside_penalty = torch.relu(outside_mass - float(outside_trim))
    loss = (1.0 - inside_mass) + float(outside_weight) * outside_penalty
    if mass_floor > 0.0:
        loss = loss + torch.relu(float(mass_floor) - inside_mass)
    return weighted_loss_reduce(loss, visibility, weights)


def simcc_peak_confidence(outputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    probs_x, probs_y = simcc_probabilities(outputs)
    conf_x = probs_x.max(dim=-1).values
    conf_y = probs_y.max(dim=-1).values
    return torch.sqrt(conf_x * conf_y).clamp_min(0.0)


def weak_quality_weights(
    outputs: Mapping[str, torch.Tensor],
    mask_det_iou: torch.Tensor,
    iou_t0: float,
    iou_t1: float,
    iou_power: float,
    conf_power: float,
) -> torch.Tensor:
    conf = simcc_peak_confidence(outputs).clamp(0.0, 1.0).pow(float(conf_power))
    denom = max(1e-6, float(iou_t1) - float(iou_t0))
    iou_w = ((mask_det_iou.float() - float(iou_t0)) / denom).clamp(0.0, 1.0).pow(float(iou_power)).unsqueeze(-1)
    return conf * iou_w


def mask_alignment_loss(
    outputs: Mapping[str, torch.Tensor],
    mask: torch.Tensor,
    split_ratio: float,
    visibility: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    exp_points, _ = expected_keypoints_from_outputs(outputs, split_ratio)
    probs_x, probs_y = simcc_probabilities(outputs)
    mask_resized = F.interpolate(
        mask.float(),
        size=(probs_y.shape[-1], probs_x.shape[-1]),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1).clamp(0.0, 1.0)
    scale = torch.tensor([probs_x.shape[-1], probs_y.shape[-1]], device=exp_points.device, dtype=exp_points.dtype)
    coords = (exp_points * float(split_ratio)) / (scale - 1.0).clamp_min(1.0)
    grid = coords * 2.0 - 1.0
    grid = grid[..., [0, 1]].unsqueeze(2)
    sampled = F.grid_sample(mask_resized.unsqueeze(1), grid, align_corners=True, mode="bilinear").squeeze(1).squeeze(-1)
    loss = 1.0 - sampled
    weight_tensor: Optional[torch.Tensor] = None
    if visibility is not None:
        weight_tensor = visibility.float().clamp_min(0.0)
    if weights is not None:
        weight_tensor = weights.float().clamp_min(0.0) if weight_tensor is None else weight_tensor * weights.float().clamp_min(0.0)
    if weight_tensor is not None:
        denom = weight_tensor.sum().clamp_min(1.0)
        return (loss * weight_tensor).sum() / denom
    return loss.mean()


def mask_variance_penalty(
    outputs: Mapping[str, torch.Tensor],
    mask: torch.Tensor,
) -> torch.Tensor:
    probs_x, probs_y = simcc_probabilities(outputs)
    device = probs_x.device
    x_axis = torch.arange(probs_x.shape[-1], device=device, dtype=probs_x.dtype)
    y_axis = torch.arange(probs_y.shape[-1], device=device, dtype=probs_y.dtype)
    exp_x = torch.einsum("bkx,x->bk", probs_x, x_axis)
    exp_y = torch.einsum("bky,y->bk", probs_y, y_axis)
    var_x = (probs_x * (x_axis[None, None, :] - exp_x.unsqueeze(-1)) ** 2).sum(dim=-1)
    var_y = (probs_y * (y_axis[None, None, :] - exp_y.unsqueeze(-1)) ** 2).sum(dim=-1)
    mask_resized = F.interpolate(
        mask.float(),
        size=(probs_y.shape[-1], probs_x.shape[-1]),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1).clamp(0.0, 1.0)
    inside_mass = torch.einsum("bky,byx,bkx->bk", probs_y, mask_resized, probs_x)
    weight = inside_mass.detach().clamp(0.0, 1.0)
    return ((var_x + var_y) * weight).mean()


def entropy_regularization(outputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    probs_x, probs_y = simcc_probabilities(outputs)
    ent_x = -(probs_x.clamp_min(1e-8) * probs_x.clamp_min(1e-8).log()).sum(dim=-1)
    ent_y = -(probs_y.clamp_min(1e-8) * probs_y.clamp_min(1e-8).log()).sum(dim=-1)
    ent_x = ent_x / math.log(max(2, probs_x.shape[-1]))
    ent_y = ent_y / math.log(max(2, probs_y.shape[-1]))
    return 0.5 * (ent_x.mean() + ent_y.mean())


def visibility_loss(
    outputs: Mapping[str, torch.Tensor],
    keypoints: torch.Tensor,
) -> torch.Tensor:
    target = keypoints[..., 2].float().clamp(0.0, 1.0)
    logits = outputs["visibility_logits"]
    return F.binary_cross_entropy_with_logits(logits, target)


def configure_backbone_freeze(model: StandaloneRTMPose, freeze_backbone: bool) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = not freeze_backbone


def supervised_total_loss(supervised_losses: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if "total_loss" in supervised_losses:
        return supervised_losses["total_loss"]
    if "loss" in supervised_losses:
        return supervised_losses["loss"]
    raise KeyError(f"Supervised loss dict is missing total loss key: {sorted(supervised_losses)}")


def compute_pose_metrics(
    pred_keypoints: torch.Tensor,
    gt_keypoints: torch.Tensor,
) -> dict[str, float]:
    visibility = gt_keypoints[..., 2] > 0.5
    if not visibility.any():
        return {
            "visible_count": 0.0,
            "sum_euclidean_error": 0.0,
            "sum_squared_coord_error": 0.0,
            "mean_score": 0.0,
            "score_count": 0.0,
        }
    diffs = pred_keypoints[..., :2] - gt_keypoints[..., :2]
    diffs = diffs[visibility]
    euclidean = torch.norm(diffs, dim=-1)
    return {
        "visible_count": float(visibility.sum().item()),
        "sum_euclidean_error": float(euclidean.sum().item()),
        "sum_squared_coord_error": float((diffs ** 2).sum().item()),
        "mean_score": float(pred_keypoints[..., 2][visibility].sum().item()),
        "score_count": float(visibility.sum().item()),
    }


def compute_visibility_metrics(
    pred_visibility: torch.Tensor,
    gt_keypoints: torch.Tensor,
) -> dict[str, float]:
    target_visibility = gt_keypoints[..., 2].float().clamp(0.0, 1.0)
    pred_binary = (pred_visibility >= 0.5).float()
    total = float(target_visibility.numel())
    correct = float((pred_binary == target_visibility).float().sum().item())
    return {
        "visibility_accuracy": correct / max(total, 1.0),
        "mean_visibility_prob": float(pred_visibility.mean().item()),
    }


@torch.no_grad()
def evaluate_pose_model(
    model: StandaloneRTMPose,
    dataloader: DataLoader,
    device: torch.device,
    lambda_pose: float,
    lambda_mask: float,
    lambda_visibility: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_visible = 0.0
    total_error = 0.0
    total_squared_error = 0.0
    total_score = 0.0
    total_score_count = 0.0
    total_visibility_acc = 0.0
    total_visibility_prob = 0.0
    euclidean_errors: list[np.ndarray] = []
    for batch in dataloader:
        image = batch["image"].to(device)
        keypoints = batch["keypoints"].to(device)
        mask = batch["mask"].to(device)
        has_mask = batch["has_mask"].to(device)
        outputs = model(image)
        supervised = model.compute_supervised_loss(outputs, keypoints)
        supervised_total = supervised_total_loss(supervised)
        split_ratio = float(model.cfg["model"]["heads"]["bodypart"].get("simcc_split_ratio", 2.0))
        if float(has_mask.max().item()) > 0.5:
            mask_loss = expected_points_mask_loss(outputs, mask, split_ratio, keypoints[..., 2])
        else:
            mask_loss = outputs["x"].sum() * 0.0
        vis_loss = visibility_loss(outputs, keypoints)
        total = lambda_pose * supervised_total + lambda_mask * mask_loss + lambda_visibility * vis_loss
        total_loss += float(total.item())
        total_batches += 1
        pred_points, pred_scores = decode_keypoints_with_predictor(model, outputs)
        pred_visibility = visibility_probabilities(outputs)
        pred_points = pred_points.to(pred_visibility.device)
        pred_scores = pred_scores.to(pred_visibility.device)
        pred = torch.cat([pred_points, pred_visibility.unsqueeze(-1)], dim=-1)
        metrics = compute_pose_metrics(pred, keypoints)
        visibility_metrics = compute_visibility_metrics(pred_visibility, keypoints)
        visibility = keypoints[..., 2] > 0.5
        if visibility.any():
            diffs = pred[..., :2] - keypoints[..., :2]
            euclidean = torch.norm(diffs[visibility], dim=-1)
            euclidean_errors.append(euclidean.detach().cpu().numpy())
        total_visible += metrics["visible_count"]
        total_error += metrics["sum_euclidean_error"]
        total_squared_error += metrics["sum_squared_coord_error"]
        total_score += metrics["mean_score"]
        total_score_count += metrics["score_count"]
        total_visibility_acc += visibility_metrics["visibility_accuracy"]
        total_visibility_prob += visibility_metrics["mean_visibility_prob"]
    rmse_90 = 0.0
    if euclidean_errors:
        rmse_90 = float(np.quantile(np.concatenate(euclidean_errors, axis=0), 0.9))
    return {
        "loss": total_loss / max(1, total_batches),
        "mean_pixel_error": total_error / max(1.0, total_visible),
        "rmse_unfiltered": math.sqrt(total_squared_error / max(1.0, 2.0 * total_visible)),
        "rmse_90": rmse_90,
        "mean_keypoint_score": total_score / max(1.0, total_score_count),
        "visibility_accuracy": total_visibility_acc / max(1, total_batches),
        "mean_visibility_prob": total_visibility_prob / max(1, total_batches),
    }


def train_one_epoch(
    model: StandaloneRTMPose,
    labeled_loader: DataLoader,
    weak_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    lambda_pose: float,
    lambda_mask: float,
    lambda_visibility: float,
    lambda_entropy: float,
    weak_sample_weight: float,
    mask_on_labeled: bool,
    mask_loss_mode: str,
    mask_variance_weight: float,
    mask_conf_threshold: float,
    weak_use_quality_weight: bool,
    weak_use_visibility_gate: bool,
    weak_iou_weight_t0: float,
    weak_iou_weight_t1: float,
    weak_iou_weight_power: float,
    weak_conf_weight_power: float,
    mask_ring_radius: int,
    mask_outside_weight: float,
    mask_outside_trim: float,
    mask_mass_floor: float,
    use_amp: bool,
    log_interval: int,
    freeze_backbone: bool,
) -> dict[str, float]:
    model.train()
    if freeze_backbone:
        model.backbone.eval()
    split_ratio = float(model.cfg["model"]["heads"]["bodypart"].get("simcc_split_ratio", 2.0))
    # Iterate through all labeled batches each epoch. Weak batches are optional and
    # consumed at most once per epoch to avoid repeating weak supervision.
    labeled_iter = iter(labeled_loader)
    weak_iter = iter(weak_loader) if weak_loader is not None else None
    steps = max(len(labeled_loader), 1)
    totals = {
        "loss": 0.0,
        "supervised": 0.0,
        "mask": 0.0,
        "visibility": 0.0,
        "entropy": 0.0,
        "cpu": 0.0,
        "h2d": 0.0,
        "gpu_fwd": 0.0,
        "gpu_bwd": 0.0,
    }

    for step_idx in range(steps):
        cpu_start = time.perf_counter()
        labeled_batch = next(labeled_iter)
        weak_batch = None
        if weak_iter is not None:
            try:
                weak_batch = next(weak_iter)
            except StopIteration:
                weak_batch = None
        cpu_time = time.perf_counter() - cpu_start

        optimizer.zero_grad(set_to_none=True)
        total_tensor = None
        supervised_log = 0.0
        mask_log = 0.0
        entropy_log = 0.0
        visibility_log = 0.0
        h2d_time = 0.0
        fwd_time = 0.0

        for batch in (labeled_batch, weak_batch):
            if batch is None:
                continue
            maybe_cuda_sync(device)
            h2d_start = time.perf_counter()
            image = batch["image"].to(device, non_blocking=True)
            keypoints = batch["keypoints"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            has_mask = batch["has_mask"].to(device, non_blocking=True)
            mask_det_iou = batch.get("mask_det_iou")
            if mask_det_iou is not None:
                mask_det_iou = mask_det_iou.to(device, non_blocking=True)
            maybe_cuda_sync(device)
            h2d_time += time.perf_counter() - h2d_start

            maybe_cuda_sync(device)
            fwd_start = time.perf_counter()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(image)
                if float(batch["is_labeled"].max().item()) > 0.5:
                    supervised = model.compute_supervised_loss(outputs, keypoints)
                    supervised_total = supervised_total_loss(supervised)
                    if mask_on_labeled and float(has_mask.max().item()) > 0.5:
                        weights = None
                        if mask_conf_threshold > 0.0:
                            conf = simcc_peak_confidence(outputs)
                            weights = (conf >= mask_conf_threshold).float()
                        if mask_loss_mode == "alignment":
                            mask_loss = mask_alignment_loss(outputs, mask, split_ratio, keypoints[..., 2], weights)
                        elif mask_loss_mode == "ring":
                            mask_loss = mask_ring_loss(
                                outputs,
                                mask,
                                ring_radius=int(mask_ring_radius),
                                outside_weight=float(mask_outside_weight),
                                mass_floor=float(mask_mass_floor),
                                visibility=keypoints[..., 2],
                                weights=weights,
                            )
                        elif mask_loss_mode == "trimmed":
                            mask_loss = mask_trimmed_outside_loss(
                                outputs,
                                mask,
                                outside_weight=float(mask_outside_weight),
                                outside_trim=float(mask_outside_trim),
                                mass_floor=float(mask_mass_floor),
                                visibility=keypoints[..., 2],
                                weights=weights,
                            )
                        else:
                            mask_loss = expected_points_mask_loss(outputs, mask, split_ratio, keypoints[..., 2], weights)
                        if mask_loss_mode == "variance" and mask_variance_weight > 0.0:
                            mask_loss = mask_loss + mask_variance_weight * mask_variance_penalty(outputs, mask)
                    else:
                        mask_loss = outputs["x"].sum() * 0.0
                    vis_loss = visibility_loss(outputs, keypoints)
                    entropy_loss = outputs["x"].sum() * 0.0
                    batch_total = (
                        lambda_pose * supervised_total
                        + lambda_mask * mask_loss
                        + lambda_visibility * vis_loss
                    )
                    supervised_log += float(supervised_total.item())
                    mask_log += float(mask_loss.item())
                    visibility_log += float(vis_loss.item())
                else:
                    if float(has_mask.max().item()) > 0.5:
                        weights = None
                        if mask_conf_threshold > 0.0:
                            conf = simcc_peak_confidence(outputs)
                            weights = (conf >= mask_conf_threshold).float()
                        if weak_use_quality_weight and mask_det_iou is not None:
                            q_weights = weak_quality_weights(
                                outputs,
                                mask_det_iou,
                                iou_t0=float(weak_iou_weight_t0),
                                iou_t1=float(weak_iou_weight_t1),
                                iou_power=float(weak_iou_weight_power),
                                conf_power=float(weak_conf_weight_power),
                            )
                            weights = q_weights if weights is None else weights * q_weights
                        if weak_use_visibility_gate:
                            vis_w = visibility_probabilities(outputs).detach().clamp(0.0, 1.0)
                            weights = vis_w if weights is None else weights * vis_w
                        if mask_loss_mode == "alignment":
                            mask_loss = mask_alignment_loss(outputs, mask, split_ratio, None, weights)
                        elif mask_loss_mode == "ring":
                            mask_loss = mask_ring_loss(
                                outputs,
                                mask,
                                ring_radius=int(mask_ring_radius),
                                outside_weight=float(mask_outside_weight),
                                mass_floor=float(mask_mass_floor),
                                visibility=None,
                                weights=weights,
                            )
                        elif mask_loss_mode == "trimmed":
                            mask_loss = mask_trimmed_outside_loss(
                                outputs,
                                mask,
                                outside_weight=float(mask_outside_weight),
                                outside_trim=float(mask_outside_trim),
                                mass_floor=float(mask_mass_floor),
                                visibility=None,
                                weights=weights,
                            )
                        else:
                            mask_loss = expected_points_mask_loss(outputs, mask, split_ratio, None, weights)
                        if mask_loss_mode == "variance" and mask_variance_weight > 0.0:
                            mask_loss = mask_loss + mask_variance_weight * mask_variance_penalty(outputs, mask)
                        entropy_loss = entropy_regularization(outputs)
                        batch_total = weak_sample_weight * (lambda_mask * mask_loss + lambda_entropy * entropy_loss)
                    else:
                        mask_loss = outputs["x"].sum() * 0.0
                        entropy_loss = outputs["x"].sum() * 0.0
                        batch_total = outputs["x"].sum() * 0.0
                    mask_log += float(mask_loss.item())
                    entropy_log += float(entropy_loss.item())
                total_tensor = batch_total if total_tensor is None else total_tensor + batch_total
            maybe_cuda_sync(device)
            fwd_time += time.perf_counter() - fwd_start

        maybe_cuda_sync(device)
        bwd_start = time.perf_counter()
        scaler.scale(total_tensor).backward()
        scaler.step(optimizer)
        scaler.update()
        maybe_cuda_sync(device)
        bwd_time = time.perf_counter() - bwd_start

        totals["loss"] += float(total_tensor.item())
        totals["supervised"] += supervised_log
        totals["mask"] += mask_log
        totals["visibility"] += visibility_log
        totals["entropy"] += entropy_log
        totals["cpu"] += cpu_time
        totals["h2d"] += h2d_time
        totals["gpu_fwd"] += fwd_time
        totals["gpu_bwd"] += bwd_time

    for key in totals:
        totals[key] /= float(steps)
    return totals


def build_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found for optimizer.")
    name = str(args.optimizer).lower()
    if name == "adamw":
        return torch.optim.AdamW(trainable_params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    if name == "adam":
        return torch.optim.Adam(trainable_params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def build_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer):
    name = str(args.scheduler).lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs)))
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(args.scheduler_step_size),
            gamma=float(args.scheduler_gamma),
        )
    if name in {"none", ""}:
        return None
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")


def resolve_selection_score(metrics: Mapping[str, float], metric_name: str) -> float:
    if metric_name not in metrics:
        raise ValueError(f"Selection metric {metric_name!r} not available in metrics: {sorted(metrics)}")
    return float(metrics[metric_name])


def metric_is_better(candidate: float, best: float, mode: str) -> bool:
    if mode == "min":
        return candidate < best
    if mode == "max":
        return candidate > best
    raise ValueError(f"Unsupported selection_mode={mode!r}")


def select_samples_for_split(split_indices: Mapping[str, SplitIndex], split: str, video_name: str | None) -> list[LabeledSample]:
    samples = split_indices[split].labeled_samples
    if video_name is None:
        return samples
    return [sample for sample in samples if sample.video_name == video_name]


def export_best_epoch_files(run_dir: Path, best_summary: Mapping[str, Any]) -> None:
    write_json(run_dir / "best_epoch.json", make_serializable(dict(best_summary)))
    write_text(
        run_dir / "best_epoch.txt",
        "\n".join(f"{key}: {value}" for key, value in best_summary.items()) + "\n",
    )


def split_eval_payload(metrics: Mapping[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    metric_payload = {
        key: float(value)
        for key, value in metrics.items()
        if key != "loss"
    }
    loss_payload = {"loss": float(metrics.get("loss", float("nan")))}
    return loss_payload, metric_payload


def serialize_detector_boxes(
    detector_boxes: Mapping[tuple[str, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "video_name": video_name,
            "frame_idx": int(frame_idx),
            "box": [float(v) for v in payload["box"]],
            "score": float(payload["score"]),
        }
        for (video_name, frame_idx), payload in sorted(detector_boxes.items())
    ]


def deserialize_detector_boxes(payload: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    detector_boxes: dict[tuple[str, int], dict[str, Any]] = {}
    for item in payload:
        video_name = str(item["video_name"])
        frame_idx = int(item["frame_idx"])
        detector_boxes[(video_name, frame_idx)] = {
            "box": [float(v) for v in item["box"]],
            "score": float(item["score"]),
        }
    return detector_boxes


def build_prepare_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "project_config": str(Path(args.project_config).resolve()),
        "labels_root": str(Path(args.labels_root).resolve()),
        "labeled_frames_root": str(Path(args.labeled_frames_root).resolve()),
        "frames_root": str(Path(args.frames_root).resolve()),
        "masks_root": str(Path(args.masks_root).resolve()),
        "detector_checkpoint": str(Path(args.detector_checkpoint).resolve()),
        "detector_model_name": str(args.detector_model_name),
        "detector_score_threshold": float(args.detector_score_threshold),
    }


def validate_prepare_metadata(args: argparse.Namespace, metadata: Mapping[str, Any]) -> None:
    expected = build_prepare_metadata(args)
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        actual_value = metadata.get(key)
        if actual_value != expected_value:
            mismatches.append(f"{key}: expected {expected_value!r}, found {actual_value!r}")
    if mismatches:
        raise RuntimeError(
            "Prepared RTMPose data does not match the current configuration. "
            "Run `prepare` again.\n" + "\n".join(mismatches)
        )


def load_prepared_training_components(
    args: argparse.Namespace,
    split_indices: Mapping[str, SplitIndex],
    prepare_dir: Path | None = None,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, SplitIndex], dict[str, Any], Path]:
    if prepare_dir is None:
        prepare_dir = build_prepare_dir(args.output_root)
    metadata_path = prepare_dir / "prepare_meta.json"
    detector_boxes_path = prepare_dir / "detector_boxes.json"
    detector_stats_path = prepare_dir / "detector_stats.json"
    if not metadata_path.is_file() or not detector_boxes_path.is_file() or not detector_stats_path.is_file():
        raise RuntimeError(
            f"Missing prepared RTMPose data in {prepare_dir}. "
            "Run `conda run -n track python scripts/keypoint_RTMPose.py prepare` first."
        )
    metadata = load_yaml_file(metadata_path)
    validate_prepare_metadata(args, metadata)
    detector_boxes = deserialize_detector_boxes(load_yaml_file(detector_boxes_path).get("detector_boxes", []))
    detector_stats = load_yaml_file(detector_stats_path)
    filtered_split_indices = filter_samples_with_detector_boxes(
        split_indices,
        detector_boxes,
        weak_sample_weight=float(getattr(args, "weak_sample_weight", 1.0)),
    )
    return detector_boxes, filtered_split_indices, detector_stats, prepare_dir


def write_prepare_bundle(
    prepare_dir: Path,
    args: argparse.Namespace,
    detector_boxes: Mapping[tuple[str, int], Mapping[str, Any]],
    detector_stats: Mapping[str, Any],
    filtered_indices: Mapping[str, SplitIndex],
    readme_context: str,
) -> None:
    if prepare_dir.exists():
        shutil.rmtree(prepare_dir)
    prepare_dir.mkdir(parents=True, exist_ok=True)
    write_json(prepare_dir / "prepare_meta.json", build_prepare_metadata(args))
    write_json(prepare_dir / "detector_boxes.json", {"detector_boxes": serialize_detector_boxes(detector_boxes)})
    write_json(prepare_dir / "detector_stats.json", make_serializable(detector_stats))
    write_json(prepare_dir / "split_summary.json", summarize_split_indices(filtered_indices))
    write_text(
        prepare_dir / "README.txt",
        "Prepared RTMPose detector boxes for training.\n"
        f"{readme_context}\n",
    )


def prepare_training_components(
    args: argparse.Namespace,
    model_cfg: Mapping[str, Any],
    split_indices: Mapping[str, SplitIndex],
) -> tuple[RamImageMaskStore, dict[tuple[str, int], dict[str, Any]], dict[str, SplitIndex], dict[str, Any]]:
    store = RamImageMaskStore()
    use_masks = float(getattr(args, "weak_sample_weight", 1.0)) > 0.0
    all_samples = [
        *split_indices["train"].labeled_samples,
        *split_indices["train"].weak_samples,
        *split_indices["val"].labeled_samples,
        *split_indices["test"].labeled_samples,
    ]
    preload_stats = store.preload(
        all_samples,
        preload_images=bool(args.preload_images),
        preload_masks=bool(args.preload_masks and use_masks),
    )
    detector_boxes: dict[tuple[str, int], dict[str, Any]] = {}
    weak_samples_for_detector: list[object] = []
    if float(getattr(args, "weak_sample_weight", 1.0)) > 0.0:
        weak_samples_for_detector.extend(split_indices["train"].weak_samples)
    if weak_samples_for_detector:
        detector_device = resolve_device(str(args.detector_device or args.device))
        detector = load_detector(args.detector_checkpoint.parent, args.detector_checkpoint, device=detector_device)
        detector_boxes = prepare_detector_boxes(
            detector,
            store,
            weak_samples_for_detector,
            detector_batch_desc="detector_rois",
            batch_size=int(args.detector_batch_size),
            score_threshold=float(args.detector_score_threshold),
        )
    filtered_split_indices = filter_samples_with_detector_boxes(
        split_indices,
        detector_boxes,
        weak_sample_weight=float(getattr(args, "weak_sample_weight", 1.0)),
    )
    detector_stats = {
        "detector_box_count": len(detector_boxes),
        "prepare_sample_count": len({(sample.video_name, int(sample.frame_idx)) for sample in all_samples}),
        "preload_stats": preload_stats,
    }
    return store, detector_boxes, filtered_split_indices, detector_stats


def sample_epoch_train_samples(
    samples: Sequence[object],
    requested_count: int,
    *,
    seed: int,
    epoch: int,
    salt: int,
) -> list[object]:
    items = list(samples)
    available = len(items)
    if available == 0:
        return []
    limit = int(requested_count)
    if limit <= 0 or limit >= available:
        return items
    rng = random.Random((int(seed) + 1) * 1_000_003 + int(epoch) * 9_176 + int(salt))
    picked = sorted(rng.sample(range(available), k=limit))
    return [items[idx] for idx in picked]


def build_train_pose_datasets_for_epoch(
    args: argparse.Namespace,
    model_cfg: Mapping[str, Any],
    config: ProjectConfig,
    store: RamImageMaskStore,
    detector_boxes: Mapping[tuple[str, int], dict[str, Any]],
    labeled_samples: Sequence[object],
    weak_samples: Sequence[object],
) -> tuple[RTMPoseDataset, Optional[RTMPoseDataset]]:
    use_masks = float(getattr(args, "weak_sample_weight", 1.0)) > 0.0
    head_cfg = model_cfg["model"]["heads"]["bodypart"]
    input_size = tuple(head_cfg.get("input_size", [256, 256]))
    image_mean = model_cfg.get("image_mean", [0.485, 0.456, 0.406])
    image_std = model_cfg.get("image_std", [0.229, 0.224, 0.225])
    data_train_cfg = dict(model_cfg.get("data", {}).get("train", {}))
    bbox_margin = float(model_cfg.get("data", {}).get("bbox_margin", getattr(args, "bbox_margin", 20.0)))
    crop_cfg = dict(data_train_cfg.get("top_down_crop", {}))
    if "width" not in crop_cfg:
        crop_cfg["width"] = int(input_size[0])
    if "height" not in crop_cfg:
        crop_cfg["height"] = int(input_size[1])
    if "margin" not in crop_cfg:
        crop_cfg["margin"] = int(getattr(args, "top_down_margin", 0))
    if "crop_with_context" not in crop_cfg:
        crop_cfg["crop_with_context"] = bool(getattr(args, "top_down_crop_with_context", True))

    train_labeled = RTMPoseDataset(
        labeled_samples,
        store,
        detector_boxes,
        config.bodyparts,
        config.skeleton,
        config.left_right_symmetry,
        input_size=input_size,
        image_mean=image_mean,
        image_std=image_std,
        crop_expand_scale=float(args.crop_expand_scale),
        bbox_margin=bbox_margin,
        train_aug_cfg=data_train_cfg,
        crop_cfg=crop_cfg,
        train_mode=True,
        include_weak=False,
        use_masks=use_masks,
        mask_select_policy=str(getattr(args, "mask_select_policy", "first")),
        weak_mask_iou_thresh=float(getattr(args, "weak_mask_iou_thresh", 0.0)),
    )
    train_weak = None
    if float(args.weak_sample_weight) > 0.0 and weak_samples:
        train_weak = RTMPoseDataset(
            weak_samples,
            store,
            detector_boxes,
            config.bodyparts,
            config.skeleton,
            config.left_right_symmetry,
            input_size=input_size,
            image_mean=image_mean,
            image_std=image_std,
            crop_expand_scale=float(args.crop_expand_scale),
            bbox_margin=bbox_margin,
            train_aug_cfg=data_train_cfg,
            crop_cfg=crop_cfg,
            train_mode=True,
            include_weak=True,
            use_masks=use_masks,
            mask_select_policy=str(getattr(args, "mask_select_policy", "first")),
            weak_mask_iou_thresh=float(getattr(args, "weak_mask_iou_thresh", 0.0)),
        )
    return train_labeled, train_weak


def build_pose_datasets(
    args: argparse.Namespace,
    model_cfg: Mapping[str, Any],
    config: ProjectConfig,
    split_indices: Mapping[str, SplitIndex],
    store: RamImageMaskStore,
    detector_boxes: Mapping[tuple[str, int], dict[str, Any]],
) -> tuple[RTMPoseDataset, Optional[RTMPoseDataset], RTMPoseDataset, RTMPoseDataset]:
    use_masks = float(getattr(args, "weak_sample_weight", 1.0)) > 0.0
    head_cfg = model_cfg["model"]["heads"]["bodypart"]
    input_size = tuple(head_cfg.get("input_size", [256, 256]))
    image_mean = model_cfg.get("image_mean", [0.485, 0.456, 0.406])
    image_std = model_cfg.get("image_std", [0.229, 0.224, 0.225])
    data_train_cfg = dict(model_cfg.get("data", {}).get("train", {}))
    bbox_margin = float(model_cfg.get("data", {}).get("bbox_margin", getattr(args, "bbox_margin", 20.0)))
    crop_cfg = dict(data_train_cfg.get("top_down_crop", {}))
    if "width" not in crop_cfg:
        crop_cfg["width"] = int(input_size[0])
    if "height" not in crop_cfg:
        crop_cfg["height"] = int(input_size[1])
    if "margin" not in crop_cfg:
        crop_cfg["margin"] = int(getattr(args, "top_down_margin", 0))
    if "crop_with_context" not in crop_cfg:
        crop_cfg["crop_with_context"] = bool(getattr(args, "top_down_crop_with_context", True))
    train_labeled = RTMPoseDataset(
        split_indices["train"].labeled_samples,
        store,
        detector_boxes,
        config.bodyparts,
        config.skeleton,
        config.left_right_symmetry,
        input_size=input_size,
        image_mean=image_mean,
        image_std=image_std,
        crop_expand_scale=float(args.crop_expand_scale),
        bbox_margin=bbox_margin,
        train_aug_cfg=data_train_cfg,
        crop_cfg=crop_cfg,
        train_mode=True,
        include_weak=False,
        use_masks=use_masks,
        mask_select_policy=str(getattr(args, "mask_select_policy", "first")),
        weak_mask_iou_thresh=float(getattr(args, "weak_mask_iou_thresh", 0.0)),
    )
    train_weak = None
    if float(args.weak_sample_weight) > 0.0 and split_indices["train"].weak_samples:
        train_weak = RTMPoseDataset(
            split_indices["train"].weak_samples,
            store,
            detector_boxes,
            config.bodyparts,
            config.skeleton,
            config.left_right_symmetry,
            input_size=input_size,
            image_mean=image_mean,
            image_std=image_std,
            crop_expand_scale=float(args.crop_expand_scale),
            bbox_margin=bbox_margin,
            train_aug_cfg=data_train_cfg,
            crop_cfg=crop_cfg,
            train_mode=True,
            include_weak=True,
            use_masks=use_masks,
            mask_select_policy=str(getattr(args, "mask_select_policy", "first")),
            weak_mask_iou_thresh=float(getattr(args, "weak_mask_iou_thresh", 0.0)),
        )
    val_set = RTMPoseDataset(
        split_indices["val"].labeled_samples,
        store,
        detector_boxes,
        config.bodyparts,
        config.skeleton,
        config.left_right_symmetry,
        input_size=input_size,
        image_mean=image_mean,
        image_std=image_std,
        crop_expand_scale=float(args.crop_expand_scale),
        bbox_margin=bbox_margin,
        train_aug_cfg=data_train_cfg,
        crop_cfg=crop_cfg,
        train_mode=False,
        include_weak=False,
        use_masks=use_masks,
        mask_select_policy=str(getattr(args, "mask_select_policy", "first")),
        weak_mask_iou_thresh=float(getattr(args, "weak_mask_iou_thresh", 0.0)),
    )
    train_eval_set = RTMPoseDataset(
        split_indices["train"].labeled_samples,
        store,
        detector_boxes,
        config.bodyparts,
        config.skeleton,
        config.left_right_symmetry,
        input_size=input_size,
        image_mean=image_mean,
        image_std=image_std,
        crop_expand_scale=float(args.crop_expand_scale),
        bbox_margin=bbox_margin,
        train_aug_cfg=data_train_cfg,
        crop_cfg=crop_cfg,
        train_mode=False,
        include_weak=False,
        use_masks=use_masks,
        mask_select_policy=str(getattr(args, "mask_select_policy", "first")),
        weak_mask_iou_thresh=float(getattr(args, "weak_mask_iou_thresh", 0.0)),
    )
    return train_labeled, train_weak, val_set, train_eval_set


def command_prepare(args: argparse.Namespace) -> int:
    model_cfg = resolve_model_config(args)
    project_cfg = load_project_config(args.project_config)
    validate_disjoint_splits(project_cfg)
    use_masks = float(getattr(args, "weak_sample_weight", 1.0)) > 0.0
    split_indices = build_all_split_indices(
        project_cfg,
        args.labels_root,
        args.labeled_frames_root,
        args.frames_root,
        args.masks_root,
        include_weak=use_masks,
        require_masks=use_masks,
        auto_val_fraction=float(args.auto_val_fraction),
        split_seed=int(args.seed),
    )
    validate_mutual_exclusion(split_indices)
    print_split_summaries(split_indices)

    prepare_dir = build_prepare_dir(args.output_root)
    print(f"Preparing RTMPose data in {prepare_dir}")

    store, detector_boxes, filtered_indices, detector_stats = prepare_training_components(args, model_cfg, split_indices)
    del store

    write_prepare_bundle(
        prepare_dir=prepare_dir,
        args=args,
        detector_boxes=detector_boxes,
        detector_stats=detector_stats,
        filtered_indices=filtered_indices,
        readme_context=(
            "This folder is required by `train` when training without `--prepare`.\n"
            "It is regenerated by the `prepare` subcommand."
        ),
    )
    print(
        f"Prepared {len(detector_boxes)} detector boxes. "
        f"Saved prepare bundle to {prepare_dir}"
    )
    return 0


def command_train(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    model_cfg = resolve_model_config(args)
    project_cfg = load_project_config(args.project_config)
    validate_disjoint_splits(project_cfg)
    use_masks = float(getattr(args, "weak_sample_weight", 1.0)) > 0.0
    split_indices = build_all_split_indices(
        project_cfg,
        args.labels_root,
        args.labeled_frames_root,
        args.frames_root,
        args.masks_root,
        include_weak=use_masks,
        require_masks=use_masks,
        auto_val_fraction=float(args.auto_val_fraction),
        split_seed=int(args.seed),
    )
    validate_mutual_exclusion(split_indices)
    run_dir = build_run_dir(args.output_root, str(args.prefix or ""))
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    console_log = (run_dir / "console.log").open("a", encoding="utf-8")
    sys.stdout = TeeStream(original_stdout, console_log)
    sys.stderr = TeeStream(original_stderr, console_log)
    try:
        print_split_summaries(split_indices)
        print(f"RTMPose run directory: {run_dir}")
        if bool(getattr(args, "prepare", False)):
            print("Preparing detector boxes in-run (train --prepare enabled)")
            store, detector_boxes, filtered_indices, detector_stats = prepare_training_components(
                args, model_cfg, split_indices
            )
            prepare_dir = run_dir / "prepare"
            write_prepare_bundle(
                prepare_dir=prepare_dir,
                args=args,
                detector_boxes=detector_boxes,
                detector_stats=detector_stats,
                filtered_indices=filtered_indices,
                readme_context="Prepared by the `train --prepare` path for this run.",
            )
            detector_stats = dict(detector_stats)
            detector_stats["train_preload_stats"] = detector_stats.get("preload_stats", {})
            print(f"Saved run-scoped prepare bundle to {prepare_dir}")
        else:
            detector_boxes, filtered_indices, detector_stats, prepare_dir = load_prepared_training_components(args, split_indices)
            print(f"Using prepared data from {prepare_dir}")
            all_samples = [
                *filtered_indices["train"].labeled_samples,
                *filtered_indices["train"].weak_samples,
                *filtered_indices["val"].labeled_samples,
                *filtered_indices["test"].labeled_samples,
            ]
            store = RamImageMaskStore()
            preload_stats = store.preload(
                all_samples,
                preload_images=bool(args.preload_images),
                preload_masks=bool(args.preload_masks),
            )
            detector_stats = dict(detector_stats)
            detector_stats["train_preload_stats"] = preload_stats

        write_yaml(run_dir / "resolved_model_config.yaml", make_serializable(model_cfg))
        write_json(run_dir / "split_summary.json", summarize_split_indices(filtered_indices))
        write_json(run_dir / "cache_stats.json", make_serializable(detector_stats))
        export_config_bundle(
            run_dir,
            project_config_path=args.project_config,
            resolved_run_args=vars(args),
            model_config_path=args.model_config,
            model_name=str(args.model_name),
            run_config_path=args.run_config,
            config_overwrite_path=args.config_overwrite,
        )

        _full_train_labeled_set, full_train_weak_set, val_set, train_eval_set = build_pose_datasets(
            args, model_cfg, project_cfg, filtered_indices, store, detector_boxes
        )
        val_loader = build_dataloader(
            val_set,
            batch_size=int(args.eval_batch_size),
            shuffle=False,
            drop_last=False,
            workers=int(args.workers),
            pin_memory=bool(args.pin_memory),
            persistent_workers=bool(args.persistent_workers),
            prefetch_factor=int(args.prefetch_factor) if int(args.workers) > 0 else None,
        )
        train_eval_loader = build_dataloader(
            train_eval_set,
            batch_size=int(args.eval_batch_size),
            shuffle=False,
            drop_last=False,
            workers=int(args.workers),
            pin_memory=bool(args.pin_memory),
            persistent_workers=bool(args.persistent_workers),
            prefetch_factor=int(args.prefetch_factor) if int(args.workers) > 0 else None,
        )

        pretrain_cfg = dict(model_cfg.get("pretrain", {}))
        init_checkpoint = args.init_checkpoint if args.init_checkpoint else None
        backbone_checkpoint: Path | None = None
        if init_checkpoint is None:
            pretrain_checkpoint = pretrain_cfg.get("checkpoint")
            if not pretrain_checkpoint:
                raise ValueError(
                    "model_cfg.pretrain.checkpoint is required when init_checkpoint is not provided."
                )
            backbone_checkpoint = Path(pretrain_checkpoint)
        model = build_pose_model(
            model_cfg,
            device=device,
            checkpoint_path=init_checkpoint,
            backbone_checkpoint_path=backbone_checkpoint,
        )
        configure_backbone_freeze(model, bool(args.freeze_backbone))
        optimizer = build_optimizer(args, model)
        scheduler = build_scheduler(args, optimizer)
        scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp))

        selection_metric_name = str(args.selection_metric)
        selection_mode = str(args.selection_mode)
        patience = int(args.patience)
        if patience < 1:
            raise ValueError(f"--patience must be >= 1, got {patience}")
        eval_every_n_epoch = max(int(args.eval_every_n_epoch), 1)
        save_every_n_epoch = max(int(args.save_every_n_epoch), 1)
        max_save = max(int(args.max_save), 0)
        best_metric = float("inf") if selection_mode == "min" else -float("inf")
        best_epoch = 0
        last_improved_epoch = 0
        history: list[dict[str, Any]] = []

        print("Running pre-training evaluation (epoch 0)")
        initial_train_eval_metrics = evaluate_pose_model(
            model,
            train_eval_loader,
            device,
            float(args.lambda_pose),
            float(args.lambda_mask),
            float(args.lambda_visibility),
        )
        initial_val_metrics = evaluate_pose_model(
            model,
            val_loader,
            device,
            float(args.lambda_pose),
            float(args.lambda_mask),
            float(args.lambda_visibility),
        )
        initial_train_loss_eval, initial_train_metric_eval = split_eval_payload(initial_train_eval_metrics)
        initial_val_loss_eval, initial_val_metric_eval = split_eval_payload(initial_val_metrics)
        initial_train_metric_value = resolve_selection_score(initial_train_eval_metrics, selection_metric_name)
        initial_val_metric_value = resolve_selection_score(initial_val_metrics, selection_metric_name)
        initial_summary = {
            "epoch": 0,
            "train": {
                "loss": float(initial_train_loss_eval["loss"]),
                "loss_eval": initial_train_loss_eval,
                "metric_eval": initial_train_metric_eval,
                "selection_metric_name": selection_metric_name,
                "selection_metric": initial_train_metric_value,
                "sampled_labeled_count": 0,
                "sampled_weak_count": 0,
                "optimization": {},
            },
            "val": {
                "loss": float(initial_val_loss_eval["loss"]),
                "loss_eval": initial_val_loss_eval,
                "metric_eval": initial_val_metric_eval,
                "selection_metric_name": selection_metric_name,
                "selection_metric": initial_val_metric_value,
            },
            "lr": float(optimizer.param_groups[0]["lr"]),
            "evaluated": True,
        }
        history.append(initial_summary)
        write_json(run_dir / "history.json", history)
        best_metric = initial_val_metric_value
        best_epoch = 0
        last_improved_epoch = 0
        best_summary = {
            "epoch": best_epoch,
            "selection_metric_name": selection_metric_name,
            "selection_metric_value": float(best_metric),
            "selection_mode": selection_mode,
            "train": initial_summary["train"],
            "val": initial_summary["val"],
            "lr": initial_summary["lr"],
            "checkpoint": "checkpoint_best.pt",
        }
        save_rtmpose_checkpoint(
            run_dir / "checkpoint_best.pt",
            model,
            optimizer=optimizer,
            scaler=scaler,
            extra_state={
                "epoch": 0,
                "history": history,
                "model_config": make_serializable(model_cfg),
                "bodyparts": list(project_cfg.bodyparts),
            },
        )
        export_best_epoch_files(run_dir, best_summary)
        print(
            "Epoch 000 pretrain_eval "
            f"train_rmse={initial_train_metric_eval.get('rmse_unfiltered', float('nan')):.4f} "
            f"train_rmse_90={initial_train_metric_eval.get('rmse_90', float('nan')):.4f} "
            f"val_rmse={initial_val_metric_eval.get('rmse_unfiltered', float('nan')):.4f} "
            f"val_rmse_90={initial_val_metric_eval.get('rmse_90', float('nan')):.4f}"
        )

        for epoch in range(1, int(args.epochs) + 1):
            sampled_train_labeled_samples = sample_epoch_train_samples(
                filtered_indices["train"].labeled_samples,
                int(getattr(args, "train_labeled_samples_per_epoch", 0)),
                seed=int(args.seed),
                epoch=epoch,
                salt=11,
            )
            sampled_train_weak_samples: list[object] = []
            if full_train_weak_set is not None:
                sampled_train_weak_samples = sample_epoch_train_samples(
                    filtered_indices["train"].weak_samples,
                    int(getattr(args, "train_weak_samples_per_epoch", 0)),
                    seed=int(args.seed),
                    epoch=epoch,
                    salt=23,
                )

            train_labeled_set, train_weak_set = build_train_pose_datasets_for_epoch(
                args=args,
                model_cfg=model_cfg,
                config=project_cfg,
                store=store,
                detector_boxes=detector_boxes,
                labeled_samples=sampled_train_labeled_samples,
                weak_samples=sampled_train_weak_samples,
            )
            if len(train_labeled_set) == 0:
                raise RuntimeError(
                    "No labeled training samples available for this epoch after subsampling. "
                    "Increase train_labeled_samples_per_epoch or check input data."
                )
            labeled_loader = build_dataloader(
                train_labeled_set,
                batch_size=int(args.labeled_batch_size),
                shuffle=True,
                drop_last=len(train_labeled_set) > 1,
                workers=int(args.workers),
                pin_memory=bool(args.pin_memory),
                persistent_workers=bool(args.persistent_workers),
                prefetch_factor=int(args.prefetch_factor) if int(args.workers) > 0 else None,
            )
            weak_loader = None
            if train_weak_set is not None and len(train_weak_set) > 1:
                weak_loader = build_dataloader(
                    train_weak_set,
                    batch_size=int(args.weak_batch_size),
                    shuffle=True,
                    drop_last=True,
                    workers=int(args.workers),
                    pin_memory=bool(args.pin_memory),
                    persistent_workers=bool(args.persistent_workers),
                    prefetch_factor=int(args.prefetch_factor) if int(args.workers) > 0 else None,
                )
            train_metrics = train_one_epoch(
                model=model,
                labeled_loader=labeled_loader,
                weak_loader=weak_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                lambda_pose=float(args.lambda_pose),
                lambda_mask=float(args.lambda_mask),
                lambda_visibility=float(args.lambda_visibility),
                lambda_entropy=float(args.lambda_entropy),
                weak_sample_weight=float(args.weak_sample_weight),
                mask_on_labeled=bool(getattr(args, "mask_on_labeled", True)),
                mask_loss_mode=str(getattr(args, "mask_loss_mode", "containment")),
                mask_variance_weight=float(getattr(args, "mask_variance_weight", 0.0)),
                mask_conf_threshold=float(getattr(args, "mask_conf_threshold", 0.0)),
                weak_use_quality_weight=bool(getattr(args, "weak_use_quality_weight", False)),
                weak_use_visibility_gate=bool(getattr(args, "weak_use_visibility_gate", False)),
                weak_iou_weight_t0=float(getattr(args, "weak_iou_weight_t0", 0.3)),
                weak_iou_weight_t1=float(getattr(args, "weak_iou_weight_t1", 0.8)),
                weak_iou_weight_power=float(getattr(args, "weak_iou_weight_power", 2.0)),
                weak_conf_weight_power=float(getattr(args, "weak_conf_weight_power", 1.0)),
                mask_ring_radius=int(getattr(args, "mask_ring_radius", 3)),
                mask_outside_weight=float(getattr(args, "mask_outside_weight", 0.5)),
                mask_outside_trim=float(getattr(args, "mask_outside_trim", 0.1)),
                mask_mass_floor=float(getattr(args, "mask_mass_floor", 0.0)),
                use_amp=bool(args.amp),
                log_interval=int(args.log_interval),
                freeze_backbone=bool(args.freeze_backbone),
            )
            force_eval_for_patience = (epoch - last_improved_epoch) >= patience
            should_eval = (epoch % eval_every_n_epoch) == 0 or force_eval_for_patience
            train_eval_metrics: dict[str, float] = {}
            train_loss_eval = {"loss": float("nan")}
            train_metric_eval: dict[str, float] = {}
            val_metrics: dict[str, float] = {}
            val_loss_eval = {"loss": float("nan")}
            val_metric_eval: dict[str, float] = {}
            train_metric_value = float("nan")
            val_metric_value = float("nan")
            if should_eval:
                train_eval_metrics = evaluate_pose_model(
                    model,
                    train_eval_loader,
                    device,
                    float(args.lambda_pose),
                    float(args.lambda_mask),
                    float(args.lambda_visibility),
                )
                val_metrics = evaluate_pose_model(
                    model,
                    val_loader,
                    device,
                    float(args.lambda_pose),
                    float(args.lambda_mask),
                    float(args.lambda_visibility),
                )
                train_loss_eval, train_metric_eval = split_eval_payload(train_eval_metrics)
                val_loss_eval, val_metric_eval = split_eval_payload(val_metrics)
                train_metric_value = resolve_selection_score(train_eval_metrics, selection_metric_name)
                val_metric_value = resolve_selection_score(val_metrics, selection_metric_name)
            epoch_summary = {
                "epoch": epoch,
                "train": {
                    "loss": float(train_loss_eval["loss"]),
                    "loss_eval": train_loss_eval,
                    "metric_eval": train_metric_eval,
                    "selection_metric_name": selection_metric_name,
                    "selection_metric": train_metric_value,
                    "sampled_labeled_count": int(len(sampled_train_labeled_samples)),
                    "sampled_weak_count": int(len(sampled_train_weak_samples)),
                    "optimization": {key: float(value) for key, value in train_metrics.items()},
                },
                "val": {
                    "loss": float(val_loss_eval["loss"]),
                    "loss_eval": val_loss_eval,
                    "metric_eval": val_metric_eval,
                    "selection_metric_name": selection_metric_name,
                    "selection_metric": val_metric_value,
                },
                "lr": float(optimizer.param_groups[0]["lr"]),
                "evaluated": should_eval,
            }
            history.append(epoch_summary)
            write_json(run_dir / "history.json", history)
            save_rtmpose_checkpoint(
                run_dir / "checkpoint_latest.pt",
                model,
                optimizer=optimizer,
                scaler=scaler,
                extra_state={
                    "epoch": epoch,
                    "history": history,
                    "model_config": make_serializable(model_cfg),
                    "bodyparts": list(project_cfg.bodyparts),
                },
            )
            epoch_message = (
                f"Epoch {epoch:03d} "
                f"loss={train_metrics.get('loss', float('nan')):.4f} "
                f"sup={train_metrics.get('supervised', float('nan')):.4f} "
                f"mask={train_metrics.get('mask', float('nan')):.4f} "
                f"vis={train_metrics.get('visibility', float('nan')):.4f} "
                f"ent={train_metrics.get('entropy', float('nan')):.4f} "
                f"n_label={len(sampled_train_labeled_samples)} "
                f"n_weak={len(sampled_train_weak_samples)} "
                f"cpu={train_metrics.get('cpu', float('nan')):.4f}s "
                f"h2d={train_metrics.get('h2d', float('nan')):.4f}s "
                f"gpu_fwd={train_metrics.get('gpu_fwd', float('nan')):.4f}s "
                f"gpu_bwd={train_metrics.get('gpu_bwd', float('nan')):.4f}s "
                f"train_rmse={train_metric_eval.get('rmse_unfiltered', float('nan')):.4f} "
                f"train_rmse_90={train_metric_eval.get('rmse_90', float('nan')):.4f}"
            )
            if should_eval:
                epoch_message += (
                    f" val_rmse={val_metric_eval.get('rmse_unfiltered', float('nan')):.4f} "
                    f"val_rmse_90={val_metric_eval.get('rmse_90', float('nan')):.4f}"
                )
            print(epoch_message)
            if (epoch % save_every_n_epoch) == 0 and max_save > 0:
                save_rtmpose_checkpoint(
                    run_dir / f"checkpoint_epoch_{epoch:03d}.pt",
                    model,
                    optimizer=optimizer,
                    scaler=scaler,
                    extra_state={
                        "epoch": epoch,
                        "history": history,
                        "model_config": make_serializable(model_cfg),
                        "bodyparts": list(project_cfg.bodyparts),
                    },
                )
                prune_periodic_checkpoints(run_dir, max_keep=max_save)
            if should_eval and metric_is_better(val_metric_value, best_metric, selection_mode):
                best_metric = val_metric_value
                best_epoch = epoch
                last_improved_epoch = epoch
                best_summary = {
                    "epoch": best_epoch,
                    "selection_metric_name": selection_metric_name,
                    "selection_metric_value": float(best_metric),
                    "selection_mode": selection_mode,
                    "train": epoch_summary["train"],
                    "val": epoch_summary["val"],
                    "lr": epoch_summary["lr"],
                    "checkpoint": "checkpoint_best.pt",
                }
                save_rtmpose_checkpoint(
                    run_dir / "checkpoint_best.pt",
                    model,
                    optimizer=optimizer,
                    scaler=scaler,
                    extra_state={
                        "epoch": epoch,
                        "history": history,
                        "model_config": make_serializable(model_cfg),
                        "bodyparts": list(project_cfg.bodyparts),
                    },
                )
                export_best_epoch_files(run_dir, best_summary)
            if should_eval and (epoch - last_improved_epoch) >= patience:
                print(
                    f"Early stopping at epoch {epoch:03d}: no val improvement in "
                    f"{patience} epoch(s) since epoch {last_improved_epoch:03d}."
                )
                break
            if scheduler is not None:
                scheduler.step()

        write_text(
            run_dir / "metrics.txt",
            "\n".join(
                [
                    f"best_{selection_metric_name}: {best_metric:.6f}",
                    f"best_epoch: {best_epoch}",
                    f"selection_mode: {selection_mode}",
                ]
            )
            + "\n",
        )
        print(f"Best epoch {best_epoch} with {selection_metric_name}={best_metric:.6f}")
        return 0
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        console_log.close()

def command_eval(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    model_cfg = resolve_model_config(args)
    project_cfg = load_project_config(args.project_config)
    use_masks = float(getattr(args, "weak_sample_weight", 1.0)) > 0.0
    split_indices = build_all_split_indices(
        project_cfg,
        args.labels_root,
        args.labeled_frames_root,
        args.frames_root,
        args.masks_root,
        include_weak=use_masks,
        require_masks=use_masks,
        auto_val_fraction=float(args.auto_val_fraction),
        split_seed=int(args.seed),
    )
    validate_mutual_exclusion(split_indices)
    store, detector_boxes, filtered_indices, detector_stats = prepare_training_components(args, model_cfg, split_indices)
    del detector_stats
    samples = select_samples_for_split(filtered_indices, args.split, args.video_name)
    data_train_cfg = dict(model_cfg.get("data", {}).get("train", {}))
    bbox_margin = float(model_cfg.get("data", {}).get("bbox_margin", getattr(args, "bbox_margin", 20.0)))
    crop_cfg = dict(data_train_cfg.get("top_down_crop", {}))
    crop_cfg.setdefault("margin", int(getattr(args, "top_down_margin", 0)))
    crop_cfg.setdefault("crop_with_context", bool(getattr(args, "top_down_crop_with_context", True)))
    dataset = RTMPoseDataset(
        samples,
        store,
        detector_boxes,
        project_cfg.bodyparts,
        project_cfg.skeleton,
        project_cfg.left_right_symmetry,
        tuple(model_cfg["model"]["heads"]["bodypart"].get("input_size", [256, 256])),
        model_cfg.get("image_mean", [0.485, 0.456, 0.406]),
        model_cfg.get("image_std", [0.229, 0.224, 0.225]),
        float(args.crop_expand_scale),
        bbox_margin=bbox_margin,
        train_aug_cfg=data_train_cfg,
        crop_cfg=crop_cfg,
        train_mode=False,
        include_weak=False,
        use_masks=use_masks,
        mask_select_policy=str(getattr(args, "mask_select_policy", "first")),
        weak_mask_iou_thresh=float(getattr(args, "weak_mask_iou_thresh", 0.0)),
    )
    loader = build_dataloader(
        dataset,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        drop_last=False,
        workers=int(args.workers),
        pin_memory=bool(args.pin_memory),
        persistent_workers=bool(args.persistent_workers),
        prefetch_factor=int(args.prefetch_factor) if int(args.workers) > 0 else None,
    )
    model, _ = load_model_from_checkpoint_for_inference(
        model_path=args.checkpoint.parent,
        checkpoint=args.checkpoint,
        device=device,
    )
    metrics = evaluate_pose_model(
        model,
        loader,
        device,
        float(args.lambda_pose),
        float(args.lambda_mask),
        float(args.lambda_visibility),
    )
    output_dir = build_run_dir(args.output_root, with_optional_prefix(f"eval_{args.split}", str(args.prefix or "")))
    payload = {
        "split": args.split,
        "video": args.video_name,
        "sample_count": len(samples),
        **metrics,
    }
    write_json(output_dir / "metrics.json", payload)
    write_text(output_dir / "metrics.txt", "\n".join(f"{k}: {v}" for k, v in payload.items()) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


def command_debug(args: argparse.Namespace) -> int:
    model_cfg = resolve_model_config(args)
    project_cfg = load_project_config(args.project_config)
    use_masks = float(getattr(args, "weak_sample_weight", 1.0)) > 0.0
    split_indices = build_all_split_indices(
        project_cfg,
        args.labels_root,
        args.labeled_frames_root,
        args.frames_root,
        args.masks_root,
        include_weak=use_masks,
        require_masks=use_masks,
        auto_val_fraction=float(args.auto_val_fraction),
        split_seed=int(args.seed),
    )
    validate_mutual_exclusion(split_indices)
    store, detector_boxes, filtered_indices, detector_stats = prepare_training_components(args, model_cfg, split_indices)
    del detector_stats
    print("Available debug sample counts by split:")
    for split in SPLIT_NAMES:
        print(
            f"  {split}: label={len(filtered_indices[split].labeled_samples)} "
            f"sam2={len(filtered_indices[split].weak_samples)}"
        )
    debug_root = build_run_dir(args.output_root, with_optional_prefix("debug", str(args.prefix or "")))
    if debug_root.exists():
        shutil.rmtree(debug_root)
    label_dir = debug_root / "label"
    sam2_dir = debug_root / "sam2"
    label_dir.mkdir(parents=True, exist_ok=True)
    sam2_dir.mkdir(parents=True, exist_ok=True)

    input_size = tuple(model_cfg["model"]["heads"]["bodypart"].get("input_size", [256, 256]))
    data_train_cfg = dict(model_cfg.get("data", {}).get("train", {}))
    bbox_margin = float(model_cfg.get("data", {}).get("bbox_margin", getattr(args, "bbox_margin", 20.0)))
    crop_cfg = dict(data_train_cfg.get("top_down_crop", {}))
    crop_cfg.setdefault("margin", int(getattr(args, "top_down_margin", 0)))
    crop_cfg.setdefault("crop_with_context", bool(getattr(args, "top_down_crop_with_context", True)))
    rng = random.Random(int(args.seed))

    def export_samples(source_name: str, samples: Sequence[object], output_dir: Path) -> None:
        selected = list(samples)
        rng.shuffle(selected)
        dataset = RTMPoseDataset(
            selected,
            store,
            detector_boxes,
            project_cfg.bodyparts,
            project_cfg.skeleton,
            project_cfg.left_right_symmetry,
            input_size,
            model_cfg.get("image_mean", [0.485, 0.456, 0.406]),
            model_cfg.get("image_std", [0.229, 0.224, 0.225]),
            float(args.crop_expand_scale),
            bbox_margin=bbox_margin,
            train_aug_cfg=data_train_cfg,
            crop_cfg=crop_cfg,
            train_mode=True,
            include_weak=(source_name == "sam2"),
            use_masks=use_masks,
            mask_select_policy=str(getattr(args, "mask_select_policy", "first")),
            weak_mask_iou_thresh=float(getattr(args, "weak_mask_iou_thresh", 0.0)),
        )
        for idx in range(len(dataset)):
            item = dataset[idx]
            image_rgb = np.transpose(item["debug_image"].numpy(), (1, 2, 0))
            image_rgb = np.ascontiguousarray((image_rgb * 255.0).clip(0, 255).astype(np.uint8))
            base = draw_box_on_image(image_rgb, item["roi_box_in_crop"], color=(255, 128, 0), thickness=2)
            if source_name == "label":
                vis = item["keypoints"].numpy()
                rendered = draw_keypoints_on_image(
                    cv2.cvtColor(base, cv2.COLOR_BGR2RGB),
                    vis,
                    project_cfg.bodyparts,
                    project_cfg.skeleton,
                )
            else:
                rendered = draw_mask_on_image(cv2.cvtColor(base, cv2.COLOR_BGR2RGB), item["mask"].numpy()[0] > 0.5)
            filename = f"{source_name}_{idx:04d}_{item['video_name'].replace(' ', '_')}_frame{int(item['frame_idx']):06d}.jpg"
            cv2.imwrite(str(output_dir / filename), rendered)
        print(f"Saved {len(selected)} {source_name} debug images to {output_dir}")

    split = str(args.split)
    export_samples("label", filtered_indices[split].labeled_samples, label_dir)
    export_samples("sam2", filtered_indices[split].weak_samples, sam2_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone RTMPose trainer with detector-driven crops.")
    parser.add_argument("--project-config", type=Path, default=DEFAULT_PROJECT_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--run-config", type=Path, default=DEFAULT_RUN_CONFIG)
    parser.add_argument("--config-overwrite", "--config_overwrite", dest="config_overwrite", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default="default")
    parser.add_argument("--labels-root", type=Path, default=PROJECT_ROOT / "input" / "labeled-data")
    parser.add_argument("--labeled-frames-root", type=Path, default=PROJECT_ROOT / "output" / "sam2" / "DLC_frames")
    parser.add_argument("--frames-root", type=Path, default=PROJECT_ROOT / "output" / "sam2" / "final")
    parser.add_argument("--masks-root", type=Path, default=PROJECT_ROOT / "output" / "sam2" / "sam2_pickle_filtered")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--detector-model-name", type=str, default="default")
    parser.add_argument("--detector-checkpoint", type=Path, default=PROJECT_ROOT / "output" / "ssdlite" / "no_weak_20260325_224005" / "checkpoint_best.pt")
    parser.add_argument("--detector-device", type=str, default="")
    parser.add_argument("--detector-score-threshold", type=float, default=0.5)
    parser.add_argument("--detector-batch-size", type=int, default=16)
    parser.add_argument(
        "--pretrained-backbone-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "input" / "DeepLabCutModelZoo-SuperAnimal-Quadruped" / "superanimal_quadruped_rtmpose_s.pt",
    )
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--preload-images", action="store_true")
    parser.add_argument("--preload-masks", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--auto-val-fraction",
        type=float,
        default=0.1,
        help="When config.yaml has no val_videos, reserve this fraction of labeled train frames for val.",
    )
    parser.add_argument("--crop-expand-scale", type=float, default=0.15)
    parser.add_argument("--bbox-margin", type=float, default=20.0)
    parser.add_argument("--top-down-margin", type=int, default=0)
    parser.add_argument("--top-down-crop-with-context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--score-cutoff", type=float, default=0.0)
    parser.add_argument("--visibility-cutoff", type=float, default=0.5)
    parser.add_argument("--weak-sample-weight", type=float, default=1.0)
    parser.add_argument(
        "--mask-select-policy",
        type=str,
        choices=("first", "largest", "best_iou"),
        default="first",
        help="Policy for selecting SAM2 mask when multiple objects are present.",
    )
    parser.add_argument(
        "--weak-mask-iou-thresh",
        type=float,
        default=0.0,
        help="Minimum IoU between SAM2 mask bbox and detector bbox for weak samples.",
    )
    parser.add_argument(
        "--mask-on-labeled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply SAM2 mask loss to labeled samples (default true).",
    )
    parser.add_argument(
        "--mask-loss-mode",
        type=str,
        choices=("containment", "variance", "alignment", "ring", "trimmed"),
        default="containment",
        help="Mask loss variant to use for weak supervision.",
    )
    parser.add_argument(
        "--mask-variance-weight",
        type=float,
        default=0.0,
        help="Additional variance penalty weight when using mask_loss_mode=variance.",
    )
    parser.add_argument(
        "--mask-conf-threshold",
        type=float,
        default=0.0,
        help="Confidence threshold for applying mask loss (SimCC peak conf).",
    )
    parser.add_argument("--weak-use-quality-weight", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--weak-use-visibility-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--weak-iou-weight-t0", type=float, default=0.3)
    parser.add_argument("--weak-iou-weight-t1", type=float, default=0.8)
    parser.add_argument("--weak-iou-weight-power", type=float, default=2.0)
    parser.add_argument("--weak-conf-weight-power", type=float, default=1.0)
    parser.add_argument("--mask-ring-radius", type=int, default=3)
    parser.add_argument("--mask-outside-weight", type=float, default=0.5)
    parser.add_argument("--mask-outside-trim", type=float, default=0.1)
    parser.add_argument("--mask-mass-floor", type=float, default=0.0)
    parser.add_argument("--save-every-n-epoch", type=int, default=1)
    parser.add_argument("--eval-every-n-epoch", type=int, default=1)
    parser.add_argument("--max-save", type=int, default=5)

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--prefix", default="")
    train_parser.add_argument(
        "--prepare",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run prepare inside train (default). Use --no-prepare to reuse an existing global prepare bundle.",
    )
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--labeled-batch-size", type=int, default=16)
    train_parser.add_argument("--weak-batch-size", type=int, default=16)
    train_parser.add_argument(
        "--train-labeled-samples-per-epoch",
        type=int,
        default=0,
        help="Max labeled train samples to use per epoch after train/val split (0 = all).",
    )
    train_parser.add_argument(
        "--train-weak-samples-per-epoch",
        type=int,
        default=0,
        help="Max weak train samples to use per epoch after filtering (0 = all).",
    )
    train_parser.add_argument("--eval-batch-size", type=int, default=16)
    train_parser.add_argument("--optimizer", type=str, default="AdamW")
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--scheduler", type=str, default="cosine")
    train_parser.add_argument("--scheduler-step-size", type=int, default=10)
    train_parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    train_parser.add_argument("--selection-metric", type=str, default="rmse_unfiltered")
    train_parser.add_argument("--selection-mode", type=str, choices=("min", "max"), default="min")
    train_parser.add_argument("--patience", type=int, default=50)
    train_parser.add_argument("--lambda-pose", type=float, default=1.0)
    train_parser.add_argument("--lambda-mask", type=float, default=0.5)
    train_parser.add_argument("--lambda-visibility", type=float, default=1.0)
    train_parser.add_argument("--lambda-entropy", type=float, default=0.05)
    train_parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument("--amp", action="store_true")
    train_parser.add_argument("--log-interval", type=int, default=10)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--prefix", default="")
    eval_parser.add_argument("--split", choices=("val", "test"), default="val")
    eval_parser.add_argument("--video-name", type=str, default=None)
    eval_parser.add_argument("--eval-batch-size", type=int, default=16)
    eval_parser.add_argument("--lambda-pose", type=float, default=1.0)
    eval_parser.add_argument("--lambda-mask", type=float, default=0.5)
    eval_parser.add_argument("--lambda-visibility", type=float, default=1.0)

    debug_parser = subparsers.add_parser("debug")
    debug_parser.add_argument("--split", choices=SPLIT_NAMES, default="train")
    debug_parser.add_argument("--prefix", default="")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args = merge_cli_with_yaml(args, parser)
    set_seed(int(args.seed))
    if args.command == "train":
        return command_train(args)
    if args.command == "prepare":
        return command_prepare(args)
    if args.command == "eval":
        return command_eval(args)
    if args.command == "debug":
        return command_debug(args)
    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
