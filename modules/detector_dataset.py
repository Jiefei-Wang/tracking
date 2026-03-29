from __future__ import annotations

import os
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import albumentations as A
import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from modules.detector_bbox_utils import expand_xyxy, keypoints_to_xyxy, mask_to_xyxy
from modules.label_csv_utils import load_keypoints

SPLIT_NAMES = ("train", "val", "test")


@dataclass
class ProjectConfig:
    train_videos: list[str]
    val_videos: list[str]
    test_videos: list[str]
    bodyparts: list[str]

    def videos_for_split(self, split: str) -> list[str]:
        if split == "train":
            return list(self.train_videos)
        if split == "val":
            return list(self.val_videos)
        if split == "test":
            return list(self.test_videos)
        raise ValueError(f"Unknown split: {split}")


@dataclass
class DetectionSample:
    split: str
    source: str
    video_name: str
    frame_idx: int
    image_path: str
    boxes: list[list[float]]
    roi_boxes: list[list[float]]
    labels: list[int]
    sample_weight: float
    is_weak: bool

    @property
    def sample_id(self) -> str:
        return f"{self.video_name}:{self.frame_idx}"


@dataclass
class SplitIndex:
    split: str
    videos: list[str]
    labeled_samples: list[DetectionSample]
    weak_samples: list[DetectionSample]
    warnings: list[str]


def load_yaml_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in YAML file: {path}")
    return payload


def load_project_config(config_path: Path) -> ProjectConfig:
    raw = load_yaml_file(config_path)
    return ProjectConfig(
        train_videos=list(raw.get("train_videos", [])),
        val_videos=list(raw.get("val_videos", [])),
        test_videos=list(raw.get("test_videos", [])),
        bodyparts=list(raw.get("bodyparts", [])),
    )


def validate_disjoint_splits(config: ProjectConfig) -> None:
    mapping = {
        "train": set(config.train_videos),
        "val": set(config.val_videos),
        "test": set(config.test_videos),
    }
    overlaps: list[str] = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        common = sorted(mapping[left] & mapping[right])
        if common:
            overlaps.append(f"{left}/{right}: {', '.join(common)}")
    if overlaps:
        raise ValueError("Overlapping split membership in config.yaml:\n" + "\n".join(overlaps))


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
        arr = np.asarray(first_mask).astype(bool)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            continue
        mask_map[frame_idx] = arr
    return mask_map


def extract_keypoints_for_row(df, row_idx: int, bodyparts: Sequence[str]) -> tuple[list[list[float]], list[int]]:
    row = df.iloc[row_idx]
    keypoints: list[list[float]] = []
    visibility: list[int] = []
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
    bbox_margin: float,
    weak_sample_weight: float,
    min_box_size: float = 1.0,
) -> SplitIndex:
    videos = config.videos_for_split(split)
    labeled_samples: list[DetectionSample] = []
    weak_samples: list[DetectionSample] = []
    warnings: list[str] = []
    include_weak = split == "train"

    for video_name in videos:
        csv_path = labels_root / video_name / "CollectedData_rats.csv"
        frames_dir = frames_root / video_name
        mask_path = masks_root / f"{video_name}.pkl"

        if not frames_dir.is_dir():
            warnings.append(f"{split}:{video_name}: missing frames dir {frames_dir}")
            continue
        frame_files = list_frame_files(frames_dir)
        if not frame_files:
            warnings.append(f"{split}:{video_name}: no frame files found")
            continue

        mask_map: dict[int, np.ndarray] = {}
        if mask_path.is_file():
            mask_map = load_mask_frame_dict(mask_path)
        elif include_weak:
            warnings.append(f"{split}:{video_name}: missing SAM2 mask file {mask_path}")

        labeled_frame_indices: set[int] = set()
        if csv_path.is_file():
            label_df = load_keypoints(str(csv_path))
            for row_idx in range(len(label_df)):
                frame_idx = int(label_df["frame"].iloc[row_idx])
                labeled_frame_indices.add(frame_idx)
                image_path = frame_files.get(frame_idx)
                if image_path is None:
                    warnings.append(f"{split}:{video_name}: labeled frame {frame_idx} missing image")
                    continue
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if image is None:
                    warnings.append(f"{split}:{video_name}: failed to read image {image_path}")
                    continue
                height, width = image.shape[:2]
                keypoints, visibility = extract_keypoints_for_row(label_df, row_idx, config.bodyparts)
                box = keypoints_to_xyxy(
                    keypoints=keypoints,
                    visibility=visibility,
                    width=width,
                    height=height,
                    margin=0.0,
                    min_size=min_box_size,
                )
                if box is None:
                    warnings.append(f"{split}:{video_name}: labeled frame {frame_idx} produced invalid bbox")
                    continue
                expanded_box = expand_xyxy(box, margin=bbox_margin, width=width, height=height)
                labeled_samples.append(
                    DetectionSample(
                        split=split,
                        source="label",
                        video_name=video_name,
                        frame_idx=frame_idx,
                        image_path=image_path,
                        boxes=[expanded_box],
                        roi_boxes=[box],
                        labels=[1],
                        sample_weight=1.0,
                        is_weak=False,
                    )
                )
        else:
            warnings.append(f"{split}:{video_name}: missing label CSV {csv_path}")

        if not include_weak:
            continue

        if not mask_map:
            warnings.append(f"{split}:{video_name}: no valid masks found for weak supervision")
            continue
        for frame_idx, image_path in frame_files.items():
            if frame_idx in labeled_frame_indices:
                continue
            mask = mask_map.get(frame_idx)
            if mask is None:
                continue
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                warnings.append(f"{split}:{video_name}: failed to read image {image_path}")
                continue
            height, width = image.shape[:2]
            box = mask_to_xyxy(mask=mask, width=width, height=height, min_size=min_box_size)
            if box is None:
                continue
            expanded_box = expand_xyxy(box, margin=bbox_margin, width=width, height=height)
            weak_samples.append(
                DetectionSample(
                    split=split,
                    source="sam2",
                    video_name=video_name,
                    frame_idx=frame_idx,
                    image_path=image_path,
                    boxes=[expanded_box],
                    roi_boxes=[box],
                    labels=[1],
                    sample_weight=float(weak_sample_weight),
                    is_weak=True,
                )
            )

    return SplitIndex(
        split=split,
        videos=videos,
        labeled_samples=sorted(labeled_samples, key=lambda sample: (sample.video_name, sample.frame_idx)),
        weak_samples=sorted(weak_samples, key=lambda sample: (sample.video_name, sample.frame_idx)),
        warnings=warnings,
    )


def build_all_split_indices(
    config: ProjectConfig,
    labels_root: Path,
    frames_root: Path,
    masks_root: Path,
    bbox_margin: float,
    weak_sample_weight: float,
    min_box_size: float = 1.0,
) -> dict[str, SplitIndex]:
    return {
        split: build_split_index(
            config=config,
            split=split,
            labels_root=labels_root,
            frames_root=frames_root,
            masks_root=masks_root,
            bbox_margin=bbox_margin,
            weak_sample_weight=weak_sample_weight,
            min_box_size=min_box_size,
        )
        for split in SPLIT_NAMES
    }


def summarize_split_indices(split_indices: Mapping[str, SplitIndex]) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for split in SPLIT_NAMES:
        index = split_indices[split]
        payload[split] = {
            "videos": list(index.videos),
            "labeled_sample_count": len(index.labeled_samples),
            "weak_sample_count": len(index.weak_samples),
            "warnings": list(index.warnings),
        }
    return payload


def validate_mutual_exclusion(split_indices: Mapping[str, SplitIndex]) -> None:
    for split in SPLIT_NAMES:
        index = split_indices[split]
        labeled_ids = {sample.sample_id for sample in index.labeled_samples}
        weak_ids = {sample.sample_id for sample in index.weak_samples}
        overlap = sorted(labeled_ids & weak_ids)
        if overlap:
            preview = ", ".join(overlap[:10])
            if len(overlap) > 10:
                preview += ", ..."
            raise ValueError(
                f"Labeled data and SAM2 weak samples overlap in split '{split}': {preview}"
            )


def print_split_summaries(split_indices: Mapping[str, SplitIndex]) -> None:
    print("=" * 80)
    print("Config-driven detector splits")
    print("=" * 80)
    for split in SPLIT_NAMES:
        index = split_indices[split]
        print(f"\n[{split.upper()}]")
        print(f"Videos ({len(index.videos)}):")
        for video_name in index.videos:
            print(f"  - {video_name}")
        print(f"Labeled samples: {len(index.labeled_samples)}")
        print(f"Weak samples: {len(index.weak_samples)}")
        print(f"Warnings: {len(index.warnings)}")


def load_image_rgb(image_path: str) -> np.ndarray:
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def build_albumentations_pipeline(
    config: Mapping[str, Any] | None,
    *,
    deterministic_only: bool = False,
) -> A.Compose:
    config = dict(config or {})
    transforms_cfg = list(config.get("transforms", []))
    transforms: list[Any] = []
    bbox_min_area = float(config.get("bbox_min_area", 0.0))
    bbox_min_visibility = float(config.get("bbox_min_visibility", 0.0))
    deterministic_transforms = {"Resize", "LongestMaxSize", "PadIfNeeded", "CenterCrop"}

    for transform_cfg in transforms_cfg:
        if not isinstance(transform_cfg, Mapping):
            continue
        name = str(transform_cfg.get("name", "")).strip()
        if not name:
            continue
        if deterministic_only and name not in deterministic_transforms:
            raise ValueError(
                f"Non-deterministic transform {name!r} is not allowed in eval_preprocess."
            )
        params = {key: value for key, value in transform_cfg.items() if key != "name"}
        if name == "HorizontalFlip":
            transforms.append(A.HorizontalFlip(**params))
        elif name == "Affine":
            transforms.append(A.Affine(**params))
        elif name == "RandomBrightnessContrast":
            transforms.append(A.RandomBrightnessContrast(**params))
        elif name == "GaussNoise":
            transforms.append(A.GaussNoise(**params))
        elif name == "MotionBlur":
            transforms.append(A.MotionBlur(**params))
        elif name == "Resize":
            transforms.append(A.Resize(**params))
        elif name == "LongestMaxSize":
            transforms.append(A.LongestMaxSize(**params))
        elif name == "PadIfNeeded":
            transforms.append(A.PadIfNeeded(**params))
        elif name == "RandomCrop":
            transforms.append(A.RandomCrop(**params))
        elif name == "CenterCrop":
            transforms.append(A.CenterCrop(**params))
        else:
            raise ValueError(f"Unsupported albumentations transform name: {name}")

    return A.Compose(
        transforms=transforms,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels", "bbox_roles"],
            min_area=bbox_min_area,
            min_visibility=bbox_min_visibility,
            clip=True,
            filter_invalid_bboxes=True,
        ),
    )


class RamImageStore:
    def __init__(self, preload_images: bool = True) -> None:
        self.preload_images = bool(preload_images)
        self._images: dict[str, np.ndarray] = {}

    def preload(self, samples: Iterable[DetectionSample]) -> dict[str, float]:
        start_time = time.perf_counter()
        image_count = 0
        image_bytes = 0
        if self.preload_images:
            unique_paths = sorted({sample.image_path for sample in samples})
            for image_path in unique_paths:
                image = load_image_rgb(image_path)
                self._images[image_path] = image
                image_count += 1
                image_bytes += int(image.nbytes)
        duration = time.perf_counter() - start_time
        return {
            "cache_build_seconds": duration,
            "cached_image_count": float(image_count),
            "cached_image_bytes": float(image_bytes),
        }

    def get(self, image_path: str) -> np.ndarray:
        if image_path in self._images:
            return self._images[image_path].copy()
        return load_image_rgb(image_path)


class DetectionDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[DetectionSample],
        image_store: RamImageStore,
        transform: A.Compose | None = None,
    ) -> None:
        self.samples = list(samples)
        self.image_store = image_store
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = self.image_store.get(sample.image_path)
        boxes = [list(box) for box in sample.boxes]
        roi_boxes = [list(box) for box in sample.roi_boxes]
        class_labels = [int(label) for label in sample.labels]

        if self.transform is not None:
            combined_boxes = boxes + roi_boxes
            bbox_roles = ["target"] * len(boxes) + ["roi"] * len(roi_boxes)
            transformed = self.transform(
                image=image,
                bboxes=combined_boxes,
                class_labels=[1] * len(combined_boxes),
                bbox_roles=bbox_roles,
            )
            image = transformed["image"]
            transformed_boxes = [list(box) for box in transformed["bboxes"]]
            transformed_roles = list(transformed["bbox_roles"])
            boxes = [box for box, role in zip(transformed_boxes, transformed_roles) if role == "target"]
            roi_boxes = [box for box, role in zip(transformed_boxes, transformed_roles) if role == "roi"]
            class_labels = [1] * len(boxes)

        if not boxes:
            boxes = [sample.boxes[0]]
        if not roi_boxes:
            roi_boxes = [sample.roi_boxes[0]]
            class_labels = [sample.labels[0]]

        image_tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255.0
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        roi_boxes_tensor = torch.as_tensor(roi_boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(class_labels, dtype=torch.int64)
        area_tensor = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
        target = {
            "boxes": boxes_tensor,
            "roi_boxes": roi_boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area_tensor,
            "iscrowd": torch.zeros((boxes_tensor.shape[0],), dtype=torch.int64),
            "sample_weight": torch.tensor(float(sample.sample_weight), dtype=torch.float32),
            "is_weak": torch.tensor(1 if sample.is_weak else 0, dtype=torch.int64),
            "sample_id": sample.sample_id,
            "video_name": sample.video_name,
            "frame_idx": sample.frame_idx,
            "source": sample.source,
        }
        return {"image": image_tensor, "target": target}


def detection_collate_fn(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    images = [item["image"] for item in batch]
    targets = [item["target"] for item in batch]
    return {"images": images, "targets": targets}


def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def select_samples_for_prediction(
    split_indices: Mapping[str, SplitIndex],
    split: str,
    video_name: str | None = None,
) -> list[DetectionSample]:
    index = split_indices[split]
    samples = list(index.labeled_samples)
    if split == "train":
        samples.extend(index.weak_samples)
    if video_name is None:
        return sorted(samples, key=lambda sample: (sample.video_name, sample.frame_idx))
    return [sample for sample in samples if sample.video_name == video_name]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
