#!/usr/bin/env python3
"""
Prepare cached ROI boxes for the HRNet keypoint pipeline.
"""

from __future__ import annotations

import argparse
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from keypoint_HRNet import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_FRAMES_ROOT,
    DEFAULT_LABELS_ROOT,
    DEFAULT_MASKS_ROOT,
    DEFAULT_ROI_CACHE_PATH,
    PROJECT_ROOT,
    SPLIT_NAMES,
    build_all_split_indices,
    load_project_config,
    load_mask_frame_dict,
    make_serializable,
    print_split_summaries,
    sample_cache_key,
    safe_video_name,
    set_seed,
    summarize_membership,
    validate_disjoint_splits,
    write_json,
)


YOLO_WEIGHTS_PATH = str(
    PROJECT_ROOT / "output" / "yolo" / "yolov8m_960_20260322_234227" / "train" / "weights" / "best.pt"
)
YOLO_IMGSZ = 960
YOLO_CONF = 0.15
YOLO_IOU = 0.5
YOLO_MAX_DET = 8
ROI_EXPAND_SCALE = 0.12
MASK_OVERLAP_KEEP_THRESHOLD = 0.02
CENTER_DISTANCE_KEEP_FACTOR = 1.5


class YoloRatDetector:
    def __init__(
        self,
        weights_path: str,
        imgsz: int = YOLO_IMGSZ,
        conf: float = YOLO_CONF,
        iou: float = YOLO_IOU,
        max_det: int = YOLO_MAX_DET,
        batch_size: int = 16,
        device: Optional[str] = None,
    ) -> None:
        self.model = YOLO(weights_path)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.batch_size = max(1, batch_size)
        self.device = device
        self.cache: Dict[Tuple[str, int], Dict[str, Tuple[int, int, int, int]]] = {}

    def _mask_bbox(self, mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    def _mask_centroid(self, mask: np.ndarray) -> Optional[Tuple[float, float]]:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        return float(xs.mean()), float(ys.mean())

    def _box_mask_overlap(self, box: Tuple[int, int, int, int], mask: np.ndarray) -> float:
        x1, y1, x2, y2 = box
        x1 = max(0, min(mask.shape[1], x1))
        x2 = max(0, min(mask.shape[1], x2))
        y1 = max(0, min(mask.shape[0], y1))
        y2 = max(0, min(mask.shape[0], y2))
        if x2 <= x1 or y2 <= y1:
            return 0.0
        crop = mask[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0
        return float(crop.mean())

    def _expand_box(
        self,
        box: Tuple[int, int, int, int],
        image_shape: Tuple[int, int, int],
        scale: float = ROI_EXPAND_SCALE,
    ) -> Tuple[int, int, int, int]:
        h, w = image_shape[:2]
        x1, y1, x2, y2 = box
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        pad_x = int(round(bw * scale))
        pad_y = int(round(bh * scale))
        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(w, x2 + pad_x),
            min(h, y2 + pad_y),
        )

    def _select_rois(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        boxes: Sequence[Tuple[int, int, int, int, float]],
    ) -> Dict[str, Tuple[int, int, int, int]]:
        h, w = image_rgb.shape[:2]
        mask_bbox = self._mask_bbox(mask)
        mask_centroid = self._mask_centroid(mask)

        if not boxes and mask_bbox is not None:
            object_roi = mask_bbox
            augmentation_roi = self._expand_box(object_roi, image_rgb.shape)
            return {"object_roi": object_roi, "augmentation_roi": augmentation_roi}
        if not boxes:
            full_frame = (0, 0, w, h)
            return {"object_roi": full_frame, "augmentation_roi": full_frame}

        scored = []
        for x1, y1, x2, y2, conf in boxes:
            overlap = self._box_mask_overlap((x1, y1, x2, y2), mask) if mask_bbox is not None else 0.0
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            center_dist = 0.0 if mask_centroid is None else math.dist((cx, cy), mask_centroid)
            scored.append(
                {
                    "box": (x1, y1, x2, y2),
                    "conf": conf,
                    "overlap": overlap,
                    "center_dist": center_dist,
                }
            )

        scored.sort(key=lambda item: (item["overlap"], item["conf"], -item["center_dist"]), reverse=True)
        primary = scored[0]
        primary_box = primary["box"]
        primary_size = max(1.0, math.sqrt((primary_box[2] - primary_box[0]) * (primary_box[3] - primary_box[1])))

        selected_boxes = [primary_box]
        for item in scored[1:]:
            box = item["box"]
            cx = 0.5 * (box[0] + box[2])
            cy = 0.5 * (box[1] + box[3])
            if mask_centroid is not None:
                dist = math.dist((cx, cy), mask_centroid)
                if item["overlap"] >= MASK_OVERLAP_KEEP_THRESHOLD or dist <= CENTER_DISTANCE_KEEP_FACTOR * primary_size:
                    selected_boxes.append(box)
            elif item["conf"] >= primary["conf"] * 0.5:
                selected_boxes.append(box)

        if mask_bbox is not None:
            selected_boxes.append(mask_bbox)

        x1 = min(box[0] for box in selected_boxes)
        y1 = min(box[1] for box in selected_boxes)
        x2 = max(box[2] for box in selected_boxes)
        y2 = max(box[3] for box in selected_boxes)
        object_roi = (x1, y1, x2, y2)
        augmentation_roi = self._expand_box(object_roi, image_rgb.shape)
        return {"object_roi": object_roi, "augmentation_roi": augmentation_roi}

    def detect_batch_rois(
        self,
        batch_items: Sequence[Tuple[str, int, np.ndarray, np.ndarray]],
    ) -> Dict[Tuple[str, int], Dict[str, Tuple[int, int, int, int]]]:
        pending = []
        results_map: Dict[Tuple[str, int], Dict[str, Tuple[int, int, int, int]]] = {}

        for video_name, frame_idx, image_rgb, mask in batch_items:
            cache_key = (video_name, frame_idx)
            if cache_key in self.cache:
                results_map[cache_key] = self.cache[cache_key]
                continue
            pending.append((cache_key, image_rgb, mask))

        if not pending:
            return results_map

        model_results = self.model.predict(
            [image_rgb for _, image_rgb, _ in pending],
            verbose=False,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            batch=self.batch_size,
            device=self.device,
        )

        for (cache_key, image_rgb, mask), result in zip(pending, model_results):
            boxes: List[Tuple[int, int, int, int, float]] = []
            if result.boxes is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.detach().cpu().numpy()
                confs = result.boxes.conf.detach().cpu().numpy()
                for coords, conf in zip(xyxy, confs):
                    x1, y1, x2, y2 = [int(round(v)) for v in coords.tolist()]
                    if x2 <= x1 or y2 <= y1:
                        continue
                    boxes.append((x1, y1, x2, y2, float(conf)))
            rois = self._select_rois(image_rgb, mask, boxes)
            self.cache[cache_key] = rois
            results_map[cache_key] = rois

        return results_map


def load_image_rgb(image_path: str) -> np.ndarray:
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def clear_cache_root(cache_root: Path) -> None:
    if cache_root.exists():
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)


def save_cropped_image(
    cache_root: Path,
    video_name: str,
    frame_idx: int,
    image_rgb: np.ndarray,
    roi: Tuple[int, int, int, int],
) -> str:
    x1, y1, x2, y2 = roi
    crop = image_rgb[y1:y2, x1:x2]
    crop_dir = cache_root / "arrays" / safe_video_name(video_name)
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop_path = crop_dir / f"{frame_idx:08d}_image.npy"
    np.save(crop_path, crop)
    return str(crop_path)


def save_cropped_mask(
    cache_root: Path,
    video_name: str,
    frame_idx: int,
    mask: np.ndarray,
    roi: Tuple[int, int, int, int],
) -> str:
    x1, y1, x2, y2 = roi
    crop = mask[y1:y2, x1:x2].astype(bool)
    mask_dir = cache_root / "arrays" / safe_video_name(video_name)
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / f"{frame_idx:08d}_mask.npy"
    np.save(mask_path, crop)
    return str(mask_path)


def save_debug_image(
    cache_root: Path,
    video_name: str,
    frame_idx: int,
    image_rgb: np.ndarray,
    augmentation_roi: Tuple[int, int, int, int],
    object_roi: Tuple[int, int, int, int],
) -> str:
    ax1, ay1, ax2, ay2 = augmentation_roi
    ox1, oy1, ox2, oy2 = object_roi
    crop = image_rgb[ay1:ay2, ax1:ax2].copy()
    rel_x1 = max(0, ox1 - ax1)
    rel_y1 = max(0, oy1 - ay1)
    rel_x2 = max(rel_x1 + 1, ox2 - ax1)
    rel_y2 = max(rel_y1 + 1, oy2 - ay1)
    debug_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    cv2.rectangle(debug_bgr, (rel_x1, rel_y1), (rel_x2 - 1, rel_y2 - 1), (0, 255, 255), 2)
    cv2.putText(
        debug_bgr,
        "object_roi",
        (max(0, rel_x1), max(14, rel_y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    debug_dir = cache_root / "debug" / safe_video_name(video_name)
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / f"{frame_idx:08d}.jpg"
    if not cv2.imwrite(str(debug_path), debug_bgr):
        raise IOError(f"Failed to write debug image: {debug_path}")
    return str(debug_path)


def build_roi_cache(args: argparse.Namespace) -> int:
    config = load_project_config(args.config)
    validate_disjoint_splits(config)
    split_indices = build_all_split_indices(config, args.labels_root, args.frames_root, args.masks_root)
    print_split_summaries(split_indices)
    clear_cache_root(args.cache_path.parent)

    detector = YoloRatDetector(YOLO_WEIGHTS_PATH, device=args.device, batch_size=args.batch_size)
    mask_cache: Dict[str, Dict[int, np.ndarray]] = {}
    roi_cache: Dict[Tuple[str, int], Dict[str, Tuple[int, int, int, int]]] = {}
    crop_paths: Dict[Tuple[str, int], str] = {}
    crop_mask_paths: Dict[Tuple[str, int], str] = {}
    debug_paths: Dict[Tuple[str, int], str] = {}
    unique_samples: Dict[Tuple[str, int], object] = {}
    for split in SPLIT_NAMES:
        for sample in split_indices[split].labeled_samples:
            unique_samples[(sample.video_name, sample.frame_idx)] = sample
        for sample in split_indices[split].weak_samples:
            unique_samples[(sample.video_name, sample.frame_idx)] = sample

    sorted_samples = sorted(unique_samples.items())
    print(f"Preparing ROI cache for {len(sorted_samples)} frames")
    for start in range(0, len(sorted_samples), args.batch_size):
        batch = sorted_samples[start:start + args.batch_size]
        batch_items = []
        for (video_name, frame_idx), sample in batch:
            if sample.mask_path not in mask_cache:
                mask_cache[sample.mask_path] = load_mask_frame_dict(Path(sample.mask_path))
            mask = mask_cache[sample.mask_path][frame_idx]
            image = load_image_rgb(sample.image_path)
            batch_items.append((video_name, frame_idx, image, mask))

        batch_rois = detector.detect_batch_rois(batch_items)
        for video_name, frame_idx, image, mask in batch_items:
            rois = batch_rois[(video_name, frame_idx)]
            roi_cache[(video_name, frame_idx)] = rois
            crop_paths[(video_name, frame_idx)] = save_cropped_image(
                args.cache_path.parent,
                video_name,
                frame_idx,
                image,
                rois["augmentation_roi"],
            )
            crop_mask_paths[(video_name, frame_idx)] = save_cropped_mask(
                args.cache_path.parent,
                video_name,
                frame_idx,
                mask,
                rois["augmentation_roi"],
            )
            debug_paths[(video_name, frame_idx)] = save_debug_image(
                args.cache_path.parent,
                video_name,
                frame_idx,
                image,
                rois["augmentation_roi"],
                rois["object_roi"],
            )
        processed = min(start + len(batch), len(sorted_samples))
        if processed % 50 == 0 or processed == len(sorted_samples):
            print(f"  processed {processed}/{len(sorted_samples)}")

    payload = {
        "created_at": datetime.now().isoformat(),
        "weights_path": YOLO_WEIGHTS_PATH,
        "array_dir": str(args.cache_path.parent / "arrays"),
        "debug_dir": str(args.cache_path.parent / "debug"),
        "entries": {
            sample_cache_key(video_name, frame_idx): {
                "roi": [int(v) for v in rois["augmentation_roi"]],
                "augmentation_roi": [int(v) for v in rois["augmentation_roi"]],
                "object_roi": [int(v) for v in rois["object_roi"]],
                "crop_path": crop_paths[(video_name, frame_idx)],
                "crop_mask_path": crop_mask_paths[(video_name, frame_idx)],
                "debug_path": debug_paths[(video_name, frame_idx)],
            }
            for (video_name, frame_idx), rois in sorted(roi_cache.items())
        },
        "split_summary": summarize_membership(split_indices),
        "args": make_serializable(vars(args)),
    }
    args.cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.cache_path, payload)
    print(f"Wrote ROI cache to {args.cache_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute ROI cache for keypoint training.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--labels-root", type=Path, default=DEFAULT_LABELS_ROOT)
    parser.add_argument("--frames-root", type=Path, default=DEFAULT_FRAMES_ROOT)
    parser.add_argument("--masks-root", type=Path, default=DEFAULT_MASKS_ROOT)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_ROI_CACHE_PATH)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    set_seed(args.seed)
    return build_roi_cache(args)


if __name__ == "__main__":
    raise SystemExit(main())
