from __future__ import annotations

import math
from typing import Sequence, Tuple

import cv2
import numpy as np


def expand_window_about_center(
    window: Sequence[int],
    frame_w: int,
    frame_h: int,
    margin_scale: float,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in window]
    width = max(1.0, float(x2 - x1))
    height = max(1.0, float(y2 - y1))
    scale = max(1.0, 1.0 + 2.0 * float(margin_scale))
    target_w = min(float(frame_w), width * scale)
    target_h = min(float(frame_h), height * scale)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    crop_w = max(1, int(math.ceil(target_w)))
    crop_h = max(1, int(math.ceil(target_h)))
    crop_x1 = int(round(cx - 0.5 * crop_w))
    crop_y1 = int(round(cy - 0.5 * crop_h))
    crop_x2 = crop_x1 + crop_w
    crop_y2 = crop_y1 + crop_h

    if crop_x1 < 0:
        crop_x2 -= crop_x1
        crop_x1 = 0
    if crop_y1 < 0:
        crop_y2 -= crop_y1
        crop_y1 = 0
    if crop_x2 > frame_w:
        shift = crop_x2 - frame_w
        crop_x1 = max(0, crop_x1 - shift)
        crop_x2 = frame_w
    if crop_y2 > frame_h:
        shift = crop_y2 - frame_h
        crop_y1 = max(0, crop_y1 - shift)
        crop_y2 = frame_h
    return int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)


def fit_roi_to_aspect(
    roi: Sequence[int],
    frame_w: int,
    frame_h: int,
    target_aspect: float,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in roi]
    roi_w = max(1.0, float(x2 - x1))
    roi_h = max(1.0, float(y2 - y1))
    target_w = roi_w
    target_h = roi_h
    if target_w / target_h < target_aspect:
        target_w = target_h * target_aspect
    else:
        target_h = target_w / target_aspect

    crop_w = min(frame_w, max(1, int(math.ceil(target_w))))
    crop_h = min(frame_h, max(1, int(math.ceil(target_h))))
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    crop_x1 = int(round(cx - 0.5 * crop_w))
    crop_y1 = int(round(cy - 0.5 * crop_h))
    crop_x2 = crop_x1 + crop_w
    crop_y2 = crop_y1 + crop_h

    if crop_x1 < 0:
        crop_x2 -= crop_x1
        crop_x1 = 0
    if crop_y1 < 0:
        crop_y2 -= crop_y1
        crop_y1 = 0
    if crop_x2 > frame_w:
        shift = crop_x2 - frame_w
        crop_x1 = max(0, crop_x1 - shift)
        crop_x2 = frame_w
    if crop_y2 > frame_h:
        shift = crop_y2 - frame_h
        crop_y1 = max(0, crop_y1 - shift)
        crop_y2 = frame_h
    return int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)


def crop_and_resize_with_aspect(
    image_rgb: np.ndarray,
    roi: Sequence[int],
    output_size: Tuple[int, int],
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    output_w, output_h = output_size
    frame_h, frame_w = image_rgb.shape[:2]
    source_window = fit_roi_to_aspect(roi, frame_w, frame_h, output_w / float(output_h))
    sx1, sy1, sx2, sy2 = source_window
    crop = image_rgb[sy1:sy2, sx1:sx2]
    if crop.size == 0:
        raise ValueError(f"Empty ROI crop after aspect fit: {source_window}")
    resized = cv2.resize(crop, (output_w, output_h), interpolation=cv2.INTER_LINEAR)
    return resized, source_window


def map_points_from_frame_to_window(
    keypoints: np.ndarray,
    visibility: np.ndarray,
    source_window: Sequence[int],
    output_size: Tuple[int, int],
) -> np.ndarray:
    sx1, sy1, sx2, sy2 = [int(v) for v in source_window]
    output_w, output_h = output_size
    source_w = max(1, sx2 - sx1)
    source_h = max(1, sy2 - sy1)
    mapped = keypoints.copy()
    visible = visibility > 0
    if np.any(visible):
        mapped[visible, 0] = (mapped[visible, 0] - float(sx1)) * (output_w / float(source_w))
        mapped[visible, 1] = (mapped[visible, 1] - float(sy1)) * (output_h / float(source_h))
    return mapped


def map_points_from_window_to_frame(
    keypoints: np.ndarray,
    source_window: Sequence[int],
    input_size: Tuple[int, int],
) -> np.ndarray:
    sx1, sy1, sx2, sy2 = [int(v) for v in source_window]
    input_w, input_h = input_size
    source_w = max(1, sx2 - sx1)
    source_h = max(1, sy2 - sy1)
    mapped = keypoints.copy()
    mapped[:, 0] = mapped[:, 0] * (source_w / float(input_w)) + sx1
    mapped[:, 1] = mapped[:, 1] * (source_h / float(input_h)) + sy1
    return mapped
