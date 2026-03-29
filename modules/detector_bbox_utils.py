from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def clip_xyxy(box: Sequence[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    return [x1, y1, x2, y2]


def expand_xyxy(box: Sequence[float], margin: float, width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    expanded = [
        x1 - float(margin),
        y1 - float(margin),
        x2 + float(margin),
        y2 + float(margin),
    ]
    return clip_xyxy(expanded, width=width, height=height)


def is_valid_xyxy(box: Sequence[float], min_size: float = 1.0) -> bool:
    x1, y1, x2, y2 = [float(v) for v in box]
    return (x2 - x1) >= min_size and (y2 - y1) >= min_size


def xyxy_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def keypoints_to_xyxy(
    keypoints: Iterable[Sequence[float]],
    visibility: Iterable[int] | None,
    width: int,
    height: int,
    margin: float,
    min_size: float = 1.0,
) -> list[float] | None:
    visible_points: list[tuple[float, float]] = []
    if visibility is None:
        for point in keypoints:
            x, y = float(point[0]), float(point[1])
            if np.isnan(x) or np.isnan(y):
                continue
            visible_points.append((x, y))
    else:
        for point, visible in zip(keypoints, visibility):
            if int(visible) <= 0:
                continue
            x, y = float(point[0]), float(point[1])
            if np.isnan(x) or np.isnan(y):
                continue
            visible_points.append((x, y))

    if not visible_points:
        return None

    xs = [point[0] for point in visible_points]
    ys = [point[1] for point in visible_points]
    box = [
        min(xs) - float(margin),
        min(ys) - float(margin),
        max(xs) + float(margin),
        max(ys) + float(margin),
    ]
    box = clip_xyxy(box, width=width, height=height)
    if not is_valid_xyxy(box, min_size=min_size):
        return None
    return box


def mask_to_xyxy(
    mask: np.ndarray,
    width: int,
    height: int,
    min_size: float = 1.0,
) -> list[float] | None:
    arr = np.asarray(mask).astype(bool)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        return None
    ys, xs = np.where(arr)
    if xs.size == 0 or ys.size == 0:
        return None
    box = [
        float(xs.min()),
        float(ys.min()),
        float(xs.max() + 1),
        float(ys.max() + 1),
    ]
    box = clip_xyxy(box, width=width, height=height)
    if not is_valid_xyxy(box, min_size=min_size):
        return None
    return box


def compute_iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    union_area = xyxy_area(box_a) + xyxy_area(box_b) - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area
