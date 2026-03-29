from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from modules.detector_bbox_utils import compute_iou_xyxy


def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for idx in range(precisions.size - 1, 0, -1):
        precisions[idx - 1] = max(precisions[idx - 1], precisions[idx])
    change_points = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[change_points + 1] - recalls[change_points]) * precisions[change_points + 1]))


def compute_detection_metrics(
    predictions: Iterable[dict[str, object]],
    targets: Iterable[dict[str, object]],
    iou_thresholds: Sequence[float] = (0.5, 0.75),
) -> dict[str, float]:
    pred_list = list(predictions)
    tgt_list = list(targets)
    if len(pred_list) != len(tgt_list):
        raise ValueError("predictions and targets must have the same length")

    results: dict[str, float] = {}
    aps: list[float] = []
    ars: list[float] = []
    for threshold in iou_thresholds:
        scored: list[tuple[float, int, float]] = []
        total_gt = 0
        for pred, target in zip(pred_list, tgt_list):
            gt_boxes = [list(map(float, box)) for box in target.get("boxes", [])]
            pred_boxes = [list(map(float, box)) for box in pred.get("boxes", [])]
            pred_scores = [float(score) for score in pred.get("scores", [])]
            total_gt += len(gt_boxes)
            matched = [False] * len(gt_boxes)
            order = np.argsort(np.asarray(pred_scores, dtype=np.float32))[::-1] if pred_scores else np.asarray([], dtype=np.int64)
            for pred_idx in order.tolist():
                box = pred_boxes[pred_idx]
                score = pred_scores[pred_idx]
                best_iou = 0.0
                best_gt_idx = -1
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if matched[gt_idx]:
                        continue
                    iou = compute_iou_xyxy(box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                if best_gt_idx >= 0 and best_iou >= threshold:
                    matched[best_gt_idx] = True
                    scored.append((score, 1, best_iou))
                else:
                    scored.append((score, 0, best_iou))
        if total_gt == 0:
            results[f"ap@{threshold:.2f}"] = 0.0
            results[f"ar@{threshold:.2f}"] = 0.0
            aps.append(0.0)
            ars.append(0.0)
            continue

        scored.sort(key=lambda item: item[0], reverse=True)
        tp = np.asarray([item[1] for item in scored], dtype=np.float32)
        fp = 1.0 - tp
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recalls = tp_cumsum / max(float(total_gt), 1.0)
        precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-8)
        ap = _compute_ap(recalls, precisions) if scored else 0.0
        ar = float(tp_cumsum[-1] / max(float(total_gt), 1.0)) if scored else 0.0
        results[f"ap@{threshold:.2f}"] = float(ap)
        results[f"ar@{threshold:.2f}"] = float(ar)
        aps.append(float(ap))
        ars.append(float(ar))

    if aps:
        results["map"] = float(np.mean(aps))
        results["mar"] = float(np.mean(ars))
    else:
        results["map"] = 0.0
        results["mar"] = 0.0
    return results
