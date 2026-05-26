#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import importlib.util
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

try:
    from scipy.interpolate import CubicSpline
    from scipy.signal import savgol_filter
except Exception:
    CubicSpline = None
    savgol_filter = None

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "keypoints"))

def _load_rtmpose_module() -> ModuleType:
    module_path = PROJECT_ROOT / "scripts" / "keypoints" / "RTMPose.py"
    module_name = "rtmpose_script"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load RTMPose module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


rtm = _load_rtmpose_module()

from modules.keypoint_rtmpose_predict_common import (  # noqa: E402
    load_model_from_checkpoint_for_inference,
    predict_dataset,
)


@dataclass
class SplitPayload:
    split_name: str
    bodyparts: list[str]
    sample_keys: list[tuple[str, int]]
    sample_videos: list[str]
    sample_frames: np.ndarray
    crop_boxes: np.ndarray
    pred_xy: np.ndarray
    pred_score: np.ndarray
    pred_visibility: np.ndarray
    gt_xy: np.ndarray
    gt_visibility: np.ndarray
    raw_predictions: list[dict[str, Any]]
    metric_eval_crop_space: dict[str, float]


@dataclass
class DenseContextPayload:
    split_name: str
    bodyparts: list[str]
    sample_keys: list[tuple[str, int]]
    sample_videos: list[str]
    sample_frames: np.ndarray
    pred_xy: np.ndarray
    pred_score: np.ndarray
    pred_visibility: np.ndarray
    eval_indices: np.ndarray


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _ensure_paths(args_ns: argparse.Namespace) -> argparse.Namespace:
    path_fields = {
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
        "detector_checkpoint",
        "init_checkpoint",
        "pretrained_backbone_checkpoint",
    }
    for field in path_fields:
        value = getattr(args_ns, field, None)
        if value is None or value == "":
            continue
        if isinstance(value, Path):
            continue
        setattr(args_ns, field, Path(value))
    return args_ns


def _load_run_args(checkpoint_path: Path) -> argparse.Namespace:
    run_dir = checkpoint_path.parent
    run_config_path = run_dir / "run_config.yaml"
    if not run_config_path.is_file():
        raise FileNotFoundError(f"Missing run_config.yaml next to checkpoint: {run_config_path}")
    payload = rtm.load_yaml_file(run_config_path)
    saved_args = payload.get("args", {}) if isinstance(payload, Mapping) else {}
    if not isinstance(saved_args, Mapping):
        raise ValueError(f"Invalid args payload in {run_config_path}")
    args_ns = argparse.Namespace(**dict(saved_args))
    args_ns = _ensure_paths(args_ns)

    defaults = {
        "detector_model_name": "default",
        "detector_score_threshold": 0.5,
        "detector_batch_size": 16,
        "seed": 42,
        "auto_val_fraction": 0.1,
        "crop_expand_scale": 0.15,
        "bbox_margin": 20.0,
        "top_down_margin": 0,
        "top_down_crop_with_context": True,
        "workers": 4,
        "prefetch_factor": 2,
        "pin_memory": False,
        "persistent_workers": True,
        "preload_images": True,
        "preload_masks": True,
        "weak_sample_weight": 0.0,
        "mask_select_policy": "first",
        "weak_mask_iou_thresh": 0.0,
        "eval_batch_size": 16,
        "device": "cuda:0",
        "detector_device": "",
        "model_name": "default",
    }
    for key, value in defaults.items():
        if not hasattr(args_ns, key):
            setattr(args_ns, key, value)
    return args_ns


def _build_runtime_args(cli_args: argparse.Namespace) -> argparse.Namespace:
    runtime = _load_run_args(cli_args.checkpoint)
    runtime.checkpoint = cli_args.checkpoint
    runtime.device = cli_args.device or runtime.device
    runtime.detector_device = cli_args.detector_device or runtime.detector_device
    if cli_args.detector_checkpoint is not None:
        runtime.detector_checkpoint = cli_args.detector_checkpoint
    runtime.eval_batch_size = int(cli_args.eval_batch_size)
    runtime.workers = int(cli_args.workers)
    runtime.prefetch_factor = int(cli_args.prefetch_factor)
    runtime.pin_memory = bool(cli_args.pin_memory)
    runtime.persistent_workers = bool(cli_args.persistent_workers)
    runtime.preload_images = bool(cli_args.preload_images)
    runtime.preload_masks = bool(cli_args.preload_masks)
    runtime.seed = int(cli_args.seed if cli_args.seed is not None else runtime.seed)
    runtime.auto_val_fraction = float(cli_args.auto_val_fraction if cli_args.auto_val_fraction is not None else runtime.auto_val_fraction)
    runtime.command = "eval"
    runtime = _ensure_paths(runtime)

    if getattr(runtime, "project_config", None) is None:
        runtime.project_config = PROJECT_ROOT / "config.yaml"
    if getattr(runtime, "labels_root", None) is None:
        runtime.labels_root = PROJECT_ROOT / "input" / "labeled-data"
    if getattr(runtime, "labeled_frames_root", None) is None:
        runtime.labeled_frames_root = PROJECT_ROOT / "output" / "sam2" / "DLC_frames"
    if getattr(runtime, "frames_root", None) is None:
        runtime.frames_root = PROJECT_ROOT / "output" / "sam2" / "final"
    if getattr(runtime, "masks_root", None) is None:
        runtime.masks_root = PROJECT_ROOT / "output" / "sam2" / "sam2_pickle_filtered"
    if getattr(runtime, "detector_checkpoint", None) is None:
        raise ValueError("Detector checkpoint was not found in run_config.yaml args; provide --detector-checkpoint.")

    return runtime


def _resolve_model_cfg(runtime_args: argparse.Namespace) -> dict[str, Any]:
    candidate = runtime_args.checkpoint.parent / "resolved_model_config.yaml"
    if candidate.is_file():
        payload = rtm.load_yaml_file(candidate)
        if isinstance(payload, Mapping) and "model" in payload:
            return dict(payload)
    candidate = runtime_args.checkpoint.parent / "model_config.yaml"
    if candidate.is_file():
        payload = rtm.load_yaml_file(candidate)
        if isinstance(payload, Mapping) and "model" in payload:
            return dict(payload)
    return dict(rtm.resolve_model_config(runtime_args))


def _build_dataset_for_split(
    runtime_args: argparse.Namespace,
    model_cfg: Mapping[str, Any],
    project_cfg: Any,
    store: Any,
    detector_boxes: Mapping[tuple[str, int], dict[str, Any]],
    filtered_indices: Mapping[str, Any],
    split_name: str,
) -> tuple[Any, list[object]]:
    samples = rtm.select_samples_for_split(filtered_indices, split_name, None)
    data_train_cfg = dict(model_cfg.get("data", {}).get("train", {}))
    bbox_margin = float(model_cfg.get("data", {}).get("bbox_margin", getattr(runtime_args, "bbox_margin", 20.0)))
    crop_cfg = dict(data_train_cfg.get("top_down_crop", {}))
    crop_cfg.setdefault("margin", int(getattr(runtime_args, "top_down_margin", 0)))
    crop_cfg.setdefault("crop_with_context", bool(getattr(runtime_args, "top_down_crop_with_context", True)))

    dataset = rtm.RTMPoseDataset(
        samples,
        store,
        detector_boxes,
        project_cfg.bodyparts,
        project_cfg.skeleton,
        project_cfg.left_right_symmetry,
        tuple(model_cfg["model"]["heads"]["bodypart"].get("input_size", [256, 256])),
        model_cfg.get("image_mean", [0.485, 0.456, 0.406]),
        model_cfg.get("image_std", [0.229, 0.224, 0.225]),
        float(runtime_args.crop_expand_scale),
        bbox_margin=bbox_margin,
        train_aug_cfg=data_train_cfg,
        crop_cfg=crop_cfg,
        train_mode=False,
        include_weak=False,
        use_masks=float(getattr(runtime_args, "weak_sample_weight", 1.0)) > 0.0,
        mask_select_policy=str(getattr(runtime_args, "mask_select_policy", "first")),
        weak_mask_iou_thresh=float(getattr(runtime_args, "weak_mask_iou_thresh", 0.0)),
    )
    return dataset, samples


def _predict_split(
    split_name: str,
    runtime_args: argparse.Namespace,
    model_cfg: Mapping[str, Any],
    project_cfg: Any,
    store: Any,
    detector_boxes: Mapping[tuple[str, int], dict[str, Any]],
    filtered_indices: Mapping[str, Any],
    model: Any,
    device: Any,
) -> SplitPayload:
    dataset, samples = _build_dataset_for_split(
        runtime_args,
        model_cfg,
        project_cfg,
        store,
        detector_boxes,
        filtered_indices,
        split_name,
    )
    dataloader = rtm.build_dataloader(
        dataset,
        batch_size=int(runtime_args.eval_batch_size),
        shuffle=False,
        drop_last=False,
        workers=int(runtime_args.workers),
        pin_memory=bool(runtime_args.pin_memory),
        persistent_workers=bool(runtime_args.persistent_workers),
        prefetch_factor=int(runtime_args.prefetch_factor) if int(runtime_args.workers) > 0 else None,
    )
    predictions = predict_dataset(
        model=model,
        dataloader=dataloader,
    )
    metric_eval_crop_space = rtm.evaluate_pose_model(
        model,
        dataloader,
        device,
        float(getattr(runtime_args, "lambda_pose", 1.0)),
        float(getattr(runtime_args, "lambda_mask", 0.5)),
        float(getattr(runtime_args, "lambda_visibility", 1.0)),
    )

    gt_map: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for sample in samples:
        sample_keypoints = getattr(sample, "keypoints", None)
        sample_visibility = getattr(sample, "visibility", None)
        sample_video_name = getattr(sample, "video_name", None)
        sample_frame_idx = getattr(sample, "frame_idx", None)
        if sample_keypoints is None or sample_visibility is None or sample_video_name is None or sample_frame_idx is None:
            continue
        key = (str(sample_video_name), int(sample_frame_idx))
        gt_map[key] = (
            np.asarray(sample_keypoints, dtype=np.float32)[..., :2],
            np.asarray(sample_visibility, dtype=np.float32),
        )

    sample_keys: list[tuple[str, int]] = []
    sample_videos: list[str] = []
    sample_frames: list[int] = []
    crop_boxes: list[list[float]] = []
    pred_xy: list[np.ndarray] = []
    pred_score: list[np.ndarray] = []
    pred_visibility: list[np.ndarray] = []
    gt_xy: list[np.ndarray] = []
    gt_visibility: list[np.ndarray] = []
    valid_predictions: list[dict[str, Any]] = []
    bodyparts = list(project_cfg.bodyparts)

    for pred in predictions:
        key = (str(pred["video_name"]), int(pred["frame_idx"]))
        if key not in gt_map:
            continue
        gt_pts, gt_vis = gt_map[key]
        kp_map = pred["keypoints"]
        xy = np.zeros((len(bodyparts), 2), dtype=np.float32)
        score = np.zeros((len(bodyparts),), dtype=np.float32)
        vis_score = np.zeros((len(bodyparts),), dtype=np.float32)
        for idx, bodypart in enumerate(bodyparts):
            rec = kp_map[bodypart]
            xy[idx, 0] = float(rec["x"])
            xy[idx, 1] = float(rec["y"])
            score[idx] = float(rec.get("score", 0.0))
            vis_score[idx] = float(rec.get("visibility_score", 0.0))

        sample_keys.append(key)
        sample_videos.append(key[0])
        sample_frames.append(key[1])
        crop_boxes.append([float(v) for v in pred.get("crop_box", [0, 0, 0, 0])])
        pred_xy.append(xy)
        pred_score.append(score)
        pred_visibility.append(vis_score)
        gt_xy.append(gt_pts)
        gt_visibility.append(gt_vis)
        valid_predictions.append(pred)

    return SplitPayload(
        split_name=split_name,
        bodyparts=bodyparts,
        sample_keys=sample_keys,
        sample_videos=sample_videos,
        sample_frames=np.asarray(sample_frames, dtype=np.int32),
        crop_boxes=np.asarray(crop_boxes, dtype=np.float32),
        pred_xy=np.asarray(pred_xy, dtype=np.float32),
        pred_score=np.asarray(pred_score, dtype=np.float32),
        pred_visibility=np.asarray(pred_visibility, dtype=np.float32),
        gt_xy=np.asarray(gt_xy, dtype=np.float32),
        gt_visibility=np.asarray(gt_visibility, dtype=np.float32),
        raw_predictions=valid_predictions,
        metric_eval_crop_space={k: float(v) for k, v in metric_eval_crop_space.items()},
    )


def _collect_context_samples_for_split(
    split_payload: SplitPayload,
    runtime_args: argparse.Namespace,
    split_name: str,
    context_radius: int,
    context_step: int,
) -> list[Any]:
    by_video: dict[str, set[int]] = {}
    step = max(1, int(context_step))
    radius = max(0, int(context_radius))
    for video_name, frame_idx in split_payload.sample_keys:
        by_video.setdefault(video_name, set())
        start = max(0, int(frame_idx) - radius)
        end = int(frame_idx) + radius
        for f in range(start, end + 1, step):
            by_video[video_name].add(int(f))

    samples: list[Any] = []
    for video_name, frame_set in by_video.items():
        frames_dir = runtime_args.frames_root / str(video_name)
        frame_files = rtm.list_frame_files(frames_dir)
        for frame_idx in sorted(frame_set):
            image_path = frame_files.get(int(frame_idx))
            if image_path is None:
                continue
            samples.append(
                rtm.WeakSample(
                    split=split_name,
                    video_name=str(video_name),
                    frame_idx=int(frame_idx),
                    image_path=str(image_path),
                    mask_path="",
                )
            )
    samples.sort(key=lambda s: (str(s.video_name), int(s.frame_idx)))
    return samples


def _predict_dense_context_for_split(
    split_payload: SplitPayload,
    runtime_args: argparse.Namespace,
    model_cfg: Mapping[str, Any],
    project_cfg: Any,
    store: Any,
    model: Any,
    *,
    context_radius: int,
    context_step: int,
) -> DenseContextPayload | None:
    context_samples = _collect_context_samples_for_split(
        split_payload,
        runtime_args,
        split_name=split_payload.split_name,
        context_radius=context_radius,
        context_step=context_step,
    )
    if not context_samples:
        return None

    _ = store.preload(
        context_samples,
        preload_images=bool(runtime_args.preload_images),
        preload_masks=False,
    )
    detector = rtm.load_detector(
        runtime_args.detector_checkpoint.parent,
        runtime_args.detector_checkpoint,
        device=rtm.resolve_device(str(runtime_args.detector_device or runtime_args.device)),
    )
    detector_boxes = rtm.prepare_detector_boxes(
        detector,
        store,
        context_samples,
        detector_batch_desc=f"context_detector_{split_payload.split_name}",
        batch_size=int(runtime_args.detector_batch_size),
        score_threshold=float(runtime_args.detector_score_threshold),
    )
    context_samples = [
        sample
        for sample in context_samples
        if (str(sample.video_name), int(sample.frame_idx)) in detector_boxes
    ]
    if not context_samples:
        return None

    data_train_cfg = dict(model_cfg.get("data", {}).get("train", {}))
    bbox_margin = float(model_cfg.get("data", {}).get("bbox_margin", getattr(runtime_args, "bbox_margin", 20.0)))
    crop_cfg = dict(data_train_cfg.get("top_down_crop", {}))
    crop_cfg.setdefault("margin", int(getattr(runtime_args, "top_down_margin", 0)))
    crop_cfg.setdefault("crop_with_context", bool(getattr(runtime_args, "top_down_crop_with_context", True)))
    dataset = rtm.RTMPoseDataset(
        context_samples,
        store,
        detector_boxes,
        project_cfg.bodyparts,
        project_cfg.skeleton,
        project_cfg.left_right_symmetry,
        tuple(model_cfg["model"]["heads"]["bodypart"].get("input_size", [256, 256])),
        model_cfg.get("image_mean", [0.485, 0.456, 0.406]),
        model_cfg.get("image_std", [0.229, 0.224, 0.225]),
        float(runtime_args.crop_expand_scale),
        bbox_margin=bbox_margin,
        train_aug_cfg=data_train_cfg,
        crop_cfg=crop_cfg,
        train_mode=False,
        include_weak=False,
        use_masks=False,
        mask_select_policy=str(getattr(runtime_args, "mask_select_policy", "first")),
        weak_mask_iou_thresh=float(getattr(runtime_args, "weak_mask_iou_thresh", 0.0)),
    )
    dataloader = rtm.build_dataloader(
        dataset,
        batch_size=int(runtime_args.eval_batch_size),
        shuffle=False,
        drop_last=False,
        workers=int(runtime_args.workers),
        pin_memory=bool(runtime_args.pin_memory),
        persistent_workers=bool(runtime_args.persistent_workers),
        prefetch_factor=int(runtime_args.prefetch_factor) if int(runtime_args.workers) > 0 else None,
    )
    predictions = predict_dataset(
        model=model,
        dataloader=dataloader,
    )

    bodyparts = list(project_cfg.bodyparts)
    sample_keys: list[tuple[str, int]] = []
    sample_videos: list[str] = []
    sample_frames: list[int] = []
    pred_xy: list[np.ndarray] = []
    pred_score: list[np.ndarray] = []
    pred_visibility: list[np.ndarray] = []
    for pred in predictions:
        key = (str(pred["video_name"]), int(pred["frame_idx"]))
        kp_map = pred["keypoints"]
        xy = np.zeros((len(bodyparts), 2), dtype=np.float32)
        score = np.zeros((len(bodyparts),), dtype=np.float32)
        vis_score = np.zeros((len(bodyparts),), dtype=np.float32)
        for idx, bodypart in enumerate(bodyparts):
            rec = kp_map[bodypart]
            xy[idx, 0] = float(rec["x"])
            xy[idx, 1] = float(rec["y"])
            score[idx] = float(rec.get("score", 0.0))
            vis_score[idx] = float(rec.get("visibility_score", 0.0))
        sample_keys.append(key)
        sample_videos.append(key[0])
        sample_frames.append(key[1])
        pred_xy.append(xy)
        pred_score.append(score)
        pred_visibility.append(vis_score)

    merged: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for i, key in enumerate(sample_keys):
        merged[key] = (pred_xy[i], pred_score[i], pred_visibility[i])
    eval_lookup = {key: i for i, key in enumerate(split_payload.sample_keys)}
    for key in split_payload.sample_keys:
        if key in merged:
            continue
        src_idx = eval_lookup[key]
        merged[key] = (
            split_payload.pred_xy[src_idx],
            split_payload.pred_score[src_idx],
            split_payload.pred_visibility[src_idx],
        )

    merged_keys = sorted(merged.keys(), key=lambda x: (x[0], x[1]))
    sample_keys = merged_keys
    sample_videos = [k[0] for k in merged_keys]
    sample_frames = [k[1] for k in merged_keys]
    pred_xy = [merged[k][0] for k in merged_keys]
    pred_score = [merged[k][1] for k in merged_keys]
    pred_visibility = [merged[k][2] for k in merged_keys]
    key_to_idx = {key: i for i, key in enumerate(sample_keys)}
    eval_indices = [key_to_idx[key] for key in split_payload.sample_keys]

    return DenseContextPayload(
        split_name=split_payload.split_name,
        bodyparts=bodyparts,
        sample_keys=sample_keys,
        sample_videos=sample_videos,
        sample_frames=np.asarray(sample_frames, dtype=np.int32),
        pred_xy=np.asarray(pred_xy, dtype=np.float32),
        pred_score=np.asarray(pred_score, dtype=np.float32),
        pred_visibility=np.asarray(pred_visibility, dtype=np.float32),
        eval_indices=np.asarray(eval_indices, dtype=np.int32),
    )


def _compute_metrics(pred_xy: np.ndarray, gt_xy: np.ndarray, gt_visibility: np.ndarray) -> dict[str, float]:
    visible = gt_visibility > 0.5
    count = int(visible.sum())
    if count <= 0:
        return {"visible_count": 0, "mean_pixel_error": 0.0, "rmse_unfiltered": 0.0, "rmse_90": 0.0}
    d = pred_xy - gt_xy
    dv = d[visible]
    eu = np.linalg.norm(dv, axis=1)
    sq = float((dv ** 2).sum())
    return {
        "visible_count": count,
        "mean_pixel_error": float(eu.mean()),
        "rmse_unfiltered": float(math.sqrt(sq / max(1.0, 2.0 * count))),
        "rmse_90": float(np.quantile(eu, 0.9)),
    }


def _video_orders(videos: Sequence[str], frames: np.ndarray) -> dict[str, list[int]]:
    d: dict[str, list[int]] = {}
    for i, name in enumerate(videos):
        d.setdefault(name, []).append(i)
    for name, arr in d.items():
        d[name] = sorted(arr, key=lambda j: int(frames[j]))
    return d


def _interp_nan_1d(y: np.ndarray) -> np.ndarray:
    x = np.arange(y.shape[0], dtype=np.float32)
    out = y.copy().astype(np.float32)
    mask = np.isfinite(out)
    if int(mask.sum()) == 0:
        return np.zeros_like(out)
    if int(mask.sum()) == 1:
        return np.full_like(out, float(out[mask][0]))
    out[~mask] = np.interp(x[~mask], x[mask], out[mask])
    return out


def _moving_avg_1d(y: np.ndarray, window: int) -> np.ndarray:
    w = max(1, int(window))
    if w <= 1:
        return y.copy()
    if w % 2 == 0:
        w += 1
    pad = w // 2
    ext = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones((w,), dtype=np.float32) / float(w)
    return np.convolve(ext, kernel, mode="valid").astype(np.float32)


def _median_1d(y: np.ndarray, window: int) -> np.ndarray:
    w = max(1, int(window))
    if w <= 1:
        return y.copy()
    if w % 2 == 0:
        w += 1
    pad = w // 2
    ext = np.pad(y, (pad, pad), mode="edge")
    out = np.empty_like(y)
    for i in range(y.shape[0]):
        out[i] = float(np.median(ext[i : i + w]))
    return out


def _savgol_1d(y: np.ndarray, window: int, order: int) -> np.ndarray:
    w = max(3, int(window))
    if w % 2 == 0:
        w += 1
    p = min(max(1, int(order)), w - 1)
    if savgol_filter is not None:
        return np.asarray(savgol_filter(y, window_length=w, polyorder=p, mode="interp"), dtype=np.float32)
    out = np.empty_like(y)
    half = w // 2
    x = np.arange(y.shape[0], dtype=np.float32)
    for i in range(y.shape[0]):
        lo = max(0, i - half)
        hi = min(y.shape[0], i + half + 1)
        xs = x[lo:hi] - float(i)
        ys = y[lo:hi]
        deg = min(p, max(1, xs.shape[0] - 1))
        coef = np.polyfit(xs, ys, deg=deg)
        out[i] = float(np.polyval(coef, 0.0))
    return out


def _cubic_interp_1d(y: np.ndarray, keep: np.ndarray) -> np.ndarray:
    x = np.arange(y.shape[0], dtype=np.float32)
    idx = np.where(keep)[0]
    if idx.shape[0] <= 1:
        return _interp_nan_1d(np.where(keep, y, np.nan))
    if CubicSpline is not None and idx.shape[0] >= 4:
        spline = CubicSpline(x[idx], y[idx], bc_type="natural")
        return spline(x).astype(np.float32)
    return _interp_nan_1d(np.where(keep, y, np.nan))


def _apply_per_video(
    base: np.ndarray,
    conf: np.ndarray,
    videos: Sequence[str],
    frames: np.ndarray,
    fn: Any,
) -> np.ndarray:
    out = base.copy()
    orders = _video_orders(videos, frames)
    for _, idx in orders.items():
        seq = out[idx]
        cseq = conf[idx]
        out[idx] = fn(seq, cseq)
    return out


def _algo_conf_threshold(base: np.ndarray, conf: np.ndarray, videos: Sequence[str], frames: np.ndarray, threshold: float) -> np.ndarray:
    def _seq(seq: np.ndarray, cseq: np.ndarray) -> np.ndarray:
        out = seq.copy()
        t, k, _ = out.shape
        keep = cseq >= float(threshold)
        for j in range(k):
            for d in range(2):
                y = out[:, j, d]
                yn = np.where(keep[:, j], y, np.nan)
                out[:, j, d] = _interp_nan_1d(yn)
        return out
    return _apply_per_video(base, conf, videos, frames, _seq)


def _algo_moving_average(base: np.ndarray, conf: np.ndarray, videos: Sequence[str], frames: np.ndarray, window: int) -> np.ndarray:
    def _seq(seq: np.ndarray, cseq: np.ndarray) -> np.ndarray:
        out = seq.copy()
        t, k, _ = out.shape
        for j in range(k):
            for d in range(2):
                out[:, j, d] = _moving_avg_1d(out[:, j, d], window)
        return out
    return _apply_per_video(base, conf, videos, frames, _seq)


def _algo_ema(base: np.ndarray, conf: np.ndarray, videos: Sequence[str], frames: np.ndarray, alpha: float) -> np.ndarray:
    def _seq(seq: np.ndarray, cseq: np.ndarray) -> np.ndarray:
        out = seq.copy()
        state = out[0].copy()
        a = float(alpha)
        for i in range(1, out.shape[0]):
            state = a * out[i] + (1.0 - a) * state
            out[i] = state
        return out
    return _apply_per_video(base, conf, videos, frames, _seq)


def _algo_median(base: np.ndarray, conf: np.ndarray, videos: Sequence[str], frames: np.ndarray, window: int) -> np.ndarray:
    def _seq(seq: np.ndarray, cseq: np.ndarray) -> np.ndarray:
        out = seq.copy()
        for j in range(out.shape[1]):
            for d in range(2):
                out[:, j, d] = _median_1d(out[:, j, d], window)
        return out
    return _apply_per_video(base, conf, videos, frames, _seq)


def _algo_savgol(base: np.ndarray, conf: np.ndarray, videos: Sequence[str], frames: np.ndarray, window: int, order: int) -> np.ndarray:
    def _seq(seq: np.ndarray, cseq: np.ndarray) -> np.ndarray:
        out = seq.copy()
        for j in range(out.shape[1]):
            for d in range(2):
                out[:, j, d] = _savgol_1d(out[:, j, d], window=window, order=order)
        return out
    return _apply_per_video(base, conf, videos, frames, _seq)


def _algo_linear_interp(base: np.ndarray, conf: np.ndarray, videos: Sequence[str], frames: np.ndarray, threshold: float) -> np.ndarray:
    return _algo_conf_threshold(base, conf, videos, frames, threshold)


def _algo_cubic_interp(base: np.ndarray, conf: np.ndarray, videos: Sequence[str], frames: np.ndarray, threshold: float) -> np.ndarray:
    def _seq(seq: np.ndarray, cseq: np.ndarray) -> np.ndarray:
        out = seq.copy()
        keep = cseq >= float(threshold)
        for j in range(out.shape[1]):
            for d in range(2):
                out[:, j, d] = _cubic_interp_1d(out[:, j, d], keep[:, j])
        return out
    return _apply_per_video(base, conf, videos, frames, _seq)


def _algo_velocity_clip(base: np.ndarray, conf: np.ndarray, videos: Sequence[str], frames: np.ndarray, vmax: float) -> np.ndarray:
    def _seq(seq: np.ndarray, cseq: np.ndarray) -> np.ndarray:
        out = seq.copy()
        vm = float(vmax)
        for i in range(1, out.shape[0]):
            delta = out[i] - out[i - 1]
            mag = np.linalg.norm(delta, axis=1, keepdims=True)
            scale = np.minimum(1.0, vm / np.maximum(1e-6, mag))
            out[i] = out[i - 1] + delta * scale
        return out
    return _apply_per_video(base, conf, videos, frames, _seq)


def _algo_accel_clip(base: np.ndarray, conf: np.ndarray, videos: Sequence[str], frames: np.ndarray, amax: float) -> np.ndarray:
    def _seq(seq: np.ndarray, cseq: np.ndarray) -> np.ndarray:
        out = seq.copy()
        am = float(amax)
        for i in range(2, out.shape[0]):
            v_prev = out[i - 1] - out[i - 2]
            v_now = out[i] - out[i - 1]
            dv = v_now - v_prev
            mag = np.linalg.norm(dv, axis=1, keepdims=True)
            scale = np.minimum(1.0, am / np.maximum(1e-6, mag))
            out[i] = out[i - 1] + v_prev + dv * scale
        return out
    return _apply_per_video(base, conf, videos, frames, _seq)


def _algo_const_vel_extrap(base: np.ndarray, conf: np.ndarray, videos: Sequence[str], frames: np.ndarray, threshold: float) -> np.ndarray:
    def _seq(seq: np.ndarray, cseq: np.ndarray) -> np.ndarray:
        out = seq.copy()
        bad = cseq < float(threshold)
        for i in range(2, out.shape[0]):
            mask = bad[i]
            if not np.any(mask):
                continue
            out[i, mask] = out[i - 1, mask] + (out[i - 1, mask] - out[i - 2, mask])
        for i in range(out.shape[0] - 3, -1, -1):
            mask = bad[i]
            if not np.any(mask):
                continue
            out[i, mask] = 0.5 * (out[i, mask] + (out[i + 1, mask] - (out[i + 2, mask] - out[i + 1, mask])))
        return out
    return _apply_per_video(base, conf, videos, frames, _seq)


def _skeleton_edges(bodyparts: Sequence[str], skeleton: Sequence[Sequence[str]]) -> list[tuple[int, int]]:
    idx = {n: i for i, n in enumerate(bodyparts)}
    out: list[tuple[int, int]] = []
    for e in skeleton:
        if len(e) != 2:
            continue
        a = idx.get(str(e[0]))
        b = idx.get(str(e[1]))
        if a is None or b is None:
            continue
        out.append((a, b))
    return out


def _fit_bone_lengths(train_data: SplitPayload, edges: list[tuple[int, int]]) -> dict[tuple[int, int], float]:
    lengths: dict[tuple[int, int], float] = {}
    vis = train_data.gt_visibility > 0.5
    for a, b in edges:
        m = vis[:, a] & vis[:, b]
        if not np.any(m):
            continue
        d = np.linalg.norm(train_data.gt_xy[m, a] - train_data.gt_xy[m, b], axis=1)
        lengths[(a, b)] = float(np.median(d))
    return lengths


def _algo_bone_length(base: np.ndarray, edges: list[tuple[int, int]], target_len: Mapping[tuple[int, int], float], strength: float) -> np.ndarray:
    out = base.copy()
    s = float(strength)
    for i in range(out.shape[0]):
        for a, b in edges:
            tl = float(target_len.get((a, b), 0.0))
            if tl <= 0.0:
                continue
            va = out[i, a]
            vb = out[i, b]
            d = vb - va
            n = float(np.linalg.norm(d))
            if n < 1e-6:
                continue
            dirv = d / n
            desired = va + dirv * tl
            out[i, b] = (1.0 - s) * vb + s * desired
    return out


def _algo_bone_length_gated(
    base: np.ndarray,
    conf: np.ndarray,
    edges: list[tuple[int, int]],
    target_len: Mapping[tuple[int, int], float],
    strength: float,
    min_conf: float,
) -> np.ndarray:
    out = base.copy()
    s = float(strength)
    for i in range(out.shape[0]):
        for a, b in edges:
            tl = float(target_len.get((a, b), 0.0))
            if tl <= 0.0:
                continue
            ca = float(conf[i, a])
            cb = float(conf[i, b])
            if max(ca, cb) < float(min_conf):
                continue
            if abs(ca - cb) < 0.1:
                continue
            if ca >= cb:
                anchor_idx, move_idx = a, b
                c_anchor, c_move = ca, cb
            else:
                anchor_idx, move_idx = b, a
                c_anchor, c_move = cb, ca
            va = out[i, anchor_idx]
            vm = out[i, move_idx]
            d = vm - va
            n = float(np.linalg.norm(d))
            if n < 1e-6:
                continue
            desired = va + (d / n) * tl
            w = s * max(0.0, min(1.0, c_anchor * (1.0 - c_move)))
            if w <= 1e-6:
                continue
            out[i, move_idx] = (1.0 - w) * vm + w * desired
    return out


def _angle_triplets(edges: list[tuple[int, int]], kcount: int) -> list[tuple[int, int, int]]:
    nbr: dict[int, set[int]] = {i: set() for i in range(kcount)}
    for a, b in edges:
        nbr[a].add(b)
        nbr[b].add(a)
    tri: list[tuple[int, int, int]] = []
    for c, neigh in nbr.items():
        arr = sorted(list(neigh))
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                tri.append((arr[i], c, arr[j]))
    return tri


def _angle(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(math.acos(c))


def _fit_angle_limits(train_data: SplitPayload, triplets: list[tuple[int, int, int]]) -> dict[tuple[int, int, int], tuple[float, float]]:
    out: dict[tuple[int, int, int], tuple[float, float]] = {}
    vis = train_data.gt_visibility > 0.5
    for a, b, c in triplets:
        m = vis[:, a] & vis[:, b] & vis[:, c]
        if int(m.sum()) < 10:
            continue
        angs = []
        for i in np.where(m)[0]:
            angs.append(_angle(train_data.gt_xy[i, a] - train_data.gt_xy[i, b], train_data.gt_xy[i, c] - train_data.gt_xy[i, b]))
        arr = np.asarray(angs, dtype=np.float32)
        out[(a, b, c)] = (float(np.quantile(arr, 0.02)), float(np.quantile(arr, 0.98)))
    return out


def _rotate(v: np.ndarray, theta: float) -> np.ndarray:
    c = float(math.cos(theta))
    s = float(math.sin(theta))
    return np.asarray([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=np.float32)


def _algo_angle_limit(base: np.ndarray, limits: Mapping[tuple[int, int, int], tuple[float, float]], strength: float) -> np.ndarray:
    out = base.copy()
    s = float(strength)
    for i in range(out.shape[0]):
        for (a, b, c), (lo, hi) in limits.items():
            vb = out[i, b]
            v1 = out[i, a] - vb
            v2 = out[i, c] - vb
            ang = _angle(v1, v2)
            target = ang
            if ang < lo:
                target = lo
            elif ang > hi:
                target = hi
            if target == ang:
                continue
            delta = (target - ang) * s
            out[i, c] = vb + _rotate(v2, delta)
    return out


def _algo_angle_limit_gated(
    base: np.ndarray,
    conf: np.ndarray,
    limits: Mapping[tuple[int, int, int], tuple[float, float]],
    strength: float,
    min_center_conf: float,
) -> np.ndarray:
    out = base.copy()
    s = float(strength)
    for i in range(out.shape[0]):
        for (a, b, c), (lo, hi) in limits.items():
            cb = float(conf[i, b])
            if cb < float(min_center_conf):
                continue
            ca = float(conf[i, a])
            cc = float(conf[i, c])
            vb = out[i, b]
            v1 = out[i, a] - vb
            v2 = out[i, c] - vb
            ang = _angle(v1, v2)
            target = ang
            if ang < lo:
                target = lo
            elif ang > hi:
                target = hi
            if target == ang:
                continue
            if ca <= cc:
                move_idx = a
                anchor_vec = v2
                move_vec = v1
                c_anchor = cc
                c_move = ca
            else:
                move_idx = c
                anchor_vec = v1
                move_vec = v2
                c_anchor = ca
                c_move = cc
            if c_anchor < float(min_center_conf):
                continue
            w = s * max(0.0, min(1.0, cb * c_anchor * (1.0 - c_move)))
            if w <= 1e-6:
                continue
            desired_angle = (target - ang) * w
            if move_idx == a:
                out[i, a] = vb + _rotate(move_vec, -desired_angle)
            else:
                out[i, c] = vb + _rotate(move_vec, desired_angle)
    return out


def _frame_dts(frame_indices: np.ndarray, max_dt: float) -> np.ndarray:
    t = int(frame_indices.shape[0])
    if t <= 0:
        return np.zeros((0,), dtype=np.float32)
    dts = np.ones((t,), dtype=np.float32)
    if t == 1:
        return dts
    diffs = np.diff(frame_indices.astype(np.float32))
    diffs = np.clip(diffs, 1.0, float(max_dt))
    dts[1:] = diffs
    return dts


def _algo_zscore(base: np.ndarray, videos: Sequence[str], frames: np.ndarray, zthr: float) -> np.ndarray:
    def _seq(seq: np.ndarray, cseq: np.ndarray) -> np.ndarray:
        out = seq.copy()
        mu = out.mean(axis=0, keepdims=True)
        sd = out.std(axis=0, keepdims=True) + 1e-6
        z = np.abs((out - mu) / sd)
        bad = np.any(z > float(zthr), axis=2)
        for j in range(out.shape[1]):
            for d in range(2):
                y = out[:, j, d]
                yn = np.where(bad[:, j], np.nan, y)
                out[:, j, d] = _interp_nan_1d(yn)
        return out
    return _apply_per_video(base, np.ones(base.shape[:2], dtype=np.float32), videos, frames, _seq)


def _algo_mad(base: np.ndarray, videos: Sequence[str], frames: np.ndarray, zthr: float) -> np.ndarray:
    def _seq(seq: np.ndarray, cseq: np.ndarray) -> np.ndarray:
        out = seq.copy()
        med = np.median(out, axis=0, keepdims=True)
        mad = np.median(np.abs(out - med), axis=0, keepdims=True) + 1e-6
        mz = 0.6745 * np.abs(out - med) / mad
        bad = np.any(mz > float(zthr), axis=2)
        for j in range(out.shape[1]):
            for d in range(2):
                out[:, j, d] = _interp_nan_1d(np.where(bad[:, j], np.nan, out[:, j, d]))
        return out
    return _apply_per_video(base, np.ones(base.shape[:2], dtype=np.float32), videos, frames, _seq)


def _algo_mahalanobis(base: np.ndarray, videos: Sequence[str], frames: np.ndarray, thr: float) -> np.ndarray:
    def _seq(seq: np.ndarray, cseq: np.ndarray) -> np.ndarray:
        out = seq.copy()
        for j in range(out.shape[1]):
            x = out[:, j, :]
            mu = x.mean(axis=0)
            cov = np.cov(x.T) + np.eye(2) * 1e-3
            inv = np.linalg.inv(cov)
            d = x - mu[None, :]
            md2 = np.einsum("ti,ij,tj->t", d, inv, d)
            bad = md2 > float(thr) ** 2
            for k in range(2):
                out[:, j, k] = _interp_nan_1d(np.where(bad, np.nan, out[:, j, k]))
        return out
    return _apply_per_video(base, np.ones(base.shape[:2], dtype=np.float32), videos, frames, _seq)


def _kf_track(z: np.ndarray, conf: np.ndarray, dts: np.ndarray, q: float, r: float, gate: float) -> np.ndarray:
    t = z.shape[0]
    out = np.zeros_like(z)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    x = np.array([z[0, 0], z[0, 1], 0.0, 0.0], dtype=np.float32)
    P = np.eye(4, dtype=np.float32) * 10.0
    for i in range(t):
        dt = float(dts[i] if i < dts.shape[0] else 1.0)
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        Q = np.diag([dt * dt, dt * dt, dt, dt]).astype(np.float32) * float(q)
        x = F @ x
        P = F @ P @ F.T + Q
        rr = float(r) / max(0.05, float(conf[i]))
        rr *= (1.0 + 0.02 * max(0.0, dt - 1.0))
        R = np.eye(2, dtype=np.float32) * rr
        y = z[i] - (H @ x)
        S = H @ P @ H.T + R
        inno = float(np.linalg.norm(y))
        gate_thr = float(gate) * math.sqrt(max(1e-6, float(np.trace(S))))
        if inno <= gate_thr:
            K = P @ H.T @ np.linalg.pinv(S)
            x = x + K @ y
            P = (np.eye(4, dtype=np.float32) - K @ H) @ P
        out[i] = x[:2]
    return out


def _algo_kalman(base: np.ndarray, conf: np.ndarray, videos: Sequence[str], frames: np.ndarray, q: float, r: float, max_dt: float, gate: float) -> np.ndarray:
    out = base.copy()
    orders = _video_orders(videos, frames)
    for _, idx in orders.items():
        seq = out[idx]
        cseq = conf[idx]
        dts = _frame_dts(frames[np.asarray(idx, dtype=np.int32)], max_dt=max_dt)
        for j in range(seq.shape[1]):
            seq[:, j, :] = _kf_track(seq[:, j, :], cseq[:, j], dts=dts, q=q, r=r, gate=gate)
        out[idx] = seq
    return out


def _ekf_track(z: np.ndarray, conf: np.ndarray, dts: np.ndarray, q: float, r: float, beta: float, gate: float) -> np.ndarray:
    t = z.shape[0]
    out = np.zeros_like(z)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    x = np.array([z[0, 0], z[0, 1], 0.0, 0.0], dtype=np.float32)
    P = np.eye(4, dtype=np.float32) * 10.0
    for i in range(t):
        dt = float(dts[i] if i < dts.shape[0] else 1.0)
        speed = float(np.linalg.norm(x[2:]))
        damp = 1.0 / (1.0 + float(beta) * speed)
        xp = np.array([x[0] + x[2] * dt, x[1] + x[3] * dt, x[2] * damp, x[3] * damp], dtype=np.float32)
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, damp, 0], [0, 0, 0, damp]], dtype=np.float32)
        Q = np.diag([dt * dt, dt * dt, dt, dt]).astype(np.float32) * float(q)
        x = xp
        P = F @ P @ F.T + Q
        rr = float(r) / max(0.05, float(conf[i]))
        rr *= (1.0 + 0.02 * max(0.0, dt - 1.0))
        R = np.eye(2, dtype=np.float32) * rr
        y = z[i] - H @ x
        S = H @ P @ H.T + R
        inno = float(np.linalg.norm(y))
        gate_thr = float(gate) * math.sqrt(max(1e-6, float(np.trace(S))))
        if inno <= gate_thr:
            K = P @ H.T @ np.linalg.pinv(S)
            x = x + K @ y
            P = (np.eye(4, dtype=np.float32) - K @ H) @ P
        out[i] = x[:2]
    return out


def _algo_ekf(base: np.ndarray, conf: np.ndarray, videos: Sequence[str], frames: np.ndarray, q: float, r: float, beta: float, max_dt: float, gate: float) -> np.ndarray:
    out = base.copy()
    orders = _video_orders(videos, frames)
    for _, idx in orders.items():
        seq = out[idx]
        cseq = conf[idx]
        dts = _frame_dts(frames[np.asarray(idx, dtype=np.int32)], max_dt=max_dt)
        for j in range(seq.shape[1]):
            seq[:, j, :] = _ekf_track(seq[:, j, :], cseq[:, j], dts=dts, q=q, r=r, beta=beta, gate=gate)
        out[idx] = seq
    return out


def _ukf_track(z: np.ndarray, conf: np.ndarray, dts: np.ndarray, q: float, r: float, alpha: float, beta: float, kappa: float, gate: float) -> np.ndarray:
    t = z.shape[0]
    out = np.zeros_like(z)
    n = 4
    lam = alpha * alpha * (n + kappa) - n
    wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)), dtype=np.float32)
    wc = wm.copy()
    wm[0] = lam / (n + lam)
    wc[0] = wm[0] + (1.0 - alpha * alpha + beta)
    x = np.array([z[0, 0], z[0, 1], 0.0, 0.0], dtype=np.float32)
    P = np.eye(4, dtype=np.float32) * 10.0
    for i in range(t):
        dt = float(dts[i] if i < dts.shape[0] else 1.0)
        P = 0.5 * (P + P.T)
        chol = None
        for jitter in (1e-6, 1e-4, 1e-3, 1e-2, 1e-1):
            try:
                chol = np.linalg.cholesky((n + lam) * (P + np.eye(n, dtype=np.float32) * float(jitter)))
                break
            except np.linalg.LinAlgError:
                continue
        if chol is None:
            chol = np.linalg.cholesky((n + lam) * (np.eye(n, dtype=np.float32) * 1.0))
        sigma = [x]
        for j in range(n):
            sigma.append(x + chol[:, j])
            sigma.append(x - chol[:, j])
        sp = []
        for s in sigma:
            sp.append(np.array([s[0] + s[2] * dt, s[1] + s[3] * dt, s[2], s[3]], dtype=np.float32))
        sp_arr = np.asarray(sp, dtype=np.float32)
        sp_arr = np.clip(sp_arr, -1e6, 1e6)
        x_pred = np.sum(wm[:, None] * sp_arr, axis=0)
        Q = np.diag([dt * dt, dt * dt, dt, dt]).astype(np.float32) * float(q)
        P_pred = Q.copy()
        for j in range(sp_arr.shape[0]):
            d = (sp_arr[j] - x_pred)[:, None]
            P_pred += wc[j] * (d @ d.T)
        P_pred = np.nan_to_num(P_pred, nan=0.0, posinf=1e6, neginf=-1e6)
        P_pred = np.clip(P_pred, -1e6, 1e6)
        P_pred = 0.5 * (P_pred + P_pred.T)
        z_sigma = sp_arr[:, :2]
        z_pred = np.sum(wm[:, None] * z_sigma, axis=0)
        rr = float(r) / max(0.05, float(conf[i]))
        rr *= (1.0 + 0.02 * max(0.0, dt - 1.0))
        R = np.eye(2, dtype=np.float32) * rr
        Szz = R.copy()
        Pxz = np.zeros((n, 2), dtype=np.float32)
        for j in range(sp_arr.shape[0]):
            dz = (z_sigma[j] - z_pred)[:, None]
            dx = (sp_arr[j] - x_pred)[:, None]
            Szz += wc[j] * (dz @ dz.T)
            Pxz += wc[j] * (dx @ dz.T)
        Szz = np.nan_to_num(Szz, nan=0.0, posinf=1e6, neginf=-1e6)
        Szz = 0.5 * (Szz + Szz.T) + np.eye(2, dtype=np.float32) * 1e-3
        innov = z[i] - z_pred
        gate_thr = float(gate) * math.sqrt(max(1e-6, float(np.trace(Szz))))
        if float(np.linalg.norm(innov)) <= gate_thr:
            K = Pxz @ np.linalg.pinv(Szz)
            x = x_pred + K @ innov
            P = P_pred - K @ Szz @ K.T
        else:
            x = x_pred
            P = P_pred
        P = 0.5 * (P + P.T)
        if not np.all(np.isfinite(x)):
            return _kf_track(z, conf, dts=dts, q=q, r=r, gate=gate)
        out[i] = x[:2]
    return out


def _algo_ukf(base: np.ndarray, conf: np.ndarray, videos: Sequence[str], frames: np.ndarray, q: float, r: float, max_dt: float, gate: float) -> np.ndarray:
    out = base.copy()
    orders = _video_orders(videos, frames)
    for _, idx in orders.items():
        seq = out[idx]
        cseq = conf[idx]
        dts = _frame_dts(frames[np.asarray(idx, dtype=np.int32)], max_dt=max_dt)
        for j in range(seq.shape[1]):
            seq[:, j, :] = _ukf_track(seq[:, j, :], cseq[:, j], dts=dts, q=q, r=r, alpha=0.3, beta=2.0, kappa=0.0, gate=gate)
        out[idx] = seq
    return out


def _particle_track(z: np.ndarray, conf: np.ndarray, n_particles: int, proc: float, meas: float) -> np.ndarray:
    t = z.shape[0]
    n = int(n_particles)
    rng = np.random.default_rng(42)
    particles = np.zeros((n, 4), dtype=np.float32)
    particles[:, 0] = z[0, 0] + rng.normal(0.0, meas, size=n)
    particles[:, 1] = z[0, 1] + rng.normal(0.0, meas, size=n)
    weights = np.ones((n,), dtype=np.float32) / float(n)
    out = np.zeros((t, 2), dtype=np.float32)
    for i in range(t):
        particles[:, 0] += particles[:, 2] + rng.normal(0.0, proc, size=n)
        particles[:, 1] += particles[:, 3] + rng.normal(0.0, proc, size=n)
        particles[:, 2] += rng.normal(0.0, proc * 0.5, size=n)
        particles[:, 3] += rng.normal(0.0, proc * 0.5, size=n)
        err2 = (particles[:, 0] - z[i, 0]) ** 2 + (particles[:, 1] - z[i, 1]) ** 2
        sigma = float(meas) / max(0.05, float(conf[i]))
        lik = np.exp(-0.5 * err2 / max(1e-6, sigma * sigma))
        weights = weights * lik
        s = float(weights.sum())
        if s <= 1e-12:
            weights[:] = 1.0 / float(n)
        else:
            weights = weights / s
        out[i] = np.sum(particles[:, :2] * weights[:, None], axis=0)
        cdf = np.cumsum(weights)
        u0 = rng.random() / float(n)
        u = u0 + np.arange(n) / float(n)
        idx = np.searchsorted(cdf, u)
        particles = particles[idx]
        weights[:] = 1.0 / float(n)
    return out


def _algo_particle(base: np.ndarray, conf: np.ndarray, videos: Sequence[str], frames: np.ndarray, n_particles: int, proc: float, meas: float) -> np.ndarray:
    def _seq(seq: np.ndarray, cseq: np.ndarray) -> np.ndarray:
        out = seq.copy()
        for j in range(out.shape[1]):
            out[:, j, :] = _particle_track(out[:, j, :], cseq[:, j], n_particles=n_particles, proc=proc, meas=meas)
        return out
    return _apply_per_video(base, conf, videos, frames, _seq)


def _solve_smoothing_1d(y: np.ndarray, w: np.ndarray, lam1: float, lam2: float) -> np.ndarray:
    t = y.shape[0]
    A = np.diag(w.astype(np.float64) + 1e-3)
    b = (w * y).astype(np.float64)
    if lam1 > 0.0 and t >= 2:
        D1 = np.zeros((t - 1, t), dtype=np.float64)
        for i in range(t - 1):
            D1[i, i] = -1.0
            D1[i, i + 1] = 1.0
        A += float(lam1) * (D1.T @ D1)
    if lam2 > 0.0 and t >= 3:
        D2 = np.zeros((t - 2, t), dtype=np.float64)
        for i in range(t - 2):
            D2[i, i] = 1.0
            D2[i, i + 1] = -2.0
            D2[i, i + 2] = 1.0
        A += float(lam2) * (D2.T @ D2)
    x = np.linalg.solve(A, b)
    return x.astype(np.float32)


def _algo_lstsq_smooth(base: np.ndarray, conf: np.ndarray, videos: Sequence[str], frames: np.ndarray, lam: float) -> np.ndarray:
    def _seq(seq: np.ndarray, cseq: np.ndarray) -> np.ndarray:
        out = seq.copy()
        for j in range(out.shape[1]):
            w = np.clip(cseq[:, j], 0.05, 1.0)
            for d in range(2):
                out[:, j, d] = _solve_smoothing_1d(out[:, j, d], w=w, lam1=lam, lam2=0.0)
        return out
    return _apply_per_video(base, conf, videos, frames, _seq)


def _algo_regularized_smooth(base: np.ndarray, conf: np.ndarray, videos: Sequence[str], frames: np.ndarray, lam1: float, lam2: float) -> np.ndarray:
    def _seq(seq: np.ndarray, cseq: np.ndarray) -> np.ndarray:
        out = seq.copy()
        for j in range(out.shape[1]):
            w = np.clip(cseq[:, j], 0.05, 1.0)
            for d in range(2):
                out[:, j, d] = _solve_smoothing_1d(out[:, j, d], w=w, lam1=lam1, lam2=lam2)
        return out
    return _apply_per_video(base, conf, videos, frames, _seq)


class _ResidualMLP(nn.Module):
    def __init__(self, inp: int, out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inp, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TemporalCNN(nn.Module):
    def __init__(self, channels: int, out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Linear(64, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        mid = z[:, :, z.shape[-1] // 2]
        return self.head(mid)


class _TemporalRNN(nn.Module):
    def __init__(self, inp: int, out: int, cell: str) -> None:
        super().__init__()
        if cell == "gru":
            self.rnn = nn.GRU(input_size=inp, hidden_size=96, num_layers=1, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size=inp, hidden_size=96, num_layers=1, batch_first=True)
        self.head = nn.Linear(96, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.rnn(x)
        mid = y[:, y.shape[1] // 2, :]
        return self.head(mid)


def _build_windows(data: SplitPayload, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    orders = _video_orders(data.sample_videos, data.sample_frames)
    half = window // 2
    feat = []
    target = []
    target_weight = []
    center_indices = []
    conf = np.clip(data.pred_score * data.pred_visibility, 0.0, 1.0)
    for _, idx in orders.items():
        seq_xy = data.pred_xy[idx]
        seq_cf = conf[idx]
        seq_gt = data.gt_xy[idx]
        t = len(idx)
        for i in range(t):
            lo = max(0, i - half)
            hi = min(t, i + half + 1)
            win_xy = seq_xy[lo:hi]
            win_cf = seq_cf[lo:hi]
            if win_xy.shape[0] < window:
                pad_l = max(0, half - i)
                pad_r = max(0, i + half + 1 - t)
                win_xy = np.pad(win_xy, ((pad_l, pad_r), (0, 0), (0, 0)), mode="edge")
                win_cf = np.pad(win_cf, ((pad_l, pad_r), (0, 0)), mode="edge")
            center_xy = seq_xy[i][None, :, :]
            rel_xy = win_xy - center_xy
            f = np.concatenate([rel_xy, win_cf[..., None]], axis=2)
            feat.append(f.astype(np.float32))
            target.append((seq_gt[i] - seq_xy[i]).astype(np.float32))
            target_weight.append((data.gt_visibility[idx[i]] > 0.5).astype(np.float32))
            center_indices.append(idx[i])
    return (
        np.asarray(feat, dtype=np.float32),
        np.asarray(target, dtype=np.float32),
        np.asarray(target_weight, dtype=np.float32),
        np.asarray(center_indices, dtype=np.int32),
    )


def _train_and_apply_residual_model(
    model_kind: str,
    train_data: SplitPayload,
    val_data: SplitPayload,
    epochs: int,
    lr: float,
    window: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    x_train, y_train, w_train, idx_train = _build_windows(train_data, window)
    x_val, y_val, w_val, idx_val = _build_windows(val_data, window)

    k = train_data.pred_xy.shape[1]
    out_dim = k * 2
    in_feat = k * 3
    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

    if model_kind == "mlp":
        model = _ResidualMLP(window * in_feat, out_dim).to(dev)
        xtr = torch.from_numpy(x_train.reshape(x_train.shape[0], -1)).to(dev)
        xva = torch.from_numpy(x_val.reshape(x_val.shape[0], -1)).to(dev)
    elif model_kind == "tcnn":
        model = _TemporalCNN(in_feat, out_dim).to(dev)
        xtr = torch.from_numpy(np.transpose(x_train.reshape(x_train.shape[0], window, in_feat), (0, 2, 1))).to(dev)
        xva = torch.from_numpy(np.transpose(x_val.reshape(x_val.shape[0], window, in_feat), (0, 2, 1))).to(dev)
    elif model_kind == "gru":
        model = _TemporalRNN(in_feat, out_dim, "gru").to(dev)
        xtr = torch.from_numpy(x_train.reshape(x_train.shape[0], window, in_feat)).to(dev)
        xva = torch.from_numpy(x_val.reshape(x_val.shape[0], window, in_feat)).to(dev)
    else:
        model = _TemporalRNN(in_feat, out_dim, "lstm").to(dev)
        xtr = torch.from_numpy(x_train.reshape(x_train.shape[0], window, in_feat)).to(dev)
        xva = torch.from_numpy(x_val.reshape(x_val.shape[0], window, in_feat)).to(dev)

    ytr = torch.from_numpy(y_train.reshape(y_train.shape[0], -1)).to(dev)
    wtr = torch.from_numpy(np.repeat(w_train[..., None], 2, axis=2).reshape(w_train.shape[0], -1)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = nn.SmoothL1Loss(reduction="none")

    bs = 128
    order = np.arange(xtr.shape[0])
    for _ in range(int(epochs)):
        np.random.shuffle(order)
        for s in range(0, order.shape[0], bs):
            b = order[s : s + bs]
            pred = model(xtr[b])
            loss_raw = loss_fn(pred, ytr[b])
            weight = torch.clamp(wtr[b], min=0.0, max=1.0)
            denom = torch.clamp(weight.sum(), min=1.0)
            loss = (loss_raw * weight).sum() / denom
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        train_res = model(xtr).detach().cpu().numpy().reshape(-1, k, 2)
        val_res = model(xva).detach().cpu().numpy().reshape(-1, k, 2)

    train_res = np.clip(train_res, -120.0, 120.0)
    val_res = np.clip(val_res, -120.0, 120.0)

    train_out = train_data.pred_xy.copy()
    val_out = val_data.pred_xy.copy()
    train_out[idx_train] = train_out[idx_train] + train_res
    val_out[idx_val] = val_out[idx_val] + val_res
    visible_ratio_train = float(w_train.mean()) if w_train.size > 0 else 0.0
    visible_ratio_val = float(w_val.mean()) if w_val.size > 0 else 0.0
    return train_out, val_out, {
        "epochs": int(epochs),
        "lr": float(lr),
        "window": int(window),
        "device": str(dev),
        "visibility_weighted_loss": True,
        "visible_ratio_train": visible_ratio_train,
        "visible_ratio_val": visible_ratio_val,
    }


def _render_predictions(raw_predictions: Sequence[Mapping[str, Any]], bodyparts: Sequence[str], pred_xy: np.ndarray) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_predictions):
        item = json.loads(json.dumps(raw))
        keypoints = item.get("keypoints", {})
        for kp_idx, bodypart in enumerate(bodyparts):
            if bodypart not in keypoints:
                continue
            keypoints[bodypart]["x"] = float(pred_xy[idx, kp_idx, 0])
            keypoints[bodypart]["y"] = float(pred_xy[idx, kp_idx, 1])
        rendered.append(item)
    return rendered


def _fit_velocity_stats(train_data: SplitPayload) -> tuple[float, float]:
    orders = _video_orders(train_data.sample_videos, train_data.sample_frames)
    vel = []
    acc = []
    for _, idx in orders.items():
        seq = train_data.pred_xy[idx]
        if seq.shape[0] >= 2:
            v = np.linalg.norm(seq[1:] - seq[:-1], axis=2).reshape(-1)
            vel.append(v)
        if seq.shape[0] >= 3:
            dv = (seq[2:] - seq[1:-1]) - (seq[1:-1] - seq[:-2])
            a = np.linalg.norm(dv, axis=2).reshape(-1)
            acc.append(a)
    vel_thr = float(np.quantile(np.concatenate(vel), 0.98)) if vel else 50.0
    acc_thr = float(np.quantile(np.concatenate(acc), 0.98)) if acc else 50.0
    return vel_thr, acc_thr


def _fit_temporal_prior(
    videos: Sequence[str],
    frames: np.ndarray,
    gt_xy: np.ndarray,
    gt_vis: np.ndarray,
    use_indices: np.ndarray,
) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    model: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for i in use_indices.tolist():
        v = str(videos[i])
        model.setdefault(v, {})
    for v in list(model.keys()):
        idx = [i for i in use_indices.tolist() if str(videos[i]) == v]
        if not idx:
            continue
        t = frames[np.asarray(idx, dtype=np.int32)]
        for k in range(gt_xy.shape[1]):
            m = gt_vis[np.asarray(idx, dtype=np.int32), k] > 0.5
            if int(m.sum()) < 2:
                continue
            tv = t[m].astype(np.float32)
            xv = gt_xy[np.asarray(idx, dtype=np.int32), k, 0][m].astype(np.float32)
            yv = gt_xy[np.asarray(idx, dtype=np.int32), k, 1][m].astype(np.float32)
            order = np.argsort(tv)
            model[v][k] = {"t": tv[order], "x": xv[order], "y": yv[order]}
    return model


def _predict_temporal_prior(
    prior: dict[str, dict[int, dict[str, np.ndarray]]],
    videos: Sequence[str],
    frames: np.ndarray,
    kcount: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(videos)
    out = np.zeros((n, kcount, 2), dtype=np.float32)
    dist = np.full((n, kcount), 1e6, dtype=np.float32)
    for i in range(n):
        v = str(videos[i])
        t = float(frames[i])
        if v not in prior:
            continue
        for k in range(kcount):
            rec = prior[v].get(k)
            if rec is None:
                continue
            tv = rec["t"]
            xv = rec["x"]
            yv = rec["y"]
            out[i, k, 0] = float(np.interp(t, tv, xv))
            out[i, k, 1] = float(np.interp(t, tv, yv))
            dist[i, k] = float(np.min(np.abs(tv - t)))
    return out, dist


def _apply_prior_blend(
    base_xy: np.ndarray,
    conf: np.ndarray,
    prior_xy: np.ndarray,
    prior_dist: np.ndarray,
    alpha: float,
    tau: float,
    conf_thr: float,
) -> np.ndarray:
    out = base_xy.copy()
    temporal_w = np.exp(-np.clip(prior_dist, 0.0, 1e6) / max(1e-6, float(tau)))
    lowconf_w = np.clip((float(conf_thr) - conf) / max(1e-6, float(conf_thr)), 0.0, 1.0)
    w = np.clip(float(alpha) * temporal_w * lowconf_w, 0.0, 0.95)
    out = (1.0 - w[..., None]) * out + w[..., None] * prior_xy
    return out


def _fit_prior_blend_params(train_data: SplitPayload) -> dict[str, float]:
    n = train_data.pred_xy.shape[0]
    rng = np.random.default_rng(42)
    order = np.arange(n, dtype=np.int32)
    rng.shuffle(order)
    cut = max(1, int(0.8 * n))
    fit_idx = np.sort(order[:cut])
    val_idx = np.sort(order[cut:])
    if val_idx.size == 0:
        return {"alpha": 0.8, "tau": 600.0, "conf_thr": 0.6}

    conf = np.clip(train_data.pred_score * train_data.pred_visibility, 0.0, 1.0)
    prior = _fit_temporal_prior(train_data.sample_videos, train_data.sample_frames, train_data.gt_xy, train_data.gt_visibility, fit_idx)
    prior_xy_all, prior_dist_all = _predict_temporal_prior(prior, train_data.sample_videos, train_data.sample_frames, train_data.pred_xy.shape[1])

    best = {"alpha": 0.8, "tau": 600.0, "conf_thr": 0.6}
    best_rmse90 = float("inf")
    for alpha in (0.4, 0.6, 0.8, 0.9):
        for tau in (120.0, 240.0, 480.0, 960.0, 1920.0):
            for conf_thr in (0.4, 0.5, 0.6, 0.7, 0.8):
                cand = _apply_prior_blend(train_data.pred_xy, conf, prior_xy_all, prior_dist_all, alpha=alpha, tau=tau, conf_thr=conf_thr)
                m = _compute_metrics(cand[val_idx], train_data.gt_xy[val_idx], train_data.gt_visibility[val_idx])
                rmse90 = float(m["rmse_90"])
                if rmse90 < best_rmse90:
                    best_rmse90 = rmse90
                    best = {"alpha": float(alpha), "tau": float(tau), "conf_thr": float(conf_thr)}
    return best


def _build_algorithms(
    train_data: SplitPayload,
    val_data: SplitPayload,
    bodyparts: Sequence[str],
    skeleton: Sequence[Sequence[str]],
    device: str,
    val_context: DenseContextPayload | None,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, list[int]]]:
    algorithms: list[dict[str, Any]] = []
    transformed_val: dict[str, np.ndarray] = {}
    conf_train = np.clip(train_data.pred_score * train_data.pred_visibility, 0.0, 1.0)
    if val_context is None:
        val_base_xy = val_data.pred_xy
        val_base_conf = np.clip(val_data.pred_score * val_data.pred_visibility, 0.0, 1.0)
        val_videos = val_data.sample_videos
        val_frames = val_data.sample_frames
        val_eval_indices = np.arange(val_data.pred_xy.shape[0], dtype=np.int32)
    else:
        val_base_xy = val_context.pred_xy
        val_base_conf = np.clip(val_context.pred_score * val_context.pred_visibility, 0.0, 1.0)
        val_videos = val_context.sample_videos
        val_frames = val_context.sample_frames
        val_eval_indices = val_context.eval_indices

    edges = _skeleton_edges(bodyparts, skeleton)
    target_bones = _fit_bone_lengths(train_data, edges)
    triplets = _angle_triplets(edges, len(bodyparts))
    angle_limits = _fit_angle_limits(train_data, triplets)
    vel_thr, acc_thr = _fit_velocity_stats(train_data)
    prior_params = _fit_prior_blend_params(train_data)
    train_prior_model = _fit_temporal_prior(
        train_data.sample_videos,
        train_data.sample_frames,
        train_data.gt_xy,
        train_data.gt_visibility,
        np.arange(train_data.pred_xy.shape[0], dtype=np.int32),
    )
    train_prior_xy, train_prior_dist = _predict_temporal_prior(
        train_prior_model,
        train_data.sample_videos,
        train_data.sample_frames,
        train_data.pred_xy.shape[1],
    )
    val_prior_xy, val_prior_dist = _predict_temporal_prior(
        train_prior_model,
        val_videos,
        val_frames,
        train_data.pred_xy.shape[1],
    )

    def _register(name: str, params: dict[str, Any], train_pred: np.ndarray, val_pred_any: np.ndarray) -> None:
        tr = _compute_metrics(train_pred, train_data.gt_xy, train_data.gt_visibility)
        if int(val_pred_any.shape[0]) == int(val_data.pred_xy.shape[0]):
            val_pred = val_pred_any
        else:
            val_pred = val_pred_any[val_eval_indices]
        va = _compute_metrics(val_pred, val_data.gt_xy, val_data.gt_visibility)
        algorithms.append({"name": name, "fit_params": params, "train_metrics": tr, "val_metrics": va})
        transformed_val[name] = val_pred

    _register("confidence_thresholding", {"threshold": 0.45},
              _algo_conf_threshold(train_data.pred_xy, conf_train, train_data.sample_videos, train_data.sample_frames, 0.45),
              _algo_conf_threshold(val_base_xy, val_base_conf, val_videos, val_frames, 0.45))

    _register("supervised_temporal_prior_blend", prior_params,
              _apply_prior_blend(train_data.pred_xy, conf_train, train_prior_xy, train_prior_dist, **prior_params),
              _apply_prior_blend(val_base_xy, val_base_conf, val_prior_xy, val_prior_dist, **prior_params))

    _register("moving_average_smoothing", {"window": 5},
              _algo_moving_average(train_data.pred_xy, conf_train, train_data.sample_videos, train_data.sample_frames, 5),
              _algo_moving_average(val_base_xy, val_base_conf, val_videos, val_frames, 5))

    _register("exponential_moving_average_smoothing", {"alpha": 0.7},
              _algo_ema(train_data.pred_xy, conf_train, train_data.sample_videos, train_data.sample_frames, 0.7),
              _algo_ema(val_base_xy, val_base_conf, val_videos, val_frames, 0.7))

    _register("median_filter", {"window": 5},
              _algo_median(train_data.pred_xy, conf_train, train_data.sample_videos, train_data.sample_frames, 5),
              _algo_median(val_base_xy, val_base_conf, val_videos, val_frames, 5))

    _register("savitzky_golay_filter", {"window": 7, "order": 3},
              _algo_savgol(train_data.pred_xy, conf_train, train_data.sample_videos, train_data.sample_frames, 7, 3),
              _algo_savgol(val_base_xy, val_base_conf, val_videos, val_frames, 7, 3))

    _register("linear_interpolation", {"threshold": 0.55},
              _algo_linear_interp(train_data.pred_xy, conf_train, train_data.sample_videos, train_data.sample_frames, 0.55),
              _algo_linear_interp(val_base_xy, val_base_conf, val_videos, val_frames, 0.55))

    _register("cubic_spline_interpolation", {"threshold": 0.55},
              _algo_cubic_interp(train_data.pred_xy, conf_train, train_data.sample_videos, train_data.sample_frames, 0.55),
              _algo_cubic_interp(val_base_xy, val_base_conf, val_videos, val_frames, 0.55))

    _register("velocity_clipping", {"vmax": vel_thr},
              _algo_velocity_clip(train_data.pred_xy, conf_train, train_data.sample_videos, train_data.sample_frames, vel_thr),
              _algo_velocity_clip(val_base_xy, val_base_conf, val_videos, val_frames, vel_thr))

    _register("acceleration_clipping", {"amax": acc_thr},
              _algo_accel_clip(train_data.pred_xy, conf_train, train_data.sample_videos, train_data.sample_frames, acc_thr),
              _algo_accel_clip(val_base_xy, val_base_conf, val_videos, val_frames, acc_thr))

    _register("constant_velocity_extrapolation", {"threshold": 0.45},
              _algo_const_vel_extrap(train_data.pred_xy, conf_train, train_data.sample_videos, train_data.sample_frames, 0.45),
              _algo_const_vel_extrap(val_base_xy, val_base_conf, val_videos, val_frames, 0.45))

    _register("bone_length_consistency_correction", {"edge_count": len(edges), "strength": 0.45, "min_conf": 0.25, "gated": True},
              _algo_bone_length_gated(train_data.pred_xy, conf_train, edges, target_bones, strength=0.45, min_conf=0.25),
              _algo_bone_length_gated(val_base_xy, val_base_conf, edges, target_bones, strength=0.45, min_conf=0.25))

    _register("joint_angle_limit_correction", {"triplet_count": len(angle_limits), "strength": 0.35, "min_center_conf": 0.3, "gated": True},
              _algo_angle_limit_gated(train_data.pred_xy, conf_train, angle_limits, strength=0.35, min_center_conf=0.3),
              _algo_angle_limit_gated(val_base_xy, val_base_conf, angle_limits, strength=0.35, min_center_conf=0.3))

    _register("zscore_outlier_detection_and_correction", {"zthr": 3.0},
              _algo_zscore(train_data.pred_xy, train_data.sample_videos, train_data.sample_frames, 3.0),
              _algo_zscore(val_base_xy, val_videos, val_frames, 3.0))

    _register("mad_outlier_detection_and_correction", {"zthr": 3.5},
              _algo_mad(train_data.pred_xy, train_data.sample_videos, train_data.sample_frames, 3.5),
              _algo_mad(val_base_xy, val_videos, val_frames, 3.5))

    _register("mahalanobis_outlier_detection_and_correction", {"thr": 3.5},
              _algo_mahalanobis(train_data.pred_xy, train_data.sample_videos, train_data.sample_frames, 3.5),
              _algo_mahalanobis(val_base_xy, val_videos, val_frames, 3.5))

    _register("kalman_filter", {"q": 0.12, "r": 8.0, "max_dt": 120.0, "gate": 4.0, "dt_aware": True},
              _algo_kalman(train_data.pred_xy, conf_train, train_data.sample_videos, train_data.sample_frames, q=0.12, r=8.0, max_dt=120.0, gate=4.0),
              _algo_kalman(val_base_xy, val_base_conf, val_videos, val_frames, q=0.12, r=8.0, max_dt=120.0, gate=4.0))

    _register("extended_kalman_filter", {"q": 0.12, "r": 8.0, "beta": 0.02, "max_dt": 120.0, "gate": 4.0, "dt_aware": True},
              _algo_ekf(train_data.pred_xy, conf_train, train_data.sample_videos, train_data.sample_frames, q=0.12, r=8.0, beta=0.02, max_dt=120.0, gate=4.0),
              _algo_ekf(val_base_xy, val_base_conf, val_videos, val_frames, q=0.12, r=8.0, beta=0.02, max_dt=120.0, gate=4.0))

    _register("unscented_kalman_filter", {"q": 0.12, "r": 8.0, "max_dt": 120.0, "gate": 4.0, "dt_aware": True},
              _algo_ukf(train_data.pred_xy, conf_train, train_data.sample_videos, train_data.sample_frames, q=0.12, r=8.0, max_dt=120.0, gate=4.0),
              _algo_ukf(val_base_xy, val_base_conf, val_videos, val_frames, q=0.12, r=8.0, max_dt=120.0, gate=4.0))

    _register("particle_filter", {"n_particles": 64, "proc": 1.0, "meas": 6.0},
              _algo_particle(train_data.pred_xy, conf_train, train_data.sample_videos, train_data.sample_frames, n_particles=64, proc=1.0, meas=6.0),
              _algo_particle(val_base_xy, val_base_conf, val_videos, val_frames, n_particles=64, proc=1.0, meas=6.0))

    _register("least_squares_temporal_smoothing", {"lam": 2.0},
              _algo_lstsq_smooth(train_data.pred_xy, conf_train, train_data.sample_videos, train_data.sample_frames, lam=2.0),
              _algo_lstsq_smooth(val_base_xy, val_base_conf, val_videos, val_frames, lam=2.0))

    _register("regularized_temporal_smoothing_optimization", {"lam1": 2.0, "lam2": 8.0},
              _algo_regularized_smooth(train_data.pred_xy, conf_train, train_data.sample_videos, train_data.sample_frames, lam1=2.0, lam2=8.0),
              _algo_regularized_smooth(val_base_xy, val_base_conf, val_videos, val_frames, lam1=2.0, lam2=8.0))

    mlp_tr, mlp_va, mlp_meta = _train_and_apply_residual_model("mlp", train_data, val_data, epochs=8, lr=1e-3, window=9, device=device)
    _register("small_residual_mlp_refinement_model", mlp_meta, mlp_tr, mlp_va)

    cnn_tr, cnn_va, cnn_meta = _train_and_apply_residual_model("tcnn", train_data, val_data, epochs=8, lr=1e-3, window=9, device=device)
    _register("tiny_temporal_cnn_refinement_model", cnn_meta, cnn_tr, cnn_va)

    gru_tr, gru_va, gru_meta = _train_and_apply_residual_model("gru", train_data, val_data, epochs=8, lr=1e-3, window=9, device=device)
    _register("small_gru_refinement_model", gru_meta, gru_tr, gru_va)

    lstm_tr, lstm_va, lstm_meta = _train_and_apply_residual_model("lstm", train_data, val_data, epochs=8, lr=1e-3, window=9, device=device)
    _register("small_lstm_refinement_model", lstm_meta, lstm_tr, lstm_va)

    algorithms.sort(key=lambda x: float(x["val_metrics"]["rmse_unfiltered"]))
    frame_manifest: dict[str, list[int]] = {}
    for v, f in zip(val_data.sample_videos, val_data.sample_frames.tolist()):
        frame_manifest.setdefault(v, []).append(int(f))
    for k in frame_manifest:
        frame_manifest[k] = sorted(set(frame_manifest[k]))
    return algorithms, transformed_val, frame_manifest


def _build_output_dir(output_root: Path, checkpoint: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = checkpoint.parent.name.replace(" ", "_")
    run_dir = output_root / f"{stem}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark RTMPose post-processing algorithms on train/val splits.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "output" / "RTMPose" / "no_weak_20260328_174401" / "checkpoint_best.pt",
        help="RTMPose checkpoint to evaluate.",
    )
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        default=None,
        help="Override detector checkpoint. Defaults to associated detector in checkpoint run_config.yaml.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--detector-device", type=str, default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--auto-val-fraction", type=float, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preload-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preload-masks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--context-radius", type=int, default=120)
    parser.add_argument("--context-step", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "output" / "post_process")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    cli_args = parser.parse_args(argv)
    runtime_args = _build_runtime_args(cli_args)
    rtm.set_seed(int(runtime_args.seed))

    output_dir = _build_output_dir(cli_args.output_root, runtime_args.checkpoint)

    device = rtm.resolve_device(str(runtime_args.device))
    model_cfg = _resolve_model_cfg(runtime_args)
    project_cfg = rtm.load_project_config(runtime_args.project_config)

    use_masks = float(getattr(runtime_args, "weak_sample_weight", 1.0)) > 0.0
    split_indices = rtm.build_all_split_indices(
        project_cfg,
        runtime_args.labels_root,
        runtime_args.labeled_frames_root,
        runtime_args.frames_root,
        runtime_args.masks_root,
        include_weak=use_masks,
        require_masks=use_masks,
        auto_val_fraction=float(runtime_args.auto_val_fraction),
        split_seed=int(runtime_args.seed),
    )
    rtm.validate_mutual_exclusion(split_indices)
    store, detector_boxes, filtered_indices, detector_stats = rtm.prepare_training_components(runtime_args, model_cfg, split_indices)

    model, _ = load_model_from_checkpoint_for_inference(
        model_path=runtime_args.checkpoint.parent,
        checkpoint=runtime_args.checkpoint,
        device=device,
    )

    train_data = _predict_split(
        "train",
        runtime_args,
        model_cfg,
        project_cfg,
        store,
        detector_boxes,
        filtered_indices,
        model,
        device,
    )
    val_data = _predict_split(
        "val",
        runtime_args,
        model_cfg,
        project_cfg,
        store,
        detector_boxes,
        filtered_indices,
        model,
        device,
    )

    val_context = _predict_dense_context_for_split(
        val_data,
        runtime_args,
        model_cfg,
        project_cfg,
        store,
        model,
        context_radius=int(cli_args.context_radius),
        context_step=int(cli_args.context_step),
    )

    algorithms, transformed_val, frame_manifest = _build_algorithms(
        train_data,
        val_data,
        bodyparts=project_cfg.bodyparts,
        skeleton=project_cfg.skeleton,
        device=str(runtime_args.device),
        val_context=val_context,
    )
    best = algorithms[0]

    raw_train_metrics = _compute_metrics(train_data.pred_xy, train_data.gt_xy, train_data.gt_visibility)
    raw_val_metrics = _compute_metrics(val_data.pred_xy, val_data.gt_xy, val_data.gt_visibility)

    benchmark_payload = {
        "checkpoint": str(runtime_args.checkpoint.resolve()),
        "associated_detector_checkpoint": str(Path(runtime_args.detector_checkpoint).resolve()),
        "seed": int(runtime_args.seed),
        "auto_val_fraction": float(runtime_args.auto_val_fraction),
        "train_sample_count": int(train_data.pred_xy.shape[0]),
        "val_sample_count": int(val_data.pred_xy.shape[0]),
        "raw_prediction_metrics_original_space": {
            "train": raw_train_metrics,
            "val": raw_val_metrics,
        },
        "metric_eval_crop_space": {
            "train": train_data.metric_eval_crop_space,
            "val": val_data.metric_eval_crop_space,
            "note": "These are RTMPose evaluate_pose_model metrics in dataset crop-coordinate space (training-log comparable).",
        },
        "detector_stats": rtm.make_serializable(detector_stats),
        "algorithm_count": len(algorithms),
        "temporal_context": {
            "enabled": bool(val_context is not None),
            "context_radius": int(cli_args.context_radius),
            "context_step": int(cli_args.context_step),
            "dense_val_frame_count": int(val_context.pred_xy.shape[0]) if val_context is not None else 0,
        },
        "algorithms_ranked_by_val_rmse": algorithms,
        "best_algorithm": {
            "name": best["name"],
            "val_rmse_unfiltered": best["val_metrics"]["rmse_unfiltered"],
            "val_mean_pixel_error": best["val_metrics"]["mean_pixel_error"],
            "val_rmse_90": best["val_metrics"]["rmse_90"],
            "delta_vs_raw_val_rmse_unfiltered": float(best["val_metrics"]["rmse_unfiltered"] - raw_val_metrics["rmse_unfiltered"]),
            "delta_vs_raw_val_rmse_90": float(best["val_metrics"]["rmse_90"] - raw_val_metrics["rmse_90"]),
        },
    }
    _write_json(output_dir / "benchmark_metrics.json", benchmark_payload)
    _write_json(output_dir / "best_algorithm.json", best)
    _write_json(output_dir / "val_frame_manifest.json", {"frames_by_video": frame_manifest})

    for algo in algorithms:
        name = str(algo["name"])
        pred_xy = transformed_val[name]
        rendered = _render_predictions(val_data.raw_predictions, val_data.bodyparts, pred_xy)
        _write_json(output_dir / f"predictions_val_{name}.json", rendered)

    summary_lines = [
        f"checkpoint: {runtime_args.checkpoint}",
        f"detector: {runtime_args.detector_checkpoint}",
        f"train_samples: {train_data.pred_xy.shape[0]}",
        f"val_samples: {val_data.pred_xy.shape[0]}",
        f"raw_val_rmse_unfiltered: {raw_val_metrics['rmse_unfiltered']:.6f}",
        f"raw_val_rmse_90: {raw_val_metrics['rmse_90']:.6f}",
        f"best_algorithm: {best['name']}",
        f"best_val_rmse_unfiltered: {best['val_metrics']['rmse_unfiltered']:.6f}",
        f"best_val_rmse_90: {best['val_metrics']['rmse_90']:.6f}",
        f"best_minus_raw_rmse_unfiltered: {best['val_metrics']['rmse_unfiltered'] - raw_val_metrics['rmse_unfiltered']:.6f}",
        f"best_minus_raw_rmse_90: {best['val_metrics']['rmse_90'] - raw_val_metrics['rmse_90']:.6f}",
        f"best_val_mean_pixel_error: {best['val_metrics']['mean_pixel_error']:.6f}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("Post-process benchmark complete")
    print(f"Output directory: {output_dir}")
    for rank, algo in enumerate(algorithms, start=1):
        vm = algo["val_metrics"]
        print(f"{rank:02d}. {algo['name']:<44s} rmse={vm['rmse_unfiltered']:.4f} rmse90={vm['rmse_90']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
