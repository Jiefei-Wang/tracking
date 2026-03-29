#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cv2
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from modules.detector_dataset import (
    DetectionDataset,
    RamImageStore,
    build_albumentations_pipeline,
    build_all_split_indices,
    cycle_loader,
    detection_collate_fn,
    load_project_config,
    load_yaml_file,
    print_split_summaries,
    select_samples_for_prediction,
    set_global_seed,
    summarize_split_indices,
    validate_disjoint_splits,
    validate_mutual_exclusion,
)
from modules.detector_metrics import compute_detection_metrics
from modules.detector_ssdlite_model import build_ssdlite_model, load_checkpoint, save_checkpoint


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_PROJECT_CONFIG = PROJECT_ROOT / "config.yaml"
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "input" / "ssdlite" / "model.yaml"
DEFAULT_RUN_CONFIG = PROJECT_ROOT / "input" / "ssdlite" / "config.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "ssdlite"


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
    model_config_path: Path,
    run_config_path: Path,
    config_overwrite_path: Path | None,
) -> None:
    write_yaml(run_dir / "project_config.yaml", load_yaml_file(project_config_path))
    write_yaml(run_dir / "model_config.yaml", load_yaml_file(model_config_path))
    write_yaml(
        run_dir / "run_config.yaml",
        resolve_merged_run_config(run_config_path, config_overwrite_path),
    )


def resolve_device(device_name: str) -> torch.device:
    requested = (device_name or "cuda:0").strip()
    if not requested.startswith("cuda"):
        raise ValueError(f"GPU training/inference is required for this script, got device={requested!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script, but torch.cuda.is_available() is False.")
    return torch.device(requested)


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


def deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge_dicts(dict(merged[key]), dict(value))
        else:
            merged[key] = value
    return merged


def merge_cli_with_yaml(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    config_path = getattr(args, "run_config", None)
    if config_path is None:
        return args
    payload = load_yaml_file(config_path)
    config_overwrite = getattr(args, "config_overwrite", None)
    if config_overwrite is not None:
        override_payload = load_yaml_file(config_overwrite)
        payload = deep_merge_dicts(payload, override_payload)
    merged: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in {"training", "inference"}:
            merged[key] = value
    section = "training" if args.command in {"train", "debug"} else "inference"
    section_payload = payload.get(section, {})
    if isinstance(section_payload, Mapping):
        merged.update(section_payload)

    defaults = collect_parser_defaults(parser)
    path_keys = {
        "project_config",
        "model_config",
        "run_config",
        "config_overwrite",
        "labels_root",
        "frames_root",
        "masks_root",
        "output_root",
        "checkpoint",
        "input_dir",
        "input_path",
    }
    for key, value in merged.items():
        if not hasattr(args, key):
            continue
        if key in defaults and getattr(args, key) != defaults.get(key):
            continue
        if key in path_keys and value is not None:
            setattr(args, key, Path(value))
        else:
            setattr(args, key, value)
    return args


def build_dataloader(
    dataset: DetectionDataset,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
) -> DataLoader:
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "collate_fn": detection_collate_fn,
        "drop_last": drop_last,
    }
    if shuffle:
        loader_kwargs["sampler"] = RandomSampler(dataset)
    if workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**loader_kwargs)


def build_split_indices_from_args(args: argparse.Namespace):
    project_cfg = load_project_config(args.project_config)
    validate_disjoint_splits(project_cfg)
    split_indices = build_all_split_indices(
        config=project_cfg,
        labels_root=args.labels_root,
        frames_root=args.frames_root,
        masks_root=args.masks_root,
        bbox_margin=float(args.bbox_margin),
        weak_sample_weight=float(args.weak_sample_weight),
        min_box_size=float(args.min_box_size),
    )
    validate_mutual_exclusion(split_indices)
    return project_cfg, split_indices


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    images = [image.to(device, non_blocking=True) for image in batch["images"]]
    targets: list[dict[str, Any]] = []
    for target in batch["targets"]:
        moved: dict[str, Any] = {}
        for key, value in target.items():
            if torch.is_tensor(value):
                moved[key] = value.to(device, non_blocking=True)
            else:
                moved[key] = value
        targets.append(moved)
    return images, targets


def strip_targets_for_model(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = {"boxes", "labels", "image_id", "area", "iscrowd"}
    stripped: list[dict[str, Any]] = []
    for target in targets:
        stripped.append({key: value for key, value in target.items() if key in allowed})
    return stripped


def draw_box(
    image_uint8: np.ndarray,
    box: Iterable[float],
    color: tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    canvas = np.ascontiguousarray(image_uint8.copy())
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color=color, thickness=thickness)
    return canvas


def draw_debug_boxes(
    image_uint8: np.ndarray,
    roi_boxes: Iterable[Iterable[float]],
    margin_boxes: Iterable[Iterable[float]],
) -> np.ndarray:
    canvas = np.ascontiguousarray(image_uint8.copy())
    for box in roi_boxes:
        canvas = draw_box(canvas, box, color=(0, 255, 0), thickness=2)
    for box in margin_boxes:
        canvas = draw_box(canvas, box, color=(255, 0, 0), thickness=2)
    return canvas


def image_tensor_to_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().clamp(0.0, 1.0).numpy()
    image = np.transpose(image, (1, 2, 0))
    return np.ascontiguousarray((image * 255.0).round().astype(np.uint8))


def build_predict_output_dir(checkpoint_path: Path) -> Path:
    checkpoint_path = checkpoint_path.resolve()
    run_dir = checkpoint_path.parent
    predict_dir_name = f"{run_dir.name}_predict"
    return run_dir.parent / predict_dir_name


def _synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_model_forward(
    model: torch.nn.Module,
    images: list[torch.Tensor],
    targets: list[dict[str, Any]] | None = None,
    amp_enabled: bool = False,
) -> Any:
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
        if targets is None:
            return model(images)
        return model(images, targets)


def predict_single_image(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    device: torch.device,
    amp_enabled: bool,
    score_threshold: float,
) -> dict[str, Any]:
    image_tensor = torch.from_numpy(np.ascontiguousarray(image_rgb.transpose(2, 0, 1))).float() / 255.0
    images = [image_tensor.to(device, non_blocking=True)]
    with torch.no_grad():
        outputs = run_model_forward(model, images, None, amp_enabled=amp_enabled)
    output = outputs[0]
    keep = output["scores"] >= float(score_threshold)
    return {
        "boxes": output["boxes"][keep].detach().cpu().tolist(),
        "scores": output["scores"][keep].detach().cpu().tolist(),
        "labels": output["labels"][keep].detach().cpu().tolist(),
    }


def draw_prediction_boxes(frame_bgr: np.ndarray, prediction: Mapping[str, Any]) -> np.ndarray:
    canvas = np.ascontiguousarray(frame_bgr.copy())
    boxes = prediction.get("boxes", [])
    scores = prediction.get("scores", [])
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
        score = float(scores[index]) if index < len(scores) else 0.0
        label_text = f"{score:.2f}"
        cv2.putText(
            canvas,
            label_text,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


def predict_video_file(
    model: torch.nn.Module,
    video_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    amp_enabled: bool,
    score_threshold: float,
) -> Path:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video for prediction: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0.0:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = None
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Failed to read video size from: {video_path}")

    output_dir = build_predict_output_dir(checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_path.stem}.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Failed to open output video writer: {output_path}")

    frame_count = 0
    try:
        with tqdm(total=total_frames, desc=f"Predicting {video_path.name}", unit="frame") as progress:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                prediction = predict_single_image(
                    model=model,
                    image_rgb=frame_rgb,
                    device=device,
                    amp_enabled=amp_enabled,
                    score_threshold=score_threshold,
                )
                annotated = draw_prediction_boxes(frame_bgr, prediction)
                writer.write(annotated)
                frame_count += 1
                progress.update(1)
    finally:
        capture.release()
        writer.release()

    print(f"Saved annotated video ({frame_count} frames) to {output_path}")
    return output_path


def compute_weighted_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    amp_enabled: bool,
    batch_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float], float, float]:
    transfer_start = time.perf_counter()
    images, targets = move_batch_to_device(batch, device=device)
    model_targets = strip_targets_for_model(targets)
    _synchronize_if_needed(device)
    transfer_seconds = time.perf_counter() - transfer_start

    gpu_forward_start = time.perf_counter()
    losses = run_model_forward(model, images, model_targets, amp_enabled=amp_enabled)
    total_loss = sum(losses.values()) * float(batch_scale)
    _synchronize_if_needed(device)
    gpu_forward_seconds = time.perf_counter() - gpu_forward_start
    loss_items = {key: float(value.detach().item()) for key, value in losses.items()}
    return total_loss, loss_items, transfer_seconds, gpu_forward_seconds


def run_validation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    score_threshold: float,
) -> dict[str, float]:
    model.eval()
    predictions: list[dict[str, Any]] = []
    targets_payload: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            images, targets = move_batch_to_device(batch, device=device)
            outputs = run_model_forward(model, images, None, amp_enabled=amp_enabled)
            for output, target in zip(outputs, targets):
                boxes = output["boxes"].detach().cpu().tolist()
                scores = output["scores"].detach().cpu().tolist()
                keep_indices = [idx for idx, score in enumerate(scores) if float(score) >= float(score_threshold)]
                predictions.append(
                    {
                        "boxes": [boxes[idx] for idx in keep_indices],
                        "scores": [scores[idx] for idx in keep_indices],
                    }
                )
                targets_payload.append({"boxes": target["boxes"].detach().cpu().tolist()})
    metrics = compute_detection_metrics(predictions, targets_payload)
    model.train()
    return metrics


def _set_batchnorm_eval(module: torch.nn.Module) -> None:
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()


def run_loss_evaluation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, float]:
    previous_mode = model.training
    model.train()
    model.apply(_set_batchnorm_eval)
    sums: dict[str, float] = {
        "loss": 0.0,
        "classification_loss": 0.0,
        "bbox_regression_loss": 0.0,
    }
    steps = 0
    with torch.no_grad():
        for batch in loader:
            images, targets = move_batch_to_device(batch, device=device)
            model_targets = strip_targets_for_model(targets)
            losses = run_model_forward(model, images, model_targets, amp_enabled=amp_enabled)
            sums["classification_loss"] += float(losses.get("classification", torch.tensor(0.0, device=device)).detach().item())
            sums["bbox_regression_loss"] += float(losses.get("bbox_regression", torch.tensor(0.0, device=device)).detach().item())
            sums["loss"] += float(sum(losses.values()).detach().item())
            steps += 1
    if previous_mode:
        model.train()
    else:
        model.eval()
    denominator = max(float(steps), 1.0)
    return {key: value / denominator for key, value in sums.items()}


def train_epoch(
    epoch: int,
    model: torch.nn.Module,
    strong_loader: DataLoader,
    weak_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    weak_sample_weight: float,
    log_interval: int,
) -> dict[str, float]:
    model.train()
    strong_iter = iter(strong_loader)
    weak_iter = cycle_loader(weak_loader) if weak_loader is not None else None
    steps = len(strong_loader)
    skipped_batches = 0
    sums = {
        "loss": 0.0,
        "strong_loss": 0.0,
        "weak_loss": 0.0,
        "cpu_batch_seconds": 0.0,
        "cpu_to_gpu_seconds": 0.0,
        "gpu_forward_seconds": 0.0,
        "gpu_backward_seconds": 0.0,
    }

    for step_idx in range(steps):
        cpu_batch_start = time.perf_counter()
        strong_batch = next(strong_iter)
        weak_batch = next(weak_iter) if weak_iter is not None else None
        cpu_batch_seconds = time.perf_counter() - cpu_batch_start

        if len(strong_batch["images"]) < 2:
            skipped_batches += 1
            print(f"Epoch {epoch:03d} step {step_idx + 1:04d}/{steps:04d} skipped strong batch with size < 2")
            continue
        if weak_batch is not None and len(weak_batch["images"]) < 2:
            weak_batch = None

        optimizer.zero_grad(set_to_none=True)

        strong_total, _, strong_transfer, strong_forward = compute_weighted_loss(
            model=model,
            batch=strong_batch,
            device=device,
            amp_enabled=amp_enabled,
            batch_scale=1.0,
        )
        total_loss = strong_total
        weak_loss_value = 0.0
        transfer_seconds = strong_transfer
        gpu_forward_seconds = strong_forward

        if weak_batch is not None and weak_sample_weight > 0.0:
            weak_total, _, weak_transfer, weak_forward = compute_weighted_loss(
                model=model,
                batch=weak_batch,
                device=device,
                amp_enabled=amp_enabled,
                batch_scale=weak_sample_weight,
            )
            total_loss = total_loss + weak_total
            weak_loss_value = float(weak_total.detach().item())
            transfer_seconds += weak_transfer
            gpu_forward_seconds += weak_forward

        backward_start = time.perf_counter()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        _synchronize_if_needed(device)
        backward_seconds = time.perf_counter() - backward_start

        total_loss_value = float(total_loss.detach().item())
        strong_loss_value = float(strong_total.detach().item())
        sums["loss"] += total_loss_value
        sums["strong_loss"] += strong_loss_value
        sums["weak_loss"] += weak_loss_value
        sums["cpu_batch_seconds"] += cpu_batch_seconds
        sums["cpu_to_gpu_seconds"] += transfer_seconds
        sums["gpu_forward_seconds"] += gpu_forward_seconds
        sums["gpu_backward_seconds"] += backward_seconds

        if log_interval > 0 and ((step_idx + 1) % log_interval == 0 or (step_idx + 1) == steps):
            print(
                f"Epoch {epoch:03d} step {step_idx + 1:04d}/{steps:04d} "
                f"loss={total_loss_value:.4f} strong={strong_loss_value:.4f} weak={weak_loss_value:.4f} "
                f"cpu={cpu_batch_seconds:.4f}s h2d={transfer_seconds:.4f}s "
                f"gpu_fwd={gpu_forward_seconds:.4f}s gpu_bwd={backward_seconds:.4f}s"
            )

    effective_steps = max(float(steps - skipped_batches), 1.0)
    metrics = {key: value / effective_steps for key, value in sums.items()}
    metrics["skipped_batches"] = float(skipped_batches)
    return metrics


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    optimizer_name = str(args.optimizer).lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def build_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace):
    scheduler_name = str(args.scheduler).lower()
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(args.epochs), 1))
    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(int(args.scheduler_step_size), 1),
            gamma=float(args.scheduler_gamma),
        )
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")


def command_train(args: argparse.Namespace) -> int:
    set_global_seed(int(args.seed))
    device = resolve_device(args.device)
    _, split_indices = build_split_indices_from_args(args)
    model_cfg_payload = load_yaml_file(args.model_config)
    model_cfg = dict(model_cfg_payload.get(args.model_name, {}))
    if not model_cfg:
        raise ValueError(f"Model name {args.model_name!r} not found in {args.model_config}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder_name = timestamp
    if args.prefix:
        safe_prefix = str(args.prefix).strip().replace(" ", "_")
        if safe_prefix:
            run_folder_name = f"{safe_prefix}_{timestamp}"
    run_dir = args.output_root / run_folder_name
    run_dir.mkdir(parents=True, exist_ok=True)

    console_log_path = run_dir / "console.log"
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    console_log = console_log_path.open("a", encoding="utf-8")
    sys.stdout = TeeStream(original_stdout, console_log)
    sys.stderr = TeeStream(original_stderr, console_log)
    try:
        print_split_summaries(split_indices)

        train_samples = split_indices["train"].labeled_samples
        use_weak_training = float(args.weak_sample_weight) > 0.0
        weak_samples = split_indices["train"].weak_samples if use_weak_training else []
        val_samples = split_indices["val"].labeled_samples
        if not train_samples:
            raise RuntimeError("No labeled training samples found.")
        if not val_samples:
            raise RuntimeError("No labeled validation samples found.")

        train_transform = build_albumentations_pipeline(args.augmentation)
        eval_transform = build_albumentations_pipeline(args.eval_preprocess, deterministic_only=True)
        image_store = RamImageStore(preload_images=bool(args.preload_images))
        preload_stats = image_store.preload([*train_samples, *weak_samples, *val_samples])

        strong_dataset = DetectionDataset(train_samples, image_store=image_store, transform=train_transform)
        weak_dataset = DetectionDataset(weak_samples, image_store=image_store, transform=train_transform) if weak_samples else None
        train_eval_dataset = DetectionDataset(train_samples, image_store=image_store, transform=eval_transform)
        val_dataset = DetectionDataset(val_samples, image_store=image_store, transform=eval_transform)

        strong_loader = build_dataloader(
            dataset=strong_dataset,
            batch_size=int(args.batch_size),
            shuffle=True,
            drop_last=True,
            workers=int(args.workers),
            pin_memory=bool(args.pin_memory),
            persistent_workers=bool(args.persistent_workers),
            prefetch_factor=int(args.prefetch_factor) if args.workers > 0 else None,
        )
        weak_loader = None
        if weak_dataset is not None and len(weak_dataset) > 0 and use_weak_training:
            weak_loader = build_dataloader(
                dataset=weak_dataset,
                batch_size=int(args.weak_batch_size),
                shuffle=True,
                drop_last=True,
                workers=int(args.workers),
                pin_memory=bool(args.pin_memory),
                persistent_workers=bool(args.persistent_workers),
                prefetch_factor=int(args.prefetch_factor) if args.workers > 0 else None,
            )
        val_loader = build_dataloader(
            dataset=val_dataset,
            batch_size=int(args.eval_batch_size),
            shuffle=False,
            drop_last=False,
            workers=int(args.workers),
            pin_memory=bool(args.pin_memory),
            persistent_workers=bool(args.persistent_workers),
            prefetch_factor=int(args.prefetch_factor) if args.workers > 0 else None,
        )
        train_eval_loader = build_dataloader(
            dataset=train_eval_dataset,
            batch_size=int(args.eval_batch_size),
            shuffle=False,
            drop_last=False,
            workers=int(args.workers),
            pin_memory=bool(args.pin_memory),
            persistent_workers=bool(args.persistent_workers),
            prefetch_factor=int(args.prefetch_factor) if args.workers > 0 else None,
        )

        write_json(run_dir / "split_summary.json", summarize_split_indices(split_indices))
        write_yaml(run_dir / "resolved_run_config.yaml", make_serializable(vars(args)))
        write_yaml(run_dir / "resolved_model_config.yaml", make_serializable(model_cfg))
        write_json(run_dir / "cache_stats.json", make_serializable(preload_stats))
        export_config_bundle(
            run_dir,
            project_config_path=args.project_config,
            model_config_path=args.model_config,
            run_config_path=args.run_config,
            config_overwrite_path=args.config_overwrite,
        )

        model = build_ssdlite_model(model_cfg).to(device)
        if args.init_checkpoint:
            checkpoint_state = load_checkpoint(model, args.init_checkpoint, device="cpu", strict=False)
            print(f"Loaded init checkpoint: {args.init_checkpoint}")
            if checkpoint_state:
                print(f"Checkpoint keys: {sorted(checkpoint_state.keys())}")

        optimizer = build_optimizer(model, args)
        scheduler = build_scheduler(optimizer, args)
        scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp))
        best_metric = -math.inf
        best_epoch = 0
        best_summary: dict[str, Any] | None = None
        history: list[dict[str, Any]] = []

        print(
            f"Training detector with batch_size={args.batch_size}, weak_batch_size={args.weak_batch_size}, "
            f"workers={args.workers}, preload_images={args.preload_images}, weak_weight={args.weak_sample_weight}, "
            f"use_weak_training={use_weak_training}"
        )
        selection_metric_name = str(args.selection_metric)
        eval_every_n_epoch = max(int(args.eval_every_n_epoch), 1)
        save_every_n_epoch = max(int(args.save_every_n_epoch), 1)
        max_save = max(int(args.max_save), 0)

        for epoch in range(1, int(args.epochs) + 1):
            train_metrics = train_epoch(
                epoch=epoch,
                model=model,
                strong_loader=strong_loader,
                weak_loader=weak_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                amp_enabled=bool(args.amp),
                weak_sample_weight=float(args.weak_sample_weight),
                log_interval=int(args.log_interval),
            )
            if scheduler is not None:
                scheduler.step()

            should_eval = (epoch % eval_every_n_epoch) == 0
            train_eval_loss = {"loss": float("nan"), "classification_loss": float("nan"), "bbox_regression_loss": float("nan")}
            train_eval_metrics: dict[str, float] = {}
            val_loss = {"loss": float("nan"), "classification_loss": float("nan"), "bbox_regression_loss": float("nan")}
            val_metrics: dict[str, float] = {}
            if should_eval:
                train_eval_loss = run_loss_evaluation(
                    model=model,
                    loader=train_eval_loader,
                    device=device,
                    amp_enabled=bool(args.amp),
                )
                train_eval_metrics = run_validation(
                    model=model,
                    loader=train_eval_loader,
                    device=device,
                    amp_enabled=bool(args.amp),
                    score_threshold=float(args.score_threshold),
                )
                val_loss = run_loss_evaluation(
                    model=model,
                    loader=val_loader,
                    device=device,
                    amp_enabled=bool(args.amp),
                )
                val_metrics = run_validation(
                    model=model,
                    loader=val_loader,
                    device=device,
                    amp_enabled=bool(args.amp),
                    score_threshold=float(args.score_threshold),
                )
            epoch_payload = {
                "epoch": epoch,
                "train": {
                    "loss": float(train_eval_loss["loss"]),
                    "strong_loss": float(train_metrics["strong_loss"]),
                    "weak_loss": float(train_metrics["weak_loss"]),
                    "selection_metric": float(train_eval_metrics.get(selection_metric_name, train_eval_metrics.get("map", float("nan")))),
                    "selection_metric_name": selection_metric_name,
                    "metric_eval": train_eval_metrics,
                    "loss_eval": train_eval_loss,
                    "timing": {
                        "cpu_batch_seconds": float(train_metrics["cpu_batch_seconds"]),
                        "cpu_to_gpu_seconds": float(train_metrics["cpu_to_gpu_seconds"]),
                        "gpu_forward_seconds": float(train_metrics["gpu_forward_seconds"]),
                        "gpu_backward_seconds": float(train_metrics["gpu_backward_seconds"]),
                        "skipped_batches": float(train_metrics["skipped_batches"]),
                    },
                },
                "val": {
                    "loss": float(val_loss["loss"]),
                    "selection_metric": float(val_metrics.get(selection_metric_name, val_metrics.get("map", float("nan")))),
                    "selection_metric_name": selection_metric_name,
                    "metric_eval": val_metrics,
                    "loss_eval": val_loss,
                },
                "lr": float(optimizer.param_groups[0]["lr"]),
                "evaluated": should_eval,
            }
            history.append(epoch_payload)
            write_json(run_dir / "history.json", history)
            train_metric_value = float(train_eval_metrics.get(selection_metric_name, train_eval_metrics.get("map", float("nan"))))
            val_metric_value = float(val_metrics.get(selection_metric_name, val_metrics.get("map", float("nan"))))
            if should_eval:
                print(
                    f"Epoch {epoch:03d} summary "
                    f"train_loss={train_eval_loss['loss']:.4f} "
                    f"train_{selection_metric_name}={train_metric_value:.4f} "
                    f"val_loss={val_loss['loss']:.4f} "
                    f"val_{selection_metric_name}={val_metric_value:.4f}"
                )

            if (epoch % save_every_n_epoch) == 0:
                save_checkpoint(
                    run_dir / "checkpoint_latest.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    extra_state={"epoch": epoch, "metrics": epoch_payload, "model_name": args.model_name},
                )
                if max_save > 0:
                    save_checkpoint(
                        run_dir / f"checkpoint_epoch_{epoch:03d}.pt",
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        extra_state={"epoch": epoch, "metrics": epoch_payload, "model_name": args.model_name},
                    )
                    prune_periodic_checkpoints(run_dir, max_keep=max_save)
            metric_value = val_metric_value
            if should_eval and metric_value > best_metric:
                best_metric = metric_value
                best_epoch = int(epoch)
                best_summary = {
                    "epoch": best_epoch,
                    "selection_metric_name": selection_metric_name,
                    "selection_metric_value": float(best_metric),
                    "train": epoch_payload["train"],
                    "val": epoch_payload["val"],
                    "lr": epoch_payload["lr"],
                    "checkpoint": "checkpoint_best.pt",
                }
                save_checkpoint(
                    run_dir / "checkpoint_best.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    extra_state={"epoch": epoch, "metrics": epoch_payload, "model_name": args.model_name},
                )
                write_json(run_dir / "best_epoch.json", make_serializable(best_summary))
                write_text(
                    run_dir / "best_epoch.txt",
                    "\n".join(
                        [
                            f"epoch: {best_epoch}",
                            f"{selection_metric_name}: {best_metric:.6f}",
                            f"checkpoint: checkpoint_best.pt",
                        ]
                    )
                    + "\n",
                )

        write_text(
            run_dir / "metrics.txt",
            "\n".join(
                [
                    f"best_{args.selection_metric}: {best_metric:.6f}",
                    f"best_epoch: {best_epoch}",
                    f"epochs: {args.epochs}",
                    f"cached_image_count: {int(preload_stats['cached_image_count'])}",
                    f"cache_build_seconds: {preload_stats['cache_build_seconds']:.6f}",
                ]
            )
            + "\n",
        )
        return 0
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        console_log.close()


def command_eval(args: argparse.Namespace) -> int:
    set_global_seed(int(args.seed))
    device = resolve_device(args.device)
    _, split_indices = build_split_indices_from_args(args)
    model_cfg_payload = load_yaml_file(args.model_config)
    model_cfg = dict(model_cfg_payload.get(args.model_name, {}))
    if not model_cfg:
        raise ValueError(f"Model name {args.model_name!r} not found in {args.model_config}")
    samples = split_indices[args.split].labeled_samples
    if not samples:
        raise RuntimeError(f"No labeled samples found for split={args.split}")
    image_store = RamImageStore(preload_images=bool(args.preload_images))
    preload_stats = image_store.preload(samples)
    dataset = DetectionDataset(
        samples,
        image_store=image_store,
        transform=build_albumentations_pipeline(args.eval_preprocess, deterministic_only=True),
    )
    loader = build_dataloader(
        dataset=dataset,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        drop_last=False,
        workers=int(args.workers),
        pin_memory=bool(args.pin_memory),
        persistent_workers=bool(args.persistent_workers),
        prefetch_factor=int(args.prefetch_factor) if args.workers > 0 else None,
    )
    model = build_ssdlite_model(model_cfg).to(device)
    load_checkpoint(model, args.checkpoint, device="cpu", strict=True)
    metrics = run_validation(model, loader, device, bool(args.amp), float(args.score_threshold))
    output_dir = args.output_root / f"eval_{args.split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "cache_stats.json", preload_stats)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def command_predict(args: argparse.Namespace) -> int:
    set_global_seed(int(args.seed))
    device = resolve_device(args.device)
    model_cfg_payload = load_yaml_file(args.model_config)
    model_cfg = dict(model_cfg_payload.get(args.model_name, {}))
    if not model_cfg:
        raise ValueError(f"Model name {args.model_name!r} not found in {args.model_config}")
    model = build_ssdlite_model(model_cfg).to(device)
    load_checkpoint(model, args.checkpoint, device="cpu", strict=True)
    model.eval()
    prediction_score_threshold = float(args.prediction_score_threshold)

    if args.video is not None:
        video_path = Path(args.video)
        if video_path.is_file():
            predict_video_file(
                model=model,
                video_path=video_path,
                checkpoint_path=args.checkpoint,
                device=device,
                amp_enabled=bool(args.amp),
                score_threshold=prediction_score_threshold,
            )
            return 0

    _, split_indices = build_split_indices_from_args(args)
    samples = select_samples_for_prediction(split_indices, split=args.split, video_name=args.video)
    if not samples:
        raise RuntimeError("No prediction samples found for the requested split/video.")
    image_store = RamImageStore(preload_images=bool(args.preload_images))
    image_store.preload(samples)
    dataset = DetectionDataset(
        samples,
        image_store=image_store,
        transform=build_albumentations_pipeline(args.eval_preprocess, deterministic_only=True),
    )
    loader = build_dataloader(
        dataset=dataset,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        drop_last=False,
        workers=int(args.workers),
        pin_memory=bool(args.pin_memory),
        persistent_workers=bool(args.persistent_workers),
        prefetch_factor=int(args.prefetch_factor) if args.workers > 0 else None,
    )

    predictions: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            images, targets = move_batch_to_device(batch, device)
            outputs = run_model_forward(model, images, None, bool(args.amp))
            for output, target in zip(outputs, targets):
                keep = output["scores"] >= prediction_score_threshold
                predictions.append(
                    {
                        "sample_id": target["sample_id"],
                        "video_name": target["video_name"],
                        "frame_idx": int(target["frame_idx"]),
                        "source": target["source"],
                        "boxes": output["boxes"][keep].detach().cpu().tolist(),
                        "scores": output["scores"][keep].detach().cpu().tolist(),
                        "labels": output["labels"][keep].detach().cpu().tolist(),
                    }
                )
    output_dir = args.output_root / f"predict_{args.split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "predictions.json", predictions)
    print(f"Saved {len(predictions)} predictions to {output_dir / 'predictions.json'}")
    return 0


def command_debug(args: argparse.Namespace) -> int:
    set_global_seed(int(args.seed))
    _, split_indices = build_split_indices_from_args(args)
    train_transform = build_albumentations_pipeline(args.augmentation)
    rng = np.random.default_rng(int(args.seed))

    print("Available debug sample counts by split:")
    for split_name in ("train", "val", "test"):
        split_index = split_indices[split_name]
        print(
            f"  {split_name}: "
            f"label={len(split_index.labeled_samples)} "
            f"sam2={len(split_index.weak_samples)}"
        )

    def choose_samples(samples: list[Any]) -> list[Any]:
        limit = int(args.limit_per_source)
        if len(samples) <= limit:
            return list(samples)
        indices = sorted(rng.choice(len(samples), size=limit, replace=False).tolist())
        return [samples[index] for index in indices]

    label_samples = choose_samples(split_indices[args.split].labeled_samples)
    sam2_samples = choose_samples(split_indices[args.split].weak_samples)
    debug_root = args.output_root / "debug"
    if debug_root.exists():
        shutil.rmtree(debug_root)
    label_dir = debug_root / "label"
    sam2_dir = debug_root / "sam2"
    label_dir.mkdir(parents=True, exist_ok=True)
    sam2_dir.mkdir(parents=True, exist_ok=True)

    image_store = RamImageStore(preload_images=bool(args.preload_images))
    image_store.preload([*label_samples, *sam2_samples])

    for source_name, samples, output_dir in (
        ("label", label_samples, label_dir),
        ("sam2", sam2_samples, sam2_dir),
    ):
        dataset = DetectionDataset(samples, image_store=image_store, transform=train_transform)
        for index in range(len(dataset)):
            item = dataset[index]
            image_uint8 = image_tensor_to_uint8(item["image"])
            boxes = item["target"]["boxes"].detach().cpu().tolist()
            roi_boxes = item["target"]["roi_boxes"].detach().cpu().tolist()
            debug_image = draw_debug_boxes(image_uint8, roi_boxes=roi_boxes, margin_boxes=boxes)
            bgr_image = cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR)
            sample = samples[index]
            output_path = output_dir / f"{index:04d}_{sample.video_name.replace('/', '_')}_{sample.frame_idx:06d}.jpg"
            cv2.imwrite(str(output_path), bgr_image)
        print(f"Saved {len(samples)} {source_name} debug images to {output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone SSDLite detector training script.")
    parser.add_argument("--project-config", type=Path, default=DEFAULT_PROJECT_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--run-config", type=Path, default=DEFAULT_RUN_CONFIG)
    parser.add_argument("--config-overwrite", "--config_overwrite", dest="config_overwrite", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-name", default="default")
    parser.add_argument("--labels-root", type=Path, default=PROJECT_ROOT / "input" / "labeled-data")
    parser.add_argument("--frames-root", type=Path, default=PROJECT_ROOT / "output" / "video_frames")
    parser.add_argument("--masks-root", type=Path, default=PROJECT_ROOT / "output" / "sam2_labels")
    parser.add_argument("--bbox-margin", type=float, default=20.0)
    parser.add_argument("--min-box-size", type=float, default=1.0)
    parser.add_argument("--weak-sample-weight", type=float, default=0.35)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--preload-images", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save-every-n-epoch", type=int, default=1)
    parser.add_argument("--eval-every-n-epoch", type=int, default=1)
    parser.add_argument("--max-save", type=int, default=5)

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--device", default="cuda:0")
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--weak-batch-size", type=int, default=8)
    train_parser.add_argument("--eval-batch-size", type=int, default=8)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--optimizer", default="AdamW")
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--momentum", type=float, default=0.9)
    train_parser.add_argument("--scheduler", default="cosine")
    train_parser.add_argument("--scheduler-step-size", type=int, default=10)
    train_parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    train_parser.add_argument("--selection-metric", default="map")
    train_parser.add_argument("--init-checkpoint", type=Path, default=None)
    train_parser.add_argument("--prefix", default="")
    train_parser.add_argument("--log-interval", type=int, default=10)
    train_parser.set_defaults(
        augmentation={},
        eval_preprocess={},
    )

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--device", default="cuda:0")
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--split", choices=("val", "test"), default="val")
    eval_parser.add_argument("--eval-batch-size", type=int, default=8)
    eval_parser.set_defaults(eval_preprocess={})

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--device", default="cuda:0")
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    predict_parser.add_argument("--video", default=None)
    predict_parser.add_argument("--eval-batch-size", type=int, default=8)
    predict_parser.add_argument("--prediction-score-threshold", type=float, default=0.5)
    predict_parser.set_defaults(eval_preprocess={})

    debug_parser = subparsers.add_parser("debug")
    debug_parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    debug_parser.add_argument("--limit-per-source", type=int, default=30)
    debug_parser.set_defaults(augmentation={})

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args = merge_cli_with_yaml(args, parser)
    if args.command == "train":
        return command_train(args)
    if args.command == "eval":
        return command_eval(args)
    if args.command == "predict":
        return command_predict(args)
    if args.command == "debug":
        return command_debug(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
