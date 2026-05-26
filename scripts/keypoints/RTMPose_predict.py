#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from modules.keypoint_rtmpose_predict_common import (
    build_predict_output_dir,
    decode_keypoints_with_predictor,
    draw_pose_prediction,
    load_model_from_checkpoint_for_inference,
    predict_dataset,
    visibility_probabilities,
)
from modules.detector_ssdlite_model import load_detector
from RTMPose import (
    DEFAULT_MODEL_CONFIG,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PROJECT_CONFIG,
    DEFAULT_RUN_CONFIG,
    PROJECT_ROOT,
    SPLIT_NAMES,
    RTMPoseDataset,
    build_all_split_indices,
    build_dataloader,
    expand_box_xyxy,
    load_project_config,
    load_yaml_file,
    merge_cli_with_yaml,
    prepare_training_components,
    resolve_device,
    resolve_model_config,
    select_samples_for_split,
    set_seed,
    validate_mutual_exclusion,
    write_json,
)


@torch.no_grad()
def predict_video_file(args: argparse.Namespace) -> Path:
    project_cfg = load_project_config(args.project_config)
    model, _ = load_model_from_checkpoint_for_inference(
        model_path=args.checkpoint.parent,
        checkpoint=args.checkpoint,
        device=resolve_device(args.device),
    )
    model_cfg = dict(model.cfg)
    detector = load_detector(
        args.detector_checkpoint.parent,
        args.detector_checkpoint,
        device=resolve_device(str(args.detector_device or args.device)),
    )
    input_w, input_h = tuple(model_cfg["model"]["heads"]["bodypart"].get("input_size", [256, 256]))
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {args.video}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_dir = build_predict_output_dir(
        args.checkpoint,
        args.output_root,
        prefix=str(args.prefix or ""),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / Path(args.video).name
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    image_mean = np.asarray(model_cfg.get("image_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    image_std = np.asarray(model_cfg.get("image_std", [0.229, 0.224, 0.225]), dtype=np.float32)
    progress = tqdm(total=total_frames if total_frames > 0 else None, desc="predict_video")
    device = next(model.parameters()).device
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        det_box, _ = detector.detect_single(frame_rgb, score_threshold=float(args.detector_score_threshold))
        if det_box is not None:
            crop_box = expand_box_xyxy(det_box, frame_rgb.shape[:2], float(args.crop_expand_scale))
            x1, y1, x2, y2 = crop_box
            crop = frame_rgb[y1:y2, x1:x2]
            resized = cv2.resize(crop, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
            image_float = resized.astype(np.float32) / 255.0
            image_norm = (image_float - image_mean[None, None, :]) / image_std[None, None, :]
            tensor = torch.from_numpy(np.ascontiguousarray(image_norm.transpose(2, 0, 1))).unsqueeze(0).to(device)
            outputs = model(tensor)
            points, coord_scores = decode_keypoints_with_predictor(model, outputs)
            visibility = visibility_probabilities(outputs)
            points = points[0].cpu().numpy()
            confidence_scores = coord_scores[0].cpu().numpy()
            visibility_scores = visibility[0].cpu().numpy()
            crop_w = max(1.0, x2 - x1)
            crop_h = max(1.0, y2 - y1)
            points[:, 0] = x1 + points[:, 0] * crop_w / float(input_w)
            points[:, 1] = y1 + points[:, 1] * crop_h / float(input_h)
            frame_bgr = draw_pose_prediction(
                frame_bgr,
                project_cfg.bodyparts,
                project_cfg.skeleton,
                points,
                confidence_scores,
                visibility_scores,
                det_box,
                crop_box,
                float(args.score_cutoff),
                float(args.visibility_cutoff),
            )
        writer.write(frame_bgr)
        progress.update(1)
    progress.close()
    writer.release()
    cap.release()
    print(f"Wrote annotated video to {output_path}")
    return output_path


def command_predict(args: argparse.Namespace) -> int:
    if args.video is not None:
        if not args.video.is_file():
            raise FileNotFoundError(
                f"--video was provided, but file does not exist: {args.video}. "
                "Pass a valid video path or omit --video to run dataset prediction mode."
            )
        predict_video_file(args)
        return 0

    device = resolve_device(args.device)
    project_cfg = load_project_config(args.project_config)
    use_masks = float(getattr(args, "weak_sample_weight", 1.0)) > 0.0
    model, _ = load_model_from_checkpoint_for_inference(
        model_path=args.checkpoint.parent,
        checkpoint=args.checkpoint,
        device=device,
    )
    model_cfg = dict(model.cfg)
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
    predictions = predict_dataset(
        model=model,
        dataloader=loader,
    )
    output_dir = build_predict_output_dir(
        args.checkpoint,
        args.output_root,
        prefix=str(args.prefix or ""),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions.json"
    write_json(output_path, predictions)
    print(f"Wrote predictions to {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone RTMPose prediction script.")
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
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--preload-images", action="store_true")
    parser.add_argument("--preload-masks", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--auto-val-fraction", type=float, default=0.1)
    parser.add_argument("--crop-expand-scale", type=float, default=0.15)
    parser.add_argument("--bbox-margin", type=float, default=20.0)
    parser.add_argument("--top-down-margin", type=int, default=0)
    parser.add_argument("--top-down-crop-with-context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--score-cutoff", type=float, default=0.0)
    parser.add_argument("--visibility-cutoff", type=float, default=0.5)
    parser.add_argument("--weak-sample-weight", type=float, default=1.0)

    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--split", choices=SPLIT_NAMES, default="test")
    parser.add_argument("--video-name", type=str, default=None)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.command = "predict"
    args = merge_cli_with_yaml(args, parser)
    set_seed(int(args.seed))
    return command_predict(args)


if __name__ == "__main__":
    raise SystemExit(main())
