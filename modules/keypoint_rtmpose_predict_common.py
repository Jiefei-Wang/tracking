from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from modules.keypoint_rtmpose_model import StandaloneRTMPose, load_rtmpose_checkpoint


def _build_run_dir(output_root: Path, prefix: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prefix = str(prefix or "").strip().replace(" ", "_")
    folder = f"{safe_prefix}_{timestamp}" if safe_prefix else timestamp
    run_dir = output_root / folder
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_predict_output_dir(checkpoint_path: Path, output_root: Path, prefix: str = "") -> Path:
    if str(prefix or "").strip():
        return _build_run_dir(output_root, str(prefix))
    checkpoint_path = checkpoint_path.resolve()
    if checkpoint_path.parent.parent.resolve() == output_root.resolve():
        return checkpoint_path.parent.parent / f"{checkpoint_path.parent.name}_predict"
    return output_root / f"predict_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def load_yaml_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in YAML file: {path}")
    return payload


def resolve_detector_model_config(detector_checkpoint: Path, detector_model_name: str) -> dict[str, Any]:
    checkpoint_path = Path(detector_checkpoint)
    detector_run_dir = checkpoint_path.parent
    model_config_path = detector_run_dir / "model_config.yaml"
    if not model_config_path.is_file():
        raise FileNotFoundError(
            f"Expected detector model config at {model_config_path}. "
            "RTMPose now requires detector checkpoints from self-contained SSDLite run folders."
        )
    detector_model_registry = load_yaml_file(model_config_path)
    if detector_model_name not in detector_model_registry:
        raise ValueError(
            f"Unknown detector model name {detector_model_name!r} in {model_config_path}"
        )
    detector_model_cfg = detector_model_registry[detector_model_name]
    if not isinstance(detector_model_cfg, Mapping):
        raise ValueError(f"Detector model config {detector_model_name!r} in {model_config_path} is invalid")
    return dict(detector_model_cfg)


def build_pose_model(
    model_cfg: Mapping[str, Any],
    device: torch.device,
    checkpoint_path: Path | None = None,
    backbone_checkpoint_path: Path | None = None,
) -> StandaloneRTMPose:
    model = StandaloneRTMPose.from_pretrained(
        dict(model_cfg),
        device=device,
        full_checkpoint_path=checkpoint_path,
        backbone_checkpoint_path=backbone_checkpoint_path,
    )
    return model


def _resolve_model_config_from_run_dir(
    run_dir: Path,
    checkpoint_payload: Mapping[str, Any] | object,
) -> dict[str, Any]:
    if isinstance(checkpoint_payload, Mapping) and isinstance(checkpoint_payload.get("model_config"), Mapping):
        return dict(checkpoint_payload["model_config"])

    for config_name in ("resolved_model_config.yaml", "model_config.yaml"):
        config_path = run_dir / config_name
        if not config_path.is_file():
            continue
        payload = load_yaml_file(config_path)
        if "model" in payload:
            return dict(payload)
        raise ValueError(
            f"Expected a resolved RTMPose model config with a 'model' mapping in {config_path}"
        )

    raise FileNotFoundError(
        "Could not resolve RTMPose model config from saved run artifacts. "
        f"Expected checkpoint model_config or one of "
        f"{run_dir / 'resolved_model_config.yaml'} / {run_dir / 'model_config.yaml'}."
    )


def _resolve_bodyparts_from_run_dir(run_dir: Path) -> list[str]:
    project_config_path = run_dir / "project_config.yaml"
    if not project_config_path.is_file():
        raise FileNotFoundError(f"RTMPose project config not found: {project_config_path}")

    payload = load_yaml_file(project_config_path)
    bodyparts = payload.get("bodyparts")
    if not isinstance(bodyparts, Sequence) or isinstance(bodyparts, (str, bytes)):
        raise ValueError(f"Expected 'bodyparts' list in {project_config_path}")

    bodypart_names = [str(name) for name in bodyparts]
    if not bodypart_names:
        raise ValueError(f"Expected non-empty 'bodyparts' list in {project_config_path}")
    return bodypart_names


def _require_model_bodyparts(model: StandaloneRTMPose) -> list[str]:
    bodyparts = getattr(model, "bodyparts", None)
    if not isinstance(bodyparts, Sequence) or isinstance(bodyparts, (str, bytes)):
        raise ValueError("RTMPose model is missing bodypart metadata. Load it from a saved run directory.")
    bodypart_names = [str(name) for name in bodyparts]
    if not bodypart_names:
        raise ValueError("RTMPose model bodypart metadata is empty.")
    return bodypart_names


def load_model_from_checkpoint_for_inference(
    model_path: str | Path,
    checkpoint: str | Path = "checkpoint_best.pt",
    device: torch.device | str = "cpu",
) -> tuple[StandaloneRTMPose, dict[str, Any]]:
    run_dir = Path(model_path)
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        if not checkpoint_path.is_file():
            checkpoint_path = run_dir / checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"RTMPose checkpoint not found: {checkpoint_path}")

    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
    model_cfg = _resolve_model_config_from_run_dir(run_dir, checkpoint_payload)
    bodyparts = _resolve_bodyparts_from_run_dir(run_dir)
    device_obj = device if isinstance(device, torch.device) else torch.device(str(device))

    model = build_pose_model(model_cfg, device_obj, None, None)
    checkpoint_payload = load_rtmpose_checkpoint(checkpoint_path, model)
    if isinstance(checkpoint_payload, Mapping) and isinstance(checkpoint_payload.get("model_config"), Mapping):
        model.cfg = dict(checkpoint_payload["model_config"])
    else:
        model.cfg = dict(model_cfg)
    model.bodyparts = bodyparts
    model.eval()
    return model, dict(checkpoint_payload) if isinstance(checkpoint_payload, Mapping) else {}


def simcc_probabilities(outputs: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    probs_x = F.softmax(outputs["x"], dim=-1)
    probs_y = F.softmax(outputs["y"], dim=-1)
    return probs_x, probs_y


def visibility_probabilities(outputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.sigmoid(outputs["visibility_logits"])


def decode_keypoints_with_predictor(
    model: StandaloneRTMPose,
    outputs: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    predictor = cast(Any, model.head.predictor)
    pred = predictor(model.output_stride, dict(outputs))
    poses = pred["poses"]
    if poses.ndim == 4 and poses.shape[1] == 1:
        poses = poses[:, 0]
    points = poses[..., :2]
    if poses.shape[-1] >= 3:
        scores = poses[..., 2]
    else:
        scores = torch.ones_like(points[..., 0])
    return points, scores


@torch.no_grad()
def predict_dataset(
    model: StandaloneRTMPose,
    dataloader: DataLoader,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    bodyparts = _require_model_bodyparts(model)
    device = next(model.parameters()).device
    for batch in dataloader:
        image = batch["image"].to(device)
        outputs = model(image)
        points, coord_scores = decode_keypoints_with_predictor(model, outputs)
        visibility = visibility_probabilities(outputs)
        points = points.cpu().numpy()
        coord_scores = coord_scores.cpu().numpy()
        visibility = visibility.cpu().numpy()
        crop_boxes = batch["crop_box"].cpu().numpy()
        orig_sizes = batch["orig_size"].cpu().numpy()
        for i in range(points.shape[0]):
            crop_box = crop_boxes[i]
            crop_w = max(1.0, crop_box[2] - crop_box[0])
            crop_h = max(1.0, crop_box[3] - crop_box[1])
            input_w = float(batch["image"].shape[-1])
            input_h = float(batch["image"].shape[-2])
            mapped = points[i].copy()
            mapped[:, 0] = crop_box[0] + mapped[:, 0] * crop_w / input_w
            mapped[:, 1] = crop_box[1] + mapped[:, 1] * crop_h / input_h
            predictions.append(
                {
                    "video_name": batch["video_name"][i],
                    "frame_idx": int(batch["frame_idx"][i].item()),
                    "source_name": batch["source_name"][i],
                    "image_width": int(orig_sizes[i, 0]),
                    "image_height": int(orig_sizes[i, 1]),
                    "crop_box": [float(v) for v in crop_box.tolist()],
                    "keypoints": {
                        bodypart: {
                            "x": float(mapped[kp_idx, 0]),
                            "y": float(mapped[kp_idx, 1]),
                            "score": float(coord_scores[i, kp_idx]),
                            "visibility_score": float(visibility[i, kp_idx]),
                        }
                        for kp_idx, bodypart in enumerate(bodyparts)
                    },
                }
            )
    return predictions


@torch.no_grad()
def keypoint_extraction_rtmpose(
    model: StandaloneRTMPose,
    images_rgb: Sequence[np.ndarray],
    detections: Sequence[Sequence[Mapping[str, Any]]],
    crop_expand_scale: float = 0.15,
) -> list[list[dict[str, Any]]]:
    if len(images_rgb) != len(detections):
        raise ValueError("images_rgb and detections must have the same length")

    bodyparts = _require_model_bodyparts(model)
    model_cfg = dict(getattr(model, "cfg", {}) or {})
    head_cfg = dict(model_cfg.get("model", {}).get("heads", {}).get("bodypart", {}))
    input_w, input_h = [int(v) for v in head_cfg.get("input_size", [256, 256])]
    image_mean = np.asarray(model_cfg.get("image_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    image_std = np.asarray(model_cfg.get("image_std", [0.229, 0.224, 0.225]), dtype=np.float32)
    device = next(model.parameters()).device

    results: list[list[dict[str, Any]]] = [[] for _ in images_rgb]
    pose_tensors: list[torch.Tensor] = []
    pose_meta: list[dict[str, Any]] = []

    for image_idx, (image_rgb, image_detections) in enumerate(zip(images_rgb, detections)):
        image_h, image_w = image_rgb.shape[:2]
        for detection in image_detections:
            bbox = detection.get("bbox")
            if bbox is None:
                raise ValueError("Each detection must include a 'bbox' field")

            x1, y1, x2, y2 = [float(v) for v in bbox]
            box_w = max(1.0, x2 - x1)
            box_h = max(1.0, y2 - y1)
            pad_x = max(0.0, box_w * float(crop_expand_scale))
            pad_y = max(0.0, box_h * float(crop_expand_scale))
            crop_box = [
                float(max(0, int(np.floor(x1 - pad_x)))),
                float(max(0, int(np.floor(y1 - pad_y)))),
                float(min(image_w, int(np.ceil(x2 + pad_x)))),
                float(min(image_h, int(np.ceil(y2 + pad_y)))),
            ]

            result_entry = {
                "bbox": [x1, y1, x2, y2],
                "score": float(detection.get("score", 0.0)),
                "crop_box": crop_box,
                "keypoints": {},
            }
            results[image_idx].append(result_entry)

            crop_x1, crop_y1, crop_x2, crop_y2 = [int(v) for v in crop_box]
            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                continue

            crop = image_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
            resized = cv2.resize(crop, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
            image_float = resized.astype(np.float32) / 255.0
            image_norm = (image_float - image_mean[None, None, :]) / image_std[None, None, :]
            tensor = torch.from_numpy(np.ascontiguousarray(image_norm.transpose(2, 0, 1))).float()
            pose_tensors.append(tensor)
            pose_meta.append(
                {
                    "image_idx": image_idx,
                    "result_idx": len(results[image_idx]) - 1,
                    "crop_box": crop_box,
                }
            )

    if not pose_tensors:
        return results

    tensor_batch = torch.stack(pose_tensors, dim=0).to(device)
    outputs = model(tensor_batch)
    points, coord_scores = decode_keypoints_with_predictor(model, outputs)
    visibility_scores = visibility_probabilities(outputs)

    points = points.detach().cpu().numpy()
    coord_scores = coord_scores.detach().cpu().numpy()
    visibility_scores = visibility_scores.detach().cpu().numpy()

    for idx, meta in enumerate(pose_meta):
        crop_box = meta["crop_box"]
        crop_w = max(1.0, crop_box[2] - crop_box[0])
        crop_h = max(1.0, crop_box[3] - crop_box[1])
        mapped = points[idx].copy()
        mapped[:, 0] = crop_box[0] + mapped[:, 0] * crop_w / float(input_w)
        mapped[:, 1] = crop_box[1] + mapped[:, 1] * crop_h / float(input_h)

        results[meta["image_idx"]][meta["result_idx"]]["keypoints"] = {
            bodypart: {
                "x": float(mapped[kp_idx, 0]),
                "y": float(mapped[kp_idx, 1]),
                "score": float(coord_scores[idx, kp_idx]),
                "visibility_score": float(visibility_scores[idx, kp_idx]),
            }
            for kp_idx, bodypart in enumerate(bodyparts)
        }

    return results


def draw_pose_prediction(
    frame_bgr: np.ndarray,
    bodyparts: Sequence[str],
    skeleton: Sequence[Sequence[str]],
    keypoints_xy: np.ndarray,
    confidence_scores: np.ndarray,
    visibility_scores: np.ndarray,
    roi_box: Sequence[int],
    expanded_roi_box: Sequence[int],
    score_cutoff: float,
    visibility_cutoff: float,
) -> np.ndarray:
    canvas = frame_bgr.copy()
    roi_x1, roi_y1, roi_x2, roi_y2 = [int(round(float(v))) for v in roi_box]
    exp_x1, exp_y1, exp_x2, exp_y2 = [int(round(float(v))) for v in expanded_roi_box]
    cv2.rectangle(canvas, (roi_x1, roi_y1), (roi_x2 - 1, roi_y2 - 1), (0, 255, 255), 2)
    cv2.rectangle(canvas, (exp_x1, exp_y1), (exp_x2 - 1, exp_y2 - 1), (255, 128, 0), 2)
    name_to_idx = {name: idx for idx, name in enumerate(bodyparts)}
    for left, right in skeleton:
        if left not in name_to_idx or right not in name_to_idx:
            continue
        li = name_to_idx[left]
        ri = name_to_idx[right]
        if (
            confidence_scores[li] < score_cutoff
            or confidence_scores[ri] < score_cutoff
            or visibility_scores[li] < visibility_cutoff
            or visibility_scores[ri] < visibility_cutoff
        ):
            continue
        p1 = (int(round(keypoints_xy[li, 0])), int(round(keypoints_xy[li, 1])))
        p2 = (int(round(keypoints_xy[ri, 0])), int(round(keypoints_xy[ri, 1])))
        cv2.line(canvas, p1, p2, (255, 255, 0), 1, cv2.LINE_AA)
    for idx, bodypart in enumerate(bodyparts):
        if confidence_scores[idx] < score_cutoff or visibility_scores[idx] < visibility_cutoff:
            continue
        x = int(round(keypoints_xy[idx, 0]))
        y = int(round(keypoints_xy[idx, 1]))
        cv2.circle(canvas, (x, y), 3, (0, 255, 0), -1)
        cv2.putText(canvas, bodypart, (x + 3, y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
    return canvas
