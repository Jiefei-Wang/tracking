from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from modules.rtmpose_model import StandaloneRTMPose, load_rtmpose_checkpoint


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


def load_model_from_checkpoint_for_inference(
    checkpoint_path: Path,
    device: torch.device,
    resolve_model_config: Callable[[], Mapping[str, Any]],
    load_yaml_file: Callable[[Path], Mapping[str, Any]],
    build_pose_model: Callable[[Mapping[str, Any], torch.device, Path | None, Path | None], StandaloneRTMPose],
) -> tuple[StandaloneRTMPose, dict[str, Any]]:
    checkpoint_payload = torch.load(Path(checkpoint_path), map_location="cpu")
    model_cfg: Mapping[str, Any] = resolve_model_config()
    if isinstance(checkpoint_payload, Mapping) and isinstance(checkpoint_payload.get("model_config"), Mapping):
        model_cfg = dict(checkpoint_payload["model_config"])
    else:
        resolved_model_config_path = Path(checkpoint_path).parent / "resolved_model_config.yaml"
        run_model_config_path = Path(checkpoint_path).parent / "model_config.yaml"
        for candidate in (resolved_model_config_path, run_model_config_path):
            if candidate.is_file():
                payload = load_yaml_file(candidate)
                if isinstance(payload, Mapping) and "model" in payload:
                    model_cfg = dict(payload)
                    break
    model = build_pose_model(
        model_cfg,
        device=device,
        checkpoint_path=None,
        backbone_checkpoint_path=None,
    )
    checkpoint_payload = load_rtmpose_checkpoint(checkpoint_path, model)
    if isinstance(checkpoint_payload, Mapping) and isinstance(checkpoint_payload.get("model_config"), Mapping):
        model.cfg = dict(checkpoint_payload["model_config"])
    else:
        model.cfg = dict(model_cfg)
    model.eval()
    return model, dict(checkpoint_payload) if isinstance(checkpoint_payload, Mapping) else {}


@torch.no_grad()
def predict_dataset(
    model: StandaloneRTMPose,
    dataloader: DataLoader,
    bodyparts: Sequence[str],
    decode_keypoints_with_predictor: Callable[[StandaloneRTMPose, Mapping[str, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]],
    visibility_probabilities: Callable[[Mapping[str, torch.Tensor]], torch.Tensor],
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
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
