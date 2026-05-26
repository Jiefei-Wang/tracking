from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
import yaml

DEFAULT_SCORE_THRESHOLD = 0.1


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def _resolve_detector_model_name(
    run_dir: Path,
    model_registry: dict[str, Any],
    checkpoint_path: Path,
) -> str:
    for run_cfg_name in ("resolved_run_config.yaml", "run_config.yaml"):
        run_cfg_path = run_dir / run_cfg_name
        if run_cfg_path.is_file():
            run_cfg = _load_yaml_mapping(run_cfg_path)
            args_payload = run_cfg.get("args")
            if isinstance(args_payload, dict) and args_payload.get("model_name"):
                return str(args_payload["model_name"])

    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint_payload, dict) and checkpoint_payload.get("model_name"):
        return str(checkpoint_payload["model_name"])

    if len(model_registry) == 1:
        return next(iter(model_registry.keys()))

    raise ValueError(
        "Could not infer detector model name from the saved detector artifacts. "
        f"Ensure run config/checkpoint in {run_dir} contains model_name."
    )


def load_detector(
    detector_path: str | Path,
    checkpoint: str | Path = "checkpoint_best.pt",
    *,
    device: torch.device | str = "cpu",
) -> "SSDLiteDetector":
    run_dir = Path(detector_path)
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = run_dir / checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Detector checkpoint not found: {checkpoint_path}")

    model_config_path = run_dir / "model_config.yaml"
    if not model_config_path.is_file():
        raise FileNotFoundError(f"Detector model config not found: {model_config_path}")
    model_registry = _load_yaml_mapping(model_config_path)

    model_name = _resolve_detector_model_name(run_dir, model_registry, checkpoint_path)
    model_cfg = model_registry.get(model_name)
    if not isinstance(model_cfg, dict):
        raise ValueError(f"Model name {model_name!r} not found in {model_config_path}")

    device_obj = device if isinstance(device, torch.device) else torch.device(str(device))
    return SSDLiteDetector(dict(model_cfg), checkpoint_path=checkpoint_path, device=device_obj)


def build_ssdlite_model(model_cfg: dict[str, Any]) -> torch.nn.Module:
    image_size = model_cfg.get("image_size", [320, 320])
    if len(image_size) != 2:
        raise ValueError("model.image_size must be [height, width]")
    image_height = int(image_size[0])
    image_width = int(image_size[1])
    if (image_height, image_width) != (320, 320):
        raise ValueError(
            "torchvision ssdlite320_mobilenet_v3_large uses a fixed internal image size of 320x320; "
            f"got image_size={[image_height, image_width]}"
        )
    num_classes = int(model_cfg.get("num_classes", 2))
    box_score_thresh = float(model_cfg.get("box_score_thresh", 0.01))
    nms_thresh = float(model_cfg.get("nms_thresh", 0.45))
    detections_per_img = int(model_cfg.get("detections_per_img", 200))
    topk_candidates = int(model_cfg.get("topk_candidates", 400))
    image_mean = [float(v) for v in model_cfg.get("image_mean", [0.485, 0.456, 0.406])]
    image_std = [float(v) for v in model_cfg.get("image_std", [0.229, 0.224, 0.225])]
    trainable_backbone_layers = model_cfg.get("trainable_backbone_layers", None)
    pretrained_backbone = bool(model_cfg.get("pretrained_backbone", False))

    weights_backbone = None
    if pretrained_backbone:
        weights_backbone = "IMAGENET1K_V2"

    return ssdlite320_mobilenet_v3_large(
        num_classes=num_classes,
        weights=None,
        weights_backbone=weights_backbone,
        trainable_backbone_layers=trainable_backbone_layers,
        score_thresh=box_score_thresh,
        nms_thresh=nms_thresh,
        detections_per_img=detections_per_img,
        topk_candidates=topk_candidates,
        image_mean=image_mean,
        image_std=image_std,
    )


def _normalize_state_dict_for_load(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(key.startswith("model.") for key in state_dict):
        return {key[len("model.") :]: value for key, value in state_dict.items() if key.startswith("model.")}
    return state_dict


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(_normalize_state_dict_for_load(state_dict), strict=strict)
    return checkpoint if isinstance(checkpoint, dict) else {}


def save_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    extra_state: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "model": {f"model.{key}": value.detach().cpu() for key, value in model.state_dict().items()}
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if extra_state:
        payload.update(extra_state)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)


def _filter_detections(
    output: dict[str, torch.Tensor],
    score_threshold: float,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if output["boxes"].numel() == 0:
        return None, None

    boxes = output["boxes"].detach().cpu().numpy()
    scores = output["scores"].detach().cpu().numpy()
    keep = scores >= float(score_threshold)
    if not keep.any():
        return None, None
    return boxes[keep], scores[keep]


class SSDLiteDetector:
    def __init__(
        self,
        model_cfg: dict[str, Any],
        checkpoint_path: str | Path,
        device: torch.device,
    ) -> None:
        self.model = build_ssdlite_model(dict(model_cfg))
        load_checkpoint(self.model, checkpoint_path, device="cpu", strict=True)
        self.model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def detect_single(
        self,
        image_rgb: np.ndarray,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> tuple[Optional[list[float]], float]:
        tensor = torch.from_numpy(image_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0).to(self.device)
        outputs = self.model([tensor])[0]
        boxes, score_vals = _filter_detections(outputs, score_threshold)
        if boxes is None or score_vals is None:
            return None, 0.0
        best_idx = int(score_vals.argmax())
        return boxes[best_idx].astype(float).tolist(), float(score_vals[best_idx])

    @torch.no_grad()
    def detect_batch(
        self,
        images_rgb: Sequence[np.ndarray],
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> list[tuple[Optional[list[float]], float]]:
        tensors = [
            torch.from_numpy(image_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0).to(self.device)
            for image_rgb in images_rgb
        ]
        outputs = self.model(list(tensors))
        results: list[tuple[Optional[list[float]], float]] = []
        for output in outputs:
            boxes, score_vals = _filter_detections(output, score_threshold)
            if boxes is None or score_vals is None:
                results.append((None, 0.0))
                continue
            best_idx = int(score_vals.argmax())
            results.append((boxes[best_idx].astype(float).tolist(), float(score_vals[best_idx])))
        return results


@torch.no_grad()
def detector_extraction_ssdlite(
    detector: SSDLiteDetector,
    images_rgb: Sequence[np.ndarray],
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> list[list[dict[str, Any]]]:
    tensors = [
        torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1) / 255.0).to(detector.device)
        for image in images_rgb
    ]

    outputs = detector.model(list(tensors))
    results: list[list[dict[str, Any]]] = []

    for output in outputs:
        boxes, scores = _filter_detections(output, score_threshold)
        if boxes is None or scores is None:
            results.append([])
            continue

        detections = [
            {
                "bbox": box.astype(float).tolist(),
                "score": float(score),
            }
            for box, score in zip(boxes, scores)
        ]

        detections.sort(key=lambda item: item["score"], reverse=True)
        results.append(detections)

    return results
