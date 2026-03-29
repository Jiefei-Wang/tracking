from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large


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
