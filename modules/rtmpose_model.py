from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from modules.dlc_pytorch.models.backbones.base import BACKBONES
from modules.dlc_pytorch.models.criterions import CRITERIONS, LOSS_AGGREGATORS
from modules.dlc_pytorch.models.heads.base import HEADS
from modules.dlc_pytorch.models.predictors import PREDICTORS
from modules.dlc_pytorch.models.target_generators import TARGET_GENERATORS

# Ensure required classes are registered.
from modules.dlc_pytorch.models.backbones import cspnext as _cspnext  # noqa: F401
from modules.dlc_pytorch.models.criterions import aggregators as _aggregators  # noqa: F401
from modules.dlc_pytorch.models.criterions import kl_discrete as _kl_discrete  # noqa: F401
from modules.dlc_pytorch.models.heads import rtmcc_head as _rtmcc_head  # noqa: F401
from modules.dlc_pytorch.models.predictors import sim_cc as _simcc_predictor  # noqa: F401
from modules.dlc_pytorch.models.target_generators import sim_cc as _simcc_generator  # noqa: F401


def _filter_state_dict(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    plen = len(prefix)
    return {key[plen:]: value for key, value in state_dict.items() if key.startswith(prefix)}


def _align_head_state_to_shape(
    head_state: dict[str, torch.Tensor],
    model_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    aligned = dict(head_state)
    for key in ("final_layer.weight", "final_layer.bias"):
        if key not in aligned or key not in model_state:
            continue
        src = aligned[key]
        dst = model_state[key]
        if src.shape == dst.shape:
            continue
        if src.ndim >= 1 and dst.ndim >= 1 and src.shape[0] >= dst.shape[0] and src.shape[1:] == dst.shape[1:]:
            aligned[key] = src[: dst.shape[0]]
    return aligned


class StandaloneRTMPose(nn.Module):
    def __init__(
        self,
        model_cfg: dict[str, Any],
        backbone: nn.Module,
        head: nn.Module,
        visibility_head: nn.Module,
    ) -> None:
        super().__init__()
        self.cfg = model_cfg
        self.backbone = backbone
        self.head = head
        self.visibility_head = visibility_head
        self.backbone_stride = float(getattr(self.backbone, "stride", 32))
        self.head_stride = float(getattr(self.head, "stride", 1))
        self.output_stride = self.backbone_stride / self.head_stride

    @classmethod
    def build_model(
        cls,
        model_cfg: dict[str, Any],
    ) -> "StandaloneRTMPose":
        backbone_cfg = copy.deepcopy(model_cfg["model"]["backbone"])
        backbone = BACKBONES.build(backbone_cfg)

        head_cfg = copy.deepcopy(model_cfg["model"]["heads"]["bodypart"])
        criterion_cfg = copy.deepcopy(head_cfg["criterion"])
        weights: dict[str, float] = {}
        criteria = {}
        for loss_name, cfg in criterion_cfg.items():
            weights[loss_name] = float(cfg.get("weight", 1.0))
            clean_cfg = {key: value for key, value in cfg.items() if key != "weight"}
            criteria[loss_name] = CRITERIONS.build(clean_cfg)

        head_cfg["aggregator"] = LOSS_AGGREGATORS.build(
            {"type": "WeightedLossAggregator", "weights": weights}
        )
        head_cfg["criterion"] = criteria
        head_cfg["target_generator"] = TARGET_GENERATORS.build(head_cfg["target_generator"])
        head_cfg["predictor"] = PREDICTORS.build(head_cfg["predictor"])
        head = HEADS.build(head_cfg)
        visibility_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(int(head_cfg["in_channels"]), int(head_cfg["out_channels"])),
        )
        return cls(model_cfg=model_cfg, backbone=backbone, head=head, visibility_head=visibility_head)

    @classmethod
    def from_pretrained(
        cls,
        model_cfg: dict[str, Any],
        device: torch.device,
        backbone_checkpoint_path: str | Path | None = None,
        full_checkpoint_path: str | Path | None = None,
    ) -> "StandaloneRTMPose":
        model = cls.build_model(model_cfg)
        if full_checkpoint_path is not None:
            checkpoint = torch.load(Path(full_checkpoint_path), map_location="cpu")
            state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
            model.backbone.load_state_dict(_filter_state_dict(state, "backbone."), strict=True)
            head_state = _filter_state_dict(state, "heads.bodypart.")
            head_state = _align_head_state_to_shape(head_state, model.head.state_dict())
            model.head.load_state_dict(head_state, strict=False)
            model.visibility_head.load_state_dict(_filter_state_dict(state, "visibility_head."), strict=True)
        elif backbone_checkpoint_path is not None:
            checkpoint = torch.load(Path(backbone_checkpoint_path), map_location="cpu")
            state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
            model.backbone.load_state_dict(_filter_state_dict(state, "backbone."), strict=True)
        model.to(device)
        return model

    def forward(self, image_batch: torch.Tensor) -> dict[str, torch.Tensor]:
        if image_batch.dim() == 3:
            image_batch = image_batch.unsqueeze(0)
        features = self.backbone(image_batch)
        outputs = self.head(features)
        outputs["visibility_logits"] = self.visibility_head(features)
        return outputs

    @torch.no_grad()
    def predict(self, image_batch: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(image_batch)
        pred = self.head.predictor(self.output_stride, outputs)
        return pred["poses"]

    def compute_supervised_loss(
        self,
        outputs: dict[str, torch.Tensor],
        keypoints: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        targets = self.head.target_generator(self.output_stride, outputs, {"keypoints": keypoints})
        losses = self.head.get_loss(outputs, targets)
        return losses


def save_rtmpose_checkpoint(
    checkpoint_path: str | Path,
    model: StandaloneRTMPose,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    extra_state: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "model": {
            **{f"backbone.{key}": value.detach().cpu() for key, value in model.backbone.state_dict().items()},
            **{f"heads.bodypart.{key}": value.detach().cpu() for key, value in model.head.state_dict().items()},
            **{f"visibility_head.{key}": value.detach().cpu() for key, value in model.visibility_head.state_dict().items()},
        }
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


def load_rtmpose_checkpoint(
    checkpoint_path: str | Path,
    model: StandaloneRTMPose,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.backbone.load_state_dict(_filter_state_dict(state, "backbone."), strict=True)
    head_state = _filter_state_dict(state, "heads.bodypart.")
    head_state = _align_head_state_to_shape(head_state, model.head.state_dict())
    model.head.load_state_dict(head_state, strict=False)
    model.visibility_head.load_state_dict(_filter_state_dict(state, "visibility_head."), strict=True)
    if optimizer is not None and isinstance(checkpoint, dict) and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and isinstance(checkpoint, dict) and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint if isinstance(checkpoint, dict) else {}
