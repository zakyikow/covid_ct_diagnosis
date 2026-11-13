"""The model (MobileNetV3 with transfer learning) and Grad-CAM heatmaps.

Transfer learning: start from MobileNetV3 pretrained on ImageNet (millions of everyday photos), replace its
last layer with one sized for our classes, and fine-tune the whole network on CT slices.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import nn

# timm's name for MobileNetV3-Large, the model used by the original baseline
MODEL_NAME = "mobilenetv3_large_100"


# Create MobileNetV3 with a new last layer for `num_classes` classes
def build_model(num_classes: int, pretrained: bool = True, drop_rate: float = 0.2) -> nn.Module:
    return timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)


# Load a trained model and the settings it was trained with
def load_model(path: str | Path, device: torch.device | str = "cpu") -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]

    model = build_model(len(cfg["classes"]), pretrained=False, drop_rate=cfg["model"]["drop_rate"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, cfg


# Run the model on every image of a loader and return the raw scores (logits), one row per image
@torch.no_grad()
def predict_logits(model: nn.Module, loader, device: torch.device) -> np.ndarray:
    model.eval()
    logits = np.zeros((len(loader.dataset), model.num_classes), dtype=np.float32)
    for images, _, rows in loader:
        logits[rows.numpy()] = model(images.to(device)).float().cpu().numpy()
    return logits


# Grad-CAM on the last feature map of the model: how much each region pushed it towards `class_idx`.
# A coarse grid (7 x 7 for a 224 px image), before it is resized to the image
def _grad_cam_features(model: nn.Module, image: torch.Tensor, class_idx: int | None) -> tuple[torch.Tensor, int]:
    model.eval()

    # Feature map of the last convolutional layer, and the class scores computed from it
    features = model.forward_features(image)
    features.retain_grad()
    logits = model.forward_head(features)
    if class_idx is None:
        class_idx = int(logits.argmax(1).item())

    # How much each feature channel matters for this class = its average gradient
    model.zero_grad(set_to_none=True)
    logits[0, class_idx].backward()
    channel_weights = features.grad.mean(dim=(2, 3), keepdim=True)

    # Weighted sum of the channels, keeping the positive part
    cam = F.relu((channel_weights * features).sum(1, keepdim=True))
    return cam.detach(), class_idx


# Scale a heatmap to 0..1 (0 = ignored, 1 = most)
def _to_unit_range(cam: torch.Tensor) -> np.ndarray:
    cam = cam - cam.min()
    return (cam / (cam.max() + 1e-8)).cpu().numpy()


# Grad-CAM heatmap at the size of the image, 0..1 (0 = ignored, 1 = most)
def grad_cam(model: nn.Module, image: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int]:
    cam, class_idx = _grad_cam_features(model, image, class_idx)
    cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    return _to_unit_range(cam), class_idx

