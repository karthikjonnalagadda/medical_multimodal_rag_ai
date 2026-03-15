"""
explainability/grad_cam.py
--------------------------
Small Grad-CAM helper tailored for PyTorch vision backbones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@dataclass
class GradCAMResult:
    heatmap: np.ndarray
    overlay: np.ndarray
    target_label: str
    score: float


class GradCAMExplainer:
    def __init__(self, model: "torch.nn.Module", target_layer: Any) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for Grad-CAM.")
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def _forward_hook(_module, _inputs, output):
            # Some backbones (e.g., DenseNet variants) use in-place ops and can yield
            # view tensors that autograd forbids modifying in-place during backward-hook flows.
            # Clone here to decouple from autograd-managed views.
            self.activations = output.detach().clone()

        def _backward_hook(_module, _grad_input, grad_output):
            # grad_output[0] can be a view; clone to avoid "view modified inplace" errors.
            self.gradients = grad_output[0].detach().clone()

        self.target_layer.register_forward_hook(_forward_hook)
        self.target_layer.register_full_backward_hook(_backward_hook)

    def explain(
        self,
        image_tensor: "torch.Tensor",
        class_index: int,
        input_image: np.ndarray,
        target_label: str,
    ) -> GradCAMResult:
        if image_tensor.ndim != 4:
            raise ValueError("Grad-CAM expects a 4D tensor.")

        self.model.zero_grad(set_to_none=True)
        output = self.model(image_tensor)
        # Ensure we don't keep a view-backed tensor that might be mutated by autograd internals.
        output = output.clone()
        if output.ndim == 2:
            score = output[:, class_index].sum().clone()
        else:
            raise ValueError("Unsupported output shape for Grad-CAM.")
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Failed to capture target activations/gradients.")

        # Stored tensors are already detached+cloned in the hooks, keep them read-only here.
        gradients = self.gradients
        activations = self.activations
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        # Avoid any chance of in-place behavior for compatibility across torch versions.
        cam = torch.relu(cam)
        cam = F.interpolate(cam, size=input_image.shape[:2], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / max(cam.max(), 1e-8)

        overlay = _overlay_heatmap(input_image, cam)
        return GradCAMResult(
            heatmap=cam,
            overlay=overlay,
            target_label=target_label,
            score=float(torch.sigmoid(score.detach()).item()),
        )


def _overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)

    color = np.zeros_like(image)
    color[..., 0] = heatmap
    color[..., 1] = np.clip(heatmap * 0.45, 0.0, 1.0)
    overlay = (1 - alpha) * image + alpha * color
    overlay = np.clip(overlay, 0.0, 1.0)
    return (overlay * 255).astype(np.uint8)
