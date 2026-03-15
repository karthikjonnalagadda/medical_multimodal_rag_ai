"""
vision/medical_models.py
------------------------
Backbone registry for medical imaging models used by the multimodal platform.

The implementations are dependency-aware:
- `torchxrayvision` is the preferred turnkey backend for chest X-rays.
- Hugging Face vision-language checkpoints can be used for MedCLIP-like
  and BioViL-compatible inference when available.
- A CheXNet-style DenseNet fallback keeps the rest of the application usable
  even when specialist checkpoints are not installed locally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from loguru import logger

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as tv_models
    import torchvision.transforms as T
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

try:
    import torchxrayvision as xrv
    _XRV_AVAILABLE = True
except ImportError:
    _XRV_AVAILABLE = False


DEFAULT_CHEST_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural Thickening",
    "Pneumonia",
    "Pneumothorax",
]


@dataclass
class VisionBackendOutput:
    probabilities: dict[str, float]
    embedding: Optional[np.ndarray] = None
    feature_map: Any = None
    attention_map: Optional[np.ndarray] = None
    backend_name: str = ""
    calibration_note: str = ""
    raw_logits: Optional[np.ndarray] = None


@dataclass
class VisionBackendSpec:
    name: str
    task: str
    model_id: Optional[str] = None
    checkpoint_path: Optional[str] = None
    labels: list[str] = field(default_factory=lambda: list(DEFAULT_CHEST_LABELS))
    image_size: int = 224
    grayscale: bool = False


class BaseVisionBackend:
    """Unified interface for interchangeable medical imaging backbones."""

    def __init__(
        self,
        spec: VisionBackendSpec,
        device: Optional[str] = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for vision backbones.")
        self.spec = spec
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: nn.Module | Any = None
        self.preprocess = None
        self.target_layer = None
        self._last_feature_map = None
        self._load()

    def _load(self) -> None:
        raise NotImplementedError

    def predict(self, image_tensor: "torch.Tensor") -> VisionBackendOutput:
        raise NotImplementedError

    def get_target_layer(self):
        return self.target_layer

    def get_last_feature_map(self):
        return self._last_feature_map


class CheXNetBackend(BaseVisionBackend):
    """DenseNet121 with a disease head suitable for CheXpert-style checkpoints."""

    def _load(self) -> None:
        self.spec.grayscale = False
        model = tv_models.densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, len(self.spec.labels))

        if self.spec.checkpoint_path:
            checkpoint_path = Path(self.spec.checkpoint_path)
            if checkpoint_path.exists():
                state = torch.load(checkpoint_path, map_location=self.device)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                cleaned_state = {
                    key.replace("module.", ""): value
                    for key, value in state.items()
                }
                missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
                if missing:
                    logger.warning("CheXNet checkpoint missing keys: {}", missing[:5])
                if unexpected:
                    logger.warning("CheXNet checkpoint unexpected keys: {}", unexpected[:5])
            else:
                logger.warning(
                    "CheXNet checkpoint not found at '{}'; continuing with randomly initialized classification head.",
                    checkpoint_path,
                )
        else:
            logger.warning("CheXNet backend loaded without a medical checkpoint; using random head weights.")

        self.model = model.to(self.device).eval()
        self.target_layer = self.model.features[-1]
        self.preprocess = T.Compose([
            T.Resize((self.spec.image_size, self.spec.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        def _hook(_module, _inputs, output):
            self._last_feature_map = output.detach()

        self.target_layer.register_forward_hook(_hook)

    def predict(self, image_tensor: "torch.Tensor") -> VisionBackendOutput:
        with torch.no_grad():
            logits = self.model(image_tensor.to(self.device))
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        embedding = None
        if self._last_feature_map is not None:
            pooled = F.adaptive_avg_pool2d(self._last_feature_map, (1, 1)).flatten(1)
            embedding = pooled.squeeze(0).cpu().numpy()
        return VisionBackendOutput(
            probabilities={label: float(prob) for label, prob in zip(self.spec.labels, probs)},
            embedding=embedding,
            feature_map=self._last_feature_map,
            backend_name="chexnet",
            raw_logits=logits.squeeze(0).detach().cpu().numpy(),
        )


class TorchXRayVisionBackend(BaseVisionBackend):
    """Chest X-ray backend using pretrained weights from TorchXRayVision."""

    def _load(self) -> None:
        if not _XRV_AVAILABLE:
            raise RuntimeError("torchxrayvision is not installed.")
        self.spec.grayscale = True
        self.model = xrv.models.DenseNet(weights="densenet121-res224-all").to(self.device).eval()
        self.spec.labels = [label.replace("_", " ") for label in self.model.pathologies]
        self.target_layer = getattr(self.model, "features", None)
        self.preprocess = T.Compose([
            T.Resize((224, 224)),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
        ])

        if self.target_layer is not None:
            def _hook(_module, _inputs, output):
                self._last_feature_map = output.detach()

            self.target_layer.register_forward_hook(_hook)

    def predict(self, image_tensor: "torch.Tensor") -> VisionBackendOutput:
        x = image_tensor.to(self.device)
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        embedding = None
        if self._last_feature_map is not None and len(self._last_feature_map.shape) == 4:
            pooled = F.adaptive_avg_pool2d(self._last_feature_map, (1, 1)).flatten(1)
            embedding = pooled.squeeze(0).cpu().numpy()
        return VisionBackendOutput(
            probabilities={label: float(prob) for label, prob in zip(self.spec.labels, probs)},
            embedding=embedding,
            feature_map=self._last_feature_map,
            backend_name="torchxrayvision",
            calibration_note="Pretrained chest X-ray priors from TorchXRayVision.",
            raw_logits=logits.squeeze(0).detach().cpu().numpy(),
        )


class HuggingFaceVisionBackend(BaseVisionBackend):
    """Wrapper for Hugging Face image encoders/classifiers used as MedCLIP/BioViL adapters."""

    def _load(self) -> None:
        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers is required for Hugging Face vision backends.")
        if not self.spec.model_id:
            raise ValueError("A Hugging Face model_id is required for this backend.")
        self.spec.grayscale = False

        self.processor = AutoImageProcessor.from_pretrained(self.spec.model_id)
        try:
            self.model = AutoModelForImageClassification.from_pretrained(self.spec.model_id).to(self.device).eval()
            self._is_classifier = True
        except Exception:
            self.model = AutoModel.from_pretrained(self.spec.model_id).to(self.device).eval()
            self._is_classifier = False
        self.preprocess = None
        self.target_layer = None

    def predict(self, image_tensor: "torch.Tensor") -> VisionBackendOutput:
        images = image_tensor.squeeze(0).cpu()
        if images.shape[0] == 1:
            images = images.repeat(3, 1, 1)
        pixel_values = self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)

        logits = getattr(outputs, "logits", None)
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        pooled = getattr(outputs, "pooler_output", None)

        if logits is not None:
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            labels = self.spec.labels
            if hasattr(self.model.config, "id2label") and self.model.config.id2label:
                labels = [self.model.config.id2label[i] for i in range(len(probs))]
            return VisionBackendOutput(
                probabilities={label: float(prob) for label, prob in zip(labels, probs)},
                embedding=(pooled.squeeze(0).cpu().numpy() if pooled is not None else None),
                backend_name=self.spec.name,
                raw_logits=logits.squeeze(0).detach().cpu().numpy(),
            )

        if pooled is None and last_hidden_state is not None:
            pooled = last_hidden_state.mean(dim=1)
        embedding = pooled.squeeze(0).cpu().numpy() if pooled is not None else None
        pseudo_probs = {label: 0.0 for label in self.spec.labels}
        return VisionBackendOutput(
            probabilities=pseudo_probs,
            embedding=embedding,
            backend_name=self.spec.name,
            calibration_note=(
                "This backend exposes an embedding encoder. Pair it with a retrieval head "
                "or contrastive report-matching objective for diagnosis scoring."
            ),
        )


def build_backend(
    backend_name: str,
    device: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    model_id: Optional[str] = None,
    labels: Optional[list[str]] = None,
) -> BaseVisionBackend:
    """
    Construct a backend by name.

    Supported names:
    - `torchxrayvision`
    - `chexnet`
    - `medclip`
    - `biovil`
    """
    spec = VisionBackendSpec(
        name=backend_name,
        task="chest_xray_multilabel",
        checkpoint_path=checkpoint_path,
        model_id=model_id,
        labels=labels or list(DEFAULT_CHEST_LABELS),
    )

    if backend_name == "torchxrayvision":
        return TorchXRayVisionBackend(spec=spec, device=device)
    if backend_name == "chexnet":
        return CheXNetBackend(spec=spec, device=device)
    if backend_name == "medclip":
        return HuggingFaceVisionBackend(
            spec=VisionBackendSpec(
                name="medclip",
                task="vision_language_embedding",
                model_id=model_id or "microsoft/swin-tiny-patch4-window7-224",
                labels=labels or list(DEFAULT_CHEST_LABELS),
            ),
            device=device,
        )
    if backend_name == "biovil":
        return HuggingFaceVisionBackend(
            spec=VisionBackendSpec(
                name="biovil",
                task="vision_language_embedding",
                model_id=model_id or "microsoft/swin-base-patch4-window7-224",
                labels=labels or list(DEFAULT_CHEST_LABELS),
            ),
            device=device,
        )
    raise ValueError(f"Unsupported medical vision backend: {backend_name}")


def list_recommended_backbones() -> list[dict[str, str]]:
    """Registry metadata surfaced in the UI and docs."""
    return [
        {
            "name": "torchxrayvision",
            "best_for": "Fast chest X-ray baseline with medical pretraining from NIH, CheXpert, PadChest, and MIMIC-style labels.",
            "notes": "Best default upgrade if you want stronger zero-shot chest findings than ImageNet-pretrained DenseNet.",
        },
        {
            "name": "chexnet",
            "best_for": "Supervised chest X-ray classification when you have a CheXpert/MIMIC-CXR checkpoint.",
            "notes": "Good for fine-tuned multilabel inference and Grad-CAM on chest pathologies.",
        },
        {
            "name": "medclip",
            "best_for": "Image-text alignment and retrieval between studies and radiology reports.",
            "notes": "Use when you want multimodal embeddings rather than only classification logits.",
        },
        {
            "name": "biovil",
            "best_for": "Radiology image-text embeddings and cross-modal retrieval pipelines.",
            "notes": "Useful as the image encoder feeding a multimodal RAG stage.",
        },
    ]
