"""
vision/xray_analysis.py
-----------------------
Medical image analysis entry point for chest X-ray, CT, and MRI studies.

This module upgrades the previous ImageNet-pretrained DenseNet flow by exposing
medical backbones such as TorchXRayVision, CheXNet-style checkpoints, and
vision-language encoders used in multimodal retrieval pipelines.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from loguru import logger

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import pydicom
    _DICOM_AVAILABLE = True
except ImportError:
    _DICOM_AVAILABLE = False

from src.explainability.calibration import top_differential
from src.explainability.grad_cam import GradCAMExplainer
from src.vision.medical_models import DEFAULT_CHEST_LABELS, build_backend, list_recommended_backbones


CHEXPERT_LABELS = list(DEFAULT_CHEST_LABELS)

CONDITION_DESCRIPTIONS = {
    "Atelectasis": "Partial or complete collapse of one or more lung lobes.",
    "Cardiomegaly": "Enlargement of the heart silhouette on chest imaging.",
    "Consolidation": "Airspace opacity compatible with fluid, pus, blood, or cells.",
    "Edema": "Interstitial or alveolar fluid accumulation in the lungs.",
    "Effusion": "Pleural fluid accumulation adjacent to the lung.",
    "Emphysema": "Hyperinflation and alveolar destruction causing air trapping.",
    "Fibrosis": "Parenchymal scarring with chronic architectural distortion.",
    "Hernia": "Herniation, often hiatal, visible on chest imaging.",
    "Infiltration": "Patchy opacity suggesting inflammatory or infectious process.",
    "Mass": "Large focal pulmonary lesion that may require malignancy workup.",
    "Nodule": "Small focal pulmonary opacity requiring interval follow-up or workup.",
    "Pleural Thickening": "Pleural surface thickening from prior inflammation or scarring.",
    "Pneumonia": "Pattern consistent with infectious airspace disease.",
    "Pneumothorax": "Air in the pleural space with potential lung collapse.",
    "No Finding": "No radiographically obvious abnormality is detected.",
}


@dataclass
class Finding:
    label: str
    confidence: float
    description: str
    is_abnormal: bool


@dataclass
class ImageAnalysisResult:
    findings: list[Finding] = field(default_factory=list)
    top_finding: Optional[Finding] = None
    image_quality: str = "unknown"
    modality: str = "unknown"
    image_embedding: Optional[np.ndarray] = None
    raw_probabilities: dict[str, float] = field(default_factory=dict)
    source_file: str = ""
    backend_name: str = ""
    calibration_note: str = ""
    differential_diagnosis: list[tuple[str, float]] = field(default_factory=list)
    gradcam_heatmap: Optional[np.ndarray] = None
    gradcam_overlay: Optional[np.ndarray] = None


class MedicalImageAnalyser:
    """
    Analyse a medical image with a configurable medical imaging backend.

    Recommended backends:
    - `torchxrayvision`: best default medical chest X-ray baseline
    - `chexnet`: supervised multilabel classification using medical checkpoints
    - `medclip`: image-text embeddings for multimodal retrieval
    - `biovil`: radiology image-text embeddings for multimodal retrieval
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_classes: int = 14,
        image_size: tuple[int, int] = (224, 224),
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
        backend_name: str = "torchxrayvision",
        model_id: Optional[str] = None,
        enable_gradcam: bool = True,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for image analysis.")

        self.num_classes = num_classes
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.enable_gradcam = enable_gradcam
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = CHEXPERT_LABELS[:num_classes]
        self.backend_name = backend_name

        try:
            self.backend = build_backend(
                backend_name=backend_name,
                device=self.device,
                checkpoint_path=model_path,
                model_id=model_id,
                labels=self.labels,
            )
        except Exception as exc:
            logger.warning(
                "Falling back to CheXNet-style backend because backend '{}' failed: {}",
                backend_name,
                exc,
            )
            self.backend_name = "chexnet"
            self.backend = build_backend(
                backend_name="chexnet",
                device=self.device,
                checkpoint_path=model_path,
                labels=self.labels,
            )

        self.labels = self.backend.spec.labels[:num_classes]
        self._gradcam = None
        target_layer = self.backend.get_target_layer()
        if enable_gradcam and target_layer is not None and hasattr(self.backend.model, "zero_grad"):
            try:
                self._gradcam = GradCAMExplainer(self.backend.model, target_layer=target_layer)
            except Exception as exc:
                logger.warning("Grad-CAM initialisation failed: {}", exc)

    def analyse(self, source: str | Path | bytes | np.ndarray | Image.Image) -> ImageAnalysisResult:
        source_name = str(source) if isinstance(source, (str, Path)) else "<in-memory>"
        pil_image, modality = self._load_image(source)
        input_image = np.array(pil_image.convert("RGB"))
        tensor = self._prepare_tensor(pil_image)

        backend_output = self.backend.predict(tensor)
        raw_probs = self._normalize_probabilities(backend_output.probabilities)
        findings = self._build_findings(raw_probs)
        top = findings[0] if findings else self._default_finding(raw_probs)

        heatmap = None
        overlay = None
        if self._gradcam and top.label in self.labels:
            try:
                class_index = self.labels.index(top.label)
                cam_result = self._gradcam.explain(
                    image_tensor=tensor.clone().detach().requires_grad_(True).to(self.backend.device),
                    class_index=class_index,
                    input_image=input_image,
                    target_label=top.label,
                )
                heatmap = cam_result.heatmap
                overlay = cam_result.overlay
            except Exception as exc:
                logger.warning("Grad-CAM generation failed for {}: {}", top.label, exc)

        result = ImageAnalysisResult(
            findings=findings,
            top_finding=top,
            image_quality=self._assess_quality(self._to_grayscale(pil_image, output_channels=1)),
            modality=modality,
            image_embedding=backend_output.embedding,
            raw_probabilities=raw_probs,
            source_file=source_name,
            backend_name=backend_output.backend_name or self.backend_name,
            calibration_note=backend_output.calibration_note,
            differential_diagnosis=top_differential(raw_probs, top_k=3),
            gradcam_heatmap=heatmap,
            gradcam_overlay=overlay,
        )
        logger.info(
            "Image analysis complete | backend={} | top={} ({:.1%})",
            result.backend_name,
            result.top_finding.label if result.top_finding else "unknown",
            result.top_finding.confidence if result.top_finding else 0.0,
        )
        return result

    def generate_embedding(self, image: Image.Image) -> np.ndarray:
        tensor = self._prepare_tensor(image)
        output = self.backend.predict(tensor)
        if output.embedding is not None:
            return output.embedding
        return np.zeros(384, dtype=np.float32)

    def _prepare_tensor(self, image: Image.Image) -> "torch.Tensor":
        output_channels = 1 if getattr(self.backend.spec, "grayscale", False) else 3
        image = self._to_grayscale(image, output_channels=output_channels)
        if self.backend.preprocess is not None:
            tensor = self.backend.preprocess(image).unsqueeze(0)
        else:
            resized = image.resize(self.image_size)
            if output_channels == 3:
                resized = resized.convert("RGB")
            else:
                resized = resized.convert("L")
            arr = np.asarray(resized).astype(np.float32) / 255.0
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=0)
            else:
                arr = np.transpose(arr, (2, 0, 1))
            tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor.to(self.backend.device)

    def _to_grayscale(self, image: Image.Image, output_channels: int = 1) -> Image.Image:
        grayscale = image.convert("L")
        if output_channels == 1:
            return grayscale
        return grayscale.convert("RGB")

    def _build_findings(self, raw_probabilities: dict[str, float]) -> list[Finding]:
        findings: list[Finding] = []
        for label, prob in raw_probabilities.items():
            if prob >= self.confidence_threshold:
                findings.append(Finding(
                    label=label,
                    confidence=float(prob),
                    description=CONDITION_DESCRIPTIONS.get(label, "Medical finding inferred from image."),
                    is_abnormal=(label != "No Finding"),
                ))
        findings.sort(key=lambda item: item.confidence, reverse=True)
        return findings

    def _normalize_probabilities(self, raw_probabilities: dict[str, float]) -> dict[str, float]:
        aliases = {
            "Pleural Thickening": "Pleural Thickening",
            "Pleural_Thickening": "Pleural Thickening",
            "Lung Opacity": "Consolidation",
            "Support Devices": "No Finding",
        }
        normalized = {label: 0.0 for label in CHEXPERT_LABELS}
        for label, prob in raw_probabilities.items():
            canonical = aliases.get(label, label)
            if canonical in normalized:
                normalized[canonical] = max(normalized[canonical], float(prob))
        if "No Finding" not in normalized:
            normalized["No Finding"] = 0.0
        return normalized

    def _default_finding(self, raw_probabilities: dict[str, float]) -> Finding:
        if raw_probabilities:
            label, probability = max(raw_probabilities.items(), key=lambda item: item[1])
            return Finding(
                label=label,
                confidence=float(probability),
                description=CONDITION_DESCRIPTIONS.get(label, "Medical finding inferred from image."),
                is_abnormal=(label != "No Finding"),
            )
        return Finding(
            label="No Finding",
            confidence=0.0,
            description=CONDITION_DESCRIPTIONS["No Finding"],
            is_abnormal=False,
        )

    def _load_image(self, source: str | Path | bytes | np.ndarray | Image.Image) -> tuple[Image.Image, str]:
        if isinstance(source, Image.Image):
            return source.convert("RGB"), "standard"
        if isinstance(source, np.ndarray):
            return Image.fromarray(source).convert("RGB"), "standard"
        if isinstance(source, bytes):
            try:
                return Image.open(io.BytesIO(source)).convert("RGB"), "standard"
            except Exception:
                if _DICOM_AVAILABLE:
                    ds = pydicom.dcmread(io.BytesIO(source))
                    return self._dicom_to_pil(ds), "DICOM"
                raise
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        if path.suffix.lower() == ".dcm":
            if not _DICOM_AVAILABLE:
                raise RuntimeError("pydicom is required to load DICOM studies.")
            ds = pydicom.dcmread(str(path))
            return self._dicom_to_pil(ds), "DICOM"
        return Image.open(path).convert("RGB"), "standard"

    def _dicom_to_pil(self, ds: "pydicom.Dataset") -> Image.Image:
        pixel_array = ds.pixel_array.astype(np.float32)
        pixel_min, pixel_max = pixel_array.min(), pixel_array.max()
        if pixel_max > pixel_min:
            pixel_array = (pixel_array - pixel_min) / (pixel_max - pixel_min)
        pixel_array = (pixel_array * 255).astype(np.uint8)
        if pixel_array.ndim == 2:
            return Image.fromarray(pixel_array, mode="L").convert("RGB")
        return Image.fromarray(pixel_array).convert("RGB")

    def _assess_quality(self, image: Image.Image) -> str:
        arr = np.array(image.convert("L"), dtype=np.float32)
        std = arr.std()
        if std < 10:
            return "very_low"
        if std < 30:
            return "low"
        if std < 70:
            return "medium"
        return "high"


def analyse_medical_image(
    source: str | Path,
    model_path: Optional[str] = None,
    confidence_threshold: float = 0.5,
    backend_name: str = "torchxrayvision",
) -> ImageAnalysisResult:
    analyser = MedicalImageAnalyser(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        backend_name=backend_name,
    )
    return analyser.analyse(source)


def format_image_report(result: ImageAnalysisResult) -> str:
    lines = [
        "=== Medical Image Analysis ===",
        f"Source   : {result.source_file}",
        f"Modality : {result.modality}",
        f"Quality  : {result.image_quality}",
        f"Backend  : {result.backend_name}",
        "",
        "--- Possible Findings ---",
    ]
    if not result.findings:
        lines.append("  No significant findings above threshold.")
    for finding in result.findings[:5]:
        lines.append(f"  [{finding.confidence:.1%}] {finding.label}")
        lines.append(f"         {finding.description}")
    if result.top_finding:
        lines.extend([
            "",
            "Recommendation:",
            "  Radiologist confirmation recommended.",
        ])
    return "\n".join(lines)


def available_medical_backbones() -> list[dict[str, str]]:
    return list_recommended_backbones()
