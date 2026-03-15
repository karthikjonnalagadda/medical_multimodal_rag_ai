"""
vision/grad_cam.py
------------------
Small convenience wrapper around the project's PyTorch-based medical image
analysis stack for Grad-CAM generation on CPU-friendly backbones.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image

from src.config import VISION_CONFIG


def load_model(
    *,
    confidence_threshold: float | None = None,
    backend_name: str | None = None,
    model_path: str | None = None,
):
    from src.vision.xray_analysis import MedicalImageAnalyser

    return MedicalImageAnalyser(
        confidence_threshold=confidence_threshold or VISION_CONFIG["confidence_threshold"],
        backend_name=backend_name or VISION_CONFIG["backend_name"],
        model_path=model_path or VISION_CONFIG["pretrained_weights"],
        model_id=VISION_CONFIG.get("hf_model_id") or None,
        device="cpu",
        enable_gradcam=True,
    )


def generate_gradcam(
    image: bytes | Image.Image | np.ndarray,
    *,
    analyser: Any | None = None,
    analysis_result: Any | None = None,
) -> dict[str, Any]:
    """
    Generate a Grad-CAM explainability bundle.

    Returns
    -------
    {
      "heatmap": image,
      "top_findings": [{"label": "...", "confidence": 0.61}],
      "original_image": image,
      "raw_heatmap": np.ndarray | None
    }
    """
    analyser = analyser or load_model()
    result = analysis_result or analyser.analyse(image)
    original = _to_image_array(image)
    heatmap = result.gradcam_overlay if getattr(result, "gradcam_overlay", None) is not None else original
    raw_heatmap = getattr(result, "gradcam_heatmap", None)

    if raw_heatmap is not None and heatmap is original:
        heatmap = overlay_heatmap(original, raw_heatmap)

    top_findings = []
    if getattr(result, "differential_diagnosis", None):
        for label, confidence in result.differential_diagnosis[:2]:
            top_findings.append({"label": label, "confidence": round(float(confidence), 2)})
    elif getattr(result, "findings", None):
        for finding in result.findings[:2]:
            top_findings.append({"label": finding.label, "confidence": round(float(finding.confidence), 2)})
    elif getattr(result, "top_finding", None):
        top_findings.append(
            {
                "label": result.top_finding.label,
                "confidence": round(float(result.top_finding.confidence), 2),
            }
        )

    return {
        "heatmap": heatmap,
        "top_findings": top_findings,
        "original_image": original,
        "raw_heatmap": raw_heatmap,
    }


def overlay_heatmap(image: bytes | Image.Image | np.ndarray, heatmap: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    image_array = _to_image_array(image).astype(np.float32)
    if image_array.max() > 1.0:
        image_array = image_array / 255.0

    if image_array.ndim == 2:
        image_array = np.stack([image_array, image_array, image_array], axis=-1)

    if heatmap.ndim == 3:
        heatmap = heatmap[..., 0]
    heatmap = heatmap.astype(np.float32)
    heatmap = heatmap - heatmap.min()
    denom = float(heatmap.max()) if float(heatmap.max()) > 0 else 1.0
    heatmap = heatmap / denom

    color = np.zeros_like(image_array)
    color[..., 0] = heatmap
    color[..., 1] = np.clip(heatmap * 0.45, 0.0, 1.0)
    color[..., 2] = np.clip(heatmap * 0.1, 0.0, 1.0)
    blended = (1 - alpha) * image_array + alpha * color
    return (np.clip(blended, 0.0, 1.0) * 255).astype(np.uint8)


def _to_image_array(image: bytes | Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    if isinstance(image, (bytes, bytearray)):
        try:
            return np.asarray(Image.open(io.BytesIO(image)).convert("RGB"))
        except Exception:
            import pydicom

            ds = pydicom.dcmread(io.BytesIO(image))
            pixels = ds.pixel_array.astype(np.float32)
            pixels = pixels - pixels.min()
            scale = float(pixels.max()) if float(pixels.max()) > 0 else 1.0
            pixels = (pixels / scale * 255).astype(np.uint8)
            if pixels.ndim == 2:
                return np.stack([pixels, pixels, pixels], axis=-1)
            return pixels
    raise TypeError("Unsupported image type for Grad-CAM generation.")
