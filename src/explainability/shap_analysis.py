"""
explainability/shap_analysis.py
--------------------------------
Model explainability for medical AI predictions.

Provides:
  * SHAP-based feature importance for tabular lab data
  * LIME-based text explanation for RAG responses
  * Grad-CAM heatmap generation for medical images
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from loguru import logger

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

try:
    from lime.lime_text import LimeTextExplainer
    _LIME_AVAILABLE = True
except ImportError:
    _LIME_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class FeatureImportance:
    feature_name: str
    importance_score: float
    direction: str          # "positive" | "negative" | "neutral"
    value: Optional[float] = None


@dataclass
class ExplainabilityReport:
    method: str
    top_features: list[FeatureImportance] = field(default_factory=list)
    explanation_text: str = ""
    heatmap: Optional[np.ndarray] = None   # For image explanations (H × W)


# ── Lab Report SHAP Explainer ──────────────────────────────────────────────────

class LabReportExplainer:
    """
    SHAP-based explainer for tabular lab report data.

    Explains which lab values contribute most to a model's prediction.

    Usage
    -----
    >>> explainer = LabReportExplainer(model_fn, feature_names)
    >>> report = explainer.explain(feature_values)
    """

    def __init__(
        self,
        model_fn,                  # Callable: np.ndarray → np.ndarray (probabilities)
        feature_names: list[str],
        background_data: Optional[np.ndarray] = None,
    ):
        self.model_fn = model_fn
        self.feature_names = feature_names
        self._explainer = None

        if not _SHAP_AVAILABLE:
            logger.warning("SHAP not installed – using gradient-free approximation.")
            return

        try:
            if background_data is not None:
                self._explainer = shap.KernelExplainer(model_fn, background_data)
            else:
                # Use zero baseline
                baseline = np.zeros((1, len(feature_names)))
                self._explainer = shap.KernelExplainer(model_fn, baseline)
            logger.info("SHAP KernelExplainer initialised.")
        except Exception as e:
            logger.error(f"SHAP init failed: {e}")

    def explain(
        self, feature_values: np.ndarray, target_class: int = 0
    ) -> ExplainabilityReport:
        """
        Explain a single prediction.

        Parameters
        ----------
        feature_values : 1-D array of lab metric values
        target_class   : class index to explain (e.g., 0 = Pneumonia)
        """
        if self._explainer is not None:
            return self._shap_explain(feature_values, target_class)
        return self._fallback_explain(feature_values)

    def _shap_explain(
        self, feature_values: np.ndarray, target_class: int
    ) -> ExplainabilityReport:
        shap_values = self._explainer.shap_values(
            feature_values.reshape(1, -1), nsamples=100
        )
        if isinstance(shap_values, list):
            values = shap_values[target_class][0]
        else:
            values = shap_values[0]

        features = []
        for name, val, imp in zip(self.feature_names, feature_values, values):
            features.append(FeatureImportance(
                feature_name=name,
                importance_score=float(abs(imp)),
                direction="positive" if imp > 0 else "negative" if imp < 0 else "neutral",
                value=float(val),
            ))
        features.sort(key=lambda f: f.importance_score, reverse=True)

        top_pos = [f for f in features if f.direction == "positive"][:3]
        top_neg = [f for f in features if f.direction == "negative"][:3]

        lines = ["Top factors INCREASING probability:"]
        for f in top_pos:
            lines.append(f"  + {f.feature_name} = {f.value:.2f}  (impact: +{f.importance_score:.3f})")
        lines.append("Top factors DECREASING probability:")
        for f in top_neg:
            lines.append(f"  - {f.feature_name} = {f.value:.2f}  (impact: -{f.importance_score:.3f})")

        return ExplainabilityReport(
            method="SHAP",
            top_features=features[:10],
            explanation_text="\n".join(lines),
        )

    def _fallback_explain(self, feature_values: np.ndarray) -> ExplainabilityReport:
        """Perturbation-based approximation when SHAP is unavailable."""
        baseline = np.zeros_like(feature_values)
        base_pred = self.model_fn(baseline.reshape(1, -1))[0]
        importances: list[FeatureImportance] = []

        for i, (name, val) in enumerate(zip(self.feature_names, feature_values)):
            perturbed = baseline.copy()
            perturbed[i] = val
            pred = self.model_fn(perturbed.reshape(1, -1))[0]
            delta = float(np.mean(pred) - np.mean(base_pred))
            importances.append(FeatureImportance(
                feature_name=name,
                importance_score=abs(delta),
                direction="positive" if delta > 0 else "negative",
                value=float(val),
            ))
        importances.sort(key=lambda f: f.importance_score, reverse=True)
        top_str = ", ".join(
            f"{f.feature_name} ({f.direction})" for f in importances[:5]
        )
        return ExplainabilityReport(
            method="perturbation",
            top_features=importances[:10],
            explanation_text=f"Most important features: {top_str}",
        )


# ── Text LIME Explainer ────────────────────────────────────────────────────────

class TextLIMEExplainer:
    """
    LIME-based explanation for free-text medical queries.

    Highlights which words in the query most influenced the RAG output.

    Usage
    -----
    >>> explainer = TextLIMEExplainer(classify_fn)
    >>> report = explainer.explain("fever cough elevated WBC lung opacity")
    """

    def __init__(self, classify_fn, class_names: list[str] | None = None):
        self.classify_fn = classify_fn
        self.class_names = class_names or ["No Disease", "Disease Present"]
        self._explainer = None
        if _LIME_AVAILABLE:
            self._explainer = LimeTextExplainer(class_names=self.class_names)
        else:
            logger.warning("LIME not installed – using keyword-weight fallback.")

    def explain(self, text: str, num_features: int = 10) -> ExplainabilityReport:
        if self._explainer is not None:
            return self._lime_explain(text, num_features)
        return self._keyword_explain(text)

    def _lime_explain(self, text: str, num_features: int) -> ExplainabilityReport:
        try:
            exp = self._explainer.explain_instance(
                text, self.classify_fn, num_features=num_features, num_samples=500
            )
            features = []
            for word, weight in exp.as_list():
                features.append(FeatureImportance(
                    feature_name=word,
                    importance_score=abs(weight),
                    direction="positive" if weight > 0 else "negative",
                ))
            features.sort(key=lambda f: f.importance_score, reverse=True)
            return ExplainabilityReport(
                method="LIME",
                top_features=features,
                explanation_text=self._features_to_text(features),
            )
        except Exception as e:
            logger.error(f"LIME explain error: {e}")
            return self._keyword_explain(text)

    def _keyword_explain(self, text: str) -> ExplainabilityReport:
        """Simple keyword weight approach as fallback."""
        medical_weights = {
            "fever": 0.8, "cough": 0.7, "pneumonia": 0.95,
            "opacity": 0.9, "wbc": 0.85, "hemoglobin": 0.75,
            "effusion": 0.85, "consolidation": 0.9, "dyspnea": 0.8,
        }
        words = text.lower().split()
        features = []
        for word in set(words):
            score = medical_weights.get(word, 0.1)
            features.append(FeatureImportance(
                feature_name=word,
                importance_score=score,
                direction="positive",
            ))
        features.sort(key=lambda f: f.importance_score, reverse=True)
        return ExplainabilityReport(
            method="keyword_weight",
            top_features=features[:10],
            explanation_text=self._features_to_text(features[:5]),
        )

    @staticmethod
    def _features_to_text(features: list[FeatureImportance]) -> str:
        parts = [f"'{f.feature_name}' (weight={f.importance_score:.3f})" for f in features[:5]]
        return "Key diagnostic terms: " + ", ".join(parts)


# ── Grad-CAM for medical images ────────────────────────────────────────────────

class GradCAMExplainer:
    """
    Gradient-weighted Class Activation Mapping for CNN medical image models.

    Produces a heatmap highlighting which image regions influence the prediction.

    Usage
    -----
    >>> explainer = GradCAMExplainer(model, target_layer="features.denseblock4")
    >>> heatmap = explainer.generate_heatmap(image_tensor, class_idx=6)  # Pneumonia
    """

    def __init__(self, model, target_layer_name: str = "features"):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for Grad-CAM.")
        self.model = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Find target layer
        self._target_layer = None
        for name, module in model.named_modules():
            if target_layer_name in name:
                self._target_layer = module

        if self._target_layer is None:
            logger.warning(f"Target layer '{target_layer_name}' not found.")
        else:
            self._target_layer.register_forward_hook(self._save_activations)
            self._target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate_heatmap(
        self, image_tensor: "torch.Tensor", class_idx: int
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap.

        Parameters
        ----------
        image_tensor : (1, C, H, W) tensor
        class_idx    : target class index

        Returns
        -------
        np.ndarray heatmap of shape (H, W), values in [0, 1]
        """
        self.model.zero_grad()
        output = self.model(image_tensor)
        score = output[0, class_idx].clone()
        score.backward()

        if self._activations is None or self._gradients is None:
            logger.warning("No activations/gradients captured.")
            return np.zeros((224, 224))

        pooled_grads = self._gradients.mean(dim=[0, 2, 3])
        activations = self._activations[0]

        for i, w in enumerate(pooled_grads):
            activations[i] *= w

        heatmap = activations.mean(dim=0).numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        # Resize to image dimensions (rough upscaling)
        try:
            import cv2
            heatmap_resized = cv2.resize(heatmap, (224, 224))
        except ImportError:
            from PIL import Image as PILImage
            heatmap_resized = np.array(
                PILImage.fromarray((heatmap * 255).astype(np.uint8)).resize((224, 224))
            ) / 255.0

        return heatmap_resized.astype(np.float32)
