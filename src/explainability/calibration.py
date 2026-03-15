"""
explainability/calibration.py
-----------------------------
Confidence calibration helpers for multilabel medical classifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@dataclass
class CalibrationSummary:
    temperature: float
    method: str
    note: str


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling for logits-based confidence calibration."""

    def __init__(self, temperature: float = 1.0) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for calibration.")
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * float(temperature))

    def forward(self, logits: "torch.Tensor") -> "torch.Tensor":
        return logits / self.temperature.clamp(min=1e-3)

    def calibrate(self, logits: np.ndarray, labels: np.ndarray, max_iter: int = 50) -> CalibrationSummary:
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.1, max_iter=max_iter)
        criterion = nn.BCEWithLogitsLoss()

        def _closure():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits_tensor), labels_tensor)
            loss.backward()
            return loss

        optimizer.step(_closure)
        return CalibrationSummary(
            temperature=float(self.temperature.detach().item()),
            method="temperature_scaling",
            note="Fit on a held-out validation split before deployment.",
        )

    def apply(self, logits: np.ndarray) -> np.ndarray:
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        calibrated = torch.sigmoid(self.forward(logits_tensor)).detach().cpu().numpy()
        return calibrated


def top_differential(probabilities: dict[str, float], top_k: int = 3) -> list[tuple[str, float]]:
    """Return the top-k differential diagnoses sorted by descending probability."""
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    return ranked[:top_k]
