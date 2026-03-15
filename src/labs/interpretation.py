"""
labs/interpretation.py
---------------------
Centralized, rule-based lab interpretation.

Goals:
- Robust numeric parsing (handles thousands separators like "17,000")
- Consistent classification: Low / Normal / High
- Targeted clinical interpretations for common labs (non-diagnostic, uncertainty-aware)

This module does not attempt full unit conversion across all lab systems; it supports a
small set of practical normalizations commonly seen in OCR output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# Reference ranges used by OCR parsing. These are generic adult reference ranges.
# Some tests (e.g., Hematocrit) are sex/age dependent; we keep a default per project requirements.
REFERENCE_RANGES: dict[str, dict] = {
    "hemoglobin": {"min": 12.0, "max": 17.5, "unit": "g/dL"},
    "hgb": {"min": 12.0, "max": 17.5, "unit": "g/dL"},
    # Requested WBC reference: 4000–11000 /µL
    "wbc": {"min": 4000, "max": 11000, "unit": "cells/uL"},
    "white blood cell": {"min": 4000, "max": 11000, "unit": "cells/uL"},
    "white blood cell count": {"min": 4000, "max": 11000, "unit": "cells/uL"},
    "platelets": {"min": 150000, "max": 400000, "unit": "cells/uL"},
    "plt": {"min": 150000, "max": 400000, "unit": "cells/uL"},
    "rbc": {"min": 4.2, "max": 5.9, "unit": "million/uL"},
    "glucose": {"min": 70, "max": 100, "unit": "mg/dL"},
    "creatinine": {"min": 0.6, "max": 1.2, "unit": "mg/dL"},
    "sodium": {"min": 136, "max": 145, "unit": "mEq/L"},
    "potassium": {"min": 3.5, "max": 5.0, "unit": "mEq/L"},
    "bun": {"min": 7, "max": 20, "unit": "mg/dL"},
    "cholesterol": {"min": 0, "max": 200, "unit": "mg/dL"},
    # HbA1c uses special classification (normal/prediabetes/diabetes). Range here is used for display only.
    "hba1c": {"min": 0, "max": 5.7, "unit": "%"},
    "tsh": {"min": 0.4, "max": 4.0, "unit": "mIU/L"},
    # Requested Hematocrit (Male): 40–50 %
    "hematocrit": {"min": 40.0, "max": 50.0, "unit": "%"},
    "hct": {"min": 40.0, "max": 50.0, "unit": "%"},
}


@dataclass(frozen=True)
class InterpretedLab:
    test: str
    value: float
    unit: str
    status: str  # "Low" | "Normal" | "High"
    interpretation: str = ""


_THOUSANDS_SEP_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")
# Includes comma or dot decimal separators and comma thousands separators (best-effort).
_NUMERIC_RE = re.compile(r"[-+]?(?:\d+(?:[.,]\d+)?|\.\d+)(?:,\d{3})*")


def parse_numeric_value(raw: str) -> Optional[float]:
    """
    Extract and parse a numeric value from an OCR token.

    Handles:
    - "17,000" -> 17000
    - "8.2%" -> 8.2
    - "  2.1 mg/dL" -> 2.1
    - "8,2" (decimal comma) -> 8.2 (best-effort)
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    # Keep only the first number-looking token to avoid swallowing dates etc.
    compact = text.replace(" ", "")
    match = _NUMERIC_RE.search(compact)
    if not match:
        return None
    token = match.group(0)

    # If we have comma as thousands separator, remove it.
    token = _THOUSANDS_SEP_RE.sub("", token)

    # If decimal comma is likely (e.g., "8,2"), convert to dot.
    if "," in token and "." not in token:
        token = token.replace(",", ".")

    try:
        return float(token)
    except Exception:
        return None


def normalize_unit(unit: str) -> str:
    """
    Normalize common OCR unit variants to a canonical form.
    """
    if not unit:
        return ""
    u = unit.strip()
    # Normalize micro sign and common variants.
    u = u.replace("µ", "u").replace("μ", "u")
    u = u.replace(" ", "")
    u_lower = u.lower()

    # Canonicalize a few frequent forms.
    if u_lower in {"cells/ul", "cells/u l"}:
        return "cells/uL"
    if u_lower in {"cells/uu", "cells/u"}:
        return "cells/uL"
    if u_lower in {"mg/dl"}:
        return "mg/dL"
    if u_lower in {"g/dl"}:
        return "g/dL"
    if u_lower in {"meq/l"}:
        return "mEq/L"
    if u_lower in {"miu/l"}:
        return "mIU/L"
    if u_lower == "%":
        return "%"
    return u


def classify_by_reference(value: float, reference_low: float, reference_high: float) -> str:
    """
    Consistent rule-based classifier for numeric values.
    """
    if value < reference_low:
        return "Low"
    if value > reference_high:
        return "High"
    return "Normal"


def interpret_metric(
    *,
    name: str,
    value: float,
    unit: str,
    reference_low: Optional[float] = None,
    reference_high: Optional[float] = None,
    reference_unit: str = "",
) -> InterpretedLab:
    """
    Interpret a lab metric using generic ref-range rules + a few domain-specific overrides.
    """
    unit_norm = normalize_unit(unit)
    test = (name or "").strip()
    test_key = " ".join(test.lower().split())

    # If no explicit reference range was provided, use the project's default range for this test.
    if reference_low is None or reference_high is None:
        ref = _find_reference_for_name(test_key)
        if ref is not None:
            reference_low = float(ref["min"])
            reference_high = float(ref["max"])
            reference_unit = str(ref.get("unit", "")) or reference_unit

    # HbA1c: special classification ranges.
    if "hba1c" in test_key or "glycated hemoglobin" in test_key:
        if value < 5.7:
            status = "Normal"
            interpretation = "HbA1c in the normal range."
        elif value < 6.5:
            status = "High"
            interpretation = "HbA1c in the prediabetes range (5.7–6.4%)."
        else:
            status = "High"
            interpretation = "HbA1c in the diabetes range (>= 6.5%)."
        return InterpretedLab(test=test, value=value, unit=unit_norm or "%", status=status, interpretation=interpretation)

    # WBC: ensure correct low/high cutoffs.
    if test_key in {"wbc", "wbc count", "white blood cell", "white blood cell count"} or "white blood cell" in test_key:
        low, high = 4000.0, 11000.0
        status = classify_by_reference(value, low, high)
        if status == "High":
            interpretation = "Leukocytosis (may suggest infection or inflammation; interpret with clinical context)."
        elif status == "Low":
            interpretation = "Leukopenia (may be seen with viral illness, marrow suppression, or medications; interpret with context)."
        else:
            interpretation = ""
        return InterpretedLab(test=test, value=value, unit=unit_norm or "cells/uL", status=status, interpretation=interpretation)

    # Hematocrit: requested male reference range 40–50%.
    if test_key in {"hematocrit", "hct"}:
        low, high = 40.0, 50.0
        status = classify_by_reference(value, low, high)
        interpretation = ""
        if status == "Low":
            interpretation = "Low hematocrit (may indicate anemia or bleeding; interpret with hemoglobin/RBC indices)."
        elif status == "High":
            interpretation = "High hematocrit (may indicate hemoconcentration or polycythemia; interpret with hydration status)."
        return InterpretedLab(test=test, value=value, unit=unit_norm or "%", status=status, interpretation=interpretation)

    # Generic range-based interpretation if we have a reference range.
    interpretation = ""
    status = "Normal"
    if reference_low is not None and reference_high is not None:
        status = classify_by_reference(value, float(reference_low), float(reference_high))

    # Add targeted interpretations for a few common tests (non-diagnostic phrasing).
    if status != "Normal":
        if "hemoglobin" in test_key or test_key == "hgb":
            if value < 7.0:
                interpretation = "Severely low hemoglobin (severe anemia range); requires prompt clinical evaluation."
            elif value < 10.0:
                interpretation = "Low hemoglobin (anemia); causes can include iron deficiency, chronic disease, or blood loss and need evaluation."
            else:
                interpretation = "Mildly low hemoglobin (possible anemia); interpret with RBC indices and clinical context."
        elif "platelet" in test_key or test_key == "plt":
            if value < 50000:
                interpretation = "Marked thrombocytopenia (increased bleeding risk); clinical correlation recommended."
            elif value < 100000:
                interpretation = "Thrombocytopenia; interpret with medications, infection, and other CBC indices."
            else:
                interpretation = "Mild thrombocytopenia; clinical correlation recommended."
        elif "glucose" in test_key:
            interpretation = "Elevated glucose (hyperglycemia); interpret based on fasting status and repeat testing if needed."
        elif "creatinine" in test_key:
            interpretation = "Elevated creatinine (may indicate reduced kidney function or dehydration); interpret with eGFR and trend."

    return InterpretedLab(test=test, value=value, unit=unit_norm, status=status, interpretation=interpretation)


def _find_reference_for_name(name_lower: str) -> Optional[dict]:
    """
    Best-effort reference lookup similar to OCR parsing.
    """
    for key, ref in REFERENCE_RANGES.items():
        key_norm = " ".join(key.lower().split())
        if key_norm in name_lower or name_lower in key_norm:
            return ref
    return None
