from __future__ import annotations

import pytest


def test_rule_based_classification_examples():
    from src.labs.interpretation import interpret_metric

    cases = [
        ("Hemoglobin", 8.4, "g/dL", "Low"),
        ("WBC", 17000, "cells/uL", "High"),
        ("Platelets", 90000, "cells/uL", "Low"),
        ("Glucose", 186, "mg/dL", "High"),
        ("HbA1c", 8.2, "%", "High"),
        ("Creatinine", 2.1, "mg/dL", "High"),
        ("Hematocrit", 28, "%", "Low"),
    ]

    for name, value, unit, expected in cases:
        interpreted = interpret_metric(name=name, value=float(value), unit=unit)
        assert interpreted.status == expected, f"{name}={value} expected {expected}, got {interpreted.status}"


def test_wbc_high_interpretation_text():
    from src.labs.interpretation import interpret_metric

    interpreted = interpret_metric(name="WBC", value=17000, unit="cells/uL")
    assert interpreted.status == "High"
    assert "leukocyt" in interpreted.interpretation.lower()


def test_hba1c_diabetes_range():
    from src.labs.interpretation import interpret_metric

    interpreted = interpret_metric(name="HbA1c", value=8.2, unit="%")
    assert interpreted.status == "High"
    assert "diabetes" in interpreted.interpretation.lower()


def test_ocr_metric_parsing_handles_commas_and_assigns_status():
    from src.ocr.extract_lab_text import MedicalOCR

    text = (
        "Hemoglobin: 8.4 g/dL\n"
        "WBC Count: 17,000 cells/uL\n"
        "Platelets: 90,000\n"
        "Glucose: 186 mg/dL\n"
        "HbA1c: 8.2 %\n"
        "Creatinine: 2.1 mg/dL\n"
        "Hematocrit: 28 %\n"
    )
    ocr = MedicalOCR.__new__(MedicalOCR)
    metrics = ocr._parse_metrics(text)
    by_name = {m.name: m for m in metrics}

    # Ensure numeric parsing is correct (no comma truncation).
    wbc = next(m for m in metrics if "wbc" in m.name or "white blood cell" in m.name)
    assert wbc.value == 17000
    assert wbc.status == "High"

    hb = next(m for m in metrics if "hemoglobin" in m.name or m.name == "hgb")
    assert hb.status == "Low"

    hct = next(m for m in metrics if "hematocrit" in m.name or m.name == "hct")
    assert hct.status == "Low"

