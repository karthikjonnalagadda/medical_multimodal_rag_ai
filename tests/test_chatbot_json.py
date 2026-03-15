from __future__ import annotations

import json

from src.chat.chatbot import clinical_json_to_structured_sections, normalize_clinical_json_response


def test_normalize_clinical_json_passthrough():
    raw = {
        "diagnosis": "Pneumonia",
        "confidence": "85%",
        "possible_conditions": ["Pneumonia", "Tuberculosis"],
        "explanation": "Likely pneumonia based on fever and cough.",
        "recommended_tests": ["Chest X-ray"],
        "next_steps": ["Seek clinician evaluation"],
    }
    normalized = normalize_clinical_json_response(json.dumps(raw))
    assert normalized["diagnosis"] == "Pneumonia"
    assert normalized["confidence"] == "85%"
    assert normalized["possible_conditions"] == ["Pneumonia", "Tuberculosis"]
    assert normalized["recommended_tests"] == ["Chest X-ray"]
    assert normalized["next_steps"] == ["Seek clinician evaluation"]


def test_normalize_clinical_json_accepts_rag_style_conditions():
    raw = {
        "conditions": [
            {"name": "Pneumonia", "confidence": 0.82},
            {"name": "Bronchitis", "confidence": 0.4},
        ],
        "analysis": "Differential based on symptoms and retrieved context.",
        "tests": ["CBC", "Chest X-ray"],
        "follow_up": ["Monitor oxygen saturation"],
    }
    normalized = normalize_clinical_json_response(json.dumps(raw))
    assert normalized["diagnosis"] == "Pneumonia"
    assert normalized["possible_conditions"] == ["Pneumonia", "Bronchitis"]
    assert normalized["explanation"].startswith("Differential")
    assert normalized["recommended_tests"] == ["CBC", "Chest X-ray"]
    assert normalized["next_steps"] == ["Monitor oxygen saturation"]


def test_normalize_clinical_json_extracts_embedded_json():
    response = """Here is the JSON:

```json
{"diagnosis":"Hypertension","confidence":"medium","possible_conditions":["Hypertension"],"explanation":"BP elevated.","recommended_tests":[],"next_steps":["Repeat BP readings"]}
```"""
    normalized = normalize_clinical_json_response(response)
    assert normalized["diagnosis"] == "Hypertension"
    assert normalized["confidence"] == "medium"
    assert normalized["possible_conditions"] == ["Hypertension"]
    assert normalized["next_steps"] == ["Repeat BP readings"]


def test_normalize_clinical_json_free_text_fallback():
    response = "I cannot be sure without vitals and labs. Please consult a clinician."
    normalized = normalize_clinical_json_response(response)
    assert normalized["diagnosis"] == ""
    assert normalized["possible_conditions"] == []
    assert "consult a clinician" in normalized["explanation"].lower()


def test_clinical_json_to_structured_sections_includes_schema():
    clinical = normalize_clinical_json_response(
        json.dumps(
            {
                "diagnosis": "Asthma",
                "confidence": "high",
                "possible_conditions": ["Asthma", "COPD"],
                "explanation": "Wheezing and reversible airflow symptoms are consistent with asthma.",
                "recommended_tests": ["Spirometry"],
                "next_steps": ["Use inhaler as prescribed"],
            }
        )
    )
    structured = clinical_json_to_structured_sections(clinical)
    assert structured["clinical_json"]["diagnosis"] == "Asthma"
    assert structured["sections"]

