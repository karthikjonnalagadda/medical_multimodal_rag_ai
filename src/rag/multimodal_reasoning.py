"""
rag/multimodal_reasoning.py
---------------------------
Multimodal prompt assembly and structured reasoning for the medical assistant.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from typing import Any

from loguru import logger

from src.labs.interpretation import interpret_metric, parse_numeric_value


MULTIMODAL_SYSTEM_PROMPT = """You are a clinical multimodal reasoning assistant.
Combine symptoms, clinical notes, image findings, and retrieved medical evidence.
Use uncertainty-aware language (e.g., "may", "could", "is consistent with") and avoid definitive etiologies.
Return only valid JSON with this schema:
{
  "differential_diagnosis": [
    {"disease": "...", "confidence": 0.82}
  ],
  "explanation": "...",
  "recommended_tests": ["...", "..."]
}"""


def build_multimodal_prompt(
    symptoms: list[str] | None,
    clinical_notes: str,
    vision_findings: list[str] | str | None,
    retrieved_docs: list[Any] | None,
) -> str:
    vision_text = _vision_text(vision_findings)
    evidence_text = _format_retrieved_docs(retrieved_docs or [])
    symptoms_text = ", ".join(symptoms or []) or "None provided"
    notes_text = clinical_notes.strip() or "None provided"

    return f"""Symptoms:
{symptoms_text}

Clinical notes:
{notes_text}

X-ray findings:
{vision_text}

Medical knowledge:
{evidence_text}"""


def reason_multimodal_case(
    *,
    symptoms: list[str] | None = None,
    clinical_notes: str = "",
    vision_findings: list[str] | str | None = None,
    retrieved_docs: list[Any] | None = None,
    llm: Any | None = None,
) -> dict[str, Any]:
    """
    Return a structured differential diagnosis bundle.

    If an LLM client is supplied it will be used first. On failure or invalid JSON,
    the module falls back to a lightweight heuristic reasoner so CPU-only flows
    stay usable offline.
    """
    prompt = build_multimodal_prompt(symptoms, clinical_notes, vision_findings, retrieved_docs)

    if llm is not None and hasattr(llm, "generate"):
        try:
            response = llm.generate(MULTIMODAL_SYSTEM_PROMPT, prompt)
            parsed = _parse_reasoning_response(response)
            if parsed["differential_diagnosis"]:
                return parsed
        except Exception as exc:
            logger.warning("LLM multimodal reasoning failed, using heuristic fallback: {}", exc)

    return _heuristic_reasoning(
        symptoms=symptoms or [],
        clinical_notes=clinical_notes,
        vision_findings=vision_findings,
        retrieved_docs=retrieved_docs or [],
    )


def _parse_reasoning_response(response: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in multimodal reasoning response.")

    payload = json.loads(match.group(0))
    if "differential_diagnosis" in payload:
        differential = payload.get("differential_diagnosis", [])
    else:
        differential = [
            {
                "disease": item.get("name", "Unknown"),
                "confidence": _normalize_confidence(item.get("confidence", 0.5)),
            }
            for item in payload.get("conditions", [])
        ]

    return {
        "differential_diagnosis": [
            {
                "disease": item.get("disease", "Unknown"),
                "confidence": _normalize_confidence(item.get("confidence", 0.0)),
            }
            for item in differential
            if item.get("disease")
        ][:3],
        "explanation": payload.get("explanation", ""),
        "recommended_tests": payload.get("recommended_tests", _recommend_tests("", "", "")),
    }


def _heuristic_reasoning(
    *,
    symptoms: list[str],
    clinical_notes: str,
    vision_findings: list[str] | str | None,
    retrieved_docs: list[Any],
) -> dict[str, Any]:
    combined_text = " ".join(
        [
            ", ".join(symptoms),
            clinical_notes,
            _vision_text(vision_findings),
            " ".join(_doc_text(doc) for doc in retrieved_docs[:4]),
        ]
    ).lower()

    # Evidence weighting (requested):
    # - Lab evidence: 40%
    # - Imaging evidence: 40%
    # - Symptoms/context: 20% (symptoms + retrieved-doc hints)
    scores: dict[str, float] = defaultdict(float)
    evidence_sources: list[str] = []

    imaging_text = _vision_text(vision_findings).lower()
    symptoms_text = ", ".join(symptoms).lower()

    # Extract a few lab signals from the combined text (best-effort from OCR/raw notes).
    lab_signals = _extract_lab_signals(combined_text)

    # Aggregate doc hints for context weighting.
    doc_disease_counts: dict[str, int] = defaultdict(int)
    for doc in retrieved_docs[:8]:
        evidence_sources.append(_doc_source(doc))
        disease = _doc_disease(doc)
        if disease:
            doc_disease_counts[disease] += 1

    disease_profiles: dict[str, dict[str, tuple[str, ...]]] = {
        "Pneumonia": {
            "labs": ("wbc_high",),
            "imaging": ("consolidation", "opacity", "infiltration", "infiltrate"),
            "symptoms": ("fever", "cough", "shortness of breath", "sputum", "productive"),
        },
        "Tuberculosis": {
            "labs": (),
            "imaging": ("upper lobe", "cavitation", "cavity"),
            "symptoms": ("night sweats", "weight loss", "hemoptysis", "chronic cough"),
        },
        "Pulmonary Edema": {
            "labs": (),
            "imaging": ("pulmonary edema", "cardiomegaly", "interstitial", "effusion"),
            "symptoms": ("shortness of breath", "orthopnea", "edema"),
        },
        "Pleural Effusion": {
            "labs": (),
            "imaging": ("effusion", "pleural"),
            "symptoms": ("shortness of breath", "chest pain", "pleuritic"),
        },
        "COPD Exacerbation": {
            "labs": (),
            "imaging": ("hyperinflation", "emphysema"),
            "symptoms": ("wheezing", "copd", "chronic cough", "dyspnea"),
        },
    }

    candidate_diseases = set(disease_profiles.keys()) | set(doc_disease_counts.keys())

    for disease in candidate_diseases:
        profile = disease_profiles.get(disease, {})
        lab_tokens = profile.get("labs", ())
        img_tokens = profile.get("imaging", ())
        sym_tokens = profile.get("symptoms", ())

        lab_score = _token_score(lab_tokens, lab_signals)
        imaging_score = _text_overlap_score(img_tokens, imaging_text)

        doc_hint = min(1.0, doc_disease_counts.get(disease, 0) / 2.0)  # 0, 0.5, 1.0...
        symptom_hint = _text_overlap_score(sym_tokens, symptoms_text)
        context_score = max(doc_hint, symptom_hint)

        scores[disease] = 0.4 * lab_score + 0.4 * imaging_score + 0.2 * context_score

    if not scores:
        seed_label = _seed_disease_from_docs(retrieved_docs) or "General Respiratory Infection"
        scores[seed_label] = 0.3

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:3]
    confidences = _normalize_scores([score for _, score in ranked])
    differential = [
        {"disease": disease, "confidence": confidence}
        for (disease, _), confidence in zip(ranked, confidences)
    ]

    explanation = _build_explanation(
        symptoms=symptoms,
        clinical_notes=clinical_notes,
        vision_findings=_vision_text(vision_findings),
        evidence_sources=[source for source in evidence_sources if source],
        differential=differential,
    )
    recommended_tests = _recommend_tests(
        ", ".join(symptoms),
        clinical_notes,
        _vision_text(vision_findings),
    )
    return {
        "differential_diagnosis": differential,
        "explanation": explanation,
        "recommended_tests": recommended_tests,
    }


def _format_retrieved_docs(retrieved_docs: list[Any]) -> str:
    if not retrieved_docs:
        return "No retrieved evidence."

    formatted = []
    for index, doc in enumerate(retrieved_docs[:4], start=1):
        metadata = _doc_metadata(doc)
        disease = metadata.get("disease") or _doc_disease(doc)
        symptoms = ", ".join(_doc_symptoms(doc)) or "n/a"
        source = _doc_source(doc) or metadata.get("source", "unknown")
        formatted.append(
            f"[{index}] Source={source}; Disease={disease or 'unknown'}; "
            f"Symptoms={symptoms}; Evidence={_doc_text(doc)[:500]}"
        )
    return "\n\n".join(formatted)


def _recommend_tests(symptoms: str, clinical_notes: str, vision_text: str) -> list[str]:
    combined = f"{symptoms} {clinical_notes} {vision_text}".lower()
    tests = ["Complete blood count", "Pulse oximetry"]
    if any(token in combined for token in ("pneumonia", "fever", "cough", "infiltration", "consolidation")):
        tests.extend(["Chest X-ray review", "CRP or ESR", "Sputum culture"])
    if any(token in combined for token in ("edema", "cardiomegaly", "effusion")):
        tests.extend(["BNP", "Echocardiogram"])
    if any(token in combined for token in ("tuberculosis", "hemoptysis", "night sweats")):
        tests.extend(["AFB smear or GeneXpert", "Chest CT if indicated"])

    deduped = []
    for item in tests:
        if item not in deduped:
            deduped.append(item)
    return deduped[:5]


def _build_explanation(
    *,
    symptoms: list[str],
    clinical_notes: str,
    vision_findings: str,
    evidence_sources: list[str],
    differential: list[dict[str, Any]],
) -> str:
    top = differential[0]["disease"] if differential else "the leading diagnosis"
    sources = ", ".join(dict.fromkeys(evidence_sources[:3])) or "retrieved medical references"
    symptom_text = ", ".join(symptoms[:4]) or "reported symptoms"
    note_hint = " Clinical notes were also incorporated." if clinical_notes.strip() else ""
    vision_hint = f" Imaging findings ({vision_findings}) support the ranking." if vision_findings and vision_findings != "None provided" else ""
    return (
        f"The leading differential is {top}, which may be consistent with symptoms such as {symptom_text}, "
        f"retrieved evidence from {sources}, and the available multimodal context."
        f"{vision_hint}{note_hint}"
    )


def _text_overlap_score(tokens: tuple[str, ...], haystack: str) -> float:
    if not tokens:
        return 0.0
    hits = sum(1 for token in tokens if token and token in haystack)
    return min(1.0, hits / max(1, len(tokens)))


def _token_score(tokens: tuple[str, ...], signals: set[str]) -> float:
    if not tokens:
        return 0.0
    hits = sum(1 for token in tokens if token in signals)
    return min(1.0, hits / max(1, len(tokens)))


def _extract_lab_signals(text: str) -> set[str]:
    """
    Extract a small set of boolean lab signals from free text for heuristic scoring.

    This is intentionally conservative: if we cannot parse a value reliably, we emit no signal.
    """
    lowered = text.lower()
    signals: set[str] = set()

    # WBC
    wbc_match = re.search(r"(?:\bwbc\b|white blood cell(?: count)?)\D{0,12}(\d[\d,\.]*)", lowered)
    if wbc_match:
        wbc_value = parse_numeric_value(wbc_match.group(1))
        if wbc_value is not None:
            interp = interpret_metric(name="wbc", value=wbc_value, unit="cells/uL")
            if interp.status == "High":
                signals.add("wbc_high")
            elif interp.status == "Low":
                signals.add("wbc_low")

    return signals


def _normalize_confidence(value: Any) -> float:
    try:
        numeric = float(value)
    except Exception:
        return 0.0
    if numeric > 1.0:
        numeric /= 100.0
    return max(0.0, min(numeric, 1.0))


def _normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    exp_values = [math.exp(min(value, 4.0)) for value in values]
    total = sum(exp_values) or 1.0
    probabilities = [value / total for value in exp_values]
    return [round(probability, 2) for probability in probabilities]


def _vision_text(vision_findings: list[str] | str | None) -> str:
    if isinstance(vision_findings, str):
        return vision_findings.strip() or "None provided"
    if not vision_findings:
        return "None provided"
    return "; ".join(str(item) for item in vision_findings if str(item).strip()) or "None provided"


def _doc_metadata(doc: Any) -> dict[str, Any]:
    if hasattr(doc, "metadata"):
        return getattr(doc, "metadata") or {}
    if isinstance(doc, dict):
        return doc.get("metadata", {}) or {}
    return {}


def _doc_text(doc: Any) -> str:
    if hasattr(doc, "text"):
        return getattr(doc, "text") or ""
    if isinstance(doc, dict):
        return doc.get("text", "") or ""
    return str(doc)


def _doc_source(doc: Any) -> str:
    if hasattr(doc, "source"):
        return getattr(doc, "source") or ""
    if isinstance(doc, dict):
        return doc.get("source", "") or _doc_metadata(doc).get("source", "")
    return ""


def _doc_disease(doc: Any) -> str:
    metadata = _doc_metadata(doc)
    disease = metadata.get("disease", "")
    if disease:
        return str(disease)

    text = _doc_text(doc)
    match = re.search(
        r"\b(Pneumonia|Tuberculosis|Pulmonary Edema|Pleural Effusion|COPD|Atelectasis)\b",
        text,
        re.IGNORECASE,
    )
    return match.group(1).title() if match else ""


def _doc_symptoms(doc: Any) -> list[str]:
    metadata = _doc_metadata(doc)
    raw = metadata.get("symptoms", [])
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
            if isinstance(payload, list):
                return [str(item) for item in payload]
        except Exception:
            pass
        return [item.strip() for item in raw.split(",") if item.strip()]
    return []


def _seed_disease_from_docs(retrieved_docs: list[Any]) -> str:
    for doc in retrieved_docs:
        disease = _doc_disease(doc)
        if disease:
            return disease
    return ""
