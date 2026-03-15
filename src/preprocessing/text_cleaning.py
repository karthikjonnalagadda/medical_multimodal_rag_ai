"""
preprocessing/text_cleaning.py
--------------------------------
NLP pipeline: tokenisation · normalisation · medical entity extraction.
Combines OCR text, image findings, and patient symptoms into a unified
context string ready for embedding and RAG retrieval.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False

try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class MedicalEntity:
    text: str
    label: str          # DISEASE | SYMPTOM | CHEMICAL | ANATOMY | PROCEDURE
    start: int
    end: int
    confidence: Optional[float] = None


@dataclass
class ProcessedMedicalText:
    raw_combined: str
    cleaned_text: str
    tokens: list[str] = field(default_factory=list)
    sentences: list[str] = field(default_factory=list)
    entities: list[MedicalEntity] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    query_ready: str = ""        # condensed string for vector search


# ── Medical term normalisations ────────────────────────────────────────────────

ABBREVIATION_MAP = {
    r"\bhb\b": "hemoglobin",
    r"\bhgb\b": "hemoglobin",
    r"\bwbc\b": "white blood cell count",
    r"\brbc\b": "red blood cell count",
    r"\bplt\b": "platelet count",
    r"\bbp\b": "blood pressure",
    r"\bhr\b": "heart rate",
    r"\btemp\b": "temperature",
    r"\bcrp\b": "c-reactive protein",
    r"\besr\b": "erythrocyte sedimentation rate",
    r"\bbun\b": "blood urea nitrogen",
    r"\bhba1c\b": "glycated hemoglobin",
    r"\btsh\b": "thyroid stimulating hormone",
    r"\bpt\b": "prothrombin time",
    r"\baptt\b": "activated partial thromboplastin time",
    r"\balt\b": "alanine transaminase",
    r"\bast\b": "aspartate transaminase",
    r"\bsob\b": "shortness of breath",
    r"\bdoe\b": "dyspnea on exertion",
    r"\bcp\b": "chest pain",
    r"\burti\b": "upper respiratory tract infection",
    r"\blrti\b": "lower respiratory tract infection",
    r"\bpna\b": "pneumonia",
    r"\bcxr\b": "chest x-ray",
    r"\bct\b": "computed tomography",
    r"\bmri\b": "magnetic resonance imaging",
}

# Symptom keyword vocabulary for entity matching (lightweight fallback)
SYMPTOM_KEYWORDS = {
    "fever", "cough", "chills", "fatigue", "dyspnea", "chest pain",
    "hemoptysis", "sputum", "wheezing", "nausea", "vomiting", "diarrhea",
    "headache", "myalgia", "arthralgia", "rash", "jaundice", "edema",
    "palpitations", "syncope", "weight loss", "night sweats",
    "shortness of breath", "confusion", "weakness", "dizziness",
}

ANATOMY_KEYWORDS = {
    "lung", "heart", "liver", "kidney", "spleen", "pancreas", "stomach",
    "intestine", "colon", "bladder", "prostate", "thyroid", "brain",
    "spine", "chest", "abdomen", "pelvis", "pleura", "alveoli",
    "bronchi", "trachea", "aorta",
}

DISEASE_KEYWORDS = {
    "pneumonia", "tuberculosis", "cancer", "tumor", "malignancy",
    "carcinoma", "lymphoma", "leukemia", "diabetes", "hypertension",
    "asthma", "copd", "emphysema", "fibrosis", "cirrhosis",
    "hepatitis", "cholecystitis", "appendicitis", "meningitis",
    "sepsis", "anemia", "thrombosis", "embolism", "infarction",
    "effusion", "consolidation", "atelectasis", "edema", "nodule",
}


# ── Text Processor ─────────────────────────────────────────────────────────────

class MedicalTextProcessor:
    """
    Multi-source NLP processor for clinical text.

    Usage
    -----
    >>> processor = MedicalTextProcessor()
    >>> result = processor.process(
    ...     lab_text="Hemoglobin: 9.5 g/dL  WBC: 15000",
    ...     image_findings=["Possible lung opacity in lower right lobe"],
    ...     symptoms=["fever", "productive cough", "shortness of breath"],
    ... )
    >>> print(result.query_ready)
    """

    def __init__(self, spacy_model: str = "en_core_web_sm", use_scispacy: bool = False):
        self.nlp = None
        self._load_spacy(spacy_model, use_scispacy)

    def _load_spacy(self, model_name: str, use_scispacy: bool):
        if not _SPACY_AVAILABLE:
            logger.warning("spaCy not available – using rule-based NLP fallback.")
            return
        try:
            if use_scispacy:
                import en_core_sci_sm
                self.nlp = en_core_sci_sm.load()
                logger.info("Loaded SciSpaCy model.")
            else:
                self.nlp = spacy.load(model_name)
                logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.warning(
                f"spaCy model '{model_name}' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(
        self,
        lab_text: str = "",
        image_findings: list[str] | None = None,
        symptoms: list[str] | None = None,
        patient_notes: str = "",
    ) -> ProcessedMedicalText:
        """
        Combine and process all medical input sources.

        Returns
        -------
        ProcessedMedicalText
        """
        image_findings = image_findings or []
        symptoms = symptoms or []

        # 1. Combine sources with clear section labels
        parts = []
        if lab_text.strip():
            parts.append(f"[LAB RESULTS]\n{lab_text.strip()}")
        if image_findings:
            parts.append("[IMAGING FINDINGS]\n" + "\n".join(f"- {f}" for f in image_findings))
        if symptoms:
            parts.append("[PATIENT SYMPTOMS]\n" + ", ".join(symptoms))
        if patient_notes.strip():
            parts.append(f"[CLINICAL NOTES]\n{patient_notes.strip()}")

        raw_combined = "\n\n".join(parts)
        cleaned = self._clean(raw_combined)

        # 2. Tokenise & sentence split
        tokens = self._tokenise(cleaned)
        sentences = self._sentence_split(cleaned)

        # 3. Entity extraction
        entities = self._extract_entities(cleaned)

        # 4. Keywords
        keywords = self._extract_keywords(entities, symptoms)

        # 5. Build compact query string for retrieval
        query_ready = self._build_query(entities, symptoms, image_findings)

        result = ProcessedMedicalText(
            raw_combined=raw_combined,
            cleaned_text=cleaned,
            tokens=tokens,
            sentences=sentences,
            entities=entities,
            keywords=keywords,
            query_ready=query_ready,
        )
        logger.debug(
            f"Text processing done | entities={len(entities)} | "
            f"keywords={len(keywords)} | query_len={len(query_ready)}"
        )
        return result

    # ── Cleaning ───────────────────────────────────────────────────────────────

    def _clean(self, text: str) -> str:
        """Normalise and clean clinical text."""
        # Unicode normalisation
        text = unicodedata.normalize("NFKD", text)
        # Expand abbreviations
        for pattern, replacement in ABBREVIATION_MAP.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        # Remove non-printable characters
        text = re.sub(r"[^\x20-\x7E\n]", " ", text)
        # Normalise whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove page numbers / headers that OCR often produces
        text = re.sub(r"page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)
        return text.strip()

    # ── Tokenisation ──────────────────────────────────────────────────────────

    def _tokenise(self, text: str) -> list[str]:
        if _NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text.lower())
                stop = stopwords.words("english")
                return [t for t in tokens if t.isalpha() and t not in stop]
            except LookupError:
                logger.warning("NLTK data not available; falling back to basic tokenization.")
        # Fallback
        return [t.lower() for t in text.split() if t.isalpha()]

    def _sentence_split(self, text: str) -> list[str]:
        if _NLTK_AVAILABLE:
            try:
                return sent_tokenize(text)
            except LookupError:
                logger.warning("NLTK punkt data not available; falling back to regex sentence split.")
        return [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]

    # ── Entity extraction ─────────────────────────────────────────────────────

    def _extract_entities(self, text: str) -> list[MedicalEntity]:
        entities: list[MedicalEntity] = []
        text_lower = text.lower()

        if self.nlp is not None:
            doc = self.nlp(text)
            for ent in doc.ents:
                entities.append(MedicalEntity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                ))
        else:
            # Rule-based fallback
            for keyword in SYMPTOM_KEYWORDS:
                for m in re.finditer(re.escape(keyword), text_lower):
                    entities.append(MedicalEntity(
                        text=keyword, label="SYMPTOM",
                        start=m.start(), end=m.end()
                    ))
            for keyword in ANATOMY_KEYWORDS:
                for m in re.finditer(r"\b" + re.escape(keyword) + r"\b", text_lower):
                    entities.append(MedicalEntity(
                        text=keyword, label="ANATOMY",
                        start=m.start(), end=m.end()
                    ))
            for keyword in DISEASE_KEYWORDS:
                for m in re.finditer(r"\b" + re.escape(keyword) + r"\b", text_lower):
                    entities.append(MedicalEntity(
                        text=keyword, label="DISEASE",
                        start=m.start(), end=m.end()
                    ))

        # Deduplicate by text + label
        seen: set[tuple[str, str]] = set()
        unique: list[MedicalEntity] = []
        for e in entities:
            key = (e.text.lower(), e.label)
            if key not in seen:
                seen.add(key)
                unique.append(e)
        return unique

    def _extract_keywords(
        self, entities: list[MedicalEntity], symptoms: list[str]
    ) -> list[str]:
        """Combine entity text + symptoms into a deduplicated keyword list."""
        kw: list[str] = [e.text.lower() for e in entities]
        kw += [s.lower() for s in symptoms]
        seen: set[str] = set()
        result: list[str] = []
        for k in kw:
            if k not in seen:
                seen.add(k)
                result.append(k)
        return result

    def _build_query(
        self,
        entities: list[MedicalEntity],
        symptoms: list[str],
        image_findings: list[str],
    ) -> str:
        """Build a compact retrieval query string."""
        parts: list[str] = []

        diseases = [e.text for e in entities if e.label in ("DISEASE", "CONDITION")]
        if diseases:
            parts.append("Conditions: " + ", ".join(diseases))

        symptom_list = list({s.lower() for s in symptoms})
        if symptom_list:
            parts.append("Symptoms: " + ", ".join(symptom_list))

        if image_findings:
            parts.append("Imaging: " + "; ".join(image_findings))

        anatomy = [e.text for e in entities if e.label == "ANATOMY"]
        if anatomy:
            parts.append("Anatomy: " + ", ".join(set(anatomy)))

        return ". ".join(parts)
