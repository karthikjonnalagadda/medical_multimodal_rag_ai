"""
pipeline/multimodal_pipeline.py
-------------------------------
Higher-level orchestration layer for research-grade multimodal medical analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.explainability.calibration import top_differential
from src.rag.hybrid_retriever import HybridMedicalRetriever
from src.rag.rag_pipeline import DiagnosisResult


@dataclass
class MultimodalInput:
    lab_text: str = ""
    symptoms: list[str] = field(default_factory=list)
    patient_notes: str = ""
    image_findings: list[str] = field(default_factory=list)
    modality: str = "unknown"
    image_embedding: Optional[object] = None
    image_summary: str = ""


@dataclass
class MultimodalEvidenceBundle:
    retrieved_chunks: list[dict] = field(default_factory=list)
    image_differential: list[tuple[str, float]] = field(default_factory=list)
    image_summary: str = ""


class MedicalMultimodalPipeline:
    """
    Connects vision, OCR, retrieval, and LLM reasoning with a richer evidence bundle.
    """

    def __init__(self, rag_pipeline, retriever: HybridMedicalRetriever) -> None:
        self.rag_pipeline = rag_pipeline
        self.retriever = retriever

    def run(self, payload: MultimodalInput, metadata_filter: Optional[dict] = None) -> tuple[DiagnosisResult, MultimodalEvidenceBundle]:
        query_text = self._build_query(payload)
        retrieved = self.retriever.search(query_text, top_k=self.rag_pipeline.top_k, metadata_filter=metadata_filter)
        retrieved_docs = [item.document for item in retrieved]

        result = self.rag_pipeline.diagnose(
            lab_text=payload.lab_text,
            image_findings=payload.image_findings,
            symptoms=payload.symptoms,
            patient_notes=payload.patient_notes,
            retrieved_docs_override=retrieved_docs,
        )

        bundle = MultimodalEvidenceBundle(
            retrieved_chunks=[
                {
                    "source": item.document.source,
                    "text": item.document.text,
                    "metadata": item.document.metadata,
                    "lexical_score": item.lexical_score,
                    "dense_score": item.dense_score,
                    "fused_score": item.fused_score,
                }
                for item in retrieved
            ],
            image_summary=payload.image_summary,
            image_differential=self._extract_image_differential(payload),
        )
        return result, bundle

    def _build_query(self, payload: MultimodalInput) -> str:
        parts = []
        if payload.lab_text:
            parts.append(f"Lab report:\n{payload.lab_text}")
        if payload.patient_notes:
            parts.append(f"Clinical notes:\n{payload.patient_notes}")
        if payload.symptoms:
            parts.append("Symptoms: " + ", ".join(payload.symptoms))
        if payload.image_findings:
            parts.append("Imaging findings: " + "; ".join(payload.image_findings))
        return "\n\n".join(parts)

    def _extract_image_differential(self, payload: MultimodalInput) -> list[tuple[str, float]]:
        if not hasattr(payload.image_embedding, "items"):
            return []
        return top_differential(payload.image_embedding, top_k=3)
