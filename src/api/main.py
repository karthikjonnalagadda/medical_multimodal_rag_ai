"""
api/main.py
-----------
FastAPI backend for the Medical AI Assistant.

Endpoints:
  POST /analyze          - full multimodal analysis
  POST /ocr              - OCR only
  POST /image-analysis   - vision only
  POST /chat             - medical text chatbot (text-only, fast)
  GET  /chat/stats       - chatbot knowledge base stats
  GET  /health           - health check
  GET  /knowledge/stats  - vector store stats
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from src.db.mongo import mongo_persistence
from src.api.chat_api import setup_chat_routes


app = FastAPI(
    title="Medical AI Assistant API",
    description="Multimodal Medical AI using RAG, Vision, and OCR",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup chat endpoints
setup_chat_routes(app)


_pipeline = None
_ocr = None
_vision = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from src.config import EMBEDDING_CONFIG, KNOWLEDGE_BASE_DIR, VECTOR_DB_CONFIG, get_llm_runtime_config
        from src.embeddings.embedding_model import MedicalEmbeddingModel
        from src.rag.knowledge_ingestion import ensure_medical_knowledge_base_loaded
        from src.rag.rag_pipeline import MedicalRAGPipeline, load_sample_knowledge_base
        from src.vector_db.faiss_store import create_vector_store

        llm_config = get_llm_runtime_config()
        embedding_model = MedicalEmbeddingModel(
            backend=EMBEDDING_CONFIG["backend"],
            model_name=EMBEDDING_CONFIG["model_name"],
            batch_size=EMBEDDING_CONFIG["batch_size"],
            device=EMBEDDING_CONFIG["device"],
        )
        vector_store = create_vector_store(
            backend=VECTOR_DB_CONFIG["backend"],
            dim=embedding_model.get_embedding_dim(),
            faiss_index_path=VECTOR_DB_CONFIG["faiss_index_path"],
            chroma_persist_dir=VECTOR_DB_CONFIG["chroma_persist_dir"],
            collection_name=VECTOR_DB_CONFIG["collection_name"],
        )
        logger.info(
            "Initialising RAG pipeline with backend={} model={}",
            llm_config["backend"],
            llm_config["model"],
        )
        _pipeline = MedicalRAGPipeline(
            vector_store=vector_store,
            embedding_model=embedding_model,
            llm_backend=llm_config["backend"],
            llm_model=llm_config["model"],
            openai_api_key=llm_config["api_key"] if llm_config["backend"] == "openai" else "",
            xai_api_key=llm_config["api_key"] if llm_config["backend"] == "xai" else "",
            groq_api_key=llm_config["api_key"] if llm_config["backend"] == "groq" else "",
            xai_base_url=llm_config["base_url"] or "https://api.x.ai/v1",
            groq_base_url=llm_config["base_url"] or "https://api.groq.com/openai/v1",
        )
        ensure_medical_knowledge_base_loaded(
            knowledge_dir=KNOWLEDGE_BASE_DIR,
            persist_dir=VECTOR_DB_CONFIG["chroma_persist_dir"],
            collection_name=VECTOR_DB_CONFIG["collection_name"],
            embedding_model=embedding_model,
            vector_store=vector_store,
        )
        if _pipeline.vector_store.count() == 0:
            load_sample_knowledge_base(_pipeline)
    return _pipeline


def get_ocr():
    global _ocr
    if _ocr is None:
        from src.ocr.extract_lab_text import MedicalOCR

        _ocr = MedicalOCR(engine="easyocr", gpu=False)
    return _ocr


def get_vision():
    global _vision
    if _vision is None:
        from src.config import VISION_CONFIG
        from src.vision.xray_analysis import MedicalImageAnalyser

        _vision = MedicalImageAnalyser(
            confidence_threshold=VISION_CONFIG["confidence_threshold"],
            backend_name=VISION_CONFIG["backend_name"],
            model_path=VISION_CONFIG["pretrained_weights"],
            model_id=VISION_CONFIG.get("hf_model_id") or None,
            device=VISION_CONFIG["device"],
            enable_gradcam=VISION_CONFIG.get("enable_gradcam", True),
        )
    return _vision


class ConditionResponse(BaseModel):
    name: str
    confidence: str
    icd_code: Optional[str] = None


class DiagnosisResponse(BaseModel):
    query_id: str
    processing_time_ms: float
    possible_conditions: list[ConditionResponse]
    evidence: list[str]
    references: list[str]
    explanation: str
    recommendation: str
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    version: str
    components: dict[str, str]


class KnowledgeStatsResponse(BaseModel):
    total_documents: int
    backend: str


class HistoryResponse(BaseModel):
    records: list[dict]


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        components={
            "ocr": "ready",
            "vision": "ready",
            "rag": "ready",
            "vector_db": "ready",
            "mongodb": mongo_persistence.health_status(),
        },
    )


@app.get("/knowledge/stats", response_model=KnowledgeStatsResponse, tags=["Knowledge"])
async def knowledge_stats():
    pipeline = get_pipeline()
    return KnowledgeStatsResponse(
        total_documents=pipeline.vector_store.count(),
        backend=type(pipeline.vector_store).__name__,
    )


@app.get("/history/recent", response_model=HistoryResponse, tags=["Database"])
async def recent_history(limit: int = 20):
    return HistoryResponse(records=mongo_persistence.recent_records(kind="analyses", limit=limit))


@app.post("/ocr", tags=["Analysis"])
async def extract_lab_report(
    file: UploadFile = File(..., description="Lab report image or PDF"),
):
    start = time.perf_counter()
    try:
        content = await file.read()
        ocr = get_ocr()
        report = ocr.extract_from_bytes(content, filename=file.filename or "")
        elapsed = (time.perf_counter() - start) * 1000

        return {
            "db_record_id": mongo_persistence.save_analysis(
                "ocr",
                {
                    "filename": file.filename,
                    "raw_text": report.raw_text[:5000],
                    "avg_confidence": round(report.confidence, 3),
                    "metrics_count": len(report.metrics),
                    "patient_info": report.patient_info,
                },
            ),
            "filename": file.filename,
            "processing_time_ms": round(elapsed, 1),
            "raw_text": report.raw_text,
            "ocr_engine": report.ocr_engine,
            "avg_confidence": round(report.confidence, 3),
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "unit": m.unit,
                    "in_normal_range": m.in_normal_range,
                    "reference_range": m.reference_range,
                    "status": m.status,
                    "interpretation": m.interpretation,
                }
                for m in report.metrics
            ],
            "patient_info": report.patient_info,
        }
    except Exception as exc:
        logger.error(f"OCR error: {exc}")
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}") from exc


@app.post("/image-analysis", tags=["Analysis"])
async def analyse_image(
    file: UploadFile = File(..., description="Medical image (X-ray, CT, MRI, DICOM)"),
):
    start = time.perf_counter()
    try:
        content = await file.read()
        vision = get_vision()
        result = vision.analyse(content)
        elapsed = (time.perf_counter() - start) * 1000

        return {
            "db_record_id": mongo_persistence.save_analysis(
                "vision",
                {
                    "filename": file.filename,
                    "modality": result.modality,
                    "image_quality": result.image_quality,
                    "backend_name": result.backend_name,
                    "top_finding": result.top_finding.label if result.top_finding else None,
                    "top_confidence": round(result.top_finding.confidence, 3) if result.top_finding else None,
                    "all_findings": [
                        {
                            "label": finding.label,
                            "confidence": round(finding.confidence, 3),
                            "description": finding.description,
                        }
                        for finding in result.findings
                    ],
                },
            ),
            "filename": file.filename,
            "processing_time_ms": round(elapsed, 1),
            "modality": result.modality,
            "image_quality": result.image_quality,
            "top_finding": {
                "label": result.top_finding.label,
                "confidence": round(result.top_finding.confidence, 3),
                "description": result.top_finding.description,
            }
            if result.top_finding
            else None,
            "all_findings": [
                {
                    "label": finding.label,
                    "confidence": round(finding.confidence, 3),
                    "description": finding.description,
                    "is_abnormal": finding.is_abnormal,
                }
                for finding in result.findings
            ],
            "backend_name": result.backend_name,
            "top_differential": [
                {"label": label, "confidence": round(score, 3)}
                for label, score in result.differential_diagnosis
            ],
        }
    except Exception as exc:
        logger.error(f"Vision error: {exc}")
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {exc}") from exc


@app.post("/analyze", response_model=DiagnosisResponse, tags=["Analysis"])
async def full_analysis(
    lab_report: Optional[UploadFile] = File(default=None, description="Lab report (PDF/image)"),
    medical_image: Optional[UploadFile] = File(default=None, description="Medical image"),
    symptoms: str = Form(default="", description="Comma-separated symptoms"),
    patient_notes: str = Form(default="", description="Free-text clinical notes"),
):
    if not lab_report and not medical_image and not symptoms.strip():
        raise HTTPException(
            status_code=400,
            detail="At least one of lab_report, medical_image, or symptoms must be provided.",
        )

    start = time.perf_counter()
    query_id = f"q{int(time.time() * 1000)}"

    try:
        lab_text = ""
        if lab_report:
            content = await lab_report.read()
            ocr = get_ocr()
            lab_report_result = ocr.extract_from_bytes(content, filename=lab_report.filename or "")
            lab_text = lab_report_result.raw_text

        image_findings: list[str] = []
        if medical_image:
            content = await medical_image.read()
            vision = get_vision()
            image_result = vision.analyse(content)
            if image_result.top_finding:
                image_findings.append(
                    f"{image_result.top_finding.label}: "
                    f"{image_result.top_finding.description} "
                    f"(confidence {image_result.top_finding.confidence:.0%})"
                )
            for finding in image_result.findings:
                image_findings.append(f"{finding.label} ({finding.confidence:.0%})")

        symptom_list = [item.strip() for item in symptoms.split(",") if item.strip()]

        pipeline = get_pipeline()
        result = pipeline.diagnose(
            lab_text=lab_text,
            image_findings=image_findings,
            symptoms=symptom_list,
            patient_notes=patient_notes,
        )

        elapsed = (time.perf_counter() - start) * 1000

        response = DiagnosisResponse(
            query_id=query_id,
            processing_time_ms=round(elapsed, 1),
            possible_conditions=[
                ConditionResponse(
                    name=condition.name,
                    confidence=f"{condition.confidence:.0%}",
                    icd_code=condition.icd_code,
                )
                for condition in result.conditions
            ],
            evidence=result.evidence,
            references=list(set(result.references))[:5],
            explanation=result.explanation,
            recommendation=result.recommendation,
            disclaimer=result.disclaimer,
        )
        mongo_persistence.save_analysis(
            "analyses",
            {
                "query_id": query_id,
                "processing_time_ms": round(elapsed, 1),
                "symptoms": symptom_list,
                "patient_notes": patient_notes,
                "image_findings": image_findings,
                "possible_conditions": [item.model_dump() for item in response.possible_conditions],
                "evidence": result.evidence,
                "references": list(set(result.references))[:5],
                "explanation": result.explanation,
                "recommendation": result.recommendation,
            },
        )
        return response
    except Exception as exc:
        logger.error(f"Full analysis error: {exc}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc


@app.post("/knowledge/ingest", tags=["Knowledge"])
async def ingest_documents(
    texts: list[str],
    sources: Optional[list[str]] = None,
):
    try:
        pipeline = get_pipeline()
        pipeline.ingest_knowledge(texts, sources=sources)
        return {
            "status": "success",
            "ingested": len(texts),
            "total": pipeline.vector_store.count(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
