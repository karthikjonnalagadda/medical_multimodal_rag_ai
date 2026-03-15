# Repository Audit And Refactor Report

Date: 2026-03-15

## 1) Project Architecture Summary

This repo is a multimodal medical assistant with three runnable surfaces:

- **FastAPI backend**: `src/api/main.py` (run with `uvicorn src.api.main:app --reload --port 8000`)
- **Streamlit research UI**: `app/streamlit_app.py` (run with `streamlit run app/streamlit_app.py`)
- **React frontend**: `frontend/` (run with `npm install` then `npm run dev`)

High-level flow:

1. OCR (`src/ocr/extract_lab_text.py`) extracts lab-report text + metrics.
2. Vision (`src/vision/xray_analysis.py` + `src/vision/medical_models.py`) analyses X-ray/DICOM and can generate Grad-CAM overlays.
3. Retrieval + reasoning (`src/rag/rag_pipeline.py`, `src/rag/hybrid_retriever.py`) retrieves evidence from Chroma/FAISS and generates a structured diagnosis object.
4. Chat (`src/chat/chatbot.py`) does fast text-only RAG and now emits a strict JSON schema.

## 2) Project Structure Analysis

Full cleaned tree listing (excluding `venv/`, `.git/`, caches, and `node_modules/`) is written to `PROJECT_TREE_CLEAN.txt`.

### Category Map

Core application files:
- `src/` (backend logic)
- `app/` (Streamlit app and pages)
- `frontend/src/` (React UI)

ML/AI model files:
- `src/embeddings/` (embedding backend + fallback)
- `src/vision/` (backbones + inference)
- `src/rag/` (retrieval, chunking, ingestion, reasoning)
- `src/explainability/` (Grad-CAM + SHAP/LIME utilities)

API/backend files:
- `src/api/main.py` (FastAPI entrypoint)
- `src/api/chat_api.py` (chat endpoints)
- `src/db/mongo.py` (optional Mongo persistence)
- `src/config.py` (runtime configuration)

Frontend/UI files:
- `frontend/` (Vite + React)
- `app/streamlit_app.py` (Streamlit UI)

Configuration files:
- `requirements.txt`, `pyproject.toml`, `pytest.ini`
- `.env.example` and frontend `frontend/.env.example`
- lint configs: `.flake8`, `.pylintrc`, `.pre-commit-config.yaml`

Temporary/generated/untracked:
- `venv/` (local virtual environment)
- pytest artifacts (`.pytest_*`, `pytest_tmp`, `pytest-cache-files-*`)
- Chroma persistence under `data/vector_store/` (generated state, already ignored by `.gitignore`)
- `.env` (local secrets)

### What Is Used In Execution (Main Paths)

- Backend: `src/api/main.py` imports and instantiates OCR/vision/RAG lazily.
- Streamlit: `app/streamlit_app.py` loads OCR/vision/RAG via `@st.cache_resource`.
- Frontend: `frontend/src/app/services/api.ts` calls backend endpoints `/ocr`, `/image-analysis`, `/analyze`, `/chat`.

### Likely Dead/Non-runtime (Keep Only If You Need Them)

- `notebooks/` (training/evaluation helpers; not required to run the app)
- `example_chatbot_usage.py` / `test_chatbot.py` (manual scripts; not required for runtime)
- `data/knowledge_base/*.pdf` (large content; runtime-ingestable but usually not committed for production)

## 3) Smoke Test Report

Detailed report: `SMOKE_TEST_REPORT.md`.

Summary:
- `pytest -q`: **PASS** (37 tests)
- Import/build: `from src.api.main import app`: **PASS**

## 4) CRITICAL FILES

Format: `file_path | reason_used | dependency_chain`

- `src/api/main.py` | FastAPI entrypoint + lazy component wiring | `src/config.py -> src/ocr/*, src/vision/*, src/rag/*, src/embeddings/*, src/vector_db/*, src/db/mongo.py`
- `src/api/chat_api.py` | `/chat` and `/chat/stats` endpoints | `src/chat/chatbot.py`
- `app/streamlit_app.py` | Streamlit UI entrypoint | `src/config.py -> src/ocr/*, src/vision/*, src/rag/*, src/embeddings/*, src/vector_db/*`
- `frontend/src/main.tsx` | React entrypoint | `frontend/src/app/App.tsx -> frontend/src/app/pages/* -> frontend/src/app/services/api.ts`
- `frontend/src/app/services/api.ts` | Backend API client | `fetch(... /ocr /image-analysis /analyze /chat)`
- `src/config.py` | All runtime config + API keys | referenced by API/Streamlit/chat/pipeline
- `src/ocr/extract_lab_text.py` | OCR engine + metric parsing | used by `/ocr` and Streamlit
- `src/vision/xray_analysis.py` | Vision inference + modality/quality | used by `/image-analysis` and Streamlit
- `src/vision/medical_models.py` | Vision backend registry + fallbacks | used by `xray_analysis.py`
- `src/rag/rag_pipeline.py` | Main RAG diagnosis object + LLM wrappers | used by API, Streamlit, chatbot
- `src/rag/hybrid_retriever.py` | Dense + lexical hybrid retrieval | used by `rag_pipeline.py` and `pipeline/multimodal_pipeline.py`
- `src/rag/chunking.py` | Chunking primitives | used by ingestion
- `src/rag/knowledge_ingestion.py` | Local KB ingestion | used by API/Streamlit to bootstrap KB
- `src/vector_db/faiss_store.py` | Vector store abstraction | used by RAG and chatbot
- `src/embeddings/embedding_model.py` | Embedding backend + stable dim fallback | used by API/Streamlit/chat/retrieval

## 5) Chunking Strategy Report

### Where Chunking Happens

- `src/rag/chunking.py`: `MedicalDocumentChunker` (section-aware + sliding window)
- `src/rag/knowledge_ingestion.py`: uses `MedicalDocumentChunker` for PDF/TXT ingestion
- `src/rag/knowledge_ingest.py`: uses `MedicalDocumentChunker` for PubMed/local corpus ingestion

### Current Settings

- Default chunking: `ChunkConfig(max_chars=1200, overlap_chars=200, preserve_sections=True)`
- Effective size: ~250 to 350 tokens per chunk (rule of thumb: 3 to 5 chars per token)

### Refactor Done

- Removed dead LangChain text-splitter path from `src/rag/knowledge_ingestion.py` so chunking is deterministic.
- Made ingestion chunk IDs deterministic (hash-based) in `src/rag/knowledge_ingestion.py` and `src/rag/knowledge_ingest.py` to prevent duplicate embeddings on re-ingest.

### Recommended Optimized Strategy (Medical Domain)

- Lab reports: smaller chunks (600 to 900 chars) and preserve full “test line” groups; overlap 80 to 150 chars.
- Clinical notes: section-preserving chunks (1200 to 1800 chars), overlap 200 to 300 chars (better recall for A/P style notes).
- Medical literature: 1000 to 1400 chars, overlap 150 to 250 chars, keep headings with the following paragraph.

## 6) Chatbot JSON Issue Debugging And Fix

Problem:
- The chatbot previously prompted for “JSON-compatible markdown”, but the LLM wrapper enforces `response_format={"type":"json_object"}` on some backends, so outputs could be inconsistent and parsing fragile.

Fix implemented:
- `src/chat/chatbot.py` now enforces and normalizes a strict schema:
  - `diagnosis`, `confidence`, `possible_conditions`, `explanation`, `recommended_tests`, `next_steps`
- Added robust JSON extraction + coercion via `normalize_clinical_json_response`.
- Exposed strict JSON on the API response:
  - `src/api/chat_api.py` now returns `clinical` in addition to `answer`/`structured`.
- Added tests: `tests/test_chatbot_json.py`.

## 7) Dependency Audit

Existing “full” dependency list stays in `requirements.txt`.

Clean runtime list:
- `requirements.runtime.txt` (minimal for local execution)

Unused/optional dependencies currently present in `requirements.txt` (recommend removing or moving to a notebook/dev file):
- `llama-index*`, `langchain-openai` (not referenced in `src/`)
- `rank-bm25` (not used; retrieval uses `src/rag/hybrid_retriever.py`)
- `sqlalchemy`, `alembic` (not referenced)
- `albumentations`, `SimpleITK`, `onnxruntime`, `pdf2image`, `torchaudio` (not referenced in runtime code)

## 8) Performance Review (Key Findings)

- Embedding startup can trigger HuggingFace downloads; set `HF_TOKEN` and/or pre-warm caches for predictable cold starts.
- Knowledge-base ingestion is expensive; keep it lazy and deterministic:
  - stable chunk IDs now prevent duplicate upserts when ingestion is re-run.
- Chroma persistence under `data/vector_store/` is generated state; keep it out of source control (already ignored) and consider clearing it for clean rebuilds when changing embedding dims/models.

## 9) Cleanup Results

Infrastructure removed (not required for local execution):
- Docker + compose: `Dockerfile`, `docker-compose*.yml`
- Jenkins: `Jenkinsfile*`, `start-jenkins.bat`, Jenkins docs
- GitHub Actions: `.github/workflows/`
- Procfile/Makefile/deploy scripts and deployment guide

Large/unnecessary artifacts removed:
- `frontend/node_modules/`
- `frontend/.git/`
- `logs/`
- accidental directory `{data/`

Known leftover artifacts (Windows permissions prevented deletion):
- `.pytest_tmp/`, `pytest_tmp/`, `pytest-cache-files-*` (see cleanup logs: `CLEANUP_REMOVED_RUN2.txt`, `CLEANUP_REMOVED_RUN3.txt`)

