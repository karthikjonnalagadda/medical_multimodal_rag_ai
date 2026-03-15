# Medical Multimodal RAG AI Assistant

Production-ready multimodal medical AI system that analyzes **lab reports**, **medical images**, and **text queries** using **OCR**, **vision models**, and **RAG-based reasoning**.

> Educational use only. Outputs are AI-generated and must be validated by qualified clinicians.

## Project Overview

This repository implements a full stack medical assistant:

- **OCR** extracts text and structured lab metrics from PDFs/images.
- **Medical vision** analyzes X-rays (and supports DICOM) with explainability hooks (Grad-CAM).
- **RAG retrieval** pulls relevant evidence from a local knowledge base stored in a vector database.
- **LLM reasoning** generates structured diagnosis suggestions and explanations.
- **Chatbot** provides fast text-only Q&A with strict JSON output.
- **UIs**: FastAPI backend, Streamlit research dashboard, and a React clinical dashboard.

## Key Features

- Lab report OCR extraction (PDF/PNG/JPG) with metric parsing and reference-range flags
- Medical image analysis (X-ray/DICOM), confidence-scored findings, optional Grad-CAM overlay
- Hybrid RAG retrieval (dense + lexical) with evidence snippets and references
- Multimodal analysis endpoint combining OCR + vision + symptoms + clinical notes
- Text-only medical chatbot with strict clinical JSON schema
- Local knowledge base ingestion and persistence in ChromaDB (optional FAISS backend)
- Caching for faster iteration (FastAPI singletons, Streamlit `st.cache_resource`, embedding cache)

## System Architecture

### Data Flow (End-to-End)

1. User input (upload lab report and/or image, enter symptoms and notes)
2. OCR extraction (lab report) -> raw text + parsed metrics
3. Vision inference (medical image) -> findings + confidences (and optional Grad-CAM)
4. Text preprocessing -> normalized query text
5. Embeddings -> dense vectors for retrieval
6. Vector search + lexical search -> fused evidence set
7. LLM reasoning -> structured diagnosis response
8. API/UI response rendering

**Flow:** User Upload -> OCR Extraction -> Text Processing -> Embeddings -> Vector Search -> LLM Reasoning -> Response Generation

### Components

- OCR pipeline: `src/ocr/extract_lab_text.py`
- Vision pipeline: `src/vision/xray_analysis.py`, `src/vision/medical_models.py`
- RAG + retrieval: `src/rag/rag_pipeline.py`, `src/rag/hybrid_retriever.py`
- Knowledge ingestion + chunking: `src/rag/knowledge_ingestion.py`, `src/rag/chunking.py`
- Chatbot: `src/chat/chatbot.py`
- API: `src/api/main.py`, `src/api/chat_api.py`
- Optional persistence: `src/db/mongo.py`
- Streamlit UI: `app/streamlit_app.py`, `app/pages/medical_chatbot.py`
- React UI: `frontend/`

## Tech Stack

Frontend
- React + TypeScript (Vite)
- TailwindCSS (styles in `frontend/src/styles/`)

Backend
- Python
- FastAPI + Uvicorn
- Pydantic

AI / ML
- PyTorch + TorchVision
- Transformers + Sentence-Transformers
- EasyOCR + PyMuPDF (PDF) + (optional) Tesseract via `pytesseract`

Vector Database
- ChromaDB (default persistent store)
- FAISS (optional backend)

## Project Folder Structure

```
medical_multimodal_rag_ai/
├── app/                      # Streamlit app entrypoint + pages
├── data/                     # Knowledge base + vector store persistence + dataset scripts
├── frontend/                 # React dashboard (Vite)
├── scripts/                  # Repo utility scripts (cleanup, tree generation)
├── src/                      # Core backend code (API, OCR, vision, RAG, embeddings, db)
├── tests/                    # Pytest suite (smoke + unit/integration)
├── .env.example              # Example backend environment variables
├── requirements.txt          # Full dependency set
├── requirements.runtime.txt  # Minimal runtime dependencies for local execution
└── README.md
```

## Installation Guide

### 1) Clone

```bash
git clone <repo_url>
cd medical_multimodal_rag_ai
```

### 2) Create Virtual Environment (Python)

Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
```

macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

### 3) Install Dependencies

Minimal runtime install (recommended):
```bash
pip install -r requirements.runtime.txt
```

Full install (heavier; includes optional packages):
```bash
pip install -r requirements.txt
```

Optional: install a spaCy model if not present (the code will fall back gracefully):
```bash
python -m spacy download en_core_web_sm
```

### 4) Configure Environment Variables

Copy `.env.example` to `.env` and fill values as needed:

```bash
cp .env.example .env
```

The system will choose the first available LLM backend in this order:
`GROQ_API_KEY` -> `XAI_API_KEY` -> `OPENAI_API_KEY` -> fallback local HuggingFace model.

## Running the Project

### Run Backend (FastAPI)

```powershell
cd medical_multimodal_rag_ai
.\venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

API docs:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### Run Frontend (React Dashboard)

```powershell
cd medical_multimodal_rag_ai\frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Configure the backend URL in `frontend/.env` (or use `frontend/.env.example`):

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

### Run Streamlit (Research Dashboard)

```powershell
cd medical_multimodal_rag_ai
.\venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

## Usage

### OCR (Lab Reports)

- Upload a PDF/PNG/JPG lab report
- The system returns:
  - extracted raw text
  - average OCR confidence
  - parsed lab metrics with reference ranges (when known)

### Vision (Medical Images)

- Upload a PNG/JPG X-ray (or DICOM when `pydicom` is installed)
- The system returns:
  - top finding + confidence
  - full ranked findings list
  - image quality estimate and modality label

### Multimodal Analysis

- Provide any combination of:
  - lab report (OCR)
  - medical image (vision)
  - symptoms (text)
  - clinical notes (text)
- The system returns:
  - possible conditions + confidence
  - evidence + references
  - explanation and recommendations

### Chatbot

- Ask a medical question (text-only)
- Receives structured output plus sources and timings

## AI Models Used

Embeddings
- Default: `sentence-transformers/all-MiniLM-L6-v2`
- File: `src/embeddings/embedding_model.py`
- Input: cleaned clinical text
- Output: dense embedding vector (float32)

Vision
- Default backend: `torchxrayvision` (configurable via `VISION_BACKEND`)
- Files: `src/vision/xray_analysis.py`, `src/vision/medical_models.py`
- Input: image bytes (PNG/JPG/DICOM)
- Output: finding probabilities + optional embedding + feature map for explainability

OCR
- Default engine: EasyOCR
- File: `src/ocr/extract_lab_text.py`
- Input: image/PDF
- Output: extracted text blocks + derived metrics

LLM Reasoning
- Providers: Groq / xAI / OpenAI (OpenAI-compatible client), or HuggingFace fallback
- File: `src/rag/rag_pipeline.py`
- Output: structured diagnosis JSON (parsed into `DiagnosisResult`)

## Data Processing Pipeline

1. OCR extraction -> raw text
2. Text cleaning and normalization -> unified query string
3. Chunking (knowledge base ingestion) -> chunk text + metadata
4. Embedding generation -> dense vectors
5. Vector retrieval (Chroma/FAISS) + lexical BM25 -> fused evidence list
6. LLM reasoning -> structured JSON response -> API/UI output

## Chunking Strategy

Chunking is implemented in `src/rag/chunking.py` and used by ingestion in `src/rag/knowledge_ingestion.py`.

- Default config: `max_chars=1200`, `overlap_chars=200`, `preserve_sections=True`
- Section-aware splitting: detects headings like `History`, `Findings`, `Impression`, `Assessment`, `Plan`, etc.
- Sliding-window chunking with overlap to preserve context across boundaries
- Deterministic chunk IDs during ingestion to avoid duplicate embeddings on re-ingest

## Chatbot Response Format (Strict JSON)

The chatbot normalizes all model outputs to this schema and exposes it via the `/chat` API as `clinical`:

```json
{
  "diagnosis": "",
  "confidence": "",
  "possible_conditions": [],
  "explanation": "",
  "recommended_tests": [],
  "next_steps": []
}
```

Field meaning:
- `diagnosis`: best single candidate (string)
- `confidence`: short string (e.g., `"low"`, `"medium"`, `"high"`, `"85%"`)
- `possible_conditions`: ordered differential diagnosis (list of strings)
- `explanation`: short rationale based on the question and retrieved evidence
- `recommended_tests`: suggested confirmatory tests (list of strings)
- `next_steps`: practical actions (list of strings)

## API Endpoints

Base URL: `http://127.0.0.1:8000`

- `GET /health` - health check
- `GET /knowledge/stats` - vector store document count
- `GET /history/recent` - recent stored analyses (MongoDB optional)
- `POST /ocr` - OCR only (multipart file field: `file`)
- `POST /image-analysis` - vision only (multipart file field: `file`)
- `POST /analyze` - full multimodal analysis (multipart: `lab_report`, `medical_image`, plus form fields)
- `POST /chat` - chatbot (JSON body)
- `GET /chat/stats` - chatbot KB stats
- `POST /knowledge/ingest` - ingest raw text strings into the KB (JSON body)

### Examples

OCR:
```bash
curl -X POST "http://127.0.0.1:8000/ocr" -F "file=@data/lab_reports/sample.pdf"
```

Image analysis:
```bash
curl -X POST "http://127.0.0.1:8000/image-analysis" -F "file=@data/medical_images/sample.png"
```

Full analysis:
```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "lab_report=@data/lab_reports/sample.pdf" \
  -F "medical_image=@data/medical_images/sample.png" \
  -F "symptoms=fever, cough" \
  -F "patient_notes=Adult with pleuritic chest pain; no prior imaging."
```

Chat:
```bash
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"What are symptoms of pneumonia?\"}"
```

## Performance Optimizations

- FastAPI lazy singletons for OCR/vision/RAG (`src/api/main.py`): avoids repeated heavy initialization
- Streamlit `st.cache_resource` for OCR/vision/RAG (`app/streamlit_app.py`)
- Chatbot embedding cache (`functools.lru_cache`) in `src/chat/chatbot.py`
- Deterministic chunk IDs during knowledge ingestion to prevent duplicate upserts on re-ingestion

## Future Improvements

- Add model checkpoint management (download/verify versioned weights under `models/`)
- Add a dedicated Grad-CAM image endpoint for the React dashboard
- Add multilingual OCR + multilingual embeddings
- Add stronger domain-specific biomedical NER (SciSpacy) with optional model download automation
- Add auth, rate-limiting, and audit logging for production deployments
- Add evaluation harness (golden test cases, retrieval quality metrics)

## Contributing

1. Fork the repository and create a feature branch.
2. Add tests for behavior changes (`pytest`).
3. Keep changes scoped and documented.
4. Open a PR with a clear problem statement and verification steps.

## License

No license file is included yet. Add a `LICENSE` file before distributing or deploying publicly.

## Author

Maintainer: (fill in your name/handle)
