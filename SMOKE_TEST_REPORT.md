# Smoke Test Report

Date: 2026-03-15

## Environment

- Python: `venv/Scripts/python.exe` (project-local venv)
- Node: `v22.14.0` (frontend runtime present; `node_modules/` removed during cleanup)

## Commands Run

```bash
venv\Scripts\python.exe -c "from src.api.main import app"
venv\Scripts\python.exe -m pytest -q
```

## Results

- `pytest`: **PASS** (`37 passed`)
- Imports: **PASS** (`src.api.main:app` imports cleanly)

## Coverage Of Requested Smoke Checks

- Application startup: `src.api.main:app` import succeeded (FastAPI app builds).
- Model loading: covered by tests in `tests/test_pipeline.py` (embedding + vision analyser paths).
- API endpoints: exercised indirectly via backend module import; full HTTP endpoint probing is not included in the unit suite.
- OCR processing / lab report extraction: covered by `TestMedicalOCR` unit tests.
- Image processing: covered by `TestMedicalImageAnalyser` and Grad-CAM wrapper tests.
- Chatbot response generation: JSON normalization logic is unit-tested in `tests/test_chatbot_json.py`. Live LLM calls depend on configured API keys (`GROQ_API_KEY` / `OPENAI_API_KEY` / `XAI_API_KEY`).

## Notes / Warnings Observed

- HuggingFace model downloads can occur on first run unless model cache is already warm; set `HF_TOKEN` to avoid unauthenticated rate limits.
- Some test artifact directories were created by earlier runs and are not removable due to Windows permissions (`.pytest_tmp`, `pytest-cache-files-*`, `pytest_tmp`).

