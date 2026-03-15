# Medical Multimodal Dashboard Frontend

React + TypeScript + Tailwind frontend imported from the provided dashboard archive and adapted to this repository's FastAPI backend.

## Run locally

```bash
cd frontend
npm install
npm run dev
```

The app expects the backend at `http://127.0.0.1:8000` by default.

To point it elsewhere, create `.env` in `frontend/`:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## Backend endpoints used

- `POST /ocr`
- `POST /image-analysis`
- `POST /analyze`
- `GET /knowledge/stats`

## Notes

- If the API is unavailable, the frontend falls back to bundled mock data so the UI remains demoable.
- Uploaded image preview is reused in the vision explainability page.
- The React app is kept under `frontend/` so it does not collide with the Python `src/` package.
