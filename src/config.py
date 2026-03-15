"""
config.py - Central configuration for Medical AI Assistant
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _default_torch_device() -> str:
    if os.getenv("FORCE_CPU", "0") == "1":
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

# ── Base Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# ── Data Sub-directories ──────────────────────────────────────────────────────
MEDICAL_IMAGES_DIR = DATA_DIR / "medical_images"
LAB_REPORTS_DIR = DATA_DIR / "lab_reports"
MEDICAL_TEXT_DIR = DATA_DIR / "medical_text"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

# ── API Keys ──────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
XAI_MODEL = os.getenv("XAI_MODEL", "grok-4-1-fast-reasoning")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
HF_TOKEN = os.getenv("HF_TOKEN", "")
MONGODB_URI = os.getenv("MONGODB_URI", "")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "medical_multimodal_ai")

# ── OCR Configuration ─────────────────────────────────────────────────────────
OCR_CONFIG = {
    "engine": "easyocr",           # "easyocr" | "tesseract" | "paddleocr"
    "languages": ["en"],
    "gpu": False,
    "confidence_threshold": 0.5,
}

# ── Vision Model Configuration ────────────────────────────────────────────────
VISION_CONFIG = {
    "backend_name": os.getenv("VISION_BACKEND", "torchxrayvision"),
    "model_name": "torchxrayvision",
    "num_classes": 14,             # NIH CheXpert labels
    "image_size": (224, 224),
    "pretrained_weights": str(MODELS_DIR / "vision_model" / "densenet121_chexpert.pth"),
    "hf_model_id": os.getenv("VISION_HF_MODEL_ID", ""),
    "device": "cpu",
    "labels": [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Effusion", "Emphysema", "Fibrosis", "Hernia",
        "Infiltration", "Mass", "Nodule", "Pleural Thickening",
        "Pneumonia", "Pneumothorax"
    ],
    "confidence_threshold": 0.5,
    "enable_gradcam": True,
}

# ── Embedding Configuration ───────────────────────────────────────────────────
EMBEDDING_CONFIG = {
    "backend": os.getenv("EMBEDDING_BACKEND", "sentence_transformer"),
    "model_name": os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
    "bio_model_name": "pritamdeka/S-PubMedBert-MS-MARCO",
    "instructor_model_name": "hkunlp/instructor-xl",
    "biobert_model_name": "dmis-lab/biobert-base-cased-v1.2",
    "embedding_dim": 384,
    "batch_size": 32,
    "max_seq_length": 512,
    "device": "cpu",
}

# ── Vector Database Configuration ─────────────────────────────────────────────
VECTOR_DB_CONFIG = {
    "backend": "chromadb",         # "faiss" | "chromadb"
    "faiss_index_path": str(VECTOR_STORE_DIR / "faiss_medical.index"),
    "chroma_persist_dir": str(VECTOR_STORE_DIR / "chroma"),
    "collection_name": "medical_knowledge",
    "top_k": 5,
    "similarity_metric": "cosine",
    "hybrid_search": True,
    "bm25_weight": 0.4,
}

# ── RAG Configuration ─────────────────────────────────────────────────────────
RAG_CONFIG = {
    "llm_provider": (
        "groq" if GROQ_API_KEY else
        "xai" if XAI_API_KEY else
        "openai" if OPENAI_API_KEY else
        "huggingface"
    ),
    "llm_model": (
        GROQ_MODEL if GROQ_API_KEY else
        XAI_MODEL if XAI_API_KEY else
        "gpt-4o" if OPENAI_API_KEY else
        "microsoft/BioGPT-Large-PubMedQA"
    ),
    "max_context_docs": 5,
    "max_tokens": 1024,
    "temperature": 0.1,
    "hybrid_search": True,
    "metadata_filters_enabled": True,
    "system_prompt": (
        "You are a specialized medical AI assistant. Analyze the provided medical "
        "data and retrieved knowledge to give evidence-based diagnostic insights. "
        "Always emphasize that results are for informational purposes only and "
        "professional medical consultation is required."
    ),
}

# ── NLP Configuration ─────────────────────────────────────────────────────────
NLP_CONFIG = {
    "spacy_model": "en_core_sci_sm",
    "entity_types": ["DISEASE", "CHEMICAL", "GENE_OR_GENE_PRODUCT", "ANATOMY"],
    "umls_linker": False,           # requires scispacy extras
}

# ── API Configuration ─────────────────────────────────────────────────────────
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 1,
    "max_upload_size_mb": 50,
    "allowed_image_types": [".jpg", ".jpeg", ".png", ".dcm", ".tif", ".tiff"],
    "allowed_report_types": [".pdf", ".png", ".jpg", ".jpeg"],
}

# MongoDB Configuration
MONGODB_CONFIG = {
    "uri": MONGODB_URI,
    "database": MONGODB_DB_NAME,
    "collections": {
        "analyses": "analysis_history",
        "ocr": "ocr_history",
        "vision": "vision_history",
    },
    "connect_timeout_ms": 5000,
}

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_CONFIG = {
    "level": "INFO",
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}",
    "log_file": str(LOGS_DIR / "medical_ai.log"),
    "rotation": "10 MB",
    "retention": "7 days",
}


def get_llm_runtime_config() -> dict:
    """Return the preferred LLM backend based on available environment variables."""
    if GROQ_API_KEY:
        return {
            "backend": "groq",
            "model": GROQ_MODEL,
            "api_key": GROQ_API_KEY,
            "base_url": GROQ_BASE_URL,
        }
    if XAI_API_KEY:
        return {
            "backend": "xai",
            "model": XAI_MODEL,
            "api_key": XAI_API_KEY,
            "base_url": XAI_BASE_URL,
        }
    if OPENAI_API_KEY:
        return {
            "backend": "openai",
            "model": "gpt-4o",
            "api_key": OPENAI_API_KEY,
            "base_url": None,
        }
    return {
        "backend": "huggingface",
        "model": "microsoft/BioGPT-Large-PubMedQA",
        "api_key": "",
        "base_url": None,
    }
