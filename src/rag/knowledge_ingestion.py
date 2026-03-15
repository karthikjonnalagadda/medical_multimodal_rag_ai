"""
rag/knowledge_ingestion.py
--------------------------
Lightweight local medical knowledge ingestion for the multimodal RAG pipeline.

This module focuses on:
- loading PDF and TXT knowledge sources from disk
- extracting text with LangChain's PyPDFLoader when available
- chunking with `MedicalDocumentChunker` (section-aware sliding windows)
- embedding with sentence-transformers/all-MiniLM-L6-v2
- persisting metadata-rich chunks into ChromaDB
"""

from __future__ import annotations

import json
import hashlib
import re
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from src.config import KNOWLEDGE_BASE_DIR, VECTOR_DB_CONFIG
from src.rag.chunking import ChunkConfig, MedicalDocumentChunker
from src.vector_db.faiss_store import ChromaVectorStore, Document

try:
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:
    PyPDFLoader = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


_SOURCE_HINTS = ("pubmed", "who", "cdc")
_SYMPTOM_VOCAB = {
    "abdominal pain",
    "chest pain",
    "chills",
    "confusion",
    "cough",
    "diarrhea",
    "dizziness",
    "fatigue",
    "fever",
    "headache",
    "hemoptysis",
    "loss of taste",
    "loss of smell",
    "myalgia",
    "nausea",
    "night sweats",
    "productive cough",
    "rash",
    "shortness of breath",
    "sore throat",
    "vomiting",
    "weakness",
    "weight loss",
    "wheezing",
}


class _SentenceTransformerAdapter:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._fallback = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.get_sentence_embedding_dimension()), dtype=np.float32)

        if self._model is None and self._fallback is None:
            self._load()

        if self._model is not None:
            embeddings = self._model.encode(
                texts,
                batch_size=32,
                show_progress_bar=len(texts) > 50,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return embeddings.astype(np.float32)

        return self._fallback.embed_batch(texts).astype(np.float32)

    def get_sentence_embedding_dimension(self) -> int:
        if self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        if self._fallback is not None:
            return self._fallback.get_embedding_dim()
        return 384

    def _load(self) -> None:
        if SentenceTransformer is not None:
            try:
                self._model = SentenceTransformer(self.model_name, device="cpu")
                return
            except Exception as exc:
                logger.warning("Falling back from SentenceTransformer in knowledge ingestion: {}", exc)

        from src.embeddings.embedding_model import MedicalEmbeddingModel

        self._fallback = MedicalEmbeddingModel(
            backend="sentence_transformer",
            model_name=self.model_name,
            device="cpu",
        )


def ingest_medical_knowledge(
    directory_path: str | Path,
    *,
    persist_dir: str | Path | None = None,
    collection_name: str | None = None,
    embedding_model: Any | None = None,
    vector_store: ChromaVectorStore | None = None,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    chunker: MedicalDocumentChunker | None = None,
) -> dict[str, Any]:
    """
    Load local PDF/TXT medical documents and persist chunk embeddings into ChromaDB.

    Parameters
    ----------
    directory_path:
        Directory containing `.pdf` and `.txt` files.
    persist_dir:
        Optional Chroma persistence directory. Defaults to config.
    collection_name:
        Optional Chroma collection name. Defaults to config.
    embedding_model:
        Optional embedding model object with `encode(texts)` or `embed_batch(texts)`.
    vector_store:
        Optional existing `ChromaVectorStore`.
    chunk_size / chunk_overlap:
        Passed through to `ChunkConfig(max_chars=..., overlap_chars=...)`.
    """
    directory = Path(directory_path)
    if not directory.exists():
        raise FileNotFoundError(f"Knowledge directory not found: {directory}")

    embedder = embedding_model or _SentenceTransformerAdapter()
    chunker = chunker or MedicalDocumentChunker(
        ChunkConfig(
            max_chars=chunk_size,
            overlap_chars=chunk_overlap,
            preserve_sections=True,
        )
    )

    if vector_store is None:
        vector_store = ChromaVectorStore(
            persist_dir=str(persist_dir or VECTOR_DB_CONFIG["chroma_persist_dir"]),
            collection_name=collection_name or VECTOR_DB_CONFIG["collection_name"],
            embedding_dim=_embedding_dim(embedder),
        )

    all_documents: list[Document] = []
    ingested_files = 0

    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in {".pdf", ".txt"}:
            continue

        try:
            records = _load_records(path)
        except Exception as exc:
            logger.warning("Skipping knowledge source {}: {}", path, exc)
            continue

        if not records:
            continue

        ingested_files += 1
        chunks_to_embed: list[str] = []
        chunk_metadata: list[dict[str, Any]] = []
        chunk_sources: list[str] = []
        chunk_ids: list[str] = []

        for record in records:
            base_metadata = _build_metadata(path=path, text=record["text"], page=record.get("page"), extra=record.get("metadata"))
            for chunk in _chunk_record_text(
                text=record["text"],
                base_metadata=base_metadata,
                chunker=chunker,
            ):
                chunk_text = chunk["text"].strip()
                if not chunk_text:
                    continue
                chunk_id = _stable_chunk_id(
                    source_path=str(path),
                    metadata=chunk["metadata"],
                    text=chunk_text,
                )
                chunks_to_embed.append(chunk_text)
                chunk_metadata.append(chunk["metadata"])
                chunk_sources.append(base_metadata["source"])
                chunk_ids.append(chunk_id)

        if not chunks_to_embed:
            continue

        embeddings = _encode_texts(embedder, chunks_to_embed)
        for idx, chunk_text in enumerate(chunks_to_embed):
            all_documents.append(
                Document(
                    id=str(chunk_ids[idx] if idx < len(chunk_ids) else uuid.uuid4()),
                    text=chunk_text,
                    embedding=embeddings[idx],
                    metadata=chunk_metadata[idx],
                    source=chunk_sources[idx],
                )
            )

    vector_store.add_documents(all_documents)
    logger.info(
        "Medical knowledge ingestion complete | files={} | chunks={} | collection={}",
        ingested_files,
        len(all_documents),
        vector_store.collection_name,
    )
    return {
        "files_indexed": ingested_files,
        "chunks_indexed": len(all_documents),
        "collection_name": vector_store.collection_name,
        "persist_dir": vector_store.persist_dir,
    }


def ensure_medical_knowledge_base_loaded(
    *,
    knowledge_dir: str | Path = KNOWLEDGE_BASE_DIR,
    persist_dir: str | Path | None = None,
    collection_name: str | None = None,
    embedding_model: Any | None = None,
    vector_store: ChromaVectorStore | None = None,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    chunker: MedicalDocumentChunker | None = None,
) -> dict[str, Any] | None:
    """
    Bootstrap the local knowledge-base folder into the vector store once.

    Returns ingestion stats when new documents were indexed, otherwise `None`.
    """
    directory = Path(knowledge_dir)
    if not directory.exists():
        return None

    source_files = [
        path for path in directory.rglob("*")
        if path.suffix.lower() in {".pdf", ".txt"}
    ]
    if not source_files:
        return None

    if vector_store is not None and vector_store.count() > 0:
        return None

    return ingest_medical_knowledge(
        directory,
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
        vector_store=vector_store,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunker=chunker,
    )


def _load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [{"text": text, "page": 1, "metadata": _load_sidecar_metadata(path)}]
    if path.suffix.lower() == ".pdf":
        return _load_pdf(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _load_pdf(path: Path) -> list[dict[str, Any]]:
    sidecar = _load_sidecar_metadata(path)

    if PyPDFLoader is not None:
        pages = PyPDFLoader(str(path)).load()
        return [
            {
                "text": page.page_content,
                "page": page.metadata.get("page", index + 1),
                "metadata": sidecar,
            }
            for index, page in enumerate(pages)
            if page.page_content and page.page_content.strip()
        ]

    import fitz

    doc = fitz.open(path)
    records = []
    for index, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            records.append({"text": text, "page": index + 1, "metadata": sidecar})
    return records
def _chunk_record_text(
    *,
    text: str,
    base_metadata: dict[str, Any],
    chunker: MedicalDocumentChunker,
) -> list[dict[str, Any]]:
    chunks = chunker.chunk_text(text, metadata=base_metadata)
    return chunks or []


def _stable_chunk_id(*, source_path: str, metadata: dict[str, Any], text: str) -> str:
    """
    Deterministic ID for a chunk so repeated ingestions upsert instead of duplicating.

    Chroma upserts by `id`, so stable ids prevent runaway duplicate embeddings when a user
    reruns ingestion for the same corpus.
    """
    page = metadata.get("page", "")
    chunk_index = metadata.get("chunk_index", "")
    section_index = metadata.get("section_index", "")
    raw = f"{source_path}|p={page}|s={section_index}|c={chunk_index}|{text}"
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()


def _load_sidecar_metadata(path: Path) -> dict[str, Any]:
    candidates = [
        path.with_suffix(path.suffix + ".meta.json"),
        path.with_suffix(".meta.json"),
        path.parent / f"{path.stem}.metadata.json",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception as exc:
            logger.warning("Unable to parse metadata sidecar {}: {}", candidate, exc)
    return {}


def _build_metadata(path: Path, text: str, page: int | None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    extra = extra or {}
    source = str(extra.get("source") or _infer_source(path, text))
    disease = str(extra.get("disease") or _infer_disease(path, text))
    symptoms = extra.get("symptoms") or _infer_symptoms(text)

    metadata = {
        "source": source,
        "disease": disease,
        "symptoms": list(symptoms),
        "path": str(path),
        "page": int(page or 1),
        "file_name": path.name,
    }
    return metadata


def _infer_source(path: Path, text: str) -> str:
    candidate = f"{path.stem} {text[:200]}".lower()
    for source in _SOURCE_HINTS:
        if source in candidate:
            return source
    return "pubmed" if "pmid" in candidate else "who"


def _infer_disease(path: Path, text: str) -> str:
    pattern = re.compile(
        r"\b(pneumonia|tuberculosis|asthma|copd|edema|effusion|atelectasis|covid-19|influenza|bronchitis)\b",
        re.IGNORECASE,
    )
    match = pattern.search(path.stem.replace("_", " "))
    if not match:
        match = pattern.search(text[:2000])
    if match:
        return match.group(1).strip().title()

    cleaned_stem = re.sub(r"[_\-]+", " ", path.stem).strip()
    return cleaned_stem[:80] if cleaned_stem else "general_medicine"


def _infer_symptoms(text: str) -> list[str]:
    lowered = text.lower()
    found = sorted(symptom for symptom in _SYMPTOM_VOCAB if symptom in lowered)
    if found:
        return found[:8]

    match = re.search(r"symptoms include ([^.]+)", lowered)
    if match:
        items = [item.strip(" ,.;") for item in match.group(1).split(",")]
        return [item for item in items if item][:8]
    return []


def _encode_texts(embedder: Any, texts: list[str]) -> np.ndarray:
    if hasattr(embedder, "encode"):
        embeddings = embedder.encode(texts)
    elif hasattr(embedder, "embed_batch"):
        embeddings = embedder.embed_batch(texts)
    else:
        raise TypeError("Embedding model must expose encode(texts) or embed_batch(texts).")
    return np.asarray(embeddings, dtype=np.float32)


def _embedding_dim(embedder: Any) -> int:
    if hasattr(embedder, "get_sentence_embedding_dimension"):
        return int(embedder.get_sentence_embedding_dimension())
    if hasattr(embedder, "get_embedding_dim"):
        return int(embedder.get_embedding_dim())
    return 384
