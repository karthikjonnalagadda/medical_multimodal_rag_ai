"""
embeddings/embedding_model.py
------------------------------
Semantic embedding generation for medical text.

Supports: sentence-transformers · PubMedBERT · BioSentVec
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import HashingVectorizer
    _SKLEARN_HASHING_AVAILABLE = True
except ImportError:
    _SKLEARN_HASHING_AVAILABLE = False


# ── Embedding Model ────────────────────────────────────────────────────────────

class MedicalEmbeddingModel:
    """
    Generates dense semantic embeddings for medical text.

    Supports domain-tuned backends for medical RAG:
    * sentence_transformer  – fast default
    * bge_large             – stronger general retrieval
    * instructor_xl         – instruction-tuned retrieval
    * pubmedbert            – biomedical dense encoder
    * biobert               – biomedical transformer encoder
    * biogpt                – BioGPT mean-pooled embeddings

    Usage
    -----
    >>> model = MedicalEmbeddingModel()
    >>> embedding = model.embed("Pneumonia with elevated WBC count")
    >>> batch = model.embed_batch(["Pneumonia", "Tuberculosis", "Asthma"])
    """

    # Default model names
    _BACKENDS = {
        "sentence_transformer": "sentence-transformers/all-MiniLM-L6-v2",
        "bge_large":            "BAAI/bge-large-en-v1.5",
        "instructor_xl":        "hkunlp/instructor-xl",
        "pubmedbert":           "pritamdeka/S-PubMedBert-MS-MARCO",
        "biobert":              "dmis-lab/biobert-base-cased-v1.2",
        "biogpt":               "microsoft/BioGPT-Large",
    }

    def __init__(
        self,
        backend: str = "sentence_transformer",
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize: bool = True,
    ):
        self.backend = backend
        self.model_name = model_name or self._BACKENDS.get(backend, self._BACKENDS["sentence_transformer"])
        self.batch_size = batch_size
        self.normalize = normalize

        if device:
            self.device = device
        else:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"

        self._model = None
        self._tokenizer = None
        self._load_model()

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_model(self):
        logger.info(f"Loading embedding model: {self.model_name} [{self.backend}]")
        try:
            if self.backend in ("sentence_transformer", "pubmedbert", "bge_large", "instructor_xl"):
                if not _ST_AVAILABLE:
                    raise ImportError("sentence-transformers not installed.")
                self._model = SentenceTransformer(self.model_name, device=self.device)
            else:
                # Generic HuggingFace mean-pool approach
                if not _TRANSFORMERS_AVAILABLE:
                    raise ImportError("transformers not installed.")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)
                self._model.to(self.device)
                self._model.eval()

            logger.success(f"Embedding model loaded: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load {self.model_name}: {e}")
            logger.warning("Falling back to TF-IDF-based pseudo-embeddings.")
            self._model = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string → 1-D numpy array."""
        if not text.strip():
            return np.zeros(self._dim())
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts → 2-D numpy array (N × dim).

        Parameters
        ----------
        texts : list of strings

        Returns
        -------
        np.ndarray, shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.empty((0, self._dim()))

        if self._model is None:
            return self._tfidf_fallback(texts)

        try:
            if isinstance(self._model, SentenceTransformer):
                if self.backend == "instructor_xl":
                    prompts = [["Represent the medical document for retrieval:", text] for text in texts]
                    embeddings = self._model.encode(
                        prompts,
                        batch_size=self.batch_size,
                        show_progress_bar=len(texts) > 50,
                        normalize_embeddings=self.normalize,
                        convert_to_numpy=True,
                    )
                else:
                    embeddings = self._model.encode(
                        texts,
                        batch_size=self.batch_size,
                        show_progress_bar=len(texts) > 50,
                        normalize_embeddings=self.normalize,
                        convert_to_numpy=True,
                    )
                return embeddings.astype(np.float32)
            else:
                return self._hf_encode(texts)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return self._tfidf_fallback(texts)

    def get_embedding_dim(self) -> int:
        return self._dim()

    # ── Private helpers ────────────────────────────────────────────────────────

    def _dim(self) -> int:
        if self._model is None:
            return 384
        if isinstance(self._model, SentenceTransformer):
            return self._model.get_sentence_embedding_dimension()
        # HF model
        try:
            return self._model.config.hidden_size
        except Exception:
            return 768

    def _hf_encode(self, texts: list[str]) -> np.ndarray:
        """Mean-pool last hidden state for generic HF models."""
        import torch
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self._model(**encoded)
            # Mean pool over token dimension
            attention_mask = encoded["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).float()
            summed = (token_embeddings * mask_expanded).sum(dim=1)
            counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
            mean_pooled = (summed / counts).cpu().numpy().astype(np.float32)
            if self.normalize:
                norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
                mean_pooled = mean_pooled / np.clip(norms, 1e-9, None)
            all_embeddings.append(mean_pooled)
        return np.vstack(all_embeddings)

    def _tfidf_fallback(self, texts: list[str]) -> np.ndarray:
        """
        Lightweight hashing-based pseudo-embedding used when model fails to load.
        Keeps a stable fixed width so vector stores do not break across requests.
        """
        dim = self._dim()
        if _SKLEARN_HASHING_AVAILABLE:
            vectorizer = HashingVectorizer(
                n_features=dim,
                alternate_sign=False,
                norm=None,
                stop_words="english",
            )
            hashed = vectorizer.transform(texts).toarray().astype(np.float32)
        else:
            hashed = np.zeros((len(texts), dim), dtype=np.float32)
            for row_index, text in enumerate(texts):
                for token in text.lower().split():
                    hashed[row_index, hash(token) % dim] += 1.0

        if self.normalize:
            norms = np.linalg.norm(hashed, axis=1, keepdims=True)
            hashed = hashed / np.clip(norms, 1e-9, None)
        return hashed


# ── Document embedding helper ──────────────────────────────────────────────────

def embed_documents(
    documents: list[str],
    model: Optional[MedicalEmbeddingModel] = None,
    backend: str = "sentence_transformer",
) -> tuple[np.ndarray, MedicalEmbeddingModel]:
    """
    Embed a list of medical documents.

    Returns
    -------
    (embeddings np.ndarray, model instance)
    """
    if model is None:
        model = MedicalEmbeddingModel(backend=backend)
    embeddings = model.embed_batch(documents)
    logger.info(f"Embedded {len(documents)} documents → shape {embeddings.shape}")
    return embeddings, model
