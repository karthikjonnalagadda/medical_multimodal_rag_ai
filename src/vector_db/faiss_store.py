"""
vector_db/faiss_store.py
------------------------
Vector database abstraction layer.

Backends supported:
  * FAISS  – local, fast, zero dependencies beyond faiss-cpu
  * ChromaDB – persistent, metadata-rich, easy to query
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

try:
    import chromadb
    from chromadb.config import Settings
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class Document:
    """A text document with its embedding and metadata."""
    id: str
    text: str
    embedding: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)
    source: str = ""
    score: float = 0.0


# ── Abstract base ──────────────────────────────────────────────────────────────

class VectorStore(ABC):
    """Abstract vector store interface."""

    @abstractmethod
    def add_documents(self, documents: list[Document]) -> None: ...

    @abstractmethod
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> list[Document]: ...

    @abstractmethod
    def update_vectors(self, doc_id: str, document: Document) -> None: ...

    @abstractmethod
    def delete_document(self, doc_id: str) -> None: ...

    @abstractmethod
    def count(self) -> int: ...


# ── FAISS Store ────────────────────────────────────────────────────────────────

class FAISSVectorStore(VectorStore):
    """
    Local FAISS-based vector store with metadata side-car.

    Usage
    -----
    >>> store = FAISSVectorStore(dim=384, index_path="data/medical.index")
    >>> store.add_documents(docs)
    >>> results = store.search_similar(query_embedding, top_k=5)
    """

    def __init__(
        self,
        dim: int = 384,
        index_path: Optional[str] = None,
        metric: str = "cosine",       # "cosine" | "l2" | "ip"
    ):
        if not _FAISS_AVAILABLE:
            raise RuntimeError("faiss-cpu not installed. pip install faiss-cpu")

        self.dim = dim
        self.index_path = Path(index_path) if index_path else None
        self.metric = metric

        self._index: faiss.Index = self._build_index()
        self._id_map: list[str] = []        # position → doc id
        self._doc_store: dict[str, Document] = {}

        # Try loading persisted index
        if self.index_path and self.index_path.exists():
            self._load()

        logger.info(f"FAISSVectorStore ready | dim={dim} | metric={metric}")

    # ── Index building ─────────────────────────────────────────────────────────

    def _build_index(self) -> faiss.Index:
        if self.metric == "cosine":
            return faiss.IndexFlatIP(self.dim)    # inner-product on L2-normed vecs
        elif self.metric == "l2":
            return faiss.IndexFlatL2(self.dim)
        else:
            return faiss.IndexFlatIP(self.dim)

    # ── VectorStore interface ──────────────────────────────────────────────────

    def add_documents(self, documents: list[Document]) -> None:
        """Add a batch of Document objects to the store."""
        if not documents:
            return
        vecs: list[np.ndarray] = []
        for doc in documents:
            if doc.embedding is None:
                logger.warning(f"Document {doc.id} has no embedding – skipped.")
                continue
            vec = doc.embedding.astype(np.float32)
            if self.metric == "cosine":
                vec = self._l2_normalize(vec)
            vecs.append(vec)
            self._id_map.append(doc.id)
            self._doc_store[doc.id] = doc

        if vecs:
            matrix = np.vstack(vecs)
            self._index.add(matrix)
        logger.info(f"Added {len(vecs)} vectors | total={self._index.ntotal}")
        if self.index_path:
            self._save()

    def search_similar(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> list[Document]:
        """Return the top-k most similar documents."""
        if self._index.ntotal == 0:
            return []
        vec = query_embedding.astype(np.float32)
        if self.metric == "cosine":
            vec = self._l2_normalize(vec)
        distances, indices = self._index.search(vec.reshape(1, -1), top_k)

        results: list[Document] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue
            doc_id = self._id_map[idx]
            doc = self._doc_store.get(doc_id)
            if doc:
                import copy
                result_doc = copy.copy(doc)
                result_doc.score = float(dist)
                results.append(result_doc)
        return results

    def update_vectors(self, doc_id: str, document: Document) -> None:
        """Update metadata for an existing document (FAISS does not support in-place update)."""
        if doc_id in self._doc_store:
            self._doc_store[doc_id] = document
            logger.debug(f"Updated metadata for {doc_id}")
        else:
            logger.warning(f"Document {doc_id} not found – adding instead.")
            self.add_documents([document])

    def delete_document(self, doc_id: str) -> None:
        """Remove a document from the metadata store (vector remains in index)."""
        self._doc_store.pop(doc_id, None)
        logger.debug(f"Deleted metadata for {doc_id}")

    def count(self) -> int:
        return self._index.ntotal

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        meta_path = self.index_path.with_suffix(".meta.json")
        meta = {
            "id_map": self._id_map,
            "documents": {
                k: {"text": v.text, "metadata": v.metadata, "source": v.source}
                for k, v in self._doc_store.items()
            },
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        logger.debug(f"FAISS index saved to {self.index_path}")

    def _load(self):
        self._index = faiss.read_index(str(self.index_path))
        meta_path = self.index_path.with_suffix(".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self._id_map = meta.get("id_map", [])
            for k, v in meta.get("documents", {}).items():
                self._doc_store[k] = Document(
                    id=k, text=v["text"],
                    metadata=v.get("metadata", {}),
                    source=v.get("source", ""),
                )
        logger.info(f"Loaded FAISS index: {self._index.ntotal} vectors")

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / max(norm, 1e-9)


# ── ChromaDB Store ─────────────────────────────────────────────────────────────

class ChromaVectorStore(VectorStore):
    """
    Persistent ChromaDB-based vector store.

    Usage
    -----
    >>> store = ChromaVectorStore(persist_dir="data/chroma", collection="medical")
    >>> store.add_documents(docs)
    >>> results = store.search_similar(query_embedding, top_k=5)
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "medical_knowledge",
        embedding_dim: int = 384,
    ):
        if not _CHROMA_AVAILABLE:
            raise RuntimeError("chromadb not installed. pip install chromadb")

        self.embedding_dim = embedding_dim
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._ensure_embedding_dimension()
        logger.info(
            f"ChromaVectorStore ready | collection={collection_name} | "
            f"docs={self._collection.count()}"
        )

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return
        ids, texts, embeddings, metadatas = [], [], [], []
        for doc in documents:
            if doc.embedding is None:
                continue
            ids.append(doc.id)
            texts.append(doc.text)
            embeddings.append(doc.embedding.tolist())
            metadatas.append(self._sanitize_metadata({**doc.metadata, "source": doc.source}))
        if ids:
            batch_size = max(1, int(self._client.get_max_batch_size()))
            try:
                self._upsert_in_batches(
                    ids=ids,
                    texts=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    batch_size=batch_size,
                )
            except Exception as exc:
                if "dimension" not in str(exc).lower():
                    raise
                logger.warning(
                    "Chroma collection '{}' has an incompatible embedding dimension. "
                    "Resetting it to {} and retrying.",
                    self.collection_name,
                    self.embedding_dim,
                )
                self._reset_collection()
                self._upsert_in_batches(
                    ids=ids,
                    texts=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    batch_size=batch_size,
                )
            logger.info(f"Added/updated {len(ids)} documents | total={self.count()}")

    def search_similar(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> list[Document]:
        if self.count() == 0:
            return []
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k, self.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            if "dimension" not in str(exc).lower():
                raise
            logger.warning(
                "Chroma collection '{}' query dimension mismatch. "
                "Resetting it to {} and returning no results for this request.",
                self.collection_name,
                self.embedding_dim,
            )
            self._reset_collection()
            return []
        docs: list[Document] = []
        for idx in range(len(results["ids"][0])):
            doc_id = results["ids"][0][idx]
            text = results["documents"][0][idx]
            metadata = results["metadatas"][0][idx] or {}
            score = 1.0 - results["distances"][0][idx]   # cosine similarity
            docs.append(Document(
                id=doc_id, text=text, metadata=metadata,
                source=metadata.get("source", ""), score=score,
            ))
        return docs

    def update_vectors(self, doc_id: str, document: Document) -> None:
        self.add_documents([document])

    def delete_document(self, doc_id: str) -> None:
        self._collection.delete(ids=[doc_id])

    def count(self) -> int:
        return self._collection.count()

    def _reset_collection(self) -> None:
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _upsert_in_batches(
        self,
        *,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        batch_size: int,
    ) -> None:
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self._collection.upsert(
                ids=ids[start:end],
                documents=texts[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )

    def _ensure_embedding_dimension(self) -> None:
        if self._collection.count() == 0:
            return
        try:
            sample = self._collection.get(limit=1, include=["embeddings"])
        except Exception as exc:
            logger.warning(
                "Unable to inspect Chroma collection '{}' for dimension validation: {}",
                self.collection_name,
                exc,
            )
            return

        embeddings = sample.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return

        stored_dim = len(embeddings[0])
        if stored_dim == self.embedding_dim:
            return

        logger.warning(
            "Chroma collection '{}' uses embedding dimension {} but the current model uses {}. "
            "Resetting the collection so it can be rebuilt.",
            self.collection_name,
            stored_dim,
            self.embedding_dim,
        )
        self._reset_collection()

    @staticmethod
    def _sanitize_metadata(metadata: dict) -> dict:
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                sanitized[key] = value
            else:
                sanitized[key] = json.dumps(value)
        return sanitized


# ── Factory function ───────────────────────────────────────────────────────────

def create_vector_store(
    backend: str = "chromadb",
    dim: int = 384,
    faiss_index_path: Optional[str] = None,
    chroma_persist_dir: str = "./chroma_db",
    collection_name: str = "medical_knowledge",
) -> VectorStore:
    """
    Instantiate the appropriate vector store backend.

    >>> store = create_vector_store("faiss", dim=384, faiss_index_path="medical.index")
    >>> store = create_vector_store("chromadb")
    """
    if backend == "faiss":
        return FAISSVectorStore(dim=dim, index_path=faiss_index_path)
    elif backend == "chromadb":
        return ChromaVectorStore(
            persist_dir=chroma_persist_dir,
            collection_name=collection_name,
            embedding_dim=dim,
        )
    else:
        raise ValueError(f"Unknown vector store backend: {backend}")


def build_knowledge_base(
    documents: list[str],
    sources: list[str] | None = None,
    metadata: list[dict] | None = None,
    embedding_model=None,
    vector_store: Optional[VectorStore] = None,
    backend: str = "chromadb",
) -> VectorStore:
    """
    Build a knowledge base from a list of text documents.

    Parameters
    ----------
    documents : list of text strings
    sources   : optional list of source names (one per document)
    metadata  : optional list of metadata dicts
    embedding_model : MedicalEmbeddingModel instance (created if None)
    vector_store    : VectorStore instance (created if None)
    backend         : "faiss" or "chromadb"
    """
    from src.embeddings.embedding_model import MedicalEmbeddingModel

    if embedding_model is None:
        embedding_model = MedicalEmbeddingModel()
    if vector_store is None:
        vector_store = create_vector_store(
            backend=backend, dim=embedding_model.get_embedding_dim()
        )

    sources = sources or ["" for _ in documents]
    metadata = metadata or [{} for _ in documents]

    embeddings = embedding_model.embed_batch(documents)

    docs = [
        Document(
            id=str(uuid.uuid4()),
            text=text,
            embedding=embeddings[i],
            metadata=meta,
            source=src,
        )
        for i, (text, src, meta) in enumerate(zip(documents, sources, metadata))
    ]
    vector_store.add_documents(docs)
    logger.success(
        f"Knowledge base built | {len(docs)} documents | backend={backend}"
    )
    return vector_store
