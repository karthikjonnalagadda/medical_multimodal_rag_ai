"""
rag/hybrid_retriever.py
-----------------------
Hybrid retrieval for medical RAG:
- lexical BM25-style retrieval
- dense vector retrieval
- reciprocal-rank style fusion
- metadata filtering
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from src.vector_db.faiss_store import Document, VectorStore


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-/\.]+")


@dataclass
class HybridSearchResult:
    document: Document
    lexical_score: float
    dense_score: float
    fused_score: float


class LightweightBM25:
    """Dependency-free BM25 implementation for medical text retrieval."""

    def __init__(self) -> None:
        self.documents: list[Document] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_freq: dict[str, int] = defaultdict(int)
        self.avgdl = 0.0

    def fit(self, documents: Iterable[Document]) -> None:
        self.documents = list(documents)
        self.doc_tokens = [self._tokenize(doc.text) for doc in self.documents]
        self.doc_freq = defaultdict(int)
        total_tokens = 0
        for tokens in self.doc_tokens:
            total_tokens += len(tokens)
            for token in set(tokens):
                self.doc_freq[token] += 1
        self.avgdl = (total_tokens / len(self.doc_tokens)) if self.doc_tokens else 0.0

    def search(self, query: str, top_k: int = 8, metadata_filter: Optional[dict] = None) -> list[tuple[Document, float]]:
        query_tokens = self._tokenize(query)
        scores: list[tuple[int, float]] = []
        for idx, (doc, tokens) in enumerate(zip(self.documents, self.doc_tokens)):
            if not _metadata_matches(doc.metadata, metadata_filter):
                continue
            score = self._bm25_score(query_tokens, tokens)
            if score > 0:
                scores.append((idx, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return [(self.documents[idx], score) for idx, score in scores[:top_k]]

    def _bm25_score(self, query_tokens: list[str], doc_tokens: list[str], k1: float = 1.5, b: float = 0.75) -> float:
        if not doc_tokens or not query_tokens:
            return 0.0
        counts = Counter(doc_tokens)
        doc_len = len(doc_tokens)
        total_docs = max(len(self.documents), 1)
        score = 0.0
        for token in query_tokens:
            if token not in counts:
                continue
            df = self.doc_freq.get(token, 0)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            tf = counts[token]
            denom = tf + k1 * (1 - b + b * doc_len / max(self.avgdl, 1e-9))
            score += idf * ((tf * (k1 + 1)) / denom)
        return score

    def _tokenize(self, text: str) -> list[str]:
        return [token.lower() for token in TOKEN_PATTERN.findall(text)]


class HybridMedicalRetriever:
    def __init__(self, vector_store: VectorStore, embedding_model) -> None:
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.lexical = LightweightBM25()
        self._documents: dict[str, Document] = {}

    def rebuild_lexical_index(self, documents: list[Document]) -> None:
        self._documents = {doc.id: doc for doc in documents}
        self.lexical.fit(documents)

    def add_documents(self, documents: list[Document]) -> None:
        for doc in documents:
            self._documents[doc.id] = doc
        self.lexical.fit(list(self._documents.values()))

    def search(
        self,
        query_text: str,
        top_k: int = 6,
        metadata_filter: Optional[dict] = None,
        dense_weight: float = 0.6,
    ) -> list[HybridSearchResult]:
        dense_results = []
        if self.vector_store.count() > 0:
            query_embedding = self.embedding_model.embed(query_text)
            dense_results = [
                doc for doc in self.vector_store.search_similar(query_embedding, top_k=max(top_k * 2, 8))
                if _metadata_matches(doc.metadata, metadata_filter)
            ]

        lexical_results = self.lexical.search(query_text, top_k=max(top_k * 2, 8), metadata_filter=metadata_filter)

        fused: dict[str, HybridSearchResult] = {}
        dense_rank_scores = {
            doc.id: dense_weight / (rank + 1)
            for rank, doc in enumerate(dense_results)
        }
        lexical_rank_scores = {
            doc.id: (1.0 - dense_weight) / (rank + 1)
            for rank, (doc, _score) in enumerate(lexical_results)
        }

        for doc in dense_results:
            fused[doc.id] = HybridSearchResult(
                document=doc,
                lexical_score=0.0,
                dense_score=float(doc.score),
                fused_score=dense_rank_scores.get(doc.id, 0.0),
            )

        for doc, lexical_score in lexical_results:
            if doc.id not in fused:
                fused[doc.id] = HybridSearchResult(
                    document=doc,
                    lexical_score=float(lexical_score),
                    dense_score=0.0,
                    fused_score=0.0,
                )
            fused[doc.id].lexical_score = float(lexical_score)
            fused[doc.id].fused_score += lexical_rank_scores.get(doc.id, 0.0)

        ranked = sorted(fused.values(), key=lambda item: item.fused_score, reverse=True)
        return ranked[:top_k]


def _metadata_matches(metadata: dict | None, metadata_filter: Optional[dict]) -> bool:
    if not metadata_filter:
        return True
    metadata = metadata or {}
    for key, expected in metadata_filter.items():
        if metadata.get(key) != expected:
            return False
    return True
