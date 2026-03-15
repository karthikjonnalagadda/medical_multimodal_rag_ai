"""
rag/knowledge_ingest.py
-----------------------
Knowledge ingestion helpers for medical RAG corpora.

This module focuses on:
- PubMed abstract ingestion via NCBI E-utilities
- Local PDF / TXT / JSONL ingestion
- Terminology ingestion from CSV / TSV files (ICD-10, SNOMED, UMLS exports)
"""

from __future__ import annotations

import csv
import hashlib
import json
import uuid
from pathlib import Path
from typing import Iterable, Optional
from xml.etree import ElementTree

import requests
from loguru import logger

from src.rag.chunking import MedicalDocumentChunker
from src.vector_db.faiss_store import Document


class MedicalKnowledgeIngester:
    def __init__(self, embedding_model, vector_store, chunker: MedicalDocumentChunker | None = None) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.chunker = chunker or MedicalDocumentChunker()

    def ingest_pubmed(self, query: str, max_results: int = 20, email: str = "research@example.com") -> int:
        pmids = self._search_pubmed(query=query, max_results=max_results, email=email)
        records = self._fetch_pubmed_records(pmids=pmids, email=email)
        docs = []
        for record in records:
            chunks = self.chunker.chunk_text(
                record["text"],
                metadata={
                    "source_type": "pubmed",
                    "pmid": record["pmid"],
                    "journal": record.get("journal", ""),
                    "year": record.get("year", ""),
                },
            )
            docs.extend(self._chunks_to_documents(
                chunks=chunks,
                source=f"PubMed PMID:{record['pmid']}",
            ))
        self._write_documents(docs)
        return len(docs)

    def ingest_local_corpus(self, directory: str | Path) -> int:
        directory = Path(directory)
        docs = []
        for path in directory.rglob("*"):
            if path.suffix.lower() not in {".txt", ".md", ".json", ".jsonl", ".pdf", ".csv", ".tsv"}:
                continue
            try:
                text_items = self._read_path(path)
            except Exception as exc:
                logger.warning("Skipping {}: {}", path, exc)
                continue
            for index, text_item in enumerate(text_items):
                chunks = self.chunker.chunk_text(
                    text_item,
                    metadata={
                        "source_type": "local_corpus",
                        "path": str(path),
                        "entry_index": index,
                    },
                )
                docs.extend(self._chunks_to_documents(chunks=chunks, source=str(path)))
        self._write_documents(docs)
        return len(docs)

    def ingest_terminology_table(
        self,
        path: str | Path,
        code_column: str,
        text_column: str,
        vocabulary: str,
        delimiter: str = ",",
    ) -> int:
        path = Path(path)
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            for row in reader:
                code = row.get(code_column, "").strip()
                description = row.get(text_column, "").strip()
                if code and description:
                    rows.append({
                        "text": f"{vocabulary} code {code}: {description}",
                        "metadata": {
                            "source_type": vocabulary.lower(),
                            "code": code,
                            "vocabulary": vocabulary,
                        },
                        "source": f"{vocabulary}:{code}",
                    })
        docs = self._records_to_documents(rows)
        self._write_documents(docs)
        return len(docs)

    def _search_pubmed(self, query: str, max_results: int, email: str) -> list[str]:
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        response = requests.get(
            url,
            params={
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                "email": email,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("esearchresult", {}).get("idlist", [])

    def _fetch_pubmed_records(self, pmids: list[str], email: str) -> list[dict]:
        if not pmids:
            return []
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        response = requests.get(
            url,
            params={
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
                "email": email,
            },
            timeout=30,
        )
        response.raise_for_status()
        root = ElementTree.fromstring(response.text)
        records = []
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID", default="")
            title = article.findtext(".//ArticleTitle", default="")
            abstract_text = " ".join(
                node.text.strip()
                for node in article.findall(".//Abstract/AbstractText")
                if node.text
            )
            journal = article.findtext(".//Journal/Title", default="")
            year = article.findtext(".//PubDate/Year", default="")
            text = "\n".join(part for part in [title, abstract_text] if part)
            if text:
                records.append({
                    "pmid": pmid,
                    "journal": journal,
                    "year": year,
                    "text": text,
                })
        return records

    def _read_path(self, path: Path) -> list[str]:
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            return [path.read_text(encoding="utf-8", errors="ignore")]
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return [json.dumps(payload)]
            return [json.dumps(item) for item in payload]
        if suffix == ".jsonl":
            return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if suffix in {".csv", ".tsv"}:
            delimiter = "\t" if suffix == ".tsv" else ","
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                reader = csv.DictReader(handle, delimiter=delimiter)
                return [json.dumps(row) for row in reader]
        if suffix == ".pdf":
            import fitz

            doc = fitz.open(path)
            return [page.get_text("text") for page in doc]
        raise ValueError(f"Unsupported file type: {suffix}")

    def _chunks_to_documents(self, chunks: list[dict], source: str) -> list[Document]:
        rows = []
        for chunk in chunks:
            rows.append({
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "source": source,
            })
        return self._records_to_documents(rows)

    def _records_to_documents(self, records: Iterable[dict]) -> list[Document]:
        records = list(records)
        if not records:
            return []
        ids = [_stable_record_id(record, index) for index, record in enumerate(records)]
        embeddings = self.embedding_model.embed_batch([record["text"] for record in records])
        docs = []
        for index, record in enumerate(records):
            docs.append(Document(
                id=ids[index] if index < len(ids) else str(uuid.uuid4()),
                text=record["text"],
                embedding=embeddings[index],
                metadata=record.get("metadata", {}),
                source=record.get("source", ""),
            ))
        return docs

    def _write_documents(self, docs: list[Document]) -> None:
        if not docs:
            logger.info("No documents produced for ingestion.")
            return
        self.vector_store.add_documents(docs)
        logger.info("Ingested {} medical knowledge chunks.", len(docs))


def _stable_record_id(record: dict, index: int) -> str:
    source = str(record.get("source", ""))
    metadata = record.get("metadata", {}) or {}
    text = str(record.get("text", ""))
    try:
        meta_text = json.dumps(metadata, sort_keys=True, ensure_ascii=True)
    except Exception:
        meta_text = str(metadata)
    raw = f"{source}|{index}|{meta_text}|{text}"
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()
