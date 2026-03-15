"""
rag/chunking.py
---------------
Chunking utilities tuned for medical literature, reports, and clinical notes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ChunkConfig:
    max_chars: int = 1200
    overlap_chars: int = 200
    preserve_sections: bool = True


class MedicalDocumentChunker:
    SECTION_SPLIT_PATTERN = re.compile(
        r"(?im)^(history|impression|findings|assessment|plan|conclusion|discussion|methods|results|diagnosis|recommendation)s?:\s*$"
    )

    def __init__(self, config: ChunkConfig | None = None) -> None:
        self.config = config or ChunkConfig()

    def chunk_text(self, text: str, metadata: dict | None = None) -> list[dict]:
        metadata = metadata or {}
        normalized = self._normalize(text)
        if not normalized:
            return []

        sections = self._split_sections(normalized) if self.config.preserve_sections else [normalized]
        chunks: list[dict] = []
        for section_index, section in enumerate(sections):
            chunks.extend(self._window_section(section, metadata=metadata, section_index=section_index))
        return chunks

    def _normalize(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _split_sections(self, text: str) -> list[str]:
        matches = list(self.SECTION_SPLIT_PATTERN.finditer(text))
        if not matches:
            return [text]
        sections: list[str] = []
        start = 0
        for match in matches:
            if match.start() > start:
                sections.append(text[start:match.start()].strip())
            start = match.start()
        sections.append(text[start:].strip())
        return [section for section in sections if section]

    def _window_section(self, section: str, metadata: dict, section_index: int) -> list[dict]:
        max_chars = self.config.max_chars
        overlap = self.config.overlap_chars
        if len(section) <= max_chars:
            return [{
                "text": section,
                "metadata": {**metadata, "chunk_index": 0, "section_index": section_index},
            }]

        chunks: list[dict] = []
        start = 0
        chunk_index = 0
        while start < len(section):
            end = min(start + max_chars, len(section))
            window = section[start:end]
            if end < len(section):
                split = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("; "))
                if split > max_chars // 2:
                    end = start + split + 1
                    window = section[start:end]
            chunks.append({
                "text": window.strip(),
                "metadata": {**metadata, "chunk_index": chunk_index, "section_index": section_index},
            })
            if end >= len(section):
                break
            start = max(end - overlap, start + 1)
            chunk_index += 1
        return chunks
