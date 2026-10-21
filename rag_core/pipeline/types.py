from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


def content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    source: str = ""
    doc_type: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    source: str = ""
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_document(cls, doc: Document) -> Chunk:
        # Loaders already split files into records, so one document is one chunk.
        return cls(f"{doc.doc_id}::0", doc.doc_id, doc.text.strip(), doc.source,
                   doc.metadata.get("section"), dict(doc.metadata))


@dataclass(frozen=True)
class ScoredChunk:
    """A retrieved chunk. ``score`` is what it is ranked by; the per-stage scores are kept."""

    chunk: Chunk
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
