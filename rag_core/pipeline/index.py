"""Hybrid (dense + BM25) retrieval over Qdrant, fused with RRF."""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import asdict, replace
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from .config import PipelineSettings
from .embeddings import make_embedder
from .sparse import BM25, tokenize
from .types import Chunk, Document, ScoredChunk, content_hash

logger = logging.getLogger("rag.pipeline")


class HybridIndex:
    def __init__(self, settings: PipelineSettings, embedder=None):
        self.settings = settings
        self.embedder = embedder or make_embedder(settings)
        if settings.qdrant_url in (":memory:", "memory"):
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        # One collection per embedding model, so switching models never mixes vector spaces.
        self.collection = f"{settings.collection_base}__{re.sub(r'[^a-z0-9]+', '-', self.embedder.model.lower())}"
        self.bm25: BM25 | None = None

    def index(self, documents: list[Document]) -> int:
        """Rebuild the collection from scratch. Returns the number of chunks written."""
        chunks, seen = [], set()
        for doc in documents:
            chunk = Chunk.from_document(doc)
            if chunk.text and content_hash(chunk.text) not in seen:
                seen.add(content_hash(chunk.text))
                chunks.append(chunk)

        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        dense = self.embedder.embed_documents(texts)
        self.bm25 = BM25(texts) if self.settings.hybrid_enabled else None
        self.client.create_collection(
            self.collection,
            vectors_config={"dense": models.VectorParams(size=len(dense[0]), distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )

        points = []
        for chunk, vector in zip(chunks, dense):
            vectors = {"dense": vector}
            sparse = self.bm25.encode(chunk.text) if self.bm25 else None
            if sparse and sparse.indices:
                vectors["sparse"] = sparse
            payload = {**asdict(chunk), "category": chunk.metadata.get("category")}
            points.append(models.PointStruct(id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)), vector=vectors, payload=payload))
        batch = self.settings.embedding_batch_size
        for i in range(0, len(points), batch):
            self.client.upsert(self.collection, points[i:i + batch], wait=True)
        logger.info("Indexed %d chunks into %s", len(chunks), self.collection)
        return len(chunks)

    def retrieve(self, query: str, *, filters: dict | None = None, top_k: int | None = None) -> list[ScoredChunk]:
        """Best chunks for ``query`` that match every payload filter (a list value matches any of its items)."""
        if not self.client.collection_exists(self.collection):
            return []
        s = self.settings
        conditions = [
            models.FieldCondition(key=key, match=models.MatchAny(any=list(value)) if isinstance(value, (list, tuple, set))
                                  else models.MatchValue(value=value))
            for key, value in (filters or {}).items()
        ]
        query_filter = models.Filter(must=conditions) if conditions else None

        hits = self._search("dense", self.embedder.embed_query(query), query_filter, s.score_threshold)
        if self.bm25:
            sparse = self.bm25.encode(query, query=True)
            sparse_hits = self._search("sparse", sparse, query_filter) if sparse.indices else []
            hits = rrf([hits, sparse_hits], s.rrf_k)[:s.candidate_top_k]
        if s.rerank_enabled:
            hits = lexical_rerank(query, hits)
        return select_context(hits[:top_k or s.top_k], s.max_context_tokens, s.dedup_similarity_threshold)

    def _search(self, using: str, vector, query_filter, score_threshold=None) -> list[ScoredChunk]:
        res = self.client.query_points(
            self.collection, query=vector, using=using, limit=self.settings.candidate_top_k,
            query_filter=query_filter, score_threshold=score_threshold, with_payload=True,
        )
        out = []
        for p in res.points:
            payload = dict(p.payload)
            payload.pop("category", None)
            out.append(ScoredChunk(Chunk(**payload), p.score, **{f"{using}_score": p.score}))
        return out


def rrf(ranked_lists: list[list[ScoredChunk]], k: int = 60) -> list[ScoredChunk]:
    """Reciprocal rank fusion. Keeps each chunk's dense/sparse scores from whichever list had them."""
    scores: dict[str, float] = {}
    best: dict[str, ScoredChunk] = {}
    for ranked in ranked_lists:
        for rank, sc in enumerate(ranked):
            cid = sc.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            if cid in best:
                prev = best[cid]
                sc = replace(prev, dense_score=prev.dense_score if prev.dense_score is not None else sc.dense_score,
                             sparse_score=prev.sparse_score if prev.sparse_score is not None else sc.sparse_score)
            best[cid] = sc
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [replace(best[c], score=scores[c], fused_score=scores[c]) for c in ordered]


def lexical_rerank(query: str, hits: list[ScoredChunk]) -> list[ScoredChunk]:
    """Re-order by cosine similarity of term counts; cheap and needs no model."""
    q = Counter(tokenize(query))

    def cosine(text: str) -> float:
        d = Counter(tokenize(text))
        if not q or not d:
            return 0.0
        dot = sum(q[t] * d[t] for t in q.keys() & d.keys())
        return dot / (math.sqrt(sum(v * v for v in q.values())) * math.sqrt(sum(v * v for v in d.values())))

    scored = sorted(((cosine(h.chunk.text), h) for h in hits), key=lambda x: x[0], reverse=True)
    return [replace(h, score=s, rerank_score=s) for s, h in scored]


def select_context(hits: list[ScoredChunk], max_tokens: int, dedup_threshold: float) -> list[ScoredChunk]:
    """Highest score first, skipping near-duplicates (token Jaccard) and anything over the token budget."""
    selected, kept, used = [], [], 0
    for h in sorted(hits, key=lambda h: h.score, reverse=True):
        tokens = set(tokenize(h.chunk.text))
        if any(tokens and k and len(tokens & k) / len(tokens | k) >= dedup_threshold for k in kept):
            continue
        cost = max(1, len(h.chunk.text) // 4)   # ~4 chars per token
        if used + cost <= max_tokens:
            selected.append(h)
            kept.append(tokens)
            used += cost
    return selected


def grounding_gate(chunks: list[ScoredChunk], min_score: float) -> tuple[bool, float]:
    """(confident enough to answer?, best similarity).

    Uses the dense cosine rather than ``score``: after RRF ``score`` is a tiny
    reciprocal-rank value whose scale says nothing about relevance.
    """
    def signal(c: ScoredChunk) -> float:
        if c.dense_score is not None:
            return c.dense_score
        if c.rerank_score is not None:
            return c.rerank_score
        return c.score

    confidence = max(map(signal, chunks), default=0.0)
    return bool(chunks) and confidence >= min_score, confidence
