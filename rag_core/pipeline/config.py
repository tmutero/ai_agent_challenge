from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineSettings:
    qdrant_url: str = ":memory:"
    qdrant_api_key: str | None = None
    collection_base: str = "rag_documents"

    embedding_provider: str = "openai"   # openai | fake
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536            # only used by the fake embedder
    embedding_batch_size: int = 64

    hybrid_enabled: bool = True          # dense + BM25, fused with RRF
    rrf_k: int = 60
    top_k: int = 5
    candidate_top_k: int = 30            # retrieved before reranking / cutting to top_k
    score_threshold: float | None = None
    rerank_enabled: bool = True          # lexical-overlap rerank of the candidates

    max_context_tokens: int = 6000
    dedup_similarity_threshold: float = 0.92
    min_grounding_score: float = 0.25    # best dense cosine below this -> refuse
