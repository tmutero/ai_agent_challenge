"""
Part 1 orchestration: question -> plan -> (retrieve | aggregate) -> grounded answer -> verification.

    question + history
        -> plan_query (LLM, structured)  -> validate_plan (deterministic)
        -> intent routing:
             lookup / catalog -> hybrid retrieval with metadata pre-filters
             compare          -> one retrieval per sub-query, merged
             aggregate        -> pandas over the full table (+ the cited rows)
             out_of_scope     -> polite refusal, no retrieval
        -> evidence gate (dense cosine >= MIN_GROUNDING_SCORE, else unfiltered retry, else refuse)
        -> generate (LLM, structured: answer + atomic facts)
        -> verify_facts (deterministic: each value must occur in its cited passage)

Everything the UI shows in the "How this answer was produced" panel comes from
``RagResponse`` -- nothing is hidden from the reviewer.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from functools import partial

from rag_core.pipeline import HybridIndex, PipelineSettings, ScoredChunk, grounding_gate
from rag_core.pipeline.embeddings import make_embedder
from rag_core.pipeline.types import content_hash

from common.llm import client, secret

from . import answerer
from .aggregate import AggregationResult, run_aggregation
from .answerer import FactCheck, GroundedAnswer, Passage
from .loaders import KnowledgeBase, field_key, load_knowledge_base
from .query_transform import QueryPlan, plan_query, validate_plan

NO_DOCUMENTS = "No documents are loaded yet - upload some in the Documents tab, or load the sample documents."
MAX_AGGREGATION_CITATIONS = 15   # source rows quoted alongside a computed result


class CachingEmbedder:
    """Process-wide cache in front of the embedder, keyed by text hash, so
    re-indexing after an upload only embeds chunks that are actually new."""

    _cache: dict[str, list[float]] = {}

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):                 # model
        return getattr(self.inner, name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [f"{self.inner.model}:{content_hash(t)}" for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in self._cache]
        if missing:
            for (k, _), vec in zip(missing, self.inner.embed_documents([t for _, t in missing])):
                self._cache[k] = vec
        return [self._cache[k] for k in keys]

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(text)


def default_settings() -> PipelineSettings:
    return PipelineSettings(
        qdrant_url=":memory:",
        collection_base="knowledge_base",
        embedding_provider=secret("EMBEDDING_PROVIDER", "openai"),
        embedding_model="text-embedding-3-small",
        embedding_dim=1536,
        top_k=6,
        candidate_top_k=20,
        rerank_enabled=False,             # see eval: lexical rerank hurt row-level recall
        min_grounding_score=0.30,
    )


@dataclass
class RagResponse:
    question: str
    plan: QueryPlan | None = None
    plan_notes: list[str] = field(default_factory=list)
    routes: list[dict] = field(default_factory=list)   # one payload filter per routed dataset
    retrieved: list[ScoredChunk] = field(default_factory=list)
    aggregation: AggregationResult | None = None
    passages: list[Passage] = field(default_factory=list)
    answer: GroundedAnswer | None = None
    fact_checks: list[FactCheck] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)   # numbers in prose found in no passage
    confidence: float = 0.0
    refused: str | None = None
    timings_ms: dict[str, int] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.refused or (self.answer.answer if self.answer else "")


class RagEngine:
    def __init__(self, settings: PipelineSettings | None = None, kb: KnowledgeBase | None = None):
        self.settings = settings or default_settings()
        if self.settings.embedding_provider == "openai":
            client()   # copies OPENAI_API_KEY from Streamlit secrets into the env the embedder reads
        self.kb = kb if kb is not None else load_knowledge_base()
        self.pipeline = HybridIndex(self.settings, embedder=CachingEmbedder(make_embedder(self.settings)))
        self.indexed_chunks = 0 if self.kb.empty else self.pipeline.index(self.kb.documents)
        self._gate = partial(grounding_gate, min_score=self.settings.min_grounding_score)
        self._docs = {f"{d.doc_id}::0": d for d in self.kb.documents}

    # --- retrieval ---------------------------------------------------------------

    def _routes(self, plan: QueryPlan) -> list[dict]:
        """One payload filter per routed dataset. A column filter only constrains its
        own dataset, so a question that filters one table can still search the others."""
        if plan.intent == "catalog":
            return [{"category": "catalog"}]
        def category(name: str) -> str:
            ds = self.kb.datasets.get(name)
            return ds.category if ds else name

        routes: dict[str, dict] = {}
        for name in [*plan.datasets, *(f.dataset for f in plan.filters)]:
            routes.setdefault(category(name), {"category": category(name)})
            # Tables usually need their surrounding prose (definitions, exceptions) too.
            ds = self.kb.datasets.get(name)
            for t in self.kb.text_sources.values():
                if ds and t.source == ds.source:
                    routes.setdefault(t.name, {"category": t.name})
        for flt in plan.filters:
            value = flt.values[0] if len(flt.values) == 1 else flt.values   # list -> MatchAny
            routes[category(flt.dataset)][f"metadata.fields.{field_key(flt.column)}"] = value
        return list(routes.values()) or [{}]

    def retrieve(self, query: str, filters: dict, top_k: int | None = None) -> list[ScoredChunk]:
        return self.pipeline.retrieve(query, filters=filters, top_k=top_k)

    # --- main entry ------------------------------------------------------------------

    def answer(self, question: str, history: list[tuple[str, str]] | None = None) -> RagResponse:
        r = RagResponse(question=question)
        if self.kb.empty:
            r.refused = NO_DOCUMENTS
            return r
        t0 = time.perf_counter()
        raw_plan = plan_query(question, history or [], self.kb)
        r.plan, r.plan_notes = validate_plan(raw_plan, self.kb)
        r.timings_ms["plan"] = int((time.perf_counter() - t0) * 1000)
        plan = r.plan

        if plan.intent == "out_of_scope":
            r.refused = ("That question is outside what the loaded documents cover "
                         f"({', '.join(self.kb.sources)}).")
            return r

        t0 = time.perf_counter()
        if plan.intent == "aggregate" and plan.aggregation:
            r.aggregation = run_aggregation(plan.aggregation, self.kb)
            r.passages.append(Passage(f"computed from {self.kb.datasets[plan.aggregation.dataset].source}",
                                      r.aggregation.as_evidence()))
            for cid in r.aggregation.rows["_chunk_id"].head(MAX_AGGREGATION_CITATIONS):
                doc = self._docs[cid]
                r.passages.append(Passage(f"{doc.source} - {doc.metadata['record_ref']}", doc.text, cid))
            r.confidence = 1.0
        else:
            r.routes = self._routes(plan)
            queries = plan.sub_queries if plan.intent == "compare" and plan.sub_queries else [plan.standalone_query]
            chunks = self._retrieve_all(queries, r.routes)
            ok, r.confidence = self._gate(chunks)
            if not ok and r.routes != [{}]:
                r.plan_notes.append(f"routed retrieval below grounding threshold "
                                    f"({r.confidence:.2f}); retried across all datasets")
                r.routes = [{}]
                chunks = self._retrieve_all(queries, r.routes)
                ok, r.confidence = self._gate(chunks)
            r.retrieved = chunks
            if not ok:
                r.refused = (f"I couldn't find sufficiently relevant information to answer that "
                             f"(best match similarity {r.confidence:.2f} < {self.settings.min_grounding_score}).")
                r.timings_ms["retrieve"] = int((time.perf_counter() - t0) * 1000)
                return r
            r.passages = [Passage(f"{c.chunk.source} - {c.chunk.metadata.get('record_ref', c.chunk.section)}",
                                  c.chunk.text, c.chunk.chunk_id) for c in chunks]
        r.timings_ms["retrieve"] = int((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        r.answer = answerer.generate(plan.standalone_query, r.passages)
        r.fact_checks = answerer.verify_facts(r.answer, r.passages)
        r.unsupported_numbers = answerer.unsupported_numbers(r.answer.answer, r.passages, f"{question} {plan.standalone_query}")
        r.timings_ms["generate"] = int((time.perf_counter() - t0) * 1000)
        return r

    def _retrieve_all(self, queries: list[str], routes: list[dict]) -> list[ScoredChunk]:
        """Fan out over (sub-query x route); split the top_k budget so every
        route/entity is represented, then merge by chunk id."""
        k = max(2, math.ceil(self.settings.top_k / (len(queries) * len(routes))))
        seen, merged = set(), []
        for q in queries:
            for filters in routes:
                for c in self.retrieve(q, filters, k):
                    if c.chunk.chunk_id not in seen:
                        seen.add(c.chunk.chunk_id)
                        merged.append(c)
        return merged
