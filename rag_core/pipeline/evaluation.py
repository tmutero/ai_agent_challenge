"""Retrieval metrics @k against labelled cases. ``relevant_ids`` are doc_ids or chunk_ids (``match_on``)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class EvalCase:
    query: str
    relevant_ids: list[str]
    match_on: str = "doc_id"   # doc_id | chunk_id
    filters: dict = field(default_factory=dict)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / k if k else 0.0


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str], k: int) -> float:
    return next((1.0 / i for i, rid in enumerate(retrieved[:k], start=1) if rid in relevant), 0.0)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, rid in enumerate(retrieved[:k], start=1) if rid in relevant)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / idcg if idcg else 0.0


METRICS = {
    "recall": recall_at_k,
    "precision": precision_at_k,
    "hit_rate": hit_rate_at_k,
    "mrr": reciprocal_rank,
    "ndcg": ndcg_at_k,
}


def evaluate(pipeline, cases: list[EvalCase], k: int = 5) -> dict:
    """Run every case through ``pipeline.retrieve`` -> {"metrics": means, "per_case": rows}."""
    totals = dict.fromkeys(METRICS, 0.0)
    per_case = []
    for case in cases:
        chunks = pipeline.retrieve(case.query, filters=case.filters, top_k=k)
        retrieved = [getattr(c.chunk, case.match_on) for c in chunks]
        row = {"query": case.query, "retrieved": retrieved[:k]}
        for name, fn in METRICS.items():
            value = fn(retrieved, set(case.relevant_ids), k)
            totals[name] += value
            row[name] = round(value, 4)
        per_case.append(row)
    n = len(cases) or 1
    return {"metrics": {name: round(total / n, 4) for name, total in totals.items()}, "per_case": per_case}
