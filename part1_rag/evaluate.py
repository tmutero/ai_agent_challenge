"""
Retrieval evaluation: labelled questions -> Recall/MRR/NDCG for several pipeline
configurations, plus the same cases run through the query planner + routing.

    python -m part1_rag.evaluate [cases.json] [data_dir]   # writes docs/eval_results.md

Cases are [{"query": ..., "relevant": [doc_id, ...]}] labelled against the documents in
``data_dir`` (default: the loader's DATA_DIR, i.e. RAG_DATA_DIR or the bundled samples).

Used to *choose* the shipped configuration rather than assert it.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

from rag_core.pipeline.evaluation import EvalCase, evaluate, hit_rate_at_k, reciprocal_rank

from .engine import RagEngine, default_settings
from .loaders import DATA_DIR, load_knowledge_base
from .query_transform import plan_query, validate_plan

CASES_PATH = Path(__file__).parent.parent / "sample_data" / "eval" / "cases.json"
OUT = Path(__file__).parent.parent / "docs" / "eval_results.md"

CONFIGS = {
    "dense only": dict(hybrid_enabled=False, rerank_enabled=False),
    "hybrid (dense+BM25, RRF)": dict(rerank_enabled=False),
    "hybrid + lexical rerank": dict(rerank_enabled=True),
}


def load_cases(path: Path = CASES_PATH) -> list[EvalCase]:
    raw = json.loads(Path(path).read_text())
    return [EvalCase(query=c["query"], relevant_ids=c["relevant"], match_on="doc_id") for c in raw]


class _PlannedPipeline:
    """Adapter so evaluate() can score planner + routing end to end."""

    def __init__(self, engine: RagEngine):
        self.engine = engine

    def retrieve(self, query, filters=None, top_k=5):
        plan, _ = validate_plan(plan_query(query, [], self.engine.kb), self.engine.kb)
        if plan.intent == "aggregate" and plan.aggregation:
            from .aggregate import run_aggregation
            ids = run_aggregation(plan.aggregation, self.engine.kb).rows["_chunk_id"].tolist()
            chunks = [_Fake(self.engine._docs[i]) for i in ids]
        else:
            queries = plan.sub_queries if plan.intent == "compare" and plan.sub_queries else [plan.standalone_query]
            chunks = self.engine._retrieve_all(queries, self.engine._routes(plan))
        return chunks


class _Fake:
    def __init__(self, doc):
        self.chunk = type("C", (), {"doc_id": doc.doc_id, "chunk_id": f"{doc.doc_id}::0"})()


def main(cases_path: Path = CASES_PATH, data_dir: Path = DATA_DIR, k: int = 5) -> None:
    cases = load_cases(cases_path)
    kb = load_knowledge_base(Path(data_dir))
    rows = []
    engine = None
    for name, overrides in CONFIGS.items():
        engine = RagEngine(replace(default_settings(), **overrides), kb)
        metrics = evaluate(engine.pipeline, cases, k=k)["metrics"]
        rows.append((name, metrics))
        print(name, metrics)

    shipped = RagEngine(default_settings(), kb)
    planned = evaluate(_PlannedPipeline(shipped), cases, k=k)
    rows.append(("shipped config + query planner/routing", planned["metrics"]))
    print("planned", planned["metrics"])

    metric_names = list(rows[0][1])
    lines = [f"# Retrieval evaluation ({len(cases)} labelled questions, k={k})", "",
             "| Configuration | " + " | ".join(metric_names) + " |",
             "|---|" + "---:|" * len(metric_names)]
    lines += [f"| {n} | " + " | ".join(f"{m[x]:.3f}" for x in metric_names) + " |" for n, m in rows]
    misses = sum(1 for c in planned["per_case"] if c["hit_rate"] == 0)
    lines += ["", "Per-case results for the shipped configuration:", "",
              "| Question | Hit@k | Reciprocal rank |", "|---|---:|---:|"]
    for c, case in zip(planned["per_case"], cases):
        hit = hit_rate_at_k(c["retrieved"], set(case.relevant_ids), k)
        rr = reciprocal_rank(c["retrieved"], set(case.relevant_ids), k)
        lines.append(f"| {case.query} | {hit:.0f} | {rr:.2f} |")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({misses} misses)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
