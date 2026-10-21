"""
Query transformation: turn a raw (possibly follow-up, possibly vague) user message
into an explicit, validated retrieval plan.

The LLM proposes; deterministic code disposes:
  * the LLM rewrites the message into a standalone query (resolving "it",
    "that one", "what about the second?" against chat history -- context retention),
    classifies intent, routes to datasets, extracts metadata filters, and for
    comparisons splits into sub-queries;
  * ``validate_plan`` then checks every dataset/column/value against the real
    catalog and *drops* anything that does not exist, recording why. A
    hallucinated filter value would otherwise silently return zero results.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from common.llm import structured

from .loaders import KnowledgeBase

Intent = Literal["lookup", "compare", "aggregate", "catalog", "out_of_scope"]


class FieldFilter(BaseModel):
    dataset: str
    column: str
    values: list[str] = Field(description="one or more exact allowed values; a row matches if it has any of them")


class Aggregation(BaseModel):
    dataset: str
    operation: Literal["min", "max", "mean", "sum", "count", "list"]
    column: str | None = Field(description="numeric column for min/max/mean/sum; null for count/list")
    filters: list[FieldFilter]
    limit: int | None = Field(description="for min/max: how many rows to return (e.g. 3 cheapest); null = 1")
    return_columns: list[str] = Field(description="extra columns to show for each returned row")


class QueryPlan(BaseModel):
    standalone_query: str = Field(description="self-contained rewrite of the user's message")
    intent: Intent
    datasets: list[str]
    filters: list[FieldFilter]
    sub_queries: list[str] = Field(description="for compare: one focused query per entity; else empty")
    aggregation: Aggregation | None
    reasoning: str = Field(description="one sentence: why this intent/routing")


SYSTEM = """You plan retrieval for a question-answering assistant over a knowledge base of documents the user
uploaded (tables from CSV/Excel/JSON/PDF/Word files, plus prose paragraphs).

Knowledge base catalog:
{catalog}

Produce a QueryPlan:
- standalone_query: rewrite the latest user message so it is fully self-contained, resolving pronouns and
  references ("it", "that one", "what about the second one") using the conversation history. Expand an
  abbreviation only when the catalog or conversation makes its meaning clear; keep the user's own terms too.
- intent:
  * lookup     - facts about specific entities or rules
  * compare    - two or more named entities side by side; fill sub_queries with one query per entity
  * aggregate  - ranking / extremes / counting / averages / sums across rows of ONE table ("cheapest",
                 "highest", "how many", "average", "which X are ..."). Fill `aggregation`.
  * catalog    - questions about what data/documents exist
  * out_of_scope - anything the catalog cannot answer (predictions, outside facts, unrelated topics)
- datasets: the dataset or text-source names from the catalog most likely to contain the answer.
- filters: ONLY when the user explicitly constrains a column listed with "allowed values", and ONLY using
  exact allowed values. Never invent values. Leave empty when unsure. Use several values for follow-ups about a
  set of entities named earlier (e.g. "which of those ..." -> the name values of the entities just discussed).
- aggregation: for aggregate intent only, else null. Use exact dataset and column names from the catalog.
"""


def plan_query(question: str, history: list[tuple[str, str]], kb: KnowledgeBase) -> QueryPlan:
    convo = "\n".join(f"User: {u}\nAssistant: {a[:400]}" for u, a in history[-3:]) or "(no prior turns)"
    user = f"Conversation so far:\n{convo}\n\nLatest user message: {question}"
    return structured(SYSTEM.format(catalog=kb.catalog()), user, QueryPlan)


def validate_plan(plan: QueryPlan, kb: KnowledgeBase) -> tuple[QueryPlan, list[str]]:
    """Drop anything in the plan that doesn't exist in the catalog. Returns the
    cleaned plan and a list of human-readable corrections for the trace."""
    notes: list[str] = []
    known = set(kb.datasets) | set(kb.text_sources)

    datasets = [d for d in plan.datasets if d in known]
    for d in set(plan.datasets) - known:
        notes.append(f"dropped unknown dataset '{d}'")

    def ok_filter(f: FieldFilter) -> bool:
        ds = kb.datasets.get(f.dataset)
        allowed = ds.filterable.get(f.column) if ds else None
        if allowed is None:
            notes.append(f"dropped filter {f.dataset}.{f.column}: not a filterable column")
            return False
        bad = [v for v in f.values if v not in allowed]
        for v in bad:
            notes.append(f"dropped filter value {f.column}='{v}': not an allowed value")
        f.values = [v for v in f.values if v in allowed]
        return bool(f.values)

    filters = [f for f in plan.filters if ok_filter(f)]

    agg = plan.aggregation
    if agg is not None:
        ds = kb.datasets.get(agg.dataset)
        if ds is None:
            notes.append(f"aggregation dataset '{agg.dataset}' unknown; falling back to retrieval")
            agg = None
        else:
            cols = set(ds.frame.columns)
            if agg.column and agg.column not in cols:
                notes.append(f"aggregation column '{agg.column}' not in {agg.dataset}; falling back to retrieval")
                agg = None
            else:
                agg = agg.model_copy(update={
                    "filters": [f for f in agg.filters if ok_filter(f)],
                    "return_columns": [c for c in agg.return_columns if c in cols],
                })

    intent = plan.intent
    if intent == "aggregate" and agg is None:
        intent = "lookup"
        notes.append("intent downgraded aggregate -> lookup (no valid aggregation)")

    cleaned = plan.model_copy(update={"datasets": datasets, "filters": filters, "aggregation": agg, "intent": intent})
    return cleaned, notes
