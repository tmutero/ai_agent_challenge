"""
Deterministic aggregation over the structured datasets.

Vector search returns the k rows most *similar* to a question; it cannot know
which item is *cheapest* unless that row happens to be retrieved. Aggregate
questions are therefore routed here: the planner emits a constrained spec
(dataset, operation, column, filters) -- never code -- and pandas executes it.
Every result row keeps its ``_chunk_id`` / ``_record_ref`` so the answer cites
the exact source rows, exactly like a retrieval answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .loaders import KnowledgeBase, to_number
from .query_transform import Aggregation


@dataclass
class AggregationResult:
    description: str
    value: float | int | None
    rows: pd.DataFrame                  # the rows that justify the answer
    excluded_missing: list[str] = field(default_factory=list)  # rows skipped for missing values

    def as_evidence(self) -> str:
        cols = [c for c in self.rows.columns if not c.startswith("_")]
        lines = [f"Computed result: {self.description}" + (f" = {self.value}" if self.value is not None else "")]
        for _, r in self.rows.iterrows():
            vals = "; ".join(f"{c}: {r[c]}" for c in cols)
            lines.append(f"- {r['_source']} {r['_record_ref']}: {vals}")
        if self.excluded_missing:
            lines.append(f"Excluded (value not reported): {', '.join(self.excluded_missing)}")
        return "\n".join(lines)


def run_aggregation(spec: Aggregation, kb: KnowledgeBase) -> AggregationResult:
    ds = kb.datasets[spec.dataset]
    df = ds.frame.copy()
    for f in spec.filters:
        df = df[df[f.column].astype(str).isin(f.values)]
    filt = ", ".join(f"{f.column} in {f.values}" for f in spec.filters) or "no filters"
    keep = list(dict.fromkeys([ds.key_column, *spec.return_columns, *( [spec.column] if spec.column else [])]))
    keep += ["_source", "_record_ref", "_chunk_id"]

    if spec.operation in ("count", "list"):
        rows = df[keep]
        desc = f"{spec.operation} of {spec.dataset} rows ({filt})"
        return AggregationResult(desc, len(rows) if spec.operation == "count" else None, rows)

    values = to_number(df[spec.column])
    missing = df.loc[values.isna(), ds.key_column].astype(str).tolist()
    df = df.assign(_v=values).dropna(subset=["_v"])
    if df.empty:
        return AggregationResult(f"{spec.operation}({spec.column}) over {spec.dataset} ({filt}): no rows", None,
                                 df[keep], missing)

    n = spec.limit or 1
    if spec.operation in ("min", "max"):
        # Small candidate sets come back fully ranked so the answer can show the
        # runners-up too ("which of those three...?"); large ones are cut to n.
        ranked = df.sort_values("_v", ascending=spec.operation == "min")
        rows = ranked if len(ranked) <= 5 else ranked.head(n)
        value = float(rows["_v"].iloc[0])
    else:  # mean / sum: every contributing row is evidence
        rows = df
        value = round(float(df["_v"].mean() if spec.operation == "mean" else df["_v"].sum()), 4)
    desc = f"{spec.operation}({spec.column}) over {spec.dataset} ({filt}, {len(df)} rows with values)"
    return AggregationResult(desc, value, rows[keep], missing)
