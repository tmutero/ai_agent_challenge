"""
Grounded, structured answer generation + post-hoc citation verification.

The model must return (a) a markdown answer with [n] citations and (b) a list of
atomic ``facts`` -- each a value copied from a specific numbered passage. We then
*check* every fact deterministically: does that value actually occur in the passage
it cites? Facts that fail are surfaced as unverified rather than trusted. That
turns "the model says it's grounded" into "we verified it's grounded".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from common.llm import structured


class Fact(BaseModel):
    claim: str = Field(description="short statement, e.g. 'Product X unit price'")
    value: str = Field(description="the value exactly as written in the passage, e.g. '0.55'")
    passage: int = Field(description="number of the passage the value is copied from")


class GroundedAnswer(BaseModel):
    answer: str = Field(description="markdown answer with [n] citations after every factual claim")
    facts: list[Fact]
    insufficient_evidence: bool


SYSTEM = """You answer questions using ONLY the numbered passages provided (extracts from the user's uploaded documents).

Rules:
- Every factual statement must cite its passage(s) as [n]. Never use outside knowledge.
- Copy numbers exactly as they appear in the passages (do not round or convert units). Add units or formatting for
  readability (e.g. "0.55%", "10,000") but list the raw value in `facts`.
- For each number or key value you state, add one entry to `facts` with the exact value and the passage it came from.
- If a value is "not reported", say it is not available - never estimate it.
- When comparing two or more entities, present a markdown table, then one sentence of interpretation.
- If a "Computed result" passage is present, it was calculated deterministically from the full dataset: trust it for
  rankings/extremes/averages and cite the row passages for the underlying values.
- If the passages do not contain the answer, set insufficient_evidence=true and say what is missing.
"""


@dataclass
class Passage:
    label: str      # e.g. "products.csv - row 3"
    text: str
    chunk_id: str | None = None


@dataclass
class FactCheck:
    fact: Fact
    verified: bool
    reason: str


def build_prompt(question: str, passages: list[Passage]) -> str:
    body = "\n\n".join(f"[{i}] {p.label}\n{p.text}" for i, p in enumerate(passages, start=1))
    return f"Passages:\n{body}\n\nQuestion: {question}"


def generate(question: str, passages: list[Passage]) -> GroundedAnswer:
    return structured(SYSTEM, build_prompt(question, passages), GroundedAnswer)


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _numbers(text: str) -> set[float]:
    out = set()
    for m in _NUM.findall(text):
        try:
            out.add(round(float(m.replace(",", "").rstrip(".")), 6))
        except ValueError:
            continue
    return out


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def verify_facts(answer: GroundedAnswer, passages: list[Passage]) -> list[FactCheck]:
    """A fact is verified if its value occurs in the passage it cites: numerically
    (0.55 == "0.55%", 10000 == "10,000") or, for text, as a normalized substring."""
    checks = []
    for f in answer.facts:
        if not 1 <= f.passage <= len(passages):
            checks.append(FactCheck(f, False, f"cites passage [{f.passage}] which does not exist"))
            continue
        text = passages[f.passage - 1].text
        nums = _numbers(f.value)
        if nums:
            ok = nums <= _numbers(text)
            reason = "number found in cited passage" if ok else f"{sorted(nums - _numbers(text))} not in passage [{f.passage}]"
        else:
            ok = _norm(f.value) in _norm(text)
            reason = "text found in cited passage" if ok else f"'{f.value}' not in passage [{f.passage}]"
        checks.append(FactCheck(f, ok, reason))
    return checks


def cited_numbers(answer_text: str) -> set[int]:
    return {int(n) for n in re.findall(r"\[(\d+)\]", answer_text)}


def unsupported_numbers(answer_text: str, passages: list[Passage], question: str = "") -> list[str]:
    """Numbers stated in the answer prose that occur in NO passage (nor in the
    question itself). Complements verify_facts, which only covers the facts the
    model chose to declare."""
    prose = re.sub(r"\[\d+\]", " ", answer_text)
    support = set().union(_numbers(question), *(_numbers(p.text) for p in passages))
    out = []
    for raw in _NUM.findall(prose):
        try:
            value = round(float(raw.replace(",", "").rstrip(".")), 6)
        except ValueError:
            continue
        if value not in support and raw not in out:
            out.append(raw)
    return out
