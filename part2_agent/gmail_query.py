"""
A small Gmail search-syntax evaluator for the offline fixtures.

The agent writes real Gmail queries ("subject:(rsvp OR registration) after:2026/09/01",
'"Cape Town" -newsletter'); the fixture mailbox must interpret them the way Gmail
would, or the offline run silently diverges from the live one. Supports implicit
AND, OR, parentheses, quoted phrases, negation, and from:/to:/subject:/after:/before:
(including field:(a OR b) groups). Unsupported operators are treated as no-ops.
"""

from __future__ import annotations

import email.utils
import re
from datetime import datetime

_TOKEN = re.compile(r'\(|\)|-?[\w.]+:"[^"]*"|-?"[^"]*"|[^\s()]+')
_FIELD_GROUP = re.compile(r'(-?)(\w+):\(([^)]*)\)')
_NOOP = {"in", "is", "label", "newer_than", "older_than", "category", "larger", "smaller"}


def _expand_field_groups(q: str) -> str:
    """subject:(a OR "b c") -> (subject:a OR subject:"b c")"""
    def repl(m):
        neg, fld, inner = m.groups()
        parts = [t if t.upper() in ("OR", "AND") else f"{neg}{fld}:{t}" for t in re.findall(r'"[^"]*"|\S+', inner)]
        return "(" + " ".join(parts) + ")"
    return _FIELD_GROUP.sub(repl, q)


def parse(query: str):
    toks = _TOKEN.findall(_expand_field_groups(query))

    def and_expr():
        items = []
        while toks and toks[0] != ")":
            if toks[0].upper() == "AND":
                toks.pop(0)
                continue
            items.append(or_expr())
        return ("and", items)

    def or_expr():
        alts = [atom()]
        while toks and toks[0].upper() == "OR":
            toks.pop(0)
            if toks and toks[0] != ")":
                alts.append(atom())
        return alts[0] if len(alts) == 1 else ("or", alts)

    def atom():
        t = toks.pop(0)
        if t == "(":
            node = and_expr()
            if toks and toks[0] == ")":
                toks.pop(0)
            return node
        neg = t.startswith("-") and len(t) > 1
        t = t[1:] if neg else t
        fld, _, val = t.partition(":") if re.match(r"^[\w.]+:", t) else ("", "", t)
        return ("term", fld.lower(), val.strip('"').lower(), neg)

    tree = and_expr()
    return tree


def _date(raw: str) -> datetime | None:
    for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def matches(node, doc: dict) -> bool:
    """``doc`` has lower-cased 'from', 'to', 'subject', 'text' and a naive datetime 'date' (may be None)."""
    kind = node[0]
    if kind == "and":
        return all(matches(n, doc) for n in node[1])
    if kind == "or":
        return any(matches(n, doc) for n in node[1])
    _, fld, val, neg = node
    if fld in ("from", "to", "subject"):
        hit = val in doc.get(fld, "")
    elif fld in ("after", "before"):
        d, md = _date(val), doc.get("date")
        hit = True if d is None or md is None else (md >= d if fld == "after" else md < d)
    elif fld == "has":
        hit = False                                   # fixtures have no attachments
    elif fld in _NOOP:
        hit = True
    else:
        hit = (f"{fld}:{val}" if fld else val) in doc["text"]   # unknown operator: literal text
    return hit != neg


def message_doc(m: dict) -> dict:
    try:
        date = email.utils.parsedate_to_datetime(m["date"]).replace(tzinfo=None)
    except (TypeError, ValueError):
        date = None
    low = {k: (m.get(k) or "").lower() for k in ("from", "to", "subject", "body")}
    return {**low, "date": date, "text": " ".join(low.values())}


def search(query: str, docs: list[dict]) -> list[int]:
    tree = parse(query)
    return [i for i, d in enumerate(docs) if matches(tree, d)]
