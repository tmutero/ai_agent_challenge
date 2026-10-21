"""BM25 for the keyword side of hybrid search, with hashed term ids so there is no vocabulary to store."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from qdrant_client import models

_TOKEN = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _term_id(term: str) -> int:
    return int(hashlib.md5(term.encode()).hexdigest(), 16) % 2**20


class BM25:
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        docs = [tokenize(t) for t in corpus]
        self.avg_len = sum(map(len, docs)) / len(docs) if docs else 0.0
        df = Counter(i for tokens in docs for i in {_term_id(t) for t in tokens})
        n = len(docs)
        self.idf = {i: max(1e-6, math.log((n - f + 0.5) / (f + 0.5) + 1.0)) for i, f in df.items()}

    def encode(self, text: str, *, query: bool = False) -> models.SparseVector:
        tokens = tokenize(text)
        counts = Counter(_term_id(t) for t in tokens)
        avg_len = self.avg_len or len(tokens) or 1.0
        indices, values = [], []
        for i, tf in counts.items():
            idf = self.idf.get(i, 1e-6)
            if query:
                weight = idf * (1.0 + math.log(tf))
            else:
                weight = idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * len(tokens) / avg_len))
            if weight > 0:
                indices.append(i)
                values.append(float(weight))
        return models.SparseVector(indices=indices, values=values)
