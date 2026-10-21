from __future__ import annotations

import hashlib
import math
import os

from .config import PipelineSettings


class OpenAIEmbedder:
    def __init__(self, settings: PipelineSettings):
        from openai import OpenAI   # lazy, so offline tests don't need a key

        self.model = settings.embedding_model
        self.batch_size = settings.embedding_batch_size
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), max_retries=4)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out = []
        for i in range(0, len(texts), self.batch_size):
            result = self._client.embeddings.create(model=self.model, input=texts[i:i + self.batch_size])
            out += [item.embedding for item in result.data]
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class FakeEmbedder:
    """Deterministic hashed bag-of-words vectors, so tests run without a network."""

    def __init__(self, settings: PipelineSettings):
        self.dim = settings.embedding_dim or 256
        self.model = f"fake-{self.dim}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in text.lower().split():
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def make_embedder(settings: PipelineSettings):
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(settings)
    if settings.embedding_provider == "fake":
        return FakeEmbedder(settings)
    raise ValueError(f"Unknown embedding provider: {settings.embedding_provider!r}")
