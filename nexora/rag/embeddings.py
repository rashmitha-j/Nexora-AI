"""Embedding clients. Documents and queries are embedded with different task types
where the provider supports it, which improves retrieval quality."""

from __future__ import annotations

import asyncio
from typing import Protocol

from nexora.config import Settings


class Embedder(Protocol):
    model: str
    dim: int

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class GeminiEmbedder:
    batch_size = 100

    def __init__(self, api_key: str, model: str, dim: int, timeout_s: int):
        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_s * 1000))
        self.model = model
        self.dim = dim

    async def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        config = self._types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim)
        for i in range(0, len(texts), self.batch_size):
            result = await self._client.aio.models.embed_content(
                model=self.model, contents=texts[i : i + self.batch_size], config=config
            )
            vectors.extend(list(e.values) for e in result.embeddings)
        return vectors

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts, "RETRIEVAL_DOCUMENT")

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([text], "RETRIEVAL_QUERY"))[0]


class OpenAIEmbedder:
    batch_size = 256

    def __init__(self, api_key: str, model: str, dim: int, timeout_s: int):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout_s)
        self.model = model
        self.dim = dim

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        batches = [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        results = await asyncio.gather(
            *(self._client.embeddings.create(model=self.model, input=b, dimensions=self.dim) for b in batches)
        )
        return [d.embedding for r in results for d in r.data]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]


def build_embedder(settings: Settings) -> Embedder:
    cls = GeminiEmbedder if settings.embedding_provider == "gemini" else OpenAIEmbedder
    return cls(
        settings.api_keys[settings.embedding_provider],
        settings.embedding_model,
        settings.embedding_dim,
        settings.llm_timeout_s,
    )
