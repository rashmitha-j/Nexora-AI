"""Ingest, retrieve and format grounded context with numbered citations."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from nexora.rag.embeddings import Embedder
from nexora.rag.ingest import chunk_sections, extract_sections
from nexora.rag.store import DocumentStore, Hit

MAX_CHUNKS_PER_DOCUMENT = 2000


class DocumentTooLargeError(ValueError):
    pass


@dataclass
class IngestResult:
    filename: str
    document_id: str
    chunk_count: int
    replaced: bool


class RagService:
    def __init__(self, store: DocumentStore, embedder: Embedder, top_k: int, min_score: float = 0.55):
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        # Cosine similarity below this is treated as irrelevant. With gemini-embedding-001 at
        # 768 dims, unrelated questions measured 0.45-0.53 and real matches 0.59-0.79.
        self.min_score = min_score

    async def ingest(
        self, owner: str, filename: str, path: Path, max_chunks: int = MAX_CHUNKS_PER_DOCUMENT
    ) -> IngestResult:
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        existing = await self.store.find_by_hash(owner, sha256)
        chunks = chunk_sections(extract_sections(path, filename))
        if len(chunks) > max_chunks:
            raise DocumentTooLargeError(
                f"{filename} is too large to index here ({len(chunks)} sections; the limit is {max_chunks})."
            )
        embeddings = await self.embedder.embed_documents([c.text for c in chunks])
        # Re-uploading the same file re-indexes it instead of duplicating it.
        if existing:
            await self.store.delete_document(owner, existing.id)
        doc_id = await self.store.add_document(owner, filename, sha256, self.embedder.model, chunks, embeddings)
        return IngestResult(filename, doc_id, len(chunks), replaced=existing is not None)

    async def reindex(self, owner: str) -> int:
        """Re-embed all of an owner's stored chunks, e.g. after changing EMBEDDING_MODEL."""
        rows = await self.store.chunks_for_owner(owner)
        if not rows:
            return 0
        vectors = await self.embedder.embed_documents([text for _, text in rows])
        await self.store.update_embeddings(
            owner, self.embedder.model, [(cid, v) for (cid, _), v in zip(rows, vectors, strict=True)]
        )
        return len(rows)

    async def retrieve(self, owner: str, query: str) -> list[Hit]:
        vector = await self.embedder.embed_query(query)
        hits = await self.store.search(owner, vector, self.top_k)
        return [h for h in hits if h.score >= self.min_score]


_TAG = re.compile(r"</?\s*document[^>]*>", re.IGNORECASE)


def format_context(hits: list[Hit]) -> str:
    """Wrap retrieved chunks in <document> tags, stripping any tags inside the text so a
    document can't close its own wrapper and pose as instructions."""
    blocks = []
    for n, hit in enumerate(hits, start=1):
        where = f"{hit.filename}" + (f", page {hit.page}" if hit.page else "")
        body = _TAG.sub("", hit.content)
        blocks.append(f'<document index="{n}" source="{_TAG.sub("", where)}">\n{body}\n</document>')
    return "\n\n".join(blocks)


def format_sources(hits: list[Hit]) -> str:
    lines = []
    for n, hit in enumerate(hits, start=1):
        page = f", p. {hit.page}" if hit.page else ""
        lines.append(f"[{n}] {hit.filename}{page} (relevance {hit.score:.2f})")
    return "\n".join(lines)
