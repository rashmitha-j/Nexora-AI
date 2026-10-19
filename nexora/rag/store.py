"""Postgres + pgvector document store. Every query is filtered by owner_id, so a user
can only retrieve, list or delete their own documents."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Protocol

from nexora.rag.ingest import Chunk


@dataclass
class DocumentInfo:
    id: str
    filename: str
    chunk_count: int
    created_at: str


@dataclass
class Hit:
    document_id: str
    filename: str
    chunk_index: int
    page: int | None
    content: str
    score: float


class DocumentStore(Protocol):
    async def ensure_schema(self) -> None: ...

    async def find_by_hash(self, owner: str, sha256: str) -> DocumentInfo | None: ...

    async def add_document(
        self,
        owner: str,
        filename: str,
        sha256: str,
        model: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> str: ...

    async def list_documents(self, owner: str) -> list[DocumentInfo]: ...

    async def delete_document(self, owner: str, document_id: str) -> bool: ...

    async def chunks_for_owner(self, owner: str) -> list[tuple[int, str]]: ...

    async def delete_owner(self, owner: str) -> int: ...

    async def purge_stale(self, owner_prefix: str, max_age_hours: int) -> int: ...

    async def update_embeddings(self, owner: str, model: str, rows: list[tuple[int, list[float]]]) -> None: ...

    async def search(self, owner: str, query_embedding: list[float], k: int) -> list[Hit]: ...


def migrations(dim: int) -> list[str]:
    """Ordered, append-only schema migrations. Never edit an applied entry."""
    return [
        f"""
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS nexora_documents (
            id uuid PRIMARY KEY,
            owner_id text NOT NULL,
            filename text NOT NULL,
            sha256 text NOT NULL,
            embedding_model text NOT NULL,
            chunk_count integer NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            UNIQUE (owner_id, sha256)
        );
        CREATE TABLE IF NOT EXISTS nexora_chunks (
            id bigserial PRIMARY KEY,
            document_id uuid NOT NULL REFERENCES nexora_documents(id) ON DELETE CASCADE,
            owner_id text NOT NULL,
            chunk_index integer NOT NULL,
            page integer,
            content text NOT NULL,
            embedding vector({dim}) NOT NULL
        );
        CREATE INDEX IF NOT EXISTS nexora_chunks_owner_idx ON nexora_chunks (owner_id);
        CREATE INDEX IF NOT EXISTS nexora_chunks_embedding_idx
            ON nexora_chunks USING hnsw (embedding vector_cosine_ops);
        """,
    ]


def _vec(values) -> list[float]:
    """pgvector's asyncpg codec takes plain float lists, so numpy isn't needed."""
    return [float(v) for v in values]


class PgVectorStore:
    """asyncpg-backed store with a small pool. asyncpg works on every platform's event
    loop, and the statement cache is off so Neon's PgBouncer pooler is safe."""

    def __init__(self, database_url: str, dim: int, max_connections: int = 3):
        from nexora.history import asyncpg_dsn

        self._url = asyncpg_dsn(database_url)
        self._dim = dim
        self._max = max_connections
        self._pool = None
        self._lock = asyncio.Lock()

    async def _get_pool(self):
        import asyncpg
        from pgvector.asyncpg import register_vector

        async with self._lock:
            if self._pool is None:
                self._pool = await asyncpg.create_pool(
                    self._url,
                    min_size=1,
                    max_size=self._max,
                    statement_cache_size=0,
                    command_timeout=30,
                    init=register_vector,
                )
        return self._pool

    async def ensure_schema(self) -> None:
        import asyncpg

        # The vector type must exist before pooled connections register its codec.
        conn = await asyncpg.connect(self._url, statement_cache_size=0, timeout=10)
        try:
            async with conn.transaction():
                await conn.execute("CREATE TABLE IF NOT EXISTS nexora_schema_migrations (version integer PRIMARY KEY)")
                # Serialise concurrent startups.
                await conn.execute("SELECT pg_advisory_xact_lock(727274)")
                applied = await conn.fetchval("SELECT coalesce(max(version), 0) FROM nexora_schema_migrations")
                for version, sql in enumerate(migrations(self._dim), start=1):
                    if version > applied:
                        await conn.execute(sql)
                        await conn.execute("INSERT INTO nexora_schema_migrations VALUES ($1)", version)
        finally:
            await conn.close()

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @staticmethod
    def _info(row) -> DocumentInfo:
        return DocumentInfo(str(row["id"]), row["filename"], row["chunk_count"], row["created_at"].isoformat())

    async def find_by_hash(self, owner: str, sha256: str) -> DocumentInfo | None:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            "SELECT id, filename, chunk_count, created_at FROM nexora_documents WHERE owner_id = $1 AND sha256 = $2",
            owner,
            sha256,
        )
        return self._info(row) if row else None

    async def add_document(self, owner, filename, sha256, model, chunks, embeddings) -> str:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings differ in length")
        doc_id = uuid.uuid4()
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "INSERT INTO nexora_documents (id, owner_id, filename, sha256, embedding_model, chunk_count) "
                "VALUES ($1, $2, $3, $4, $5, $6)",
                doc_id,
                owner,
                filename,
                sha256,
                model,
                len(chunks),
            )
            await conn.executemany(
                "INSERT INTO nexora_chunks (document_id, owner_id, chunk_index, page, content, embedding) "
                "VALUES ($1, $2, $3, $4, $5, $6)",
                [(doc_id, owner, c.index, c.page, c.text, _vec(e)) for c, e in zip(chunks, embeddings, strict=True)],
            )
        return str(doc_id)

    async def list_documents(self, owner: str) -> list[DocumentInfo]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            "SELECT id, filename, chunk_count, created_at FROM nexora_documents "
            "WHERE owner_id = $1 ORDER BY created_at",
            owner,
        )
        return [self._info(r) for r in rows]

    async def delete_document(self, owner: str, document_id: str) -> bool:
        pool = await self._get_pool()
        status = await pool.execute(
            "DELETE FROM nexora_documents WHERE owner_id = $1 AND id = $2", owner, uuid.UUID(document_id)
        )
        return status != "DELETE 0"

    async def delete_owner(self, owner: str) -> int:
        pool = await self._get_pool()
        status = await pool.execute("DELETE FROM nexora_documents WHERE owner_id = $1", owner)
        return int(status.split()[-1])

    async def purge_stale(self, owner_prefix: str, max_age_hours: int) -> int:
        """Delete documents of temporary owners (e.g. anonymous sessions) older than the limit."""
        pool = await self._get_pool()
        status = await pool.execute(
            "DELETE FROM nexora_documents "
            "WHERE starts_with(owner_id, $1) AND created_at < now() - make_interval(hours => $2)",
            owner_prefix,
            max_age_hours,
        )
        return int(status.split()[-1])

    async def chunks_for_owner(self, owner: str) -> list[tuple[int, str]]:
        pool = await self._get_pool()
        rows = await pool.fetch("SELECT id, content FROM nexora_chunks WHERE owner_id = $1 ORDER BY id", owner)
        return [(r["id"], r["content"]) for r in rows]

    async def update_embeddings(self, owner: str, model: str, rows: list[tuple[int, list[float]]]) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            await conn.executemany(
                "UPDATE nexora_chunks SET embedding = $1 WHERE id = $2 AND owner_id = $3",
                [(_vec(e), cid, owner) for cid, e in rows],
            )
            await conn.execute("UPDATE nexora_documents SET embedding_model = $1 WHERE owner_id = $2", model, owner)

    async def search(self, owner: str, query_embedding: list[float], k: int) -> list[Hit]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            "SELECT c.document_id, d.filename, c.chunk_index, c.page, c.content, "
            "       1 - (c.embedding <=> $1) AS score "
            "FROM nexora_chunks c JOIN nexora_documents d ON d.id = c.document_id "
            "WHERE c.owner_id = $2 "
            "ORDER BY c.embedding <=> $1 LIMIT $3",
            _vec(query_embedding),
            owner,
            k,
        )
        return [
            Hit(str(r["document_id"]), r["filename"], r["chunk_index"], r["page"], r["content"], float(r["score"]))
            for r in rows
        ]
