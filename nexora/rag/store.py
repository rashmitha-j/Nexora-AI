"""Postgres + pgvector document store. Every query is filtered by owner_id, so a user
can only retrieve, list or delete their own documents."""

from __future__ import annotations

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


class PgVectorStore:
    def __init__(self, database_url: str, dim: int):
        self._url = database_url
        self._dim = dim

    async def _connect(self):
        import psycopg
        from pgvector.psycopg import register_vector_async

        conn = await psycopg.AsyncConnection.connect(self._url, connect_timeout=10)
        try:
            await register_vector_async(conn)
        except Exception:
            await conn.close()
            raise
        return conn

    async def ensure_schema(self) -> None:
        import psycopg

        # The vector type must exist before register_vector_async can run.
        async with await psycopg.AsyncConnection.connect(self._url, connect_timeout=10) as conn:
            await conn.execute("CREATE TABLE IF NOT EXISTS nexora_schema_migrations (version integer PRIMARY KEY)")
            # Serialise concurrent startups.
            await conn.execute("SELECT pg_advisory_xact_lock(727274)")
            cur = await conn.execute("SELECT coalesce(max(version), 0) FROM nexora_schema_migrations")
            applied = (await cur.fetchone())[0]
            for version, sql in enumerate(migrations(self._dim), start=1):
                if version > applied:
                    await conn.execute(sql)
                    await conn.execute("INSERT INTO nexora_schema_migrations VALUES (%s)", (version,))
            await conn.commit()

    async def find_by_hash(self, owner: str, sha256: str) -> DocumentInfo | None:
        async with await self._connect() as conn:
            cur = await conn.execute(
                "SELECT id, filename, chunk_count, created_at FROM nexora_documents "
                "WHERE owner_id = %s AND sha256 = %s",
                (owner, sha256),
            )
            row = await cur.fetchone()
        return DocumentInfo(str(row[0]), row[1], row[2], row[3].isoformat()) if row else None

    async def add_document(self, owner, filename, sha256, model, chunks, embeddings) -> str:
        import numpy as np

        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings differ in length")
        doc_id = str(uuid.uuid4())
        async with await self._connect() as conn:
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO nexora_documents (id, owner_id, filename, sha256, embedding_model, chunk_count) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (doc_id, owner, filename, sha256, model, len(chunks)),
                )
                async with conn.cursor() as cur:
                    await cur.executemany(
                        "INSERT INTO nexora_chunks (document_id, owner_id, chunk_index, page, content, embedding) "
                        "VALUES (%s, %s, %s, %s, %s, %s)",
                        [
                            (doc_id, owner, c.index, c.page, c.text, np.asarray(e, dtype=np.float32))
                            for c, e in zip(chunks, embeddings, strict=True)
                        ],
                    )
        return doc_id

    async def list_documents(self, owner: str) -> list[DocumentInfo]:
        async with await self._connect() as conn:
            cur = await conn.execute(
                "SELECT id, filename, chunk_count, created_at FROM nexora_documents "
                "WHERE owner_id = %s ORDER BY created_at",
                (owner,),
            )
            rows = await cur.fetchall()
        return [DocumentInfo(str(r[0]), r[1], r[2], r[3].isoformat()) for r in rows]

    async def delete_document(self, owner: str, document_id: str) -> bool:
        async with await self._connect() as conn:
            cur = await conn.execute(
                "DELETE FROM nexora_documents WHERE owner_id = %s AND id = %s",
                (owner, document_id),
            )
            await conn.commit()
            return cur.rowcount > 0

    async def chunks_for_owner(self, owner: str) -> list[tuple[int, str]]:
        async with await self._connect() as conn:
            cur = await conn.execute("SELECT id, content FROM nexora_chunks WHERE owner_id = %s ORDER BY id", (owner,))
            return [(r[0], r[1]) for r in await cur.fetchall()]

    async def update_embeddings(self, owner: str, model: str, rows: list[tuple[int, list[float]]]) -> None:
        import numpy as np

        async with await self._connect() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    await cur.executemany(
                        "UPDATE nexora_chunks SET embedding = %s WHERE id = %s AND owner_id = %s",
                        [(np.asarray(e, dtype=np.float32), cid, owner) for cid, e in rows],
                    )
                await conn.execute(
                    "UPDATE nexora_documents SET embedding_model = %s WHERE owner_id = %s", (model, owner)
                )

    async def search(self, owner: str, query_embedding: list[float], k: int) -> list[Hit]:
        import numpy as np

        async with await self._connect() as conn:
            cur = await conn.execute(
                "SELECT c.document_id, d.filename, c.chunk_index, c.page, c.content, "
                "       1 - (c.embedding <=> %s) AS score "
                "FROM nexora_chunks c JOIN nexora_documents d ON d.id = c.document_id "
                "WHERE c.owner_id = %s "
                "ORDER BY c.embedding <=> %s LIMIT %s",
                (
                    np.asarray(query_embedding, dtype=np.float32),
                    owner,
                    np.asarray(query_embedding, dtype=np.float32),
                    k,
                ),
            )
            rows = await cur.fetchall()
        return [Hit(str(r[0]), r[1], r[2], r[3], r[4], float(r[5])) for r in rows]
