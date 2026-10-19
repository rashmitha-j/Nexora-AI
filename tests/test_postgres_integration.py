"""Runs against a real Postgres with pgvector when NEXORA_TEST_DATABASE_URL is set.

Use a throwaway database: the test creates tables and deletes only the rows it adds.
"""

import asyncio
import os
import uuid

import pytest

from nexora.rag.ingest import Chunk

DB_URL = os.getenv("NEXORA_TEST_DATABASE_URL")
DIM = 768  # matches the app default, so the test can share a database with it
UNIT = [1.0] + [0.0] * (DIM - 1)
pytestmark = pytest.mark.skipif(not DB_URL, reason="NEXORA_TEST_DATABASE_URL not set")


def test_vector_store_round_trip_and_owner_isolation():
    from nexora.rag.store import PgVectorStore

    owner, other = f"test-{uuid.uuid4().hex[:8]}", f"test-{uuid.uuid4().hex[:8]}"

    async def scenario():
        store = PgVectorStore(DB_URL, dim=DIM)
        await store.ensure_schema()
        await store.ensure_schema()  # idempotent
        doc = await store.add_document(owner, "a.md", uuid.uuid4().hex, "fake", [Chunk(0, "apples", 1)], [UNIT])
        await store.add_document(other, "b.md", uuid.uuid4().hex, "fake", [Chunk(0, "secret", None)], [UNIT])
        hits = await store.search(owner, UNIT, 5)
        docs = await store.list_documents(owner)
        deleted = await store.delete_document(owner, doc)
        not_mine = await store.delete_document(owner, (await store.list_documents(other))[0].id)
        for d in await store.list_documents(other):
            await store.delete_document(other, d.id)
        await store.close()
        return hits, docs, deleted, not_mine

    hits, docs, deleted, not_mine = asyncio.run(scenario())
    assert [(h.filename, h.page, round(h.score, 3)) for h in hits] == [("a.md", 1, 1.0)]
    assert [d.filename for d in docs] == ["a.md"]
    assert deleted is True and not_mine is False


def test_chat_history_schema_and_thread_round_trip():
    from chainlit.types import Pagination, ThreadFilter
    from chainlit.user import User

    from nexora.history import NexoraDataLayer

    identifier = f"test-{uuid.uuid4().hex[:8]}"
    thread_id = str(uuid.uuid4())

    async def scenario():
        dl = NexoraDataLayer(DB_URL)
        user = await dl.create_user(User(identifier=identifier, metadata={}))
        await dl.update_thread(thread_id, name="Integration", user_id=user.id)
        page = await dl.list_threads(Pagination(first=5), ThreadFilter(userId=user.id))
        author = await dl.get_thread_author(thread_id)
        await dl.delete_thread(thread_id)
        await dl.execute_query('DELETE FROM "User" WHERE id = $1', {"id": user.id})
        await dl.close()
        return page, author

    page, author = asyncio.run(scenario())
    assert [t["name"] for t in page.data] == ["Integration"]
    assert author == identifier


def test_session_cleanup_delete_owner_and_purge_stale():
    from nexora.rag.store import PgVectorStore

    session, keeper = f"session:{uuid.uuid4().hex}", f"test-{uuid.uuid4().hex[:8]}"

    async def scenario():
        store = PgVectorStore(DB_URL, dim=DIM)
        await store.ensure_schema()
        await store.add_document(session, "a.md", uuid.uuid4().hex, "fake", [Chunk(0, "x", None)], [UNIT])
        await store.add_document(session, "b.md", uuid.uuid4().hex, "fake", [Chunk(0, "y", None)], [UNIT])
        await store.add_document(keeper, "c.md", uuid.uuid4().hex, "fake", [Chunk(0, "z", None)], [UNIT])
        fresh_purged = await store.purge_stale("session:", 24)  # nothing is old enough yet
        pool = await store._get_pool()
        await pool.execute(
            "UPDATE nexora_documents SET created_at = now() - interval '25 hours' WHERE owner_id = $1", session
        )
        old_purged = await store.purge_stale("session:", 24)
        left = await store.list_documents(session)
        removed_keeper = await store.delete_owner(keeper)
        await store.close()
        return fresh_purged, old_purged, left, removed_keeper

    fresh_purged, old_purged, left, removed_keeper = asyncio.run(scenario())
    assert fresh_purged == 0 and old_purged == 2 and left == []
    assert removed_keeper == 1  # purge only touched session:* owners
