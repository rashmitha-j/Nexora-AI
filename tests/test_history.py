import asyncio

from nexora.history import NexoraDataLayer, asyncpg_dsn, history_from_steps


def test_asyncpg_dsn_strips_libpq_only_options():
    url = "postgresql://u:p@ep-x-pooler.neon.tech/neondb?sslmode=require&channel_binding=require"
    assert asyncpg_dsn(url) == "postgresql://u:p@ep-x-pooler.neon.tech/neondb?sslmode=require"
    assert asyncpg_dsn("postgres://u:p@h/db").startswith("postgresql://")


def test_history_from_steps_keeps_only_real_turns():
    steps = [
        {"type": "assistant_message", "output": "# Welcome", "start": "1"},
        {"type": "user_message", "output": "What is RAG?", "start": "2"},
        {"type": "assistant_message", "output": "Retrieval...", "start": "3"},
        {"type": "user_message", "output": "/usage", "start": "4"},
        {"type": "assistant_message", "output": "This chat: 1 replies", "start": "5"},
        {"type": "run", "output": "internal", "start": "6"},
        {"type": "user_message", "output": "And embeddings?", "start": "7"},
        {"type": "assistant_message", "output": "Vectors...", "start": "8"},
    ]
    assert history_from_steps(steps, max_turns=10) == [
        ("user", "What is RAG?"),
        ("assistant", "Retrieval..."),
        ("user", "And embeddings?"),
        ("assistant", "Vectors..."),
    ]
    assert history_from_steps(steps, max_turns=1) == [("user", "And embeddings?"), ("assistant", "Vectors...")]


def test_data_layer_degrades_when_database_is_down(monkeypatch):
    import asyncpg

    attempts = []

    async def refuse(*args, **kwargs):
        attempts.append(1)
        raise ConnectionRefusedError("refused")

    monkeypatch.setattr(asyncpg, "create_pool", refuse)
    now = [0.0]
    dl = NexoraDataLayer("postgresql://u:p@127.0.0.1:1/db", clock=lambda: now[0])

    async def scenario():
        user = await dl.get_user("alice")
        from chainlit.types import Pagination, ThreadFilter

        threads = await dl.list_threads(Pagination(first=5), ThreadFilter())
        author = await dl.get_thread_author("t1")
        return user, threads, author

    user, threads, author = asyncio.run(scenario())
    assert user is None and threads.data == [] and author == ""
    # Only one connection attempt during the back-off window, not one per call.
    assert len(attempts) == 1
    now[0] = 31.0
    asyncio.run(dl.get_user("alice"))
    assert len(attempts) == 2
