"""Chat history persistence on Postgres via Chainlit's built-in data layer.

Chainlit switches its asyncpg data layer on whenever DATABASE_URL is set, but it never
creates the tables it needs. This subclass makes that safe for Neon:

- creates Chainlit's tables on first connect (idempotent; never drops anything);
- removes connection options asyncpg doesn't understand (Neon URLs include
  `channel_binding=require`, which asyncpg would send to the server as a setting);
- keeps the pool small for the free tier and disables asyncpg's statement cache, which
  breaks behind Neon's PgBouncer pooler in transaction mode.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import asyncpg
from chainlit.data.chainlit_data_layer import ChainlitDataLayer

logger = logging.getLogger(__name__)

# libpq options that asyncpg doesn't accept as connection parameters.
_UNSUPPORTED_PARAMS = {"channel_binding", "options", "target_session_attrs", "gssencmode"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS "User" (
    id text PRIMARY KEY,
    identifier text NOT NULL UNIQUE,
    metadata jsonb NOT NULL DEFAULT '{}',
    "createdAt" timestamp(3) NOT NULL DEFAULT now(),
    "updatedAt" timestamp(3) NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS "Thread" (
    id text PRIMARY KEY,
    "createdAt" timestamp(3) NOT NULL DEFAULT now(),
    "updatedAt" timestamp(3) NOT NULL DEFAULT now(),
    "deletedAt" timestamp(3),
    name text,
    metadata jsonb NOT NULL DEFAULT '{}',
    tags text[] NOT NULL DEFAULT '{}',
    "userId" text REFERENCES "User"(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS "Step" (
    id text PRIMARY KEY,
    "createdAt" timestamp(3) NOT NULL DEFAULT now(),
    "updatedAt" timestamp(3) NOT NULL DEFAULT now(),
    "parentId" text REFERENCES "Step"(id) ON DELETE CASCADE,
    "threadId" text REFERENCES "Thread"(id) ON DELETE CASCADE,
    input text,
    metadata jsonb NOT NULL DEFAULT '{}',
    name text,
    output text,
    type text NOT NULL,
    "showInput" text DEFAULT 'json',
    "isError" boolean DEFAULT false,
    "startTime" timestamp(3),
    "endTime" timestamp(3)
);
CREATE TABLE IF NOT EXISTS "Element" (
    id text PRIMARY KEY,
    "createdAt" timestamp(3) NOT NULL DEFAULT now(),
    "updatedAt" timestamp(3) NOT NULL DEFAULT now(),
    "threadId" text REFERENCES "Thread"(id) ON DELETE CASCADE,
    "stepId" text NOT NULL REFERENCES "Step"(id) ON DELETE CASCADE,
    metadata jsonb NOT NULL DEFAULT '{}',
    mime text,
    name text NOT NULL,
    "objectKey" text,
    url text,
    "chainlitKey" text,
    display text,
    size text,
    language text,
    page integer,
    "autoPlay" boolean,
    "playerConfig" jsonb,
    props jsonb DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS "Feedback" (
    id text PRIMARY KEY,
    "createdAt" timestamp(3) NOT NULL DEFAULT now(),
    "updatedAt" timestamp(3) NOT NULL DEFAULT now(),
    "stepId" text NOT NULL REFERENCES "Step"(id) ON DELETE CASCADE,
    name text NOT NULL,
    value double precision NOT NULL,
    comment text
);
CREATE INDEX IF NOT EXISTS "Thread_userId_idx" ON "Thread" ("userId");
CREATE INDEX IF NOT EXISTS "Thread_updatedAt_idx" ON "Thread" ("updatedAt");
CREATE INDEX IF NOT EXISTS "Step_threadId_idx" ON "Step" ("threadId");
CREATE INDEX IF NOT EXISTS "Element_threadId_idx" ON "Element" ("threadId");
CREATE INDEX IF NOT EXISTS "Element_stepId_idx" ON "Element" ("stepId");
CREATE INDEX IF NOT EXISTS "Feedback_stepId_idx" ON "Feedback" ("stepId");
"""


def asyncpg_dsn(url: str) -> str:
    """Normalise a Postgres URL for asyncpg: postgres:// scheme, no libpq-only options."""
    parts = urlsplit(url)
    scheme = "postgresql" if parts.scheme in ("postgres", "postgresql") else parts.scheme
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k not in _UNSUPPORTED_PARAMS]
    return urlunsplit((scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


class DatabaseUnavailable(RuntimeError):
    pass


# Errors that mean "the database can't be reached", as opposed to a bug in a query.
_CONNECTION_ERRORS = (
    OSError,
    asyncio.TimeoutError,
    asyncpg.exceptions.InterfaceError,
    asyncpg.exceptions.PostgresConnectionError,
    asyncpg.exceptions.CannotConnectNowError,
    asyncpg.exceptions.TooManyConnectionsError,
)
RETRY_AFTER_SECONDS = 30.0


class NexoraDataLayer(ChainlitDataLayer):
    """Chat history is optional: if the database is unreachable, history is switched off
    for RETRY_AFTER_SECONDS (calls return empty results) instead of breaking the chat."""

    def __init__(self, database_url: str, clock=time.monotonic):
        super().__init__(database_url=asyncpg_dsn(database_url))
        self._connect_lock = asyncio.Lock()
        self._clock = clock
        self._down_until = 0.0

    async def connect(self):
        async with self._connect_lock:
            if self.pool:
                return
            if self._clock() < self._down_until:
                raise DatabaseUnavailable("chat history database recently unreachable")
            try:
                pool = await asyncpg.create_pool(
                    self.database_url,
                    min_size=1,
                    max_size=5,
                    statement_cache_size=0,
                    command_timeout=30,
                    timeout=10,
                )
                async with pool.acquire() as conn, conn.transaction():
                    await conn.execute("SET LOCAL lock_timeout = '15s'")
                    await conn.execute("SELECT pg_advisory_xact_lock(727275)")
                    await conn.execute(SCHEMA)
            except _CONNECTION_ERRORS as exc:
                self._down_until = self._clock() + RETRY_AFTER_SECONDS
                logger.warning(
                    "Chat history database unreachable (%s); history paused for %ds",
                    type(exc).__name__,
                    RETRY_AFTER_SECONDS,
                )
                raise DatabaseUnavailable(str(exc)) from exc
            self.pool = pool
            logger.info("Chat history storage ready")

    async def execute_query(self, query, params=None):
        try:
            return await super().execute_query(query, params)
        except _CONNECTION_ERRORS as exc:
            await self.cleanup()
            self._down_until = self._clock() + RETRY_AFTER_SECONDS
            raise DatabaseUnavailable(str(exc)) from exc


def _empty_threads():
    from chainlit.types import PageInfo, PaginatedResponse

    return PaginatedResponse(pageInfo=PageInfo(hasNextPage=False, startCursor=None, endCursor=None), data=[])


# What each data-layer method returns while the database is unavailable.
_FALLBACKS = {
    "get_user": lambda: None,
    "create_user": lambda: None,
    "get_thread": lambda: None,
    "get_thread_author": lambda: "",
    "get_step": lambda: None,
    "get_element": lambda: None,
    "list_threads": _empty_threads,
    "get_favorite_steps": list,
    "delete_feedback": lambda: False,
    "upsert_feedback": lambda: "",
    "create_element": lambda: None,
    "delete_element": lambda: None,
    "create_step": lambda: None,
    "update_step": lambda: None,
    "delete_step": lambda: None,
    "delete_thread": lambda: None,
    "update_thread": lambda: None,
}


def _degrade(name: str, fallback):
    method = getattr(ChainlitDataLayer, name)

    @functools.wraps(method)
    async def wrapper(self, *args, **kwargs):
        try:
            return await method(self, *args, **kwargs)
        except DatabaseUnavailable:
            return fallback()

    return wrapper


for _name, _fallback in _FALLBACKS.items():
    setattr(NexoraDataLayer, _name, _degrade(_name, _fallback))


def history_from_steps(steps: list[dict], max_turns: int) -> list[tuple[str, str]]:
    """Rebuild (role, text) chat history from a stored thread's steps, keeping only real
    question/answer pairs: slash commands and their replies, and the welcome message, are
    left out."""
    pairs: list[tuple[str, str]] = []
    pending_user: str | None = None
    for step in sorted(steps, key=lambda s: s.get("start") or s.get("createdAt") or ""):
        text = (step.get("output") or "").strip()
        if step.get("type") == "user_message":
            pending_user = None if (not text or text.startswith("/")) else text
        elif step.get("type") == "assistant_message" and pending_user and text:
            pairs += [("user", pending_user), ("assistant", text)]
            pending_user = None
    return pairs[-max_turns * 2 :]
