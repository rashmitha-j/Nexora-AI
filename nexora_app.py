"""Nexora AI Chainlit app.

Run locally: chainlit run nexora_app.py
"""

import asyncio
import logging
import time
from pathlib import Path

import chainlit as cl
from chainlit.input_widget import Select, Switch, TextInput
from dotenv import load_dotenv

from nexora.auth import authenticate, parse_users
from nexora.commands import help_text, parse_command
from nexora.config import load_settings
from nexora.jobs import JobClient, JobClientError
from nexora.limits import RateLimiter
from nexora.prompts import BASE_SYSTEM, rag_system
from nexora.providers import ChatMessage, ProviderError, Usage, build_provider, candidate_models, stream_with_fallback
from nexora.rag.ingest import UnsupportedFileError
from nexora.rag.service import RagService, format_context, format_sources

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nexora")

settings = load_settings()
limiter = RateLimiter(settings.rate_limit_per_minute)
jobs = JobClient(settings.worker_url, settings.worker_token) if settings.jobs_enabled else None

rag: RagService | None = None
_schema_ready = asyncio.Lock()
_schema_done = False
if settings.rag_enabled:
    from nexora.rag.embeddings import build_embedder
    from nexora.rag.store import PgVectorStore

    rag = RagService(
        PgVectorStore(settings.database_url, settings.embedding_dim),
        build_embedder(settings),
        settings.rag_top_k,
        settings.rag_min_score,
    )

JOB_POLL_SECONDS = 5
JOB_WATCH_LIMIT_SECONDS = 20 * 60

if settings.auth_enabled:
    _users = parse_users(settings.users_spec)

    @cl.password_auth_callback
    async def password_auth(username: str, password: str):
        ok = await asyncio.to_thread(authenticate, _users, username, password)
        return cl.User(identifier=username, metadata={"provider": "password"}) if ok else None


# Chainlit enables its own asyncpg data layer whenever DATABASE_URL is set, without
# creating its tables. Always register ours instead. Chat history is only useful (and
# only visible) for signed-in users, so anonymous sessions aren't stored.
history_enabled = bool(settings.database_url) and settings.auth_enabled
if settings.database_url:

    @cl.data_layer
    def data_layer():
        if not history_enabled:
            return None
        from nexora.history import NexoraDataLayer

        return NexoraDataLayer(settings.database_url)


logger.info(
    "Nexora AI starting: providers=%s default=%s rag=%s jobs=%s auth=%s history=%s",
    settings.providers,
    settings.default_provider,
    settings.rag_enabled,
    settings.jobs_enabled,
    settings.auth_enabled,
    history_enabled,
)


# -- helpers -----------------------------------------------------------------


def current_owner() -> str | None:
    """The signed-in user's id. Documents and jobs need one, so they are only
    available when sign-in (NEXORA_USERS) is configured."""
    user = cl.user_session.get("user")
    return user.identifier if user else None


def rate_limit_key() -> str:
    return current_owner() or f"session:{cl.context.session.id}"


async def ensure_rag_schema() -> None:
    global _schema_done
    async with _schema_ready:
        if not _schema_done:
            await rag.store.ensure_schema()
            _schema_done = True


async def rag_ready() -> bool:
    """Make sure document tables exist; tell the user if the database is unreachable."""
    try:
        await ensure_rag_schema()
        return True
    except Exception:
        logger.exception("Document storage unavailable")
        await say("Document storage is unavailable right now (the database didn't respond). Chat still works.")
        return False


def available_help() -> str:
    signed_in = current_owner() is not None
    return help_text(docs=rag is not None and signed_in, build=jobs is not None and signed_in)


async def say(content: str) -> None:
    await cl.Message(content=content).send()


# -- lifecycle ---------------------------------------------------------------


async def init_session(history: list[ChatMessage]) -> bool:
    """Set up per-chat state and the settings panel. Returns whether documents are available."""
    cl.user_session.set("history", history)
    cl.user_session.set("usage", {"input": 0, "output": 0, "requests": 0})
    cl.user_session.set("provider", settings.default_provider)
    cl.user_session.set("model", "")
    docs_available = rag is not None and current_owner() is not None
    cl.user_session.set("use_docs", docs_available)

    widgets = [
        Select(
            id="provider",
            label="AI provider",
            values=settings.providers,
            initial_value=settings.default_provider,
        ),
        TextInput(
            id="model",
            label="Model (blank = server default for the provider)",
            initial="",
            placeholder=settings.models[settings.default_provider],
        ),
    ]
    if docs_available:
        widgets.append(Switch(id="use_docs", label="Answer from my documents", initial=True))
    await cl.ChatSettings(widgets).send()
    return docs_available


@cl.on_chat_start
async def on_chat_start():
    docs_available = await init_session([])
    owner = current_owner()
    lines = [
        "# Welcome to Nexora AI",
        f"Chatting with **{settings.default_provider}** "
        f"(`{settings.models[settings.default_provider]}`). Change it in the settings panel.",
    ]
    if docs_available:
        lines.append("Attach PDF, text or Markdown files to a message to add them to your knowledge base.")
    elif rag is not None:
        lines.append("Sign-in is required to use documents.")
    if jobs is not None and owner is not None:
        lines.append("Use `/build <idea>` to have a MetaGPT software team plan and write a project.")
    lines.append("\nCommands:\n" + available_help())
    await say("\n\n".join(lines))


@cl.on_chat_resume
async def on_chat_resume(thread: dict):
    """Reopen a saved conversation from the history sidebar with its context restored."""
    from nexora.history import history_from_steps

    pairs = history_from_steps(thread.get("steps") or [], settings.history_turns)
    await init_session([ChatMessage(role, text) for role, text in pairs])


@cl.on_settings_update
async def on_settings_update(new_settings: dict):
    provider = new_settings.get("provider", settings.default_provider)
    if provider not in settings.providers:
        await say(f"Provider `{provider}` isn't configured on this server.")
        return
    cl.user_session.set("provider", provider)
    cl.user_session.set("model", (new_settings.get("model") or "").strip())
    if "use_docs" in new_settings:
        cl.user_session.set("use_docs", bool(new_settings["use_docs"]))
    model = cl.user_session.get("model") or settings.models[provider]
    await say(f"Now using **{provider}** (`{model}`).")


@cl.on_message
async def on_message(message: cl.Message):
    try:
        await handle_message(message)
    except Exception:
        # Last line of defence: log the details, never leave the user without a reply.
        logger.exception("Unhandled error while handling a message")
        await say("Something went wrong on our side. Please try again in a moment.")


async def handle_message(message: cl.Message) -> None:
    wait = limiter.check(rate_limit_key())
    if wait:
        await say(f"You're sending messages too quickly. Try again in {int(wait) + 1} seconds.")
        return

    files = [e for e in (message.elements or []) if getattr(e, "path", None)]
    if files:
        await handle_uploads(files)

    text = (message.content or "").strip()
    if not text:
        if not files:
            await say("Type a message to get started.")
        return
    if len(text) > settings.max_message_chars:
        await say(f"Please keep messages under {settings.max_message_chars} characters.")
        return

    command = parse_command(text)
    if command:
        await handle_command(command.name, command.arg)
        return
    await chat(text)


# -- chat --------------------------------------------------------------------


async def chat(text: str) -> None:
    provider_name = cl.user_session.get("provider")
    try:
        providers = [
            build_provider(settings, provider_name, model)
            for model in candidate_models(settings, provider_name, cl.user_session.get("model"))
        ]
    except ProviderError as exc:
        await say(exc.user_message)
        return
    provider = providers[0]

    history: list[ChatMessage] = cl.user_session.get("history")
    system, hits = BASE_SYSTEM, []
    owner = current_owner()
    if rag is not None and owner and cl.user_session.get("use_docs"):
        try:
            await ensure_rag_schema()
            hits = await rag.retrieve(owner, text)
        except Exception:
            logger.exception("Retrieval failed")
            await say("Couldn't search your documents right now, so this answer doesn't use them.")
        if hits:
            system = rag_system(format_context(hits))

    messages = history + [ChatMessage("user", text)]
    usage = Usage()
    reply = cl.Message(content="")
    started = time.monotonic()
    try:
        async for chunk in stream_with_fallback(providers, system, messages, usage, settings.llm_timeout_s):
            await reply.stream_token(chunk)
    except ProviderError as exc:
        logger.error(
            "Chat failed for provider=%s models=%s", provider.name, [p.model for p in providers], exc_info=exc.__cause__
        )
        if reply.content:
            await reply.stream_token(f"\n\n_Response interrupted: {exc.user_message}_")
            await reply.send()
        else:
            await say(exc.user_message)
        return

    if not reply.content.strip():
        await say(f"{provider.name} returned an empty response (it may have been blocked by safety filters).")
        return
    if hits:
        reply.content += "\n\n---\n**Sources**\n" + format_sources(hits)
    await reply.send()

    history.extend([ChatMessage("user", text), ChatMessage("assistant", reply.content)])
    del history[: max(0, len(history) - settings.history_turns * 2)]
    totals = cl.user_session.get("usage")
    totals["input"] += usage.input_tokens or 0
    totals["output"] += usage.output_tokens or 0
    totals["requests"] += 1
    logger.info(
        "chat provider=%s model=%s in=%s out=%s sources=%d seconds=%.1f",
        provider.name,
        usage.model,
        usage.input_tokens,
        usage.output_tokens,
        len(hits),
        time.monotonic() - started,
    )


# -- documents ---------------------------------------------------------------


async def handle_uploads(files) -> None:
    owner = current_owner()
    if rag is None:
        await say("Document upload isn't enabled on this server (DATABASE_URL is not set).")
        return
    if owner is None:
        await say("Sign in to upload documents.")
        return
    if not await rag_ready():
        return
    for element in files:
        path = Path(element.path)
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > settings.max_upload_mb:
            await say(f"**{element.name}** is {size_mb:.1f} MB; the limit is {settings.max_upload_mb} MB.")
            continue
        async with cl.Step(name=f"Indexing {element.name}") as step:
            try:
                result = await rag.ingest(owner, element.name, path)
            except UnsupportedFileError as exc:
                step.output = str(exc)
                await say(str(exc))
                continue
            except Exception:
                logger.exception("Ingest failed for %s", element.name)
                step.output = "Failed"
                await say(f"Couldn't index **{element.name}**. Please try again.")
                continue
            verb = "Re-indexed" if result.replaced else "Indexed"
            step.output = f"{verb} {result.chunk_count} chunks"
        await say(f"{verb} **{result.filename}** ({result.chunk_count} chunks). Ask me about it.")


async def handle_command(name: str, arg: str) -> None:
    if name == "help":
        await say(available_help())
        return
    if name == "usage":
        u = cl.user_session.get("usage")
        await say(f"This chat: {u['requests']} replies, {u['input']} input tokens, {u['output']} output tokens.")
        return
    if name == "build":
        await handle_build(arg)
        return

    owner = current_owner()
    if rag is None or owner is None:
        await say("Documents need sign-in and a configured database.")
        return
    if not await rag_ready():
        return
    documents = await rag.store.list_documents(owner)
    if name == "docs":
        if not documents:
            await say("You haven't uploaded any documents yet.")
        else:
            await say("\n".join(f"{i}. **{d.filename}** — {d.chunk_count} chunks" for i, d in enumerate(documents, 1)))
    elif name == "forget":
        if not arg.isdigit() or not 1 <= int(arg) <= len(documents):
            await say("Usage: `/forget <number>` using the numbers from `/docs`.")
            return
        doc = documents[int(arg) - 1]
        res = await cl.AskActionMessage(
            content=f"Delete **{doc.filename}** from your knowledge base? This can't be undone.",
            actions=[
                cl.Action(name="confirm", payload={"ok": True}, label="Delete"),
                cl.Action(name="cancel", payload={"ok": False}, label="Cancel"),
            ],
        ).send()
        if res and res.get("payload", {}).get("ok"):
            deleted = await rag.store.delete_document(owner, doc.id)
            await say(f"Deleted **{doc.filename}**." if deleted else "That document was already gone.")
        else:
            await say("Kept it.")
    elif name == "reindex":
        async with cl.Step(name="Re-indexing documents") as step:
            count = await rag.reindex(owner)
            step.output = f"{count} chunks"
        await say(f"Re-embedded {count} chunks with `{rag.embedder.model}`.")


# -- MetaGPT jobs ------------------------------------------------------------


async def handle_build(idea: str) -> None:
    owner = current_owner()
    if jobs is None:
        await say("The MetaGPT worker isn't configured on this server (METAGPT_WORKER_URL).")
        return
    if owner is None:
        await say("Sign in to run build jobs.")
        return
    if not idea:
        await say("Usage: `/build <project idea>`, e.g. `/build a CLI to-do app in Python`.")
        return

    approval = await cl.AskActionMessage(
        content=(
            "**Run a MetaGPT software team?**\n\n"
            f"Idea: _{idea[:300]}_\n\n"
            "A product manager, architect, project manager and engineer will write requirements, "
            "a design and code. This uses the worker's LLM quota and can take several minutes. "
            "Generated code is not executed; you get a zip to review."
        ),
        actions=[
            cl.Action(name="approve", payload={"ok": True}, label="Run job"),
            cl.Action(name="cancel", payload={"ok": False}, label="Cancel"),
        ],
        timeout=120,
    ).send()
    if not (approval and approval.get("payload", {}).get("ok")):
        await say("Build cancelled.")
        return

    try:
        job = await jobs.submit(owner, idea)
    except JobClientError as exc:
        await say(f"Couldn't start the job: {exc}")
        return

    status_msg = cl.Message(content=f"Job `{job['id'][:8]}` queued…")
    await status_msg.send()
    deadline = time.monotonic() + JOB_WATCH_LIMIT_SECONDS
    last = job["status"]
    while job["status"] in ("queued", "running") and time.monotonic() < deadline:
        await asyncio.sleep(JOB_POLL_SECONDS)
        try:
            job = await jobs.status(owner, job["id"])
        except JobClientError as exc:
            logger.warning("Job poll failed: %s", exc)
            continue
        if job["status"] != last:
            last = job["status"]
            ellipsis = "…" if last in ("queued", "running") else ""
            status_msg.content = f"Job `{job['id'][:8]}` {last}{ellipsis}"
            await status_msg.update()

    if job["status"] in ("queued", "running"):
        await say(
            f"Job `{job['id'][:8]}` is still {job['status']}. I stopped watching it; it keeps running on the worker."
        )
        return

    summary = [f"Job `{job['id'][:8]}` **{job['status']}**."]
    if job.get("cost_usd") is not None:
        summary.append(f"Estimated LLM cost: ${job['cost_usd']}.")
    if job.get("error"):
        summary.append(f"Error: {job['error']}")
    elements = []
    if job.get("files"):
        listing = "\n".join(f"- `{f}`" for f in job["files"][:40])
        more = f"\n- …and {len(job['files']) - 40} more" if len(job["files"]) > 40 else ""
        summary.append(f"Files:\n{listing}{more}")
        try:
            data = await jobs.archive(owner, job["id"])
            elements.append(cl.File(name=f"nexora-job-{job['id'][:8]}.zip", content=data, display="inline"))
        except JobClientError as exc:
            summary.append(f"Couldn't fetch the zip: {exc}")
    await cl.Message(content="\n\n".join(summary), elements=elements).send()
