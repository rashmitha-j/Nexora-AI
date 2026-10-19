# Nexora AI

Nexora AI is a Chainlit chat app with:

- **Multi-provider chat.** Google Gemini, Anthropic Claude, OpenAI, or a local Ollama model, streamed token by token, with per-user rate limits, timeouts, retries and token-usage tracking.
- **Answers from your documents (RAG).** Upload PDF, text, Markdown, CSV or JSON files. They're chunked, embedded and stored in Postgres with pgvector. Answers cite the passages they used, and each user only ever retrieves their own documents.
- **Chat history.** Signed-in users' conversations are saved in the same Postgres database and listed in the sidebar; reopening one restores its context.
- **MetaGPT build jobs (optional, runs locally).** `/build <idea>` runs a MetaGPT software team (product manager → architect → project manager → engineer) in a separate worker service and returns the generated project as a zip. The worker needs more memory than Render's free plan, so it isn't part of the free deployment.

## Architecture

```
Browser ──websocket──▶ Chainlit web app (nexora_app.py)
                         │  nexora/providers.py   Gemini · Claude · OpenAI · Ollama
                         │  nexora/rag/*          extract → chunk → embed → pgvector
                         │  nexora/history.py     chat history (Chainlit data layer)
                         │  nexora/auth.py        password sign-in (PBKDF2 hashes)
                         │  nexora/limits.py      per-user rate limiting
                         ├──────────────▶ Postgres + pgvector (Neon, Supabase, …)
                         └──HTTPS + token─▶ MetaGPT worker (worker/app.py, FastAPI)
                                              one job at a time, each in its own
                                              process with a hard timeout
```

- Every key is read from server-side environment variables. Nothing secret reaches the browser.
- Retrieved document text is wrapped in `<document>` tags, and the system prompt tells the model to treat it as data, never instructions. Tags inside documents are stripped so a file can't break out of its wrapper.
- The worker only uses MetaGPT roles that write documents and code. `QaEngineer`, `Engineer2` and `DataAnalyst` are excluded because they execute generated code or shell commands. Each job needs the user's explicit approval in the chat.

## Requirements

- Python 3.11 (MetaGPT supports `>=3.9,<3.12`)
- At least one LLM API key (or a running Ollama server)
- Optional: Postgres 15+ with the `vector` extension for documents
- Optional: a second Python 3.11 environment for the MetaGPT worker

## Run locally

```bash
python -m venv .venv
.venv/Scripts/activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env              # then fill in at least GEMINI_API_KEY
chainlit run nexora_app.py
```

Open http://localhost:8000.

### Turn on sign-in (needed for documents and `/build`)

```bash
python -m nexora.auth             # prompts for a username and password, prints "user:hash"
chainlit create-secret            # prints CHAINLIT_AUTH_SECRET=...
```

Put the printed values in `NEXORA_USERS` (comma-separate several users) and `CHAINLIT_AUTH_SECRET`.

### Turn on documents and chat history

Create a Postgres database with pgvector (Neon and Supabase both support it), set `DATABASE_URL`, and restart. Paste Neon's connection string as it is: options that only `psql` understands, such as `channel_binding`, are removed automatically. All tables are created on first use with `CREATE TABLE IF NOT EXISTS`; nothing is ever dropped. `EMBEDDING_DIM` is fixed when the tables are created, so changing it later needs a fresh database.

**Without sign-in (public site):** anyone can attach documents, but each chat gets its own private document space: up to 5 files of about 300 KB of text each, never visible to other visitors. They're deleted when the visitor clicks **New chat**, and anything older than 24 hours is purged automatically. A closed tab or a dropped connection doesn't delete them straight away, because Chainlit lets the browser reconnect to the same chat. Chat history isn't saved for anonymous visitors.

**With sign-in (`NEXORA_USERS`):** each user keeps their documents across chats, and their conversations are saved and listed in the sidebar.

To save the free embedding quota, a question is only matched against documents when that chat actually has some.

If the database is unreachable, chat keeps working: history pauses for 30 seconds at a time and document commands say storage is unavailable.

> **Why this matters on Chainlit:** Chainlit turns on its own database layer whenever `DATABASE_URL` is set, and that layer needs `asyncpg` plus tables Chainlit never creates. Without `nexora/history.py` the page goes blank (`/project/settings` returns HTTP 500).

### Run the MetaGPT worker

Use a separate virtual environment, because MetaGPT pins many dependencies:

```bash
python -m venv .venv-worker
PYTHON=.venv-worker/Scripts/python sh worker/install.sh      # macOS/Linux: .venv-worker/bin/python
METAGPT_WORKER_TOKEN=<random 32+ chars> METAGPT_API_KEY=<gemini key> \
  .venv-worker/Scripts/python -m uvicorn worker.app:app --port 8001
```

Then set `METAGPT_WORKER_URL=http://127.0.0.1:8001` and the same `METAGPT_WORKER_TOKEN` for the web app, and sign in (jobs belong to a user).

Notes on running upstream MetaGPT:
- `worker/requirements.txt` is upstream MetaGPT's dependency list at a pinned commit, with two changes. `lancedb==0.4.0` is removed because it's no longer on PyPI. `pyarrow` is added because `qianfan` otherwise injects a fake `pyarrow` that breaks pandas.
- With `METAGPT_API_TYPE=gemini` (the default) the worker talks to Gemini through Google's OpenAI-compatible endpoint. MetaGPT's native Gemini client drops all system prompts, so its first step fails.
- Each job runs in its own process, starting inside its own workspace folder, with a per-job home directory and a fixed git identity, and is killed at `METAGPT_JOB_TIMEOUT_SECONDS`. The project path is never taken from the user's idea, so generated files stay in the job's workspace.
- MetaGPT 1.0's "quick think" shortcut is disabled for the hired roles; otherwise small ideas get a chat answer instead of running the pipeline.
- A full run makes dozens of LLM calls. On Gemini's free tier that can use a large share of a model's daily quota, so point the worker at a different model (or key) than the public chat if you can.

## Commands in the chat

| Command | What it does |
|---|---|
| `/help` | List available commands |
| `/docs` | List your documents |
| `/forget <n>` | Delete document *n* from `/docs` (asks first) |
| `/reindex` | Re-embed your documents after changing `EMBEDDING_MODEL` |
| `/build <idea>` | Run a MetaGPT job (asks first) |
| `/usage` | Token usage for this chat |

## Environment variables

See [.env.example](.env.example) for the full list with defaults.

| Variable | Used by | Purpose |
|---|---|---|
| `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OLLAMA_BASE_URL` | web | Chat providers (at least one) |
| `NEXORA_PROVIDER`, `*_MODEL`, `*_FALLBACK_MODELS` | web | Default provider, models, and models to fall back to on quota/overload errors |
| `LLM_TIMEOUT_SECONDS`, `MAX_OUTPUT_TOKENS`, `RATE_LIMIT_PER_MINUTE`, `MAX_MESSAGE_CHARS`, `HISTORY_TURNS` | web | Limits |
| `NEXORA_USERS`, `CHAINLIT_AUTH_SECRET` | web | Sign-in |
| `DATABASE_URL`, `EMBEDDING_*`, `RAG_TOP_K`, `RAG_MIN_SCORE`, `MAX_UPLOAD_MB` | web | Documents |
| `METAGPT_WORKER_URL`, `METAGPT_WORKER_TOKEN` | web + worker | Worker connection |
| `METAGPT_API_TYPE`, `METAGPT_MODEL`, `METAGPT_API_KEY`, `METAGPT_JOB_TIMEOUT_SECONDS`, `METAGPT_MAX_TOKENS`, `WORKER_JOBS_DIR` | worker | MetaGPT runs |

## Tests and checks

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

The tests use fakes for LLM providers, the vector store and MetaGPT, so they need no keys or network. To also run the Postgres integration tests, point them at a **throwaway** database with pgvector:

```bash
NEXORA_TEST_DATABASE_URL=postgresql://... python -m pytest tests/test_postgres_integration.py
```

## Deploy on Render (free)

The live service is a Render **free web service** with:

- Build command: `pip install -r requirements.txt`
- Start command: `chainlit run nexora_app.py --host 0.0.0.0 --port $PORT --headless`
- Health check path: `/health`
- Environment: `PYTHON_VERSION=3.11.9`, `GEMINI_API_KEY`, and optionally `GEMINI_MODEL`, `DATABASE_URL`, `NEXORA_USERS`, `CHAINLIT_AUTH_SECRET`

`render.yaml` describes the same service if you ever recreate it as a Blueprint. Set secrets in the Render dashboard (**Environment** tab), never in the repository.

## Known limitations (free tier)

- **Cold starts.** Render's free web services sleep after 15 minutes without traffic; the first visit afterwards takes about a minute.
- **Gemini quotas.** The free Gemini tier limits requests per minute and per day, **per model**. When the main model (`gemini-3.5-flash`) hits its quota or is overloaded, Nexora automatically retries with `GEMINI_FALLBACK_MODELS` (default `gemini-3.7-flash`, then `gemini-3.1-flash-lite`). Only when all of them are exhausted do users see a "rate limit or quota reached" message. `gemini-2.5-flash` no longer works for new API keys.
- **Neon free plan.** Storage is limited (0.5 GB at the time of writing) and the database suspends when idle, so the first query after a pause is slower. Pool sizes are kept small (history 5, documents 3 connections).
- **Single instance.** Rate limits and the job queue live in memory, so run one instance. With more instances, move them to Redis.
- **MetaGPT** needs more than the free plan's 512 MB of RAM, so `/build` is only available when you run the worker yourself.
- **Uploaded files aren't kept.** Only their extracted text and embeddings are stored; the original file isn't (no file storage is configured).
- **Scanned PDFs** have no text layer and are rejected. Run OCR first.
- **CORS.** `.chainlit/config.toml` allows all origins (Chainlit's default). Set `allow_origins` to your deployed URL for production.
