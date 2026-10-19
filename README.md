# Nexora AI

Nexora AI is a Chainlit chat app with:

- **Multi-provider chat.** Google Gemini, Anthropic Claude, OpenAI, or a local Ollama model, streamed token by token, with per-user rate limits, timeouts, retries and token-usage tracking.
- **Answers from your documents (RAG).** Upload PDF, text, Markdown, CSV or JSON files. They're chunked, embedded and stored in Postgres with pgvector. Answers cite the passages they used, and each user only ever retrieves their own documents.
- **MetaGPT build jobs.** `/build <idea>` runs a MetaGPT software team (product manager → architect → project manager → engineer) in a separate worker service and returns the generated project as a zip.

## Architecture

```
Browser ──websocket──▶ Chainlit web app (nexora_app.py)
                         │  nexora/providers.py   Gemini · Claude · OpenAI · Ollama
                         │  nexora/rag/*          extract → chunk → embed → pgvector
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

### Turn on documents

Create a Postgres database with pgvector (Neon and Supabase both support it), set `DATABASE_URL`, and restart. Tables are created automatically on first use. `EMBEDDING_DIM` is fixed when the tables are created, so changing it later needs a fresh database.

### Run the MetaGPT worker

Use a separate virtual environment, because MetaGPT pins many dependencies:

```bash
python -m venv .venv-worker
PYTHON=.venv-worker/Scripts/python sh worker/install.sh      # macOS/Linux: .venv-worker/bin/python
METAGPT_WORKER_TOKEN=<random 32+ chars> METAGPT_API_KEY=<gemini key> \
  .venv-worker/Scripts/python -m uvicorn worker.app:app --port 8001
```

Then set `METAGPT_WORKER_URL=http://127.0.0.1:8001` and the same `METAGPT_WORKER_TOKEN` for the web app.

`worker/requirements.txt` is upstream MetaGPT's dependency list at a pinned commit, with two changes. `lancedb==0.4.0` is removed because it's no longer on PyPI. `pyarrow` is added because `qianfan` otherwise injects a fake `pyarrow` that breaks pandas.

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
| `NEXORA_PROVIDER`, `*_MODEL` | web | Default provider and models |
| `LLM_TIMEOUT_SECONDS`, `MAX_OUTPUT_TOKENS`, `RATE_LIMIT_PER_MINUTE`, `MAX_MESSAGE_CHARS`, `HISTORY_TURNS` | web | Limits |
| `NEXORA_USERS`, `CHAINLIT_AUTH_SECRET` | web | Sign-in |
| `DATABASE_URL`, `EMBEDDING_*`, `RAG_TOP_K`, `RAG_MIN_SCORE`, `MAX_UPLOAD_MB` | web | Documents |
| `METAGPT_WORKER_URL`, `METAGPT_WORKER_TOKEN` | web + worker | Worker connection |
| `METAGPT_API_TYPE`, `METAGPT_MODEL`, `METAGPT_API_KEY`, `METAGPT_JOB_TIMEOUT_SECONDS`, `WORKER_JOBS_DIR` | worker | MetaGPT runs |

## Tests and checks

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

The tests use fakes for LLM providers, the vector store and MetaGPT, so they need no keys or network.

## Deploy on Render

`render.yaml` is a Render Blueprint for two services: `nexora-ai` (web) and `nexora-metagpt-worker`.

1. In Render, choose **New → Blueprint** and select this repository.
2. Enter the secrets Render asks for: `GEMINI_API_KEY` (plus any other provider keys), `DATABASE_URL`, `NEXORA_USERS` and `METAGPT_API_KEY`. `CHAINLIT_AUTH_SECRET` and the worker token are generated for you.
3. After the worker deploys, copy its public URL into the web service's `METAGPT_WORKER_URL`.
4. Health checks: web `/health`, worker `/healthz`.

Not deployed: neither service has been deployed to Render from this repository yet, so the Blueprint is untested on Render.

## Known limitations

- **Gemini capacity.** During testing, `gemini-3.8-flash` (Google's current recommendation) often returned 503 "high demand". Set `GEMINI_MODEL=gemini-3.5-flash` if that persists. `gemini-2.5-flash` no longer works for new API keys.
- **Single instance.** Rate limits and the job queue live in memory, so run one instance of each service. With more instances, move them to Redis.
- **Worker storage is temporary** on Render's free plan. Finished jobs disappear when the worker restarts.
- **MetaGPT memory use.** The worker may exceed the free plan's 512 MB of RAM. Use a larger plan if builds crash.
- **Scanned PDFs** have no text layer and are rejected. Run OCR first.
- **CORS.** `.chainlit/config.toml` allows all origins (Chainlit's default). Set `allow_origins` to your deployed URL for production.
