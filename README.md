# Nexora-AI

[![CI](https://github.com/rashmitha-j/Nexora-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/rashmitha-j/Nexora-AI/actions/workflows/ci.yml)
![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Chainlit](https://img.shields.io/badge/Chainlit-2.12-F80061)
![Google Gemini](https://img.shields.io/badge/Google-Gemini-4285F4?logo=googlegemini&logoColor=white)
![PostgreSQL + pgvector](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?logo=postgresql&logoColor=white)
![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?logo=render&logoColor=black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Nexora-AI is a generative-AI assistant built with Chainlit and Google Gemini. It chats with streaming answers, answers questions about uploaded PDF, text and Markdown files with cited sources (retrieval-augmented generation on PostgreSQL + pgvector), and can hand software-building tasks to a MetaGPT multi-agent team running as a separate worker. It is designed to run on free tiers: Render, Neon and the Gemini free quota.

- **Live app:** https://nexora-ai-jcdz.onrender.com
- **Repository:** https://github.com/rashmitha-j/Nexora-AI

> The live app runs on Render's free plan, which sleeps after 15 minutes without traffic. The first visit after that can take about a minute to load.

## Screenshots

| Answering from an uploaded document, with its source | Welcome screen |
|---|---|
| ![Document question answered with a citation](docs/screenshots/02-document-qa.jpg) | ![Welcome screen](docs/screenshots/01-welcome.jpg) |
| **Model choice limited to server-approved models** | **Mobile layout** |
| ![Settings panel with the model dropdown](docs/screenshots/03-model-settings.jpg) | <img src="docs/screenshots/04-mobile.jpg" alt="Mobile layout" width="260"> |

Screenshots were taken from this code running locally, with a real Gemini answer.

## Contents

- [Overview](#overview) · [Features](#features-and-status) · [Architecture](#architecture) · [Tech stack](#tech-stack) · [Repository structure](#repository-structure)
- [Quick start (Windows)](#quick-start-windows) · [Configuration](#environment-variables) · [Gemini quota and fallback](#gemini-quota-and-fallback)
- [Sign-in and chat history](#sign-in-and-chat-history) · [Documents (RAG)](#documents-rag) · [MetaGPT worker](#metagpt-worker-local-only)
- [Deploy on Render](#deploy-on-render-free) · [CI](#continuous-integration) · [Security](#security) · [Testing](#testing) · [Troubleshooting](#troubleshooting)
- [Limitations and future work](#limitations-and-future-work) · [Contributing](#contributing) · [License](#license) · [Links](#links)

## Overview

**Problem.** General chatbots can't answer questions about your own documents. Hosting an AI app publicly also exposes a limited API quota to anyone on the internet.

**Objectives.**
1. A public chat assistant backed by Google Gemini, with streaming answers.
2. Grounded answers from user-uploaded documents, with citations, kept private per user or per chat.
3. Optional sign-in with saved conversation history.
4. A MetaGPT multi-agent workflow that turns an idea into requirements, a design and code.
5. Running at ₹0/month, while staying usable when free quotas run out and protecting the quota from abuse.

## Features and status

| Feature | Status |
|---|---|
| Streaming chat with Gemini (Claude, OpenAI and Ollama also supported by the provider layer) | Live. Gemini verified; other providers implemented, not live-tested |
| Automatic fallback across Gemini models on quota or overload | Live |
| Document upload (PDF, TXT, MD, CSV, JSON) and cited answers | Live, verified with a PDF |
| Per-chat private documents for anonymous visitors (auto-deleted) | Live |
| Abuse protection: per-IP and global rate limits, size limits, model allow-list | In code (see [Security](#security)) |
| Password sign-in, saved chat history in PostgreSQL | Implemented and tested locally; off on the public site by choice |
| MetaGPT software-team jobs (`/build`) | Local only. Needs more memory than Render's free plan; a full run hasn't completed yet |

## Architecture

```
Browser ──websocket──▶ Chainlit app (nexora_app.py) ── one free Render web service
                         ├─ nexora/providers.py   Gemini · Claude · OpenAI · Ollama, retries, model fallback
                         ├─ nexora/rag/*          extract → chunk → embed → pgvector → cited answer
                         ├─ nexora/history.py     chat history (Chainlit data layer on PostgreSQL)
                         ├─ nexora/auth.py        password sign-in (PBKDF2), login throttling
                         └─ nexora/limits.py      rate limiting
                              │
                              ├──▶ Neon PostgreSQL + pgvector (documents, chunks, embeddings, history)
                              └──HTTPS + token──▶ MetaGPT worker (worker/app.py, FastAPI), optional and local
```

**Request flow for a document question:** the file is uploaded, then its text is extracted and cleaned, split into about 1,200-character chunks, embedded with `gemini-embedding-001` (768 dimensions), and stored in pgvector under the owner's ID. Each question is embedded, the top matches for the same owner are retrieved, and Gemini answers using them as untrusted reference text, citing `[1]`, `[2]`, and so on.

## Tech stack

| Layer | Technology |
|---|---|
| UI and server | [Chainlit](https://chainlit.io) 2.12 (Python 3.11) |
| LLM | Google Gemini via `google-genai`; optional Anthropic Claude, OpenAI, Ollama |
| Embeddings | `gemini-embedding-001` (or OpenAI `text-embedding-3-small`) |
| Database | PostgreSQL with `pgvector` (Neon), via `asyncpg` |
| Agents | [MetaGPT](https://github.com/FoundationAgents/MetaGPT) 1.0 in a FastAPI worker |
| Quality | pytest, ruff, GitHub Actions |
| Hosting | Render (free web service) |

## Repository structure

| Path | Purpose |
|---|---|
| [`nexora_app.py`](nexora_app.py) | Chainlit entry point: chat, uploads, commands, limits |
| [`nexora/providers.py`](nexora/providers.py) | Provider layer, error classification, retries, model fallback |
| [`nexora/config.py`](nexora/config.py) | Environment-variable settings and validation |
| [`nexora/rag/`](nexora/rag) | Extraction, chunking, embeddings, pgvector store, retrieval |
| [`nexora/history.py`](nexora/history.py) | Chat history data layer (creates its tables, survives DB outages) |
| [`nexora/auth.py`](nexora/auth.py) | Password hashing, account parsing, login throttling |
| [`worker/`](worker) | MetaGPT job worker (FastAPI, job queue, sandboxed runner) |
| [`tests/`](tests) | Unit tests with mocked providers, plus optional PostgreSQL integration tests |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | CI: lint and tests on every push and pull request |
| [`render.yaml`](render.yaml) | Render service definition (build and start commands) |
| [`.env.example`](.env.example) | All settings with safe placeholders |
| [`docs/screenshots/`](docs/screenshots) | README screenshots |

## Quick start (Windows)

Prerequisites: Python 3.11, Git, and a [Gemini API key](https://aistudio.google.com/apikey). PostgreSQL with pgvector is optional (for documents and history).

```bash
git clone https://github.com/rashmitha-j/Nexora-AI.git
```

```bash
cd Nexora-AI
```

```bash
py -3.11 -m venv .venv
```

```bash
.\.venv\Scripts\activate
```

```bash
pip install -r requirements-dev.txt
```

```bash
copy .env.example .env
```

Open `.env`, set `GEMINI_API_KEY`, then start the app:

```bash
chainlit run nexora_app.py
```

Open http://localhost:8000. On macOS or Linux, use `python3.11 -m venv .venv`, `source .venv/bin/activate` and `cp .env.example .env`.

## Environment variables

[`.env.example`](.env.example) lists every setting with defaults. The important ones:

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Gemini key (or configure `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OLLAMA_BASE_URL`) |
| `GEMINI_MODEL`, `GEMINI_FALLBACK_MODELS` | Main model and the models tried after it |
| `DATABASE_URL` | PostgreSQL with pgvector; turns on documents (and history when sign-in is on) |
| `NEXORA_USERS`, `CHAINLIT_AUTH_SECRET` | Sign-in accounts and session secret |
| `RATE_LIMIT_PER_MINUTE`, `GLOBAL_RATE_LIMIT_PER_MINUTE` | Per-visitor and site-wide message limits |
| `MAX_MESSAGE_CHARS`, `MAX_UPLOAD_MB`, `MAX_OUTPUT_TOKENS`, `LLM_TIMEOUT_SECONDS` | Request size, upload size, answer length and time limits |
| `METAGPT_WORKER_URL`, `METAGPT_WORKER_TOKEN` | Connection to the optional MetaGPT worker |

Secrets live only in `.env` (git-ignored) or the Render dashboard, never in the repository.

## Gemini quota and fallback

Gemini's free tier limits requests per minute and per day, **separately for each model**. Nexora handles this as follows:
- Each answer tries `GEMINI_MODEL` first (default `gemini-3.5-flash`), then `GEMINI_FALLBACK_MODELS` in order (default `gemini-3.8-flash`, `gemini-3.7-flash`, `gemini-3.6-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`).
- **Daily quota used up (429 with a per-day quota ID):** that model is skipped until Google's reported reset time, with no wasted calls.
- **Overloaded (503):** go straight to the next model.
- **Temporary errors (other 429s, 500s, timeouts):** up to 3 attempts with backoff, and only before any text has streamed, so users never see duplicated text.
- **Invalid key or bad request:** no fallback; a clear message is shown.
- The Gemini SDK's own retries are turned off, so retries happen in one place and requests are bounded.
- If every model fails, the user sees a plain message (for example "rate limit or quota reached"). Technical details go only to the server log.

## Sign-in and chat history

Off by default, so the site is public. To turn it on:

```bash
python -m nexora.auth
```

That prints a `username:hash` line for `NEXORA_USERS` (use commas between several users).

```bash
chainlit create-secret
```

That prints the value for `CHAINLIT_AUTH_SECRET`.

With both set (and `DATABASE_URL`), the site becomes login-only. Each user's conversations are saved in PostgreSQL and listed in the sidebar, and reopening one restores its context. Sign-out is in the user menu. Users can only open their own threads and documents.

Safeguards:
- Five failed logins lock that username for 15 minutes.
- A malformed account entry is skipped and logged rather than crashing the site.
- If `CHAINLIT_AUTH_SECRET` is missing, a temporary one is used, so sign-ins reset on each restart.

## Documents (RAG)

Attach a PDF, TXT, MD, CSV or JSON file to a message, then ask about it. Answers cite their sources.

Commands:
- `/docs` lists your documents.
- `/forget <n>` deletes one (after a confirmation).
- `/reindex` re-embeds your documents after changing the embedding model.
- `/usage` shows token usage for the chat.

How documents are handled:
- Tables are created automatically with `CREATE TABLE IF NOT EXISTS`; nothing is ever dropped.
- Re-uploading the same file re-indexes it instead of duplicating it.
- Extracted text has control characters (such as NUL) removed. Damaged, encrypted or scanned (image-only) PDFs get a clear message.
- **Anonymous visitors:** documents are private to the current chat (up to 5 files, about 300 KB of text each). They're deleted when the visitor starts a new chat, or after 24 hours.
- **Signed-in users:** documents are kept across chats.
- Questions are only embedded when the chat actually has documents, which saves quota.
- Original files aren't stored, only their extracted text and embeddings, so Render's temporary disk isn't relied on.

## MetaGPT worker (local only)

`/build <idea>` asks for confirmation, then sends the idea to a separate worker. There a MetaGPT team (product manager, architect, project manager and engineer) writes requirements, a design and code, and the result comes back as a zip.
- **Bounded:** one job at a time, a hard timeout per job, and a bounded queue.
- **Sandboxed:** each job runs in its own process, with its own folder and home directory.
- **No code execution:** roles that run generated code are excluded.

Setup (needs a second Python 3.11 environment):

```bash
py -3.11 -m venv .venv-worker
```

```bash
set PYTHON=.venv-worker\Scripts\python && sh worker/install.sh
```

Start it with `METAGPT_WORKER_TOKEN` and `METAGPT_API_KEY` set:

```bash
.venv-worker\Scripts\python -m uvicorn worker.app:app --port 8001
```

Then set `METAGPT_WORKER_URL=http://127.0.0.1:8001` and the same token for the app, and sign in.

**Current status:**
- The pipeline starts and produced a full requirements document in testing.
- A complete end-to-end run hasn't finished yet, because Gemini's free daily quota and overload ran out first.
- The worker isn't deployed, because MetaGPT needs more than Render's free 512 MB of memory.

Upstream issues fixed in this repo (no MetaGPT changes; all in configuration and wrapping):
- Gemini system prompts were dropped.
- The quick-think shortcut skipped the pipeline.
- The project path came from user text.
- `lancedb` and `pyarrow` dependency problems.

## Deploy on Render (free)

Settings on Render ([`render.yaml`](render.yaml) describes the same):

| Setting | Value |
|---|---|
| Build command | `pip install -r requirements.txt` |
| Start command | `chainlit run nexora_app.py --host 0.0.0.0 --port $PORT --headless` |
| Health check path | `/health` |
| Environment | `PYTHON_VERSION=3.11.9`, `GEMINI_API_KEY`, `DATABASE_URL` (optional: `NEXORA_USERS`, `CHAINLIT_AUTH_SECRET`) |

Automatic deploys are off on the current service. After pushing, open the service in the [Render dashboard](https://dashboard.render.com/) and use **Manual Deploy → Deploy latest commit**.

## Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push and pull request to `main`. It installs the dependencies on Python 3.11.9, runs `ruff check`, `ruff format --check` and `pytest`. CI needs no API keys or database: providers are mocked, and the PostgreSQL integration tests skip themselves unless `NEXORA_TEST_DATABASE_URL` is set. [See the runs](https://github.com/rashmitha-j/Nexora-AI/actions).

## Security

- **Server-side limits:**
  - Messages per minute: per signed-in user, or **per IP address** for anonymous visitors, so opening new chats doesn't reset the allowance.
  - A site-wide per-minute cap.
  - Maximum message length, upload size, document count and size, and answer length.
- **No client-chosen models:** the model dropdown only offers server-approved models, and the server rejects anything else sent over the socket. Cost and quota can't be bypassed by picking an arbitrary model.
- **One request at a time per chat,** so double-sending can't trigger parallel model calls.
- **Timeouts everywhere:**
  - Model calls, embeddings and database operations all have timeouts.
  - Retries are bounded and never unlimited.
  - Users always get a reply or a clear error.
- **Prompt-injection hardening:** retrieved document text is wrapped and marked as untrusted data, and tags inside documents are stripped.
- **Data isolation:** every document query is filtered by owner; chat threads are checked against their author.
- **Secrets stay server-side:**
  - API keys and database URLs live only in environment variables.
  - Logs and user-facing errors never include them.
  - The repository contains only placeholders.
- **Parameterized SQL** throughout (`asyncpg`); no debug or admin endpoints are exposed.
- **Known limitation:** rate limits are kept in memory, so they apply per instance (fine for one Render instance; use Redis if you scale out).

## Testing

```bash
python -m pytest
```

```bash
python -m ruff check .
```

```bash
python -m ruff format --check .
```

Tests cover configuration validation, password hashing and login throttling, rate limiting, streaming, retries and error classification, model fallback and the model allow-list, chunking and text cleaning, owner isolation in retrieval, the MetaGPT job queue and API, and chat-history degradation when the database is down.

To also run the PostgreSQL integration tests, point them at a **throwaway** database with pgvector:

```bash
set NEXORA_TEST_DATABASE_URL=postgresql://user:pass@host/db && python -m pytest tests/test_postgres_integration.py
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| Blank page, `/project/settings` returns 500 | Fixed: Chainlit's own data layer needed `asyncpg` and tables. The app now registers its own data layer |
| "rate limit or quota reached" | Every Gemini model is out of its daily quota; it resets daily. Check [AI Studio usage](https://aistudio.google.com/usage) |
| "is overloaded right now" | Google-side load; try again shortly. Fallback models are already tried |
| "Document storage is unavailable" | The database didn't respond (Neon may be waking up). Chat still works; retry |
| "couldn't read this PDF" / "no extractable text" | The file is damaged, encrypted or scanned. Use a text-based PDF or run OCR first |
| First visit is slow | Render free service waking up (about 1 minute) |

## Limitations and future work

- Free-tier limits: Render sleeps when idle, Gemini's daily quotas, and Neon's small storage.
- MetaGPT runs locally only and hasn't completed a full run yet.
- Rate limits are per instance (in memory).
- Possible next steps:
  - A Redis-backed rate limiter.
  - OCR for scanned PDFs.
  - Self-service sign-up.
  - A hosted worker once a suitable free option exists.

## Contributing

Issues and pull requests are welcome: [open an issue](https://github.com/rashmitha-j/Nexora-AI/issues). Please run `ruff check .`, `ruff format .` and `pytest` before submitting; CI runs the same checks.

## License

[MIT](LICENSE) © 2026 Rashmitha J. MetaGPT is used as a dependency under its own MIT license.

## Links

| | |
|---|---|
| Live app | https://nexora-ai-jcdz.onrender.com |
| Repository | https://github.com/rashmitha-j/Nexora-AI |
| Source code | https://github.com/rashmitha-j/Nexora-AI/tree/main |
| Issues | https://github.com/rashmitha-j/Nexora-AI/issues |
| Commit history | https://github.com/rashmitha-j/Nexora-AI/commits/main |
| GitHub Actions | https://github.com/rashmitha-j/Nexora-AI/actions |
| Render dashboard | https://dashboard.render.com/ |
| Google AI Studio | https://aistudio.google.com/ |
| Neon console | https://console.neon.tech/ |
