# Claudio

A RAG-powered code assistant CLI. Point it at a codebase and ask questions in
natural language — it indexes your source with tree-sitter, retrieves relevant
chunks from a hybrid (dense + sparse) Qdrant vector store, and answers with an
LLM agent that can read files, run commands, plan multi-step tasks, and call
MCP tools. Conversations persist across sessions, and repeated questions are
served from a Redis-backed semantic cache.

## Features

- **Hybrid RAG** over your codebase (dense + sparse retrieval via Qdrant + FastEmbed)
- **AST-aware chunking** with tree-sitter across many languages
- **Agent** with file, terminal, skills, and MCP tools
- **Persistent memory** per session (SQLite checkpointer) with auto-summarization
- **Semantic response cache** (Redis) with automatic invalidation on code changes
- **`/plan`** — generate and execute multi-step task plans

## Quickstart — try the CLI (no setup)

Run Claudio on any codebase with **no API key, no Qdrant, and no Docker**. It
routes LLM calls through a hosted proxy and uses an in-memory vector store, so a
single install is all it takes.

Requires **Python 3.11–3.13** and **[pipx](https://pipx.pypa.io/en/stable/installation/)**.

```bash
# 1. Install straight from GitHub
pipx install git+https://github.com/edgarjoel18/Claudio.git

# 2. Set the demo access token (provided alongside this link)
export CLAUDIO_API_KEY=<token>

# 3. Run inside any project you want to ask about
cd some-project
claudio
```

Then type a question (or `/ask <question>`), and `/exit` to quit. The token
routes through a rate-limited, spend-capped proxy — no OpenAI key required.

> Prefer your own OpenAI key instead of the proxy? Just `export OPENAI_API_KEY=...`
> (don't set `CLAUDIO_API_KEY`) and Claudio calls OpenAI directly.

## Development / full setup

For the complete hybrid-Qdrant + Redis experience and local development:

### Prerequisites

- **Python 3.11, 3.12, or 3.13** (not 3.14 yet)
- **[Poetry](https://python-poetry.org/docs/#installation)**
- An **OpenAI API key**
- A running **Qdrant** instance (enables hybrid retrieval; without it, Claudio
  falls back to the in-memory store automatically)
- **Redis** — optional; enables the semantic cache (the app runs fine without it)

The easiest way to run Qdrant and Redis locally is Docker:

```bash
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
docker run -d -p 6379:6379 --name redis  redis
```

## Setup

```bash
# 1. Clone
git clone https://github.com/<your-username>/claudio.git
cd claudio

# 2. Configure secrets
cp claudio/.env.example claudio/.env
#   then edit claudio/.env and set at least OPENAI_API_KEY and QDRANT_URL

# 3. Install dependencies
poetry install

# 4. Start Qdrant (and optionally Redis) — see Prerequisites

# 5. Run — from the root of the project you want to analyze
poetry run claudio
```

On first launch Claudio indexes the current working directory, so run it from
inside the codebase you want to ask about.

## Usage

Once the REPL is up:

| Command | Description |
| --- | --- |
| `<question>` | Ask a question about the codebase |
| `/show_index` | Show all indexed chunks |
| `/reindex` | Re-index the current directory (also clears the cache) |
| `/new_session` | Start a fresh conversation |
| `/switch <session_id>` | Resume a past session |
| `/session` | Show the current session id |
| `/plan <goal>` | Generate and execute a multi-step plan |
| `/task_status` | Show task progress for the active plan |
| `/exit` | Quit |

## Configuration

Runtime behavior (LLM/embedding models, RAG mode, cache thresholds, etc.) is set
in [`claudio/config.yml`](claudio/config.yml). Secrets and connection URLs come
from `claudio/.env`.
