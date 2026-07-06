# Wakiru

An agentic personal assistant built on **LangGraph + LangChain**, using the **Codex CLI**
as its reasoning/execution engine. This is the basic plumbing — a minimal LangGraph graph
that drives `codex exec` and returns its reply — meant to be extended with real capabilities.

> **On the name:** *Wakiru* blends the Arabic *wakīl* (وكيل — "agent," one who acts on your
> behalf) with the Japanese *wakaru* (分かる — "to understand"): an agent that understands you.

## How it works

```
HTTP (/chat)  ->  LangGraph StateGraph  ->  CodexChatModel  ->  `codex exec` subprocess
```

- **`codex_runner.py`** — thin subprocess wrapper around `codex exec` (captures the final message).
- **`llm.py`** — the **provider abstraction**. `build_model()` selects a LangChain `BaseChatModel`
  by `LLM_PROVIDER`. `codex` is wired (`CodexChatModel` over the runner); `openai` and `anthropic`
  are ready-to-fill stubs (each stub documents the exact steps to enable it).
- **`agent.py`** — a single-node LangGraph graph; the extension point for more nodes/routing/memory.
- **`api.py`** — FastAPI app: `GET /health`, `POST /chat`.

Codex is itself an autonomous agent (its own model, tools, and sandbox), so tool-use happens
inside Codex rather than as LangChain tools.

## Prerequisites

- Python 3.13, [`uv`](https://github.com/astral-sh/uv)
- The [Codex CLI](https://github.com/openai/codex), authenticated:

  ```sh
  codex login          # ChatGPT sign-in — no API key needed
  codex login status   # should print "Logged in using ChatGPT"
  ```

## Setup

```sh
uv sync
cp .env.example .env   # optional — all settings have defaults
```

## Run

```sh
uv run uvicorn assistant.api:app --reload
```

```sh
curl localhost:8000/health
# {"status":"ok"}

curl -sX POST localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"message":"What time is it right now?"}'
# {"reply":"..."}
```

## Docker

The image bundles the Codex CLI but **not** its credentials — Codex auth lives in `~/.codex`
on your host and must be mounted in.

```sh
# 1. Authenticate on the host first (once):
codex login

# 2. Build:
docker build -t agentic-assistent .

# 3. Run, mounting your Codex credentials read-only:
docker run --rm -p 8000:8000 \
  -v "$HOME/.codex:/root/.codex:ro" \
  agentic-assistent

curl localhost:8000/health
```

Notes:
- The default `CODEX_SANDBOX=read-only` is safest in a container. If Codex needs to run shell
  commands and the container can't apply its OS sandbox, either widen the sandbox via
  `-e CODEX_SANDBOX=workspace-write` or give the container the privileges Codex's sandbox needs.
- Pass any settings as env, e.g. `-e LLM_PROVIDER=codex -e CODEX_MODEL=...`.

## Test

```sh
uv run pytest
```

Smoke tests build the graph and hit `/health` without invoking Codex.

## Configuration

See `.env.example`. Notably `CODEX_SANDBOX` defaults to `read-only`; widen it deliberately.

## Adding an API-backed provider later

`LLM_PROVIDER=openai` or `anthropic` are registered but stubbed. To enable one, open the
matching `_build_*` function in `llm.py` and follow the inline steps (add the LangChain
integration package, add config fields, return the chat model). No other file changes.

## Roadmap / not yet wired

- OpenAI / Claude providers at the `llm.py` stubs.
- Conversation memory / persistence (a LangGraph checkpointer).
- Additional tools, routing nodes, streaming, and API auth.
