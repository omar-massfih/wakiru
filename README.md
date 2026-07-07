# Wakiru

An agentic personal assistant built on **LangGraph + LangChain**, using the **Codex CLI**
as its reasoning/execution engine. This is the basic plumbing — a minimal LangGraph graph
that drives `codex exec` and returns its reply — meant to be extended with real capabilities.

> **On the name:** *Wakiru* blends the Arabic *wakīl* (وكيل — "agent," one who acts on your
> behalf) with the Japanese *wakaru* (分かる — "to understand"): an agent that understands you.

## How it works

```
HTTP (/chat)  \
               ->  LangGraph StateGraph  ->  CodexChatModel  ->  `codex exec` subprocess
Telegram bot  /
```

- **`codex_runner.py`** — thin subprocess wrapper around `codex exec` (captures the final message).
- **`llm.py`** — the **provider abstraction**. `build_model()` selects a LangChain `BaseChatModel`
  by `LLM_PROVIDER`. `codex` is wired (`CodexChatModel` over the runner); `openai` and `anthropic`
  are ready-to-fill stubs (each stub documents the exact steps to enable it).
- **`agent.py`** — the LangGraph graph: `recall -> codex -> summarize`, with a SQLite
  checkpointer for conversation history.
- **`chat.py`** — the channel-agnostic core: one turn of conversation plus its
  post-reply upkeep (memory, summary folding, calendar, consolidation), shared by
  every channel so they all behave identically.
- **`api.py`** — FastAPI app: `GET /health`, `POST /chat`, `GET /memory` and
  `POST /memory/consolidate` for inspecting and consolidating the brain, `GET /calendar`,
  and `POST /reminders/run` for firing due reminders.
- **`telegram.py`** — the Telegram channel (see below): a stdlib-only long-polling
  bridge started alongside the server when a bot token is configured.

Codex is itself an autonomous agent (its own model, tools, and sandbox), so tool-use happens
inside Codex rather than as LangChain tools. Live web search is one such tool: set
`CODEX_WEB_SEARCH=true` to pass Codex's global `--search` flag, which turns on the native
Responses `web_search` tool (off by default — extra tokens/latency per turn).

## The brain (memory)

Memory lives in `memory/` and has two layers:

- **Working memory** — the conversation, persisted per `thread_id` by the LangGraph
  SQLite checkpointer. Once it grows past a threshold, older turns are folded into a
  rolling summary (in the background, after the reply) so context stays bounded.
- **Long-term memory** — durable markdown notes on disk (the source of truth), in three
  cognitive kinds:
  - `semantic/` — durable facts, preferences, goals ("the user prefers Norwegian").
  - `procedural/` — learned how-to knowledge ("deploy with uv").
  - `episodic/` — timestamped traces of what happened (decays and is pruned over time).

  A local, offline vector index (`sqlite-vec` + fastembed, multilingual by default)
  is *derived* from the files and rebuilt from them on startup (`reindex`), so hand-edits
  never drift.

How it learns (`src/assistant/memory/`):

- **Recall** (`recall.py`) — embeds the incoming message, pulls a candidate pool, and
  re-ranks it by blending similarity + recency + reuse + salience. Recalling a note
  *reinforces* it, so useful memories rise over time.
- **Online learning** (`learn.py`) — after each turn (in the background), it writes an
  episodic trace (skipping small talk and repeated exchanges) and runs a **reconciling**
  extraction: Codex sees the exchange *and the memories already relevant to it*, then
  emits `save` / `update` / `forget` ops. Seeing current memory lets it fix
  contradictions in place ("moved from Oslo to Bergen") instead of piling up duplicates.
- **Consolidation** (`consolidate.py`) — periodically (or via `POST /memory/consolidate`)
  it decays and prunes old episodes, promotes recurring patterns into semantic/procedural
  memory, merges duplicates, resolves contradictions store-wide, and flushes reinforcement
  counters back into the files. It also keeps long-term memory *finite*: durable notes
  that never get recalled fade in ranking priority, and each kind is held under a hard
  note cap (lowest retention value evicted first). Only the most valuable note titles are
  injected into the prompt each turn, so context stays bounded no matter how much the
  assistant has learned.

## Talk to it on Telegram

The assistant is also a Telegram bot, so it lives in your pocket instead of behind
`curl`. The channel long-polls the Bot API (the server *pulls* messages), so it works
from a laptop behind NAT — no public URL, no webhook, no open port.

Setup:

1. Message [@BotFather](https://t.me/BotFather) on Telegram, send `/newbot`, and copy
   the token it gives you into `.env`:

   ```sh
   TELEGRAM_BOT_TOKEN=123456:ABC-your-token
   ```

2. Start the server and message your new bot. That's it — the first chat to reach the
   bot is **paired** (trust-on-first-use, persisted in `memory/telegram_chats.json`)
   and answered from then on; every other chat gets silence.

Since first contact wins, keep the bot's username to yourself until you've paired.
The cautious can skip pairing entirely and pin chats up front via
`TELEGRAM_ALLOWED_CHAT_IDS=[...]` in `.env` (merged with the paired set); un-pair by
deleting `memory/telegram_chats.json`.

Each chat maps to a stable conversation thread (`telegram:<chat_id>`), so working
memory and the rolling summary persist across restarts. Proactive reminders are
pushed to the same chats, so "Dentist in 1 hour" lands where you already talk.
Replies longer than Telegram's 4096-char limit are split at newline boundaries.

## Proactive reminders

The calendar can nudge you *before* an event, unprompted — "Dentist in 1 hour" —
instead of only answering when you ask. A ticker inside the server (no cron needed)
checks the calendar every `REMINDER_TICK_SECONDS` and, for each event entering the
`REMINDER_LEAD_MINUTES` window, pushes one message to `REMINDER_WEBHOOK_URL`. A small
SQLite ledger guarantees each reminder fires exactly once (a rescheduled event nudges
again for its new time).

```sh
# Point it at an ntfy topic (install the ntfy app and subscribe to the same topic):
REMINDER_WEBHOOK_URL=https://ntfy.sh/your-private-topic
```

Delivery fans out to every configured channel: the webhook (any endpoint that accepts
a plain POST — ntfy, a Discord/Slack webhook, … — the message is the body, the event
title the `Title` header) and, when the Telegram channel is set up, every allowed
Telegram chat. Configure neither and reminders are still computed, just not pushed.

`POST /reminders/run` runs one pass on demand (handy for testing, and idempotent thanks
to the ledger). Prefer external cron? Set `REMINDER_TICK_SECONDS=0` to disable the
built-in ticker and curl that endpoint on a schedule instead.

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
`CODEX_WEB_SEARCH` is similarly off by default; turn it on deliberately too.

## Adding an API-backed provider later

`LLM_PROVIDER=openai` or `anthropic` are registered but stubbed. To enable one, open the
matching `_build_*` function in `llm.py` and follow the inline steps (add the LangChain
integration package, add config fields, return the chat model). No other file changes.

## Roadmap / not yet wired

- OpenAI / Claude providers at the `llm.py` stubs.
- Additional tools, routing nodes, streaming, and API auth.

> **Note on the embedding model:** the default `EMBEDDING_MODEL` is
> `intfloat/multilingual-e5-large` (1024-dim, strong Norwegian recall). Its first use
> downloads ~2GB into the HuggingFace cache; set a smaller model (e.g.
> `sentence-transformers/all-MiniLM-L6-v2`) if you don't need multilingual recall.
