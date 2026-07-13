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
  by `LLM_PROVIDER`. `codex` (default), `openai`, and `anthropic` are all wired; the API-backed
  providers read `LLM_API_KEY` / `LLM_MODEL` (and `LLM_BASE_URL` for openai).
- **`agent.py`** — the LangGraph graph: `START -> recall -> agenda -> tasks -> profile -> agent`,
  then an `agent <-> tools` loop (bounded by `TOOL_MAX_ROUNDS`) until the model answers in plain
  text. Conversation history persists via a SQLite checkpointer by default, or Postgres
  checkpoints when `STORAGE_BACKEND=postgres`. Working-memory summarization runs off the reply
  path (in the background), not as a graph node.
- **`tools.py`** — the tool registry the model acts through: calendar
  (create/reschedule/cancel/skip/move), tasks (add/complete/update/remove), memory
  (remember/forget/search), document search, and email (list/read/draft; `send_email` exists
  only when `ENABLE_EMAIL_SEND` is on). Each tool wraps a guarded write path, so ambiguity
  guards, conflict notes, and the undo ledger all apply.
- **`chat.py`** — the channel-agnostic core: one turn of conversation plus its
  post-reply upkeep (memory learning, summary folding, consolidation), shared by
  every channel so they all behave identically. Calendar/task writes happen through
  the tool loop during the turn.
- **`api.py`** — FastAPI app: `GET /health`, `POST /chat`, `POST /chat/stream` (SSE),
  `GET /memory` and `POST /memory/consolidate` for inspecting and consolidating the brain,
  `GET /calendar`, `GET /tasks` for the to-do list, `POST|GET /documents` (+ `/documents/search`,
  `/documents/{id}/summarize`) for documents, and `POST /reminders/run` for firing
  due reminders (events and tasks). Swagger UI stays at `/docs`.
- **`tasks/`** — the to-do list: a store, a read path (open tasks injected each turn), a
  tool-driven write path (add/complete/update/remove), a due-task reminder path, and an undo
  ledger — mirroring the `calendar/` package for work with no fixed time and a done state.
- **`docs/`** — ingested documents, chunked and embedded into their own `docs.db` vector
  index. The most relevant chunks ride in on the `recall` node each turn (so "what did I
  write about X" works), and a whole document can be summarized on demand.
- **`mail/`** — the only subsystem that talks to an external service, and the only one
  **off by default** (`ENABLE_EMAIL`). Stdlib IMAP/SMTP with XOAUTH2 or an app password.
  Reads use `BODY.PEEK` (never marks your mail read) and are surfaced on request only,
  not injected each turn. Drafting is the default write; **sending** needs a second,
  independent switch (`ENABLE_EMAIL_SEND`) and never happens in the background.
- **`telegram.py`** — the Telegram channel (see below): a stdlib-only long-polling
  bridge started alongside the server when a bot token is configured. Free text goes to
  the model; the slash commands `/help`, `/tasks`, `/calendar`, `/memory`, and `/reset`
  are answered locally (no model call), and `"undo"` reverts the last calendar/task write.
- **`cli.py`** — a terminal REPL over the same `chat.py` seam (`assistant-cli`), for chatting
  without the HTTP server or a bot token; it uses one stable thread so history persists.
- **`slack.py`** — the Slack channel: an Events API bridge (`POST /slack/events`) authenticated
  by HMAC signature over the raw body, or — when `SLACK_APP_TOKEN` is set — a Socket Mode
  websocket that needs no public URL (works behind NAT, like Telegram). Only allowlisted user
  ids are answered — with no pairing handshake, an empty allowlist fails closed. Reminders can
  fan out to Slack too.
- **`webui.py`** — a single self-contained HTML page at `GET /ui` that streams replies from
  `/chat/stream`. No build step, no CDN. Prompts for `API_TOKEN` when one is configured.

The assistant's own tools work uniformly across providers: the API-backed providers use
native function calling, while the Codex provider emulates `bind_tools` over plain text —
tool schemas ride in the prompt and the model marks calls with a fenced ` ```tool_call `
block that is parsed back into structured calls (and never leaks to the user; streaming
withholds it). Codex is additionally an autonomous agent of its own (model, tools, sandbox):
set `CODEX_WEB_SEARCH=true` to pass its global `--search` flag, which turns on the native
Responses `web_search` tool (off by default — extra tokens/latency per turn).

## The brain (memory)

Memory lives in `memory/` by default. In deployment, set `STORAGE_BACKEND=postgres` with `DATABASE_URL` from a Vercel Marketplace Postgres provider such as Neon to store conversation checkpoints, long-term memory, and document vectors in Postgres/pgvector. It has two layers:

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

2. Start the server and message your new bot. It replies asking for a **pairing
   code**, which is printed in the server log — send that code back and the chat
   is paired (persisted in `memory/telegram_chats.json`) and answered from then
   on; every other chat gets silence.

The code round-trip means only whoever can read the server log can claim the bot,
so a stranger finding the bot first can't hijack it. You can also skip the
handshake entirely and pin chats up front via `TELEGRAM_ALLOWED_CHAT_IDS=[...]`
in `.env` (merged with the paired set); un-pair by deleting
`memory/telegram_chats.json`.

Each chat maps to a stable conversation thread (`telegram:<chat_id>`), so working
memory and the rolling summary persist across restarts. Proactive reminders are
pushed to the same chats, so "Dentist in 1 hour" lands where you already talk.
Replies longer than Telegram's 4096-char limit are split at newline boundaries.

## Proactive reminders

The calendar can nudge you *before* an event, unprompted — "Dentist in 1 hour" —
instead of only answering when you ask. Every delivered push (reminders and the
daily briefing) is also recorded into each paired Telegram chat's working memory
(`ENABLE_PROACTIVE_LOOP_IN`, on by default), so the conversation knows what it
already sent you — "what was that reminder about?" just works, and the assistant
can follow up on its own nudges. A ticker inside the server (no cron needed)
checks the calendar every `REMINDER_TICK_SECONDS` and, for each event entering the
`REMINDER_LEAD_MINUTES` window, pushes one message to `REMINDER_WEBHOOK_URL`. A small
SQLite ledger guarantees each reminder fires exactly once (a rescheduled event nudges
again for its new time).

```sh
# Point it at an ntfy topic (install the ntfy app and subscribe to the same topic):
REMINDER_WEBHOOK_URL=https://ntfy.sh/your-private-topic
```

By default each configured lead fires once. Set `REMINDER_REPEAT_MINUTES` (e.g. `15`)
to instead re-nudge on that cadence from the outermost lead onward, until the event
starts — so an event no longer goes quiet after the first "in 1 hour". Dated tasks
also keep nagging *past* their due time ("Task overdue: … (30 min ago)") until you
mark them done, bounded by `REMINDER_OVERDUE_MAX_MINUTES` (default 24h).

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

# 3. Run, mounting your Codex credentials (Codex needs a writable CODEX_HOME at
#    runtime for its app-server client; auth still lives on the host) and the
#    memory directory (with the default local backend the assistant's entire
#    brain lives there — without the mount it dies with the container):
docker run --rm -p 8000:8000 \
  -v "$HOME/.codex:/home/assistant/.codex" \
  -v "$PWD/memory:/app/memory" \
  -e API_TOKEN=change-me \
  agentic-assistent

curl localhost:8000/health
```

Notes:
- Host-mounted `./memory` (and `./models` when using docker-compose) must be writable
  by uid 1000 — the container runs as the non-root `assistant` user.
- `API_TOKEN` is required in the container: the image binds `0.0.0.0`, and the server
  refuses to start on a non-loopback bind without a token. Set `ALLOW_UNAUTHENTICATED=1`
  only if something in front (reverse proxy, VPN) does the authentication.
- The default `CODEX_SANDBOX=read-only` is safest in a container. If Codex needs to run shell
  commands and the container can't apply its OS sandbox, either widen the sandbox via
  `-e CODEX_SANDBOX=workspace-write` or give the container the privileges Codex's sandbox needs.
- Pass any settings as env, e.g. `-e LLM_PROVIDER=codex -e CODEX_MODEL=...`.

## Test

```sh
uv run pytest
```

Smoke tests build the graph and hit `/health` without invoking Codex.

The real-embedder recall tests are skipped by default (they load the ~2GB
embedding model); run them with `REAL_EMBEDDINGS=1 uv run pytest tests/test_recall_real.py`.

## Vercel / Neon storage

For durable deployment storage, provision Neon from Vercel Marketplace and expose
its connection string as `DATABASE_URL`:

```sh
vercel install neon
```

Then run the assistant with:

```sh
STORAGE_BACKEND=postgres
DATABASE_URL=postgres://...
```

The local backend remains the default for development. The Postgres backend stores
LangGraph checkpoints, long-term memory notes, memory embeddings, documents, and
document chunk embeddings in Postgres using `pgvector`; `memory/MEMORY.md` becomes
an export artifact instead of the source of truth.

## Configuration

See `.env.example`. Notably `CODEX_SANDBOX` defaults to `read-only`; widen it deliberately.
`CODEX_WEB_SEARCH` is similarly off by default; turn it on deliberately too.

Set `API_TOKEN` to require `Authorization: Bearer <token>` on every endpoint except
`/health`. Unset, the server trusts anyone who can reach the port — fine on the
default loopback bind, but on any non-loopback bind (the Docker image binds
`0.0.0.0`) startup fails without a token. `ALLOW_UNAUTHENTICATED=1` overrides the
refusal for deployments where a reverse proxy or VPN does the authentication.

## Using an API-backed provider

Set `LLM_PROVIDER=openai` or `anthropic` and `LLM_API_KEY=<your key>`. Optionally
override `LLM_MODEL` (defaults: `gpt-4o` for openai, `claude-opus-4-8` for anthropic)
and, for openai-compatible endpoints, `LLM_BASE_URL`. Unlike the default `codex`
provider, these support token streaming (see below).

## Streaming

`POST /chat/stream` returns the reply as Server-Sent Events: `data: <text>` frames
as the model produces the reply, a final `event: done` frame with the `thread_id`,
and `event: error` if the model fails mid-stream. Post-reply upkeep runs once, after
the stream closes, exactly as `POST /chat` does. Every provider streams: the `codex`
provider parses the CLI's `--json` event stream and emits each agent message as it
completes (the CLI does not expose token deltas, so granularity is per message).

## Newer capabilities

- **Daily briefing** — one digest per day (agenda + due tasks + unread mail) pushed
  through the reminder channels at `BRIEFING_TIME`; `POST /briefing/run` on demand.
- **Personalization** — durable memories tagged `profile` (working hours, locations,
  quiet hours, tone) are injected every turn, and stated quiet hours hold
  reminders/briefings until morning.
- **External calendar sync** — `CALENDAR_ICS_URLS` mirrors Google/Outlook/CalDAV
  ICS feeds into the local calendar (read-only, one-way) every
  `CALENDAR_SYNC_MINUTES`; agenda, conflicts, and reminders see the real calendar.
- **Richer document ingest** — `POST /documents/upload` accepts PDF/DOCX/text
  files; `POST /documents` can also take a `url` (opt-in, `ENABLE_DOCS_URL_INGEST`).
- **Voice notes** — with `ENABLE_VOICE=true`, Telegram voice messages are
  transcribed locally (faster-whisper) and answered like typed text.

## Roadmap / not yet wired

- Two-way calendar sync (CalDAV writes) — the current sync is deliberately pull-only.

> **Note on the embedding model:** the default `EMBEDDING_MODEL` is
> `intfloat/multilingual-e5-large` (1024-dim, strong Norwegian recall). Its first use
> downloads ~2GB into the HuggingFace cache; set a smaller model (e.g.
> `sentence-transformers/all-MiniLM-L6-v2`) if you don't need multilingual recall.
