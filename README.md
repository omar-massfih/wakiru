# Wakiru

An agentic personal assistant built on **LangGraph + LangChain**, using the **Codex CLI**
as its reasoning/execution engine. This is the basic plumbing — a minimal LangGraph graph
that drives `codex exec` and returns its reply — meant to be extended with real capabilities.

> **On the name:** *Wakiru* blends the Arabic *wakīl* (وكيل — "agent," one who acts on your
> behalf) with the Japanese *wakaru* (分かる — "to understand"): an agent that understands you.

## How it works

```
Telegram bot  \
               ->  LangGraph StateGraph  ->  CodexChatModel  ->  `codex exec` subprocess
Slack / CLI   /
```

Wakiru runs as a **living daemon**: a long-lived process that is nothing but its
background loops (the heartbeat that drives reminders, digests, and data
refreshes; the nightly memory pass) and its chat channels (Telegram, Slack).
There is no web server and no REST API — the only HTTP is a bare, unauthenticated
`GET /health` for container liveness probes. What used to be an endpoint is now
either the heartbeat's own job or a Telegram admin command.

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
  (remember/forget/search), document search + whole-document summarize, web
  (`read_url`/`ingest_url` when `ENABLE_DOCS_URL_INGEST` is on), and email
  (list/search/read/draft, threaded replies, archive, mark read, label, attachment
  ingestion into documents; `send_email`/`send_reply` exist only when `ENABLE_EMAIL_SEND`
  is on). Each tool wraps a guarded write path, so ambiguity guards, conflict notes, and
  the undo ledger all apply.
- **`chat.py`** — the channel-agnostic core: one turn of conversation plus its
  post-reply upkeep (memory learning, summary folding, consolidation), shared by
  every channel so they all behave identically. Calendar/task writes happen through
  the tool loop during the turn.
- **`daemon.py`** — the process entry point (`assistant`): starts the heartbeat
  loop, the nightly sleep loop, Telegram long-polling, and Slack Socket Mode, then
  waits on a signal to shut down cleanly. Its only HTTP is a bare `GET /health`
  liveness endpoint on `HEALTH_PORT`. On-demand jobs that used to be REST
  endpoints are Telegram admin commands (`/heartbeat`, `/briefing`, `/review`,
  `/sync`, `/sleep`); everything else is the heartbeat's own job.
- **`tasks/`** — the to-do list: a store, a read path (open tasks injected each turn), a
  tool-driven write path (add/complete/update/remove), a due-task reminder path, and an undo
  ledger — mirroring the `calendar/` package for work with no fixed time and a done state.
- **`docs/`** — ingested documents, chunked and embedded into their own `docs.db` vector
  index. The most relevant chunks ride in on the `recall` node each turn (so "what did I
  write about X" works), and a whole document can be summarized on demand.
- **`mail/`** — the only subsystem that talks to an external service, and the only one
  **off by default** (`ENABLE_EMAIL`). Stdlib IMAP/SMTP with XOAUTH2 or an app password.
  Reads use `BODY.PEEK` (never marks your mail read — that is its own deliberate tool) and
  are surfaced on request only, not injected each turn. The assistant can also *manage* the
  mailbox: threaded replies (drafted by default), archive (recoverable — Gmail keeps the
  message in All Mail), mark read/unread, and labels; every mutation lands in an audit
  ledger (`mail.db`). The background heartbeat can triage the inbox too, but only when
  `EMAIL_TRIAGE_MAX_ACTIONS` opts in, capped per wake — and **sending** needs a second,
  independent switch (`ENABLE_EMAIL_SEND`) and never happens in the background.
- **`telegram.py`** — the Telegram channel (see below): a stdlib-only long-polling
  bridge started by the daemon when a bot token is configured. Everything goes to
  the model — slash commands like `/tasks` or `/memory` become natural turns it answers
  itself, and `"undo"` makes it call the `undo` tool. Answered locally instead: `/reset`
  (and the pairing handshake), so it works even when the model or history is broken, and
  the admin commands (`/heartbeat`, `/briefing`, `/review`, `/sync`, `/sleep`) that trigger
  the on-demand background jobs.
- **`cli.py`** — a terminal REPL over the same `chat.py` seam (`assistant-cli`), for chatting
  without a bot token; it uses one stable thread so history persists.
- **`slack.py`** — the Slack channel: a Socket Mode websocket (needs `SLACK_APP_TOKEN`)
  that requires no public URL and works behind NAT, like Telegram. Only allowlisted user
  ids are answered — with no pairing handshake, an empty allowlist fails closed. Proactive
  pushes can fan out to Slack too. While a turn runs, the incoming message gets an ⏳ reaction
  as an "I'm on it" signal — grant the app the `reactions:write` scope for this (without it the
  reaction is silently skipped and everything else still works).

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
- **Consolidation** (`consolidate.py`) — periodically (riding the nightly sleep pass)
  it decays and prunes old episodes, promotes recurring patterns into semantic/procedural
  memory, merges duplicates, resolves contradictions store-wide, and flushes reinforcement
  counters back into the files. It also keeps long-term memory *finite*: durable notes
  that never get recalled fade in ranking priority, and each kind is held under a hard
  note cap (lowest retention value evicted first). Only the most valuable note titles are
  injected into the prompt each turn, so context stays bounded no matter how much the
  assistant has learned.

## Persona & voice

The assistant speaks with one voice everywhere (`src/assistant/persona.py`). The
register is configurable with `PERSONA_STYLE`: `warm` (default — a natural,
direct personal assistant that matches your energy and allows itself a moment
of warmth), `neutral` (professional and plain), or `minimal` (terse). The
prompt stays byte-stable per style, so provider prompt caching keeps working.

Tone also personalizes over time: communication preferences you state in
conversation (language, brevity, humor, when not to be disturbed) are captured
as `profile`-tagged memories and injected every turn. A good way to seed this
after a fresh deploy is one onboarding message, e.g.:

> Remember how I like to work: answer me in Norwegian unless I write in
> English, keep replies short on weekdays, humor is welcome, and don't ping me
> between 22:00 and 07:30.

The extractor turns that into profile notes; a stated quiet-hours preference
overrides the `QUIET_HOURS_DEFAULT` window (22:00–07:30 unless you change it),
during which reminders, the briefing, and heartbeat check-ins hold.

For the full "feels like a human assistant" experience on a personal deploy,
also enable the proactive layer: `ENABLE_BRIEFING=true`, `ENABLE_HEARTBEAT=true`,
and `HEARTBEAT_CONTACT_GAP_HOURS=24`.

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

The calendar can nudge you *before* an event, unprompted — "Heads up: Dentist
at 14:00 (in 1 hour)." — instead of only answering when you ask. Nudges are
composed by the model in the assistant's own voice, with its memory and agenda
in context; if the model fails, deterministic templates
(`src/assistant/phrasing.py`) carry the nudge instead, so a claimed reminder
is never lost. Every delivered push (reminders and the
daily briefing) is also recorded into each paired Telegram chat's working memory,
and into Slack conversations living in `SLACK_NOTIFY_CHANNEL`
(`ENABLE_PROACTIVE_LOOP_IN`, on by default), so the conversation knows what it
already sent you — "what was that reminder about?" just works, and the assistant
can follow up on its own nudges. The heartbeat (no cron needed) wakes right when
an event enters its `REMINDER_LEAD_MINUTES` window, surfaces the due reminder in
its situation report, and the model decides how to phrase (or bundle) the nudge.
A small SQLite ledger guarantees each reminder surfaces exactly once (a
rescheduled event nudges again for its new time).

```sh
# Point it at an ntfy topic (install the ntfy app and subscribe to the same topic):
REMINDER_WEBHOOK_URL=https://ntfy.sh/your-private-topic
```

By default each configured lead fires once. Set `REMINDER_REPEAT_MINUTES` (e.g. `15`)
to instead re-nudge on that cadence from the outermost lead onward, until the event
starts — so an event no longer goes quiet after the first "in 1 hour". Dated tasks
also keep nagging *past* their due time ("Still open: … — was due 30 min ago.")
until you mark them done, bounded by `REMINDER_OVERDUE_MAX_MINUTES` (default 24h)
*and* `REMINDER_OVERDUE_MAX_NUDGES` (default 4), whichever is hit first, so a
forgotten task can't chase you dozens of times. A purely informational one-time
reminder ("remind me the session resets at 14:50") is instead recorded as a
*notify-only* task: it fires once at its time and never nags overdue. And each
heartbeat wake is shown what was already pushed recently
(`HEARTBEAT_DEDUP_PUSH_HOURS`, default 6) so it doesn't re-send the same nudge in
different words.

Delivery fans out to every configured channel: the webhook (any endpoint that accepts
a plain POST — ntfy, a Discord/Slack webhook, … — the message is the body, the event
title the `Title` header) and, when the Telegram channel is set up, every allowed
Telegram chat. Configure neither and the nudge is composed but has nowhere to land.

The `/heartbeat` Telegram command runs one wake on demand (handy for testing, and
idempotent thanks to the ledgers).

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

Wakiru is a daemon — start it and talk to it over a chat channel (Telegram/Slack),
or use the terminal REPL for a quick local chat with no channel at all.

```sh
# The living daemon: heartbeat + nightly pass + whatever channels are configured.
# Set TELEGRAM_BOT_TOKEN (or SLACK_APP_TOKEN + SLACK_BOT_TOKEN) in .env first.
uv run assistant

# Or chat locally in the terminal, no bot token needed:
uv run assistant-cli
```

```sh
# The only HTTP surface is a liveness probe:
curl localhost:8000/health
# {"status":"ok"}
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
  -e TELEGRAM_BOT_TOKEN=... \
  agentic-assistent

curl localhost:8000/health
```

Notes:
- Host-mounted `./memory` (and `./models` when using docker-compose) must be writable
  by uid 1000 — the container runs as the non-root `assistant` user.
- The daemon has no REST surface, so there is no `API_TOKEN`: configure at least one
  chat channel (`TELEGRAM_BOT_TOKEN`, or `SLACK_APP_TOKEN` + `SLACK_BOT_TOKEN`) so you
  can reach it. The published `:8000` is only the `/health` liveness endpoint.
- The default `CODEX_SANDBOX=read-only` is safest in a container. If Codex needs to run shell
  commands and the container can't apply its OS sandbox, either widen the sandbox via
  `-e CODEX_SANDBOX=workspace-write` or give the container the privileges Codex's sandbox needs.
- Pass any settings as env, e.g. `-e LLM_PROVIDER=codex -e CODEX_MODEL=...`.

## Test

```sh
uv run pytest
```

Smoke tests build the graph and exercise the daemon's `/health` liveness
endpoint without invoking Codex.

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

The daemon exposes no REST surface, so there is no token to configure — access is
whoever can message the bot. Lock that down at the channel: Telegram pairs to the
first chat and ignores the rest (or pin `TELEGRAM_ALLOWED_CHAT_IDS`), and Slack
answers only `SLACK_ALLOWED_USER_IDS`. `HEALTH_PORT` (default 8000) serves only the
unauthenticated liveness probe; set it to 0 to disable even that.

## Using an API-backed provider

Set `LLM_PROVIDER=openai` or `anthropic` and `LLM_API_KEY=<your key>`. Optionally
override `LLM_MODEL` (defaults: `gpt-4o` for openai, `claude-opus-4-8` for anthropic)
and, for openai-compatible endpoints, `LLM_BASE_URL`.

## Newer capabilities

- **Daily briefing** — one digest per day (agenda + due tasks + unread mail) pushed
  through the chat channels at `BRIEFING_TIME`; the `/briefing` Telegram command on demand.
- **Weekly review** — with `ENABLE_WEEKLY_REVIEW=true`, one look-back +
  week-ahead digest per week (tasks completed, habit streaks, and spending from
  the last seven days; calendar, due tasks, trips, birthdays, and subscription
  renewals for the next seven), due at `WEEKLY_REVIEW_DAY` + `WEEKLY_REVIEW_TIME`
  (default Sunday 17:00), exactly once per ISO week; the `/review` Telegram command
  on demand.
- **Personalization** — durable memories tagged `profile` (working hours, locations,
  quiet hours, tone) are injected every turn, and quiet hours (a stated
  preference, else `QUIET_HOURS_DEFAULT`) hold reminders/briefings until morning.
- **Configurable voice** — `PERSONA_STYLE` selects the reply register
  (warm/neutral/minimal); see "Persona & voice" above.
- **External calendar sync** — `CALENDAR_ICS_URLS` mirrors Google/Outlook/CalDAV
  ICS feeds into the local calendar (read-only, one-way) every
  `CALENDAR_SYNC_MINUTES`; agenda, conflicts, and reminders see the real calendar.
- **Richer document ingest** — send a PDF/DOCX/text file to the bot and it is
  extracted, chunked, and embedded; the model can also read a `url` into the store
  with its document tools (opt-in, `ENABLE_DOCS_URL_INGEST`).
- **Voice notes** — with `ENABLE_VOICE=true`, Telegram voice messages are
  transcribed locally (faster-whisper) and answered like typed text.
- **People / contacts (lightweight CRM)** — with `ENABLE_PEOPLE=true`, a store
  of the people you know (relationship, keep-in-touch cadence, last contact,
  birthday, notes). A compact roster rides in each turn with anyone overdue for
  contact or with a birthday soon flagged first, so "who is this meeting with?"
  and "haven't spoken to Kari in a while" both work; the assistant manages it
  with the people tools (add / update / log-contact / remove), every write
  undoable. Birthdays within `PEOPLE_BIRTHDAY_LEAD_DAYS` fire a proactive
  reminder and appear in the daily briefing.
- **Health / habits log** — with `ENABLE_HABITS=true`, log habits and health
  metrics ("slept 7 hours", "ran 5k", "gym") with their numbers; `habit_summary`
  reports streaks and recent trends. It's the "I did it, here's the value" side
  of a recurring task, so the assistant can reflect real progress back to you.
- **Subscriptions / bills** — with `ENABLE_SUBSCRIPTIONS=true`, track recurring
  charges (amount, cadence, renewal date); "what am I paying for?" lists them
  with an estimated monthly-spend rollup, and each renewal fires a heads-up a few
  days ahead (exactly-once via the fired ledger) so nothing surprises you.
- **Expense log** — with `ENABLE_EXPENSES=true`, log one-off spending as it
  happens ("250 kr on groceries") with `log_expense`; `expense_summary` rolls
  any month up per currency and category, `remove_expense` fixes a mis-log, and
  on the 1st the daily briefing opens with last month's rollup. The complement
  of subscriptions: that's what recurs, this is where the money actually went.
- **Work log / time tracking** — with `ENABLE_WORKLOG=true`, track working time
  per project: `start_work` starts the clock ("back to the report" — a running
  timer is stopped and logged first, so switching is one call), `stop_work`
  logs the stretch, `log_work` records time after the fact ("2 hours on client
  X yesterday"), and `work_summary` rolls today and the last N days up per
  project. While the clock runs a small Work-timer block rides in each turn,
  and the weekly review carries last week's hours per project. The work twin of
  the expense log: that's where the money went, this is where the time went.
- **Reading list (read-it-later)** — with `ENABLE_READING=true`, a place to
  save links to get back to: "save this for later" stores it, "what's on my
  reading list?" lists the unread ones, and the assistant can mark them read or
  drop them. Tool-only, so it never bloats the per-turn prompt.
- **Weather** — with `ENABLE_WEATHER=true` and a location (`WEATHER_LATITUDE`/
  `WEATHER_LONGITUDE`, or a `WEATHER_LOCATION_NAME` that gets geocoded), a short
  forecast is fetched off the reply path (Open-Meteo, keyless) and folded into
  each turn's context and the daily briefing — "bring a jacket" grounded in real
  conditions. The `get_weather` tool answers about any other city or a multi-day
  outlook on demand.

## Roadmap / not yet wired

- Two-way calendar sync (CalDAV writes) — the current sync is deliberately pull-only.

> **Note on the embedding model:** the default `EMBEDDING_MODEL` is
> `intfloat/multilingual-e5-large` (1024-dim, strong Norwegian recall). Its first use
> downloads ~2GB into the HuggingFace cache; set a smaller model (e.g.
> `sentence-transformers/all-MiniLM-L6-v2`) if you don't need multilingual recall.
