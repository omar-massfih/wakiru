"""Telegram channel — talk to the assistant from your phone.

A long-polling bridge to the Telegram Bot API, stdlib-only (urllib) like
:mod:`assistant.notify` — no runtime HTTP dependency. Long polling means the
server *pulls* updates, so it works behind NAT with no public webhook URL and
no open inbound port. Enable it by setting ``TELEGRAM_BOT_TOKEN`` (from
@BotFather); the API lifespan then runs :func:`poll_loop` alongside the
reminder ticker.

Security: pairing-code handshake. While the bot has no owner, a chat that
messages it receives a prompt to echo back a short code — which is printed only
to the *server log*, so only whoever runs the server can complete the pairing.
The paired chat is persisted under the memory directory and answered from then
on; every other chat gets silence. Pin or add chats explicitly via
``TELEGRAM_ALLOWED_CHAT_IDS`` (it is merged with the paired set, and bypasses
the handshake); un-pair by deleting ``telegram_chats.json`` from the memory
directory. Each chat maps to a stable thread (``telegram:<chat_id>``), so the
conversation — with its working memory and rolling summary — survives restarts.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import secrets
import tempfile
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from langgraph.graph.state import CompiledStateGraph

from .chat import error_reply, run_chat, run_upkeep
from .config import Settings, postgres_backend
from .telegram_render import _render_chunks

logger = logging.getLogger(__name__)

_API_BASE = "https://api.telegram.org"
# How long one getUpdates call blocks server-side waiting for a message.
_POLL_SECONDS = 30
# Socket-timeout head-room on top of the long poll.
_TIMEOUT_MARGIN_SECONDS = 15
# Back-off after a failed poll so an outage doesn't spin the loop.
_RETRY_SECONDS = 5
# How often the typing bubble is re-sent while a turn runs (Telegram expires
# each sendChatAction after ~5s).
_TYPING_REFRESH_SECONDS = 4.0


def _call(token: str, method: str, payload: dict, timeout: float = 15) -> object:
    """POST one Bot API method and return its ``result`` (raises on failure)."""
    request = urllib.request.Request(
        f"{_API_BASE}/bot{token}/{method}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    if not body.get("ok"):
        raise RuntimeError(f"telegram {method} failed: {body.get('description')}")
    return body.get("result")


def _paired_path(settings: Settings):
    return settings.memory_path / "telegram_chats.json"


def _paired_chats(settings: Settings) -> list[int]:
    """Chats paired at runtime (trust-on-first-use), persisted across restarts."""
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.paired_telegram_chats(settings)
    try:
        return [int(c) for c in json.loads(_paired_path(settings).read_text())]
    except FileNotFoundError:
        return []
    except (ValueError, OSError):
        logger.warning("unreadable %s; treating as no paired chats", _paired_path(settings))
        return []


def _pair(settings: Settings, chat_id: int) -> None:
    """Persist ``chat_id`` as paired so it survives restarts.

    Written atomically (temp file + ``os.replace``): the reminder ticker reads
    this file from another thread, and a partial read is swallowed as "no
    paired chats", which would silently drop a reminder fan-out.
    """
    if storage_postgres := postgres_backend(settings):
        storage_postgres.pair_telegram_chat(settings, chat_id)
        return
    settings.memory_path.mkdir(parents=True, exist_ok=True)
    chats = _paired_chats(settings)
    if chat_id not in chats:
        chats.append(chat_id)
        path = _paired_path(settings)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(chats))
        os.replace(tmp, path)


def authorized_chats(settings: Settings) -> list[int]:
    """Every chat the assistant answers: the env allowlist plus paired chats."""
    chats = list(settings.telegram_allowed_chat_ids)
    chats.extend(c for c in _paired_chats(settings) if c not in chats)
    return chats


# Chats mid-handshake: chat_id -> the code they must echo back. In-memory only;
# a restart simply restarts the handshake. Only consulted while the bot has no
# owner, and handle_update runs sequentially in the poll loop, so no lock.
_pending_pairings: dict[int, str] = {}


def _handle_pairing(settings: Settings, token: str, chat_id: int, text: str) -> None:
    """One step of the pairing handshake for an ownerless bot.

    First contact gets a short code — printed only to the server log, so only
    whoever runs the server can read it — and the chat is paired when it echoes
    the code back. This closes the trust-on-first-use window where whoever
    happened to find the bot first silently became its owner.
    """
    code = _pending_pairings.get(chat_id)
    if code is not None and text.strip() == code:
        _pending_pairings.pop(chat_id, None)
        _pair(settings, chat_id)
        logger.info("paired telegram chat %s (pairing code verified)", chat_id)
        send_message(token, chat_id, "Paired — this chat now talks to your assistant.")
        return
    if code is None:
        code = secrets.token_hex(3)
        _pending_pairings[chat_id] = code
    logger.warning("telegram pairing code for chat %s: %s", chat_id, code)
    send_message(
        token,
        chat_id,
        "This assistant isn't paired yet. Reply with the pairing code "
        "printed in its server log to pair this chat.",
    )


@contextlib.contextmanager
def _typing(token: str, chat_id: int):
    """Keep the chat's "typing…" bubble alive while the body runs.

    Telegram expires each ``sendChatAction`` after ~5 seconds — far shorter
    than a model turn — so a daemon thread re-sends it until the reply is
    ready. Every send is best-effort: presence must never break the turn.
    """
    stop = threading.Event()

    def _keepalive() -> None:
        while not stop.is_set():
            with contextlib.suppress(urllib.error.URLError, OSError, RuntimeError):
                _call(token, "sendChatAction", {"chat_id": chat_id, "action": "typing"})
            stop.wait(_TYPING_REFRESH_SECONDS)

    thread = threading.Thread(target=_keepalive, daemon=True, name="telegram-typing")
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1)


def send_message(token: str, chat_id: int, text: str) -> None:
    """Deliver ``text`` to a chat, split into API-sized chunks.

    Each chunk is sent as HTML, falling back to plain text when Telegram
    rejects the markup or the network hiccups. A failed chunk is logged and the
    rest still go out; only total failure (nothing delivered) raises, so
    callers can tell a dead channel from a partial delivery.
    """
    delivered = False
    last_error: Exception | None = None
    for piece, html_piece in _render_chunks(text):
        try:
            if html_piece is not None:
                try:
                    _call(
                        token,
                        "sendMessage",
                        {"chat_id": chat_id, "text": html_piece, "parse_mode": "HTML"},
                    )
                    delivered = True
                    continue
                except (urllib.error.URLError, OSError, RuntimeError) as exc:
                    logger.warning(
                        "telegram HTML delivery failed; retrying as plain text: %s", exc
                    )
            _call(token, "sendMessage", {"chat_id": chat_id, "text": piece})
            delivered = True
        except (urllib.error.URLError, OSError, RuntimeError) as exc:
            logger.warning("telegram delivery of one chunk failed: %s", exc)
            last_error = exc
    if not delivered and last_error is not None:
        raise last_error


def _mergeable_text(update: dict) -> tuple[int, str] | None:
    """``(chat_id, text)`` when the update can join a coalesced run.

    Only plain-text, non-command messages qualify; voice notes, media, and
    slash commands are handled individually and break a run.
    """
    message = update.get("message") or {}
    chat_id = (message.get("chat") or {}).get("id")
    text = message.get("text")
    if chat_id is None or not isinstance(text, str) or not text or text.startswith("/"):
        return None
    return chat_id, text


def _coalesce(updates: list[dict]) -> list[dict]:
    """Merge runs of consecutive plain-text messages from the same chat.

    Messages sent while a turn was running are queued by Telegram and all
    arrive in the next poll batch; answered one by one they read like a bot
    replying to fragments of a thought. Merged texts are joined with newlines
    into a synthetic update that keeps the *last* message's ``update_id``, so
    offset semantics are unchanged.
    """
    merged: list[dict] = []
    for update in updates:
        current = _mergeable_text(update)
        previous = _mergeable_text(merged[-1]) if merged else None
        if current and previous and current[0] == previous[0]:
            combined = dict(update)
            combined["message"] = dict(update["message"])
            combined["message"]["text"] = f"{previous[1]}\n{current[1]}"
            merged[-1] = combined
            continue
        merged.append(update)
    return merged


def get_updates(token: str, offset: int | None) -> list[dict]:
    """One long-poll round; returns whatever updates arrived (possibly none)."""
    payload: dict = {"timeout": _POLL_SECONDS, "allowed_updates": ["message"]}
    if offset is not None:
        payload["offset"] = offset
    result = _call(
        token, "getUpdates", payload, timeout=_POLL_SECONDS + _TIMEOUT_MARGIN_SECONDS
    )
    return result if isinstance(result, list) else []


# Slash commands the bot advertises (via setMyCommands). /reset and the admin
# commands (_ADMIN_COMMANDS below — the on-demand background jobs that used to
# be REST endpoints) are answered locally; the rest map to natural-language
# turns the model answers itself, from its own context and memory
# (_COMMAND_PROMPTS below).
_COMMANDS = [
    ("start", "Show what I can do"),
    ("help", "Show what I can do"),
    ("reset", "Forget this conversation's history"),
    ("memory", "Show what I remember about you"),
    ("tasks", "Show your open to-do list"),
    ("calendar", "Show upcoming events"),
    ("email", "Show unread mail (if email is enabled)"),
    ("heartbeat", "Run one heartbeat wake now"),
    ("briefing", "Send today's briefing now"),
    ("review", "Send this week's review now"),
    ("sync", "Sync external calendars now"),
    ("sleep", "Run the nightly memory pass now"),
]

_COMMAND_PROMPTS = {
    "start": "Introduce yourself: who are you and what can you do for me here?",
    "help": "Introduce yourself: who are you and what can you do for me here?",
    "tasks": "Show my open to-do list.",
    "calendar": "What's coming up on my calendar?",
    "email": "Any unread mail?",
    "memory": "What do you remember about me?",
}


def _admin_heartbeat(agent, settings: Settings) -> str:
    from .heartbeat import run_heartbeat

    result = run_heartbeat(settings, agent=agent, force=True)
    if result.get("sent"):
        return "Heartbeat ran — message on its way."
    return f"Heartbeat ran — nothing to send ({result.get('reason', 'silent')})."


def _admin_briefing(agent, settings: Settings) -> str:
    from .briefing import run_briefing

    result = run_briefing(settings, force=True, agent=agent)
    if result.get("sent"):
        return "Briefing on its way."
    return f"No briefing sent ({result.get('reason', 'unknown')})."


def _admin_review(agent, settings: Settings) -> str:
    from .weekly_review import run_weekly_review

    result = run_weekly_review(settings, force=True, agent=agent)
    if result.get("sent"):
        return "Weekly review on its way."
    return f"No review sent ({result.get('reason', 'unknown')})."


def _admin_sync(agent, settings: Settings) -> str:
    from .calendar import remote as calendar_remote
    from .calendar import sync as calendar_sync
    from .refreshes import caldav_once

    parts = []
    if settings.calendar_ics_urls:
        calendar_sync.pull_feeds(settings)
        parts.append("feeds pulled")
    if calendar_remote.is_configured(settings):
        caldav_once(settings)
        parts.append("CalDAV synced")
    return "Calendar sync done" + (f" ({', '.join(parts)})." if parts else " — nothing configured.")


def _admin_sleep(agent, settings: Settings) -> str:
    from .sleep import run_sleep

    run_sleep(settings, agent, force=True)
    return "Nightly memory pass ran."


# The on-demand background jobs — each idempotent via its own ledger, so a
# repeated command is safe. Anything a job decides to *send* (a briefing, a
# heartbeat push) arrives through the normal proactive channels; the return
# value here is only the operator acknowledgement.
_ADMIN_COMMANDS: dict[str, Callable[..., str]] = {
    "heartbeat": _admin_heartbeat,
    "briefing": _admin_briefing,
    "review": _admin_review,
    "sync": _admin_sync,
    "sleep": _admin_sleep,
}


def set_commands(token: str) -> None:
    """Register the slash-command menu with Telegram (best-effort, once at startup)."""
    try:
        _call(
            token,
            "setMyCommands",
            {"commands": [{"command": c, "description": d} for c, d in _COMMANDS]},
        )
    except Exception:
        logger.warning("setMyCommands failed; the command menu may be stale", exc_info=True)


def _reset_thread(agent: CompiledStateGraph, thread_id: str) -> None:
    """Clear one thread's checkpointed conversation history and rolling summary."""
    from langchain_core.messages import RemoveMessage
    from langchain_core.runnables import RunnableConfig

    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    snapshot = agent.get_state(config)
    messages = snapshot.values.get("messages", [])
    removals = [RemoveMessage(id=m.id) for m in messages if m.id is not None]
    # as_node="agent" matches the graph's message-producing node, mirroring how
    # maybe_summarize applies its trims.
    agent.update_state(config, {"messages": removals, "summary": ""}, as_node="agent")


def _reset_reply(agent: CompiledStateGraph, thread_id: str) -> str:
    """Perform /reset and report it — deterministic on purpose: clearing a
    broken history must not depend on the model (or that history) working."""
    try:
        _reset_thread(agent, thread_id)
    except Exception:
        logger.exception("reset failed for thread %s", thread_id)
        return "Couldn't reset — try again."
    return "Done — I've forgotten this conversation's history."


def _command_turn(text: str) -> str:
    """The natural-language turn a non-reset ``/command`` message becomes.

    Known commands map to plain requests the model answers from its injected
    context (agenda, tasks, mail, memory) in its own voice; an unknown command
    runs as its text minus the slash, and a bare "/" asks for the intro.
    """
    # "/tasks@MyBot arg" -> "tasks". Split before indexing: a bare "/" or a
    # "/" followed only by spaces has no first word.
    parts = text[1:].split()
    command = parts[0].split("@")[0].lower() if parts else ""
    return _COMMAND_PROMPTS.get(command) or text[1:].strip() or _COMMAND_PROMPTS["help"]


def _transcribe_voice(token: str, settings: Settings, voice: dict) -> str:
    """Download one Telegram voice note and return its transcript.

    Raises on any failure (download, decode, model); the caller turns that into
    a friendly reply. Runs the same thread as the turn — transcription time is
    part of the reply latency, which is why clip length is bounded.
    """
    info = _call(token, "getFile", {"file_id": voice.get("file_id")})
    file_path = (info or {}).get("file_path") if isinstance(info, dict) else None
    if not file_path:
        raise ValueError("telegram getFile returned no file_path")
    url = f"{_API_BASE}/file/bot{token}/{file_path}"
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    from .stt import transcribe

    suffix = Path(file_path).suffix or ".oga"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        return transcribe(tmp.name, settings)


def _voice_turn_text(
    token: str, settings: Settings, chat_id: int, voice: dict
) -> str | None:
    """Turn an authorized chat's voice note into turn text, or ``None`` to drop it.

    Owns the whole voice policy — feature gate, duration cap, transcription, and
    every user-facing reply on the way out (a refusal, a failure apology, or the
    transcript echo that makes a mishearing obvious before the reply lands).
    """
    if not settings.enable_voice:
        send_message(token, chat_id, "Voice notes are off. Set ENABLE_VOICE=true to use them.")
        return None
    if (voice.get("duration") or 0) > settings.voice_max_seconds:
        send_message(
            token,
            chat_id,
            f"That voice note is too long — keep it under {settings.voice_max_seconds}s.",
        )
        return None
    try:
        with _typing(token, chat_id):
            text = _transcribe_voice(token, settings, voice)
    except Exception:
        logger.exception("voice transcription failed for chat %s", chat_id)
        send_message(token, chat_id, "Sorry — I couldn't make out that voice note. Try again?")
        return None
    if not text:
        send_message(token, chat_id, "I couldn't hear any speech in that voice note.")
        return None
    # Echo the transcript so a mishearing is obvious before the reply lands.
    send_message(token, chat_id, f"🎙 Heard: “{text}”")
    return text


def handle_update(
    agent: CompiledStateGraph, settings: Settings, update: dict
) -> Callable[[], None] | None:
    """Answer one incoming message: authorize, run the turn, reply.

    Returns the turn's post-reply upkeep as a zero-arg callable (or ``None`` when
    the update produced no turn). The poll loop runs it off the reply path: upkeep
    makes further Codex calls, and awaiting them here would block the *next*
    message for as long as they take.
    """
    token = settings.telegram_bot_token
    message = update.get("message") or {}
    chat_id = (message.get("chat") or {}).get("id")
    text = message.get("text")
    voice = message.get("voice") or {}
    if token is None or chat_id is None or not (text or voice):
        return None  # not a text/voice message (sticker, photo, member event, …)

    allowed = authorized_chats(settings)
    if chat_id not in allowed:
        if allowed or not text:
            # Once anyone is paired/allowlisted, strangers get silence — and the
            # pairing handshake itself is text-only (a stranger's audio is never
            # downloaded, let alone transcribed).
            logger.warning("ignoring telegram message from unauthorized chat %s", chat_id)
            return None
        # No owner yet: run the pairing handshake (code round-trip via the
        # server log) instead of trusting first contact blindly. The handshake
        # messages themselves never reach the model.
        _handle_pairing(settings, token, chat_id, text)
        return None

    if text is None:  # a voice note from an authorized chat
        text = _voice_turn_text(token, settings, chat_id, voice)
        if text is None:
            return None

    thread_id = f"telegram:{chat_id}"

    # Slash commands: /reset and the admin commands are answered locally
    # (/reset must work even when the model or the checkpointed history is
    # broken; the admin commands are deterministic job triggers). Every other
    # command becomes a natural-language turn the model answers itself.
    if text.startswith("/"):
        parts = text[1:].split()
        command = parts[0].split("@")[0].lower() if parts else ""
        if command == "reset":
            send_message(token, chat_id, _reset_reply(agent, thread_id))
            return None
        if (admin := _ADMIN_COMMANDS.get(command)) is not None:
            try:
                with _typing(token, chat_id):
                    reply = admin(agent, settings)
            except Exception:
                logger.exception("admin command /%s failed", command)
                reply = f"Couldn't run /{command} — check the server logs."
            send_message(token, chat_id, reply)
            return None
        text = _command_turn(text)

    try:
        with _typing(token, chat_id):
            reply = run_chat(agent, text, thread_id, settings=settings)
    except Exception as exc:
        # Whatever failed, the user must hear something better than silence.
        logger.exception("telegram chat turn failed")
        send_message(token, chat_id, error_reply(exc))
        return None
    send_message(token, chat_id, reply)
    return lambda: run_upkeep(agent, settings, text, reply, thread_id)


async def poll_loop(agent: CompiledStateGraph, settings: Settings) -> None:
    """Long-poll Telegram forever, answering messages one at a time.

    Replies are sequential by design: a turn can take as long as a full Codex
    run, during which Telegram queues further messages server-side (they are
    delivered on the next poll). Each turn's upkeep, however, runs as a
    background task — it makes further Codex calls, and awaiting it inline would
    make the *next* message wait on the *previous* turn's maintenance. Every
    failure is logged and retried so the channel survives network blips and API
    hiccups. Blocking work runs in worker threads to keep the event loop (and
    the reminder ticker) free.
    """
    token = settings.telegram_bot_token
    if not token:  # the lifespan only starts this loop when the token is set
        raise ValueError("poll_loop requires TELEGRAM_BOT_TOKEN")
    offset: int | None = None
    # Strong refs so fire-and-forget upkeep tasks are never garbage-collected.
    upkeep_tasks: set[asyncio.Task] = set()
    await asyncio.to_thread(set_commands, token)  # advertise the /command menu
    logger.info("telegram channel started (long polling)")
    while True:
        try:
            updates = await asyncio.to_thread(get_updates, token, offset)
        except Exception:
            logger.exception("telegram getUpdates failed; retrying in %ss", _RETRY_SECONDS)
            await asyncio.sleep(_RETRY_SECONDS)
            continue
        if updates:
            # Advance past the whole batch first (updates arrive in id order):
            # a poison update must not be redelivered forever.
            offset = updates[-1]["update_id"] + 1
        if settings.telegram_coalesce_messages:
            updates = _coalesce(updates)
        for update in updates:
            try:
                upkeep = await asyncio.to_thread(handle_update, agent, settings, update)
            except Exception:
                logger.exception(
                    "handling telegram update %s failed", update.get("update_id")
                )
                continue
            if upkeep is not None:
                task = asyncio.create_task(asyncio.to_thread(upkeep))
                upkeep_tasks.add(task)
                task.add_done_callback(upkeep_tasks.discard)
