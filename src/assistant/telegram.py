"""Telegram channel — talk to the assistant from your phone.

A long-polling bridge to the Telegram Bot API, stdlib-only (urllib) like
:mod:`assistant.notify` — no runtime HTTP dependency. Long polling means the
server *pulls* updates, so it works behind NAT with no public webhook URL and
no open inbound port. Enable it by setting ``TELEGRAM_BOT_TOKEN`` (from
@BotFather); the API lifespan then runs :func:`poll_loop` alongside the
reminder ticker.

Security: trust-on-first-use. The first chat to message the bot is *paired* —
persisted under the memory directory — and answered from then on; every other
chat gets silence. So setup is just: set the token, message the bot. Pin or add
chats explicitly via ``TELEGRAM_ALLOWED_CHAT_IDS`` (it is merged with the paired
set); un-pair by deleting ``telegram_chats.json`` from the memory directory.
Each chat maps to a stable thread (``telegram:<chat_id>``), so the conversation
— with its working memory and rolling summary — survives restarts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.request

from langgraph.graph.state import CompiledStateGraph

from .chat import run_chat, run_upkeep
from .codex_runner import CodexError
from .config import Settings

logger = logging.getLogger(__name__)

_API_BASE = "https://api.telegram.org"
# Telegram rejects messages longer than this; longer replies are split.
_MAX_MESSAGE_CHARS = 4096
# How long one getUpdates call blocks server-side waiting for a message.
_POLL_SECONDS = 30
# Socket-timeout head-room on top of the long poll.
_TIMEOUT_MARGIN_SECONDS = 15
# Back-off after a failed poll so an outage doesn't spin the loop.
_RETRY_SECONDS = 5


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
    try:
        return [int(c) for c in json.loads(_paired_path(settings).read_text())]
    except FileNotFoundError:
        return []
    except (ValueError, OSError):
        logger.warning("unreadable %s; treating as no paired chats", _paired_path(settings))
        return []


def _pair(settings: Settings, chat_id: int) -> None:
    """Persist ``chat_id`` as paired so it survives restarts."""
    settings.memory_path.mkdir(parents=True, exist_ok=True)
    chats = _paired_chats(settings)
    if chat_id not in chats:
        chats.append(chat_id)
        _paired_path(settings).write_text(json.dumps(chats))


def authorized_chats(settings: Settings) -> list[int]:
    """Every chat the assistant answers: the env allowlist plus paired chats."""
    chats = list(settings.telegram_allowed_chat_ids)
    chats.extend(c for c in _paired_chats(settings) if c not in chats)
    return chats


def _chunks(text: str) -> list[str]:
    """Split a reply into Telegram-sized pieces, preferring newline boundaries."""
    text = text.strip()
    if not text:
        return ["(empty reply)"]
    pieces: list[str] = []
    while text:
        if len(text) <= _MAX_MESSAGE_CHARS:
            pieces.append(text)
            break
        cut = text.rfind("\n", 1, _MAX_MESSAGE_CHARS)
        if cut < 1:
            cut = _MAX_MESSAGE_CHARS
        pieces.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return pieces


def send_message(token: str, chat_id: int, text: str) -> None:
    """Deliver ``text`` to a chat, split into API-sized chunks."""
    for piece in _chunks(text):
        _call(token, "sendMessage", {"chat_id": chat_id, "text": piece})


def get_updates(token: str, offset: int | None) -> list[dict]:
    """One long-poll round; returns whatever updates arrived (possibly none)."""
    payload: dict = {"timeout": _POLL_SECONDS, "allowed_updates": ["message"]}
    if offset is not None:
        payload["offset"] = offset
    result = _call(
        token, "getUpdates", payload, timeout=_POLL_SECONDS + _TIMEOUT_MARGIN_SECONDS
    )
    return result if isinstance(result, list) else []


def handle_update(
    agent: CompiledStateGraph, settings: Settings, update: dict
) -> None:
    """Answer one incoming message: authorize, run the turn, reply, upkeep."""
    token = settings.telegram_bot_token
    message = update.get("message") or {}
    chat_id = (message.get("chat") or {}).get("id")
    text = message.get("text")
    if token is None or chat_id is None or not text:
        return  # not a text message (sticker, photo, member event, …)

    allowed = authorized_chats(settings)
    if chat_id not in allowed:
        if allowed:
            # Once anyone is paired/allowlisted, strangers get silence.
            logger.warning("ignoring telegram message from unauthorized chat %s", chat_id)
            return
        # Trust-on-first-use: the very first chat to reach the bot becomes its
        # owner, so setup needs no id copying, no .env edit, no restart.
        _pair(settings, chat_id)
        logger.info("paired telegram chat %s (first contact)", chat_id)
        send_message(token, chat_id, "Paired — this chat now talks to your assistant.")

    # Show "typing…" while the model thinks (best-effort; it expires after ~5s).
    try:
        _call(token, "sendChatAction", {"chat_id": chat_id, "action": "typing"})
    except (urllib.error.URLError, OSError, RuntimeError):
        pass

    thread_id = f"telegram:{chat_id}"
    try:
        reply = run_chat(agent, text, thread_id)
    except CodexError as exc:
        logger.error("telegram chat turn failed: %s", exc)
        send_message(token, chat_id, "Sorry — I hit an error answering that. Try again.")
        return
    send_message(token, chat_id, reply)
    # The reply is already out, so upkeep here costs the user nothing.
    run_upkeep(agent, settings, text, reply, thread_id)


async def poll_loop(agent: CompiledStateGraph, settings: Settings) -> None:
    """Long-poll Telegram forever, answering messages one at a time.

    Sequential by design: a turn can take as long as a full Codex run, during
    which Telegram queues further messages server-side (they are delivered on
    the next poll). Every failure is logged and retried so the channel survives
    network blips and API hiccups. Blocking work runs in worker threads to keep
    the event loop (and the reminder ticker) free.
    """
    token = settings.telegram_bot_token
    offset: int | None = None
    logger.info("telegram channel started (long polling)")
    while True:
        try:
            updates = await asyncio.to_thread(get_updates, token, offset)
        except Exception:
            logger.exception("telegram getUpdates failed; retrying in %ss", _RETRY_SECONDS)
            await asyncio.sleep(_RETRY_SECONDS)
            continue
        for update in updates:
            # Advance first: a poison update must not be redelivered forever.
            offset = update["update_id"] + 1
            try:
                await asyncio.to_thread(handle_update, agent, settings, update)
            except Exception:
                logger.exception(
                    "handling telegram update %s failed", update.get("update_id")
                )
