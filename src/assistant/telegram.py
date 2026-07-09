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
import html
import json
import logging
import os
import secrets
import urllib.error
import urllib.request
from collections.abc import Callable
from urllib.parse import urlparse

from langgraph.graph.state import CompiledStateGraph
from markdown_it import MarkdownIt
from markdown_it.token import Token

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
_SAFE_LINK_SCHEMES = {"http", "https", "mailto", "tg"}
_MARKDOWN = MarkdownIt("default", {"html": False, "linkify": False, "typographer": False})


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
    """Persist ``chat_id`` as paired so it survives restarts.

    Written atomically (temp file + ``os.replace``): the reminder ticker reads
    this file from another thread, and a partial read is swallowed as "no
    paired chats", which would silently drop a reminder fan-out.
    """
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


def _safe_href(href: str | None) -> str | None:
    if not href:
        return None
    scheme = urlparse(href).scheme.lower()
    if scheme not in _SAFE_LINK_SCHEMES:
        return None
    return html.escape(href, quote=True)


def _language_class(info: str) -> str:
    language = info.strip().split(maxsplit=1)[0] if info.strip() else ""
    safe = "".join(c for c in language if c.isalnum() or c in {"+", "-", "_", "#"})
    return f' class="language-{html.escape(safe, quote=True)}"' if safe else ""


def _render_inline(tokens: list[Token]) -> str:
    pieces: list[str] = []
    link_stack: list[str | None] = []
    for token in tokens:
        if token.type == "text":
            pieces.append(html.escape(token.content))
        elif token.type == "code_inline":
            pieces.append(f"<code>{html.escape(token.content)}</code>")
        elif token.type in {"softbreak", "hardbreak"}:
            pieces.append("\n")
        elif token.type == "strong_open":
            pieces.append("<b>")
        elif token.type == "strong_close":
            pieces.append("</b>")
        elif token.type == "em_open":
            pieces.append("<i>")
        elif token.type == "em_close":
            pieces.append("</i>")
        elif token.type == "s_open":
            pieces.append("<s>")
        elif token.type == "s_close":
            pieces.append("</s>")
        elif token.type == "link_open":
            href = _safe_href(token.attrGet("href"))
            link_stack.append(href)
            if href is not None:
                pieces.append(f'<a href="{href}">')
        elif token.type == "link_close":
            href = link_stack.pop() if link_stack else None
            if href is not None:
                pieces.append("</a>")
        elif token.children:
            pieces.append(_render_inline(token.children))
        elif token.content:
            pieces.append(html.escape(token.content))
    return "".join(pieces)


def _matching_close(tokens: list[Token], start: int, close_type: str) -> int:
    depth = 0
    for idx in range(start, len(tokens)):
        if tokens[idx].type == tokens[start].type:
            depth += 1
        elif tokens[idx].type == close_type:
            depth -= 1
            if depth == 0:
                return idx
    return len(tokens) - 1


def _render_blocks(tokens: list[Token]) -> str:
    pieces: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.type == "inline":
            pieces.append(_render_inline(token.children or []))
        elif token.type == "paragraph_open":
            inline = tokens[i + 1] if i + 1 < len(tokens) else None
            if inline and inline.type == "inline":
                pieces.append(f"{_render_inline(inline.children or [])}\n\n")
            i = _matching_close(tokens, i, "paragraph_close")
        elif token.type == "heading_open":
            inline = tokens[i + 1] if i + 1 < len(tokens) else None
            if inline and inline.type == "inline":
                pieces.append(f"<b>{_render_inline(inline.children or [])}</b>\n\n")
            i = _matching_close(tokens, i, "heading_close")
        elif token.type in {"fence", "code_block"}:
            content = html.escape(token.content.rstrip("\n"))
            language_class = _language_class(token.info) if token.type == "fence" else ""
            if language_class:
                pieces.append(f"<pre><code{language_class}>{content}</code></pre>\n\n")
            else:
                pieces.append(f"<pre>{content}</pre>\n\n")
        elif token.type == "blockquote_open":
            close = _matching_close(tokens, i, "blockquote_close")
            inner = _render_blocks(tokens[i + 1 : close]).strip()
            if inner:
                pieces.append(f"<blockquote>{inner}</blockquote>\n\n")
            i = close
        elif token.type in {"bullet_list_open", "ordered_list_open"}:
            close = _matching_close(
                tokens,
                i,
                "bullet_list_close" if token.type == "bullet_list_open" else "ordered_list_close",
            )
            number = int(token.attrGet("start") or 1)
            item = 0
            j = i + 1
            while j < close:
                if tokens[j].type == "list_item_open":
                    item_close = _matching_close(tokens, j, "list_item_close")
                    body = _render_blocks(tokens[j + 1 : item_close]).strip()
                    if body:
                        prefix = "• " if token.type == "bullet_list_open" else f"{number + item}. "
                        pieces.append(prefix + body.replace("\n", "\n  ") + "\n")
                    item += 1
                    j = item_close
                j += 1
            pieces.append("\n")
            i = close
        elif token.type == "html_block":
            pieces.append(f"{html.escape(token.content.strip())}\n\n")
        elif token.type == "hr":
            pieces.append("---\n\n")
        elif token.content:
            pieces.append(html.escape(token.content))
        i += 1
    return "".join(pieces)


def _telegram_html(markdown: str) -> str:
    """Render assistant Markdown into Telegram-supported HTML."""
    rendered = _render_blocks(_MARKDOWN.parse(markdown)).strip()
    return rendered or html.escape(markdown.strip()) or "(empty reply)"


def _split_piece(piece: str) -> list[str] | None:
    """Split a markdown piece roughly in half, preferring a block boundary.

    Tries a blank line first (keeps whole blocks together), then a newline,
    then a space. ``None`` when no split point yields two non-empty halves.
    """
    mid = len(piece) // 2
    for sep in ("\n\n", "\n", " "):
        cut = piece.rfind(sep, 1, mid)
        if cut < 1:
            cut = piece.find(sep, mid, len(piece) - 1)
        if cut > 0:
            head, tail = piece[:cut].strip("\n"), piece[cut:].strip("\n")
            if head and tail:
                return [head, tail]
    return None


def _render_chunks(text: str) -> list[tuple[str, str | None]]:
    """Markdown pieces paired with rendered HTML that fits the API limit.

    Rendering grows the text (entity escaping, ``<b>``/``<a>`` tags), so the
    limit can only be checked *after* rendering: chunk, render, and re-split any
    piece whose rendered form is still too long. A piece that cannot be split
    further comes back with ``None`` HTML — the caller sends it as plain text.
    """
    queue = _chunks(text)
    rendered: list[tuple[str, str | None]] = []
    while queue:
        piece = queue.pop(0)
        html_piece = _telegram_html(piece)
        if len(html_piece) <= _MAX_MESSAGE_CHARS:
            rendered.append((piece, html_piece))
            continue
        halves = _split_piece(piece)
        if halves is None:
            rendered.append((piece, None))
        else:
            queue[:0] = halves
    return rendered


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
    if token is None or chat_id is None or not text:
        return None  # not a text message (sticker, photo, member event, …)

    allowed = authorized_chats(settings)
    if chat_id not in allowed:
        if allowed:
            # Once anyone is paired/allowlisted, strangers get silence.
            logger.warning("ignoring telegram message from unauthorized chat %s", chat_id)
            return None
        # No owner yet: run the pairing handshake (code round-trip via the
        # server log) instead of trusting first contact blindly. The handshake
        # messages themselves never reach the model.
        _handle_pairing(settings, token, chat_id, text)
        return None

    # Show "typing…" while the model thinks (best-effort; it expires after ~5s).
    try:
        _call(token, "sendChatAction", {"chat_id": chat_id, "action": "typing"})
    except (urllib.error.URLError, OSError, RuntimeError):
        pass

    thread_id = f"telegram:{chat_id}"
    try:
        reply = run_chat(agent, text, thread_id, settings=settings)
    except CodexError as exc:
        logger.error("telegram chat turn failed: %s", exc)
        send_message(token, chat_id, "Sorry — I hit an error answering that. Try again.")
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
    offset: int | None = None
    # Strong refs so fire-and-forget upkeep tasks are never garbage-collected.
    upkeep_tasks: set[asyncio.Task] = set()
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
