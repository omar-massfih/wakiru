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
import html
import json
import logging
import urllib.error
import urllib.request
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


def send_message(token: str, chat_id: int, text: str) -> None:
    """Deliver ``text`` to a chat, split into API-sized chunks."""
    for piece in _chunks(text):
        payload = {"chat_id": chat_id, "text": _telegram_html(piece), "parse_mode": "HTML"}
        try:
            _call(token, "sendMessage", payload)
        except (urllib.error.HTTPError, RuntimeError) as exc:
            logger.warning("telegram HTML delivery failed; retrying as plain text: %s", exc)
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
        reply = run_chat(agent, text, thread_id, settings=settings)
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
