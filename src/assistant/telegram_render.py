"""Render assistant Markdown into Telegram-supported HTML, chunked to the API limit.

Telegram accepts a small HTML subset (``<b>``/``<i>``/``<a>``/``<pre>``/…), not
Markdown, and caps a message at 4096 characters. This module parses the model's
Markdown with markdown-it and emits that HTML subset, escaping everything else,
then splits a reply into pieces whose *rendered* form fits the limit. It is the
pure formatting half of :mod:`assistant.telegram` — no network, no transport.
"""

from __future__ import annotations

import html
from urllib.parse import urlparse

from markdown_it import MarkdownIt
from markdown_it.token import Token

# Telegram rejects messages longer than this; longer replies are split.
_MAX_MESSAGE_CHARS = 4096
_SAFE_LINK_SCHEMES = {"http", "https", "mailto", "tg"}
_MARKDOWN = MarkdownIt("default", {"html": False, "linkify": False, "typographer": False})


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


def _safe_href(href: str | int | float | None) -> str | None:
    # markdown-it's attrGet is typed str|int|float|None; an href is always a
    # string in practice, but coerce so the type checker and urlparse agree.
    if not href:
        return None
    href = str(href)
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
