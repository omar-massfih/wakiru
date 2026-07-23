"""Minimal RSS/Atom fetching for feed watches — stdlib only, netguard-checked.

A feed watch (:mod:`assistant.watches`, kind ``feed``) polls a URL the user
registered in conversation and fires when a new entry matches. This module is
the deterministic fetch half: pull the XML (public addresses only, size- and
timeout-capped), parse out entry titles and links namespace-agnostically, and
cache the result briefly so a burst of heartbeat wakes does not hammer the
origin. Titles are arbitrary-origin text — callers must frame them as content,
never instructions.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from .netguard import BlockedURLError, urlopen_public

_FETCH_TIMEOUT_SECONDS = 10.0
_MAX_BYTES = 1_000_000
_MAX_ENTRIES = 50
_TITLE_MAX_CHARS = 200

# One fetch per feed per TTL window, per process — heartbeat wakes can come
# minutes apart and feeds move slowly.
CACHE_TTL_SECONDS = 600.0
_cache: dict[str, tuple[float, list[FeedEntry]]] = {}


class FeedError(Exception):
    """The feed could not be fetched or parsed."""


@dataclass(frozen=True)
class FeedEntry:
    title: str
    link: str = ""


def _local(tag: object) -> str:
    """An element's tag without its XML namespace."""
    text = tag if isinstance(tag, str) else ""
    return text.rsplit("}", 1)[-1].lower()


def _entry_from(element: ET.Element) -> FeedEntry | None:
    title = ""
    link = ""
    for child in element:
        name = _local(child.tag)
        if name == "title" and not title:
            title = (child.text or "").strip()
        elif name == "link" and not link:
            # RSS puts the URL in the text; Atom in an href attribute.
            link = (child.text or "").strip() or child.get("href", "").strip()
    if not title:
        return None
    return FeedEntry(title=title[:_TITLE_MAX_CHARS], link=link)


def parse_entries(xml_text: str) -> list[FeedEntry]:
    """Feed entries in document order — RSS ``<item>`` and Atom ``<entry>``
    alike, namespaces ignored. Raises :class:`FeedError` on unparseable XML."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise FeedError(f"not parseable as RSS/Atom XML ({exc})") from exc
    entries: list[FeedEntry] = []
    for element in root.iter():
        if _local(element.tag) not in ("item", "entry"):
            continue
        entry = _entry_from(element)
        if entry is not None:
            entries.append(entry)
        if len(entries) >= _MAX_ENTRIES:
            break
    return entries


def fetch_entries(url: str, force: bool = False) -> list[FeedEntry]:
    """The feed's current entries, through the short per-process cache.

    ``force`` skips the cache (used at registration, so a stale error or a
    just-fixed feed is re-checked). Raises :class:`FeedError` on any fetch or
    parse problem — blocked (non-public) addresses included.
    """
    cached = _cache.get(url)
    if cached and not force and time.monotonic() - cached[0] < CACHE_TTL_SECONDS:
        return cached[1]
    try:
        with urlopen_public(url, timeout=_FETCH_TIMEOUT_SECONDS) as response:
            raw = response.read(_MAX_BYTES + 1)
    except BlockedURLError as exc:
        raise FeedError(str(exc)) from exc
    except Exception as exc:
        raise FeedError(f"fetch failed ({exc})") from exc
    if len(raw) > _MAX_BYTES:
        raise FeedError(f"feed larger than {_MAX_BYTES:,} bytes")
    entries = parse_entries(raw.decode("utf-8", errors="replace"))
    _cache[url] = (time.monotonic(), entries)
    return entries
