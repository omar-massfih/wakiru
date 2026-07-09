"""Document recall — the chunks injected alongside memory recall each turn.

:func:`docs_context` embeds the latest user turn and returns the most relevant
document chunks as a plain-text block, which the graph's ``recall`` node appends
to the memory-recall context. This is why "what did I write about X" works: the
answer rides in on the same channel durable memories do.
"""

from __future__ import annotations

from ..config import Settings, get_settings
from . import store


def docs_context(settings: Settings | None = None, query: str = "") -> str:
    """A text block of the document chunks most relevant to ``query`` (empty when
    docs are disabled, nothing is ingested, or nothing clears the similarity floor)."""
    settings = settings or get_settings()
    if not settings.enable_docs or not query.strip():
        return ""
    chunks = store.search_chunks(settings, query)
    if not chunks:
        return ""
    lines = ["## Relevant excerpts from your documents"]
    for chunk in chunks:
        lines.append(f"\n### From “{chunk.doc_title}”\n{chunk.text}")
    return "\n".join(lines)
