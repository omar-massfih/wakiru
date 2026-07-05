"""Semantic recall: turn a query into the most relevant notes and a context block.

This is how the assistant "searches its memory": embed the incoming message,
take the top-k nearest notes from the vector index, keep those above the
similarity floor, and hand them back to the graph to inject before Codex runs.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import SystemMessage

from ..config import Settings, get_settings
from . import index, store
from .embeddings import embed_one
from .store import Note

# Base persona + operating instructions, prepended to every turn. The key job of
# this prompt is to tell the model that its memory is maintained *for it* out of
# band (recall injection here, background save/forget after the turn), so it stops
# disclaiming abilities it actually has.
SYSTEM_PROMPT = """\
You are a personal assistant with persistent, long-term memory that carries
across conversations.

How your memory works:
- Memories relevant to the current message are provided to you below under
  "Relevant memories", and everything you know is listed by title under "Memory
  index". Rely on these; never invent memories you were not given.
- Your memory is maintained for you automatically. After each turn the system
  records new durable facts and applies any request to remember or forget
  something — you do not (and cannot) call a tool to do this yourself.
- Therefore, when the user asks you to remember or forget something, just
  acknowledge it naturally (e.g. "Got it — I'll remember that." / "Okay, I've
  forgotten that."). Never say you are unable to remember, store, update, or
  delete information, and never tell the user to manage memory in some settings
  screen. The system handles it.
- Honor the preferences recorded in memory (for example, the user's preferred
  reply language)."""


def search_memory(
    settings: Settings, query: str, k: int | None = None
) -> list[tuple[Note, float]]:
    """Relevant notes for ``query`` as ``(note, similarity)``, best first."""
    if not query.strip():
        return []
    k = k or settings.recall_top_k
    hits = index.search(settings, embed_one(query, settings), k)
    results: list[tuple[Note, float]] = []
    for name, path, _desc, similarity in hits:
        if similarity < settings.recall_min_similarity:
            continue
        note = _load(settings, name, path)
        if note is not None:
            results.append((note, similarity))
    return results


def _load(settings: Settings, name: str, path: str) -> Note | None:
    p = Path(path)
    if p.exists():
        try:
            return store.read_note(p)
        except (ValueError, KeyError):
            return None
    return store.find_note(settings, name)  # file moved/renamed — fall back to name


def build_context_message(
    settings: Settings, results: list[tuple[Note, float]]
) -> SystemMessage:
    """Compose the system message injected ahead of the user's turn.

    Always includes the base :data:`SYSTEM_PROMPT` and the compact ``MEMORY.md``
    index (so the model knows what it knows), plus the full text of any relevant
    recalled notes.
    """
    parts = [
        SYSTEM_PROMPT,
        "\n## Memory index\n" + store.read_index(settings),
    ]
    if results:
        recalled = "\n\n".join(
            f"### {note.name} ({note.type})\n{note.body}" for note, _ in results
        )
        parts.append("\n## Relevant memories for this request\n" + recalled)
    return SystemMessage(content="\n".join(parts))


def recall_context(settings: Settings | None, query: str) -> SystemMessage:
    """Convenience: search and build the context message in one call."""
    settings = settings or get_settings()
    return build_context_message(settings, search_memory(settings, query))
