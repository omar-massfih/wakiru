"""Semantic recall: turn a query into the most relevant notes and a context block.

This is how the assistant "searches its memory". Rather than a flat top-k cosine
lookup, recall pulls a wider candidate pool from the vector index and **re-ranks**
it by blending four signals, so the brain surfaces what is not just similar but
*useful*:

* **similarity** — cosine to the query (the base signal),
* **recency** — exponential decay since the note was last recalled/updated,
* **reuse** — how often the note has been recalled (reinforcement),
* **salience** — the note's stored importance,

plus a small per-kind bias (semantic/procedural facts slightly favored over raw
episodic traces for answering). Recalling a note *reinforces* it: the injected
notes get their reuse counters bumped, so memories that keep proving useful rise.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

from langchain_core.messages import SystemMessage

from ..config import Settings, get_settings
from . import index, store
from .embeddings import embed_query
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
  "Relevant memories", and a selection of what you know is listed by title under
  "Memory index" (it may be partial; relevant memories are always retrieved for
  you). Rely on these; never invent memories you were not given.
- Your memory has three kinds: semantic (durable facts and preferences),
  procedural (learned how-to knowledge), and episodic (things that happened).
- Your memory is maintained for you automatically. After each turn the system
  records new durable facts, reconciles anything that changed, and applies any
  request to remember or forget something — you do not (and cannot) call a tool
  to do this yourself.
- Therefore, when the user asks you to remember or forget something, just
  acknowledge it naturally (e.g. "Got it — I'll remember that." / "Okay, I've
  forgotten that."). Never say you are unable to remember, store, update, or
  delete information, and never tell the user to manage memory in some settings
  screen. The system handles it.
- Honor the preferences recorded in memory (for example, the user's preferred
  reply language).

Time and calendar:
- You know the current date and time: they are provided each turn under "Current
  date and time". Use them to answer time questions and to interpret relative
  dates like "tomorrow" or "next Friday". Never claim you don't know the time.
- You have a personal calendar, maintained for you automatically. The events
  coming up are listed under "Upcoming events". Rely on that list; do not invent
  events you were not shown.
- When the user asks you to schedule, move, or cancel something, just acknowledge
  it naturally (e.g. "Done — dentist booked for Friday at 3pm."). The system
  records the calendar change out of band after the turn; you do not (and cannot)
  call a tool to do it yourself. Never say you are unable to manage the calendar
  or tell the user to use some other app."""


def _recency(stamp: str, half_life_days: float) -> float:
    """Exponential recency score in ``[0, 1]`` from an ISO date (blank -> 0)."""
    if not stamp:
        return 0.0
    try:
        days = (date.today() - date.fromisoformat(stamp)).days
    except ValueError:
        return 0.0
    days = max(days, 0)
    return math.exp(-math.log(2) * days / max(half_life_days, 1e-6))


def retention_score(
    settings: Settings,
    kind: str,
    salience: float,
    recall_count: int,
    last_recalled: str,
    updated: str,
) -> float:
    """How valuable a note is independent of any query.

    The non-similarity part of the recall blend (recency + reuse + salience +
    kind bias). Also ranks the bounded index view and drives eviction when
    consolidation enforces the per-kind note caps.
    """
    recency = max(_recency(last_recalled, settings.recall_recency_half_life_days),
                  _recency(updated, settings.recall_recency_half_life_days))
    reuse = math.log1p(max(recall_count, 0)) / math.log(2 + settings.recall_reuse_cap)
    reuse = min(reuse, 1.0)
    return (
        settings.recall_w_recency * recency
        + settings.recall_w_reuse * reuse
        + settings.recall_w_salience * salience
        + settings.recall_kind_bias.get(kind, 0.0)
    )


def _blended_score(
    settings: Settings,
    similarity: float,
    kind: str,
    salience: float,
    recall_count: int,
    last_recalled: str,
    updated: str,
) -> float:
    return settings.recall_w_similarity * similarity + retention_score(
        settings, kind, salience, recall_count, last_recalled, updated
    )


def search_memory(
    settings: Settings, query: str, k: int | None = None
) -> list[tuple[Note, float]]:
    """Relevant notes for ``query`` as ``(note, blended_score)``, best first.

    Pure read — does *not* reinforce. Use :func:`recall_context` for the per-turn
    answering path (which reinforces the notes it injects).
    """
    if not query.strip():
        return []
    k = k or settings.recall_top_k
    pool = max(k * settings.recall_candidate_multiplier, k)
    hits = index.search_ranked(settings, embed_query(query, settings), pool)

    scored: list[tuple[Note, float]] = []
    for name, path, _desc, kind, salience, recall_count, last_recalled, sim in hits:
        if sim < settings.recall_min_similarity:
            continue
        note = _load(settings, name, path)
        if note is None:
            continue
        score = _blended_score(
            settings, sim, kind, salience, recall_count, last_recalled, note.updated
        )
        scored.append((note, score))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:k]


def _load(settings: Settings, name: str, path: str) -> Note | None:
    p = Path(path)
    if p.exists():
        try:
            return store.read_note(p)
        except (ValueError, KeyError):
            return None
    return store.find_note(settings, name)  # file moved/renamed — fall back to name


def build_index_view(settings: Settings) -> str:
    """A bounded, per-kind view of the memory index for prompt injection.

    ``MEMORY.md`` on disk stays complete (the human-readable artifact); this
    view trims each kind to its ``context_index_max_per_kind`` most valuable
    entries by :func:`retention_score`, so the injected index cannot grow
    without bound as notes accumulate. Built entirely from the index DB — no
    file reads.
    """
    by_kind: dict[str, list[tuple[float, str, str]]] = {}
    for name, desc, kind, salience, rc, lr, updated in index.list_entries(settings):
        score = retention_score(settings, kind, salience, rc, lr, updated)
        by_kind.setdefault(kind, []).append((score, name, desc))
    if not by_kind:
        return "_(empty)_"

    order = [k for k in store._KIND_ORDER if k in by_kind]
    order += [k for k in sorted(by_kind) if k not in store._KIND_ORDER]

    counts = ", ".join(f"{len(by_kind[k])} {k}" for k in order)
    caps = settings.context_index_max_per_kind
    trimmed = False
    lines = [f"Memory: {counts}."]
    for kind in order:
        cap = caps.get(kind, -1)
        if cap == 0:
            trimmed = True
            continue
        group = sorted(by_kind[kind], reverse=True)
        if cap > 0 and len(group) > cap:
            group = group[:cap]
            trimmed = True
        lines.append(f"\n### {kind.capitalize()}")
        lines.extend(f"- **{name}** — {desc}" for _score, name, desc in group)
    if trimmed:
        lines[0] += " Showing the most valuable entries."
    return "\n".join(lines)


def build_context_message(
    settings: Settings, results: list[tuple[Note, float]]
) -> SystemMessage:
    """Compose the system message injected ahead of the user's turn.

    Always includes the base :data:`SYSTEM_PROMPT` and a bounded view of the
    memory index (so the model knows what it knows), plus the full text of any
    relevant recalled notes, labelled by kind.
    """
    parts = [
        SYSTEM_PROMPT,
        "\n## Memory index\n" + build_index_view(settings),
    ]
    if settings.enable_calendar and settings.enable_write_confirmation:
        parts.append(
            "\n## Undo\nCalendar writes can be undone: after booking, moving, or "
            "cancelling something, you may mention the user can reply \"undo\" "
            f"within {settings.write_undo_window_minutes} minutes to revert it, "
            "if it fits naturally."
        )
    if results:
        recalled = "\n\n".join(
            f"### {note.name} ({note.kind})\n{note.body}" for note, _ in results
        )
        parts.append("\n## Relevant memories for this request\n" + recalled)
    return SystemMessage(content="\n".join(parts))


def recall_context(settings: Settings | None, query: str) -> SystemMessage:
    """Search, reinforce, and build the injected context message in one call.

    This is the per-turn answering path: the notes it selects get their reuse
    counters bumped (reinforcement), so useful memories strengthen over time.
    """
    settings = settings or get_settings()
    results = search_memory(settings, query)
    index.bump_recall(settings, [note.name for note, _ in results])
    return build_context_message(settings, results)
