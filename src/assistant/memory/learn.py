"""Memory formation — the "learning" side of the brain.

A single LLM-driven path handles everything. After each turn, Codex reads the
exchange and returns a list of memory *operations*:

* ``save``   — a durable fact/preference/learning worth keeping (whether the user
  said "remember …" or just mentioned it in passing).
* ``forget`` — something the user asked to drop.

This runs in the background (it makes a second Codex call), so it never adds
latency to the reply. Every save goes through :func:`save_memory`, which dedupes
by embedding similarity — a near-duplicate updates the existing note instead of
piling up a second copy.
"""

from __future__ import annotations

import json
import re

from ..codex_runner import run_codex
from ..config import Settings, get_settings
from . import index, store
from .embeddings import embed_one
from .store import Note, slugify

# Cosine-similarity floor above which a new memory is treated as a duplicate of
# an existing one (update in place rather than create).
DEDUP_THRESHOLD = 0.85
# Lower bar for a "forget" query: it's fuzzy, so accept a looser match.
FORGET_THRESHOLD = 0.45


def save_memory(
    settings: Settings,
    body: str,
    description: str | None = None,
    type: str | None = None,
) -> Note:
    """Persist a memory (file + index), deduping against existing notes."""
    body = body.strip()
    note = Note(
        name=slugify(description or body),
        description=(description or body).strip(),
        body=body,
        type=type or "fact",
    )

    # Dedup: if a very similar note already exists, update it in place.
    vector = embed_one(note.index_text, settings)
    hits = index.search(settings, vector, k=1)
    if hits and hits[0][3] >= DEDUP_THRESHOLD:
        existing = store.find_note(settings, hits[0][0])
        if existing is not None:
            note.name = existing.name
            note.created = existing.created

    path = store.write_note(settings, note)
    index.upsert(settings, note.name, str(path), note.description, vector)
    store.regenerate_index(settings)
    return note


def forget_memory(settings: Settings, query: str) -> Note | None:
    """Delete the memory best matching ``query`` (file + index)."""
    hits = index.search(settings, embed_one(query, settings), k=1)
    if not hits or hits[0][3] < FORGET_THRESHOLD:
        return None
    name = hits[0][0]
    deleted = store.delete_note(settings, name)
    index.remove(settings, name)
    store.regenerate_index(settings)
    return deleted


# --------------------------------------------------------------------------- #
# LLM-driven extraction (background)
# --------------------------------------------------------------------------- #

_EXTRACT_PROMPT = """\
You maintain the long-term memory of a personal assistant. Read the exchange
below and decide what should change in long-term memory.

Honor explicit instructions: if the user asks you to remember something, save it;
if they ask you to forget something, mark it for deletion. Also proactively
capture DURABLE, user-specific facts, preferences, goals, or learnings worth
recalling in future conversations. Ignore greetings, small talk, and anything
ephemeral. Split compound statements into separate atomic memories, and phrase
each saved memory as a clear third-person sentence (e.g. "The user's name is …").

Return a JSON array of operations, each one of:
  {{"op": "save", "type": "fact" | "learning", "description": "<short one-line summary>", "body": "<the memory as one clear sentence>"}}
  {{"op": "forget", "query": "<what the user wants forgotten>"}}
Return [] if nothing should change. Output JSON only — no prose, no code fences.

User: {user}
Assistant: {assistant}
"""


def _parse_ops(text: str) -> list[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict) and d.get("op") in {"save", "forget"}]


def update_memory(
    settings: Settings | None, user_msg: str, assistant_msg: str
) -> list[str]:
    """Extract and apply memory operations for one turn (save + forget).

    Intended to run in the background — it makes a second Codex call. Returns a
    short log of what changed. No-ops when ``enable_auto_memory`` is false.
    """
    settings = settings or get_settings()
    if not settings.enable_auto_memory:
        return []
    prompt = _EXTRACT_PROMPT.format(user=user_msg, assistant=assistant_msg)
    try:
        raw = run_codex(prompt, settings=settings)
    except Exception:
        return []  # memory upkeep is best-effort; never break the main flow

    applied: list[str] = []
    for op in _parse_ops(raw):
        if op["op"] == "save" and op.get("body"):
            note = save_memory(
                settings,
                body=str(op["body"]),
                description=op.get("description"),
                type=op.get("type"),
            )
            applied.append(f"saved: {note.description}")
        elif op["op"] == "forget" and op.get("query"):
            deleted = forget_memory(settings, str(op["query"]))
            if deleted is not None:
                applied.append(f"forgot: {deleted.description}")
    return applied
