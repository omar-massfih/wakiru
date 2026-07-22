"""Memory formation — the "learning" side of the brain.

Learning happens in two phases:

**Online (per turn, in the background).** After each exchange we:

1. write a compact **episodic** trace of the turn (cheap, no LLM), and
2. run a **reconciling extractor**: Codex reads the exchange *together with the
   memories already relevant to it* and returns operations —

   * ``save``   — a new durable fact/preference/learning,
   * ``update`` — supersede an existing note in place (fixes contradictions like
     "moved from Oslo to Bergen" that dedup-by-similarity would miss),
   * ``forget`` — drop something by name or description.

Because the extractor sees current memory, it stops piling up near-duplicate or
contradictory notes. Every ``save`` still dedups by embedding similarity as a
backstop.

**Consolidation ("sleep").** Periodically :mod:`.consolidate` reviews recent
episodes against long-term memory to promote, merge, decay, and prune. That lives
in its own module; this one handles the per-turn path.
"""

from __future__ import annotations

import logging
from datetime import date

from ..config import Settings, get_settings
from ..llm import complete_text
from ..ops_parse import parse_ops
from . import graph, index, store
from .embeddings import embed_one, embed_query
from .locks import locked
from .store import Note, slugify

logger = logging.getLogger(__name__)


def _today() -> str:
    return date.today().isoformat()


def _write_and_index(settings: Settings, note: Note, vector: list[float]) -> Note:
    """Persist a note (file + vector index + MEMORY.md) with all its metadata."""
    store.purge_stale_files(settings, note.name, note.kind)  # drop any other-kind copy
    path = store.write_note(settings, note)
    index.upsert(
        settings,
        note.name,
        str(path),
        note.description,
        vector,
        kind=note.kind,
        salience=note.salience,
        updated=note.updated,
        last_recalled=note.last_recalled,
        recall_count=note.recall_count,
        text_hash=index.content_hash(note.index_text),
    )
    # Mirror the note's triples into the knowledge graph (rebuildable from the
    # file just written). Unconditional sync — an absorb/revise that dropped a
    # relation must clear the stale edge, so this runs even when relations is [].
    if settings.enable_graph_memory:
        graph.sync_note(settings, note)
    store.regenerate_index(settings)
    return note


@locked
def save_memory(
    settings: Settings,
    body: str,
    description: str | None = None,
    kind: str | None = None,
    salience: float | None = None,
    confidence: float | None = None,
    source: str = "",
    tags: list[str] | None = None,
    relations: list[dict] | None = None,
) -> Note:
    """Persist a memory, deduping against existing notes.

    A near-duplicate (embedding cosine ≥ ``settings.dedup_threshold``) of the same
    kind updates the existing note in place — keeping its name, creation date, and
    reinforcement counters — instead of creating a second copy. A durable save may
    also absorb a durable note of the *other* kind at the stricter
    ``dedup_cross_kind_threshold``; episodic traces never merge across kinds.
    """
    body = body.strip()
    desc = (description or body).strip()
    note = Note(
        name=slugify(desc),
        description=desc,
        body=body,
        kind=kind or "semantic",
        salience=0.5 if salience is None else float(salience),
        confidence=0.8 if confidence is None else float(confidence),
        tags=[str(t) for t in tags] if tags else [],
        source=source,
        updated=_today(),
        relations=relations or [],
    )

    vector = embed_one(note.index_text, settings)

    def _absorb(existing: Note) -> Note:
        """Take over an existing note's identity (name/created/tags/counters)."""
        note.name = existing.name
        note.created = existing.created
        # Union, not replace: a restatement must not strip e.g. the profile tag.
        note.tags += [t for t in existing.tags if t not in note.tags]
        stats = index.get_stats(settings, existing.name)
        note.recall_count, note.last_recalled = stats or (
            existing.recall_count,
            existing.last_recalled,
        )
        return _write_and_index(settings, note, vector)

    hits = index.search_ranked(settings, vector, k=settings.dedup_candidates)

    # Dedup pass 1 — same kind only: an episodic trace must never swallow a
    # distilled semantic/procedural fact just because they read alike.
    for name, _path, _desc, hit_kind, _sal, _rc, _lr, sim in hits:
        if sim < settings.dedup_threshold:
            break  # results are sorted best-first; nothing else will qualify
        if hit_kind != note.kind:
            continue
        existing = store.find_note(settings, name)
        if existing is not None:
            return _absorb(existing)

    # Dedup pass 2 — across the durable kinds (semantic <-> procedural) at a
    # stricter threshold, so a rephrased fact that arrives labelled as the other
    # kind updates the original instead of duplicating it. Episodic never merges.
    if note.kind != "episodic":
        for name, _path, _desc, hit_kind, _sal, _rc, _lr, sim in hits:
            if sim < settings.dedup_cross_kind_threshold:
                break
            if hit_kind == note.kind or hit_kind == "episodic":
                continue
            existing = store.find_note(settings, name)
            if existing is not None:
                return _absorb(existing)

    note.name = store.unique_name(settings, note.name)
    return _write_and_index(settings, note, vector)


@locked
def revise_memory(
    settings: Settings,
    name: str,
    body: str | None = None,
    description: str | None = None,
    kind: str | None = None,
    salience: float | None = None,
    tags: list[str] | None = None,
    relations: list[dict] | None = None,
) -> Note | None:
    """Supersede an existing note in place, keeping its identity and counters.

    Returns the revised note, or ``None`` if no note named ``name`` exists (the
    caller may then fall back to :func:`save_memory`).
    """
    existing = store.find_note(settings, name)
    if existing is None:
        return None

    # Direct attribute assignment skips Note.__post_init__, so gate the kind here:
    # an unknown value from an LLM op keeps the note's current kind.
    kind = store.normalize_kind(kind)

    old_path = store.note_path(settings, existing)
    if body is not None:
        existing.body = body.strip()
    if description is not None:
        existing.description = description.strip()
    if kind is not None:
        existing.kind = kind
    if salience is not None:
        existing.salience = float(salience)
    if tags:  # additive only — an update never strips e.g. the profile tag
        existing.tags += [str(t) for t in tags if str(t) not in existing.tags]
    if relations is not None:
        # Replace, not append: an update restates the note's facts, so its
        # triples supersede the old ones (a moved-away edge must not linger).
        existing.relations = store.normalize_relations(relations)
    existing.updated = _today()

    new_path = store.note_path(settings, existing)
    if old_path != new_path and settings.storage_backend != "postgres":
        old_path.unlink(missing_ok=True)

    # Preserve reinforcement counters from the authoritative index (the file's
    # copy lags until consolidation flushes it) so an update never wipes them.
    stats = index.get_stats(settings, name)
    if stats is not None:
        existing.recall_count, existing.last_recalled = stats

    vector = embed_one(existing.index_text, settings)
    return _write_and_index(settings, existing, vector)


@locked
def forget_memory(
    settings: Settings, query: str, *, allow_fuzzy: bool = True
) -> Note | None:
    """Delete the memory best matching ``query`` (by name, else by similarity).

    ``allow_fuzzy=False`` restricts deletion to an exact name match. Callers
    applying an LLM ``forget`` op that carries a *name* must pass it: a
    hallucinated name falling through to the similarity match could delete an
    unrelated real memory.
    """
    by_name = store.find_note(settings, query)
    if by_name is not None:
        # Episodes are a log pruned by consolidation, never forgotten by name —
        # an LLM-hallucinated name must not be able to punch a hole in it.
        if by_name.kind == "episodic":
            logger.warning("refusing to forget episodic trace %r", query)
            return None
        deleted = store.delete_note(settings, query)
        index.remove(settings, query)
        if settings.enable_graph_memory:
            graph.remove(settings, query)
        store.regenerate_index(settings)
        return deleted

    if not allow_fuzzy:
        logger.warning("forget target %r does not exist; skipping (exact-only)", query)
        return None

    # Fuzzy fallback — never fuzzy-delete an episodic trace (those are a log,
    # pruned by consolidation, not by a loose text match).
    candidates = [
        (name, sim)
        for name, _path, _desc, kind, _sal, _rc, _lr, sim in index.search_ranked(
            settings, embed_query(query, settings), k=5
        )
        if kind != "episodic" and sim >= settings.forget_threshold
    ]
    if not candidates:
        return None
    # Ambiguity guard: when two notes match about equally well, deleting nothing
    # beats deleting the wrong memory — the caller can retry by exact name.
    if (
        len(candidates) > 1
        and candidates[0][1] - candidates[1][1] < settings.forget_ambiguity_margin
    ):
        logger.warning(
            "fuzzy forget %r is ambiguous between %r (%.3f) and %r (%.3f); skipping",
            query, candidates[0][0], candidates[0][1],
            candidates[1][0], candidates[1][1],
        )
        return None
    name = candidates[0][0]
    deleted = store.delete_note(settings, name)
    index.remove(settings, name)
    if settings.enable_graph_memory:
        graph.remove(settings, name)
    store.regenerate_index(settings)
    return deleted


@locked
def record_episode(
    settings: Settings, user_msg: str, assistant_msg: str, source: str = ""
) -> Note | None:
    """Write a compact episodic trace of one turn (no LLM, low salience).

    Gated so the log stays signal, not noise: returns ``None`` (no trace) when
    the user message is too short to matter (greetings, "ok") or when the trace
    would near-duplicate an existing episode.
    """
    user_msg = user_msg.strip()
    assistant_msg = assistant_msg.strip()
    min_chars = settings.episodic_min_chars
    if min_chars > 0 and len(user_msg) < min_chars:
        return None
    desc = f"{_today()}: " + (user_msg[:80] + ("…" if len(user_msg) > 80 else ""))
    body = f"User: {user_msg[:600]}\nAssistant: {assistant_msg[:600]}".strip()
    note = Note(
        name=store.unique_name(settings, slugify(f"{_today()} {user_msg}")),
        description=desc,
        body=body,
        kind="episodic",
        salience=settings.episodic_initial_salience,
        confidence=1.0,
        source=source,
        updated=_today(),
    )
    vector = embed_one(note.index_text, settings)
    for _name, _path, _desc, hit_kind, _sal, _rc, _lr, sim in index.search_ranked(
        settings, vector, k=3
    ):
        if sim < settings.episodic_dedup_threshold:
            break
        if hit_kind == "episodic":
            return None  # a repeat of an exchange the log already covers
    return _write_and_index(settings, note, vector)


# --------------------------------------------------------------------------- #
# LLM-driven extraction (background, reconciling)
# --------------------------------------------------------------------------- #

_EXTRACT_PROMPT = """\
You maintain the long-term memory of a personal assistant. Read the exchange and
the memories already known to be relevant, then decide what should change.

Honor explicit instructions: if the user asks you to remember something, save it;
if they ask you to forget something, forget it. Also proactively capture DURABLE,
user-specific facts, preferences, goals, or learnings worth recalling later.
Ignore greetings, small talk, and anything ephemeral. Split compound statements
into separate atomic memories. Phrase each memory as a clear third-person
sentence (e.g. "The user's name is …").

CRITICAL — reconcile against what is already known:
- If a known memory is now WRONG, OUTDATED, or REFINED by this exchange, emit an
  "update" for it (use its exact name) instead of saving a near-duplicate.
- Only "save" genuinely new information not already covered below.
- If the exchange shows a memory was already saved, updated, or forgotten via a
  tool this turn (e.g. the assistant reports it remembered/forgot something),
  do not repeat that operation.

Choose a kind for each saved/updated memory:
- "semantic"   — durable facts, preferences, goals about the user or world.
- "procedural" — how-to knowledge, methods, the way the user likes things done.

Additionally, when a memory describes how the user LIVES or WORKS — working
hours, home/work locations, quiet hours ("don't ping me after 22:00"), commute,
or preferred tone/format of replies — add "tags": ["profile"] to the operation.
Treat communication preferences as first-class profile facts: preferred
language, formality, humor tolerance, brevity ("keep answers short"), how they
like to be greeted, and when not to be disturbed. These profile memories
personalize scheduling, reminders, and tone every turn.

On a "save" or "update", also extract the entity RELATIONSHIPS the memory states
as (subject, relation, object) triples, when any are present. These build a graph
that answers multi-hop questions ("where does my sister work?"). Use short slug
relations (works_at, lives_in, sister, brother, parent, spouse, friend, owns,
member_of); name the user as "user". Only include a relationship the memory
actually asserts — omit "relations" entirely when there are none. If the exchange
makes clear when a relationship started or ended, add ISO-date "valid_from" /
"valid_to". Example: "My sister Sara works at Acme" ->
  "relations": [{{"subj": "user", "rel": "sister", "obj": "Sara"}},
                {{"subj": "Sara", "rel": "works_at", "obj": "Acme"}}]

Return a JSON array of operations, each one of:
  {{"op": "save", "kind": "semantic|procedural", "description": "<short summary>", "body": "<one clear sentence>", "salience": <0..1>, "tags": ["profile"]?, "relations": [{{"subj": "..", "rel": "..", "obj": "..", "valid_from": "..?", "valid_to": "..?"}}]?}}
  {{"op": "update", "name": "<existing memory name>", "description": "<short summary>", "body": "<the corrected sentence>", "tags": ["profile"]?, "relations": [..]?}}
  {{"op": "forget", "name": "<existing memory name>"}}   (or {{"op": "forget", "query": "<what to drop>"}})
For "forget", always use the exact name when the memory appears in the known
list; only use "query" for memories not shown.
Return [] if nothing should change. Output JSON only — no prose, no code fences.

Known relevant memories:
{memories}

User: {user}
Assistant: {assistant}
"""


def _format_memories(settings: Settings, query: str) -> str:
    """Durable memories relevant to ``query``, for the extractor to reconcile.

    Episodic traces are deliberately excluded: they are a raw log managed by
    consolidation, not something the per-turn extractor should edit or forget.
    """
    from .recall import search_memory

    # Search a wider pool before dropping episodics: the current turn's own
    # trace (written just before this) near-duplicates the query and would
    # otherwise crowd every durable note out of a plain top-k.
    pool = settings.recall_top_k * settings.recall_candidate_multiplier
    durable = [
        (note, score)
        for note, score in search_memory(settings, query, k=pool)
        if note.kind != "episodic"
    ][: settings.recall_top_k]
    if not durable:
        return "(none)"
    return "\n".join(
        f"- name: {note.name} [{note.kind}] — {note.description}\n  {note.body}"
        for note, _ in durable
    )


_ALLOWED_OPS = frozenset({"save", "update", "forget"})


def _parse_ops(text: str) -> list[dict]:
    return parse_ops(text, _ALLOWED_OPS)


def update_memory(
    settings: Settings | None, user_msg: str, assistant_msg: str, source: str = ""
) -> list[str]:
    """Extract and apply memory operations for one turn (episode + save/update/forget).

    Intended to run in the background — it makes a second Codex call. Returns a
    short log of what changed. No-ops when ``enable_auto_memory`` is false.
    """
    settings = settings or get_settings()
    if not settings.enable_auto_memory:
        return []

    applied: list[str] = []

    # 1. Episodic trace — cheap, gated (skips small talk and repeats).
    try:
        if record_episode(settings, user_msg, assistant_msg, source=source) is None:
            logger.debug("episodic trace skipped (too short or near-duplicate)")
    except Exception:
        logger.exception("failed to record episodic trace")

    # 2. Reconciling extraction.
    prompt = _EXTRACT_PROMPT.format(
        memories=_format_memories(settings, user_msg),
        user=user_msg,
        assistant=assistant_msg,
    )
    try:
        raw = complete_text(prompt, settings)
    except Exception:
        logger.exception("memory extraction (LLM) failed; skipping this turn")
        return applied  # memory upkeep is best-effort; never break the main flow

    for op in _parse_ops(raw):
        try:
            if op["op"] == "save" and op.get("body"):
                tags = op.get("tags")
                rels = op.get("relations")
                note = save_memory(
                    settings,
                    body=str(op["body"]),
                    description=op.get("description"),
                    kind=op.get("kind"),
                    salience=op.get("salience"),
                    source=source,
                    tags=tags if isinstance(tags, list) else None,
                    relations=rels if isinstance(rels, list) else None,
                )
                applied.append(f"saved: {note.description}")
            elif op["op"] == "update" and op.get("name"):
                tags = op.get("tags")
                tags = tags if isinstance(tags, list) else None
                rels = op.get("relations")
                rels = rels if isinstance(rels, list) else None
                revised = revise_memory(
                    settings,
                    name=str(op["name"]),
                    body=op.get("body"),
                    description=op.get("description"),
                    kind=op.get("kind"),
                    tags=tags,
                    relations=rels,
                )
                if revised is None:  # name didn't exist — treat as a save
                    if op.get("body"):
                        note = save_memory(
                            settings,
                            body=str(op["body"]),
                            description=op.get("description"),
                            kind=op.get("kind"),
                            source=source,
                            tags=tags,
                            relations=rels,
                        )
                        applied.append(f"saved: {note.description}")
                else:
                    applied.append(f"updated: {revised.description}")
            elif op["op"] == "forget":
                # A "name" must match exactly — a hallucinated name falling
                # through to the fuzzy match could delete the wrong memory.
                # Only an explicit "query" opts into similarity matching.
                name, fuzzy = op.get("name"), op.get("query")
                deleted = None
                if name:
                    deleted = forget_memory(settings, str(name), allow_fuzzy=False)
                elif fuzzy:
                    deleted = forget_memory(settings, str(fuzzy))
                if deleted is not None:
                    applied.append(f"forgot: {deleted.description}")
        except Exception:
            logger.exception("failed to apply memory op: %s", op)

    if applied:
        logger.info("memory updated: %s", "; ".join(applied))
    return applied
