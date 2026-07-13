"""Consolidation — the brain's "sleep" pass over long-term memory.

Runs periodically (every N turns) or on demand, off the reply path. It does the
slow, whole-store work that the per-turn learner can't:

* **Decay + prune** episodic traces — salience falls with age; traces below the
  floor or past the age horizon are dropped, so the episodic log stays a rolling
  window rather than growing without bound.
* **Flush reinforcement** — mirror the index's authoritative ``recall_count`` /
  ``last_recalled`` counters back into the markdown frontmatter, so a rebuild from
  files (or a hand-edit) never loses what the brain learned about usefulness.
* **Promote / merge / reconcile (LLM)** — Codex reviews recent episodes against
  existing semantic/procedural memory and emits ``save`` / ``update`` / ``forget``
  ops to lift recurring, important patterns into durable memory, merge duplicates,
  and resolve contradictions store-wide.
* **Decay + cap durable memory** — the effective salience of durable notes that
  never get recalled fades gently, and each kind is held under a hard note cap
  (lowest retention value evicted first), so long-term memory — and with it the
  injected prompt context — cannot grow without bound.
"""

from __future__ import annotations

import logging
import math
from datetime import date

from ..config import Settings, get_settings
from ..llm import complete_text
from . import index, learn, recall, store
from .locks import CONSOLIDATE_LOCK, MEMORY_LOCK, locked

logger = logging.getLogger(__name__)


def _age_days(settings: Settings, stamp: str) -> int:
    try:
        return max((recall.local_today(settings) - date.fromisoformat(stamp)).days, 0)
    except ValueError:
        return 0


@locked
def _decay_and_prune(settings: Settings) -> int:
    """Age-decay episodic salience; delete traces past the horizon or floor."""
    pruned = 0
    half_life = max(settings.episodic_max_age_days / 2, 1)
    for note in store.list_notes(settings):
        if note.kind != "episodic":
            continue
        age = _age_days(settings, note.updated or note.created)
        decayed = settings.episodic_initial_salience * math.exp(
            -math.log(2) * age / half_life
        )
        if age > settings.episodic_max_age_days or decayed < settings.salience_prune_floor:
            store.delete_note(settings, note.name)
            index.remove(settings, note.name)
            pruned += 1
            continue
        if abs(decayed - note.salience) > 1e-3:
            note.salience = round(decayed, 3)
            store.write_note(settings, note)
            index.set_salience(settings, note.name, note.salience)
    return pruned


@locked
def _decay_durable(settings: Settings) -> int:
    """Gently decay the *effective* salience of durable notes never recalled.

    Index-only: the file keeps the note's original salience — the fixed base the
    decay is computed from, so repeated passes never compound — while the index
    carries the decayed value that recall ranking and cap eviction actually use.
    Never deletes: a faded note just loses priority until something recalls it
    (any recall history exempts a note entirely — proving useful is what protects
    a memory from fading).
    """
    half_life = settings.durable_decay_half_life_days
    if half_life <= 0:
        return 0
    decayed = 0
    for note in store.list_notes(settings):
        if note.kind == "episodic":
            continue
        stats = index.get_stats(settings, note.name)
        if stats is not None and stats[0] > 0:
            continue
        age = _age_days(settings, note.updated or note.created)
        target = note.salience * math.exp(-math.log(2) * age / half_life)
        target = round(max(target, settings.salience_prune_floor), 3)
        if target < note.salience - 1e-3:
            index.set_salience(settings, note.name, target)
            decayed += 1
    return decayed


@locked
def _enforce_kind_caps(settings: Settings) -> int:
    """Hard safety net: evict the lowest-value notes beyond each kind's cap.

    Ranks by :func:`recall.retention_score` over the index's effective salience
    and reinforcement counters, so recalled/recent/salient notes survive. The LLM
    passes shrink memory *judiciously*; this bound guarantees it stays finite.
    """
    by_kind: dict[str, list[tuple[float, str]]] = {}
    for name, _desc, kind, salience, rc, lr, updated in index.list_entries(settings):
        score = recall.retention_score(settings, kind, salience, rc, lr, updated)
        by_kind.setdefault(kind, []).append((score, name))

    evicted = 0
    for kind, ranked in by_kind.items():
        cap = settings.max_notes_per_kind.get(kind, 0)
        if cap <= 0 or len(ranked) <= cap:
            continue
        ranked.sort(reverse=True)
        for _score, name in ranked[cap:]:
            store.delete_note(settings, name)
            index.remove(settings, name)
            evicted += 1
    return evicted


@locked
def _flush_counters(settings: Settings) -> int:
    """Mirror the index's reinforcement counters into the note frontmatter."""
    flushed = 0
    for note in store.list_notes(settings):
        stats = index.get_stats(settings, note.name)
        if stats is None:
            continue
        recall_count, last_recalled = stats
        if recall_count != note.recall_count or last_recalled != note.last_recalled:
            note.recall_count = recall_count
            note.last_recalled = last_recalled
            store.write_note(settings, note)
            flushed += 1
    return flushed


_CONSOLIDATE_PROMPT = """\
You are consolidating the long-term memory of a personal assistant (its "sleep").
Below are recent episodic traces (things that happened) and the existing durable
memories. Decide what should be lifted into durable memory and what should be
cleaned up.

Do:
- PROMOTE recurring or important patterns from the episodes into durable
  "semantic" (facts/preferences/goals) or "procedural" (how-to) memories.
- MERGE duplicates and RESOLVE contradictions among existing durable memories
  (update the surviving one, forget the rest).
- Do NOT restate trivia or one-off chit-chat. Be conservative; quality over
  quantity. Do not touch episodic traces (they are managed automatically).

Return a JSON array of operations, each one of:
  {{"op": "save", "kind": "semantic|procedural", "description": "<short>", "body": "<one clear sentence>", "salience": <0..1>}}
  {{"op": "update", "name": "<existing name>", "description": "<short>", "body": "<corrected sentence>"}}
  {{"op": "forget", "name": "<existing name>"}}
Return [] if nothing should change. Output JSON only — no prose, no code fences.

Recent episodes:
{episodes}

Existing durable memories:
{durable}
"""


def _llm_consolidate(settings: Settings) -> list[str]:
    notes = store.list_notes(settings)
    episodes = [n for n in notes if n.kind == "episodic"]
    durable = [n for n in notes if n.kind != "episodic"]
    if not episodes:
        return []

    episodes = sorted(episodes, key=lambda n: n.updated, reverse=True)
    episodes = episodes[: settings.consolidate_max_episodes]

    episodes_txt = "\n".join(f"- {n.description}\n  {n.body}" for n in episodes)
    durable_txt = (
        "\n".join(f"- name: {n.name} [{n.kind}] — {n.description}\n  {n.body}" for n in durable)
        or "(none)"
    )
    prompt = _CONSOLIDATE_PROMPT.format(episodes=episodes_txt, durable=durable_txt)

    try:
        raw = complete_text(prompt, settings)
    except Exception:
        logger.exception("consolidation (LLM) failed")
        return []

    applied: list[str] = []
    for op in learn._parse_ops(raw):
        try:
            if op["op"] == "save" and op.get("body"):
                note = learn.save_memory(
                    settings,
                    body=str(op["body"]),
                    description=op.get("description"),
                    kind=op.get("kind"),
                    salience=op.get("salience"),
                    source="consolidation",
                )
                applied.append(f"promoted: {note.description}")
            elif op["op"] == "update" and op.get("name"):
                revised = learn.revise_memory(
                    settings,
                    name=str(op["name"]),
                    body=op.get("body"),
                    description=op.get("description"),
                    kind=op.get("kind"),
                )
                if revised is not None:
                    applied.append(f"merged: {revised.description}")
            elif op["op"] == "forget":
                # Exact-name only (matching the prompt's op schema): a
                # hallucinated name must not fuzzy-delete an unrelated note.
                name, fuzzy = op.get("name"), op.get("query")
                deleted = None
                if name:
                    deleted = learn.forget_memory(settings, str(name), allow_fuzzy=False)
                elif fuzzy:
                    deleted = learn.forget_memory(settings, str(fuzzy))
                if deleted is not None:
                    applied.append(f"forgot: {deleted.description}")
        except Exception:
            logger.exception("failed to apply consolidation op: %s", op)
    return applied


def consolidate_memory(settings: Settings | None = None) -> dict:
    """Run one consolidation pass. Returns a summary of what changed.

    Passes are mutually exclusive: a turn-triggered pass overlapping a manual
    ``POST /memory/consolidate`` would double-evict and burn Codex tokens on the
    same episodes, so the later one is skipped rather than queued (the work is
    periodic; the next trigger covers it). Each mutating step takes the memory
    lock itself — the pass never holds it across the Codex call.
    """
    settings = settings or get_settings()
    if not CONSOLIDATE_LOCK.acquire(blocking=False):
        logger.info("consolidation already running; skipping this pass")
        return {"skipped": "a consolidation pass is already running"}
    try:
        pruned = _decay_and_prune(settings)
        changes = _llm_consolidate(settings) if settings.enable_auto_memory else []
        flushed = _flush_counters(settings)
        durable_decayed = _decay_durable(settings)  # before caps: eviction sees decay
        evicted = _enforce_kind_caps(settings)
        with MEMORY_LOCK:
            trash_pruned = store.prune_trash(settings, settings.trash_retention_days)
            store.regenerate_index(settings)
        summary = {
            "pruned_episodes": pruned,
            "changes": changes,
            "counters_flushed": flushed,
            "durable_decayed": durable_decayed,
            "evicted": evicted,
            "trash_pruned": trash_pruned,
        }
        logger.info("consolidation: %s", summary)
        return summary
    finally:
        CONSOLIDATE_LOCK.release()
