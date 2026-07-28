"""One-shot migration from Postgres (Neon) back into the local SQLite + markdown stores.

The inverse of :mod:`assistant.import_local`. Reads every domain from Postgres by
calling the *same* store module with a postgres-configured ``Settings`` (which
dispatches to ``storage_postgres``), and writes each object into a local
``Settings`` (SQLite + markdown). Used to reclaim data stranded in a retired Neon
instance and fold it into the SQLite backend.

Merge, not overwrite: this runs against a memory dir that already holds live
data, so every domain skips rows whose id/name already exists locally — fresh
state always wins on a collision. Notes are written as markdown only (their
bodies are the source of truth); the vector index and knowledge graph are
rebuilt from those files by the daemon's startup reindex, so they are not
migrated directly.
"""

from __future__ import annotations

import logging
from datetime import datetime

from .calendar import store as calendar_store
from .config import Settings
from .memory import store as memory_store
from .people import store as people_store
from .tasks import store as task_store

logger = logging.getLogger(__name__)


def _dt(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def import_notes(pg: Settings, local: Settings) -> tuple[int, int]:
    """Write every Neon note whose name isn't already a local note (markdown)."""
    existing = {n.name for n in memory_store.list_notes(local)}
    added = skipped = 0
    for note in memory_store.list_notes(pg):
        if note.name in existing:
            skipped += 1
            continue
        memory_store.write_note(local, note)
        added += 1
    return added, skipped


def import_tasks(pg: Settings, local: Settings) -> tuple[int, int]:
    existing = {t.id for t in task_store.list_tasks(local, include_done=True)}
    added = skipped = 0
    for task in task_store.list_tasks(pg, include_done=True):
        if task.id in existing:
            skipped += 1
            continue
        task_store.restore_task(local, task)
        added += 1
    return added, skipped


def import_events(pg: Settings, local: Settings) -> tuple[int, int]:
    existing = {e.id for e in calendar_store.list_events(local)}
    added = skipped = 0
    for event in calendar_store.list_events(pg):
        if event.id in existing:
            skipped += 1
            continue
        calendar_store.restore_event(local, event)
        added += 1
    return added, skipped


def import_people(pg: Settings, local: Settings) -> tuple[int, int]:
    existing = {p.id for p in people_store.list_people(local)}
    added = skipped = 0
    for person in people_store.list_people(pg):
        if person.id in existing:
            skipped += 1
            continue
        people_store.restore_person(local, person)
        added += 1
    return added, skipped


def import_followups(pg: Settings, local: Settings) -> tuple[int, int]:
    """Best-effort: followups get fresh ids (add() mints them); dedupe on topic+due."""
    from . import followups

    existing = {(f.topic, f.due) for f in followups.list_open(local)}
    added = skipped = 0
    for f in followups.list_open(pg):
        due = _dt(f.due)
        if due is None or (f.topic, f.due) in existing:
            skipped += 1
            continue
        followups.add(local, due, f.topic, f.context, f.thread_id)
        added += 1
    return added, skipped


def import_goals(pg: Settings, local: Settings) -> tuple[int, int]:
    from . import goals

    existing = {g.title for g in goals.list_open(local)}
    added = skipped = 0
    for g in goals.list_open(pg):
        if g.title in existing:
            skipped += 1
            continue
        goals.open_goal(local, g.title, g.state, _dt(g.next_action_at), g.thread_id)
        added += 1
    return added, skipped


def import_watches(pg: Settings, local: Settings) -> tuple[int, int]:
    from . import watches

    existing = {(w.kind, w.pattern) for w in watches.list_active(local)}
    added = skipped = 0
    for w in watches.list_active(pg):
        if (w.kind, w.pattern) in existing:
            skipped += 1
            continue
        watches.add(
            local,
            w.kind,
            w.pattern,
            note=w.note,
            until=_dt(w.expires_at),
            repeat=w.repeat,
            lead_minutes=w.lead_minutes,
            url=w.url,
        )
        added += 1
    return added, skipped


# domain -> importer. Notes/tasks/events/people are exact restores; the rest are
# best-effort (fresh ids, param-based inserts) and are wrapped so one bad domain
# never aborts the migration.
_IMPORTERS = {
    "notes": import_notes,
    "tasks": import_tasks,
    "events": import_events,
    "people": import_people,
    "followups": import_followups,
    "goals": import_goals,
    "watches": import_watches,
}


def import_all(pg: Settings, local: Settings) -> dict[str, tuple[int, int]]:
    """Run every importer; returns ``{domain: (added, skipped)}``."""
    local.memory_path.mkdir(parents=True, exist_ok=True)
    summary: dict[str, tuple[int, int]] = {}
    for name, fn in _IMPORTERS.items():
        try:
            summary[name] = fn(pg, local)
        except Exception:
            logger.exception("reverse import: domain %r failed", name)
            summary[name] = (-1, -1)
    return summary
