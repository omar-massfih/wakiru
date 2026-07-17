"""Reflection — the nightly outcome review that closes the autonomy loop.

The assistant acts proactively (pushes, background mailbox triage, calendar
and task writes), and the user reacts: they reply or go quiet, undo writes,
mute reminders with a stated reason. Those signals were all recorded — the
push log here, the undo ledgers, the mutes store, the mail audit — but
nothing ever read them back. This module does: once per night (riding
:func:`assistant.sleep.run_sleep`) it assembles a *deterministic* digest of
only counted events, and — when the digest is non-empty — makes one
conservative LLM call that may write a few ``self``-tagged procedural notes
("your late-evening ambient pushes go unanswered; hold them for the morning").

Those notes then flow back into judgment: recall picks them up like any
procedural memory, and the heartbeat's compose prefix gives the freshest ones
a guaranteed slot. This is the honest trust ladder — the model *sees* how its
autonomy landed and adjusts, while the structural gates (quiet hours, send
exclusion, budgets) stay exactly where they are.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from . import followups, threads
from .calendar.context import now
from .calendar.store import parse_dt
from .config import Settings, get_settings, postgres_backend
from .ops_parse import parse_ops

logger = logging.getLogger(__name__)

# How long after a push a user reply still counts as a response to it.
_RESPONSE_WINDOW = timedelta(hours=2)
_DIGEST_WINDOW = timedelta(hours=24)
_EXCERPT_CHARS = 120

_REFLECT_PROMPT = """\
You are a personal assistant reviewing your own proactive behavior. Below is a
factual digest of what you did on your own initiative over the last day and how
the user measurably reacted, plus the lessons you have already recorded.

Decide what, if anything, you should learn about HOW you act proactively —
timing, frequency, tone, which background actions get reversed.

Rules:
- Conclude ONLY from what the digest states. Never infer motives, moods, or
  facts beyond it. A single data point is rarely worth a note.
- Lessons describe your own behavior, not the user's life (that is what the
  regular memory pass is for).
- Prefer updating an existing lesson over saving a near-duplicate.
- At most {cap} operations; an empty answer is the common, correct outcome.

Return a JSON array of operations, each one of:
  {{"op": "save", "description": "<short>", "body": "<one clear lesson>"}}
  {{"op": "update", "name": "<existing name>", "body": "<revised lesson>"}}
Return [] if nothing should change. Output JSON only — no prose, no code fences.

Lessons you already recorded:
{existing}

Digest of the last day:
{digest}
"""

_ALLOWED_OPS = frozenset({"save", "update"})
_SELF_TAG = "self"


# --------------------------------------------------------------------------- #
# Push log — the record of what was actually delivered, and when
# --------------------------------------------------------------------------- #

def _ensure_push_log(conn) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS push_log ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,"
        " kind TEXT NOT NULL, excerpt TEXT NOT NULL DEFAULT '')"
    )


def log_push(settings: Settings, kind: str, text: str) -> None:
    """Record one delivered push. Never raises — logging must not turn a
    delivered push into a failure."""
    try:
        ts = now(settings).isoformat(timespec="seconds")
        excerpt = " ".join(str(text).split())[:_EXCERPT_CHARS]
        if storage_postgres := postgres_backend(settings):
            storage_postgres.record_push_log(settings, ts, str(kind), excerpt)
            return
        with followups._connect(settings) as conn:
            _ensure_push_log(conn)
            conn.execute(
                "INSERT INTO push_log (ts, kind, excerpt) VALUES (?, ?, ?)",
                (ts, str(kind), excerpt),
            )
    except Exception:
        logger.exception("recording the push log entry failed")


def recent_pushes(settings: Settings, since: datetime) -> list[dict]:
    """Pushes delivered at/after ``since``, oldest first."""
    if storage_postgres := postgres_backend(settings):
        rows = storage_postgres.recent_push_log(settings)
    else:
        with followups._connect(settings) as conn:
            _ensure_push_log(conn)
            rows = [
                dict(row)
                for row in conn.execute(
                    "SELECT ts, kind, excerpt FROM push_log ORDER BY id"
                ).fetchall()
            ]
    return [
        row
        for row in rows
        if (ts := parse_dt(str(row["ts"]))) is not None and ts >= since
    ]


# --------------------------------------------------------------------------- #
# The deterministic digest — counted events only, no LLM
# --------------------------------------------------------------------------- #

def _push_lines(settings: Settings, current: datetime, since: datetime) -> list[str]:
    contacts = [
        ts
        for info in threads.known_threads(settings)
        if (ts := parse_dt(info.last_user_at)) is not None
    ]
    lines: list[str] = []
    for row in recent_pushes(settings, since):
        ts = parse_dt(str(row["ts"]))
        if ts is None:
            continue
        replied = any(ts <= c <= ts + _RESPONSE_WINDOW for c in contacts)
        if replied:
            reaction = "the user wrote back within 2h"
        elif any(c > ts for c in contacts):
            reaction = "the user was active later (reply timing unknown)"
        elif current - ts < _RESPONSE_WINDOW:
            reaction = "too recent to judge"
        else:
            reaction = "no user message since"
        lines.append(
            f"- {row['ts']} {row['kind']} push: \"{row['excerpt']}\" — {reaction}"
        )
    return lines


def _undone_write_lines(settings: Settings, since: datetime) -> list[str]:
    """Writes the user reverted — each one a proactive change they rejected."""
    from .calendar.undo import _SPEC as calendar_spec
    from .tasks.undo import _SPEC as tasks_spec
    from .write_ledger import connect

    lines: list[str] = []
    for spec in (calendar_spec, tasks_spec):
        try:
            if storage_postgres := postgres_backend(settings):
                rows: list[dict] = []
                for info in threads.known_threads(settings):
                    rows += [
                        dict(r)
                        for r in getattr(storage_postgres, spec.pg_rows)(
                            settings, info.thread_id
                        )
                    ]
            else:
                with connect(spec, settings) as conn:
                    rows = [
                        dict(r)
                        for r in conn.execute(
                            "SELECT summary, undone_at FROM write_log"
                            " WHERE undone_at IS NOT NULL"
                        ).fetchall()
                    ]
        except Exception:
            logger.exception("reading the %s undo ledger for reflection failed", spec.kind)
            continue
        for row in rows:
            undone = parse_dt(str(row.get("undone_at") or ""))
            if undone is not None and undone >= since:
                lines.append(
                    f"- The user undid a {spec.kind} write: {row.get('summary', '')}"
                )
    return lines


def _mute_lines(settings: Settings, since: datetime) -> list[str]:
    """Mutes the user created — each a push stream they asked to stop, with
    the stated reason when one was given (the clearest signal there is)."""
    from . import mutes

    if postgres_backend(settings):
        # The Postgres mutes mirror keeps only (scope, target, until) — no
        # reason/created_at — so this section has nothing reliable to say there.
        return []
    try:
        with mutes._connect(settings) as conn:
            rows = [
                dict(row)
                for row in conn.execute("SELECT * FROM reminder_mutes").fetchall()
            ]
    except Exception:
        logger.exception("reading the mutes store for reflection failed")
        return []
    lines = []
    for row in rows:
        created = parse_dt(str(row.get("created_at") or ""))
        if created is None or created < since:
            continue
        label = row.get("scope", "")
        reason = str(row.get("reason") or "").strip()
        lines.append(
            f"- Reminders muted (scope: {label})"
            + (f' — stated reason: "{reason}"' if reason else " — no reason given")
        )
    return lines


def _triage_lines(settings: Settings, since: datetime) -> list[str]:
    if not settings.enable_email:
        return []
    from .mail import audit as mail_audit

    try:
        actions = mail_audit.recent(settings, limit=20, actor="heartbeat")
    except Exception:
        logger.exception("reading the mail audit for reflection failed")
        return []
    lines = []
    for row in actions:
        at = parse_dt(str(row.get("at") or ""))
        if at is not None and at >= since:
            lines.append(f"- Background mailbox action: {row.get('detail', '')}")
    return lines


def build_digest(settings: Settings, current: datetime) -> str:
    """The last day's measurable autonomy outcomes; '' when there were none."""
    since = current - _DIGEST_WINDOW
    sections = (
        _push_lines(settings, current, since)
        + _undone_write_lines(settings, since)
        + _mute_lines(settings, since)
        + _triage_lines(settings, since)
    )
    return "\n".join(sections)


# --------------------------------------------------------------------------- #
# Self-notes — the lessons, written and read back
# --------------------------------------------------------------------------- #

def self_notes(settings: Settings, k: int = 5) -> list:
    """The freshest ``self``-tagged lessons, for the heartbeat's compose prefix."""
    from .memory import store

    notes = [
        note
        for note in store.list_notes(settings)
        if _SELF_TAG in note.tags and note.kind == "procedural"
    ]
    notes.sort(key=lambda n: n.updated or n.created, reverse=True)
    return notes[: max(k, 0)]


def run_reflection(settings: Settings | None = None, current: datetime | None = None) -> dict:
    """One nightly review: digest, then at most one conservative LLM call.

    An empty digest skips the LLM entirely — a day without proactive activity
    has nothing to learn from and costs no tokens.
    """
    settings = settings or get_settings()
    if not settings.enable_reflection:
        return {"ran": False, "reason": "disabled"}
    current = current or now(settings)

    digest = build_digest(settings, current)
    if not digest:
        return {"ran": False, "reason": "nothing to review"}

    from .llm import complete_text
    from .memory.learn import revise_memory, save_memory

    existing = self_notes(settings, k=10)
    existing_txt = (
        "\n".join(f"- name: {n.name} — {n.body}" for n in existing) or "(none yet)"
    )
    prompt = _REFLECT_PROMPT.format(
        cap=max(settings.reflection_max_ops, 1),
        existing=existing_txt,
        digest=digest,
    )
    try:
        raw = complete_text(prompt, settings)
    except Exception:
        logger.exception("reflection (LLM) failed")
        return {"ran": False, "reason": "llm failed"}

    applied: list[str] = []
    for op in parse_ops(raw, _ALLOWED_OPS)[: max(settings.reflection_max_ops, 1)]:
        try:
            if op["op"] == "save" and op.get("body"):
                note = save_memory(
                    settings,
                    body=str(op["body"]),
                    description=op.get("description"),
                    kind="procedural",
                    source="reflection",
                    tags=[_SELF_TAG],
                )
                applied.append(f"learned: {note.description}")
            elif op["op"] == "update" and op.get("name"):
                revised = revise_memory(
                    settings,
                    name=str(op["name"]),
                    body=op.get("body"),
                    description=op.get("description"),
                    kind="procedural",
                    tags=[_SELF_TAG],
                )
                if revised is not None:
                    applied.append(f"revised: {revised.description}")
        except Exception:
            logger.exception("applying reflection op %r failed", op)
    logger.info("reflection pass applied %d ops", len(applied))
    return {"ran": True, "applied": applied}
