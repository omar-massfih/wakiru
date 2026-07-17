"""Event importance classification — per-event reminder cadence.

A dentist appointment deserves a heads-up days ahead; a coffee chat needs 15
minutes. :func:`tiers_for` grades upcoming events into two tiers with one
batched LLM call, and :func:`leads_for` maps each tier to its lead schedule
(:attr:`Settings.reminder_lead_minutes_critical` vs the normal
:attr:`Settings.reminder_lead_minutes`). :mod:`.reminders` consults both when
deciding which lead windows apply to an event.

Verdicts are cached in ``calendar.db`` keyed on the event id plus a hash of the
title, so each event is classified once ever — not per tick — and a retitled
event is regraded. A failed LLM call degrades to the normal tier, cached with
``source='fallback'`` and retried after a backoff, so a down model never blocks
reminders and never causes a call per tick.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta

from ..config import Settings
from ..sqlite_util import open_db, transaction
from .store import Event, parse_dt

logger = logging.getLogger(__name__)

TIER_CRITICAL = "critical"
TIER_NORMAL = "normal"

# A failed classification (source='fallback') is retried once this much time has
# passed, so a model outage costs at most one call per window — not one per
# tick — yet recovers well inside the multi-day critical lead.
FALLBACK_RETRY = timedelta(hours=1)
# Verdict rows older than this are pruned on write so the cache, which only
# ever grows, stays bounded (same discipline as the fired ledgers).
RETENTION_DAYS = 90

_SYSTEM = (
    "You grade calendar events by how important it is that the user is "
    "reminded well in advance. Reply with strict JSON only — a single object "
    "mapping each event id to \"critical\" or \"normal\". No other text.\n"
    "critical: events that are costly to miss or need preparation — medical "
    "appointments (doctor, dentist, hospital), flights and other booked "
    "travel, exams, job interviews, official/legal deadlines.\n"
    "normal: everything routine — meetings, social plans, errands, workouts.\n"
    "Titles may be in Norwegian (e.g. legetime, tannlege, eksamen, frist)."
)


def _open(settings: Settings) -> sqlite3.Connection:
    """Open calendar.db and ensure the verdict cache exists (WAL, fresh
    connection — the same discipline as the fired ledgers)."""
    conn = open_db(settings.calendar_db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS event_importance ("
        " event_id TEXT PRIMARY KEY, title_hash TEXT NOT NULL,"
        " tier TEXT NOT NULL, source TEXT NOT NULL, updated TEXT NOT NULL)"
    )
    return conn


@contextmanager
def _connect(settings: Settings) -> Iterator[sqlite3.Connection]:
    with transaction(_open(settings)) as conn:
        yield conn


def _title_hash(title: str) -> str:
    return hashlib.sha256(title.encode("utf-8")).hexdigest()[:16]


def leads_for(settings: Settings, tier: str) -> list[int]:
    """The lead schedule (minutes before start) for one tier."""
    if tier == TIER_CRITICAL:
        return settings.reminder_lead_minutes_critical
    return settings.reminder_lead_minutes


def max_lead_minutes(settings: Settings) -> int:
    """The furthest-out lead across all tiers — the reminder scan horizon."""
    return max(
        [*settings.reminder_lead_minutes, *settings.reminder_lead_minutes_critical],
        default=0,
    )


def _prune(conn: sqlite3.Connection, current: datetime) -> None:
    """Drop verdicts older than the retention window (and unparseable ones)."""
    cutoff = current - timedelta(days=RETENTION_DAYS)
    stale = [
        (row["event_id"],)
        for row in conn.execute("SELECT event_id, updated FROM event_importance")
        if (updated := parse_dt(row["updated"])) is None or updated < cutoff
    ]
    conn.executemany("DELETE FROM event_importance WHERE event_id = ?", stale)


def _classify_llm(settings: Settings, events: list[Event]) -> dict[str, str]:
    """One batched LLM grading call: ``{event_id: tier}``. Raises on failure.

    The reply is parsed defensively — the first ``{...}`` block is taken,
    unknown ids and tiers dropped — so a chatty model costs coverage, not a
    crash; uncovered events fall back per event in :func:`tiers_for`.
    """
    from ..llm import complete_text

    lines = "\n".join(
        f"{e.id} | {e.title} | starts {e.start}"
        + (f" | {e.notes[:120]}" if e.notes else "")
        for e in events
    )
    reply = complete_text(f"Events:\n{lines}", settings, system=_SYSTEM)
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in classification reply: {reply!r}")
    parsed = json.loads(match.group(0))
    known = {e.id for e in events}
    return {
        str(event_id): tier
        for event_id, tier in parsed.items()
        if str(event_id) in known and tier in (TIER_CRITICAL, TIER_NORMAL)
    }


def tiers_for(settings: Settings, events: Sequence[Event]) -> dict[str, str]:
    """The importance tier per event id, classifying cache misses in one call.

    Cached verdicts are reused only while the title hash still matches (a
    retitled event is regraded) and, for ``fallback`` rows, only inside the
    retry backoff. All misses go to the model in one batch; on any failure each
    miss degrades to :data:`TIER_NORMAL`, recorded as ``fallback`` so the next
    window retries. Never raises.
    """
    unique = list({e.id: e for e in events}.values())
    current = datetime.now().astimezone()
    tiers: dict[str, str] = {}
    misses: list[Event] = []
    with _connect(settings) as conn:
        rows = {
            row["event_id"]: row
            for row in conn.execute(
                "SELECT event_id, title_hash, tier, source, updated"
                " FROM event_importance"
            )
        }
    for event in unique:
        row = rows.get(event.id)
        if row is not None and row["title_hash"] == _title_hash(event.title):
            fresh_fallback = (
                row["source"] == "fallback"
                and (updated := parse_dt(row["updated"])) is not None
                and current - updated < FALLBACK_RETRY
            )
            if row["source"] == "llm" or fresh_fallback:
                tiers[event.id] = row["tier"]
                continue
        misses.append(event)

    if not misses:
        return tiers

    try:
        verdicts = _classify_llm(settings, misses)
        source = "llm"
    except Exception:
        logger.exception("importance classification failed; defaulting to normal")
        verdicts = {}
        source = "fallback"
    stamp = current.isoformat(timespec="seconds")
    with _connect(settings) as conn:
        _prune(conn, current)
        for event in misses:
            tier = verdicts.get(event.id, TIER_NORMAL)
            # An id the model skipped is a fallback verdict even on a call that
            # otherwise succeeded: it gets retried, not frozen at normal.
            row_source = source if event.id in verdicts else "fallback"
            tiers[event.id] = tier
            conn.execute(
                "INSERT OR REPLACE INTO event_importance"
                " (event_id, title_hash, tier, source, updated) VALUES (?, ?, ?, ?, ?)",
                (event.id, _title_hash(event.title), tier, row_source, stamp),
            )
    return tiers
