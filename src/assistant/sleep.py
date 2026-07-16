"""Clock-driven sleep — nightly memory maintenance without a user turn.

The per-turn upkeep (:func:`assistant.chat.run_upkeep`) folds working memory and
counts toward consolidation, but it only runs when the user talks: a quiet week
gets no memory maintenance at all, and turn-counted consolidation never fires.
This is the time-driven counterpart — one pass per local date that keeps the
brain tidy whether or not anyone spoke.

Two things happen each pass:

* **Fold sweep** — :func:`assistant.agent.maybe_summarize` on every known thread,
  catching conversations whose per-turn fold failed or that ended mid-turn.
  Cheap no-op for a thread already under the working-memory cap.
* **Consolidation** — the deterministic steps (decay, prune, counter flush, caps,
  trash) run every night (they are free and time-driven). The LLM promote/merge/
  forget step is gated on there being an episode newer than the last LLM pass, so
  an idle night with nothing new to review costs no tokens.

Unlike reminders and the briefing, sleep runs **during quiet hours by design**:
it never pushes anything, and its default ``sleep_time`` sits inside the default
quiet window. State is a fired ledger (the shared :mod:`assistant.fired_ledger`
driver, exactly-once per local date) plus a tiny ``sleep_state`` KV holding the
last LLM-pass date — both in ``briefing.db``. The ticker and a manual
``POST /sleep/run`` drive the same function; the ledger makes them safe to race.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import time as dtime

from . import fired_ledger, threads
from .agent import maybe_summarize
from .calendar.context import now
from .config import Settings, get_settings, postgres_backend
from .memory import consolidate_memory, store

logger = logging.getLogger(__name__)

_LEDGER = fired_ledger.FiredLedgerSpec(
    table="sleep_fired",
    columns=(("local_date", "TEXT"),),
    db_path=lambda settings: settings.briefing_db_path,
)


def _due_time(settings: Settings) -> dtime:
    """Parse ``sleep_time`` (HH:MM); a malformed value falls back to 03:30."""
    try:
        hour, _, minute = settings.sleep_time.partition(":")
        return dtime(int(hour), int(minute))
    except ValueError:
        logger.warning("invalid SLEEP_TIME %r; using 03:30", settings.sleep_time)
        return dtime(3, 30)


_KV_NAMESPACE = "sleep"


# The sleep KV rides the briefing db alongside the fired ledger (same file the
# briefing/sleep ledgers live in), created lazily like heartbeat's state table.
# Under Postgres it lands in the shared assistant_kv table instead.
def _state_get(settings: Settings, key: str) -> str:
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.kv_get(settings, _KV_NAMESPACE, key)
    with fired_ledger.connect(_LEDGER, settings) as conn:
        _ensure_state(conn)
        row = conn.execute(
            "SELECT value FROM sleep_state WHERE key = ?", (key,)
        ).fetchone()
    return row["value"] if row else ""


def _state_set(settings: Settings, key: str, value: str) -> None:
    if storage_postgres := postgres_backend(settings):
        storage_postgres.kv_set(settings, _KV_NAMESPACE, key, value)
        return
    with fired_ledger.connect(_LEDGER, settings) as conn:
        _ensure_state(conn)
        conn.execute(
            "INSERT OR REPLACE INTO sleep_state (key, value) VALUES (?, ?)",
            (key, value),
        )


def _ensure_state(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS sleep_state"
        " (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )


def _newest_episode_date(settings: Settings) -> str:
    """The most recent episodic stamp (ISO date), or '' when there are none."""
    stamps = [
        note.updated or note.created
        for note in store.list_notes(settings)
        if note.kind == "episodic"
    ]
    return max(stamps, default="")


def _fold_all_threads(settings: Settings, agent) -> int:
    """Fold working memory on every known thread; return how many were touched."""
    if agent is None:
        return 0
    folded = 0
    for info in threads.known_threads(settings):
        try:
            maybe_summarize(agent, settings, info.thread_id)
            folded += 1
        except Exception:
            logger.exception("sleep: folding thread %s failed", info.thread_id)
    return folded


def run_sleep(
    settings: Settings | None = None, agent=None, force: bool = False
) -> dict:
    """Run tonight's maintenance pass if it is due and unclaimed.

    Gated by ``enable_sleep`` (``force`` overrides), a time-of-day check
    (``force`` bypasses), and the once-per-day ledger — claimed before any work
    so the ticker and ``POST /sleep/run`` never double-run. Quiet hours are
    *not* a gate here: this pass never pushes, and it is meant to run overnight.
    """
    settings = settings or get_settings()
    if not settings.enable_sleep and not force:
        return {"ran": False, "reason": "disabled"}

    current = now(settings)
    local_date = current.date().isoformat()
    if not force and current.time() < _due_time(settings):
        return {"ran": False, "reason": "not due yet"}

    claimed = fired_ledger.claim(
        _LEDGER, settings, [(local_date,)], current.isoformat(timespec="seconds"), current
    )
    if not claimed:
        return {"ran": False, "reason": "already ran today"}

    folded = _fold_all_threads(settings, agent)

    # The LLM step only earns its tokens when there is something new to review;
    # the free deterministic steps run either way.
    # ">" (not ">="): an episode dated the same day as the last pass was already
    # in view then, so it must not re-trigger a review every following night. The
    # LLM step reviews the most recent episodes wholesale, so a genuinely newer
    # one pulls the same-day stragglers in with it.
    last_pass = _state_get(settings, "last_llm_pass_at")
    include_llm = not last_pass or _newest_episode_date(settings) > last_pass
    consolidation = consolidate_memory(settings, include_llm=include_llm)
    if include_llm:
        _state_set(settings, "last_llm_pass_at", local_date)

    result = {
        "ran": True,
        "date": local_date,
        "folded": folded,
        "llm": include_llm,
        "consolidation": consolidation,
    }
    logger.info("sleep pass: %s", result)
    return result
