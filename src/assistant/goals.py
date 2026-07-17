"""Goals — the assistant's standing multi-step intentions.

A followup is one guaranteed check-in; a goal is an ongoing project the
assistant carries across wakes and conversations: "plan the Oslo trip",
"find a better electricity deal". The ``state`` field is the model's own
working document — plan, progress, dead ends — which it rewrites with
``update_goal`` as it advances, so every future wake (and every chat turn,
via the goals context provider) picks up where the last one left off.

Unlike followups, goals are standing, not consumed: ``next_action_at`` says
when the next step is worth attempting, and the heartbeat raises a ready goal
without claiming it — the model itself moves ``next_action_at`` forward when
it finishes a step. Rows live in ``followups.db`` (same file, same
claim-discipline conventions); under ``STORAGE_BACKEND=postgres`` they live
in ``assistant_goals``.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

from .calendar.context import now
from .calendar.store import parse_dt
from .config import Settings, postgres_backend
from .sqlite_util import open_db, transaction

logger = logging.getLogger(__name__)

# The state field is the model's scratchpad, not an essay: capped so a
# runaway rewrite can't bloat every future prompt that carries the goal.
STATE_MAX_CHARS = 2000


@dataclass(frozen=True)
class Goal:
    id: str
    title: str
    state: str = ""
    next_action_at: str = ""
    thread_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    status: str = "open"  # open | done | abandoned
    outcome: str = ""


def _open(settings: Settings) -> sqlite3.Connection:
    conn = open_db(settings.followups_db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS goals ("
        " id TEXT PRIMARY KEY, title TEXT NOT NULL,"
        " state TEXT NOT NULL DEFAULT '', next_action_at TEXT NOT NULL DEFAULT '',"
        " thread_id TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,"
        " updated_at TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'open',"
        " outcome TEXT NOT NULL DEFAULT '')"
    )
    return conn


@contextmanager
def _connect(settings: Settings) -> Iterator[sqlite3.Connection]:
    with transaction(_open(settings)) as conn:
        yield conn


def _from_row(row: sqlite3.Row) -> Goal:
    return Goal(**dict(row))


def _clip_state(state: str) -> str:
    state = str(state).strip()
    return state[:STATE_MAX_CHARS]


def open_goal(
    settings: Settings,
    title: str,
    state: str = "",
    next_action_at: datetime | None = None,
    thread_id: str = "",
) -> Goal | None:
    """Open one standing goal; ``None`` when the open-goal cap is reached."""
    if len(list_open(settings)) >= max(settings.goals_max_open, 0):
        return None
    stamp = now(settings).isoformat(timespec="seconds")
    goal = Goal(
        id=uuid.uuid4().hex[:12],
        title=str(title).strip(),
        state=_clip_state(state),
        next_action_at=(
            next_action_at.isoformat(timespec="seconds") if next_action_at else ""
        ),
        thread_id=thread_id,
        created_at=stamp,
        updated_at=stamp,
    )
    if storage_postgres := postgres_backend(settings):
        storage_postgres.add_goal(settings, goal)
        return goal
    with _connect(settings) as conn:
        conn.execute(
            "INSERT INTO goals"
            " (id, title, state, next_action_at, thread_id, created_at,"
            "  updated_at, status, outcome)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, 'open', '')",
            (
                goal.id,
                goal.title,
                goal.state,
                goal.next_action_at,
                goal.thread_id,
                goal.created_at,
                goal.updated_at,
            ),
        )
    return goal


def list_open(settings: Settings) -> list[Goal]:
    """Every open goal, soonest next action first (never-scheduled ones last)."""
    if storage_postgres := postgres_backend(settings):
        rows = storage_postgres.list_open_goals(settings)
    else:
        with _connect(settings) as conn:
            rows = [
                _from_row(row)
                for row in conn.execute(
                    "SELECT * FROM goals WHERE status = 'open'"
                    " ORDER BY next_action_at = '', next_action_at"
                ).fetchall()
            ]
    return rows


def ready(settings: Settings, current: datetime | None = None) -> list[Goal]:
    """Open goals whose next action is due — raised, never claimed."""
    current = current or now(settings)
    return [
        goal
        for goal in list_open(settings)
        if (due := parse_dt(goal.next_action_at)) is not None and due <= current
    ]


def _match_open(settings: Settings, ident: str, action: str) -> Goal | None:
    """Resolve an open goal by id or an unambiguous title reference.

    Refuses ambiguity rather than guessing — the same rule followups and the
    calendar targets follow.
    """
    ident = str(ident).strip()
    if not ident:
        return None
    candidates = list_open(settings)
    matches = [g for g in candidates if g.id == ident]
    if not matches:
        needle = ident.lower()
        matches = [g for g in candidates if needle in g.title.lower()]
    if len(matches) != 1:
        if len(matches) > 1:
            logger.warning(
                "goal %s target %r is ambiguous between %d goals; skipping",
                action,
                ident,
                len(matches),
            )
        return None
    return matches[0]


def update(
    settings: Settings,
    ident: str,
    state: str | None = None,
    next_action_at: datetime | None = None,
    title: str | None = None,
    clear_next_action: bool = False,
) -> Goal | None:
    """Revise an open goal's working state, next action time, or title.

    ``clear_next_action`` parks the goal (no scheduled next step) — the model's
    way of saying "waiting on the world, don't raise me". Returns the revised
    row, or ``None`` when nothing matched or no field was given.
    """
    if state is None and next_action_at is None and title is None and not clear_next_action:
        return None
    target = _match_open(settings, ident, "update")
    if target is None:
        return None
    if clear_next_action:
        next_stamp = ""
    elif next_action_at is not None:
        next_stamp = next_action_at.isoformat(timespec="seconds")
    else:
        next_stamp = target.next_action_at
    revised = Goal(
        id=target.id,
        title=str(title).strip() if title is not None else target.title,
        state=_clip_state(state) if state is not None else target.state,
        next_action_at=next_stamp,
        thread_id=target.thread_id,
        created_at=target.created_at,
        updated_at=now(settings).isoformat(timespec="seconds"),
        status=target.status,
    )
    if storage_postgres := postgres_backend(settings):
        updated = storage_postgres.update_goal(
            settings,
            revised.id,
            revised.title,
            revised.state,
            revised.next_action_at,
            revised.updated_at,
        )
        return revised if updated else None
    with _connect(settings) as conn:
        cursor = conn.execute(
            "UPDATE goals SET title = ?, state = ?, next_action_at = ?, updated_at = ?"
            " WHERE id = ? AND status = 'open'",
            (
                revised.title,
                revised.state,
                revised.next_action_at,
                revised.updated_at,
                revised.id,
            ),
        )
    return revised if cursor.rowcount else None


def close(
    settings: Settings, ident: str, outcome: str = "", abandoned: bool = False
) -> Goal | None:
    """Close an open goal as done (or abandoned) with a one-line outcome.

    The closure is recorded as an episodic trace, so nightly consolidation can
    promote what worked (or what kept failing) into durable lessons.
    """
    target = _match_open(settings, ident, "close")
    if target is None:
        return None
    status = "abandoned" if abandoned else "done"
    stamp = now(settings).isoformat(timespec="seconds")
    outcome = str(outcome).strip()
    if storage_postgres := postgres_backend(settings):
        if not storage_postgres.close_goal(settings, target.id, status, outcome, stamp):
            return None
    else:
        with _connect(settings) as conn:
            cursor = conn.execute(
                "UPDATE goals SET status = ?, outcome = ?, updated_at = ?"
                " WHERE id = ? AND status = 'open'",
                (status, outcome, stamp, target.id),
            )
        if not cursor.rowcount:
            return None
    try:
        from .memory.learn import record_episode

        record_episode(
            settings,
            f"Goal {status}: {target.title}",
            outcome or target.state[:600],
            source="goal",
        )
    except Exception:
        logger.exception("recording the goal closure episode failed")
    return Goal(
        id=target.id,
        title=target.title,
        state=target.state,
        next_action_at=target.next_action_at,
        thread_id=target.thread_id,
        created_at=target.created_at,
        updated_at=stamp,
        status=status,
        outcome=outcome,
    )


def stale(settings: Settings, current: datetime | None = None) -> list[Goal]:
    """Open goals untouched for ``goal_stale_days`` — surfaced, never auto-closed."""
    days = settings.goal_stale_days
    if days <= 0:
        return []
    current = current or now(settings)
    out: list[Goal] = []
    for goal in list_open(settings):
        touched = parse_dt(goal.updated_at) or parse_dt(goal.created_at)
        if touched is not None and (current - touched).days >= days:
            out.append(goal)
    return out
