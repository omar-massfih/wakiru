"""Goal store for the Postgres backend.

Mirrors :mod:`assistant.goals` (standing multi-step intentions) so that under
``STORAGE_BACKEND=postgres`` goals live in Neon, not a prod-only
``followups.db``. Timestamps are ISO-8601 TEXT, byte-identical to the sqlite
store; goals are standing (raised, never claimed), so the only guarded write
is the ``WHERE status = 'open'`` on update/close.
"""

from __future__ import annotations

from ..config import Settings
from .core import _rows, _schema_done, _schema_mark, connect


def ensure_goals_schema(settings: Settings) -> None:
    if _schema_done(settings, "goals"):
        return
    with connect(settings) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_goals (
              id TEXT PRIMARY KEY,
              title TEXT NOT NULL,
              state TEXT NOT NULL DEFAULT '',
              next_action_at TEXT NOT NULL DEFAULT '',
              thread_id TEXT NOT NULL DEFAULT '',
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL DEFAULT '',
              status TEXT NOT NULL DEFAULT 'open',
              outcome TEXT NOT NULL DEFAULT ''
            )
            """
        )
    _schema_mark(settings, "goals")


def _from_row(row: dict):
    from ..goals import Goal

    return Goal(
        id=str(row["id"]),
        title=str(row["title"]),
        state=str(row.get("state") or ""),
        next_action_at=str(row.get("next_action_at") or ""),
        thread_id=str(row.get("thread_id") or ""),
        created_at=str(row.get("created_at") or ""),
        updated_at=str(row.get("updated_at") or ""),
        status=str(row.get("status") or "open"),
        outcome=str(row.get("outcome") or ""),
    )


def add_goal(settings: Settings, goal) -> None:
    ensure_goals_schema(settings)
    with connect(settings) as conn:
        conn.execute(
            "INSERT INTO assistant_goals"
            " (id, title, state, next_action_at, thread_id, created_at,"
            "  updated_at, status, outcome)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, 'open', '')",
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


def list_open_goals(settings: Settings) -> list:
    ensure_goals_schema(settings)
    with connect(settings) as conn:
        rows = _rows(
            conn.execute(
                "SELECT * FROM assistant_goals WHERE status = 'open'"
                " ORDER BY (next_action_at = ''), next_action_at"
            )
        )
    return [_from_row(row) for row in rows]


def update_goal(
    settings: Settings,
    ident: str,
    title: str,
    state: str,
    next_action_at: str,
    updated_at: str,
) -> bool:
    ensure_goals_schema(settings)
    with connect(settings) as conn:
        cur = conn.execute(
            "UPDATE assistant_goals SET title = %s, state = %s,"
            " next_action_at = %s, updated_at = %s"
            " WHERE id = %s AND status = 'open' RETURNING id",
            (title, state, next_action_at, updated_at, ident),
        )
        return cur.fetchone() is not None


def close_goal(
    settings: Settings, ident: str, status: str, outcome: str, updated_at: str
) -> bool:
    ensure_goals_schema(settings)
    with connect(settings) as conn:
        cur = conn.execute(
            "UPDATE assistant_goals SET status = %s, outcome = %s, updated_at = %s"
            " WHERE id = %s AND status = 'open' RETURNING id",
            (status, outcome, updated_at, ident),
        )
        return cur.fetchone() is not None
