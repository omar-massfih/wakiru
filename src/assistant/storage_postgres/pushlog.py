"""Push log for the Postgres backend.

Mirrors the ``push_log`` table :mod:`assistant.reflect` keeps in
``followups.db``: one row per delivered proactive push, read back by the
nightly reflection digest. Append-heavy and small; rows older than the digest
window are pruned on write.
"""

from __future__ import annotations

from ..config import Settings
from .core import _rows, _schema_done, _schema_mark, connect


def ensure_push_log_schema(settings: Settings) -> None:
    if _schema_done(settings, "push_log"):
        return
    with connect(settings) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_push_log (
              id BIGSERIAL PRIMARY KEY,
              ts TEXT NOT NULL,
              kind TEXT NOT NULL,
              excerpt TEXT NOT NULL DEFAULT ''
            )
            """
        )
    _schema_mark(settings, "push_log")


def record_push_log(settings: Settings, ts: str, kind: str, excerpt: str) -> None:
    ensure_push_log_schema(settings)
    with connect(settings) as conn:
        conn.execute(
            "INSERT INTO assistant_push_log (ts, kind, excerpt) VALUES (%s, %s, %s)",
            (ts, kind, excerpt),
        )


def recent_push_log(settings: Settings) -> list[dict]:
    ensure_push_log_schema(settings)
    with connect(settings) as conn:
        rows = _rows(
            conn.execute("SELECT ts, kind, excerpt FROM assistant_push_log ORDER BY id")
        )
    return [dict(row) for row in rows]
