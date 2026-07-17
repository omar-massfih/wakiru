"""Append-only audit ledger for mailbox mutations.

Every archive / label / mark-read / reply-draft the assistant performs is
recorded here with its actor (``chat:<thread>`` or ``heartbeat``) and the
human summary the client returned. This is deliberately *not* the write-undo
ledger: that one is thread-scoped (it no-ops for the heartbeat), and a true
IMAP undo can't restore an expunged INBOX uid anyway. The ops themselves are
hand-recoverable; the ledger's job is accountability — the heartbeat is shown
its own recent actions on later wakes so it never re-triages the same mail.

Lives in ``mail.db`` locally, or ``assistant_mail_audit`` under Postgres.
"""

from __future__ import annotations

import logging

from ..calendar.context import now
from ..config import Settings, postgres_backend
from ..sqlite_util import open_db, transaction

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS mail_audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  actor TEXT NOT NULL,
  action TEXT NOT NULL,
  uid TEXT NOT NULL,
  detail TEXT NOT NULL,
  at TEXT NOT NULL
)
"""


def _connect(settings: Settings):
    conn = open_db(settings.mail_db_path)
    conn.execute(_SCHEMA)
    return conn


def record(settings: Settings, actor: str, action: str, uid: str, detail: str) -> None:
    """Log one mailbox mutation. Never raises — a failed audit write must not
    turn a succeeded mailbox operation into a tool failure."""
    at = now(settings).isoformat(timespec="seconds")
    try:
        if storage_postgres := postgres_backend(settings):
            storage_postgres.record_mail_audit(settings, actor, action, uid, detail, at)
            return
        with transaction(_connect(settings)) as conn:
            conn.execute(
                "INSERT INTO mail_audit (actor, action, uid, detail, at)"
                " VALUES (?, ?, ?, ?, ?)",
                (actor, action, uid, detail, at),
            )
    except Exception:
        logger.exception("mail audit write failed (the mailbox operation itself succeeded)")


def recent(settings: Settings, limit: int = 5, actor: str | None = None) -> list[dict]:
    """Newest-first audit rows, optionally filtered to one actor."""
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.mail_audit_rows(settings, limit=limit, actor=actor)
    with transaction(_connect(settings)) as conn:
        if actor is None:
            cur = conn.execute(
                "SELECT * FROM mail_audit ORDER BY id DESC LIMIT ?", (limit,)
            )
        else:
            cur = conn.execute(
                "SELECT * FROM mail_audit WHERE actor = ? ORDER BY id DESC LIMIT ?",
                (actor, limit),
            )
        return [dict(row) for row in cur.fetchall()]
