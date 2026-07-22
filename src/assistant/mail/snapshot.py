"""Cached unread-mail snapshot — mail context without IMAP on the reply path.

Every other feature contributes a per-turn context block; mail historically
could not, because rendering it means a network round-trip to the mailbox.
This module closes that gap with a snapshot: :func:`maybe_refresh` runs off
the reply path (riding the reminder ticker on its own
``email_snapshot_minutes`` cadence) and persists what it saw;
:func:`current` — the context provider — only ever reads the persisted
snapshot, stamped with its fetch time so the model never over-claims
freshness. The live path (``/email`` command, ``GET /email``) keeps using
:func:`assistant.mail.context.unread_summary`.

Persisted as a small JSON file under the memory dir (like
``telegram_chats.json``), so a restart doesn't blank the block.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ..calendar.context import now
from ..config import Settings, postgres_backend

logger = logging.getLogger(__name__)

# A snapshot older than this is withheld entirely: yesterday's inbox presented
# as context misleads more than it helps.
_MAX_AGE_HOURS = 24


# The snapshot lives in the shared KV table (namespace/key) under Postgres, or a
# small JSON file under the memory dir on the local backend.
_KV_NAMESPACE = "mail"
_KV_KEY = "snapshot"


def _path(settings: Settings) -> Path:
    return settings.memory_path / "mail_snapshot.json"


def _decode(payload: str) -> tuple[str, datetime] | None:
    try:
        raw = json.loads(payload)
        fetched_at = datetime.fromisoformat(raw["fetched_at"])
    except (KeyError, TypeError, ValueError):
        logger.warning("unreadable mail snapshot; refetching on the next tick")
        return None
    return str(raw.get("text", "")), fetched_at


def _load(settings: Settings) -> tuple[str, datetime] | None:
    if storage_postgres := postgres_backend(settings):
        payload = storage_postgres.kv_get(settings, _KV_NAMESPACE, _KV_KEY)
        return _decode(payload) if payload else None
    try:
        payload = _path(settings).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        logger.warning("unreadable mail snapshot; refetching on the next tick")
        return None
    return _decode(payload)


def _save(settings: Settings, text: str, fetched_at: datetime) -> None:
    payload = json.dumps(
        {"text": text, "fetched_at": fetched_at.isoformat(timespec="seconds")}
    )
    if storage_postgres := postgres_backend(settings):
        storage_postgres.kv_set(settings, _KV_NAMESPACE, _KV_KEY, payload)
        return
    settings.memory_path.mkdir(parents=True, exist_ok=True)
    target = _path(settings)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(target)


def _render(messages: list) -> str:
    if not messages:
        return "No unread mail."
    lines = [f"{len(messages)} unread message(s):"]
    lines += [
        f"- {m.subject or '(no subject)'} — from {m.sender}" for m in messages
    ]
    return "\n".join(lines)


def refresh(settings: Settings) -> str | None:
    """Fetch the mailbox now and persist the snapshot; ``None`` when disabled.

    The one place IMAP runs for the snapshot. Raises nothing: a failed fetch
    logs and leaves the previous snapshot in place (stale-but-honest beats
    blank, and beats error text riding into every prompt).
    """
    if not (settings.enable_email and settings.email_snapshot_minutes > 0):
        return None
    from . import client

    try:
        messages = client.list_recent(settings, unread_only=True)
    except Exception:
        logger.exception("mail snapshot refresh failed; keeping the previous one")
        return None
    text = _render(messages)
    _save(settings, text, now(settings))
    return text


def invalidate(settings: Settings) -> None:
    """Mark the persisted snapshot stale so the next ticker tick refetches.

    Called after a mutation that changes the unread set (archive, mark read):
    :func:`current` then withholds the outdated block instead of presenting a
    mailbox that no longer exists — without any IMAP on the tool path.
    """
    stored = _load(settings)
    if stored is None:
        return
    text, _ = stored
    _save(settings, text, datetime(1970, 1, 1, tzinfo=UTC))


def maybe_refresh(settings: Settings) -> None:
    """Refresh when the snapshot is older than its cadence (the ticker hook)."""
    if not (settings.enable_email and settings.email_snapshot_minutes > 0):
        return
    stored = _load(settings)
    if stored is not None:
        _, fetched_at = stored
        age = now(settings) - fetched_at
        if age < timedelta(minutes=settings.email_snapshot_minutes):
            return
    refresh(settings)


def content(settings: Settings) -> str:
    """The snapshot's raw text with no ``as of HH:MM`` stamp — ``""`` when
    disabled, never fetched, or too old.

    For change-detection (the heartbeat's mail-changed check) and pattern
    matching (mail_from watches): both must key off the unread set itself, not
    the fetch time. Hashing or scanning :func:`current`'s stamped block instead
    would flip on every refresh (a new stamp) even when the inbox is unchanged.
    """
    if not (settings.enable_email and settings.email_snapshot_minutes > 0):
        return ""
    stored = _load(settings)
    if stored is None:
        return ""
    text, fetched_at = stored
    if not text or now(settings) - fetched_at > timedelta(hours=_MAX_AGE_HOURS):
        return ""
    return text


def current(settings: Settings) -> str:
    """The snapshot as a context block, or ``""`` — never any I/O.

    Stamped with its fetch time ("as of 09:12") so the model presents it as a
    snapshot, not a live view. Empty when disabled, never fetched, or too old
    to be honest about.
    """
    if not (settings.enable_email and settings.email_snapshot_minutes > 0):
        return ""
    stored = _load(settings)
    if stored is None:
        return ""
    text, fetched_at = stored
    if not text or now(settings) - fetched_at > timedelta(hours=_MAX_AGE_HOURS):
        return ""
    stamp = fetched_at.strftime("%H:%M")
    return f"## Unread mail (snapshot as of {stamp})\n{text}"
