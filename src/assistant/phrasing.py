"""Deterministic reminder phrasing — the reflex path's fallback text.

Reminder nudges are normally composed by the model in the assistant's own
voice (:func:`assistant.compose.compose_push`); these templates seed the facts
the model composes from and are what its composition falls back to when it
fails or stalls, so a claimed reminder is never lost. Each message carries the
local wall-clock time and one of a few natural phrasings, chosen by a stable
hash of the reminder's identity: the same reminder always renders the same
text, different reminders vary.

Messages never include the ⏰ prefix — the delivery channels prepend it.
"""

from __future__ import annotations

import hashlib
from datetime import timedelta

from .calendar.context import resolve_tz
from .calendar.store import parse_dt
from .config import Settings


def _humanize(delta: timedelta) -> str:
    """Render a positive time-until as a short phrase: 'in 30 min' / 'in 1 hour'."""
    minutes = round(delta.total_seconds() / 60)
    if minutes < 1:
        return "now"
    if minutes < 60:
        return f"in {minutes} min"
    if minutes < 1440:
        hours = round(minutes / 60)
        return f"in {hours} hour{'s' if hours != 1 else ''}"
    days = round(minutes / 1440)
    return f"in {days} day{'s' if days != 1 else ''}"


def _humanize_ago(delta: timedelta) -> str:
    """Render a positive time-since as a short phrase: '30 min ago' / '1 hour ago'."""
    minutes = round(delta.total_seconds() / 60)
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes} min ago"
    if minutes < 1440:
        hours = round(minutes / 60)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = round(minutes / 1440)
    return f"{days} day{'s' if days != 1 else ''} ago"


def _pick(variants: list[str], *identity: object) -> str:
    """A stable member of ``variants`` for this identity.

    hashlib, not built-in ``hash()``: the latter is salted per process, and the
    choice must survive restarts so re-renders of the same reminder match.
    """
    digest = hashlib.sha256("|".join(str(part) for part in identity).encode()).digest()
    return variants[digest[0] % len(variants)]


def _clock(settings: Settings, iso: str) -> str | None:
    """The local wall-clock time (HH:MM) of an ISO instant, if parseable."""
    dt = parse_dt(iso)
    if dt is None:
        return None
    return dt.astimezone(resolve_tz(settings)).strftime("%H:%M")


def event_reminder_message(
    settings: Settings,
    event_id: str,
    title: str,
    start_iso: str,
    remaining: timedelta,
    slot: int,
) -> str:
    """A natural one-line nudge for an event starting in ``remaining``."""
    clock = _clock(settings, start_iso)
    if clock is None:
        return f"{title} {_humanize(remaining)}"
    rel = _humanize(remaining)
    minutes = remaining.total_seconds() / 60
    if minutes < 1:
        variants = [
            f"{title} is starting now ({clock}).",
            f"It's time: {title} — starting now ({clock}).",
        ]
    elif minutes < 10:
        variants = [
            f"{title} starts {rel} ({clock}).",
            f"Almost time: {title} at {clock} ({rel}).",
        ]
    elif minutes < 1440:
        variants = [
            f"Heads up: {title} at {clock} ({rel}).",
            f"{title} coming up at {clock} ({rel}).",
            f"Reminder: {title} at {clock} ({rel}).",
        ]
    else:
        day = parse_dt(start_iso).astimezone(resolve_tz(settings)).strftime("%A")
        variants = [
            f"Looking ahead: {title} {rel} ({day} at {clock}).",
            f"Coming up {rel}: {title}, {day} at {clock}.",
        ]
    return _pick(variants, event_id, start_iso, slot)


def task_reminder_message(
    settings: Settings,
    task_id: str,
    title: str,
    due_iso: str,
    remaining: timedelta,
    slot: int,
) -> str:
    """A natural one-line nudge for a task due in ``remaining`` (or overdue)."""
    if remaining < timedelta(0):
        return _overdue_task_message(settings, task_id, title, due_iso, -remaining, slot)
    clock = _clock(settings, due_iso)
    if clock is None:
        return f"Task due: {title} {_humanize(remaining)}"
    rel = _humanize(remaining)
    minutes = remaining.total_seconds() / 60
    if minutes < 10:
        variants = [
            f"{title} is due {rel} ({clock}).",
            f"Due {rel}: {title} ({clock}).",
        ]
    elif minutes < 1440:
        variants = [
            f"Coming due: {title} at {clock} ({rel}).",
            f"Don't forget: {title} — due at {clock} ({rel}).",
            f"Reminder: {title} is due at {clock} ({rel}).",
        ]
    else:
        day = parse_dt(due_iso).astimezone(resolve_tz(settings)).strftime("%A")
        variants = [
            f"Looking ahead: {title} is due {rel} ({day} at {clock}).",
            f"Coming due {rel}: {title}, {day} at {clock}.",
        ]
    return _pick(variants, task_id, due_iso, slot)


def _overdue_task_message(
    settings: Settings,
    task_id: str,
    title: str,
    due_iso: str,
    elapsed: timedelta,
    slot: int,
) -> str:
    ago = _humanize_ago(elapsed)
    variants = [
        f"Still open: {title} — was due {ago}.",
        f"{title} is still open (due {ago}).",
        f"Overdue: {title} — was due {ago}.",
    ]
    return _pick(variants, task_id, due_iso, slot)
