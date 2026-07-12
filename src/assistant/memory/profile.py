"""The user profile — durable preference notes put to work (personalization).

No new data model: a *profile* note is any durable memory tagged ``profile``
(the per-turn extractor in :mod:`.learn` applies the tag to facts about how the
user lives and works — working hours, locations, quiet hours, tone). This module
gives those notes two outputs:

* :func:`profile_context` — a system-prompt block injected every turn (see the
  ``profile`` node in :mod:`assistant.agent`), so the model adapts tone,
  scheduling suggestions, and conflict warnings to the person it serves.
* :func:`quiet_hours` / :func:`in_quiet_hours` — the one preference *code*
  needs: the reminder tickers and the daily briefing hold non-urgent pushes
  while the user has asked not to be pinged.

Parsing free-text notes is deliberately lenient and fail-open: an unparseable
note simply contributes context, never an exception or a wrong hold.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, time as dtime

from ..config import Settings
from . import store
from .store import Note

logger = logging.getLogger(__name__)

_PROFILE_TAG = "profile"

# "22-07", "22:00-07:00", "10 pm to 7 am" — first time wins as start, second as end.
_TIME_RANGE_RE = re.compile(
    r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*(?:-|–|—|to|until)\s*"
    r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
    re.IGNORECASE,
)
# "after 22:00", "past 10pm" — start only; quiet then runs to a morning default.
_AFTER_RE = re.compile(r"(?:after|past|från|fra|etter)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", re.IGNORECASE)
_QUIET_HINTS = ("quiet", "do not disturb", "don't ping", "dont ping", "no notif", "ikke forstyrr")
_DEFAULT_QUIET_END = dtime(7, 0)


def _hour(raw_hour: str, raw_minute: str | None, meridiem: str | None) -> dtime | None:
    hour, minute = int(raw_hour), int(raw_minute or 0)
    if meridiem:
        hour = hour % 12 + (12 if meridiem.lower() == "pm" else 0)
    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return dtime(hour, minute)
    return None


def profile_notes(settings: Settings) -> list[Note]:
    """Durable notes tagged ``profile``, freshest first."""
    notes = [
        n
        for n in store.list_notes(settings)
        if _PROFILE_TAG in n.tags and n.kind != "episodic"
    ]
    notes.sort(key=lambda n: n.updated, reverse=True)
    return notes


def profile_context(settings: Settings) -> str:
    """The system-prompt block describing who the assistant serves ('' if none)."""
    if not settings.enable_profile:
        return ""
    notes = profile_notes(settings)
    if not notes:
        return ""
    lines = [
        "## User profile",
        "Durable preferences about how this user lives and works. Respect them",
        "when scheduling, suggesting times, warning about conflicts, and choosing",
        "your tone:",
        "",
    ]
    lines.extend(f"- {note.body}" for note in notes)
    return "\n".join(lines)


# The reminder tickers ask for the quiet window every tick (each minute), and
# listing notes is a full store scan — a network round-trip on Postgres. Quiet
# hours change rarely, so the parsed window is cached briefly per store; a new
# stated preference takes effect within this TTL.
_QUIET_TTL_SECONDS = 300
_quiet_cache: dict[str, tuple[float, tuple[dtime, dtime] | None]] = {}


def quiet_hours(settings: Settings) -> tuple[dtime, dtime] | None:
    """The (start, end) quiet window from the profile, or ``None`` if unstated.

    Fail-open by design: a storage error means "no quiet window" (pushes flow)
    rather than an exception in the tick loop.
    """
    if not settings.enable_profile:
        return None
    cache_key = f"{settings.storage_backend}:{settings.memory_dir}"
    cached = _quiet_cache.get(cache_key)
    if cached is not None and time.monotonic() - cached[0] < _QUIET_TTL_SECONDS:
        return cached[1]
    try:
        window = _parse_quiet_hours(settings)
    except Exception:
        logger.exception("reading profile notes for quiet hours failed; assuming none")
        window = None
    _quiet_cache[cache_key] = (time.monotonic(), window)
    return window


def _parse_quiet_hours(settings: Settings) -> tuple[dtime, dtime] | None:
    for note in profile_notes(settings):
        text = note.body.lower()
        if not any(hint in text for hint in _QUIET_HINTS):
            continue
        if match := _TIME_RANGE_RE.search(text):
            start = _hour(match.group(1), match.group(2), match.group(3))
            end = _hour(match.group(4), match.group(5), match.group(6))
            if start and end:
                return start, end
        if match := _AFTER_RE.search(text):
            start = _hour(match.group(1), match.group(2), match.group(3))
            if start:
                return start, _DEFAULT_QUIET_END
    return None


def in_quiet_hours(settings: Settings, current: datetime) -> bool:
    """Whether ``current`` falls inside the user's stated quiet window.

    A window that crosses midnight (22:00–07:00) is the common case and handled
    explicitly. No profile / no quiet note => never quiet (fail open: pushes flow).
    """
    window = quiet_hours(settings)
    if window is None:
        return False
    start, end = window
    now_t = current.time()
    if start <= end:
        return start <= now_t < end
    return now_t >= start or now_t < end
