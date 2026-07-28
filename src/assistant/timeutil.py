"""Timezone-aware timestamp helpers shared by the local stores.

Every store stamps ``created``/``updated`` in the assistant's configured
timezone and attaches that timezone to naive ISO datetimes on the way in. These
two helpers hold that shared shape. Each wraps
:func:`assistant.calendar.context.resolve_tz`, imported lazily inside the body
because that module imports the stores at top level — a module-level import here
would close an import cycle.
"""

from __future__ import annotations

from datetime import datetime

from .config import Settings


def stamp_now(settings: Settings) -> str:
    """Current time in the assistant's timezone (seconds precision).

    Used for ``created``/``updated`` stamps, matching how every other stamp in
    the system is resolved.
    """
    from .calendar.context import resolve_tz

    return datetime.now(resolve_tz(settings)).isoformat(timespec="seconds")


def normalize_stamp(settings: Settings, value: str) -> str:
    """Attach the assistant's timezone to a naive ISO datetime on its way in.

    The write-path extractor is told to emit offsets, but an LLM slip must not
    poison the store. Blank or unparseable values pass through unchanged (they
    are filtered on read); a value that already carries an offset is left as-is.
    """
    value = value.strip()
    if not value:
        return value
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return value
    if dt.tzinfo is not None:
        return value
    from .calendar.context import resolve_tz

    return dt.replace(tzinfo=resolve_tz(settings)).isoformat()
