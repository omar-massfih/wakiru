"""The people read path: the roster injected into each turn, plus the shared
attention helpers (overdue-for-contact, upcoming birthday) the briefing and the
heartbeat reuse.

:func:`people_context` renders a compact roster — people needing attention
first — as a plain-text block the agent graph prepends as a ``SystemMessage``,
the same mechanism recall, the agenda, and tasks use. :func:`briefing_people`
and :func:`attention_lines` surface the same signals to the morning briefing and
the proactive heartbeat.
"""

from __future__ import annotations

from datetime import datetime

from ..calendar.context import now
from ..calendar.store import parse_dt
from ..config import Settings, get_settings
from . import store
from .store import Person


def parse_birthday(birthday: str) -> tuple[int, int] | None:
    """(month, day) from an ``MM-DD`` or ``YYYY-MM-DD`` birthday, or ``None``."""
    birthday = birthday.strip()
    if not birthday:
        return None
    parts = birthday.split("-")
    try:
        month, day = int(parts[-2]), int(parts[-1])
    except (IndexError, ValueError):
        return None
    if 1 <= month <= 12 and 1 <= day <= 31:
        return month, day
    return None


def days_until_birthday(person: Person, current: datetime) -> int | None:
    """Whole days until this person's next birthday (0 = today), or ``None``."""
    md = parse_birthday(person.birthday)
    if md is None:
        return None
    month, day = md
    today = current.date()
    year = today.year
    # Feb 29 in a common year falls back to Mar 1 so the count still resolves.
    for candidate_year in (year, year + 1):
        try:
            nxt = today.replace(year=candidate_year, month=month, day=day)
        except ValueError:
            nxt = today.replace(year=candidate_year, month=month, day=28)
            if day > 28:
                from datetime import timedelta

                nxt = nxt + timedelta(days=day - 28)
        if nxt >= today:
            return (nxt - today).days
    return None


def contact_gap_days(person: Person, current: datetime) -> int | None:
    """Whole days since the user last contacted this person, or ``None`` if never."""
    last = parse_dt(person.last_contact)
    if last is None:
        return None
    return max(0, (current - last).days)


def is_overdue(person: Person, current: datetime) -> bool:
    """True when a keep-in-touch cadence is set and it has lapsed (or was never
    logged) — the "haven't spoken to X in a while" signal."""
    if person.cadence_days <= 0:
        return False
    gap = contact_gap_days(person, current)
    return gap is None or gap >= person.cadence_days


def _birthday_soon(person: Person, settings: Settings, current: datetime) -> int | None:
    """Days-until when a birthday is within the lead window, else ``None``."""
    days = days_until_birthday(person, current)
    if days is not None and days <= settings.people_birthday_lead_days:
        return days
    return None


def _needs_attention(person: Person, settings: Settings, current: datetime) -> bool:
    return is_overdue(person, current) or _birthday_soon(person, settings, current) is not None


def _birthday_phrase(days: int) -> str:
    if days == 0:
        return "🎂 birthday today"
    if days == 1:
        return "🎂 birthday tomorrow"
    return f"🎂 birthday in {days}d"


def _render_person(person: Person, settings: Settings, current: datetime, with_id: bool) -> str:
    line = f"- {person.name}"
    if person.relationship:
        line += f" — {person.relationship}"
    notes: list[str] = []
    gap = contact_gap_days(person, current)
    if gap is not None:
        note = f"last contact {gap}d ago"
        if is_overdue(person, current):
            note += f"; overdue (every {person.cadence_days}d)"
        notes.append(note)
    elif person.cadence_days > 0:
        notes.append(f"no contact logged; keep in touch every {person.cadence_days}d")
    bday = _birthday_soon(person, settings, current)
    if bday is not None:
        notes.append(_birthday_phrase(bday))
    if notes:
        line += " (" + "; ".join(notes) + ")"
    if with_id:
        line += f"  [id: {person.id}]"
    return line


def roster(settings: Settings, current: datetime) -> list[Person]:
    """Everyone, people needing attention first, capped by ``people_max_open``."""
    people = store.list_people(settings)
    people.sort(
        key=lambda p: (not _needs_attention(p, settings, current), p.name.lower())
    )
    return people[: settings.people_max_open]


def render_people(
    settings: Settings, people: list[Person], current: datetime, with_ids: bool = True
) -> str:
    if not people:
        return "(no people recorded)"
    return "\n".join(_render_person(p, settings, current, with_ids) for p in people)


def people_context(settings: Settings | None = None) -> str:
    """The people roster injected ahead of the user's turn."""
    settings = settings or get_settings()
    current = now(settings)
    return "## People\n" + render_people(settings, roster(settings, current), current)


def attention_lines(settings: Settings, current: datetime) -> list[str]:
    """One line per person overdue for contact or with a birthday coming — for
    the heartbeat situation report and the briefing. Empty when nothing is due."""
    lines: list[str] = []
    for person in store.list_people(settings):
        bday = _birthday_soon(person, settings, current)
        if bday is not None:
            who = f"{person.name}" + (f" ({person.relationship})" if person.relationship else "")
            lines.append(f"{who}: {_birthday_phrase(bday)}")
        elif is_overdue(person, current):
            who = f"{person.name}" + (f" ({person.relationship})" if person.relationship else "")
            gap = contact_gap_days(person, current)
            when = f"{gap}d ago" if gap is not None else "not logged"
            lines.append(f"{who}: overdue to reach out (last contact {when})")
    return lines


def briefing_people(settings: Settings | None = None) -> str:
    """The people section for the daily briefing — birthdays and overdue
    contacts only (nothing when none), so the digest stays about what's due."""
    settings = settings or get_settings()
    lines = attention_lines(settings, now(settings))
    if not lines:
        return ""
    return "## People\n" + "\n".join(f"- {line}" for line in lines)
