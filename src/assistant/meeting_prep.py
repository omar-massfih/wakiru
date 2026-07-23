"""Meeting prep — who the user is about to meet, injected just in time.

A context provider joining two stores that already exist: when a calendar
event starts (or is running) within ``meeting_prep_minutes``, and someone
from the people store is named in its title or notes, the block carries the
event plus each matched person's stored detail (relationship, last contact,
birthday, notes). "What do I need to know for this meeting?" then answers
itself — and the persona can brief the user unprompted.

Matching is deterministic and token-free: a person matches an event when
their full name appears in the event's title or notes (case-insensitive), or
when their first name appears as a whole word *and* no one else in the store
shares that first name — "1:1 with Kari" finds Kari Nordmann, but an
ambiguous "Alex" matches no one rather than guessing. Events nobody matches
render nothing: the agenda already lists what's scheduled, this block only
adds who it's with.
"""

from __future__ import annotations

import re
from datetime import timedelta

from .calendar.context import busy_events, format_when, now
from .calendar.store import Event
from .config import Settings, get_settings
from .people import store as people_store
from .people.context import describe_person
from .people.store import Person

# At most this many upcoming events get a prep section — a token dial; two
# back-to-back meetings is the realistic worst case worth briefing at once.
_MAX_EVENTS = 2


def matched_people(people: list[Person], event: Event) -> list[Person]:
    """Everyone named in the event's title or notes, store order preserved."""
    text = f"{event.title} {event.notes}".lower()
    words = set(re.findall(r"[^\W\d_]+", text, re.UNICODE))
    first_names: dict[str, int] = {}
    for person in people:
        first = person.name.strip().lower().split()[0] if person.name.strip() else ""
        if first:
            first_names[first] = first_names.get(first, 0) + 1
    matches = []
    for person in people:
        name = person.name.strip().lower()
        if not name:
            continue
        first = name.split()[0]
        if name in text or (first in words and first_names.get(first) == 1):
            matches.append(person)
    return matches


def meeting_prep_context(settings: Settings | None = None) -> str:
    """The Meeting-prep block for the next ``meeting_prep_minutes`` — empty
    (and cheap) whenever no imminent event names a known person."""
    settings = settings or get_settings()
    lead = settings.meeting_prep_minutes
    if lead <= 0:
        return ""
    current = now(settings)
    events = busy_events(settings, current, current + timedelta(minutes=lead))
    if not events:
        return ""
    people = people_store.list_people(settings)
    if not people:
        return ""
    sections = []
    for event in events:
        matches = matched_people(people, event)
        if not matches:
            continue
        where = f", {event.location}" if event.location else ""
        lines = [f"### {event.title} — {format_when(settings, event.start)}{where}"]
        lines += [describe_person(settings, person, current) for person in matches]
        sections.append("\n".join(lines))
        if len(sections) >= _MAX_EVENTS:
            break
    if not sections:
        return ""
    header = (
        "## Meeting prep\n"
        "People from the user's circle in an imminent meeting — brief the "
        "user proactively if it helps, and log_contact afterwards when they "
        "say it happened."
    )
    return "\n".join([header, *sections])
