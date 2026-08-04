"""Meeting-prep tests — name matching, the time window, and provider gating.

Everything runs for real (plain SQLite); times are pinned relative to the
assistant's own "now" so the tests hold at any hour.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from assistant.calendar import store as calendar_store
from assistant.calendar.context import now
from assistant.config import Settings
from assistant.context_providers import build_context
from assistant.meeting_prep import meeting_prep_context
from assistant.people import store as people_store


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        enable_people=True,
    )


def _event(settings: Settings, title: str, minutes_from_now: int, **kwargs):
    start = now(settings) + timedelta(minutes=minutes_from_now)
    end = start + timedelta(minutes=45)
    return calendar_store.create_event(
        settings,
        title=title,
        start=start.isoformat(timespec="seconds"),
        end=end.isoformat(timespec="seconds"),
        **kwargs,
    )
# --- matching ----------------------------------------------------------------- #


def test_full_name_in_title_matches(settings) -> None:
    people_store.create_person(
        settings, name="Kari Nordmann", relationship="client", notes="prefers mornings"
    )
    _event(settings, "Contract review with Kari Nordmann", 30)
    block = meeting_prep_context(settings)
    assert "Meeting prep" in block
    assert "Contract review" in block
    assert "Kari Nordmann — client" in block
    assert "prefers mornings" in block


def test_unique_first_name_matches_ambiguous_does_not(settings) -> None:
    people_store.create_person(settings, name="Kari Nordmann")
    people_store.create_person(settings, name="Alex Berg")
    people_store.create_person(settings, name="Alex Dahl")
    _event(settings, "1:1 with Kari", 20)
    _event(settings, "Sync with Alex", 40)
    block = meeting_prep_context(settings)
    assert "Kari Nordmann" in block
    assert "1:1 with Kari" in block
    assert "Alex" not in block  # two Alexes — guessing would misbrief


def test_name_in_notes_matches_too(settings) -> None:
    people_store.create_person(settings, name="Ola Hansen", relationship="accountant")
    _event(settings, "Quarterly numbers", 30, notes="walkthrough with Ola Hansen")
    assert "Ola Hansen — accountant" in meeting_prep_context(settings)


def test_no_match_renders_nothing(settings) -> None:
    people_store.create_person(settings, name="Kari Nordmann")
    _event(settings, "Dentist", 30)
    assert meeting_prep_context(settings) == ""


def test_generic_title_matches_attendee_email_before_misleading_name(settings) -> None:
    kari = people_store.create_person(
        settings, name="Kari Nordmann", email="kari@example.com",
        relationship="client", notes="renewal is due",
    )
    people_store.create_person(settings, name="Ola Hansen", relationship="vendor")
    event = _event(settings, "Catch-up with Ola Hansen", 30)
    event.attendees = calendar_store.dump_attendees([
        {"email": " MAILTO:KARI@EXAMPLE.COM ", "status": "accepted"},
        {"email": "kari@example.com"},
    ])
    calendar_store.restore_event(settings, event)
    block = meeting_prep_context(settings)
    assert kari.name in block and "renewal is due" in block
    assert "Ola Hansen — vendor" not in block


def test_organizer_and_multiple_attendees_match_in_people_order(settings) -> None:
    alpha = people_store.create_person(settings, "Alpha One", email="alpha@example.com")
    beta = people_store.create_person(settings, "Beta Two", email="beta@example.com")
    event = _event(settings, "Planning", 15)
    event.organizer = calendar_store.dump_organizer({"email": "beta@example.com"})
    event.attendees = calendar_store.dump_attendees([
        {"email": "alpha@example.com"}, {"email": "beta@example.com"},
    ])
    calendar_store.restore_event(settings, event)
    block = meeting_prep_context(settings)
    assert block.index(alpha.name) < block.index(beta.name)
    assert block.count(beta.name) == 1


def test_ambiguous_or_malformed_email_does_not_guess(settings) -> None:
    people_store.create_person(settings, "Alex One", email="shared@example.com")
    people_store.create_person(settings, "Alex Two", email="shared@example.com")
    event = _event(settings, "Catch-up", 10)
    event.attendees = calendar_store.dump_attendees([{"email": "shared@example.com"}])
    calendar_store.restore_event(settings, event)
    assert meeting_prep_context(settings) == ""
    event.attendees = "not-json"
    event.title = "Catch-up with Alex One"
    calendar_store.restore_event(settings, event)
    assert "Alex One" in meeting_prep_context(settings)  # safe name fallback


def test_imported_ics_attendee_drives_meeting_prep(settings) -> None:
    from datetime import UTC

    from assistant.calendar import sync

    people_store.create_person(
        settings, "Kari Nordmann", email="kari@example.com",
        relationship="partner", notes="ask about launch",
    )
    start = (now(settings) + timedelta(minutes=20)).astimezone(UTC)
    end = start + timedelta(minutes=30)
    ical = (
        "BEGIN:VCALENDAR\r\nVERSION:2.0\r\nBEGIN:VEVENT\r\nUID:prep-1\r\n"
        f"DTSTART:{start.strftime('%Y%m%dT%H%M%SZ')}\r\n"
        f"DTEND:{end.strftime('%Y%m%dT%H%M%SZ')}\r\n"
        "SUMMARY:Catch-up\r\nATTENDEE:mailto:kari@example.com\r\n"
        "END:VEVENT\r\nEND:VCALENDAR\r\n"
    )
    event = next(iter(sync.parse_vevents(ical, settings).values()))
    event.id = "ics-prep"
    calendar_store.restore_event(settings, event)
    block = meeting_prep_context(settings)
    assert "Kari Nordmann — partner" in block
    assert "ask about launch" in block


# --- the window ---------------------------------------------------------------- #


def test_event_outside_the_window_is_silent(settings) -> None:
    people_store.create_person(settings, name="Kari Nordmann")
    _event(settings, "Lunch with Kari Nordmann", 240)
    assert meeting_prep_context(settings) == ""


def test_in_progress_event_still_preps(settings) -> None:
    people_store.create_person(settings, name="Kari Nordmann")
    _event(settings, "Workshop with Kari Nordmann", -10)  # started, not over
    assert "Workshop" in meeting_prep_context(settings)


def test_section_cap_holds(settings) -> None:
    people_store.create_person(settings, name="Kari Nordmann")
    for i in range(3):
        _event(settings, f"Meeting {i} with Kari Nordmann", 10 + i * 15)
    block = meeting_prep_context(settings)
    assert block.count("###") == 2


# --- gating --------------------------------------------------------------------- #


def test_disabled_by_zero_lead(settings) -> None:
    people_store.create_person(settings, name="Kari Nordmann")
    _event(settings, "1:1 with Kari Nordmann", 10)
    off = settings.model_copy(update={"meeting_prep_minutes": 0})
    assert meeting_prep_context(off) == ""


def test_provider_gated_on_people_and_lead(settings) -> None:
    people_store.create_person(settings, name="Kari Nordmann")
    _event(settings, "1:1 with Kari Nordmann", 10)
    blocks = build_context(settings, query="", thread_id="t")
    assert "Kari Nordmann" in blocks["meeting_prep"]
    off = settings.model_copy(update={"enable_people": False})
    assert "meeting_prep" not in build_context(off, query="", thread_id="t")
