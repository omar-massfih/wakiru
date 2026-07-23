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
