"""Profile layer tests — tagging, context injection, and quiet-hours parsing.

Notes are written through the real store (tmp dir); only embeddings would need
the model, so notes are created via store.write_note directly, not save_memory.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from assistant.config import Settings
from assistant.memory import profile, store
from assistant.memory.store import Note


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(memory_dir=str(tmp_path / "memory"))


def _note(settings: Settings, body: str, tags: list[str], kind: str = "semantic") -> Note:
    note = Note(name=store.slugify(body), description=body, body=body, kind=kind, tags=tags)
    store.write_note(settings, note)
    return note


def test_profile_context_empty_without_profile_notes(settings) -> None:
    _note(settings, "The user's dog is named Rex", tags=[])
    assert profile.profile_context(settings) == ""


def test_profile_context_lists_only_profile_notes(settings) -> None:
    _note(settings, "The user works 9-17 on weekdays", tags=["profile"])
    _note(settings, "The user's dog is named Rex", tags=[])
    _note(settings, "A raw episode", tags=["profile"], kind="episodic")  # never included
    context = profile.profile_context(settings)
    assert "works 9-17" in context
    assert "Rex" not in context and "episode" not in context
    assert context.startswith("## User profile")


def test_profile_context_surfaces_quiet_window(settings) -> None:
    # The effective window is surfaced even with no profile notes, so the agent
    # can reason about a reminder it's setting landing inside quiet hours.
    settings.quiet_hours_default = "22:00-07:30"
    context = profile.profile_context(settings)
    assert context.startswith("## User profile")
    assert "Quiet hours: 22:00–07:30" in context
    assert "held during this window" in context


def test_profile_disabled_is_empty(settings) -> None:
    settings.enable_profile = False
    _note(settings, "The user works 9-17 on weekdays", tags=["profile"])
    assert profile.profile_context(settings) == ""
    assert profile.quiet_hours(settings) is None


def test_quiet_hours_from_range(settings) -> None:
    _note(settings, "The user wants quiet hours from 22:00 to 07:00", tags=["profile"])
    start, end = profile.quiet_hours(settings)
    assert (start.hour, end.hour) == (22, 7)


def test_quiet_hours_from_after_phrase(settings) -> None:
    _note(settings, "Don't ping the user after 22:30", tags=["profile"])
    start, end = profile.quiet_hours(settings)
    assert (start.hour, start.minute) == (22, 30)
    assert end == profile._DEFAULT_QUIET_END


def test_non_quiet_profile_note_yields_no_window(settings) -> None:
    settings.quiet_hours_default = ""
    _note(settings, "The user works 9-17 on weekdays", tags=["profile"])
    assert profile.quiet_hours(settings) is None


def _at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 7, 11, hour, minute)


def test_in_quiet_hours_crossing_midnight(settings) -> None:
    _note(settings, "Quiet hours 22:00 to 07:00 please", tags=["profile"])
    assert profile.in_quiet_hours(settings, _at(23))
    assert profile.in_quiet_hours(settings, _at(3))
    assert not profile.in_quiet_hours(settings, _at(12))
    assert profile.in_quiet_hours(settings, _at(22, 0))
    assert not profile.in_quiet_hours(settings, _at(7, 0))


def test_heartbeat_quiet_hold_uses_trip_local_clock(settings, monkeypatch) -> None:
    from assistant import heartbeat
    from assistant.calendar import context as calendar_context
    from assistant.trips import store as trips_store

    traveling = settings.model_copy(
        update={
            "timezone": "Europe/Oslo",
            "enable_trips": True,
            "enable_heartbeat": True,
            "quiet_hours_default": "22:00-01:00",
        }
    )
    trips_store.create_trip(
        traveling, "New York", start="2026-03-07", end="2026-03-08",
        timezone="America/New_York",
    )
    instant = datetime(2026, 3, 8, 4, 0, tzinfo=UTC)  # 23:00 EST, but 05:00 in Oslo
    real_datetime = datetime

    class FrozenDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return instant.astimezone(tz) if tz is not None else instant.replace(tzinfo=None)

    monkeypatch.setattr(calendar_context, "datetime", FrozenDateTime)
    assert heartbeat.gather_situation(traveling) is None

    # The same held work is eligible on the first wake at the destination-local
    # end boundary (01:00 EST), without changing the trip or home configuration.
    released = datetime(2026, 3, 8, 6, 0, tzinfo=UTC)
    instant = released
    assert heartbeat.gather_situation(traveling) is not None


def test_in_quiet_hours_without_notes_or_default_is_false(settings) -> None:
    settings.quiet_hours_default = ""
    assert not profile.in_quiet_hours(settings, _at(3))


def test_default_quiet_window_applies_without_notes(settings) -> None:
    settings.quiet_hours_default = "22:00-07:30"  # conftest blanks it suite-wide
    start, end = profile.quiet_hours(settings)
    assert (start.hour, start.minute) == (22, 0)
    assert (end.hour, end.minute) == (7, 30)
    assert profile.in_quiet_hours(settings, _at(23))
    assert not profile.in_quiet_hours(settings, _at(8))


def test_stated_note_overrides_the_default_window(settings) -> None:
    settings.quiet_hours_default = "22:00-07:30"
    _note(settings, "Quiet hours 23:00 to 06:00 please", tags=["profile"])
    start, end = profile.quiet_hours(settings)
    assert (start.hour, end.hour) == (23, 6)


def test_unparseable_default_fails_open(settings) -> None:
    settings.quiet_hours_default = "night-time"
    assert profile.quiet_hours(settings) is None


def test_reminders_held_during_quiet_hours(settings, monkeypatch) -> None:
    # The quiet-hours hold lives in the heartbeat now: gather_situation returns
    # None before any reminder is computed or claimed, so a window surviving
    # the night surfaces on the first wake after quiet ends.
    from assistant import heartbeat
    from assistant.calendar import reminders as cal_reminders

    settings.enable_heartbeat = True
    _note(settings, "Quiet hours 22:00 to 07:00", tags=["profile"])
    monkeypatch.setattr(
        cal_reminders, "due_reminders", lambda s, c: pytest.fail("must hold during quiet hours")
    )
    assert heartbeat.gather_situation(settings, _at(23)) is None
