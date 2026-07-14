"""Phrasing tests — deterministic natural reminder text, no LLM involved."""

from __future__ import annotations

from datetime import timedelta

from assistant import phrasing
from assistant.calendar.context import now
from assistant.config import Settings

_SETTINGS = Settings(timezone="Europe/Oslo")


def _start_in(minutes: int) -> str:
    return (now(_SETTINGS) + timedelta(minutes=minutes)).isoformat(timespec="seconds")


def test_humanize() -> None:
    assert phrasing._humanize(timedelta(minutes=30)) == "in 30 min"
    assert phrasing._humanize(timedelta(minutes=60)) == "in 1 hour"
    assert phrasing._humanize(timedelta(hours=2)) == "in 2 hours"
    assert phrasing._humanize(timedelta(days=1)) == "in 1 day"


def test_humanize_rounds_up_to_clean_boundaries() -> None:
    # The ticker fires seconds after each band boundary, so the countdown must
    # read as the clean boundary — "in 45 min", never the jittery "in 44 min".
    assert phrasing._humanize(timedelta(minutes=44, seconds=46)) == "in 45 min"
    assert phrasing._humanize(timedelta(minutes=14, seconds=26)) == "in 15 min"
    assert phrasing._humanize(timedelta(minutes=59, seconds=56)) == "in 1 hour"
    assert phrasing._humanize(timedelta(0)) == "now"
    assert phrasing._humanize(timedelta(seconds=-40)) == "now"  # at-start nudge


def test_humanize_ago() -> None:
    assert phrasing._humanize_ago(timedelta(seconds=10)) == "just now"
    assert phrasing._humanize_ago(timedelta(minutes=15)) == "15 min ago"
    assert phrasing._humanize_ago(timedelta(minutes=60)) == "1 hour ago"
    assert phrasing._humanize_ago(timedelta(hours=2)) == "2 hours ago"


def test_event_message_is_deterministic() -> None:
    start = _start_in(30)
    args = (_SETTINGS, "ev-1", "Dentist", start, timedelta(minutes=30), 60)
    assert phrasing.event_reminder_message(*args) == phrasing.event_reminder_message(*args)


def test_event_message_carries_title_countdown_and_clock() -> None:
    start = _start_in(30)
    message = phrasing.event_reminder_message(
        _SETTINGS, "ev-1", "Dentist", start, timedelta(minutes=30), 60
    )
    assert "Dentist" in message
    assert "30 min" in message
    assert start[11:16] in message  # local wall-clock HH:MM
    assert "⏰" not in message  # the delivery channels prepend the prefix


def test_event_messages_vary_across_events() -> None:
    start = _start_in(30)
    rendered = {
        phrasing.event_reminder_message(
            _SETTINGS, f"ev-{i}", "Dentist", start, timedelta(minutes=30), 60
        )
        for i in range(20)
    }
    assert len(rendered) > 1  # not everyone gets the same template


def test_task_overdue_phrasing() -> None:
    due = _start_in(-30)
    message = phrasing.task_reminder_message(
        _SETTINGS, "task-1", "submit form", due, timedelta(minutes=-30), -30
    )
    assert "submit form" in message
    assert "30 min ago" in message


def test_unparseable_start_falls_back_to_plain_phrase() -> None:
    message = phrasing.event_reminder_message(
        _SETTINGS, "ev-1", "Dentist", "not-a-date", timedelta(minutes=30), 60
    )
    assert message == "Dentist in 30 min"
