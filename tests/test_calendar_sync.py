"""ICS pull-sync tests — parsing, upsert/delete mirroring, and the write guard.

The feed is injected as ICS text (urlopen monkeypatched), so parsing and the
store round-trip run for real against a tmp SQLite calendar.
"""

from __future__ import annotations

import json

import pytest

from assistant.calendar import store, sync
from assistant.config import Settings

FEED_URL = "https://calendar.example.com/secret/basic.ics"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(memory_dir=str(tmp_path / "memory"), timezone="Europe/Oslo")


def _ics(body: str) -> str:
    return (
        "BEGIN:VCALENDAR\r\nVERSION:2.0\r\nPRODID:-//test//EN\r\n"
        + body
        + "END:VCALENDAR\r\n"
    )


def _vevent(uid: str, summary: str, start: str, end: str = "", extra: str = "") -> str:
    lines = [f"BEGIN:VEVENT\r\nUID:{uid}\r\nDTSTAMP:20260701T000000Z\r\n"]
    lines.append(f"DTSTART:{start}\r\n")
    if end:
        lines.append(f"DTEND:{end}\r\n")
    lines.append(f"SUMMARY:{summary}\r\n")
    if extra:
        lines.append(extra)
    lines.append("END:VEVENT\r\n")
    return "".join(lines)


def _feed(monkeypatch, text: str) -> None:
    import contextlib
    import io

    from assistant import netguard

    @contextlib.contextmanager
    def fake_open(request, timeout=None):
        yield io.BytesIO(text.encode())

    monkeypatch.setattr(netguard, "require_public_url", lambda url: None)
    monkeypatch.setattr(netguard, "_open", fake_open)


def test_pull_mirrors_events(settings, monkeypatch) -> None:
    _feed(
        monkeypatch,
        _ics(
            _vevent("uid-1", "Dentist", "20260715T090000Z", "20260715T100000Z")
            + _vevent("uid-2", "Standup", "20260716T081500Z")
        ),
    )
    result = sync.pull_feed(settings, FEED_URL)
    assert result["added"] == 2 and result["removed"] == 0

    events = store.list_events(settings)
    assert {e.title for e in events} == {"Dentist", "Standup"}
    assert all(sync.is_synced_id(e.id) for e in events)


def test_repull_is_idempotent_and_updates_changes(settings, monkeypatch) -> None:
    _feed(monkeypatch, _ics(_vevent("uid-1", "Dentist", "20260715T090000Z")))
    sync.pull_feed(settings, FEED_URL)

    # Same feed again: nothing changes, ids stay stable.
    first_id = store.list_events(settings)[0].id
    result = sync.pull_feed(settings, FEED_URL)
    assert result["added"] == 0 and result["updated"] == 0
    assert store.list_events(settings)[0].id == first_id

    # The event moves in the source calendar: same row, new start.
    _feed(monkeypatch, _ics(_vevent("uid-1", "Dentist", "20260715T110000Z")))
    result = sync.pull_feed(settings, FEED_URL)
    assert result["updated"] == 1 and result["added"] == 0
    event = store.list_events(settings)[0]
    assert event.id == first_id and event.start.startswith("2026-07-15T11:00")


def test_event_deleted_in_source_is_removed(settings, monkeypatch) -> None:
    _feed(
        monkeypatch,
        _ics(
            _vevent("uid-1", "Dentist", "20260715T090000Z")
            + _vevent("uid-2", "Standup", "20260716T081500Z")
        ),
    )
    sync.pull_feed(settings, FEED_URL)
    _feed(monkeypatch, _ics(_vevent("uid-2", "Standup", "20260716T081500Z")))
    result = sync.pull_feed(settings, FEED_URL)
    assert result["removed"] == 1
    assert [e.title for e in store.list_events(settings)] == ["Standup"]


def test_local_events_are_untouched_by_sync(settings, monkeypatch) -> None:
    local = store.create_event(settings, "My own thing", "2026-07-20T12:00:00+02:00")
    _feed(monkeypatch, _ics(_vevent("uid-1", "Dentist", "20260715T090000Z")))
    sync.pull_feed(settings, FEED_URL)
    _feed(monkeypatch, _ics(""))  # feed empties out
    sync.pull_feed(settings, FEED_URL)
    remaining = store.list_events(settings)
    assert [e.id for e in remaining] == [local.id]


def test_recurring_event_keeps_rrule_and_exdates(settings, monkeypatch) -> None:
    _feed(
        monkeypatch,
        _ics(
            _vevent(
                "uid-r", "Weekly sync", "20260706T090000Z",
                extra="RRULE:FREQ=WEEKLY;BYDAY=MO\r\nEXDATE:20260713T090000Z\r\n",
            )
        ),
    )
    sync.pull_feed(settings, FEED_URL)
    event = store.list_events(settings)[0]
    assert "FREQ=WEEKLY" in event.rrule
    assert json.loads(event.exdates) and "2026-07-13" in json.loads(event.exdates)[0]


def test_all_day_event_becomes_local_midnight(settings, monkeypatch) -> None:
    _feed(
        monkeypatch,
        _ics(_vevent("uid-d", "Norway Day", "20260517", extra="")),
    )
    sync.pull_feed(settings, FEED_URL)
    event = store.list_events(settings)[0]
    assert event.start.startswith("2026-05-17T00:00")
    # No DTEND on a date-only DTSTART: RFC 5545 implies one full day — without
    # it the event would block only the store's default 60 minutes.
    assert event.end.startswith("2026-05-18T00:00")


def test_all_day_event_with_dtend_keeps_it(settings, monkeypatch) -> None:
    _feed(
        monkeypatch,
        _ics(_vevent("uid-e", "Offsite", "20260517", end="20260519")),
    )
    sync.pull_feed(settings, FEED_URL)
    event = store.list_events(settings)[0]
    assert event.end.startswith("2026-05-19T00:00")  # the implied-P1D fix never overrides


def test_write_path_refuses_synced_events(settings, monkeypatch) -> None:
    from assistant.calendar import ops

    _feed(monkeypatch, _ics(_vevent("uid-1", "Dentist", "20260715T090000Z")))
    sync.pull_feed(settings, FEED_URL)

    assert ops.apply_op(settings, {"op": "cancel", "query": "Dentist"}) is None
    assert store.list_events(settings), "synced event must survive a cancel attempt"


def test_pull_feeds_survives_a_broken_feed(settings, monkeypatch) -> None:
    from assistant import netguard

    settings.calendar_ics_urls = ["https://bad.example.com/x.ics"]

    def boom(request, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(netguard, "require_public_url", lambda url: None)
    monkeypatch.setattr(netguard, "_open", boom)
    assert sync.pull_feeds(settings) == []
