"""Google Calendar REST provider tests — pull, push, undo, conflict, recreate.

The single seam ``google_calendar._request`` is monkeypatched with a recorder that
returns canned ``(status, headers, json-body)`` tuples, so the store round-trip and
the shared push/pull/undo machinery run for real offline.
"""

from __future__ import annotations

import json

import pytest

from assistant.calendar import google_calendar, ops, outbox, remote, store, sync, undo
from assistant.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        caldav_provider="google",
        google_calendar_id="primary",
        enable_caldav=True,
        enable_caldav_write=True,
        enable_write_confirmation=True,
        caldav_auth="oauth",
        caldav_oauth_client_id="cid",
        caldav_oauth_client_secret="secret",
        caldav_oauth_refresh_token="refresh",
    )


class FakeGoogle:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.list_items: list[dict] = []
        self.insert_status = 200
        self.update_status = 200
        self.delete_status = 204
        self.etag = '"g-1"'
        self.raise_exc: Exception | None = None

    def __call__(self, method, url, *, settings, body=b"", headers=None):
        self.calls.append(
            {"method": method, "url": url,
             "body": body.decode() if body else "", "headers": headers or {}}
        )
        if self.raise_exc is not None:
            raise self.raise_exc
        if method == "GET":
            return 200, {}, json.dumps({"items": self.list_items}).encode()
        if method == "POST":
            payload = json.loads(body)
            return self.insert_status, {}, json.dumps(
                {"id": payload.get("id", "gen"), "etag": self.etag}
            ).encode()
        if method == "PUT":
            gid = url.rsplit("/", 1)[-1]
            return self.update_status, {}, json.dumps({"id": gid, "etag": self.etag}).encode()
        if method == "DELETE":
            return self.delete_status, {}, b""
        return 200, {}, b"{}"

    def of(self, method: str) -> list[dict]:
        return [c for c in self.calls if c["method"] == method]


@pytest.fixture
def gserver(monkeypatch) -> FakeGoogle:
    fake = FakeGoogle()
    monkeypatch.setattr(google_calendar, "_request", fake)
    return fake


def test_provider_selected(settings) -> None:
    assert remote.is_google(settings) and remote.is_configured(settings)


def test_pull_maps_google_events(settings, gserver) -> None:
    gserver.list_items = [
        {"id": "aaa", "summary": "Dentist",
         "start": {"dateTime": "2026-12-01T09:00:00+01:00"},
         "end": {"dateTime": "2026-12-01T10:00:00+01:00"}, "etag": '"e1"'},
        {"id": "bbb", "summary": "Standup",
         "start": {"dateTime": "2026-12-02T08:15:00+01:00"}, "etag": '"e2"'},
    ]
    result = sync.pull_caldav(settings)
    assert result["added"] == 2
    events = {e.title: e for e in store.list_events(settings)}
    assert events["Dentist"].caldav_href == "aaa"
    assert events["Dentist"].caldav_etag == '"e1"'
    assert not sync.is_synced_id(events["Dentist"].id)  # writable


def test_pull_skips_cancelled_and_folds_instances(settings, gserver) -> None:
    gserver.list_items = [
        {"id": "gone", "status": "cancelled"},
        {"id": "series", "summary": "Weekly",
         "start": {"dateTime": "2026-12-07T09:00:00+01:00"},
         "recurrence": ["RRULE:FREQ=WEEKLY;BYDAY=MO"], "etag": '"s1"'},
        {"id": "inst", "summary": "Weekly (moved)", "recurringEventId": "series",
         "originalStartTime": {"dateTime": "2026-12-14T09:00:00+01:00"},
         "start": {"dateTime": "2026-12-14T11:00:00+01:00"}, "etag": '"i1"'},
    ]
    sync.pull_caldav(settings)
    rows = store.list_events(settings)
    assert [e.title for e in rows] == ["Weekly"]
    overrides = store.load_overrides(rows[0])
    assert any(v.get("title") == "Weekly (moved)" for v in overrides.values())


def test_create_posts_with_our_id(settings, gserver) -> None:
    summary = ops.apply_op(
        settings, {"op": "create", "title": "Lunch", "start": "2026-12-05T12:00:00+01:00"}
    )
    assert summary and "not yet synced" not in summary
    posts = gserver.of("POST")
    assert len(posts) == 1
    payload = json.loads(posts[0]["body"])
    assert payload["summary"] == "Lunch"
    event = store.find_event(settings, "Lunch")
    assert payload["id"] == event.id            # our id becomes the Google id
    assert payload["colorId"] in {str(value) for value in range(1, 12)}
    assert event.caldav_href == event.id and event.caldav_etag == '"g-1"'


def test_reschedule_puts_with_if_match(settings, gserver) -> None:
    ops.apply_op(settings, {"op": "create", "title": "Lunch", "start": "2026-12-05T12:00:00+01:00"})
    original_color = json.loads(gserver.of("POST")[0]["body"])["colorId"]
    gserver.calls.clear()
    ops.apply_op(
        settings, {"op": "reschedule", "query": "Lunch", "start": "2026-12-05T13:00:00+01:00"}
    )
    puts = gserver.of("PUT")
    assert len(puts) == 1
    assert puts[0]["headers"].get("If-Match") == '"g-1"'
    assert "2026-12-05T13:00:00+01:00" in puts[0]["body"]
    assert json.loads(puts[0]["body"])["colorId"] == original_color


def test_event_colors_vary_but_are_stable() -> None:
    colors = [google_calendar._color_id(f"event-{index}") for index in range(30)]

    assert len(set(colors)) > 1
    assert colors == [google_calendar._color_id(f"event-{index}") for index in range(30)]
    assert set(colors) <= {str(value) for value in range(1, 12)}


def test_cancel_deletes(settings, gserver) -> None:
    ops.apply_op(settings, {"op": "create", "title": "Lunch", "start": "2026-12-05T12:00:00+01:00"})
    event = store.find_event(settings, "Lunch")
    gserver.calls.clear()
    ops.apply_op(settings, {"op": "cancel", "query": "Lunch"})
    deletes = gserver.of("DELETE")
    assert len(deletes) == 1 and deletes[0]["url"].endswith(event.caldav_href)
    assert deletes[0]["headers"].get("If-Match") == '"g-1"'


def test_update_gone_recreates(settings, gserver) -> None:
    ops.apply_op(settings, {"op": "create", "title": "Lunch", "start": "2026-12-05T12:00:00+01:00"})
    gserver.calls.clear()
    gserver.update_status = 404  # remote resource vanished
    ops.apply_op(settings, {"op": "reschedule", "query": "Lunch", "start": "2026-12-05T13:00:00+01:00"})
    assert gserver.of("PUT") and gserver.of("POST")  # PUT 404 → fell back to POST insert


def test_conflict_lands_locally_and_queues(settings, gserver) -> None:
    ops.apply_op(settings, {"op": "create", "title": "Lunch", "start": "2026-12-05T12:00:00+01:00"})
    gserver.update_status = 412
    summary = ops.apply_op(
        settings, {"op": "reschedule", "query": "Lunch", "start": "2026-12-05T13:00:00+01:00"}
    )
    assert summary and "not yet synced" in summary
    assert store.find_event(settings, "Lunch").start.startswith("2026-12-05T13:00")
    assert len(outbox.pending(settings)) == 1


def test_outage_queues_then_reconcile_drains(settings, gserver) -> None:
    gserver.raise_exc = google_calendar.GoogleCalError("down")
    summary = ops.apply_op(
        settings, {"op": "create", "title": "Lunch", "start": "2026-12-05T12:00:00+01:00"}
    )
    assert "not yet synced" in summary
    assert len(outbox.pending(settings)) == 1
    gserver.raise_exc = None
    gserver.calls.clear()
    result = sync.reconcile_caldav(settings)
    assert result["reconciled"] == 1 and outbox.pending(settings) == []
    assert gserver.of("POST")  # re-inserted on reconcile


def test_undo_create_deletes_remote(settings, gserver) -> None:
    ops.apply_op(
        settings, {"op": "create", "title": "Lunch", "start": "2026-12-05T12:00:00+01:00"},
        thread_id="t1", batch_id="b1",
    )
    gserver.calls.clear()
    undo.undo_latest(settings, "t1", window_minutes=60)
    assert store.find_event(settings, "Lunch") is None
    assert len(gserver.of("DELETE")) == 1
