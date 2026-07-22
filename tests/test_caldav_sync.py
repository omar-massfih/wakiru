"""Two-way CalDAV sync tests — pull, push, undo, conflict, and reconcile.

The single network seam ``caldav._request`` is monkeypatched with a recorder that
returns canned ``(status, headers, body)`` tuples, so protocol construction (REPORT
XML, PUT/DELETE headers + VCALENDAR body) and the store round-trip run for real
offline — the same discipline as ``test_calendar_sync.py``.
"""

from __future__ import annotations

import json

import pytest

from assistant.calendar import caldav, caldav_oauth, ops, outbox, store, sync, undo
from assistant.config import Settings

CALDAV_URL = "https://dav.example.com/cal/"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        caldav_url=CALDAV_URL,
        caldav_username="me@example.com",
        caldav_password="app-password",
        enable_caldav=True,
        enable_caldav_write=True,
        enable_write_confirmation=True,
    )


class FakeServer:
    """Records requests and returns programmed responses for the ``_request`` seam."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.report_body = _multistatus()
        self.put_status = 201
        self.put_etag = '"etag-1"'
        self.delete_status = 204
        self.raise_exc: Exception | None = None

    def __call__(self, method, url, *, settings, body=b"", headers=None):
        self.calls.append(
            {"method": method, "url": url,
             "body": body.decode() if body else "", "headers": headers or {}}
        )
        if self.raise_exc is not None:
            raise self.raise_exc
        if method == "REPORT":
            return 207, {}, self.report_body
        if method == "PUT":
            return self.put_status, {"etag": self.put_etag}, b""
        if method == "DELETE":
            return self.delete_status, {}, b""
        return 200, {}, b""

    def of_method(self, method: str) -> list[dict]:
        return [c for c in self.calls if c["method"] == method]


@pytest.fixture
def server(monkeypatch) -> FakeServer:
    fake = FakeServer()
    monkeypatch.setattr(caldav, "_request", fake)
    return fake


def _vevent(uid: str, summary: str, start: str, end: str = "", extra: str = "") -> str:
    lines = [f"BEGIN:VEVENT\nUID:{uid}\nDTSTAMP:20260701T000000Z\n", f"DTSTART:{start}\n"]
    if end:
        lines.append(f"DTEND:{end}\n")
    lines.append(f"SUMMARY:{summary}\n")
    if extra:
        lines.append(extra)
    lines.append("END:VEVENT\n")
    return "".join(lines)


def _vcal(*vevents: str) -> str:
    return "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//test//EN\n" + "".join(vevents) + "END:VCALENDAR\n"


def _multistatus(*resources: tuple[str, str, str]) -> bytes:
    """resources = (href, etag, ical) tuples → a DAV:multistatus body."""
    blocks = "".join(
        f"<D:response><D:href>{href}</D:href><D:propstat><D:prop>"
        f"<D:getetag>{etag}</D:getetag>"
        f"<C:calendar-data>{ical}</C:calendar-data>"
        f"</D:prop></D:propstat></D:response>"
        for href, etag, ical in resources
    )
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<D:multistatus xmlns:D="DAV:" xmlns:C="urn:ietf:params:xml:ns:caldav">'
        f"{blocks}</D:multistatus>"
    ).encode()


# --- pull ----------------------------------------------------------------------


def test_pull_creates_writable_rows(settings, server) -> None:
    server.report_body = _multistatus(
        ("/cal/a.ics", '"e-a"', _vcal(_vevent("uid-a", "Dentist", "20261201T090000Z"))),
        ("/cal/b.ics", '"e-b"', _vcal(_vevent("uid-b", "Standup", "20261202T081500Z"))),
    )
    result = sync.pull_caldav(settings)
    assert result["added"] == 2

    events = {e.title: e for e in store.list_events(settings)}
    assert set(events) == {"Dentist", "Standup"}
    # CalDAV rows are read+WRITE — not refused by the write path.
    assert all(not sync.is_synced_id(e.id) for e in events.values())
    assert all(e.id.startswith("cdv") for e in events.values())
    assert events["Dentist"].caldav_href == "/cal/a.ics"
    assert events["Dentist"].caldav_etag == "e-a"


def test_repull_is_idempotent_by_etag(settings, server) -> None:
    server.report_body = _multistatus(
        ("/cal/a.ics", '"e-a"', _vcal(_vevent("uid-a", "Dentist", "20261201T090000Z"))),
    )
    sync.pull_caldav(settings)
    result = sync.pull_caldav(settings)
    assert result["added"] == 0 and result["updated"] == 0


def test_pull_removes_vanished_resource(settings, server) -> None:
    server.report_body = _multistatus(
        ("/cal/a.ics", '"e-a"', _vcal(_vevent("uid-a", "Dentist", "20261201T090000Z"))),
        ("/cal/b.ics", '"e-b"', _vcal(_vevent("uid-b", "Standup", "20261202T081500Z"))),
    )
    sync.pull_caldav(settings)
    server.report_body = _multistatus(
        ("/cal/b.ics", '"e-b"', _vcal(_vevent("uid-b", "Standup", "20261202T081500Z"))),
    )
    result = sync.pull_caldav(settings)
    assert result["removed"] == 1
    assert [e.title for e in store.list_events(settings)] == ["Standup"]


# --- push ----------------------------------------------------------------------


def test_push_create_puts_with_if_none_match(settings, server) -> None:
    summary = ops.apply_op(
        settings, {"op": "create", "title": "Lunch", "start": "2026-12-05T12:00:00+01:00"}
    )
    assert summary and "not yet synced" not in summary
    puts = server.of_method("PUT")
    assert len(puts) == 1
    assert puts[0]["headers"].get("If-None-Match") == "*"
    assert "SUMMARY:Lunch" in puts[0]["body"]
    assert puts[0]["url"].endswith(".ics")

    # The returned ETag landed on the row.
    event = store.find_event(settings, "Lunch")
    assert event is not None and event.caldav_etag == "etag-1"
    assert event.caldav_href.endswith(".ics")


def test_push_reschedule_uses_if_match(settings, server) -> None:
    ops.apply_op(settings, {"op": "create", "title": "Lunch", "start": "2026-12-05T12:00:00+01:00"})
    server.calls.clear()
    ops.apply_op(
        settings,
        {"op": "reschedule", "query": "Lunch", "start": "2026-12-05T13:00:00+01:00"},
    )
    puts = server.of_method("PUT")
    assert len(puts) == 1
    assert puts[0]["headers"].get("If-Match") == '"etag-1"'
    # 13:00 +01:00 == 12:00Z
    assert "DTSTART:20261205T120000Z" in puts[0]["body"]


def test_push_cancel_deletes(settings, server) -> None:
    ops.apply_op(settings, {"op": "create", "title": "Lunch", "start": "2026-12-05T12:00:00+01:00"})
    href = store.find_event(settings, "Lunch").caldav_href
    server.calls.clear()
    ops.apply_op(settings, {"op": "cancel", "query": "Lunch"})
    deletes = server.of_method("DELETE")
    assert len(deletes) == 1
    assert deletes[0]["url"] == href
    assert deletes[0]["headers"].get("If-Match") == '"etag-1"'


def test_writes_do_not_push_when_write_disabled(settings, server) -> None:
    settings.enable_caldav_write = False
    ops.apply_op(settings, {"op": "create", "title": "Lunch", "start": "2026-12-05T12:00:00+01:00"})
    assert server.of_method("PUT") == []


# --- conflict + outage ---------------------------------------------------------


def test_etag_conflict_lands_locally_and_queues(settings, server) -> None:
    ops.apply_op(settings, {"op": "create", "title": "Lunch", "start": "2026-12-05T12:00:00+01:00"})
    server.put_status = 412  # someone changed the remote first
    summary = ops.apply_op(
        settings,
        {"op": "reschedule", "query": "Lunch", "start": "2026-12-05T13:00:00+01:00"},
    )
    assert summary and "not yet synced" in summary
    # Local edit still landed.
    assert store.find_event(settings, "Lunch").start.startswith("2026-12-05T13:00")
    # And it is queued for reconcile.
    pending = outbox.pending(settings)
    assert len(pending) == 1 and pending[0]["op"] == outbox.OP_PUT


def test_outage_queues_then_reconcile_drains(settings, server) -> None:
    server.raise_exc = caldav.CalDavError("connection refused")
    summary = ops.apply_op(
        settings, {"op": "create", "title": "Lunch", "start": "2026-12-05T12:00:00+01:00"}
    )
    assert summary and "not yet synced" in summary
    assert store.find_event(settings, "Lunch") is not None  # local write survived
    assert len(outbox.pending(settings)) == 1

    server.raise_exc = None  # server comes back
    server.calls.clear()
    result = sync.reconcile_caldav(settings)
    assert result["reconciled"] == 1
    assert outbox.pending(settings) == []
    assert len(server.of_method("PUT")) == 1


def test_reconcile_drops_persistent_conflict_toward_remote(settings, server) -> None:
    server.raise_exc = caldav.CalDavError("down")
    ops.apply_op(settings, {"op": "create", "title": "Lunch", "start": "2026-12-05T12:00:00+01:00"})
    server.raise_exc = None
    server.put_status = 412  # remote moved while we were offline
    result = sync.reconcile_caldav(settings)
    assert result["dropped"] == 1
    assert outbox.pending(settings) == []  # deferred to the server, not retried forever


def test_reconcile_isolates_a_poison_row_from_the_rest(settings, server, monkeypatch) -> None:
    from assistant.calendar import remote

    # Two edits queued during an outage.
    server.raise_exc = caldav.CalDavError("down")
    ops.apply_op(settings, {"op": "create", "title": "Alpha", "start": "2026-12-05T09:00:00+01:00"})
    ops.apply_op(settings, {"op": "create", "title": "Beta", "start": "2026-12-06T09:00:00+01:00"})
    server.raise_exc = None
    assert len(outbox.pending(settings)) == 2

    # The first row hits an unexpected error (not a RemoteError) — a raw failure
    # like a corrupt event or a DB hiccup. It must not abort the drain and strand
    # the healthy row behind it.
    real_upsert = remote.upsert
    seen: list[str] = []

    def flaky_upsert(s, event):
        seen.append(event.title)
        if len(seen) == 1:
            raise RuntimeError("boom building the request")
        return real_upsert(s, event)

    monkeypatch.setattr(remote, "upsert", flaky_upsert)

    result = sync.reconcile_caldav(settings)  # must not raise
    assert len(seen) == 2  # the second row was still attempted
    assert result["reconciled"] == 1 and result["still_pending"] == 1
    assert len(outbox.pending(settings)) == 1  # only the poison row stays queued


# --- undo ----------------------------------------------------------------------


def test_undo_create_deletes_remote(settings, server) -> None:
    ops.apply_op(
        settings,
        {"op": "create", "title": "Lunch", "start": "2026-12-05T12:00:00+01:00"},
        thread_id="t1",
        batch_id="b1",
    )
    href = store.find_event(settings, "Lunch").caldav_href
    server.calls.clear()

    undo.undo_latest(settings, "t1", window_minutes=60)
    assert store.find_event(settings, "Lunch") is None  # local row gone
    deletes = server.of_method("DELETE")
    assert len(deletes) == 1 and deletes[0]["url"] == href


# --- guard split ---------------------------------------------------------------


def test_ics_rows_still_refuse_writes(settings, server) -> None:
    store.restore_event(
        settings,
        store.Event(id="ics_abc123", title="External", start="2026-12-09T09:00:00+01:00"),
    )
    assert ops.apply_op(settings, {"op": "cancel", "query": "External"}) is None
    assert store.find_event(settings, "External") is not None
    assert server.of_method("DELETE") == []


def test_caldav_rows_accept_writes(settings, server) -> None:
    server.report_body = _multistatus(
        ("/cal/a.ics", '"e-a"', _vcal(_vevent("uid-a", "Dentist", "20261201T090000Z"))),
    )
    sync.pull_caldav(settings)
    server.calls.clear()
    summary = ops.apply_op(settings, {"op": "cancel", "query": "Dentist"})
    assert summary is not None
    assert len(server.of_method("DELETE")) == 1


# --- recurring round-trip ------------------------------------------------------


def test_recurring_round_trip(settings) -> None:
    event = store.Event(
        id="abc123",
        title="Weekly sync",
        start="2026-12-07T09:00:00+01:00",
        rrule="FREQ=WEEKLY;BYDAY=MO",
        exdates=json.dumps(["2026-12-14T09:00:00+01:00"]),
        overrides=json.dumps({"2026-12-21T09:00:00+01:00": {"title": "Weekly sync (moved)"}}),
    )
    ical = caldav.event_to_ical(settings, event)
    parsed = sync.parse_vevents(ical, settings)
    assert len(parsed) == 1
    master = next(iter(parsed.values()))
    assert "FREQ=WEEKLY" in master.rrule
    assert "2026-12-14" in store.load_exdates(master)[0]
    overrides = store.load_overrides(master)
    assert any(v.get("title") == "Weekly sync (moved)" for v in overrides.values())


# --- google oauth --------------------------------------------------------------


def test_oauth_auth_header_uses_bearer(settings, monkeypatch) -> None:
    settings.caldav_auth = "oauth"
    monkeypatch.setattr(caldav_oauth, "access_token", lambda s: "tok-123")
    assert caldav._auth_header(settings) == {"Authorization": "Bearer tok-123"}


def test_oauth_refresh_exchanges_and_caches_0600(tmp_path, monkeypatch) -> None:
    import os
    import stat

    settings = Settings(
        memory_dir=str(tmp_path / "memory"),
        caldav_auth="oauth",
        caldav_oauth_client_id="cid",
        caldav_oauth_client_secret="secret",
        caldav_oauth_refresh_token="refresh",
    )

    calls: list = []

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"access_token": "AT", "expires_in": 3600}).encode()

    def fake_urlopen(request, timeout=None):
        calls.append(request)
        return FakeResp()

    monkeypatch.setattr(caldav_oauth.urllib.request, "urlopen", fake_urlopen)

    assert caldav_oauth.access_token(settings) == "AT"
    # A second call is served from the 0600 cache — no extra network round-trip.
    assert caldav_oauth.access_token(settings) == "AT"
    assert len(calls) == 1
    mode = stat.S_IMODE(os.stat(settings.caldav_token_path).st_mode)
    assert mode == 0o600


def test_oauth_missing_credentials_raises_auth_error(tmp_path) -> None:
    settings = Settings(memory_dir=str(tmp_path / "memory"), caldav_auth="oauth")
    with pytest.raises(caldav.CalDavAuthError):
        caldav_oauth.access_token(settings)


def test_own_write_pulls_back_without_duplicating(settings, server) -> None:
    # A locally-created event pushed to CalDAV, then pulled back by UID, must upsert
    # the same row (id stripped from '<id>@wakiru') — not create a 'cdv…' duplicate.
    ops.apply_op(settings, {"op": "create", "title": "Lunch", "start": "2026-12-05T12:00:00+01:00"})
    local = store.find_event(settings, "Lunch")
    server.report_body = _multistatus(
        (local.caldav_href, '"etag-1"',
         _vcal(_vevent(f"{local.id}@wakiru", "Lunch", "20261205T110000Z"))),
    )
    sync.pull_caldav(settings)
    lunches = [e for e in store.list_events(settings) if e.title == "Lunch"]
    assert len(lunches) == 1 and lunches[0].id == local.id
