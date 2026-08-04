"""Daily briefing tests — the due-time gate, the once-per-day ledger, delivery.

No network and no LLM: delivery is monkeypatched, composition is stubbed to
its fallback (compose_push's own behavior lives in test_compose.py), and the
heartbeat is off in the fixture so the local path runs. The ledger runs for
real against a tmp SQLite file.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from assistant import briefing, weather
from assistant.calendar import context as calendar_context
from assistant.calendar import store as calendar_store
from assistant.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    # Heartbeat off by default here: these exercise the template composition
    # path. The delegation tests flip enable_heartbeat=True explicitly.
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        enable_briefing=True,
        enable_email=False,
        enable_heartbeat=False,
    )


@pytest.fixture(autouse=True)
def _compose_fallback(monkeypatch) -> None:
    """Stand-in composer: behaves like a failed model (returns the fallback)."""
    monkeypatch.setattr(
        "assistant.compose.compose_push", lambda s, **kw: kw["fallback"]
    )


@pytest.fixture
def delivered(monkeypatch) -> list[dict]:
    sent: list[dict] = []
    monkeypatch.setattr(
        briefing, "deliver_reminder", lambda s, reminder, **kw: sent.append(reminder) or True
    )
    return sent


def _freeze_clock(monkeypatch, settings: Settings, hhmm: str) -> None:
    hour, minute = map(int, hhmm.split(":"))
    from assistant.calendar.context import resolve_tz

    frozen = datetime(2026, 7, 11, hour, minute, tzinfo=resolve_tz(settings))
    monkeypatch.setattr(briefing, "now", lambda s: frozen)


def test_not_due_before_briefing_time(settings, delivered, monkeypatch) -> None:
    _freeze_clock(monkeypatch, settings, "06:00")
    assert briefing.run_briefing(settings) == {"sent": False, "reason": "not due yet"}
    assert delivered == []


def test_due_gate_and_ledger_date_use_trip_local_time(settings, delivered, monkeypatch) -> None:
    from assistant.calendar import context as calendar_context
    from assistant.trips import store as trips_store

    traveling = settings.model_copy(
        update={"timezone": "Europe/Oslo", "enable_trips": True, "briefing_time": "07:30"}
    )
    trips_store.create_trip(
        traveling, "New York", start="2026-03-08", end="2026-03-08",
        timezone="America/New_York",
    )
    instant = datetime(2026, 3, 8, 10, 0, tzinfo=UTC)  # 06:00 EDT, 11:00 home
    real_datetime = datetime

    class FrozenDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return instant.astimezone(tz) if tz is not None else instant.replace(tzinfo=None)

    monkeypatch.setattr(calendar_context, "datetime", FrozenDateTime)
    assert briefing.run_briefing(traveling) == {"sent": False, "reason": "not due yet"}

    instant = datetime(2026, 3, 8, 12, 0, tzinfo=UTC)  # 08:00 EDT
    result = briefing.run_briefing(traveling)
    assert result["sent"] and result["date"] == "2026-03-08"


def test_due_gate_reverts_to_home_time_after_trip_ends(settings, delivered, monkeypatch) -> None:
    from assistant.calendar import context as calendar_context
    from assistant.trips import store as trips_store

    traveling = settings.model_copy(
        update={"timezone": "Europe/Oslo", "enable_trips": True, "briefing_time": "07:00"}
    )
    trips_store.create_trip(
        traveling, "New York", start="2026-03-08", end="2026-03-08",
        timezone="America/New_York",
    )
    # The trip ended at 04:00 UTC. At 06:30 it is 07:30 at home, so the home
    # briefing is due even though New York's clock would only read 02:30.
    instant = datetime(2026, 3, 9, 6, 30, tzinfo=UTC)
    real_datetime = datetime

    class FrozenDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return instant.astimezone(tz) if tz is not None else instant.replace(tzinfo=None)

    monkeypatch.setattr(calendar_context, "datetime", FrozenDateTime)
    result = briefing.run_briefing(traveling)

    assert result["sent"] and result["date"] == "2026-03-09"


def test_fires_once_after_due_time(settings, delivered, monkeypatch) -> None:
    _freeze_clock(monkeypatch, settings, "08:00")
    first = briefing.run_briefing(settings)
    assert first["sent"] and first["delivered"]
    assert len(delivered) == 1
    assert delivered[0]["title"] == "Daily briefing"
    assert "Upcoming events" in delivered[0]["message"]

    second = briefing.run_briefing(settings)
    assert second == {"sent": False, "reason": "already sent today"}
    assert len(delivered) == 1


def test_disabled_is_noop_unless_forced(settings, delivered, monkeypatch) -> None:
    settings.enable_briefing = False
    _freeze_clock(monkeypatch, settings, "08:00")
    assert briefing.run_briefing(settings) == {"sent": False, "reason": "disabled"}
    assert briefing.run_briefing(settings, force=True)["sent"]
    assert len(delivered) == 1


def test_force_skips_time_gate_but_claims_the_day(settings, delivered, monkeypatch) -> None:
    _freeze_clock(monkeypatch, settings, "06:00")
    assert briefing.run_briefing(settings, force=True)["sent"]
    # The scheduled firing later the same day must not duplicate it.
    _freeze_clock(monkeypatch, settings, "08:00")
    assert briefing.run_briefing(settings)["reason"] == "already sent today"
    assert len(delivered) == 1


def test_briefing_is_composed_by_the_model(settings, delivered, monkeypatch) -> None:
    composed: dict = {}

    def fake_compose(s, **kwargs):
        composed.update(kwargs)
        return "God morgen! Rolig dag i dag."

    monkeypatch.setattr("assistant.compose.compose_push", fake_compose)
    _freeze_clock(monkeypatch, settings, "08:00")
    assert briefing.run_briefing(settings)["sent"]
    assert delivered[0]["message"] == "God morgen! Rolig dag i dag."
    # The assembled digest is both the source material and the fallback.
    assert "Upcoming events" in composed["facts"]
    assert composed["fallback"] == composed["facts"]


def test_compose_failure_falls_back_to_the_verbatim_digest(settings, delivered, monkeypatch) -> None:
    # The autouse fixture already models a failed composition (fallback text);
    # the digest must go out verbatim.
    _freeze_clock(monkeypatch, settings, "08:00")
    assert briefing.run_briefing(settings)["sent"]
    assert "Upcoming events" in delivered[0]["message"]


def test_briefing_includes_pending_invitation_verbatim(settings) -> None:
    invite = calendar_store.Event(
        id="briefing-invite", title="Planning review",
        start=(calendar_context.now(settings) + timedelta(days=1)).isoformat(),
        organizer=calendar_store.dump_organizer({"email": "owner@example.com"}),
        attendees=calendar_store.dump_attendees([
            {"email": "me@example.com", "status": "needsaction", "self": True},
        ]),
        caldav_href="/invite.ics",
    )
    calendar_store.restore_event(settings, invite)
    digest = briefing.build_briefing(settings)
    assert "## Pending invitations" in digest
    assert "briefing-invite" in digest
    assert "read-only/non-actionable" in digest


def test_heartbeat_enabled_delegates_composition(settings, delivered, monkeypatch) -> None:
    settings.enable_heartbeat = True
    calls: dict = {}

    def fake_run_heartbeat(s, agent=None, force=False, force_briefing=False):
        calls.update(force=force, force_briefing=force_briefing)
        return {"sent": True, "delivered": True, "message": "God morgen!"}

    monkeypatch.setattr("assistant.heartbeat.run_heartbeat", fake_run_heartbeat)
    _freeze_clock(monkeypatch, settings, "08:00")

    result = briefing.run_briefing(settings)
    assert result["message"] == "God morgen!"
    assert calls == {"force": False, "force_briefing": False}
    assert delivered == []  # delivery happens inside the heartbeat, not here


def test_heartbeat_delegation_still_respects_due_time(settings, monkeypatch) -> None:
    settings.enable_heartbeat = True
    monkeypatch.setattr(
        "assistant.heartbeat.run_heartbeat",
        lambda *a, **k: pytest.fail("not due yet — the heartbeat must not run"),
    )
    _freeze_clock(monkeypatch, settings, "06:00")
    assert briefing.run_briefing(settings) == {"sent": False, "reason": "not due yet"}


def test_heartbeat_forced_briefing_passes_force_through(settings, monkeypatch) -> None:
    settings.enable_heartbeat = True
    calls: dict = {}

    def fake_run_heartbeat(s, agent=None, force=False, force_briefing=False):
        calls.update(force=force, force_briefing=force_briefing)
        return {"sent": True}

    monkeypatch.setattr("assistant.heartbeat.run_heartbeat", fake_run_heartbeat)
    _freeze_clock(monkeypatch, settings, "06:00")  # before due time
    assert briefing.run_briefing(settings, force=True)["sent"]
    assert calls == {"force": True, "force_briefing": True}


class _RecordingAgent:
    def __init__(self) -> None:
        self.recorded: list[tuple[str, str]] = []

    def update_state(self, config, update, as_node=None) -> None:
        thread = config["configurable"]["thread_id"]
        self.recorded.append((thread, update["messages"][0].content))


def test_briefing_loops_into_authorized_threads(settings, delivered, monkeypatch) -> None:
    settings.telegram_bot_token = "tok"
    settings.telegram_allowed_chat_ids = [42, 43]
    agent = _RecordingAgent()
    _freeze_clock(monkeypatch, settings, "08:00")
    assert briefing.run_briefing(settings, agent=agent)["sent"]
    threads = {t for t, _ in agent.recorded}
    assert threads == {"telegram:42", "telegram:43"}
    assert all(text.startswith("Daily briefing:") for _, text in agent.recorded)


def test_briefing_loop_in_disabled_records_nothing(settings, delivered, monkeypatch) -> None:
    settings.telegram_bot_token = "tok"
    settings.telegram_allowed_chat_ids = [42]
    settings.enable_proactive_loop_in = False
    agent = _RecordingAgent()
    _freeze_clock(monkeypatch, settings, "08:00")
    assert briefing.run_briefing(settings, agent=agent)["sent"]
    assert agent.recorded == []


def test_malformed_briefing_time_defaults(settings) -> None:
    settings.briefing_time = "not-a-time"
    assert briefing._due_time(settings).hour == 7


def test_briefing_weather_tracks_trip_and_home(settings, monkeypatch) -> None:
    from assistant.trips import store as trips_store

    traveling = settings.model_copy(
        update={
            "enable_weather": True,
            "enable_trips": True,
            "weather_latitude": 59.9,
            "weather_longitude": 10.7,
            "weather_location_name": "Oslo",
            "weather_refresh_minutes": 60,
        }
    )
    trips_store.create_trip(
        traveling, "Tokyo", start="2026-08-04", end="2026-08-04", timezone="UTC"
    )
    payload = {
        "current": {"temperature_2m": 20, "weather_code": 0},
        "daily": {
            "temperature_2m_max": [24],
            "temperature_2m_min": [16],
            "weather_code": [0],
        },
    }
    instant = datetime(2026, 8, 4, 12, tzinfo=UTC)
    monkeypatch.setattr(weather, "now", lambda s: instant)
    monkeypatch.setattr(weather, "_geocode", lambda s, name: (35.68, 139.65))
    monkeypatch.setattr(weather, "_fetch", lambda *args, **kwargs: payload)
    weather.refresh(traveling)
    assert "Location: Tokyo" in briefing.build_briefing(traveling)

    instant = datetime(2026, 8, 5, 0, tzinfo=UTC)
    assert "## Weather" not in briefing.build_briefing(traveling)
    weather.refresh(traveling)
    assert "Location: Oslo" in briefing.build_briefing(traveling)
