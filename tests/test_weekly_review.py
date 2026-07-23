"""Weekly review tests — the day/time gate, the once-per-week ledger, sections.

No network and no LLM: delivery is monkeypatched and composition is stubbed to
its fallback (compose_push's own behavior lives in test_compose.py). The
ledger and the subsystem stores run for real against tmp SQLite files.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from assistant import weekly_review
from assistant.calendar.context import resolve_tz
from assistant.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        enable_weekly_review=True,
        enable_email=False,
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
        weekly_review,
        "deliver_reminder",
        lambda s, reminder, **kw: sent.append(reminder) or True,
    )
    return sent


def _freeze_clock(monkeypatch, settings: Settings, day: int, hhmm: str) -> datetime:
    """Freeze now() to the given ISO weekday (Mon=0) of the week of 2026-07-06."""
    hour, minute = map(int, hhmm.split(":"))
    # 2026-07-06 is a Monday.
    frozen = datetime(2026, 7, 6 + day, hour, minute, tzinfo=resolve_tz(settings))
    monkeypatch.setattr(weekly_review, "now", lambda s: frozen)
    return frozen


def test_not_due_before_review_day(settings, delivered, monkeypatch) -> None:
    _freeze_clock(monkeypatch, settings, day=4, hhmm="18:00")  # Friday evening
    result = weekly_review.run_weekly_review(settings)
    assert result == {"sent": False, "reason": "not due yet"}
    assert delivered == []


def test_not_due_before_review_time(settings, delivered, monkeypatch) -> None:
    _freeze_clock(monkeypatch, settings, day=6, hhmm="12:00")  # Sunday noon
    result = weekly_review.run_weekly_review(settings)
    assert result == {"sent": False, "reason": "not due yet"}


def test_fires_once_per_week(settings, delivered, monkeypatch) -> None:
    _freeze_clock(monkeypatch, settings, day=6, hhmm="18:00")  # Sunday evening
    first = weekly_review.run_weekly_review(settings)
    assert first["sent"] and first["delivered"]
    assert first["week"] == "2026-W28"
    assert len(delivered) == 1
    assert delivered[0]["title"] == "Weekly review"
    assert "Calendar next 7 days" in delivered[0]["message"]

    second = weekly_review.run_weekly_review(settings)
    assert second == {"sent": False, "reason": "already sent this week"}
    assert len(delivered) == 1


def test_next_week_fires_again(settings, delivered, monkeypatch) -> None:
    _freeze_clock(monkeypatch, settings, day=6, hhmm="18:00")
    assert weekly_review.run_weekly_review(settings)["sent"]
    _freeze_clock(monkeypatch, settings, day=13, hhmm="18:00")  # next Sunday
    result = weekly_review.run_weekly_review(settings)
    assert result["sent"] and result["week"] == "2026-W29"
    assert len(delivered) == 2


def test_disabled_is_noop_unless_forced(settings, delivered, monkeypatch) -> None:
    settings.enable_weekly_review = False
    _freeze_clock(monkeypatch, settings, day=6, hhmm="18:00")
    result = weekly_review.run_weekly_review(settings)
    assert result == {"sent": False, "reason": "disabled"}
    assert weekly_review.run_weekly_review(settings, force=True)["sent"]
    assert len(delivered) == 1


def test_force_skips_gate_but_claims_the_week(settings, delivered, monkeypatch) -> None:
    _freeze_clock(monkeypatch, settings, day=1, hhmm="09:00")  # Tuesday morning
    assert weekly_review.run_weekly_review(settings, force=True)["sent"]
    # The scheduled firing later the same week must not duplicate it.
    _freeze_clock(monkeypatch, settings, day=6, hhmm="18:00")
    result = weekly_review.run_weekly_review(settings)
    assert result["reason"] == "already sent this week"
    assert len(delivered) == 1


def test_custom_review_day(settings, delivered, monkeypatch) -> None:
    settings.weekly_review_day = "friday"
    _freeze_clock(monkeypatch, settings, day=4, hhmm="18:00")  # Friday evening
    assert weekly_review.run_weekly_review(settings)["sent"]


def test_completed_and_due_tasks_ride_in(settings, delivered, monkeypatch) -> None:
    from assistant.tasks import store as tasks_store

    frozen = _freeze_clock(monkeypatch, settings, day=6, hhmm="18:00")
    done = tasks_store.create_task(settings, title="Ship the report")
    monkeypatch.setattr(
        tasks_store, "_stamp_now", lambda s: frozen.isoformat(timespec="seconds")
    )
    tasks_store.complete_task(settings, done.id)
    tasks_store.create_task(
        settings,
        title="Prepare slides",
        due=(frozen + timedelta(days=2)).isoformat(timespec="seconds"),
    )
    assert weekly_review.run_weekly_review(settings)["sent"]
    message = delivered[0]["message"]
    assert "Completed last week (1)" in message
    assert "Ship the report" in message
    assert "Tasks due this week" in message
    assert "Prepare slides" in message


def test_expenses_and_subscriptions_ride_in(settings, delivered, monkeypatch) -> None:
    settings.enable_expenses = True
    settings.enable_subscriptions = True
    from assistant.expenses import store as expenses_store
    from assistant.subscriptions import store as subs_store

    frozen = _freeze_clock(monkeypatch, settings, day=6, hhmm="18:00")
    today = frozen.date()
    expenses_store.log_entry(
        settings, amount=250, currency="kr", category="food",
        on=(today - timedelta(days=2)).isoformat(),
    )
    expenses_store.log_entry(  # outside the window — must not count
        settings, amount=999, currency="kr", category="food",
        on=(today - timedelta(days=20)).isoformat(),
    )
    expenses_store.set_budget(settings, "food", "2000", currency="kr")
    subs_store.create_subscription(
        settings, name="Spotify", amount=129, currency="kr", cadence="monthly",
        renews_on=(today + timedelta(days=3)).isoformat(),
    )
    assert weekly_review.run_weekly_review(settings)["sent"]
    message = delivered[0]["message"]
    assert "Spending last 7 days" in message
    assert "250 kr" in message and "top: food 250" in message
    assert "999" not in message
    assert "Budgets (month to date):" in message
    assert "food: 250 of 2000 kr (12%)" in message
    assert "Renewals this week" in message
    assert "Spotify" in message


def test_compose_failure_falls_back_to_the_verbatim_digest(
    settings, delivered, monkeypatch
) -> None:
    # The autouse fixture already models a failed composition (fallback text).
    _freeze_clock(monkeypatch, settings, day=6, hhmm="18:00")
    assert weekly_review.run_weekly_review(settings)["sent"]
    assert "Weekly review" in delivered[0]["message"]


def test_review_loops_into_authorized_threads(settings, delivered, monkeypatch) -> None:
    settings.telegram_bot_token = "tok"
    settings.telegram_allowed_chat_ids = [42]
    recorded: list[tuple[str, str]] = []

    class _Agent:
        def update_state(self, config, update, as_node=None) -> None:
            recorded.append(
                (config["configurable"]["thread_id"], update["messages"][0].content)
            )

    _freeze_clock(monkeypatch, settings, day=6, hhmm="18:00")
    assert weekly_review.run_weekly_review(settings, agent=_Agent())["sent"]
    assert {t for t, _ in recorded} == {"telegram:42"}
    assert all(text.startswith("Weekly review:") for _, text in recorded)


def test_malformed_day_and_time_default(settings) -> None:
    settings.weekly_review_day = "someday"
    settings.weekly_review_time = "not-a-time"
    assert weekly_review._due_day(settings) == 6
    assert weekly_review._due_time(settings).hour == 17
