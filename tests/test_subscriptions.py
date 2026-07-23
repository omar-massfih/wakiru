"""Subscription subsystem tests — store, next-renewal math, cost rollup, the
tool paths, and renewal reminders (exactly-once via the fired ledger).
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from assistant.calendar import context as cal_context
from assistant.config import Settings
from assistant.subscriptions import store
from assistant.subscriptions.context import monthly_totals
from assistant.subscriptions.reminders import run_subscription_reminders
from assistant.subscriptions.store import monthly_amount, next_renewal
from assistant.tools import ToolContext, tool_map


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        memory_dir=str(tmp_path / "memory"),
        timezone="Europe/Oslo",
        enable_subscriptions=True,
        subscriptions_renewal_lead_days=3,
    )


# --- store + math ----------------------------------------------------------- #


def test_create_and_list(settings) -> None:
    store.create_subscription(settings, "Spotify", amount="129", currency="NOK", cadence="monthly")
    subs = store.list_subscriptions(settings)
    assert [s.name for s in subs] == ["Spotify"]
    assert subs[0].amount == 129.0
    assert subs[0].cadence == "monthly"


def test_cadence_aliases_normalize(settings) -> None:
    s = store.create_subscription(settings, "Insurance", cadence="annually")
    assert s.cadence == "yearly"


def test_next_renewal_rolls_past_anchor_forward() -> None:
    today = date(2026, 7, 23)
    sub = store.Subscription(id="x", name="s", cadence="monthly", renews_on="2026-01-10")
    nxt = next_renewal(sub, today)
    assert nxt == date(2026, 8, 10)  # rolled forward from January


def test_next_renewal_future_anchor_unchanged() -> None:
    today = date(2026, 7, 23)
    sub = store.Subscription(id="x", name="s", cadence="yearly", renews_on="2026-12-01")
    assert next_renewal(sub, today) == date(2026, 12, 1)


def test_monthly_amount_normalizes_cadence() -> None:
    yearly = store.Subscription(id="x", name="s", amount=120.0, cadence="yearly")
    assert monthly_amount(yearly) == 10.0
    quarterly = store.Subscription(id="y", name="s", amount=30.0, cadence="quarterly")
    assert monthly_amount(quarterly) == 10.0


def test_monthly_totals_sum_by_currency(settings) -> None:
    store.create_subscription(settings, "A", amount="100", currency="NOK", cadence="monthly")
    store.create_subscription(settings, "B", amount="120", currency="NOK", cadence="yearly")
    totals = monthly_totals(store.list_subscriptions(settings))
    assert totals["NOK"] == pytest.approx(110.0)  # 100 + 120/12


# --- tools ------------------------------------------------------------------ #


def _run(settings, tool, **args) -> str:
    return tool_map(settings)[tool].run(ToolContext(settings=settings), **args)


def test_add_dedupe_and_list_tools(settings) -> None:
    assert "Tracking subscription" in _run(settings, "add_subscription", name="Netflix", amount="99", currency="NOK", cadence="monthly")
    assert "already exists" in _run(settings, "add_subscription", name="Netflix")
    listing = _run(settings, "list_subscriptions")
    assert "Netflix" in listing and "monthly spend" in listing.lower()


def test_update_and_remove_tools(settings) -> None:
    _run(settings, "add_subscription", name="Gym", amount="400", cadence="monthly")
    assert "Updated" in _run(settings, "update_subscription", query="Gym", amount="450")
    assert store.find_subscription(settings, "Gym").amount == 450.0
    assert "Stopped tracking" in _run(settings, "remove_subscription", query="Gym")
    assert store.list_subscriptions(settings) == []


# --- renewal reminders ------------------------------------------------------ #


@pytest.fixture
def _delivery(monkeypatch):
    sent: list[dict] = []
    monkeypatch.setattr("assistant.compose.compose_push", lambda s, **kw: kw["fallback"])
    monkeypatch.setattr(
        "assistant.subscriptions.reminders.deliver_reminder",
        lambda s, r: bool(sent.append(r)) or True,
    )
    monkeypatch.setattr("assistant.proactive.record_push", lambda *a, **k: None)
    return sent


def test_renewal_reminder_fires_once(settings, _delivery) -> None:
    soon = (cal_context.now(settings).date() + timedelta(days=2)).isoformat()
    store.create_subscription(settings, "Spotify", amount="129", currency="NOK", cadence="monthly", renews_on=soon)
    sent = run_subscription_reminders(settings)
    assert len(sent) == 1
    assert "Spotify" in _delivery[0]["message"]
    assert run_subscription_reminders(settings) == []  # exactly-once


def test_renewal_outside_lead_window(settings, _delivery) -> None:
    far = (cal_context.now(settings).date() + timedelta(days=20)).isoformat()
    store.create_subscription(settings, "Gym", cadence="monthly", renews_on=far)
    assert run_subscription_reminders(settings) == []


def test_renewal_reminders_respect_switches(settings, _delivery) -> None:
    soon = (cal_context.now(settings).date() + timedelta(days=1)).isoformat()
    store.create_subscription(settings, "X", cadence="monthly", renews_on=soon)
    assert run_subscription_reminders(settings.model_copy(update={"enable_reminders": False})) == []
    assert run_subscription_reminders(settings.model_copy(update={"enable_subscriptions": False})) == []
