"""Weekly review — one look-back + week-ahead digest per week.

The daily briefing answers "what matters today"; this answers the planning
question a Sunday evening raises: what happened last week (tasks completed,
habit streaks, spending) and what is coming in the next seven days (calendar,
due tasks, trips, birthdays, subscription renewals). Same architecture as
:mod:`assistant.briefing`: no data model of its own — the subsystem read paths
are assembled into a digest, the model composes the push in Wakiru's voice
(:func:`assistant.compose.compose_push`, digest as fallback), delivery goes
through :func:`assistant.notify.deliver_reminder`, and a fired ledger keyed on
the ISO week makes the ticker and the manual ``POST /weekly-review/run``
exactly-once together.

The review becomes *due* at ``weekly_review_day`` + ``weekly_review_time``
(local wall clock) and fires on the first call at or after that moment in the
same ISO week — a server asleep Sunday evening still reviews when it wakes
Sunday night; a whole missed week is skipped, not double-sent. Both windows
are rolling seven-day spans around "now", not calendar-week aligned, so the
digest is equally useful when it fires late.
"""

from __future__ import annotations

import logging
from datetime import time as dtime
from datetime import timedelta

from . import fired_ledger
from .calendar.context import busy_events, now, render_events
from .calendar.store import parse_dt
from .config import Settings, get_settings
from .notify import deliver_reminder

logger = logging.getLogger(__name__)

_LEDGER = fired_ledger.FiredLedgerSpec(
    table="weekly_reviews_fired",
    columns=(("iso_week", "TEXT"),),
    db_path=lambda settings: settings.briefing_db_path,
)

_WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def _due_day(settings: Settings) -> int:
    """Parse ``weekly_review_day`` to a weekday index (Mon=0); fallback Sunday."""
    prefix = settings.weekly_review_day.strip().lower()[:3]
    if prefix in _WEEKDAYS:
        return _WEEKDAYS.index(prefix)
    logger.warning(
        "invalid WEEKLY_REVIEW_DAY %r; using sunday", settings.weekly_review_day
    )
    return 6


def _due_time(settings: Settings) -> dtime:
    """Parse ``weekly_review_time`` (HH:MM); a malformed value falls back to 17:00."""
    try:
        hour, _, minute = settings.weekly_review_time.partition(":")
        return dtime(int(hour), int(minute))
    except ValueError:
        logger.warning(
            "invalid WEEKLY_REVIEW_TIME %r; using 17:00", settings.weekly_review_time
        )
        return dtime(17, 0)


def _tasks_sections(settings: Settings, current) -> list[str]:
    """Due-this-week (overdue included) and completed-last-week task blocks."""
    from .tasks import store as tasks_store
    from .tasks.context import render_tasks

    week_ago = current - timedelta(days=7)
    week_ahead = current + timedelta(days=7)
    everything = tasks_store.list_tasks(settings, include_done=True)
    due = [
        t
        for t in everything
        if not t.done
        and t.due
        and (when := parse_dt(t.due)) is not None
        and when <= week_ahead
    ]
    finished = [
        t
        for t in everything
        if t.done
        and t.done_at
        and (when := parse_dt(t.done_at)) is not None
        and when >= week_ago
    ]
    parts = []
    if due:
        parts.append("## Tasks due this week\n" + render_tasks(settings, due))
    if finished:
        lines = "\n".join(f"- {t.title}" for t in finished)
        parts.append(f"## Completed last week ({len(finished)})\n{lines}")
    return parts


def _trips_section(settings: Settings, today) -> str:
    """Active trips and departures within the next seven days."""
    from .trips import store as trips_store

    soon = today + timedelta(days=7)
    lines = []
    for trip in trips_store.list_trips(settings, today=today):
        start = trips_store.parse_date(trip.start)
        if start is not None and start > soon:
            continue
        name = trip.name or trip.destination
        span = " to ".join(part for part in (trip.start, trip.end) if part)
        lines.append(f"- {name}" + (f": {span}" if span else ""))
    return "## Trips this week\n" + "\n".join(lines) if lines else ""


def _subscriptions_section(settings: Settings, today) -> str:
    """Subscriptions renewing within the next seven days."""
    from .subscriptions import store as subs_store
    from .subscriptions.context import render_subscription

    renewing = [
        sub
        for sub in subs_store.list_subscriptions(settings)
        if (nxt := subs_store.next_renewal(sub, today)) is not None
        and (nxt - today).days <= 7
    ]
    if not renewing:
        return ""
    lines = "\n".join(render_subscription(s, today, with_id=False) for s in renewing)
    return "## Renewals this week\n" + lines


def _habits_section(settings: Settings, today) -> str:
    """The habit overview (streaks ride along), only when something is tracked."""
    from .habits import store as habits_store
    from .habits.context import overview

    if not habits_store.habit_names(settings):
        return ""
    return "## Habits\n" + overview(settings, today)


def _expenses_section(settings: Settings, today) -> str:
    """Last seven days of spending, totalled per currency with top categories."""
    from .expenses import store as expenses_store
    from .expenses.context import _num, totals_by_category, totals_by_currency

    week_ago = today - timedelta(days=7)
    months = {today.isoformat()[:7], week_ago.isoformat()[:7]}
    entries = [
        e
        for month in sorted(months)
        for e in expenses_store.list_entries(settings, month=month)
        if week_ago.isoformat() <= e.spent_on <= today.isoformat()
    ]
    if not entries:
        return ""
    by_category = totals_by_category(entries)
    lines = []
    for currency, total in sorted(totals_by_currency(entries).items()):
        top = sorted(
            by_category.get(currency, {}).items(), key=lambda kv: -kv[1]
        )[:3]
        detail = ", ".join(f"{cat} {_num(amount)}" for cat, amount in top)
        lines.append(
            f"- {_num(total)} {currency}" + (f" (top: {detail})" if detail else "")
        )
    return "## Spending last 7 days\n" + "\n".join(lines)


def _people_section(settings: Settings) -> str:
    """Birthdays and overdue contacts — the same block the briefing shows."""
    from .people.context import briefing_people

    return briefing_people(settings)


def build_weekly_review(settings: Settings) -> str:
    """Assemble the digest text from the subsystem read paths (no LLM)."""
    current = now(settings)
    today = current.date()
    stamp = current.strftime("%A, %d %B %Y").strip()
    parts = [f"## Weekly review\nIt is {stamp}."]
    try:
        events = busy_events(settings, current, current + timedelta(days=7))
        parts.append("## Calendar next 7 days\n" + render_events(settings, events))
    except Exception:
        logger.exception("weekly review: calendar section failed; skipping it")
    for flag, section in (
        (settings.enable_tasks, lambda: _tasks_sections(settings, current)),
        (settings.enable_trips, lambda: [_trips_section(settings, today)]),
        (settings.enable_people, lambda: [_people_section(settings)]),
        (settings.enable_subscriptions, lambda: [_subscriptions_section(settings, today)]),
        (settings.enable_habits, lambda: [_habits_section(settings, today)]),
        (settings.enable_expenses, lambda: [_expenses_section(settings, today)]),
    ):
        if not flag:
            continue
        try:
            parts.extend(section())
        except Exception:
            logger.exception("weekly review: a section failed; skipping it")
    return "\n\n".join(p for p in parts if p)


def run_weekly_review(
    settings: Settings | None = None, force: bool = False, agent=None
) -> dict:
    """Fire this week's review if it is due and unsent; return what happened.

    ``force=True`` (the manual endpoint) skips the day/time gate but still
    claims the ISO-week ledger, so a forced review replaces — not duplicates —
    the scheduled one. With ``agent`` given (and ``enable_proactive_loop_in``),
    the delivered review is recorded into each conversation's working memory.
    """
    settings = settings or get_settings()
    if not settings.enable_weekly_review and not force:
        return {"sent": False, "reason": "disabled"}

    current = now(settings)
    if not force:
        # Due from (day, time) through the end of the ISO week — a server
        # asleep at the due moment still reviews when it wakes that week.
        if (current.weekday(), current.time()) < (_due_day(settings), _due_time(settings)):
            return {"sent": False, "reason": "not due yet"}
        # Quiet hours and an all-scope mute hold the review (nothing is
        # claimed) until the first tick after they end, like the briefing.
        from .memory.profile import in_quiet_hours

        if in_quiet_hours(settings, current):
            return {"sent": False, "reason": "quiet hours"}
        from .mutes import all_muted

        if all_muted(settings, current):
            return {"sent": False, "reason": "muted"}

    year, week, _ = current.isocalendar()
    iso_week = f"{year}-W{week:02d}"
    fired_at = current.isoformat(timespec="seconds")
    claimed = fired_ledger.claim(_LEDGER, settings, [(iso_week,)], fired_at, current)
    if not claimed:
        return {"sent": False, "reason": "already sent this week"}

    from .compose import compose_push

    digest = build_weekly_review(settings)
    message = compose_push(
        settings,
        instruction=(
            "Compose the user's weekly review from the sections below — a "
            "short reflective message in your own voice, plain text, in the "
            "user's language. Celebrate what got done, note streaks and "
            "spending briefly, then set up the week ahead: the busiest days, "
            "what is due, anything renewing or upcoming. If the week looks "
            "quiet, say so. Reply with the review only."
        ),
        facts=digest,
        query="weekly review last week next week agenda tasks habits spending",
        fallback=digest,
    )
    delivered = deliver_reminder(
        settings, {"title": "Weekly review", "message": message}, kind="briefing"
    )
    if not delivered:
        # Claim stands even if no channel is configured — retrying every tick
        # would rebuild a push that can never land.
        logger.warning("weekly review built but no delivery channel accepted it")
    else:
        from .proactive import record_push

        record_push(agent, settings, f"Weekly review: {message}")
    return {"sent": True, "delivered": delivered, "week": iso_week}
