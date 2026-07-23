"""Read paths over the work log — the running-timer block and the summaries."""

from __future__ import annotations

from datetime import date, timedelta

from ..config import Settings
from . import store
from .store import WorkEntry


def fmt_minutes(total: int) -> str:
    """Whole minutes as "45m" / "2h" / "2h 05m"."""
    hours, minutes = divmod(max(0, int(total)), 60)
    if not hours:
        return f"{minutes}m"
    return f"{hours}h" if not minutes else f"{hours}h {minutes:02d}m"


def totals_by_project(entries: list[WorkEntry]) -> dict[str, int]:
    """Minutes per project (case-insensitive; the newest spelling labels it),
    largest first. Assumes ``entries`` is newest-first, as list_entries returns."""
    labels: dict[str, str] = {}
    totals: dict[str, int] = {}
    for entry in entries:
        key = entry.project.lower()
        label = labels.setdefault(key, entry.project)
        totals[label] = totals.get(label, 0) + entry.minutes
    return dict(sorted(totals.items(), key=lambda kv: -kv[1]))


def _window(settings: Settings, since: date, until: date) -> list[WorkEntry]:
    """Finished entries whose date falls inside ``[since, until]``."""
    return [
        e
        for e in store.list_entries(settings)
        if e.minutes > 0 and since.isoformat() <= e.worked_on <= until.isoformat()
    ]


def timer_context(settings: Settings) -> str:
    """The running-timer block injected per turn — empty (and free) off the clock."""
    running = store.running_entry(settings)
    if running is None:
        return ""
    elapsed = fmt_minutes(store.elapsed_minutes(settings, running))
    note = f" ({running.note})" if running.note else ""
    return (
        "## Work timer\n"
        f"⏱ On the clock: {running.project}{note} — {elapsed} so far. "
        "Stop it with stop_work when the user finishes or switches; "
        "start_work switches projects in one step."
    )


def summary(settings: Settings, today: date, days: int = 7) -> str:
    """Today + the last ``days`` days per project, and recent entries with ids."""
    entries = store.list_entries(settings)
    if not entries and store.running_entry(settings) is None:
        return "No work logged yet."
    lines = []
    running = store.running_entry(settings)
    if running is not None:
        elapsed = fmt_minutes(store.elapsed_minutes(settings, running))
        lines.append(f"Running now: {running.project} — {elapsed} so far  [id: {running.id}]")
    todays = _window(settings, today, today)
    if todays:
        per = ", ".join(f"{p} {fmt_minutes(m)}" for p, m in totals_by_project(todays).items())
        lines.append(f"Today: {fmt_minutes(sum(e.minutes for e in todays))} ({per})")
    span = _window(settings, today - timedelta(days=days - 1), today)
    if span:
        lines.append(f"Last {days} days: {fmt_minutes(sum(e.minutes for e in span))}")
        for project, minutes in totals_by_project(span).items():
            lines.append(f"- {project}: {fmt_minutes(minutes)}")
    finished = [e for e in entries if e.minutes > 0][:5]
    if finished:
        lines.append("Recent entries:")
        for e in finished:
            note = f" ({e.note})" if e.note else ""
            lines.append(f"- {e.worked_on} {e.project} {fmt_minutes(e.minutes)}{note}  [id: {e.id}]")
    return "\n".join(lines) if lines else "No work logged yet."


def weekly_section(settings: Settings, today: date) -> str:
    """The time-per-project block for the weekly review — empty when idle."""
    span = _window(settings, today - timedelta(days=6), today)
    if not span:
        return ""
    lines = [f"total {fmt_minutes(sum(e.minutes for e in span))}"]
    lines += [
        f"{project}: {fmt_minutes(minutes)}"
        for project, minutes in totals_by_project(span).items()
    ]
    return "## Time worked last 7 days\n" + "\n".join(f"- {line}" for line in lines)
