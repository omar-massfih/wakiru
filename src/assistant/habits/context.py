"""Streak/trend computation and rendering over the habit log — shared by the
habit tools and the GET /habits endpoint.
"""

from __future__ import annotations

from datetime import date, timedelta

from ..config import Settings
from . import store
from .store import HabitEntry, parse_date


def _num(value: float) -> str:
    return str(int(value)) if value == int(value) else str(round(value, 2))


def current_streak(entries: list[HabitEntry], today: date) -> int:
    """Consecutive days ending at the most recent entry (0 if the last entry is
    older than yesterday, so a lapsed habit reads as broken)."""
    days = sorted({d for e in entries if (d := parse_date(e.logged_on))}, reverse=True)
    if not days:
        return 0
    latest = days[0]
    if (today - latest).days > 1:
        return 0  # streak broken — last log was before yesterday
    streak = 1
    prev = latest
    for d in days[1:]:
        if (prev - d).days == 1:
            streak += 1
            prev = d
        elif d == prev:
            continue
        else:
            break
    return streak


def summarize(settings: Settings, habit: str, today: date) -> str:
    """A detailed summary for one habit: streak, last entry, recent count/average."""
    entries = store.list_entries(settings, habit)
    if not entries:
        return f"No entries logged for {habit!r} yet."
    name = entries[0].habit
    lines = [f"{name}:"]
    lines.append(f"  logged {len(entries)}× total")
    streak = current_streak(entries, today)
    if streak:
        lines.append(f"  current streak: {streak} day(s)")
    last = entries[0]
    val = f" — {_num(last.value)} {last.unit}".rstrip() if last.value else ""
    lines.append(f"  last: {last.logged_on}{val}")
    week_ago = (today - timedelta(days=7)).isoformat()
    recent = [e for e in entries if e.logged_on >= week_ago]
    if recent:
        lines.append(f"  last 7 days: {len(recent)} entr(y/ies)")
        valued = [e.value for e in recent if e.value]
        if valued:
            avg = sum(valued) / len(valued)
            lines.append(f"  7-day average value: {_num(avg)} {recent[0].unit}".rstrip())
    # A few recent entries with ids, so the model can correct a mis-log.
    lines.append("  recent:")
    for e in entries[:5]:
        v = f" {_num(e.value)} {e.unit}".rstrip() if e.value else ""
        note = f" ({e.note})" if e.note else ""
        lines.append(f"    - {e.logged_on}{v}{note}  [id: {e.id}]")
    return "\n".join(lines)


def overview(settings: Settings, today: date) -> str:
    """One line per tracked habit: total count, streak, and last log."""
    names = store.habit_names(settings)
    if not names:
        return "No habits logged yet."
    lines = ["Habits:"]
    for name in names:
        entries = store.list_entries(settings, name)
        streak = current_streak(entries, today)
        streak_str = f", {streak}-day streak" if streak else ""
        last = entries[0]
        val = ""
        if last.value:
            unit = f" {last.unit}" if last.unit else ""
            val = f" ({_num(last.value)}{unit})"
        lines.append(f"- {name}: {len(entries)}× logged{streak_str}, last {last.logged_on}{val}")
    return "\n".join(lines)
