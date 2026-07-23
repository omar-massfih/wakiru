"""Daily briefing — one proactive digest per day, pushed like a reminder.

Assembles the existing read paths (agenda, open tasks, unread mail when email
is on) into a single morning digest, composed by the model in Wakiru's own
voice either way: with the heartbeat enabled the briefing is a heartbeat wake
trigger; without it, :func:`assistant.compose.compose_push` writes it from
the assembled sections (which double as the verbatim fallback when the model
fails) and it goes out through
:func:`assistant.notify.deliver_reminder`. Nothing here has its own data
model: the only state is a fired ledger (the shared
:mod:`assistant.fired_ledger` driver) so the ticker, the heartbeat, and a
manual ``POST /briefing/run`` can all drive it without double-sending.

The briefing becomes *due* at ``briefing_time`` (local wall clock in
``TIMEZONE``) and fires on the first call at or after it that day — a server
that was asleep at 07:30 still briefs when it wakes. It never fires twice for
the same local date.
"""

from __future__ import annotations

import logging
from datetime import time as dtime

from . import fired_ledger
from .calendar.context import agenda_context, now
from .config import Settings, get_settings
from .notify import deliver_reminder
from .tasks.context import tasks_context

logger = logging.getLogger(__name__)

_LEDGER = fired_ledger.FiredLedgerSpec(
    table="briefings_fired",
    columns=(("local_date", "TEXT"),),
    db_path=lambda settings: settings.briefing_db_path,
)


def _due_time(settings: Settings) -> dtime:
    """Parse ``briefing_time`` (HH:MM); a malformed value falls back to 07:30."""
    try:
        hour, _, minute = settings.briefing_time.partition(":")
        return dtime(int(hour), int(minute))
    except ValueError:
        logger.warning("invalid BRIEFING_TIME %r; using 07:30", settings.briefing_time)
        return dtime(7, 30)


def build_briefing(settings: Settings) -> str:
    """Assemble the digest text from the subsystem read paths (no LLM)."""
    parts = [agenda_context(settings)]
    if settings.enable_weather:
        try:
            from .weather import current as weather_current

            if block := weather_current(settings):
                parts.append(block)
        except Exception:
            logger.exception("briefing: weather section failed; skipping it")
    if settings.enable_tasks:
        try:
            parts.append(tasks_context(settings))
        except Exception:
            logger.exception("briefing: tasks section failed; skipping it")
    if settings.enable_people:
        try:
            from .people.context import briefing_people

            if block := briefing_people(settings):
                parts.append(block)
        except Exception:
            logger.exception("briefing: people section failed; skipping it")
    if settings.enable_expenses:
        # Non-empty only on the 1st (last month's rollup), so it costs the
        # briefing nothing the rest of the month.
        try:
            from .expenses.context import briefing_expenses

            if block := briefing_expenses(settings, now(settings).date()):
                parts.append(block)
        except Exception:
            logger.exception("briefing: expenses section failed; skipping it")
    if settings.enable_email:
        # Imported lazily so the briefing works with the mail extra not installed.
        try:
            from .mail.context import unread_summary

            parts.append("## Unread mail\n" + unread_summary(settings))
        except Exception:
            logger.exception("briefing: mail section failed; skipping it")
    return "\n\n".join(p for p in parts if p)


def run_briefing(
    settings: Settings | None = None, force: bool = False, agent=None
) -> dict:
    """Fire today's briefing if it is due and unsent; return what happened.

    With the heartbeat enabled, the briefing is one of its wake triggers: the
    model composes it in Wakiru's own voice, and this function is a thin
    dispatcher into :func:`assistant.heartbeat.run_heartbeat` (same
    once-per-day ledger, so flipping the flag never double-briefs). With the
    heartbeat off, the model composes it here from the assembled digest,
    which is delivered verbatim when the model fails.

    ``force=True`` (the manual endpoint) skips the time-of-day gate but still
    claims the ledger, so a forced briefing replaces — not duplicates — the
    scheduled one. With ``agent`` given (and ``enable_proactive_loop_in``), the
    delivered briefing is also recorded into each conversation's working
    memory, so the chat knows what it was sent.
    """
    settings = settings or get_settings()
    if not settings.enable_briefing and not force:
        return {"sent": False, "reason": "disabled"}

    current = now(settings)
    local_date = current.date().isoformat()
    if not force and current.time() < _due_time(settings):
        return {"sent": False, "reason": "not due yet"}

    if settings.enable_heartbeat:
        from .heartbeat import run_heartbeat

        return run_heartbeat(settings, agent=agent, force=force, force_briefing=force)

    if not force:
        # A quiet window reaching past briefing_time holds the briefing (nothing
        # is claimed) until the first tick after quiet ends.
        from .memory.profile import in_quiet_hours

        if in_quiet_hours(settings, current):
            return {"sent": False, "reason": "quiet hours"}
        # An all-scope mute ("no nudges today") holds the briefing the same way.
        from .mutes import all_muted

        if all_muted(settings, current):
            return {"sent": False, "reason": "muted"}

    fired_at = current.isoformat(timespec="seconds")
    claimed = fired_ledger.claim(_LEDGER, settings, [(local_date,)], fired_at, current)
    if not claimed:
        return {"sent": False, "reason": "already sent today"}

    # Composed by the model in the assistant's own voice; the assembled digest
    # is both the source material and the fallback, so a model failure still
    # briefs — verbatim, like before.
    from .compose import compose_push

    digest = build_briefing(settings)
    message = compose_push(
        settings,
        instruction=(
            "Compose the user's morning briefing from the sections below — a "
            "few sentences in your own voice, plain text, in the user's "
            "language. Lead with what matters most today; if the day looks "
            "quiet, say so briefly. Reply with the briefing only."
        ),
        facts=digest,
        query="daily briefing today's agenda open tasks unread mail",
        fallback=digest,
    )
    delivered = deliver_reminder(
        settings, {"title": "Daily briefing", "message": message}, kind="briefing"
    )
    if not delivered:
        # Claim stands even if no channel is configured — retrying every tick
        # would rebuild a push that can never land.
        logger.warning("briefing built but no delivery channel accepted it")
    else:
        from .proactive import record_push

        record_push(agent, settings, f"Daily briefing: {message}")
    return {"sent": True, "delivered": delivered, "date": local_date}
