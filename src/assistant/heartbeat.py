"""The heartbeat — the single entry point for all of Wakiru's background activity.

On every beat the assistant *wakes up, looks around, and decides* whether
reaching out helps right now: a due calendar/task/birthday/renewal reminder,
a due followup it scheduled for itself, the daily briefing or weekly review,
the inbox changing, not having heard from the user in a while — or nothing at
all (the common case, answered SILENT). Due reminders are claimed into the
situation report, never auto-fired to a channel: the model bundles, phrases,
or skips them like an assistant would. Each wake also starts by running the
due data refreshes (:mod:`assistant.refreshes`), so nothing else needs its
own ticker. It composes every message itself, so proactive contact reads
like the assistant, not a template.

The model is the judge; only what is not its to override stays deterministic:

* Quiet hours and an all-scope mute hold everything, exactly as they hold
  reminders — the model is never woken against the user's stated do-not-disturb.
* ``heartbeat_min_gap_minutes`` bounds *delivery*, not judgment: an ambient
  push (no due followup, no briefing) within the gap since the last push is
  suppressed, so a chatty model can't become a barrage. Scheduled intent
  always delivers.
* Every beat is a model call — ``heartbeat_minutes`` is the direct token-cost
  dial.

The wake runs as its own bounded tool loop over the restricted
``mode="heartbeat"`` registry (never ``send_email``) with **no checkpointer**:
a silent wake leaves no trace in any conversation. Only when the model
actually speaks does the message go out through the reminder channels and get
recorded into working memory via :func:`assistant.proactive.record_push` —
the same loop-in path reminders use, so "what was that about?" works.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from . import followups, goals, people, persona, threads, watches
from .calendar.context import now
from .calendar.store import parse_dt
from .config import Settings, get_settings, postgres_backend
from .context_providers import build_context
from .followups import Followup
from .llm import build_model
from .tools import ToolContext, available_tools, execute_tool

logger = logging.getLogger(__name__)

_SILENT = "SILENT"

# A bare sentinel (any case) with tolerated trailing punctuation, or a reply
# that *ends* in a standalone, uppercase SILENT token. The trailing-token branch
# is case-sensitive on purpose: a real message ending "…keep it silent." uses
# lowercase and must still deliver, while all-caps SILENT is the model signaling
# the sentinel even when it narrated its way there ("reasoning… SILENT?").
_TRAILING_PUNCT = r"[\s?.!…\"'’)\]]*"
_BARE_SILENT = re.compile(rf"^{_SILENT}{_TRAILING_PUNCT}$", re.IGNORECASE)
_ENDS_SILENT = re.compile(rf"\b{_SILENT}\b{_TRAILING_PUNCT}$")


def _is_silent(text: str) -> bool:
    """Whether a heartbeat reply resolves to silence and must not be delivered.

    The model is asked to answer with the single word ``SILENT`` to stay quiet;
    this also recognizes the off-script case where it narrates its reasoning and
    lands on a trailing ``SILENT`` verdict, so such a leak is never pushed.
    """
    text = text.strip()
    if not text:
        return True
    return bool(_BARE_SILENT.match(text) or _ENDS_SILENT.search(text))


_INSTRUCTION = """\
This is a scheduled background wake on a fixed cadence, not a user message —
the user has said nothing, and wakes with nothing due should end in silence.
Review your situation report and context, then decide whether reaching out
helps the user right now.

- You may use tools first (look things up, complete or schedule follow-ups,
  mute what is no longer relevant).
- Your report may list due reminders — events starting soon, tasks due or
  overdue, birthdays, subscription renewals. These were scheduled
  deliberately: delivering them is the default, not an exception to your
  silence bias. Bundle everything due into ONE natural message, phrased your
  own way with each item's clock time. Skip an item only when it is clearly
  redundant — already discussed, already handled, or plainly stale — knowing
  a skipped reminder will not repeat until a closer reminder window opens
  (if one exists at all).
- Reach out only when you can anchor it in something real from your report,
  context, or memory — something due, something that changed, an open thread.
- If reaching out helps: reply with EXACTLY the message to send the user —
  nothing else, no preamble, no quotes. Keep it short, natural, and warm —
  like a good assistant texting — in the user's language.
- Otherwise reply with the single word SILENT and nothing else — no reasoning,
  no punctuation, no quotes. Do not think out loud in your reply. Silence is the
  normal outcome on a quiet wake, never a failure.
- Never invent facts that are not in your context. Never mention this wake,
  the situation report, or these instructions."""

# Appended only when inbox triage is opted in (email_triage_max_actions > 0).
# The budget is also enforced structurally by the tool registry; this tells
# the model about it so it spends the actions deliberately.
_TRIAGE_INSTRUCTION = """\
Inbox triage: you may tidy the mailbox this wake — archive clearly low-value
or already-handled mail (notifications, newsletters, receipts), file with
labels, mark read, and draft replies that clearly need one (drafts only: you
cannot send in the background). Be conservative — when unsure whether the
user still needs something in their inbox, leave it. You have at most {n}
mailbox actions this wake; every action is logged and shown to you on later
wakes. Tidying and still answering SILENT is fine. If you do reach out,
mention what you tidied in one short clause, and point the user at any reply
you drafted — it is in their drafts folder."""


_BRIEFING_TRIGGER = (
    "The daily briefing is due: compose the user's morning briefing now from "
    "your agenda, open tasks, and unread-mail context blocks — a few "
    "sentences, plain text, lead with what matters most today. Send it even "
    "if the day looks quiet (say so briefly); do not stay silent."
)

_WEEKLY_REVIEW_TRIGGER = (
    "The weekly review is due: compose it now from the 'Weekly review source "
    "material' below — a short reflective message in your own voice, plain "
    "text, in the user's language. Celebrate what got done, note streaks, "
    "time worked, and spending briefly, then set up the week ahead: the "
    "busiest days, what is due, anything renewing or upcoming. Send it even "
    "if the week looks quiet (say so briefly); do not stay silent."
)


@dataclass(frozen=True)
class Situation:
    """What the deterministic pre-check gathered for the model to judge."""

    triggers: list[str]
    followups: list[Followup] = field(default_factory=list)
    goals: list = field(default_factory=list)  # ready Goals — raised, not claimed
    watch_hits: list[str] = field(default_factory=list)  # fired watch trigger lines
    reminders: list[str] = field(default_factory=list)  # claimed due-reminder lines
    info: list[str] = field(default_factory=list)

    @property
    def scheduled(self) -> bool:
        """Explicit intent (a due reminder, a due follow-up, a goal's due next
        step, a fired watch, the briefing, or the weekly review) vs a purely
        ambient wake — scheduled intent is exempt from the delivery throttle:
        a set due time, ``next_action_at``, or watch condition is deliberate."""
        return (
            bool(self.reminders)
            or bool(self.followups)
            or bool(self.goals)
            or bool(self.watch_hits)
            or _BRIEFING_TRIGGER in self.triggers
            or _WEEKLY_REVIEW_TRIGGER in self.triggers
        )

    def report(self) -> str:
        lines = ["## Situation report (background wake)"]
        lines += [f"- {trigger}" for trigger in self.triggers]
        lines += [f"- {hit}" for hit in self.watch_hits]
        lines += [f"- Due reminder: {line}" for line in self.reminders]
        for item in self.followups:
            lines.append(
                f"- Due follow-up: {item.topic}"
                + (f" — context: {item.context}" if item.context else "")
            )
        for goal in self.goals:
            lines.append(
                f"- Goal ready for its next step: {goal.title} (id {goal.id})"
                + (f" — state: {goal.state}" if goal.state else "")
                + " — advance it now with your tools, then update_goal with the"
                " new state and next_action (or park it); working on a goal and"
                " still answering SILENT is fine."
            )
        if (
            not self.triggers
            and not self.followups
            and not self.goals
            and not self.watch_hits
            and not self.reminders
        ):
            lines.append(
                "- Nothing specific happened since your last wake. Review your "
                "context and decide; most such wakes should stay SILENT."
            )
        lines += [f"- {line}" for line in self.info]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Wake state (last wake / last push / last seen mail) — a tiny KV in followups.db
# --------------------------------------------------------------------------- #

_KV_NAMESPACE = "heartbeat"


def state_get(settings: Settings, key: str) -> str:
    if storage_postgres := postgres_backend(settings):
        return storage_postgres.kv_get(settings, _KV_NAMESPACE, key)
    with followups._connect(settings) as conn:
        _ensure_state(conn)
        row = conn.execute(
            "SELECT value FROM heartbeat_state WHERE key = ?", (key,)
        ).fetchone()
    return row["value"] if row else ""


def state_set(settings: Settings, key: str, value: str) -> None:
    if storage_postgres := postgres_backend(settings):
        storage_postgres.kv_set(settings, _KV_NAMESPACE, key, value)
        return
    with followups._connect(settings) as conn:
        _ensure_state(conn)
        conn.execute(
            "INSERT OR REPLACE INTO heartbeat_state (key, value) VALUES (?, ?)",
            (key, value),
        )


def state_clear(settings: Settings, *keys: str) -> None:
    if storage_postgres := postgres_backend(settings):
        storage_postgres.kv_clear(settings, _KV_NAMESPACE, list(keys))
        return
    with followups._connect(settings) as conn:
        _ensure_state(conn)
        conn.executemany(
            "DELETE FROM heartbeat_state WHERE key = ?", [(key,) for key in keys]
        )


# Back-compat aliases: the state KV was private before the self-pacing round.
_state_get = state_get
_state_set = state_set


def _ensure_state(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS heartbeat_state"
        " (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )


def _minutes_since(settings: Settings, key: str, current: datetime) -> float | None:
    stamp = parse_dt(_state_get(settings, key))
    if stamp is None:
        return None
    return (current - stamp).total_seconds() / 60


# --------------------------------------------------------------------------- #
# Self-pacing — when the scheduler should wake the model next
# --------------------------------------------------------------------------- #

def next_wake_at(settings: Settings, current: datetime) -> datetime:
    """When the loop should next wake the model — a pure read over the state KV.

    The fixed ``heartbeat_minutes`` cadence is the default and (by default) the
    ceiling. The model may pull the next wake earlier or, when
    ``heartbeat_wake_max_minutes`` is raised, push it later via ``set_next_wake``
    (stored in the ``next_wake_at`` KV); the request is clamped into
    ``[anchor + wake_min, anchor + (wake_max or heartbeat_minutes)]``. The
    scheduler also never sleeps past the soonest deliberate intent — an open
    follow-up, a goal's next step, a watch window, a reminder band opening, a
    scheduled digest, or a data refresh coming due — but never wakes before
    the floor. During quiet hours or an all-scope mute nothing can be
    delivered, so it simply ticks at the base cadence.
    """
    base = timedelta(minutes=max(settings.heartbeat_minutes, 1))
    if not settings.enable_heartbeat:
        return current + base

    from .memory.profile import in_quiet_hours
    from .mutes import all_muted

    if in_quiet_hours(settings, current) or all_muted(settings, current):
        return current + base

    anchor_raw = _state_get(settings, "last_wake_at")
    anchor = parse_dt(anchor_raw) if anchor_raw else None
    anchor = anchor or current
    floor = anchor + timedelta(minutes=max(settings.heartbeat_wake_min_minutes, 0))
    ceiling = anchor + timedelta(
        minutes=settings.heartbeat_wake_max_minutes or max(settings.heartbeat_minutes, 1)
    )

    requested_raw = _state_get(settings, "next_wake_at")
    requested = parse_dt(requested_raw) if requested_raw else None
    target = requested or (anchor + base)
    target = min(max(target, floor), ceiling)

    # A soon-due open follow-up pulls the wake earlier — but never before the
    # floor, so a follow-up due "now" still can't busy-loop the model.
    due = [d for f in followups.list_open(settings) if (d := parse_dt(f.due)) is not None]
    # A goal's next_action_at is the same kind of deliberate intent as a
    # followup's due time, so it pulls the wake the same way — and so do a
    # calendar-window watch's opening and a silence watch's deadline.
    due += [
        d
        for g in goals.list_open(settings)
        if (d := parse_dt(g.next_action_at)) is not None
    ]
    try:
        due += watches.wake_times(settings, current)
    except Exception:
        logger.exception("computing watch wake times failed")
    # A reminder band opening is the most deliberate intent of all — the wake
    # is pulled to the window's start so the base cadence can't make a
    # "15 minutes before" nudge land late. Pure reads (cached importance
    # tiers only); each source is best-effort so one failing store never
    # blocks the scheduler.
    try:
        from .calendar import reminders as calendar_reminders
        from .tasks import reminders as task_reminders

        due += calendar_reminders.next_due_at(settings, current, ceiling)
        due += task_reminders.next_due_at(settings, current, ceiling)
    except Exception:
        logger.exception("computing reminder wake times failed")
    if settings.enable_reminders and (
        settings.enable_people or settings.enable_subscriptions
    ):
        # Birthday/renewal windows are day-granular: they only change at the
        # next local midnight.
        due.append(
            (current + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        )
    due += _scheduled_wakes(settings, current)
    try:
        from . import refreshes

        # The anti-starvation cap: a model that backed its wake far off still
        # wakes when a data refresh (mail, weather, calendar sync) comes due.
        due += refreshes.next_due_at(settings, current)
    except Exception:
        logger.exception("computing refresh wake times failed")
    if due:
        target = min(target, max(min(due), floor))
    return target


def _scheduled_wakes(settings: Settings, current: datetime) -> list[datetime]:
    """The next briefing / weekly-review instants — pure wall-clock computation.

    No ledger check: waking at an already-claimed time is a harmless ambient
    wake, and the claim in :func:`gather_situation` keeps it exactly-once.
    """
    wakes: list[datetime] = []
    if settings.enable_briefing:
        from . import briefing

        at = current.replace(
            hour=briefing._due_time(settings).hour,
            minute=briefing._due_time(settings).minute,
            second=0,
            microsecond=0,
        )
        wakes.append(at if at > current else at + timedelta(days=1))
    if settings.enable_weekly_review:
        from . import weekly_review

        due_time = weekly_review._due_time(settings)
        days_ahead = (weekly_review._due_day(settings) - current.weekday()) % 7
        at = (current + timedelta(days=days_ahead)).replace(
            hour=due_time.hour, minute=due_time.minute, second=0, microsecond=0
        )
        wakes.append(at if at > current else at + timedelta(days=7))
    return wakes


# --------------------------------------------------------------------------- #
# The deterministic pre-check
# --------------------------------------------------------------------------- #

def gather_situation(
    settings: Settings,
    current: datetime | None = None,
    force_briefing: bool = False,
    force_weekly_review: bool = False,
) -> Situation | None:
    """Gather the situation for the model to judge; hold only when it must.

    The model — not a trigger table — decides whether to reach out, so every
    beat returns a :class:`Situation` (triggers when something happened,
    ambient facts either way). ``None`` only when the heartbeat is disabled or
    during quiet hours / an all-scope mute: the user's stated do-not-disturb
    is not the model's to override. Holds claim nothing, so a followup or
    reminder due at 03:00 is raised on the first wake after quiet ends.
    ``force_briefing`` / ``force_weekly_review`` bypass those digests'
    time gates (``POST /briefing/run`` / ``POST /weekly-review/run``), still
    claiming their exactly-once ledgers.
    """
    if not settings.enable_heartbeat:
        return None
    current = current or now(settings)

    from .memory.profile import in_quiet_hours
    from .mutes import all_muted

    if in_quiet_hours(settings, current) or all_muted(settings, current):
        return None

    triggers: list[str] = []

    # Scheduled intent, claimed here (exactly-once); a wake that then stays
    # SILENT still consumes the claim — the same at-most-once tradeoff the
    # reminder ledgers make (the briefing instruction tells the model not to).
    due = followups.claim_due(settings, current)
    if _briefing_due(settings, current, force=force_briefing):
        triggers.append(_BRIEFING_TRIGGER)
    review_due = _weekly_review_due(settings, current, force=force_weekly_review)
    if review_due:
        triggers.append(_WEEKLY_REVIEW_TRIGGER)

    # Due reminders (calendar events, dated tasks, birthdays, renewals) are
    # claimed here — surfaced for the model's judgment, never auto-fired to a
    # channel. Claims are at-most-once per band: a reminder the model skips
    # resurfaces only when a closer band opens. Each family is best-effort,
    # like the watches, so one failing store never blocks the wake.
    reminder_lines = _due_reminder_lines(settings, current)

    # Ready goals are raised, never claimed: the model moves next_action_at
    # forward itself (update_goal). The raise-stamp KV keeps a goal the model
    # ignored from re-raising on every short self-paced wake — it comes back
    # once per base-cadence window, or immediately once the model touches it.
    ready_goals = _raisable_goals(settings, current)

    # Model-registered perception: evaluate every active watch (deterministic,
    # token-free); a firing consumes the watch (one-shots) and hands the model
    # back its own note-to-self.
    try:
        watch_hits = [line for _watch, line in watches.evaluate(settings, current)]
    except Exception:
        logger.exception("heartbeat: evaluating watches failed")
        watch_hits = []

    # Ambient observations — information for the model's judgment, not gates.
    mail_line = _mail_changed(settings)
    if mail_line:
        triggers.append(mail_line)
    stale_line = _contact_stale(settings, current)
    if stale_line:
        triggers.append(stale_line)
    people_line = _people_attention(settings, current)
    if people_line:
        triggers.append(people_line)

    info: list[str] = []
    if review_due:
        # The digest the template path would have sent — handed to the model
        # as source material so the review is grounded, not improvised.
        try:
            from .weekly_review import build_weekly_review

            info.append("Weekly review source material:\n" + build_weekly_review(settings))
        except Exception:
            logger.exception("heartbeat: building the weekly review digest failed")
    since_push = _minutes_since(settings, "last_push_at", current)
    if since_push is not None:
        info.append(f"You last reached out proactively {int(since_push)} minutes ago.")
    last_heard = threads.last_contact(settings)
    if last_heard is not None:
        minutes = int((current - last_heard).total_seconds() / 60)
        info.append(f"You last heard from the user {minutes} minutes ago.")

    # The standing intentions carried across wakes (open follow-ups, active
    # watches, stale goals) and what earlier wakes already did (mailbox actions,
    # nudges delivered) — all surfaced for judgment, never auto-actioned.
    info += _standing_info(settings, current)
    info += _accountability_info(settings, current)

    return Situation(
        triggers=triggers,
        followups=due,
        goals=ready_goals,
        watch_hits=watch_hits,
        reminders=reminder_lines,
        info=info,
    )


def _due_reminder_lines(settings: Settings, current: datetime) -> list[str]:
    """Due-reminder lines across the four families — calendar events, dated
    tasks, birthdays, subscription renewals.

    Claimed here (at-most-once per band): a reminder the model skips resurfaces
    only when a closer band opens. Each family is best-effort, so one failing
    store never blocks the wake. Empty when reminders are disabled.
    """
    if not settings.enable_reminders:
        return []
    from .calendar import reminders as calendar_reminders
    from .calendar.context import format_when
    from .people import reminders as birthday_reminders
    from .subscriptions import reminders as subscription_reminders
    from .tasks import reminders as task_reminders

    families = (
        (
            calendar_reminders.surface_due,
            lambda r: f"{r['message']} (starts {format_when(settings, r['start'])})",
        ),
        (
            task_reminders.surface_due,
            lambda r: f"{r['message']} (due {format_when(settings, r['due'])})",
        ),
        (birthday_reminders.surface_due, lambda r: r["message"]),
        (subscription_reminders.surface_due, lambda r: r["message"]),
    )
    lines: list[str] = []
    for surface, render in families:
        try:
            lines += [render(r) for r in surface(settings, current)]
        except Exception:
            logger.exception("heartbeat: surfacing due reminders failed")
    return lines


def _standing_info(settings: Settings, current: datetime) -> list[str]:
    """The standing intentions carried across wakes: open follow-ups not yet due
    (the due ones are claimed in :func:`gather_situation`), active watches, and
    goals left untouched too long. Surfaced every beat so the model can act
    toward, reschedule, prune, or advance them — never auto-actioned.
    """
    info: list[str] = []
    open_items = followups.list_open(settings)
    if open_items:
        from .calendar.context import format_when

        info.append(
            "Open follow-ups you are carrying (reschedule, update, or cancel "
            "them as things change):"
        )
        for item in open_items[:5]:
            line = f"  - {item.topic} @ {format_when(settings, item.due)} (id {item.id})"
            info.append(line + (f" — {item.context}" if item.context else ""))

    active_watches = watches.list_active(settings, current)
    if active_watches:
        info.append("Watches you have set (drop stale ones with unwatch):")
        info += [
            f"  - [{w.kind}] {w.url or w.pattern or 'user silence'}"
            + (f" — {w.note}" if w.note else "")
            + f" (id {w.id})"
            for w in active_watches[:8]
        ]

    # Open goals already ride along via the goals context block; only nudge
    # about ones the model has left untouched too long.
    for goal in goals.stale(settings, current):
        info.append(
            f"Stale goal: {goal.title} (id {goal.id}) has not moved in "
            f"{settings.goal_stale_days}+ days — advance it, reschedule its "
            "next step, or close it as abandoned."
        )
    return info


def _accountability_info(settings: Settings, current: datetime) -> list[str]:
    """What earlier wakes already did — mailbox actions taken and nudges
    delivered — so this wake never re-archives/re-labels/re-drafts what an
    earlier one handled, nor re-sends the same nudge in different words. Both
    ledgers are best-effort; a read failure just yields nothing.
    """
    info: list[str] = []
    if settings.enable_email and settings.email_triage_max_actions > 0:
        try:
            from .mail import audit as mail_audit

            actions = mail_audit.recent(settings, limit=5, actor="heartbeat")
        except Exception:
            logger.exception("heartbeat: reading the mail audit ledger failed")
            actions = []
        if actions:
            info.append("Mailbox actions you took on recent wakes (do not redo them):")
            info += [f"  - {row['detail']}" for row in actions]

    if settings.heartbeat_dedup_push_hours > 0:
        try:
            from .reflect import recent_pushes

            delivered = recent_pushes(
                settings, current - timedelta(hours=settings.heartbeat_dedup_push_hours)
            )
        except Exception:
            logger.exception("heartbeat: reading the recent-push log failed")
            delivered = []
        if delivered:
            info.append(
                "Nudges already delivered to the user recently — do NOT send these "
                "again, or a reworded version; they have been handled:"
            )
            info += [f"  - [{row['kind']}] {row['excerpt']}" for row in delivered[-6:]]
    return info


def _raisable_goals(settings: Settings, current: datetime) -> list:
    """Ready goals worth raising this wake, with a per-goal raise stamp.

    A goal the model advanced (``updated_at`` newer than the stamp) raises
    immediately; one it ignored comes back only after a full base-cadence
    window, so short self-paced wakes don't nag the same stuck goal.
    """
    window = timedelta(
        minutes=settings.heartbeat_wake_max_minutes or max(settings.heartbeat_minutes, 1)
    )
    raisable = []
    for goal in goals.ready(settings, current):
        # The stamp remembers when the goal was last raised and the
        # updated_at it had then: a different updated_at means the model
        # touched it since, so it raises again right away.
        raised_at_raw, _, seen_updated = _state_get(
            settings, f"goal_raise:{goal.id}"
        ).partition("|")
        raised_at = parse_dt(raised_at_raw) if raised_at_raw else None
        if (
            raised_at is None
            or goal.updated_at != seen_updated
            or current - raised_at >= window
        ):
            _state_set(
                settings,
                f"goal_raise:{goal.id}",
                f"{current.isoformat(timespec='seconds')}|{goal.updated_at}",
            )
            raisable.append(goal)
    return raisable


def _briefing_due(settings: Settings, current: datetime, force: bool = False) -> bool:
    """Claim today's briefing slot when it is due (or forced) and unclaimed.

    Rides the same once-per-day ledger the template briefing uses, so flipping
    ``enable_heartbeat`` mid-day never double-briefs.
    """
    from . import briefing, fired_ledger

    if not (settings.enable_briefing or force):
        return False
    if not force and current.time() < briefing._due_time(settings):
        return False
    local_date = current.date().isoformat()
    claimed = fired_ledger.claim(
        briefing._LEDGER,
        settings,
        [(local_date,)],
        current.isoformat(timespec="seconds"),
        current,
    )
    return bool(claimed)


def _weekly_review_due(settings: Settings, current: datetime, force: bool = False) -> bool:
    """Claim this week's review slot when it is due (or forced) and unclaimed.

    Rides the same once-per-ISO-week ledger the template review uses, so
    flipping ``enable_heartbeat`` mid-week never double-sends. Due from
    (day, time) through the end of the ISO week — a server asleep at the due
    moment still reviews when it wakes that week.
    """
    from . import fired_ledger, weekly_review

    if not (settings.enable_weekly_review or force):
        return False
    if not force and (current.weekday(), current.time()) < (
        weekly_review._due_day(settings),
        weekly_review._due_time(settings),
    ):
        return False
    year, week, _ = current.isocalendar()
    claimed = fired_ledger.claim(
        weekly_review._LEDGER,
        settings,
        [(f"{year}-W{week:02d}",)],
        current.isoformat(timespec="seconds"),
        current,
    )
    return bool(claimed)


def _mail_changed(settings: Settings) -> str:
    """A trigger line when the unread-mail snapshot changed since last seen.

    Consumes the change (stores the new hash) as soon as it triggers, so a
    SILENT verdict doesn't re-raise the same inbox every wake.
    """
    if not (settings.enable_email and settings.email_snapshot_minutes > 0):
        return ""
    from .mail.snapshot import content as snapshot_content

    # Key off the raw unread set, never current()'s stamped block — the "as of
    # HH:MM" stamp advances on every refresh and would fake a change each tick.
    snapshot = snapshot_content(settings)
    if not snapshot:
        return ""
    digest = hashlib.sha256(snapshot.encode("utf-8")).hexdigest()
    if digest == _state_get(settings, "last_mail_hash"):
        return ""
    _state_set(settings, "last_mail_hash", digest)
    return "The unread-mail snapshot changed since your last wake (see the mail block)."


def _people_attention(settings: Settings, current: datetime) -> str:
    """A trigger line when someone in the CRM is overdue for contact or has a
    birthday coming (see :func:`assistant.people.attention_lines`).

    Recomputed each wake (like ``_contact_stale``, not consumed like a claim):
    the birthday/overdue signal simply persists while it is true. The ambient
    push gap keeps a chatty model from acting on it every beat.
    """
    if not settings.enable_people:
        return ""
    lines = people.attention_lines(settings, current)
    if not lines:
        return ""
    return (
        "People to keep in touch with: " + "; ".join(lines) + ". Reach out only "
        "if you can anchor it in something real (their birthday, a shared thread) "
        "— never empty small talk, and don't repeat outreach you already made."
    )


def _contact_stale(settings: Settings, current: datetime) -> str:
    if settings.heartbeat_contact_gap_hours <= 0:
        return ""
    last = threads.last_contact(settings)
    if last is None:
        return ""  # never spoken — nothing to follow up on yet
    gap = current - last
    if gap < timedelta(hours=settings.heartbeat_contact_gap_hours):
        return ""
    hours = int(gap.total_seconds() // 3600)
    return (
        f"You haven't heard from the user in about {hours} hours. Reach out "
        "only if you can anchor it in something real — an open thread, "
        "something due, something you remembered — not empty small talk."
    )


# --------------------------------------------------------------------------- #
# The wake itself — a bounded, checkpointer-less tool loop
# --------------------------------------------------------------------------- #

def _latest_summary(settings: Settings, agent) -> str:
    """The rolling summary of the most recently active thread, best-effort."""
    if agent is None:
        return ""
    try:
        infos = [t for t in threads.known_threads(settings) if t.last_user_at]
        if not infos:
            return ""
        latest = max(infos, key=lambda t: t.last_user_at)
        state = agent.get_state(
            {"configurable": {"thread_id": latest.thread_id}}
        ).values
        return state.get("summary", "") or ""
    except Exception:
        logger.exception("heartbeat: reading the latest thread summary failed")
        return ""


def _compose(settings: Settings, situation: Situation, agent) -> str:
    """One bounded tool loop over the restricted registry; returns final text."""
    model = build_model(settings)
    specs = available_tools(settings, mode="heartbeat")
    by_name = {spec.name: spec for spec in specs}
    bound = model.bind_tools([spec.to_openai_tool() for spec in specs]) if specs else model

    query = " ".join(
        [item.topic for item in situation.followups]
        + [goal.title for goal in situation.goals]
        + situation.triggers
        + situation.watch_hits
        + situation.reminders
    ) or "check in with the user"
    prefix: list[BaseMessage] = [persona.system_message(settings)]
    for _name, block in build_context(settings, query, "").items():
        if block:
            prefix.append(SystemMessage(content=block))
    summary = _latest_summary(settings, agent)
    if summary:
        if settings.heartbeat_summary_as_tool:
            # Don't let a possibly-stale conversation narrative sit in the prompt
            # and override fresh facts (the morning-briefing-reports-yesterday's-
            # stats bug). Point at it instead; the model fetches it on demand.
            from .tools.memory import CONVERSATION_POINTER

            prefix.append(SystemMessage(content=CONVERSATION_POINTER))
        else:
            prefix.append(
                SystemMessage(content="Latest conversation so far:\n" + summary)
            )
    prefix.append(SystemMessage(content=_INSTRUCTION))
    # The freshest lessons the nightly reflection distilled about how this
    # assistant's own proactivity lands — a guaranteed slot, not left to recall.
    try:
        from .reflect import self_notes

        lessons = self_notes(settings, k=5)
    except Exception:
        logger.exception("heartbeat: reading reflection self-notes failed")
        lessons = []
    if lessons:
        prefix.append(
            SystemMessage(
                content="What you have learned about your own proactivity:\n"
                + "\n".join(f"- {note.body}" for note in lessons)
            )
        )
    if settings.enable_email and settings.email_triage_max_actions > 0:
        prefix.append(
            SystemMessage(
                content=_TRIAGE_INSTRUCTION.format(n=settings.email_triage_max_actions)
            )
        )

    ctx = ToolContext(
        settings=settings, thread_id="", batch_id=uuid.uuid4().hex, summary=summary
    )
    messages: list[BaseMessage] = [HumanMessage(content=situation.report())]
    for _round in range(max(settings.tool_max_rounds, 1)):
        reply = bound.invoke(prefix + messages)
        messages.append(reply)
        calls = reply.tool_calls if isinstance(reply, AIMessage) else []
        if not calls:
            break
        for call in calls:
            spec = by_name.get(call["name"])
            if spec is None:
                output = f"Unknown tool: {call['name']}."
            else:
                output = execute_tool(spec, ctx, call.get("args") or {})
            messages.append(
                ToolMessage(content=output, tool_call_id=call["id"], name=call["name"])
            )
    else:
        # Budget exhausted mid-tool-call: one final tool-less pass for text.
        reply = model.invoke(prefix + messages)
        messages.append(reply)

    text = messages[-1].content
    return text if isinstance(text, str) else str(text)


def run_heartbeat(
    settings: Settings | None = None,
    agent=None,
    force: bool = False,
    force_briefing: bool = False,
    force_weekly_review: bool = False,
) -> dict:
    """One heartbeat: refresh data, gather the situation, wake the model.

    Returns what happened (``sent`` / ``reason`` / ``triggers``), mirroring
    :func:`assistant.briefing.run_briefing`'s shape. With ``agent`` given, a
    delivered message is looped into working memory on every target thread.
    ``force`` (the manual endpoint) bypasses the ambient delivery throttle,
    never the quiet-hours/mute holds.
    """
    settings = settings or get_settings()
    current = now(settings)

    # Data refreshes ride every wake — before the gather, so the situation and
    # context blocks see fresh mail/weather/calendar. Deliberately outside the
    # holds below: plumbing must run even when the deliberative layer is
    # disabled, quiet, or muted.
    try:
        from . import refreshes

        refreshes.run_due(settings)
    except Exception:
        logger.exception("heartbeat: running due refreshes failed")

    situation = gather_situation(
        settings,
        current,
        force_briefing=force_briefing,
        force_weekly_review=force_weekly_review,
    )
    if situation is None:
        return {"sent": False, "reason": "held"}

    # Consume any wake the model scheduled for itself: clear it so the next tick
    # falls back to the base cadence unless the model re-schedules, and surface
    # the reason it left so it knows why it is awake now.
    reason = _state_get(settings, "next_wake_reason")
    state_clear(settings, "next_wake_at", "next_wake_reason")
    if reason:
        situation.info.append(f"You scheduled this wake yourself: {reason}")

    _state_set(settings, "last_wake_at", current.isoformat(timespec="seconds"))
    triggers = (
        situation.triggers
        + situation.watch_hits
        + [f"reminder: {line}" for line in situation.reminders]
        + [f"followup: {item.topic}" for item in situation.followups]
        + [f"goal: {goal.title}" for goal in situation.goals]
    )
    try:
        text = _compose(settings, situation, agent).strip()
    except Exception:
        logger.exception("heartbeat composition failed")
        return {"sent": False, "reason": "composition failed", "triggers": triggers}

    if _is_silent(text):
        logger.info("heartbeat stayed silent (triggers: %s)", "; ".join(triggers))
        return {"sent": False, "reason": "silent", "triggers": triggers}

    # The min-gap bound applies to ambient pushes only: the model's judgment
    # decides *whether* to speak, this decides no more often than the user can
    # stand. Scheduled intent (a due followup, the briefing) always delivers.
    if not situation.scheduled and not force:
        since_push = _minutes_since(settings, "last_push_at", current)
        if since_push is not None and since_push < settings.heartbeat_min_gap_minutes:
            logger.info(
                "heartbeat suppressed an ambient push (last push %d min ago)",
                int(since_push),
            )
            return {"sent": False, "reason": "throttled", "triggers": triggers}

    from .notify import deliver_reminder
    from .proactive import record_push

    delivered = deliver_reminder(
        settings, {"title": "Wakiru", "message": text}, kind="heartbeat"
    )
    if delivered:
        _state_set(settings, "last_push_at", current.isoformat(timespec="seconds"))
        record_push(agent, settings, text)
    else:
        logger.warning("heartbeat composed a message but no channel accepted it")
    return {"sent": True, "delivered": delivered, "message": text, "triggers": triggers}
