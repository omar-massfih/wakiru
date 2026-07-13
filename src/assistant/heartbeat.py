"""The heartbeat — Wakiru's deliberative proactivity.

Calendar and task reminders are the reflex arc: deterministic, minute-precise,
never dependent on a model's judgment. The heartbeat is the layer above them —
on a slow cadence the assistant *wakes up, looks around, and decides* whether
reaching out helps right now: a due followup it scheduled for itself, the
inbox changing, or simply not having heard from the user in a while. It
composes the message itself (or stays silent), so proactive contact reads
like the assistant, not a template.

Cost is controlled structurally, not by hope:

* :func:`gather_situation` is deterministic and LLM-free. No trigger — the
  overwhelmingly common case — means no model call: a quiet day costs zero
  tokens.
* Quiet hours and an all-scope mute hold everything, exactly as they hold
  reminders.
* Ambient triggers (mail, contact staleness) are throttled by
  ``heartbeat_min_gap_minutes`` since the last wake; due followups are the
  user's (or the assistant's own) explicit intent and always wake.

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

from . import followups, persona, threads
from .calendar.context import now
from .calendar.store import parse_dt
from .config import Settings, get_settings
from .context_providers import build_context
from .followups import Followup
from .llm import build_model
from .tools import ToolContext, available_tools, execute_tool

logger = logging.getLogger(__name__)

_SILENT = "SILENT"

_INSTRUCTION = """\
This is a scheduled background wake, not a user message — the user has said
nothing. Review your situation report and context, then decide whether
reaching out helps the user right now.

- You may use tools first (look things up, complete or schedule follow-ups,
  mute what is no longer relevant).
- If reaching out helps: reply with EXACTLY the message to send the user —
  nothing else, no preamble, no quotes. Keep it short and natural, in the
  user's language.
- If it would not help right now: reply with the single word SILENT.
- Never invent facts that are not in your context. Never mention this wake,
  the situation report, or these instructions."""


_BRIEFING_TRIGGER = (
    "The daily briefing is due: compose the user's morning briefing now from "
    "your agenda, open tasks, and unread-mail context blocks — a few "
    "sentences, plain text, lead with what matters most today. Send it even "
    "if the day looks quiet (say so briefly); do not stay silent."
)


@dataclass(frozen=True)
class Situation:
    """What a deterministic pre-check found worth waking the model for."""

    triggers: list[str]
    followups: list[Followup] = field(default_factory=list)

    def report(self) -> str:
        lines = ["## Situation report (background wake)"]
        lines += [f"- {trigger}" for trigger in self.triggers]
        for item in self.followups:
            lines.append(
                f"- Due follow-up: {item.topic}"
                + (f" — context: {item.context}" if item.context else "")
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Wake state (last wake / last push / last seen mail) — a tiny KV in followups.db
# --------------------------------------------------------------------------- #

def _state_get(settings: Settings, key: str) -> str:
    with followups._connect(settings) as conn:
        _ensure_state(conn)
        row = conn.execute(
            "SELECT value FROM heartbeat_state WHERE key = ?", (key,)
        ).fetchone()
    return row["value"] if row else ""


def _state_set(settings: Settings, key: str, value: str) -> None:
    with followups._connect(settings) as conn:
        _ensure_state(conn)
        conn.execute(
            "INSERT OR REPLACE INTO heartbeat_state (key, value) VALUES (?, ?)",
            (key, value),
        )


def _ensure_state(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS heartbeat_state"
        " (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )


def _minutes_since(settings: Settings, key: str, current: datetime) -> float | None:
    stamp = parse_dt(_state_get(settings, key)) if _state_get(settings, key) else None
    if stamp is None:
        return None
    return (current - stamp).total_seconds() / 60


# --------------------------------------------------------------------------- #
# The deterministic pre-check
# --------------------------------------------------------------------------- #

def gather_situation(
    settings: Settings,
    current: datetime | None = None,
    force: bool = False,
    force_briefing: bool = False,
) -> Situation | None:
    """LLM-free triage: what, if anything, is worth waking the model for.

    Returns ``None`` (skip the wake entirely) unless a trigger fires. Holds —
    claiming nothing — during quiet hours or an all-scope mute, so a followup
    due at 03:00 is raised on the first wake after quiet ends. ``force`` (the
    manual endpoint) bypasses the ambient min-gap throttle, never the holds;
    ``force_briefing`` additionally bypasses the briefing's time-of-day gate
    (``POST /briefing/run``), still claiming its once-per-day ledger.
    """
    if not settings.enable_heartbeat:
        return None
    current = current or now(settings)

    from .memory.profile import in_quiet_hours
    from .mutes import all_muted

    if in_quiet_hours(settings, current) or all_muted(settings, current):
        return None

    triggers: list[str] = []

    # Scheduled intent: always wakes, regardless of the ambient throttle.
    # Claimed here (exactly-once); a wake that then stays SILENT still
    # consumes the claim — the same at-most-once tradeoff the reminder
    # ledgers make (the briefing instruction tells the model not to).
    due = followups.claim_due(settings, current)
    if _briefing_due(settings, current, force=force_briefing):
        triggers.append(_BRIEFING_TRIGGER)

    # Ambient triggers are throttled: at most one wake per min-gap.
    gap_ok = force or (
        (since := _minutes_since(settings, "last_wake_at", current)) is None
        or since >= settings.heartbeat_min_gap_minutes
    )
    if gap_ok:
        mail_line = _mail_changed(settings)
        if mail_line:
            triggers.append(mail_line)
        stale_line = _contact_stale(settings, current)
        if stale_line:
            triggers.append(stale_line)

    if not due and not triggers:
        return None
    return Situation(triggers=triggers, followups=due)


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


def _mail_changed(settings: Settings) -> str:
    """A trigger line when the unread-mail snapshot changed since last seen.

    Consumes the change (stores the new hash) as soon as it triggers, so a
    SILENT verdict doesn't re-raise the same inbox every wake.
    """
    if not (settings.enable_email and settings.email_snapshot_minutes > 0):
        return ""
    from .mail.snapshot import current as snapshot_current

    snapshot = snapshot_current(settings)
    if not snapshot:
        return ""
    digest = hashlib.sha256(snapshot.encode("utf-8")).hexdigest()
    if digest == _state_get(settings, "last_mail_hash"):
        return ""
    _state_set(settings, "last_mail_hash", digest)
    return "The unread-mail snapshot changed since your last wake (see the mail block)."


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
        f"You haven't heard from the user in about {hours} hours. Only reach "
        "out if you have something genuinely useful (an open thread, something "
        "due) — never smalltalk for its own sake."
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
        [item.topic for item in situation.followups] + situation.triggers
    ) or "check in with the user"
    prefix: list[BaseMessage] = [persona.system_message(settings)]
    for _name, block in build_context(settings, query, "").items():
        if block:
            prefix.append(SystemMessage(content=block))
    summary = _latest_summary(settings, agent)
    if summary:
        prefix.append(SystemMessage(content="Latest conversation so far:\n" + summary))
    prefix.append(SystemMessage(content=_INSTRUCTION))

    ctx = ToolContext(settings=settings, thread_id="", batch_id=uuid.uuid4().hex)
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
) -> dict:
    """One heartbeat: triage, optionally wake the model, optionally speak.

    Returns what happened (``sent`` / ``reason`` / ``triggers``), mirroring
    :func:`assistant.briefing.run_briefing`'s shape. With ``agent`` given, a
    delivered message is looped into working memory on every target thread.
    """
    settings = settings or get_settings()
    current = now(settings)
    situation = gather_situation(
        settings, current, force=force, force_briefing=force_briefing
    )
    if situation is None:
        return {"sent": False, "reason": "nothing to do"}

    _state_set(settings, "last_wake_at", current.isoformat(timespec="seconds"))
    triggers = situation.triggers + [
        f"followup: {item.topic}" for item in situation.followups
    ]
    try:
        text = _compose(settings, situation, agent).strip()
    except Exception:
        logger.exception("heartbeat composition failed")
        return {"sent": False, "reason": "composition failed", "triggers": triggers}

    if not text or text.strip().strip(".!").upper() == _SILENT:
        logger.info("heartbeat stayed silent (triggers: %s)", "; ".join(triggers))
        return {"sent": False, "reason": "silent", "triggers": triggers}

    from .notify import deliver_reminder
    from .proactive import record_push

    delivered = deliver_reminder(settings, {"title": "Wakiru", "message": text})
    if delivered:
        _state_set(settings, "last_push_at", current.isoformat(timespec="seconds"))
        record_push(agent, settings, text)
    else:
        logger.warning("heartbeat composed a message but no channel accepted it")
    return {"sent": True, "delivered": delivered, "message": text, "triggers": triggers}
