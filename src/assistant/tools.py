"""Model-dispatched tools — the assistant's hands.

Each tool wraps an existing, already-guarded write or read path (the same
``apply_op`` functions the background extractors use), so ambiguity guards,
conflict notes, and the undo ledger all keep working unchanged. The model calls
these through the graph's tool loop (:mod:`assistant.agent`); the registry here
only describes and dispatches them.

Two rules the registry enforces structurally rather than by prompt:

* Every tool is gated by its subsystem's enable flag — a disabled capability is
  simply not offered to the model.
* ``send_email`` is registered only when ``enable_email_send`` is set (the
  second, independent switch) **and** only in ``mode="chat"`` — background
  paths (the heartbeat) request ``mode="heartbeat"``, which never contains
  ``send_email``, so mail can never be sent except from a live conversation.

``execute_tool`` never raises: a failure becomes the tool's result string, so
the loop always produces a ``ToolMessage`` and the model can self-correct.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from .config import Settings

logger = logging.getLogger(__name__)

# What the model sees when a mutating op couldn't resolve its target (the
# underlying apply_op returns None for both "not found" and "ambiguous").
_NO_MATCH = (
    "No matching item, or the reference was ambiguous. Check the exact id "
    "against the list you were shown and try again."
)


@dataclass(frozen=True)
class ToolContext:
    """Per-turn execution context threaded into every tool call.

    ``batch_id`` is minted once per user turn so all of a turn's writes share
    one undo batch — replying "undo" reverts the whole turn, as before.
    """

    settings: Settings
    thread_id: str = ""
    batch_id: str = ""


@dataclass(frozen=True)
class ToolSpec:
    """One tool: an OpenAI-format schema plus its implementation."""

    name: str
    description: str
    parameters: dict
    run: Callable[..., str] = field(repr=False)

    def to_openai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _params(props: dict[str, tuple[str, str]], required: list[str]) -> dict:
    """A flat JSON-Schema object — string/boolean args only, no $defs."""
    return {
        "type": "object",
        "properties": {
            name: {"type": json_type, "description": desc}
            for name, (json_type, desc) in props.items()
        },
        "required": required,
    }


# --------------------------------------------------------------------------- #
# Calendar / tasks — thin op builders over the existing apply_op paths
# --------------------------------------------------------------------------- #

def _calendar_op(ctx: ToolContext, op: dict) -> str:
    from .calendar import ops as calendar_ops

    result = calendar_ops.apply_op(ctx.settings, op, ctx.thread_id, ctx.batch_id)
    return result or _NO_MATCH


def _task_op(ctx: ToolContext, op: dict) -> str:
    from .tasks import ops as task_ops

    result = task_ops.apply_op(ctx.settings, op, ctx.thread_id, ctx.batch_id)
    return result or _NO_MATCH


def _op_runner(
    apply: Callable[[ToolContext, dict], str], kind: str
) -> Callable[..., str]:
    def run(ctx: ToolContext, **args: object) -> str:
        op: dict[str, object] = {"op": kind}
        op.update({k: v for k, v in args.items() if v not in (None, "")})
        return apply(ctx, op)

    return run


_ISO = "Absolute ISO-8601 datetime with timezone offset"


def _calendar_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            "create_event",
            "Schedule a new calendar event.",
            _params(
                {
                    "title": ("string", "Short event title"),
                    "start": ("string", _ISO),
                    "end": ("string", f"{_ISO} (omit for a default 1h)"),
                    "location": ("string", "Where"),
                    "notes": ("string", "Free-form notes"),
                    "rrule": ("string", "RFC 5545 RRULE for a repeating event"),
                },
                ["title", "start"],
            ),
            _op_runner(_calendar_op, "create"),
        ),
        ToolSpec(
            "reschedule_event",
            "Change an existing event's time or details (whole series if recurring).",
            _params(
                {
                    "id": ("string", "Exact event id from Upcoming events"),
                    "start": ("string", _ISO),
                    "end": ("string", _ISO),
                    "title": ("string", "New title"),
                    "location": ("string", "New location"),
                    "notes": ("string", "New notes"),
                },
                ["id"],
            ),
            _op_runner(_calendar_op, "reschedule"),
        ),
        ToolSpec(
            "cancel_event",
            "Cancel an event (whole series if recurring).",
            _params({"id": ("string", "Exact event id")}, ["id"]),
            _op_runner(_calendar_op, "cancel"),
        ),
        ToolSpec(
            "skip_occurrence",
            "Drop a single occurrence of a recurring event — e.g. the user is "
            "skipping it today. Also stops that occurrence's remaining reminder "
            "nudges.",
            _params(
                {
                    "id": ("string", "Series id"),
                    "occurrence": ("string", f"{_ISO} of the occurrence to drop"),
                },
                ["id", "occurrence"],
            ),
            _op_runner(_calendar_op, "skip"),
        ),
        ToolSpec(
            "move_occurrence",
            "Move a single occurrence of a recurring event, leaving the series.",
            _params(
                {
                    "id": ("string", "Series id"),
                    "occurrence": ("string", f"{_ISO} of the original occurrence"),
                    "start": ("string", f"New start, {_ISO}"),
                    "end": ("string", _ISO),
                    "title": ("string", "New title for this occurrence"),
                    "location": ("string", "New location for this occurrence"),
                },
                ["id", "occurrence", "start"],
            ),
            _op_runner(_calendar_op, "move"),
        ),
    ]


def _task_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            "add_task",
            "Add a to-do (no fixed meeting time; use the calendar for those).",
            _params(
                {
                    "title": ("string", "Short task title"),
                    "due": ("string", f"Optional due date, {_ISO}"),
                    "notes": ("string", "Free-form notes"),
                },
                ["title"],
            ),
            _op_runner(_task_op, "add"),
        ),
        ToolSpec(
            "complete_task",
            "Mark a task done (also stops its reminder nagging).",
            _params({"id": ("string", "Exact task id from Open tasks")}, ["id"]),
            _op_runner(_task_op, "complete"),
        ),
        ToolSpec(
            "update_task",
            "Change a task's title, due date, or notes.",
            _params(
                {
                    "id": ("string", "Exact task id"),
                    "title": ("string", "New title"),
                    "due": ("string", f"New due date, {_ISO}"),
                    "notes": ("string", "New notes"),
                },
                ["id"],
            ),
            _op_runner(_task_op, "update"),
        ),
        ToolSpec(
            "remove_task",
            "Delete a task without completing it.",
            _params({"id": ("string", "Exact task id")}, ["id"]),
            _op_runner(_task_op, "remove"),
        ),
    ]


# --------------------------------------------------------------------------- #
# Reminder mutes — silence nudges without touching the calendar or tasks
# --------------------------------------------------------------------------- #

def _resolve_mute_target(settings: Settings, target: str) -> tuple[str, str, str] | None:
    """Resolve ``target`` to ``(scope, target_id, label)``; None when no match
    or ambiguous (the same refuse-don't-guess rule as calendar._target_id)."""
    target = str(target).strip()
    if target.lower() == "all":
        return ("all", "", "all reminders")
    from .calendar import store as calendar_store
    from .tasks import store as tasks_store

    events = calendar_store.find_events(settings, target)
    if len(events) == 1:
        return ("event", events[0].id, events[0].title)
    if len(events) > 1:
        return None
    tasks = tasks_store.find_tasks(settings, target)
    if len(tasks) == 1:
        return ("task", tasks[0].id, tasks[0].title)
    return None


def _mute_reminders(ctx: ToolContext, target: str, until: str = "", reason: str = "") -> str:
    from .calendar.context import format_when, now
    from .calendar.store import parse_dt
    from .mutes import set_mute

    resolved = _resolve_mute_target(ctx.settings, target)
    if resolved is None:
        return _NO_MATCH
    scope, target_id, label = resolved
    current = now(ctx.settings)
    if until:
        expiry = parse_dt(str(until))
        if expiry is None:
            return f"Tool failed: until must be {_ISO}."
        if expiry <= current:
            return "Tool failed: until is already in the past."
    else:
        # The ergonomic default: quiet for the rest of today (local time).
        expiry = current.replace(hour=23, minute=59, second=59, microsecond=0)
    set_mute(ctx.settings, scope, target_id, expiry, str(reason), current)
    return f"Muted reminders for {label} until {format_when(ctx.settings, expiry.isoformat())}."


def _unmute_reminders(ctx: ToolContext, target: str) -> str:
    from .mutes import clear_mute

    resolved = _resolve_mute_target(ctx.settings, target)
    if resolved is None:
        return _NO_MATCH
    scope, target_id, label = resolved
    if clear_mute(ctx.settings, scope, target_id):
        return f"Unmuted reminders for {label}."
    return f"No active mute for {label}."


def _reminder_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            "mute_reminders",
            "Silence reminder nudges for one event, one task, or everything "
            '(target="all") until a time, without changing the calendar or '
            "tasks. Use when the user declines a reminder or asks for quiet.",
            _params(
                {
                    "target": ("string", 'Event/task id or title, or "all"'),
                    "until": ("string", f"{_ISO} the mute expires (omit = rest of today)"),
                    "reason": ("string", 'Why, e.g. "user is sick"'),
                },
                ["target"],
            ),
            _mute_reminders,
        ),
        ToolSpec(
            "unmute_reminders",
            "Lift a reminder mute so nudges resume.",
            _params(
                {"target": ("string", 'Event/task id or title, or "all"')},
                ["target"],
            ),
            _unmute_reminders,
        ),
    ]


# --------------------------------------------------------------------------- #
# Memory — explicit remember/forget/search (implicit learning stays background)
# --------------------------------------------------------------------------- #

def _remember(ctx: ToolContext, content: str, kind: str = "semantic",
              profile: bool = False) -> str:
    from .memory.learn import save_memory

    if kind not in ("semantic", "procedural"):
        kind = "semantic"
    note = save_memory(
        ctx.settings,
        body=str(content),
        kind=kind,
        source=ctx.thread_id,
        tags=["profile"] if profile else None,
    )
    return f"Saved: {note.description}"


def _forget(ctx: ToolContext, target: str) -> str:
    from .memory.learn import forget_memory

    deleted = forget_memory(ctx.settings, str(target))
    if deleted is None:
        return (
            "No memory matched (or the match was ambiguous). "
            "Use the exact name from the memory index."
        )
    return f"Forgot: {deleted.description}"


def _search_memory(ctx: ToolContext, query: str) -> str:
    from .memory.recall import search_memory

    results = search_memory(ctx.settings, str(query))
    if not results:
        return "No relevant memories."
    return "\n".join(f"- {note.name} [{note.kind}]: {note.body}" for note, _ in results)


def _memory_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            "remember",
            "Save a durable fact, preference, or how-to the user asked you to remember.",
            _params(
                {
                    "content": ("string", "One clear third-person sentence"),
                    "kind": ("string", '"semantic" (facts) or "procedural" (how-to)'),
                    "profile": ("boolean", "True if it describes how the user lives/works"),
                },
                ["content"],
            ),
            _remember,
        ),
        ToolSpec(
            "forget",
            "Delete a stored memory the user asked you to forget.",
            _params(
                {"target": ("string", "Exact memory name, or a description of it")},
                ["target"],
            ),
            _forget,
        ),
        ToolSpec(
            "search_memory",
            "Search long-term memory beyond what was auto-recalled this turn.",
            _params({"query": ("string", "What to look for")}, ["query"]),
            _search_memory,
        ),
    ]


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #

def _search_documents(ctx: ToolContext, query: str) -> str:
    from .docs import store as docs_store

    chunks = docs_store.search_chunks(ctx.settings, str(query))
    if not chunks:
        return "No matching document passages."
    return "\n\n".join(f"From “{c.doc_title}”:\n{c.text}" for c in chunks)


def _docs_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            "search_documents",
            "Search the user's ingested documents and notes for relevant passages.",
            _params({"query": ("string", "What to look for")}, ["query"]),
            _search_documents,
        ),
    ]


# --------------------------------------------------------------------------- #
# Email — read/draft when enabled; send only behind the second switch
# --------------------------------------------------------------------------- #

def _list_email(ctx: ToolContext, unread_only: bool = True) -> str:
    from .mail import client as mail_client

    messages = mail_client.list_recent(ctx.settings, unread_only=bool(unread_only))
    if not messages:
        return "No messages." if not unread_only else "No unread messages."
    return "\n".join(
        f"- [{m.uid}] {'(unread) ' if m.unread else ''}{m.sender} — {m.subject} ({m.date})"
        for m in messages
    )


def _read_email(ctx: ToolContext, uid: str) -> str:
    from .mail import client as mail_client

    message = mail_client.read_message(ctx.settings, str(uid))
    if message is None:
        return f"No message with uid {uid}."
    return (
        f"From: {message.sender}\nSubject: {message.subject}\n"
        f"Date: {message.date}\n\n{message.body}"
    )


def _draft_email(ctx: ToolContext, to: str, subject: str, body: str) -> str:
    from .mail import client as mail_client

    return mail_client.save_draft(ctx.settings, str(to), str(subject), str(body))


def _send_email(ctx: ToolContext, to: str, subject: str, body: str) -> str:
    from .mail import client as mail_client

    return mail_client.send_message(ctx.settings, str(to), str(subject), str(body))


def _email_tools(settings: Settings) -> list[ToolSpec]:
    tools = [
        ToolSpec(
            "list_email",
            "List recent mailbox messages (never marks anything read).",
            _params(
                {"unread_only": ("boolean", "Only unread messages (default true)")},
                [],
            ),
            _list_email,
        ),
        ToolSpec(
            "read_email",
            "Read one message's body by uid.",
            _params({"uid": ("string", "Message uid from list_email")}, ["uid"]),
            _read_email,
        ),
        ToolSpec(
            "draft_email",
            "Save an email draft to the drafts folder (does not send).",
            _params(
                {
                    "to": ("string", "Recipient address"),
                    "subject": ("string", "Subject line"),
                    "body": ("string", "Plain-text body"),
                },
                ["to", "subject", "body"],
            ),
            _draft_email,
        ),
    ]
    if settings.enable_email_send:
        tools.append(
            ToolSpec(
                "send_email",
                "Send an email. Only after the user explicitly confirmed sending "
                "this exact message in this conversation.",
                _params(
                    {
                        "to": ("string", "Recipient address"),
                        "subject": ("string", "Subject line"),
                        "body": ("string", "Plain-text body"),
                    },
                    ["to", "subject", "body"],
                ),
                _send_email,
            )
        )
    return tools


# --------------------------------------------------------------------------- #
# Followups — the assistant schedules its own future check-ins
# --------------------------------------------------------------------------- #

def _schedule_followup(ctx: ToolContext, when: str, topic: str, context: str = "") -> str:
    from . import followups
    from .calendar.context import format_when, now
    from .calendar.store import parse_dt

    due = parse_dt(str(when))
    if due is None:
        return f"Tool failed: when must be {_ISO}."
    if due <= now(ctx.settings):
        return "Tool failed: when is already in the past."
    saved = followups.add(
        ctx.settings, due, str(topic), str(context), thread_id=ctx.thread_id
    )
    return (
        f"Follow-up scheduled: {saved.topic}"
        f" @ {format_when(ctx.settings, saved.due)} (id {saved.id})"
    )


def _cancel_followup(ctx: ToolContext, target: str) -> str:
    from . import followups

    cancelled = followups.cancel(ctx.settings, str(target))
    if cancelled is None:
        return _NO_MATCH
    return f"Cancelled follow-up: {cancelled.topic}"


def _update_followup(
    ctx: ToolContext,
    target: str,
    when: str = "",
    topic: str = "",
    context: str = "",
) -> str:
    from . import followups
    from .calendar.context import format_when, now
    from .calendar.store import parse_dt

    due = None
    if when:
        due = parse_dt(str(when))
        if due is None:
            return f"Tool failed: when must be {_ISO}."
        if due <= now(ctx.settings):
            return "Tool failed: when is already in the past."
    if due is None and not topic and not context:
        return "Tool failed: give at least one of when, topic, or context to change."
    revised = followups.update(
        ctx.settings,
        str(target),
        due=due,
        topic=topic or None,
        context=context or None,
    )
    if revised is None:
        return _NO_MATCH
    return (
        f"Follow-up updated: {revised.topic}"
        f" @ {format_when(ctx.settings, revised.due)} (id {revised.id})"
    )


def _list_followups(ctx: ToolContext) -> str:
    from . import followups
    from .calendar.context import format_when

    open_items = followups.list_open(ctx.settings)
    if not open_items:
        return "No follow-ups scheduled."
    return "\n".join(
        f"- {f.topic} @ {format_when(ctx.settings, f.due)} (id {f.id})"
        for f in open_items
    )


def _followup_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            "schedule_followup",
            "Schedule yourself to check in with the user about something "
            "later. Use for 'remind me to ask …', promises you made, or "
            "things you decide are worth following up on. When it comes due "
            "you will be woken to compose the check-in yourself. The context "
            "field is your note-to-self that every future wake reads, so keep "
            "your working state there (\"waiting for their reply\", \"step 2 "
            "of 3: draft done\") and revise it with update_followup as things "
            "move.",
            _params(
                {
                    "when": ("string", _ISO),
                    "topic": ("string", "What to follow up about (one line)"),
                    "context": (
                        "string",
                        "What future-you needs to know to write a good check-in",
                    ),
                },
                ["when", "topic"],
            ),
            _schedule_followup,
        ),
        ToolSpec(
            "update_followup",
            "Revise an open follow-up you are carrying: push its due time out, "
            "reword the topic, or (most often) update its context with what you "
            "just learned. Target it by id or an unambiguous topic reference. "
            "Give at least one of when/topic/context to change.",
            _params(
                {
                    "target": ("string", "Follow-up id or topic"),
                    "when": ("string", "New due time — " + _ISO),
                    "topic": ("string", "New one-line topic"),
                    "context": ("string", "Revised note-to-self for future wakes"),
                },
                ["target"],
            ),
            _update_followup,
        ),
        ToolSpec(
            "cancel_followup",
            "Cancel a scheduled follow-up by id or an unambiguous topic reference.",
            _params({"target": ("string", "Follow-up id or topic")}, ["target"]),
            _cancel_followup,
        ),
        ToolSpec(
            "list_followups",
            "List your scheduled follow-ups.",
            _params({}, []),
            _list_followups,
        ),
    ]


# --------------------------------------------------------------------------- #
# Undo — revert the latest calendar/task write on this conversation
# --------------------------------------------------------------------------- #

def _undo(ctx: ToolContext) -> str:
    from .undo import undo_latest

    return undo_latest(
        ctx.settings, ctx.thread_id, ctx.settings.write_undo_window_minutes
    )


def _undo_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            "undo",
            "Revert the user's most recent calendar or task write in this "
            "conversation. Call it when they ask to undo, revert, or take "
            "back your latest change; the result says what was reverted.",
            _params({}, []),
            _undo,
        )
    ]


# --------------------------------------------------------------------------- #
# Registry + dispatch
# --------------------------------------------------------------------------- #

def available_tools(settings: Settings, mode: str = "chat") -> list[ToolSpec]:
    """Every tool the current configuration offers the model.

    ``mode="heartbeat"`` is the background variant: identical except that
    ``send_email`` is structurally absent — no prompt, bug, or jailbreak can
    make a background wake send mail — and so is ``undo``, since a background
    wake has no conversation whose latest write it could revert.
    """
    tools: list[ToolSpec] = []
    if settings.enable_calendar:
        tools += _calendar_tools()
    if settings.enable_tasks:
        tools += _task_tools()
    if settings.enable_reminders:
        tools += _reminder_tools()
    if settings.enable_write_confirmation and (
        settings.enable_calendar or settings.enable_tasks
    ):
        tools += _undo_tools()
    tools += _memory_tools()
    if settings.enable_docs:
        tools += _docs_tools()
    if settings.enable_email:
        tools += _email_tools(settings)
    if settings.enable_heartbeat:
        tools += _followup_tools()
    if mode == "heartbeat":
        tools = [spec for spec in tools if spec.name not in ("send_email", "undo")]
    return tools


def tool_map(settings: Settings) -> dict[str, ToolSpec]:
    return {spec.name: spec for spec in available_tools(settings)}


def execute_tool(spec: ToolSpec, ctx: ToolContext, args: dict) -> str:
    """Run one tool call; any failure becomes the result string, never a raise."""
    if not isinstance(args, dict):
        args = {}
    known = spec.parameters.get("properties", {})
    missing = [name for name in spec.parameters.get("required", []) if not args.get(name)]
    if missing:
        return f"Tool failed: missing required argument(s): {', '.join(missing)}."
    kwargs = {k: v for k, v in args.items() if k in known}
    try:
        return spec.run(ctx, **kwargs)
    except Exception as exc:
        logger.exception("tool %s failed", spec.name)
        return f"Tool failed: {exc}"
