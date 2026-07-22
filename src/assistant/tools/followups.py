"""Followup tools — the assistant schedules its own future check-ins."""
from __future__ import annotations

from ._base import _ISO, _NO_MATCH, ToolContext, ToolSpec, _params


def _ambiguous_followups_message(matches: list) -> str:
    shown = ", ".join(f'{f.id} ("{f.topic}")' for f in matches[:5])
    more = f", +{len(matches) - 5} more" if len(matches) > 5 else ""
    return (
        f"Ambiguous — {len(matches)} follow-ups match: {shown}{more}. "
        "Retry with one exact id from list_followups."
    )

def _schedule_followup(ctx: ToolContext, when: str, topic: str, context: str = "") -> str:
    from .. import followups
    from ..calendar.context import format_when, now
    from ..calendar.store import parse_dt

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
    from .. import followups

    cancelled = followups.cancel(ctx.settings, str(target))
    if isinstance(cancelled, list):
        return _ambiguous_followups_message(cancelled)
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
    from .. import followups
    from ..calendar.context import format_when, now
    from ..calendar.store import parse_dt

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
    if isinstance(revised, list):
        return _ambiguous_followups_message(revised)
    if revised is None:
        return _NO_MATCH
    return (
        f"Follow-up updated: {revised.topic}"
        f" @ {format_when(ctx.settings, revised.due)} (id {revised.id})"
    )

def _list_followups(ctx: ToolContext) -> str:
    from .. import followups
    from ..calendar.context import format_when

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
            "later, where you need to compose new content when it comes due "
            "(promises you made, an interview, a decision they postponed). "
            "NOT for a plain 'remind me at TIME that X' — that's add_task "
            "with a due time instead. When it comes due "
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
