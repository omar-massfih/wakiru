"""Goal tools — standing multi-step intentions advanced across wakes."""
from __future__ import annotations

from ._base import _ISO, _NO_MATCH, ToolContext, ToolSpec, _params


def _format_goal(ctx: ToolContext, goal) -> str:
    from ..calendar.context import format_when

    when = (
        f" — next step {format_when(ctx.settings, goal.next_action_at)}"
        if goal.next_action_at
        else " — parked (no next step scheduled)"
    )
    return f"{goal.title}{when} (id {goal.id})"

def _ambiguous_goals_message(matches: list) -> str:
    shown = ", ".join(f'{g.id} ("{g.title}")' for g in matches[:5])
    more = f", +{len(matches) - 5} more" if len(matches) > 5 else ""
    return (
        f"Ambiguous — {len(matches)} goals match: {shown}{more}. "
        "Retry with one exact id from Open goals."
    )

def _open_goal(ctx: ToolContext, title: str, state: str = "", next_action: str = "") -> str:
    from .. import goals
    from ..calendar.store import parse_dt

    dupe = goals.find_exact_open_title(ctx.settings, str(title))
    if dupe is not None:
        return (
            f"Not opened — an open goal already has this exact title: "
            f"{dupe.id} (\"{dupe.title}\"). Use update_goal with id {dupe.id} "
            "to change it, or open_goal again with a more distinguishing "
            "title if this is genuinely a separate goal."
        )
    due = None
    if next_action:
        due = parse_dt(str(next_action))
        if due is None:
            return f"Tool failed: next_action must be {_ISO}."
    saved = goals.open_goal(
        ctx.settings, str(title), str(state), due, thread_id=ctx.thread_id
    )
    if saved is None:
        return (
            f"Tool failed: you already carry {ctx.settings.goals_max_open} open "
            "goals. Close or abandon one before opening another."
        )
    return f"Goal opened: {_format_goal(ctx, saved)}"

def _update_goal(
    ctx: ToolContext,
    target: str,
    state: str = "",
    next_action: str = "",
    title: str = "",
    park: bool = False,
) -> str:
    from .. import goals
    from ..calendar.store import parse_dt

    due = None
    if next_action:
        due = parse_dt(str(next_action))
        if due is None:
            return f"Tool failed: next_action must be {_ISO}."
    if not state and due is None and not title and not park:
        return "Tool failed: give at least one of state, next_action, title, or park."
    revised = goals.update(
        ctx.settings,
        str(target),
        state=state or None,
        next_action_at=due,
        title=title or None,
        clear_next_action=bool(park),
    )
    if isinstance(revised, list):
        return _ambiguous_goals_message(revised)
    if revised is None:
        return _NO_MATCH
    return f"Goal updated: {_format_goal(ctx, revised)}"

def _close_goal(ctx: ToolContext, target: str, outcome: str = "", abandoned: bool = False) -> str:
    from .. import goals

    closed = goals.close(ctx.settings, str(target), str(outcome), bool(abandoned))
    if isinstance(closed, list):
        return _ambiguous_goals_message(closed)
    if closed is None:
        return _NO_MATCH
    return f"Goal {closed.status}: {closed.title}"

def _list_goals(ctx: ToolContext) -> str:
    from .. import goals

    open_items = goals.list_open(ctx.settings)
    if not open_items:
        return "No open goals."
    lines = []
    for goal in open_items:
        lines.append(f"- {_format_goal(ctx, goal)}")
        if goal.state:
            lines.append(f"  state: {goal.state}")
    return "\n".join(lines)

def _goal_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            "open_goal",
            "Open a standing goal: an ongoing multi-step project you will "
            "advance across future wakes (research, plan, prepare — not a "
            "one-off check-in; schedule_followup is for those). The state "
            "field is your working document — plan, progress, open questions "
            "— that every future wake and conversation reads. Set "
            "next_action when you (not the user) should attempt the next "
            "step.",
            _params(
                {
                    "title": ("string", "Short goal title, e.g. 'Plan the Oslo trip'"),
                    "state": (
                        "string",
                        "Your plan and current progress — what future-you needs "
                        "to pick this up cold",
                    ),
                    "next_action": (
                        "string",
                        f"When to attempt the next step, {_ISO} (omit to park)",
                    ),
                },
                ["title", "state"],
            ),
            _open_goal,
        ),
        ToolSpec(
            "update_goal",
            "Advance a goal you carry: rewrite its state with what you just "
            "did or learned, and ALWAYS set next_action to when the next step "
            "is worth attempting (or park=true while waiting on the world) — "
            "otherwise the same goal is raised to you again next wake. Target "
            "by id or an unambiguous title reference.",
            _params(
                {
                    "target": ("string", "Goal id or title"),
                    "state": ("string", "Rewritten working state (replaces the old)"),
                    "next_action": ("string", f"Next step time, {_ISO}"),
                    "title": ("string", "New title"),
                    "park": (
                        "boolean",
                        "True to clear the next step — waiting, don't raise me",
                    ),
                },
                ["target"],
            ),
            _update_goal,
        ),
        ToolSpec(
            "close_goal",
            "Close a goal as done — or abandoned=true when it is no longer "
            "worth pursuing — with a one-line outcome. The outcome is "
            "remembered, so say what worked or why it died.",
            _params(
                {
                    "target": ("string", "Goal id or title"),
                    "outcome": ("string", "One line: the result, or why abandoned"),
                    "abandoned": ("boolean", "True to abandon instead of complete"),
                },
                ["target"],
            ),
            _close_goal,
        ),
        ToolSpec(
            "list_goals",
            "List your open goals with their working state.",
            _params({}, []),
            _list_goals,
        ),
    ]
