"""Registry + dispatch: assemble the enabled tools for a turn and run them."""
from __future__ import annotations

from dataclasses import replace

from ..config import Settings
from ._base import ToolContext, ToolSpec, logger
from .calendar import _calendar_tools
from .code import _code_tools
from .docs import _docs_tools, _web_ingest_tools, _web_tools
from .email import _email_tools, _mail_mutated
from .expenses import _expense_tools
from .followups import _followup_tools
from .goals import _goal_tools
from .habits import _habit_tools
from .lists import _lists_tools
from .memory import _memory_tools
from .people import _people_tools
from .reading import _reading_tools
from .reminders import _reminder_tools
from .subscriptions import _subscription_tools
from .tasks import _task_tools
from .trips import _trip_tools
from .undo import _undo_tools
from .wake import _wake_tools
from .watches import _chat_only_feed, _watch_tools
from .weather import _weather_tools

_MAIL_MUTATING = frozenset(
    {"reply_email", "archive_email", "mark_email_read", "label_email"}
)


def _budgeted(spec: ToolSpec, budget: dict[str, int]) -> ToolSpec:
    """Cap a mutating mail tool to the heartbeat's per-wake triage budget.

    The registry is rebuilt on every wake, so the shared counter is naturally
    per-wake. Only performed mutations consume budget — a "no message with
    uid" miss or a refusal does not. The ceiling holds structurally, whatever
    the prompt says.
    """
    inner = spec.run

    def run(ctx: ToolContext, **args: object) -> str:
        if budget["n"] <= 0:
            return (
                "Tool failed: the mailbox triage budget for this wake is used "
                "up. Leave the rest of the inbox for the next wake."
            )
        result = inner(ctx, **args)
        if _mail_mutated(result):
            budget["n"] -= 1
        return result

    return replace(spec, run=run)


def available_tools(settings: Settings, mode: str = "chat") -> list[ToolSpec]:
    """Every tool the current configuration offers the model.

    ``mode="heartbeat"`` is the background variant: ``send_email`` and
    ``send_reply`` are structurally absent — no prompt, bug, or jailbreak can
    make a background wake send mail — and so is ``undo``, since a background
    wake has no conversation whose latest write it could revert. The mutating
    mail tools (archive / label / mark-read / draft-reply) are absent too
    unless ``email_triage_max_actions`` opts in, and then they share a
    per-wake mutation budget.
    """
    tools: list[ToolSpec] = []
    if settings.enable_calendar:
        tools += _calendar_tools()
    if settings.enable_tasks:
        tools += _task_tools()
    if settings.enable_people:
        tools += _people_tools()
    if settings.enable_reading:
        tools += _reading_tools()
    if settings.enable_lists:
        tools += _lists_tools()
    if settings.enable_trips:
        tools += _trip_tools()
    if settings.enable_habits:
        tools += _habit_tools()
    if settings.enable_subscriptions:
        tools += _subscription_tools()
    if settings.enable_expenses:
        tools += _expense_tools()
    if settings.enable_weather:
        tools += _weather_tools()
    if settings.enable_reminders:
        tools += _reminder_tools()
    if settings.enable_write_confirmation and (
        settings.enable_calendar or settings.enable_tasks
    ):
        tools += _undo_tools()
    tools += _memory_tools()
    if settings.enable_docs:
        tools += _docs_tools()
    if settings.enable_docs_url_ingest:
        # The same opt-in that authorizes server-side URL fetching for the
        # /documents endpoint. Both web tools are chat-only (excluded below):
        # a page's text is arbitrary-origin, and an unattended background wake
        # holding write tools must not read attacker-controllable instructions.
        tools += _web_tools()
        if settings.enable_docs:
            tools += _web_ingest_tools()
    if settings.enable_email:
        tools += _email_tools(settings)
    if settings.enable_code_execution:
        # Offered in both chat and heartbeat: the sandbox has no access to the
        # user's data or secrets, so an unattended wake computing on its own
        # initiative is no more dangerous than a chat call.
        tools += _code_tools()
    if settings.enable_heartbeat:
        tools += _followup_tools()
        tools += _goal_tools()
        tools += _watch_tools()
    if mode == "heartbeat":
        # set_next_wake is background-only: in chat, "wake me before X" is a
        # follow-up. The send exclusion is untouched below it. Ingest and
        # whole-document summarize stay chat-only too: a background wake should
        # not grow docs.db or spend a map-reduce of LLM calls unprompted.
        tools += _wake_tools()
        tools = [
            spec
            for spec in tools
            if spec.name not in (
                "send_email", "send_reply", "undo",
                "ingest_attachment", "summarize_document", "save_note",
                "read_url", "ingest_url",
                # On-demand weather does a network fetch; like the web tools it
                # is chat-only — a background wake must not fetch arbitrary places.
                "get_weather",
                # People writes are chat-only: a background wake surfaces who is
                # due (see heartbeat) and composes outreach, but does not mutate
                # the CRM unattended.
                "add_person", "update_person", "remove_person", "log_contact",
                # Reading-list writes are chat-only too — nothing a background
                # wake should be saving or pruning on its own.
                "save_reading", "mark_read", "remove_reading",
                # Checklist writes likewise: the user says what goes on a list;
                # show_list stays readable for briefing enrichment.
                "add_to_list", "check_off_item", "remove_from_list",
                # Trip writes are chat-only; list_trips stays readable so a
                # wake can reason about imminent travel.
                "add_trip", "update_trip", "remove_trip",
                # Subscription writes are chat-only; the heartbeat only fires
                # renewal reminders, it does not edit what's tracked.
                "add_subscription", "update_subscription", "remove_subscription",
                # Habit writes are chat-only — the user logs what they did; a
                # background wake has nothing to record on their behalf.
                "log_habit", "remove_habit_entry",
                # Expense writes likewise — only the user knows what they
                # spent; expense_summary stays readable for rollup questions.
                "log_expense", "remove_expense",
            )
        ]
        tools = [
            _chat_only_feed(spec) if spec.name == "watch" else spec for spec in tools
        ]
        if settings.email_triage_max_actions > 0:
            budget = {"n": settings.email_triage_max_actions}
            tools = [
                _budgeted(spec, budget) if spec.name in _MAIL_MUTATING else spec
                for spec in tools
            ]
        else:
            tools = [spec for spec in tools if spec.name not in _MAIL_MUTATING]
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
