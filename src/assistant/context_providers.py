"""Per-turn context assembly — every feature feeds the brain the same way.

The graph used to wire one hand-written node per feature (recall, agenda,
tasks, profile), and anything without a node — email — simply never reached
the model. This registry replaces that: a feature contributes context by
registering a :class:`ContextProvider` (a name, an enabled-flag, and a
provide function), and :func:`build_context` runs every enabled provider for
the turn. Registry order is prompt order.

Providers are isolated: one failing provider logs and contributes nothing,
exactly as the old per-node try/excepts did. A provider must be fast — it runs
on the reply path — so anything slow (network I/O like IMAP) must serve from a
cache refreshed off-path (see :mod:`assistant.mail.snapshot`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from .config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurnContext:
    """What a provider may look at when composing its block."""

    settings: Settings
    query: str  # the expanded recall query for this turn
    thread_id: str


@dataclass(frozen=True)
class ContextProvider:
    """One feature's per-turn contribution to the prompt."""

    name: str  # state key + log label
    enabled: Callable[[Settings], bool]
    provide: Callable[[TurnContext], str]  # "" contributes nothing


def _recall(ctx: TurnContext) -> str:
    """Recalled memories plus the most relevant document excerpts.

    Folded into one block so "what did I write about X" is answered from
    ingested docs as readily as from memory notes.
    """
    from .docs import docs_context
    from .memory import recall_context

    content = recall_context(ctx.settings, ctx.query).content
    if not isinstance(content, str):
        content = str(content)
    if ctx.settings.enable_docs:
        try:
            docs = docs_context(ctx.settings, ctx.query)
        except Exception:
            logger.exception("document recall failed; continuing without it")
            docs = ""
        if docs:
            content = f"{content}\n\n{docs}" if content else docs
    return content


def _profile(ctx: TurnContext) -> str:
    from .memory.profile import profile_context

    return profile_context(ctx.settings)


def _agenda(ctx: TurnContext) -> str:
    from .calendar import agenda_context

    return agenda_context(ctx.settings)


def _tasks(ctx: TurnContext) -> str:
    from .tasks import tasks_context

    return tasks_context(ctx.settings)


def _mail(ctx: TurnContext) -> str:
    """The cached unread-mail snapshot — never live IMAP on the reply path."""
    from .mail.snapshot import current

    return current(ctx.settings)


def _weather(ctx: TurnContext) -> str:
    """The cached weather forecast — never an outbound fetch on the reply path."""
    from .weather import current

    return current(ctx.settings)


def _trip(ctx: TurnContext) -> str:
    """The active or imminent trip — empty (and free) between travels."""
    from .trips import trips_context

    return trips_context(ctx.settings)


def _worklog(ctx: TurnContext) -> str:
    """The running work timer — empty (and free) whenever the clock is off."""
    from .worklog.context import timer_context

    return timer_context(ctx.settings)


def _people(ctx: TurnContext) -> str:
    """The compact people roster, with anyone due for contact / a birthday soon
    flagged first — so "who is this with?" and "reach out to X" both work."""
    from .people import people_context

    return people_context(ctx.settings)


def _goals(ctx: TurnContext) -> str:
    """The standing goals the assistant carries — the same intentions the
    heartbeat advances, so a chat turn about one picks up its working state
    and the user can steer or drop it."""
    from . import goals
    from .calendar.context import format_when

    open_items = goals.list_open(ctx.settings)
    if not open_items:
        return ""
    lines = [
        "## Your standing goals",
        "Ongoing projects you are advancing in the background. If the "
        "conversation touches one, use its state; record progress or changes "
        "of direction with update_goal, and close_goal when it is finished "
        "or the user drops it.",
    ]
    for goal in open_items:
        when = (
            f" — next step {format_when(ctx.settings, goal.next_action_at)}"
            if goal.next_action_at
            else " — parked"
        )
        lines.append(f"- {goal.title}{when} (id {goal.id})")
        if goal.state:
            lines.append(f"  state: {goal.state}")
    return "\n".join(lines)


def default_providers() -> list[ContextProvider]:
    """The standard registry, in prompt order."""
    return [
        ContextProvider("recall", lambda s: True, _recall),
        ContextProvider("profile", lambda s: s.enable_profile, _profile),
        ContextProvider("agenda", lambda s: s.enable_calendar, _agenda),
        ContextProvider("tasks", lambda s: s.enable_tasks, _tasks),
        ContextProvider("worklog", lambda s: s.enable_worklog, _worklog),
        ContextProvider("trip", lambda s: s.enable_trips, _trip),
        ContextProvider("people", lambda s: s.enable_people, _people),
        ContextProvider("goals", lambda s: s.enable_heartbeat, _goals),
        ContextProvider(
            "mail", lambda s: s.enable_email and s.email_snapshot_minutes > 0, _mail
        ),
        ContextProvider(
            "weather",
            lambda s: s.enable_weather and s.weather_refresh_minutes > 0,
            _weather,
        ),
    ]


def build_context(
    settings: Settings,
    query: str,
    thread_id: str,
    providers: list[ContextProvider] | None = None,
) -> dict[str, str]:
    """Run every enabled provider for one turn, each isolated from the others."""
    ctx = TurnContext(settings=settings, query=query, thread_id=thread_id)
    blocks: dict[str, str] = {}
    for provider in providers if providers is not None else default_providers():
        if not provider.enabled(settings):
            continue
        try:
            blocks[provider.name] = provider.provide(ctx)
        except Exception:
            logger.exception("context provider %r failed; continuing without it", provider.name)
            blocks[provider.name] = ""
    return blocks
