"""Undo tool — revert the latest calendar/task write on this conversation."""
from __future__ import annotations

from ._base import ToolContext, ToolSpec, _params


def _undo(ctx: ToolContext) -> str:
    from ..undo import undo_latest

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
