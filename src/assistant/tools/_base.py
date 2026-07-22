"""Shared primitives for the tools package: the tool records, the per-turn
context, and the tiny arg/op helpers every family builder leans on."""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from ..config import Settings

logger = logging.getLogger(__name__)

# What the model sees when a mutating op couldn't resolve its target at all
# (no match, or the single match was vetoed, e.g. an ICS-mirrored calendar
# row). A resolvable-but-ambiguous match now returns its own descriptive
# message from apply_op instead of falling through to this generic one.
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
def _op_runner(
    apply: Callable[[ToolContext, dict], str], kind: str
) -> Callable[..., str]:
    def run(ctx: ToolContext, **args: object) -> str:
        op: dict[str, object] = {"op": kind}
        op.update({k: v for k, v in args.items() if v not in (None, "")})
        return apply(ctx, op)

    return run
_ISO = "Absolute ISO-8601 datetime with timezone offset"
def _int_arg(value: object, default: int) -> int | None:
    """A tool's numeric string arg as an int; default when blank, None when junk."""
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(text)
    except ValueError:
        return None
