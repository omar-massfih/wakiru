"""Memory tools — explicit remember/forget/search."""
from __future__ import annotations

from ._base import ToolContext, ToolSpec, _params


def _remember(ctx: ToolContext, content: str, kind: str = "semantic",
              profile: bool = False) -> str:
    from ..memory.learn import save_memory

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
    from ..memory.learn import forget_memory

    deleted = forget_memory(ctx.settings, str(target))
    if isinstance(deleted, list):
        return (
            f"Ambiguous — {len(deleted)} memories are too close to call: "
            f"{', '.join(deleted)}. Use the exact name from the memory index."
        )
    if deleted is None:
        return (
            "No memory matched. Use the exact name from the memory index."
        )
    return f"Forgot: {deleted.description}"

def _search_memory(ctx: ToolContext, query: str) -> str:
    from ..memory.recall import search_memory

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
