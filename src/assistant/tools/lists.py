"""Checklist tools — add/show/check-off/remove over the named-lists store."""
from __future__ import annotations

from ._base import ToolContext, ToolSpec, _params


def _split_items(raw: str) -> list[str]:
    """One item per line or comma — 'milk, eggs, bread' is three entries."""
    parts: list[str] = []
    for line in str(raw).splitlines():
        parts += [p.strip() for p in line.split(",")]
    return [p for p in parts if p]


def _render_entry(entry) -> str:
    box = "[x]" if entry.done else "[ ]"
    return f"- {box} {entry.item}  [id: {entry.id}]"


def _render_list(ctx: ToolContext, name: str, include_done: bool = False) -> str:
    from ..lists import store

    entries = store.list_items(ctx.settings, name, include_done=include_done)
    display = store.canonical_name(ctx.settings, name)
    if not entries:
        return f"The {display} list is empty."
    return f"{display}:\n" + "\n".join(_render_entry(e) for e in entries)


def _add_to_list(ctx: ToolContext, **args: object) -> str:
    from ..lists import store

    name = str(args.get("list", "")).strip()
    if not name:
        return "Tool failed: a list name is required."
    items = _split_items(str(args.get("items", "")))
    if not items:
        return "Tool failed: at least one item is required."
    added = [store.add_item(ctx.settings, name, item) for item in items]
    display = added[0].list_name
    return f"Added to the {display} list: " + ", ".join(e.item for e in added)


def _show_list(ctx: ToolContext, **args: object) -> str:
    from ..lists import store

    name = str(args.get("list", "")).strip()
    include_done = str(args.get("include_done", "")).strip().lower() in ("1", "true", "yes")
    if name:
        return _render_list(ctx, name, include_done)
    names = store.list_names(ctx.settings)
    if not names:
        return "There are no lists yet."
    return "\n\n".join(_render_list(ctx, n, include_done) for n, _open in names)


def _check_off_item(ctx: ToolContext, **args: object) -> str:
    from ..lists import store

    query = str(args.get("item", "")).strip()
    if not query:
        return "Tool failed: an item (text or id) is required."
    entry = store.find_item(ctx.settings, query, str(args.get("list", "") or ""))
    if entry is None:
        return f"No list item matches {query!r}."
    updated = store.set_done(ctx.settings, entry.id)
    return (
        f"Checked off “{updated.item}” on the {updated.list_name} list."
        if updated
        else f"No item with id {entry.id}."
    )


def _remove_from_list(ctx: ToolContext, **args: object) -> str:
    from ..lists import store

    query = str(args.get("item", "")).strip()
    if not query:
        return "Tool failed: an item (text or id) is required."
    entry = store.find_item(ctx.settings, query, str(args.get("list", "") or ""))
    if entry is None:
        return f"No list item matches {query!r}."
    removed = store.delete_item(ctx.settings, entry.id)
    return (
        f"Removed “{removed.item}” from the {removed.list_name} list."
        if removed
        else "Nothing removed."
    )


def _lists_tools() -> list[ToolSpec]:
    _scope = "Which list it's on (only needed when the item text is ambiguous)"
    return [
        ToolSpec(
            "add_to_list",
            "Add item(s) to a named checklist (shopping, errands, packing …). "
            "The list is created the first time it's named.",
            _params(
                {
                    "list": ("string", "The list's name, e.g. \"shopping\""),
                    "items": ("string", "One item, or several separated by commas/newlines"),
                },
                ["list", "items"],
            ),
            _add_to_list,
        ),
        ToolSpec(
            "show_list",
            "Show a checklist's open items — or every list when no name is given.",
            _params(
                {
                    "list": ("string", "The list to show (omit for all lists)"),
                    "include_done": ("string", "\"true\" to include checked-off items"),
                },
                [],
            ),
            _show_list,
        ),
        ToolSpec(
            "check_off_item",
            "Check an item off a list (bought / done). It leaves the visible list "
            "but stays recoverable.",
            _params(
                {
                    "item": ("string", "The item's text or exact id"),
                    "list": ("string", _scope),
                },
                ["item"],
            ),
            _check_off_item,
        ),
        ToolSpec(
            "remove_from_list",
            "Delete an item from a list outright (added by mistake).",
            _params(
                {
                    "item": ("string", "The item's text or exact id"),
                    "list": ("string", _scope),
                },
                ["item"],
            ),
            _remove_from_list,
        ),
    ]
