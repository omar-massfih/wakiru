"""Reading-list tools — save/list/mark-read/remove over the read-it-later store."""
from __future__ import annotations

from ._base import ToolContext, ToolSpec, _params


def _render_item(item, with_id: bool = True) -> str:
    line = f"- {item.title}"
    if item.url and item.url != item.title:
        line += f" <{item.url}>"
    if item.note:
        line += f" — {item.note}"
    if with_id:
        line += f"  [id: {item.id}]"
    return line


def _save_reading(ctx: ToolContext, **args: object) -> str:
    from ..reading import store

    url = str(args.get("url", "")).strip()
    if not url:
        return "Tool failed: a url is required."
    if not (url.startswith("http://") or url.startswith("https://")):
        return f"Tool failed: {url!r} is not an http(s) URL."
    item = store.create_item(
        ctx.settings,
        url=url,
        title=str(args.get("title", "") or ""),
        note=str(args.get("note", "") or ""),
    )
    return f"Saved to your reading list: {item.title}"


def _list_reading(ctx: ToolContext, **args: object) -> str:
    from ..reading import store

    include_read = str(args.get("include_read", "")).strip().lower() in ("1", "true", "yes")
    items = store.list_items(ctx.settings, include_read=include_read)
    if not include_read:
        items = items[: ctx.settings.reading_max_open]
    if not items:
        return "Your reading list is empty."
    header = "Reading list:" if include_read else "Unread in your reading list:"
    return header + "\n" + "\n".join(_render_item(i) for i in items)


def _mark_read(ctx: ToolContext, **args: object) -> str:
    from ..reading import store

    query = str(args.get("query", "")).strip()
    if not query:
        return "Tool failed: a url, title, or id is required."
    item = store.find_item(ctx.settings, query)
    if item is None:
        return f"No reading-list item matches {query!r}."
    updated = store.mark_read(ctx.settings, item.id)
    return f"Marked read: {updated.title}" if updated else f"No item with id {item.id}."


def _remove_reading(ctx: ToolContext, **args: object) -> str:
    from ..reading import store

    query = str(args.get("query", "")).strip()
    if not query:
        return "Tool failed: a url, title, or id is required."
    item = store.find_item(ctx.settings, query)
    if item is None:
        return f"No reading-list item matches {query!r}."
    removed = store.delete_item(ctx.settings, item.id)
    return f"Removed from reading list: {removed.title}" if removed else "Nothing removed."


def _reading_tools() -> list[ToolSpec]:
    _ref = "The item's url, title, or exact id"
    return [
        ToolSpec(
            "save_reading",
            "Save a link to the user's read-it-later list. Use it when they share "
            "an article/page to get back to, or say \"save this for later\".",
            _params(
                {
                    "url": ("string", "The http(s) URL to save"),
                    "title": ("string", "A short title (defaults to the URL)"),
                    "note": ("string", "Why they're saving it / what to read it for"),
                },
                ["url"],
            ),
            _save_reading,
        ),
        ToolSpec(
            "list_reading",
            "List the user's reading list — unread items by default. Use it for "
            "\"what's on my reading list?\" or \"what did I save to read?\".",
            _params(
                {"include_read": ("string", "\"true\" to include already-read items")},
                [],
            ),
            _list_reading,
        ),
        ToolSpec(
            "mark_read",
            "Mark a saved link as read (done). ",
            _params({"query": ("string", _ref)}, ["query"]),
            _mark_read,
        ),
        ToolSpec(
            "remove_reading",
            "Delete a link from the reading list.",
            _params({"query": ("string", _ref)}, ["query"]),
            _remove_reading,
        ),
    ]
