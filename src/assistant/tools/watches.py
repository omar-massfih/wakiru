"""Watch tools — register what background wakes should look for."""
from __future__ import annotations

from dataclasses import replace

from ._base import _ISO, _NO_MATCH, ToolContext, ToolSpec, _params


def _watch(
    ctx: ToolContext,
    kind: str,
    pattern: str = "",
    note: str = "",
    until: str = "",
    repeat: bool = False,
    lead_minutes: str = "",
    url: str = "",
) -> str:
    from .. import watches
    from ..calendar.context import format_when, now
    from ..calendar.store import parse_dt

    if kind not in watches.KINDS:
        return f"Tool failed: kind must be one of {', '.join(watches.KINDS)}."
    if kind == "mail_from" and not (
        ctx.settings.enable_email and ctx.settings.email_snapshot_minutes > 0
    ):
        return (
            "Tool failed: mail_from watches need email configured (ENABLE_EMAIL "
            "with periodic snapshots) — without it this watch could never fire."
        )
    if kind not in ("silence", "feed") and not str(pattern).strip():
        return "Tool failed: pattern is required for this kind."
    initial_hash = ""
    if kind == "feed":
        from .. import feeds

        url = str(url).strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            return "Tool failed: a feed watch needs an absolute http(s) url."
        try:
            entries = feeds.fetch_entries(url, force=True)
        except feeds.FeedError as exc:
            return f"Tool failed: could not read that feed — {exc}."
        # Prime the digest with what is already published, so the watch fires
        # on the next *new* matching entry, not on today's back catalogue.
        initial_hash = watches.feed_digest(
            watches.matched_feed_entries(entries, str(pattern))
        )
    expiry = None
    if until:
        expiry = parse_dt(str(until))
        if expiry is None:
            return f"Tool failed: until must be {_ISO}."
        if expiry <= now(ctx.settings):
            return "Tool failed: until is already in the past."
    elif kind == "silence":
        return f"Tool failed: a silence watch needs until (its deadline, {_ISO})."
    lead = watches.DEFAULT_LEAD_MINUTES
    if lead_minutes:
        try:
            lead = max(int(str(lead_minutes)), 0)
        except ValueError:
            return "Tool failed: lead_minutes must be a whole number of minutes."
    saved = watches.add(
        ctx.settings,
        kind,
        str(pattern),
        str(note),
        until=expiry,
        repeat=bool(repeat),
        lead_minutes=lead,
        url=url,
        initial_hash=initial_hash,
    )
    if saved is None:
        return (
            f"Tool failed: you already have {ctx.settings.watches_max_active} "
            "active watches. Drop one with unwatch first."
        )
    return (
        f"Watching ({saved.kind}): {_watched_thing(saved)}"
        f" until {format_when(ctx.settings, saved.expires_at)} (id {saved.id})"
    )


def _watched_thing(watch) -> str:
    """What a watch is looking at, for confirmations and listings."""
    if watch.kind == "feed":
        target = watch.url
        if watch.pattern:
            target += f' (matching "{watch.pattern}")'
        return target
    return watch.pattern or "user silence"

def _ambiguous_watches_message(matches: list) -> str:
    shown = ", ".join(f'{w.id} ("{w.pattern or w.kind}")' for w in matches[:5])
    more = f", +{len(matches) - 5} more" if len(matches) > 5 else ""
    return (
        f"Ambiguous — {len(matches)} watches match: {shown}{more}. "
        "Retry with one exact id from list_watches."
    )

def _unwatch(ctx: ToolContext, target: str) -> str:
    from .. import watches

    cancelled = watches.cancel(ctx.settings, str(target))
    if isinstance(cancelled, list):
        return _ambiguous_watches_message(cancelled)
    if cancelled is None:
        return _NO_MATCH
    return f"Stopped watching: {_watched_thing(cancelled)}"

def _list_watches(ctx: ToolContext) -> str:
    from .. import watches
    from ..calendar.context import format_when

    active = watches.list_active(ctx.settings)
    if not active:
        return "No active watches."
    return "\n".join(
        f"- [{w.kind}] {_watched_thing(w)}"
        + (f" — {w.note}" if w.note else "")
        + f" (until {format_when(ctx.settings, w.expires_at)}, id {w.id})"
        for w in active
    )

def _chat_only_feed(spec: ToolSpec) -> ToolSpec:
    """The heartbeat variant of the ``watch`` tool: feed registration refused.

    A background wake reads attacker-controllable text (mail subjects, feed
    titles); letting it register fetch URLs would be an exfiltration channel.
    The other kinds stay available — they only look at local state.
    """
    inner = spec.run

    def run(ctx: ToolContext, **args: object) -> str:
        if str(args.get("kind", "")).strip() == "feed":
            return (
                "Tool failed: feed watches can only be registered in "
                "conversation with the user, not from a background wake."
            )
        return inner(ctx, **args)

    return replace(spec, run=run)


def _watch_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            "watch",
            "Register something your background wakes should look for, so you "
            "notice it without being asked: kind=mail_from fires when unread "
            "mail matches the pattern (sender or subject substring), "
            "kind=calendar_window fires when an event matching the pattern is "
            "about to start (and wakes you for it), kind=silence fires if the "
            "user has not written by the until deadline, kind=feed fires when "
            "a registered RSS/Atom feed publishes a new entry matching the "
            "pattern (empty pattern = any new entry). When it fires you get "
            "your note back, so write the note to your future self.",
            _params(
                {
                    "kind": (
                        "string",
                        '"mail_from", "calendar_window", "silence", or "feed"',
                    ),
                    "pattern": (
                        "string",
                        "Substring to match (sender/subject, event title, or "
                        "feed-entry title); not used for silence",
                    ),
                    "url": ("string", "feed only: the RSS/Atom feed's http(s) URL"),
                    "note": ("string", "What future-you should do when this fires"),
                    "until": (
                        "string",
                        f"Expiry — or the deadline for silence — {_ISO} "
                        "(default: 2 weeks out)",
                    ),
                    "repeat": (
                        "boolean",
                        "mail_from/feed only: keep firing on new matches instead of once",
                    ),
                    "lead_minutes": (
                        "string",
                        "calendar_window only: minutes before the event (default 30)",
                    ),
                },
                ["kind"],
            ),
            _watch,
        ),
        ToolSpec(
            "unwatch",
            "Drop an active watch by id or an unambiguous pattern/note reference.",
            _params({"target": ("string", "Watch id, pattern, or note")}, ["target"]),
            _unwatch,
        ),
        ToolSpec(
            "list_watches",
            "List your active watches.",
            _params({}, []),
            _list_watches,
        ),
    ]
