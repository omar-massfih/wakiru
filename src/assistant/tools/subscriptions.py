"""Subscription tools — add/list/update/remove recurring charges."""
from __future__ import annotations

from ._base import ToolContext, ToolSpec, _params

_CADENCE = "How often it renews: weekly, monthly, quarterly, or yearly"


def _add_subscription(ctx: ToolContext, **args: object) -> str:
    from ..subscriptions import store

    name = str(args.get("name", "")).strip()
    if not name:
        return "Tool failed: a name is required."
    dupe = store.find_exact_name(ctx.settings, name)
    if dupe is not None:
        return (
            f"Not added — a subscription named {dupe.name!r} already exists "
            f"(id {dupe.id}). Update it instead, or use a distinct name."
        )
    sub = store.create_subscription(
        ctx.settings,
        name=name,
        amount=args.get("amount", 0) or 0,
        currency=str(args.get("currency", "") or ""),
        cadence=str(args.get("cadence", "monthly") or "monthly"),
        renews_on=str(args.get("renews_on", "") or ""),
        notes=str(args.get("notes", "") or ""),
    )
    return f"Tracking subscription: {sub.name} ({sub.cadence})"


def _list_subscriptions(ctx: ToolContext, **args: object) -> str:
    from datetime import date

    from ..subscriptions import store
    from ..subscriptions.context import rollup

    subs = store.list_subscriptions(ctx.settings)
    return "Subscriptions:\n" + rollup(ctx.settings, subs, date.today())


def _update_subscription(ctx: ToolContext, **args: object) -> str:
    from ..subscriptions import store

    query = str(args.get("query", "")).strip()
    if not query:
        return "Tool failed: a name or id is required."
    sub = store.find_subscription(ctx.settings, query)
    if sub is None:
        return f"No subscription matches {query!r}."
    updated = store.update_subscription(
        ctx.settings, sub.id,
        name=args.get("name"), amount=args.get("amount"),
        currency=args.get("currency"), cadence=args.get("cadence"),
        renews_on=args.get("renews_on"), notes=args.get("notes"),
    )
    return f"Updated subscription: {updated.name}" if updated else "Nothing updated."


def _remove_subscription(ctx: ToolContext, **args: object) -> str:
    from ..subscriptions import store

    query = str(args.get("query", "")).strip()
    if not query:
        return "Tool failed: a name or id is required."
    sub = store.find_subscription(ctx.settings, query)
    if sub is None:
        return f"No subscription matches {query!r}."
    removed = store.delete_subscription(ctx.settings, sub.id)
    return f"Stopped tracking: {removed.name}" if removed else "Nothing removed."


def _subscription_tools() -> list[ToolSpec]:
    _ref = "The subscription's name or exact id"
    return [
        ToolSpec(
            "add_subscription",
            "Track a recurring charge / bill the user pays (streaming service, "
            "gym, insurance, SaaS…) so it can be summed and renewals flagged.",
            _params(
                {
                    "name": ("string", "What it is, e.g. \"Spotify\""),
                    "amount": ("string", "Price per cycle, e.g. \"129\""),
                    "currency": ("string", "Currency, e.g. \"NOK\" / \"USD\""),
                    "cadence": ("string", _CADENCE),
                    "renews_on": ("string", "Next/last renewal date, YYYY-MM-DD"),
                    "notes": ("string", "Free-form notes"),
                },
                ["name"],
            ),
            _add_subscription,
        ),
        ToolSpec(
            "list_subscriptions",
            "List tracked subscriptions with their next renewal and the estimated "
            "monthly spend. Use it for \"what am I paying for?\" / \"my subscriptions\".",
            _params({}, []),
            _list_subscriptions,
        ),
        ToolSpec(
            "update_subscription",
            "Change a subscription's amount, currency, cadence, renewal date, or notes.",
            _params(
                {
                    "query": ("string", _ref),
                    "name": ("string", "New name"),
                    "amount": ("string", "New price per cycle"),
                    "currency": ("string", "New currency"),
                    "cadence": ("string", _CADENCE),
                    "renews_on": ("string", "New renewal date, YYYY-MM-DD"),
                    "notes": ("string", "New notes"),
                },
                ["query"],
            ),
            _update_subscription,
        ),
        ToolSpec(
            "remove_subscription",
            "Stop tracking a subscription (e.g. the user cancelled it).",
            _params({"query": ("string", _ref)}, ["query"]),
            _remove_subscription,
        ),
    ]
