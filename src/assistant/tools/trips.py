"""Trip tools — add/list/update/remove over the trips store."""
from __future__ import annotations

from ._base import ToolContext, ToolSpec, _params


def _validated_dates(start: str, end: str) -> str:
    """An error message when the dates don't form a trip, else ``""``."""
    from ..trips import store

    began, ended = store.parse_date(start), store.parse_date(end)
    if start.strip() and began is None:
        return f"Tool failed: start {start!r} is not a YYYY-MM-DD date."
    if end.strip() and ended is None:
        return f"Tool failed: end {end!r} is not a YYYY-MM-DD date."
    if began is not None and ended is not None and ended < began:
        return "Tool failed: the trip ends before it starts."
    return ""


def _render_trip(trip) -> str:
    line = f"- {trip.name}"
    if trip.destination and trip.destination != trip.name:
        line += f" ({trip.destination})"
    if trip.start or trip.end:
        line += f" {trip.start or '?'} → {trip.end or '?'}"
    if trip.timezone:
        line += f" [{trip.timezone}]"
    if trip.notes:
        line += f" — {trip.notes}"
    return line + f"  [id: {trip.id}]"


def _add_trip(ctx: ToolContext, **args: object) -> str:
    from ..trips import store

    destination = str(args.get("destination", "")).strip()
    if not destination:
        return "Tool failed: a destination is required."
    start = str(args.get("start", "") or "")
    end = str(args.get("end", "") or "")
    problem = _validated_dates(start, end)
    if problem:
        return problem
    timezone = str(args.get("timezone", "") or "").strip()
    if not store.valid_timezone(timezone):
        return f"Tool failed: {timezone!r} is not an IANA timezone (like Europe/Lisbon)."
    trip = store.create_trip(
        ctx.settings,
        destination=destination,
        name=str(args.get("name", "") or ""),
        start=start,
        end=end,
        timezone=timezone,
        notes=str(args.get("notes", "") or ""),
    )
    when = f" {trip.start} → {trip.end}" if trip.start or trip.end else ""
    return f"Trip saved: {trip.name}{when} (id {trip.id})"


def _list_trips(ctx: ToolContext, **args: object) -> str:
    from ..trips import store

    include_past = str(args.get("include_past", "")).strip().lower() in ("1", "true", "yes")
    trips = store.list_trips(ctx.settings, include_past=include_past)
    if not trips:
        return "No trips on file."
    header = "Trips (past included):" if include_past else "Current and upcoming trips:"
    return header + "\n" + "\n".join(_render_trip(t) for t in trips)


def _update_trip(ctx: ToolContext, **args: object) -> str:
    from ..trips import store

    query = str(args.get("trip", "")).strip()
    if not query:
        return "Tool failed: which trip? Pass its name, destination, or id."
    target = store.find_trip(ctx.settings, query)
    if target is None:
        return f"No trip matches {query!r}."
    fields = {
        k: str(args[k]) for k in ("name", "destination", "start", "end", "timezone", "notes")
        if args.get(k) not in (None, "")
    }
    if not fields:
        return "Tool failed: nothing to change — pass at least one field."
    problem = _validated_dates(
        fields.get("start", target.start), fields.get("end", target.end)
    )
    if problem:
        return problem
    if "timezone" in fields and not store.valid_timezone(fields["timezone"]):
        return (
            f"Tool failed: {fields['timezone']!r} is not an IANA timezone "
            "(like Europe/Lisbon)."
        )
    updated = store.update_trip(ctx.settings, target.id, **fields)
    return f"Trip updated: {_render_trip(updated)}" if updated else "Nothing updated."


def _remove_trip(ctx: ToolContext, **args: object) -> str:
    from ..trips import store

    query = str(args.get("trip", "")).strip()
    if not query:
        return "Tool failed: which trip? Pass its name, destination, or id."
    target = store.find_trip(ctx.settings, query)
    if target is None:
        return f"No trip matches {query!r}."
    removed = store.delete_trip(ctx.settings, target.id)
    return f"Trip removed: {removed.name}" if removed else "Nothing removed."


def _trip_tools() -> list[ToolSpec]:
    _ref = "The trip's name, destination, or exact id"
    _date = "YYYY-MM-DD"
    return [
        ToolSpec(
            "add_trip",
            "Save a trip (travel with dates) so it's kept in mind before and "
            "during — flights, holidays, work travel.",
            _params(
                {
                    "destination": ("string", "Where to (city or place)"),
                    "name": ("string", "Optional label (defaults to the destination)"),
                    "start": ("string", f"Departure date, {_date}"),
                    "end": ("string", f"Return date (inclusive), {_date}"),
                    "timezone": (
                        "string",
                        "Destination IANA timezone (like Europe/Lisbon), if known",
                    ),
                    "notes": ("string", "Flights, hotel, who with — anything worth recalling"),
                },
                ["destination"],
            ),
            _add_trip,
        ),
        ToolSpec(
            "list_trips",
            "List current and upcoming trips (past ones with include_past).",
            _params(
                {"include_past": ("string", "\"true\" to include finished trips")},
                [],
            ),
            _list_trips,
        ),
        ToolSpec(
            "update_trip",
            "Change a saved trip's dates, destination, timezone, or notes.",
            _params(
                {
                    "trip": ("string", _ref),
                    "name": ("string", "New label"),
                    "destination": ("string", "New destination"),
                    "start": ("string", f"New departure date, {_date}"),
                    "end": ("string", f"New return date, {_date}"),
                    "timezone": ("string", "New IANA timezone"),
                    "notes": ("string", "Replacement notes"),
                },
                ["trip"],
            ),
            _update_trip,
        ),
        ToolSpec(
            "remove_trip",
            "Delete a trip (cancelled or added by mistake).",
            _params({"trip": ("string", _ref)}, ["trip"]),
            _remove_trip,
        ),
    ]
