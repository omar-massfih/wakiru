"""Weather tool — an on-demand forecast for any place the user asks about.

Distinct from the home-location block injected each turn: this answers "what's
the weather in Bergen this weekend?" by geocoding and fetching that place on the
spot. Chat-only (it does network I/O), gated on ``enable_weather``.
"""
from __future__ import annotations

from ._base import ToolContext, ToolSpec, _int_arg, _params


def _get_weather(ctx: ToolContext, **args: object) -> str:
    from .. import weather

    location = str(args.get("location", "")).strip()
    if not location:
        return "Tool failed: a location (city or place name) is required."
    days = _int_arg(args.get("days", ""), 1) or 1
    result = weather.forecast_for(ctx.settings, location, days)
    if result is None:
        return (
            f"Couldn't get the weather for {location!r} — the place didn't "
            "resolve or the service was unavailable. Check the spelling or try a "
            "nearby larger place."
        )
    return result


def _weather_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            "get_weather",
            "Get the current conditions and forecast for a place the user names "
            "(a city or town). Use it for weather anywhere other than their home "
            "location, or when they want a multi-day outlook.",
            _params(
                {
                    "location": ("string", "City or place name, e.g. \"Bergen\""),
                    "days": ("string", "Days of forecast, 1-7 (default 1)"),
                },
                ["location"],
            ),
            _get_weather,
        ),
    ]
