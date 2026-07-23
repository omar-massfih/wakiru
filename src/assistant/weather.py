"""Cached weather snapshot — a forecast block in context without I/O on the reply path.

The daily briefing has an agenda, tasks, and mail but no weather; "should I
bring a jacket?" had no grounding. This module closes that gap the same way
:mod:`assistant.mail.snapshot` closes it for mail: :func:`maybe_refresh` runs
off the reply path (riding the reminder ticker on its own
``weather_refresh_minutes`` cadence) and persists what it saw; :func:`current`
— the context provider — only ever reads the persisted snapshot, stamped with
its fetch time so the model never over-claims freshness.

The forecast comes from Open-Meteo (https://open-meteo.com) — keyless, free
for non-commercial use — fetched through the same SSRF guard every other
outbound fetch uses. The location is a configured latitude/longitude
(``weather_latitude`` / ``weather_longitude``); resolving a place *name* to
coordinates (geocoding) is left as a future enhancement so this stays a single,
robust endpoint.

Persisted as a small JSON file under the memory dir (like the mail snapshot),
or the shared KV table under Postgres, so a restart doesn't blank the block.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

from .calendar.context import now
from .config import Settings, postgres_backend
from .netguard import urlopen_public

logger = logging.getLogger(__name__)

# A forecast older than this is withheld entirely — a stale forecast presented
# as current misleads more than it helps.
_MAX_AGE_HOURS = 6

# The snapshot lives in the shared KV table under Postgres, or a small JSON file
# under the memory dir on the local backend (mirrors mail.snapshot).
_KV_NAMESPACE = "weather"
_KV_KEY = "snapshot"

_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_FETCH_TIMEOUT = 10.0

# WMO weather-interpretation codes → short prose (Open-Meteo's `weather_code`).
# Ranges collapsed to the phrase a person would actually say.
_WMO = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "rime fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "freezing drizzle",
    57: "freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light showers",
    81: "showers",
    82: "heavy showers",
    85: "snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with hail",
}


def _describe_code(code: object) -> str:
    if not isinstance(code, (int, float)):
        return ""
    return _WMO.get(int(code), "")


def _units(settings: Settings) -> tuple[str, str, str, str]:
    """(temperature_unit, wind_speed_unit, temp_symbol, wind_symbol) for the API."""
    if settings.weather_units.strip().lower() == "imperial":
        return "fahrenheit", "mph", "°F", "mph"
    return "celsius", "kmh", "°C", "km/h"


def location(settings: Settings) -> tuple[float, float] | None:
    """The configured coordinates, or ``None`` when weather has no location."""
    lat, lon = settings.weather_latitude, settings.weather_longitude
    if lat is None or lon is None:
        return None
    return lat, lon


def enabled(settings: Settings) -> bool:
    """Weather is on, on a cadence, and has somewhere to forecast for."""
    return (
        settings.enable_weather
        and settings.weather_refresh_minutes > 0
        and location(settings) is not None
    )


def _path(settings: Settings):
    return settings.memory_path / "weather_snapshot.json"


def _decode(payload: str) -> tuple[str, datetime] | None:
    try:
        raw = json.loads(payload)
        fetched_at = datetime.fromisoformat(raw["fetched_at"])
    except (KeyError, TypeError, ValueError):
        logger.warning("unreadable weather snapshot; refetching on the next tick")
        return None
    return str(raw.get("text", "")), fetched_at


def _load(settings: Settings) -> tuple[str, datetime] | None:
    if storage_postgres := postgres_backend(settings):
        payload = storage_postgres.kv_get(settings, _KV_NAMESPACE, _KV_KEY)
        return _decode(payload) if payload else None
    try:
        payload = _path(settings).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        logger.warning("unreadable weather snapshot; refetching on the next tick")
        return None
    return _decode(payload)


def _save(settings: Settings, text: str, fetched_at: datetime) -> None:
    payload = json.dumps(
        {"text": text, "fetched_at": fetched_at.isoformat(timespec="seconds")}
    )
    if storage_postgres := postgres_backend(settings):
        storage_postgres.kv_set(settings, _KV_NAMESPACE, _KV_KEY, payload)
        return
    settings.memory_path.mkdir(parents=True, exist_ok=True)
    target = _path(settings)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(target)


def _fetch(settings: Settings, lat: float, lon: float) -> dict:
    temp_unit, wind_unit, _, _ = _units(settings)
    params = (
        f"latitude={lat}&longitude={lon}"
        "&current=temperature_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code"
        f"&timezone=auto&forecast_days=1&temperature_unit={temp_unit}"
        f"&wind_speed_unit={wind_unit}"
    )
    url = f"{_FORECAST_URL}?{params}"
    with urlopen_public(
        url, timeout=_FETCH_TIMEOUT, headers={"User-Agent": "wakiru-assistant"}
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def _render(settings: Settings, data: dict) -> str:
    """Turn the Open-Meteo payload into one or two plain-text lines."""
    _, _, temp_sym, wind_sym = _units(settings)
    lines: list[str] = []
    label = settings.weather_location_name.strip()
    if label:
        lines.append(f"Location: {label}")

    current = data.get("current") or {}
    temp = current.get("temperature_2m")
    if temp is not None:
        feels = current.get("apparent_temperature")
        cond = _describe_code(current.get("weather_code"))
        wind = current.get("wind_speed_10m")
        now_line = f"Now: {temp}{temp_sym}"
        if feels is not None and feels != temp:
            now_line += f" (feels {feels}{temp_sym})"
        if cond:
            now_line += f", {cond}"
        if wind is not None:
            now_line += f", wind {wind} {wind_sym}"
        lines.append(now_line)

    daily = data.get("daily") or {}

    def _first(key: str):
        seq = daily.get(key)
        return seq[0] if isinstance(seq, list) and seq else None

    tmax, tmin = _first("temperature_2m_max"), _first("temperature_2m_min")
    if tmax is not None and tmin is not None:
        day_line = f"Today: {tmin}–{tmax}{temp_sym}"
        cond = _describe_code(_first("weather_code"))
        if cond:
            day_line += f", {cond}"
        pop = _first("precipitation_probability_max")
        if pop is not None:
            day_line += f", {pop}% chance of precipitation"
        lines.append(day_line)

    return "\n".join(lines)


def refresh(settings: Settings) -> str | None:
    """Fetch the forecast now and persist the snapshot; ``None`` when disabled.

    The one place the network runs for weather. Raises nothing: a failed fetch
    logs and leaves the previous snapshot in place (stale-but-honest beats
    blank, and beats error text riding into every prompt).
    """
    if not enabled(settings):
        return None
    loc = location(settings)
    assert loc is not None  # enabled() guarantees it
    try:
        data = _fetch(settings, *loc)
        text = _render(settings, data)
    except Exception:
        logger.exception("weather refresh failed; keeping the previous snapshot")
        return None
    if not text:
        logger.warning("weather fetch returned nothing usable; keeping the previous snapshot")
        return None
    _save(settings, text, now(settings))
    return text


def maybe_refresh(settings: Settings) -> None:
    """Refresh when the snapshot is older than its cadence (the ticker hook)."""
    if not enabled(settings):
        return
    stored = _load(settings)
    if stored is not None:
        _, fetched_at = stored
        if now(settings) - fetched_at < timedelta(minutes=settings.weather_refresh_minutes):
            return
    refresh(settings)


def current(settings: Settings) -> str:
    """The snapshot as a context block, or ``""`` — never any I/O.

    Stamped with its fetch time ("as of 09:12") so the model presents it as a
    forecast fetched then, not a live reading. Empty when disabled, never
    fetched, or too old to be honest about.
    """
    if not enabled(settings):
        return ""
    stored = _load(settings)
    if stored is None:
        return ""
    text, fetched_at = stored
    if not text or now(settings) - fetched_at > timedelta(hours=_MAX_AGE_HOURS):
        return ""
    stamp = fetched_at.strftime("%H:%M")
    return f"## Weather (as of {stamp})\n{text}"
