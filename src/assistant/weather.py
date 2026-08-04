"""Cached weather snapshot — a forecast block in context without I/O on the reply path.

The cached location follows the active trip destination when trips are enabled,
and otherwise uses the configured home location. :func:`maybe_refresh` runs off
the reply path and persists both what it saw and which effective location it
belongs to; :func:`current` only exposes a fresh snapshot whose location still
matches.

The forecast and geocoding come from Open-Meteo (https://open-meteo.com) and
are fetched through the same SSRF guard every other outbound fetch uses.

Persisted as a small JSON file under the memory dir (like the mail snapshot),
or the shared KV table under Postgres, so a restart doesn't blank the block.
"""

from __future__ import annotations

import json
import logging
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .calendar.context import now, resolve_home_tz
from .config import Settings, postgres_backend
from .netguard import urlopen_public
from .trips.store import active_trip_at, list_trips, parse_date

logger = logging.getLogger(__name__)

# A forecast older than this is withheld entirely — a stale forecast presented
# as current misleads more than it helps.
_MAX_AGE_HOURS = 6

# The snapshot lives in the shared KV table under Postgres, or a small JSON file
# under the memory dir on the local backend (mirrors mail.snapshot).
_KV_NAMESPACE = "weather"
_KV_KEY = "snapshot"
# The resolved-coordinates cache (geocoding a place name), keyed under the same
# namespace so re-geocoding only happens when the configured name changes.
_KV_GEO_KEY = "geocode"

_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FETCH_TIMEOUT = 10.0

# WMO codes worth a proactive heads-up: heavy rain, freezing rain, heavy snow,
# violent showers, heavy snow showers, and thunderstorms.
_SEVERE_CODES = {65, 66, 67, 75, 82, 85, 86, 95, 96, 99}

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


@dataclass(frozen=True)
class _EffectiveLocation:
    key: str
    label: str
    name: str = ""
    coordinates: tuple[float, float] | None = None


def _normalized_name(value: str) -> str:
    return " ".join(value.split()).casefold()


def _named_location(name: str) -> _EffectiveLocation | None:
    label = name.strip()
    normalized = _normalized_name(label)
    if not normalized:
        return None
    return _EffectiveLocation(key=f"name:{normalized}", label=label, name=label)


def _effective_location(
    settings: Settings, instant: datetime | None = None
) -> _EffectiveLocation | None:
    """Select the trip destination or home location at one aware instant."""
    if not settings.enable_weather or settings.weather_refresh_minutes <= 0:
        return None
    captured = instant or now(settings)
    if captured.tzinfo is None:
        raise ValueError("instant must be timezone-aware")
    if settings.enable_trips:
        trip = active_trip_at(settings, captured, resolve_home_tz(settings))
        if trip is not None:
            destination = _named_location(trip.destination)
            if destination is not None:
                return destination

    coords = _configured_coords(settings)
    label = settings.weather_location_name.strip()
    if coords is not None:
        lat, lon = coords
        key = json.dumps(["coords", float(lat), float(lon), label], separators=(",", ":"))
        return _EffectiveLocation(key=key, label=label, coordinates=(lat, lon))
    return _named_location(label)


def effective_location_key(settings: Settings, instant: datetime | None = None) -> str | None:
    """Stable identity of the currently selected snapshot location, without network I/O."""
    location = _effective_location(settings, instant)
    return location.key if location is not None else None


def next_location_boundary(settings: Settings, instant: datetime) -> datetime | None:
    """Return the next trip start/end instant that may change the location.

    Trip dates are inclusive and use the destination timezone when supplied,
    matching :func:`active_trip_at`.  Returning boundaries for overlapping or
    destination-less trips can cause a harmless extra wake, while ensuring the
    scheduler never sleeps through a real location transition.
    """
    if not settings.enable_weather or not settings.enable_trips:
        return None
    if instant.tzinfo is None:
        raise ValueError("instant must be timezone-aware")

    home_tz = resolve_home_tz(settings)
    home_date = instant.astimezone(home_tz).date()
    boundaries: list[datetime] = []
    for trip in list_trips(settings, include_past=True, today=home_date):
        started, ends = parse_date(trip.start), parse_date(trip.end)
        if started is None or ends is None:
            continue
        trip_tz = home_tz
        if trip.timezone:
            with suppress(ValueError, ZoneInfoNotFoundError):
                trip_tz = ZoneInfo(trip.timezone)
        start_boundary = datetime.combine(started, time.min, tzinfo=trip_tz)
        end_boundary = datetime.combine(ends + timedelta(days=1), time.min, tzinfo=trip_tz)
        if start_boundary > instant:
            boundaries.append(start_boundary)
        if end_boundary > instant:
            boundaries.append(end_boundary)
    return min(boundaries) if boundaries else None


def _describe_code(code: object) -> str:
    if not isinstance(code, (int, float)):
        return ""
    return _WMO.get(int(code), "")


def _units(settings: Settings) -> tuple[str, str, str, str]:
    """(temperature_unit, wind_speed_unit, temp_symbol, wind_symbol) for the API."""
    if settings.weather_units.strip().lower() == "imperial":
        return "fahrenheit", "mph", "°F", "mph"
    return "celsius", "kmh", "°C", "km/h"


def _configured_coords(settings: Settings) -> tuple[float, float] | None:
    """The explicitly configured latitude/longitude, or ``None``. No I/O."""
    lat, lon = settings.weather_latitude, settings.weather_longitude
    if lat is None or lon is None:
        return None
    return lat, lon


def enabled(settings: Settings) -> bool:
    """Whether weather currently has an effective location to forecast for."""
    return effective_location_key(settings) is not None


def _geocode(settings: Settings, name: str) -> tuple[float, float] | None:
    """Resolve a place name to coordinates via Open-Meteo's geocoder (keyless).

    Network I/O — call only off the reply path (from :func:`refresh`). Returns
    ``None`` when the name doesn't resolve or the fetch fails.
    """
    url = f"{_GEOCODE_URL}?name={quote(name)}&count=1&format=json"
    try:
        with urlopen_public(
            url, timeout=_FETCH_TIMEOUT, headers={"User-Agent": "wakiru-assistant"}
        ) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        logger.exception("weather: geocoding %r failed", name)
        return None
    results = data.get("results") or []
    if not results:
        logger.warning("weather: no geocoding match for %r", name)
        return None
    top = results[0]
    lat, lon = top.get("latitude"), top.get("longitude")
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        return None
    return float(lat), float(lon)


def _geo_cache_load(settings: Settings) -> dict[str, tuple[float, float]]:
    if storage_postgres := postgres_backend(settings):
        payload = storage_postgres.kv_get(settings, _KV_NAMESPACE, _KV_GEO_KEY)
    else:
        try:
            payload = (settings.memory_path / "weather_geocode.json").read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            return {}
    try:
        raw = json.loads(payload) if payload else {}
    except (TypeError, ValueError):
        return {}
    result: dict[str, tuple[float, float]] = {}
    # Legacy cache: {name, latitude, longitude}.
    if isinstance(raw, dict) and "name" in raw:
        lat, lon = raw.get("latitude"), raw.get("longitude")
        key = _normalized_name(str(raw.get("name", "")))
        if key and isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            result[key] = float(lat), float(lon)
        return result
    locations = raw.get("locations", {}) if isinstance(raw, dict) else {}
    if not isinstance(locations, dict):
        return result
    for key, entry in locations.items():
        if not isinstance(entry, dict):
            continue
        lat, lon = entry.get("latitude"), entry.get("longitude")
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            result[str(key)] = float(lat), float(lon)
    return result


def _geo_cache_save(settings: Settings, locations: dict[str, tuple[float, float]]) -> None:
    payload = json.dumps(
        {
            "locations": {
                key: {"latitude": lat, "longitude": lon} for key, (lat, lon) in locations.items()
            }
        }
    )
    if storage_postgres := postgres_backend(settings):
        storage_postgres.kv_set(settings, _KV_NAMESPACE, _KV_GEO_KEY, payload)
        return
    settings.memory_path.mkdir(parents=True, exist_ok=True)
    target = settings.memory_path / "weather_geocode.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(target)


def _resolve_coords(settings: Settings, location: _EffectiveLocation) -> tuple[float, float] | None:
    """Resolve explicit coordinates or a named effective location.

    May do network I/O (the geocode) — refresh-path only, never on a reply.
    """
    if location.coordinates is not None:
        return location.coordinates
    name = location.name
    cache_key = _normalized_name(name)
    if not cache_key:
        return None
    cached = _geo_cache_load(settings)
    if cache_key in cached:
        return cached[cache_key]
    coords = _geocode(settings, name)
    if coords is not None:
        cached[cache_key] = coords
        _geo_cache_save(settings, cached)
    return coords


def _path(settings: Settings):
    return settings.memory_path / "weather_snapshot.json"


def _decode(payload: str) -> tuple[str, datetime, str | None] | None:
    try:
        raw = json.loads(payload)
        fetched_at = datetime.fromisoformat(raw["fetched_at"])
    except (KeyError, TypeError, ValueError):
        logger.warning("unreadable weather snapshot; refetching on the next tick")
        return None
    location_key = raw.get("location_key")
    return (
        str(raw.get("text", "")),
        fetched_at,
        str(location_key) if location_key is not None else None,
    )


def _load(settings: Settings) -> tuple[str, datetime, str | None] | None:
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


def _save(settings: Settings, text: str, fetched_at: datetime, location_key: str) -> None:
    payload = json.dumps(
        {
            "text": text,
            "fetched_at": fetched_at.isoformat(timespec="seconds"),
            "location_key": location_key,
        }
    )
    if storage_postgres := postgres_backend(settings):
        storage_postgres.kv_set(settings, _KV_NAMESPACE, _KV_KEY, payload)
        return
    settings.memory_path.mkdir(parents=True, exist_ok=True)
    target = _path(settings)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(target)


def _fetch(settings: Settings, lat: float, lon: float, days: int = 1) -> dict:
    temp_unit, wind_unit, _, _ = _units(settings)
    params = (
        f"latitude={lat}&longitude={lon}"
        "&current=temperature_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code"
        f"&timezone=auto&forecast_days={days}&temperature_unit={temp_unit}"
        f"&wind_speed_unit={wind_unit}"
    )
    url = f"{_FORECAST_URL}?{params}"
    with urlopen_public(
        url, timeout=_FETCH_TIMEOUT, headers={"User-Agent": "wakiru-assistant"}
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def _now_line(settings: Settings, current: dict) -> str:
    """The "Now: …" current-conditions line, or ``""`` if no reading."""
    _, _, temp_sym, wind_sym = _units(settings)
    temp = current.get("temperature_2m")
    if temp is None:
        return ""
    feels = current.get("apparent_temperature")
    cond = _describe_code(current.get("weather_code"))
    wind = current.get("wind_speed_10m")
    line = f"Now: {temp}{temp_sym}"
    if feels is not None and feels != temp:
        line += f" (feels {feels}{temp_sym})"
    if cond:
        line += f", {cond}"
    if wind is not None:
        line += f", wind {wind} {wind_sym}"
    return line


def _weekday(date: str) -> str:
    """ "Mon" from a YYYY-MM-DD date, or the raw string if unparseable."""
    try:
        return datetime.strptime(date, "%Y-%m-%d").strftime("%a")
    except (TypeError, ValueError):
        return date


def _render(settings: Settings, data: dict, label: str | None = None) -> str:
    """Turn the Open-Meteo payload into one or two plain-text lines."""
    _, _, temp_sym, _ = _units(settings)
    lines: list[str] = []
    label = settings.weather_location_name.strip() if label is None else label.strip()
    if label:
        lines.append(f"Location: {label}")

    now_line = _now_line(settings, data.get("current") or {})
    if now_line:
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

    # A prominent heads-up for severe conditions, inserted right after the
    # location label so it leads the block (and rides into the briefing).
    day_code = _first("weather_code")
    if isinstance(day_code, (int, float)) and int(day_code) in _SEVERE_CODES:
        alert = f"⚠ {_describe_code(day_code).capitalize()} expected today."
        lines.insert(1 if label else 0, alert)

    return "\n".join(lines)


def _render_forecast(settings: Settings, name: str, data: dict, days: int) -> str:
    """A multi-day forecast for an on-demand location query (the get_weather tool)."""
    _, _, temp_sym, _ = _units(settings)
    lines = [f"Weather for {name}:"]
    now_line = _now_line(settings, data.get("current") or {})
    if now_line:
        lines.append(now_line)

    daily = data.get("daily") or {}
    times = daily.get("time") or []
    tmax = daily.get("temperature_2m_max") or []
    tmin = daily.get("temperature_2m_min") or []
    codes = daily.get("weather_code") or []
    pops = daily.get("precipitation_probability_max") or []
    for i in range(min(days, len(times))):
        if i >= len(tmin) or i >= len(tmax):
            continue
        line = f"{_weekday(times[i])} {times[i]}: {tmin[i]}–{tmax[i]}{temp_sym}"
        cond = _describe_code(codes[i]) if i < len(codes) else ""
        if cond:
            line += f", {cond}"
        if i < len(pops) and pops[i] is not None:
            line += f", {pops[i]}% precip"
        lines.append(line)

    return "\n".join(lines)


def forecast_for(settings: Settings, location_name: str, days: int = 1) -> str | None:
    """An on-demand forecast for a named place — geocode, fetch, render.

    Distinct from the injected home snapshot: it resolves an *arbitrary* place
    (not cached against the configured home) and can span several days. Network
    I/O, so chat-only (never a background wake). ``None`` on any failure.
    """
    name = location_name.strip()
    if not name:
        return None
    coords = _geocode(settings, name)
    if coords is None:
        return None
    days = max(1, min(days, 7))
    try:
        data = _fetch(settings, *coords, days=days)
    except Exception:
        logger.exception("weather: on-demand fetch for %r failed", name)
        return None
    return _render_forecast(settings, name, data, days)


def _refresh_selected(
    settings: Settings, location: _EffectiveLocation, captured: datetime
) -> str | None:
    coords = _resolve_coords(settings, location)
    if coords is None:
        logger.warning("weather: could not resolve a location; skipping refresh")
        return None
    try:
        data = _fetch(settings, *coords)
        text = _render(settings, data, location.label)
    except Exception:
        logger.exception("weather refresh failed; keeping the previous snapshot")
        return None
    if not text:
        logger.warning("weather fetch returned nothing usable; keeping the previous snapshot")
        return None
    _save(settings, text, captured, location.key)
    return text


def refresh(settings: Settings, instant: datetime | None = None) -> str | None:
    """Fetch and persist the forecast selected at one captured instant."""
    captured = instant or now(settings)
    location = _effective_location(settings, captured)
    if location is None:
        return None
    return _refresh_selected(settings, location, captured)


def maybe_refresh(settings: Settings, instant: datetime | None = None) -> None:
    """Refresh at one captured instant when the snapshot is old or mismatched."""
    captured = instant or now(settings)
    location = _effective_location(settings, captured)
    if location is None:
        return
    stored = _load(settings)
    if stored is not None:
        _, fetched_at, location_key = stored
        if location_key == location.key and captured - fetched_at < timedelta(
            minutes=settings.weather_refresh_minutes
        ):
            return
    _refresh_selected(settings, location, captured)


def current(settings: Settings, instant: datetime | None = None) -> str:
    """The snapshot as a context block, or ``""`` — never any I/O.

    Stamped with its fetch time ("as of 09:12") so the model presents it as a
    forecast fetched then, not a live reading. Empty when disabled, never
    fetched, or too old to be honest about.
    """
    captured = instant or now(settings)
    location = _effective_location(settings, captured)
    if location is None:
        return ""
    stored = _load(settings)
    if stored is None:
        return ""
    text, fetched_at, location_key = stored
    if (
        location_key != location.key
        or not text
        or captured - fetched_at > timedelta(hours=_MAX_AGE_HOURS)
    ):
        return ""
    stamp = fetched_at.strftime("%H:%M")
    return f"## Weather (as of {stamp})\n{text}"
