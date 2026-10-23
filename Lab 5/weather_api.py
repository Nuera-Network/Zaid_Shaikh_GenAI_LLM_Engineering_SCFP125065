"""Open-Meteo weather API adapter: deterministic parsing, no LLM weather claims."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class WeatherUnavailable(RuntimeError):
    """Raised when no trustworthy forecast can be retrieved for a request."""


def _get_json(base_url: str, params: dict) -> dict:
    url = base_url + "?" + urlencode(params)
    try:
        with urlopen(url, timeout=12) as response:
            return json.load(response)
    except (URLError, HTTPError, TimeoutError, ValueError) as exc:
        raise WeatherUnavailable(f"Weather service request failed: {exc}") from exc


def describe_weather_code(code: int) -> str:
    if code == 0:
        return "clear sky"
    if code in {1, 2, 3}:
        return "partly cloudy or overcast"
    if code in {45, 48}:
        return "fog"
    if code in {51, 53, 55, 56, 57}:
        return "drizzle"
    if code in {61, 63, 65, 66, 67}:
        return "rain"
    if code in {71, 73, 75, 77, 85, 86}:
        return "snow"
    if code in {80, 81, 82}:
        return "rain showers"
    if code in {95, 96, 99}:
        return "thunderstorms"
    return "conditions not classified"


def fetch_forecast(location: str, event_date: str) -> dict:
    """Get a forecast using location geocoding and a daily Open-Meteo query.

    Returns raw normalized data in exactly the WeatherData field names.
    It deliberately refuses dates outside the provider's 16-day forecast.
    """
    try:
        target = date.fromisoformat(event_date)
    except ValueError as exc:
        raise WeatherUnavailable("Event date must be YYYY-MM-DD") from exc

    places = _get_json(
        "https://geocoding-api.open-meteo.com/v1/search",
        {"name": location, "count": 1, "language": "en", "format": "json"},
    ).get("results") or []
    if not places:
        raise WeatherUnavailable(f"Location could not be identified: {location}")
    place = places[0]
    tz_name = place.get("timezone", "UTC")
    try:
        today = datetime.now(ZoneInfo(tz_name)).date()
    except ZoneInfoNotFoundError as exc:
        raise WeatherUnavailable("Unknown timezone returned by geocoding") from exc
    if not (today <= target <= today + timedelta(days=15)):
        raise WeatherUnavailable(
            f"Only dates from {today} through {today + timedelta(days=15)} "
            "are supported for current forecasts. Check again closer to the event."
        )

    forecast = _get_json(
        "https://api.open-meteo.com/v1/forecast",
        {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "daily": "temperature_2m_max,precipitation_probability_max,weather_code",
            "start_date": event_date,
            "end_date": event_date,
            "timezone": tz_name,
        },
    )
    try:
        daily = forecast["daily"]
        idx = daily["time"].index(event_date)
        temp = daily["temperature_2m_max"][idx]
        chance = daily["precipitation_probability_max"][idx]
        code = daily["weather_code"][idx]
        if temp is None or chance is None or code is None:
            raise ValueError("Some forecast values were missing")
        return {
            "temperature": float(temp),
            "precipitation_chance": float(chance),
            "conditions": describe_weather_code(int(code)),
        }
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        raise WeatherUnavailable("The forecast did not contain complete daily data") from exc

