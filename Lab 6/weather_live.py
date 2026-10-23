"""Open-Meteo current-weather adapter, kept separate for offline testing."""

import json
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen


def get_json(base_url: str, params: dict) -> dict:
    try:
        with urlopen(base_url + "?" + urlencode(params), timeout=12) as response:
            return json.load(response)
    except (URLError, TimeoutError, ValueError) as exc:
        raise RuntimeError(f"Weather service unavailable: {exc}") from exc


def conditions_for_code(code: int) -> str:
    if code == 0:
        return "clear"
    if code in {1, 2, 3}:
        return "cloudy"
    if code in {45, 48}:
        return "foggy"
    if code in {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82}:
        return "rainy"
    if code in {71, 73, 75, 77, 85, 86}:
        return "snowy"
    if code in {95, 96, 99}:
        return "stormy"
    return "unknown"


def current_weather(city: str) -> dict:
    places = get_json(
        "https://geocoding-api.open-meteo.com/v1/search",
        {"name": city, "count": 1, "format": "json"},
    ).get("results") or []
    if not places:
        raise ValueError(f"Unknown city: {city}")
    place = places[0]
    report = get_json(
        "https://api.open-meteo.com/v1/forecast",
        {"latitude": place["latitude"], "longitude": place["longitude"],
         "current": "temperature_2m,weather_code", "timezone": "auto"},
    )
    try:
        current = report["current"]
        return {
            "city": place["name"],
            "temperature": float(current["temperature_2m"]),
            "conditions": conditions_for_code(int(current["weather_code"])),
            "observed_at": current["time"],
            "source": "Open-Meteo model-based current conditions",
        }
    except (KeyError, ValueError, TypeError) as exc:
        raise RuntimeError("Incomplete weather response") from exc

