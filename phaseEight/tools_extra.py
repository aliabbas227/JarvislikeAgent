"""
tools_extra.py -- Phase 8, milestone 1: the accuracy layer.

Phase 8's job is polish, not new capability for its own sake. This
module exists because a live diagnostic against the real DeepSeek API
(see README's "What the diagnostic actually showed") found the merged
Phase 4 + Phase 6 tool registry giving confidently wrong answers to two
of the most ordinary questions a person asks a voice assistant: "what
time is it?" and "what's the weather?"

Three tools live here:

  get_current_time -- NEW. There was no clock in the registry at all.
      Asked the time, the live model called get_active_window; asked the
      date, it called get_weather(location="Today") and read_screen.
      Those aren't random -- with no time tool available, the model
      reaches for whatever sounds adjacent and narrates a plausible
      answer from it. The fix is a real clock, not a better prompt.

  get_weather -- OVERRIDES Phase 4's. Phase 4's version returns
      hardcoded stub data ("partly cloudy", 21C) with a `note` field
      admitting it. The model does not reliably relay that admission,
      so a weather question produces a confident, invented forecast.
      This version is real, via Open-Meteo.

  search_web -- OVERRIDES Phase 4's. Phase 4's version works and
      returns genuinely current results (confirmed in the same
      diagnostic -- search was NOT the outdated part), but it asks
      Tavily for the bare minimum: 3 results, 300-character snippets,
      no synthesized answer, no recency control. This version asks for
      more and better.

Nothing here edits Phase 4's tools.py. Phase 4 is a finished,
standalone mini-project whose own CLI still works exactly as it always
did; Phase 8 layers over it at the registry level instead (see
app.py's register_tools call). That keeps the change reversible and
keeps the phase boundary intact -- Phase 8 is polish ON Phase 7, not a
rewrite of Phase 4.

Open-Meteo is used for weather specifically because it needs no API
key and no signup -- worth choosing over a key-gated service for a tool
whose whole job is to stop being a stub with the least new setup
burden. It is two calls: geocode the place name, then fetch current
conditions for those coordinates.
"""

import os
from datetime import datetime, timezone as _timezone

import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TAVILY_URL = "https://api.tavily.com/search"

HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "12"))

SEARCH_MAX_RESULTS = int(os.getenv("SEARCH_MAX_RESULTS", "10"))
SEARCH_SNIPPET_CHARS = int(os.getenv("SEARCH_SNIPPET_CHARS", "600"))

# WMO weather interpretation codes, which is what Open-Meteo returns
# instead of a text description. Condensed to the granularity a spoken
# reply can actually use -- "light drizzle" and "dense drizzle" are
# worth distinguishing, "moderate" vs "dense freezing drizzle" is not.
WMO_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "freezing drizzle",
    57: "heavy freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "heavy freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "rain showers",
    82: "violent rain showers",
    85: "light snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with heavy hail",
}


def get_current_time() -> dict:
    """
    The machine's own clock -- the whole point being that it is not the
    model guessing. Returns the local time and date in several forms
    because a spoken reply and a date calculation want different ones,
    and making the model reformat a single ISO string is a needless
    place for it to make an arithmetic mistake.

    Local time comes first and UTC second, deliberately: someone asking
    a desktop assistant "what time is it" means their own wall clock
    essentially always.
    """
    now = datetime.now().astimezone()
    return {
        "iso": now.isoformat(timespec="seconds"),
        "date": now.strftime("%Y-%m-%d"),
        "time_24h": now.strftime("%H:%M"),
        "time_12h": now.strftime("%I:%M %p").lstrip("0"),
        "weekday": now.strftime("%A"),
        "spoken": now.strftime("%A, %B %d, %Y at %I:%M %p").replace(" 0", " "),
        "timezone": now.tzname(),
        "utc_offset": now.strftime("%z"),
        "utc_iso": datetime.now(_timezone.utc).isoformat(timespec="seconds"),
    }


def _geocode(location: str) -> dict:
    """Resolves a place name to coordinates via Open-Meteo's geocoder.
    Returns an {"error": ...} dict rather than raising, matching the
    convention every tool in Phases 4 and 6 already follows: the model
    should see what went wrong and be able to say so, instead of the
    whole turn dying on an exception."""
    try:
        response = requests.get(
            GEOCODE_URL,
            params={"name": location, "count": 1, "format": "json"},
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Could not look up '{location}': {e}"}
    except ValueError as e:
        return {"error": f"Geocoder returned unreadable data for '{location}': {e}"}

    results = data.get("results") or []
    if not results:
        return {"error": f"No place found matching '{location}'."}
    return results[0]


def get_weather(location: str) -> dict:
    """
    Real current conditions via Open-Meteo (no API key required).

    Replaces Phase 4's stub, which returned invented values for every
    location on earth. Reports the resolved place name alongside the
    conditions so a geocoding near-miss ("Cambridge" landing in the
    wrong country) is visible in the answer rather than silently
    changing what the numbers mean.
    """
    place = _geocode(location)
    if "error" in place:
        return place

    try:
        response = requests.get(
            FORECAST_URL,
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": (
                    "temperature_2m,apparent_temperature,relative_humidity_2m,"
                    "precipitation,weather_code,wind_speed_10m"
                ),
                "timezone": "auto",
            },
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Weather lookup failed: {e}"}
    except ValueError as e:
        return {"error": f"Weather service returned unreadable data: {e}"}

    current = data.get("current") or {}
    resolved = ", ".join(
        part for part in (place.get("name"), place.get("admin1"), place.get("country")) if part
    )
    return {
        "location": resolved or location,
        "observed_at": current.get("time"),
        "condition": WMO_CODES.get(current.get("weather_code"), "unknown"),
        "temp_c": current.get("temperature_2m"),
        "feels_like_c": current.get("apparent_temperature"),
        "humidity_pct": current.get("relative_humidity_2m"),
        "precipitation_mm": current.get("precipitation"),
        "wind_kmh": current.get("wind_speed_10m"),
        "source": "Open-Meteo",
    }


def search_web(
    query: str,
    max_results: int = SEARCH_MAX_RESULTS,
    time_range: str = None,
    topic: str = "general",
) -> dict:
    """
    Web search via Tavily, asking for considerably more than Phase 4's
    version did.

    Phase 4's search was never the broken part -- the same live
    diagnostic that caught the missing clock confirmed it returns
    genuinely current results. What it asked for was thin: 3 results,
    300-character snippets, no synthesized answer, and no way to say
    "only recent things." Four changes here, each aimed at a specific
    way a thin result set turns into a wrong spoken answer:

      - include_answer: Tavily synthesizes a direct answer across the
        results. For a spoken assistant this matters more than for a
        text one -- the alternative is the model stitching an answer
        out of three truncated snippets and filling the seams itself.
      - search_depth "advanced": more thorough retrieval per query.
      - time_range: lets the model force recency explicitly when the
        question is time-sensitive, rather than hoping the ranking
        happens to favour fresh pages.
      - longer snippets and more of them, so there is less for the
        model to have to guess at.

    published_date is passed through when Tavily supplies it (it does
    for topic="news"), since a spoken answer that can say how old a
    claim is beats one that cannot.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {
            "error": "TAVILY_API_KEY not set. Sign up free at https://tavily.com and add it to .env."
        }

    payload = {
        "query": query,
        "max_results": max_results,
        "search_depth": "advanced",
        "include_answer": "advanced",
        "topic": topic,
    }
    if time_range:
        payload["time_range"] = time_range

    try:
        response = requests.post(
            TAVILY_URL,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Web search failed: {e}"}
    except ValueError as e:
        return {"error": f"Search service returned unreadable data: {e}"}

    results = []
    for r in (data.get("results") or [])[:max_results]:
        entry = {
            "title": r.get("title"),
            "url": r.get("url"),
            "snippet": (r.get("content") or "")[:SEARCH_SNIPPET_CHARS],
        }
        if r.get("published_date"):
            entry["published_date"] = r["published_date"]
        results.append(entry)

    out = {"query": query, "results": results}
    if data.get("answer"):
        out["answer"] = data["answer"]
    return out


TOOL_FUNCTIONS = {
    "get_current_time": get_current_time,
    "get_weather": get_weather,
    "search_web": search_web,
}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "Get the current local date and time from this machine's own clock. "
                "Use this for ANY question about the current time, today's date, the "
                "day of the week, or how long until/since a date. You have no other "
                "way to know the current time -- never answer from memory, and never "
                "use another tool to infer it."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get real current weather conditions for a location, via Open-Meteo. "
                "Requires an actual place name (a city, town, or region) -- it cannot "
                "resolve 'here', 'today', or anything that is not a place."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City or place name, e.g. 'Toronto' or 'Lahore'.",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the web for current information. Use this whenever the answer "
                "depends on anything that changes over time, or on events after your "
                "training data -- news, prices, releases, schedules, who currently "
                "holds a position. Prefer searching over answering from memory for "
                "anything time-sensitive."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "max_results": {
                        "type": "integer",
                        "description": f"How many results to return (default {SEARCH_MAX_RESULTS}).",
                    },
                    "time_range": {
                        "type": "string",
                        "enum": ["day", "week", "month", "year"],
                        "description": (
                            "Restrict results to this recency window. Set it for "
                            "genuinely time-sensitive questions ('latest', 'right now', "
                            "'this week'); leave unset otherwise."
                        ),
                    },
                    "topic": {
                        "type": "string",
                        "enum": ["general", "news"],
                        "description": "Use 'news' for current-events questions, 'general' otherwise.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]
