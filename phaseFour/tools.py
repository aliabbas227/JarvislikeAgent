import os
import time
import threading

import requests

def get_weather(location: str) -> dict:
    return {
        "location": location,
        "condition": "partly cloudy",
        "temp_c": 21,
        "note": "This is stub data — replace with a real weather API call when ready.",
    }


def set_timer(duration_seconds: int, label: str = "") -> dict:
    """
    Real timer: schedules a background thread that prints to the console
    when it fires. Non-blocking — the CLI stays responsive while it
    counts down. daemon=True so a pending timer never keeps the process
    alive if the user exits before it fires.

    This intentionally does NOT use asyncio (flagged as "mandatory by
    Phase 4" in the roadmap's prerequisites) — threading.Timer is the
    "small and ugly first" version. A real async event loop is Phase 7's
    job, once wake-word + voice + tools + memory all need to run
    concurrently at once.
    """
    label = label or "unnamed timer"

    def _fire():
        print(f"\n⏰ Timer done: {label}\n")

    timer = threading.Timer(duration_seconds, _fire)
    timer.daemon = True
    timer.start()

    return {
        "status": "scheduled",
        "duration_seconds": duration_seconds,
        "label": label,
        "scheduled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": f"Real timer running in the background — will print to the console in {duration_seconds}s.",
    }


def search_web(query: str, max_results: int = 3) -> dict:
    """
    Real web search via Tavily (https://tavily.com), built for LLM
    agents rather than human-facing search results. Requires
    TAVILY_API_KEY in .env — free tier covers plenty for dev/testing.

    Returns an error dict (not an exception) on missing key or network
    failure, so the LLM can see what went wrong and explain it to the
    user instead of the whole turn crashing.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {
            "error": "TAVILY_API_KEY not set. Sign up free at https://tavily.com and add it to .env."
        }

    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "max_results": max_results},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Web search failed: {e}"}

    results = [
        {
            "title": r.get("title"),
            "url": r.get("url"),
            "snippet": (r.get("content") or "")[:300],
        }
        for r in data.get("results", [])[:max_results]
    ]
    return {"query": query, "results": results}


TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "set_timer": set_timer,
    "search_web": search_web,
}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name."}
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_timer",
            "description": "Set a real background timer for a given duration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration_seconds": {"type": "integer", "description": "Seconds."},
                    "label": {"type": "string", "description": "Optional name."},
                },
                "required": ["duration_seconds"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information on a topic or question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "max_results": {
                        "type": "integer",
                        "description": "How many results to return (default 3).",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def execute_tool(name: str, arguments: dict) -> dict:
    if name not in TOOL_FUNCTIONS:
        raise KeyError(f"Unknown tool: {name}")
    return TOOL_FUNCTIONS[name](**arguments)