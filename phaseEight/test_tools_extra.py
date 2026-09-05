"""
test_tools_extra.py -- Phase 8 milestone 1 tests for the accuracy layer.

Fully mocked: no network, no API keys, no clock dependence beyond
"the clock is the clock". Every outbound HTTP call is faked at the
`requests` seam.

Same caveat this project has earned the hard way in every phase so far
and restated in phaseEight/README.md: this suite proves the mapping,
truncation, and error-handling logic is right. It cannot prove
Open-Meteo and Tavily actually respond the way the code assumes they
do -- that took a real, live call, logged in the README's verification
section.
"""

import pytest
import requests

import tools_extra


# ---------------------------------------------------------------------------
# get_current_time
# ---------------------------------------------------------------------------

def test_get_current_time_returns_all_expected_fields():
    now = tools_extra.get_current_time()
    for key in (
        "iso", "date", "time_24h", "time_12h", "weekday",
        "spoken", "timezone", "utc_offset", "utc_iso",
    ):
        assert key in now, f"missing {key}"


def test_get_current_time_fields_agree_with_each_other():
    """The several formats are a convenience for the model, not
    independent sources -- if they ever disagree, the tool is handing
    the model a contradiction to resolve, which is exactly the kind of
    thing it resolves by inventing something."""
    now = tools_extra.get_current_time()
    assert now["iso"].startswith(now["date"])
    assert now["time_24h"] in now["iso"]
    assert now["weekday"] in now["spoken"]
    assert now["date"][:4] in now["spoken"]


def test_get_current_time_takes_no_arguments():
    """Guards the schema: an empty properties block means DeepSeek will
    call this with {}, so the function must accept exactly that."""
    assert tools_extra.get_current_time() is not None


# ---------------------------------------------------------------------------
# get_weather
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, payload, raise_for_status=None):
        self._payload = payload
        self._raise = raise_for_status

    def raise_for_status(self):
        if self._raise:
            raise self._raise

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


GEOCODE_HIT = {
    "results": [
        {
            "name": "Toronto",
            "admin1": "Ontario",
            "country": "Canada",
            "latitude": 43.7,
            "longitude": -79.42,
        }
    ]
}

FORECAST_HIT = {
    "current": {
        "time": "2026-08-31T13:15",
        "temperature_2m": 20.3,
        "apparent_temperature": 23.4,
        "relative_humidity_2m": 97,
        "precipitation": 0.0,
        "weather_code": 3,
        "wind_speed_10m": 5.1,
    }
}


@pytest.fixture
def fake_weather_api(monkeypatch):
    """Routes the two Open-Meteo calls by URL so a test can override
    either half independently."""
    responses = {"geocode": GEOCODE_HIT, "forecast": FORECAST_HIT}

    def fake_get(url, **kwargs):
        key = "geocode" if url == tools_extra.GEOCODE_URL else "forecast"
        payload = responses[key]
        if isinstance(payload, requests.exceptions.RequestException):
            raise payload
        return _FakeResponse(payload)

    monkeypatch.setattr(tools_extra.requests, "get", fake_get)
    return responses


def test_get_weather_returns_real_conditions(fake_weather_api):
    result = tools_extra.get_weather("Toronto")
    assert result["temp_c"] == 20.3
    assert result["condition"] == "overcast"  # WMO code 3
    assert result["source"] == "Open-Meteo"


def test_get_weather_reports_the_resolved_place_not_the_query(fake_weather_api):
    """A geocoding near-miss silently changes what the numbers mean --
    'Cambridge' resolving to the wrong country reads as a correct
    answer unless the resolved name comes back with it."""
    result = tools_extra.get_weather("toronto")
    assert result["location"] == "Toronto, Ontario, Canada"


def test_get_weather_has_no_stub_note_field(fake_weather_api):
    """Phase 4's version carried a `note` admitting it was fake data.
    Its absence here is the actual point of this tool existing."""
    assert "note" not in tools_extra.get_weather("Toronto")


def test_get_weather_unknown_wmo_code_does_not_invent_a_condition(fake_weather_api):
    fake_weather_api["forecast"] = {"current": {"weather_code": 12345, "temperature_2m": 5}}
    assert tools_extra.get_weather("Toronto")["condition"] == "unknown"


def test_get_weather_unfindable_place_returns_error_not_exception(fake_weather_api):
    fake_weather_api["geocode"] = {"results": []}
    result = tools_extra.get_weather("Xyzzyville")
    assert "error" in result
    assert "Xyzzyville" in result["error"]


def test_get_weather_network_failure_returns_error_not_exception(fake_weather_api):
    """Matches the convention every Phase 4/6 tool follows: the model
    should see what went wrong and be able to say so, rather than the
    whole voice turn dying on an exception."""
    fake_weather_api["geocode"] = requests.exceptions.ConnectionError("no route to host")
    result = tools_extra.get_weather("Toronto")
    assert "error" in result


def test_get_weather_unreadable_json_returns_error_not_exception(fake_weather_api):
    fake_weather_api["forecast"] = ValueError("not json")
    result = tools_extra.get_weather("Toronto")
    assert "error" in result


# ---------------------------------------------------------------------------
# search_web
# ---------------------------------------------------------------------------

TAVILY_HIT = {
    "answer": "A synthesized answer across the results.",
    "results": [
        {
            "title": "First result",
            "url": "https://example.com/1",
            "content": "x" * 2000,
            "published_date": "2026-08-30",
        },
        {"title": "Second result", "url": "https://example.com/2", "content": "short"},
    ],
}


@pytest.fixture
def fake_tavily(monkeypatch):
    """Captures the outbound request so tests can assert on what was
    actually asked for, not just what came back."""
    captured = {"payload": None, "headers": None, "response": TAVILY_HIT}

    def fake_post(url, json=None, headers=None, **kwargs):
        captured["payload"] = json
        captured["headers"] = headers
        payload = captured["response"]
        if isinstance(payload, requests.exceptions.RequestException):
            raise payload
        return _FakeResponse(payload)

    monkeypatch.setattr(tools_extra.requests, "post", fake_post)
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    return captured


def test_search_web_without_api_key_returns_actionable_error(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    result = tools_extra.search_web("anything")
    assert "TAVILY_API_KEY" in result["error"]


def test_search_web_authenticates_with_a_bearer_header(fake_tavily):
    """Phase 4 put the key in the request body. The header form is
    Tavily's current documented auth -- confirmed working against the
    live API, see README."""
    tools_extra.search_web("query")
    assert fake_tavily["headers"]["Authorization"] == "Bearer tvly-test-key"
    assert "api_key" not in fake_tavily["payload"]


def test_search_web_returns_the_synthesized_answer(fake_tavily):
    """The single biggest upgrade over Phase 4's version: without this,
    the model stitches an answer out of truncated snippets and fills
    the seams itself."""
    assert tools_extra.search_web("query")["answer"] == TAVILY_HIT["answer"]


def test_search_web_omits_answer_key_when_tavily_sends_none(fake_tavily):
    fake_tavily["response"] = {"results": []}
    assert "answer" not in tools_extra.search_web("query")


def test_search_web_requests_advanced_depth_and_answer(fake_tavily):
    tools_extra.search_web("query")
    assert fake_tavily["payload"]["search_depth"] == "advanced"
    assert fake_tavily["payload"]["include_answer"] == "advanced"


def test_search_web_only_sends_time_range_when_asked(fake_tavily):
    """An always-present time_range would quietly hide older results
    for questions where age is irrelevant."""
    tools_extra.search_web("query")
    assert "time_range" not in fake_tavily["payload"]

    tools_extra.search_web("query", time_range="week")
    assert fake_tavily["payload"]["time_range"] == "week"


def test_search_web_passes_topic_through(fake_tavily):
    tools_extra.search_web("query", topic="news")
    assert fake_tavily["payload"]["topic"] == "news"


def test_search_web_truncates_snippets_to_the_configured_length(fake_tavily):
    results = tools_extra.search_web("query")["results"]
    assert len(results[0]["snippet"]) == tools_extra.SEARCH_SNIPPET_CHARS


def test_search_web_snippets_are_longer_than_phase_fours_300(fake_tavily):
    """Regression guard on the actual complaint: Phase 4 cut every
    snippet at 300 characters, routinely mid-fact."""
    assert len(tools_extra.search_web("query")["results"][0]["snippet"]) > 300


def test_search_web_passes_published_date_through_when_present(fake_tavily):
    results = tools_extra.search_web("query")["results"]
    assert results[0]["published_date"] == "2026-08-30"
    assert "published_date" not in results[1]


def test_search_web_network_failure_returns_error_not_exception(fake_tavily):
    fake_tavily["response"] = requests.exceptions.Timeout("timed out")
    assert "error" in tools_extra.search_web("query")


def test_search_web_respects_max_results(fake_tavily):
    assert len(tools_extra.search_web("query", max_results=1)["results"]) == 1


# ---------------------------------------------------------------------------
# Registry contract
# ---------------------------------------------------------------------------

def test_schemas_and_functions_cover_the_same_names():
    """orchestrator.register_tools() raises on a mismatch; catching it
    here points at the actual typo instead of at a startup crash."""
    schema_names = {s["function"]["name"] for s in tools_extra.TOOL_SCHEMAS}
    assert schema_names == set(tools_extra.TOOL_FUNCTIONS)


def test_overrides_reuse_phase_four_tool_names_exactly():
    """get_weather and search_web must match Phase 4's names character
    for character -- a typo would register a second, differently-named
    tool alongside the broken original rather than replacing it, and
    nothing would fail loudly."""
    assert "get_weather" in tools_extra.TOOL_FUNCTIONS
    assert "search_web" in tools_extra.TOOL_FUNCTIONS
