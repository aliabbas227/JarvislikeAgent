"""
test_agent.py — mocked tests for Phase 4 (DeepSeek backend).

Same philosophy as phaseThree/test_wake_word.py: no real network calls,
no real API key needed to run the suite. The DeepSeek client is mocked
at the point run_turn() calls it.
"""

from unittest.mock import MagicMock, patch
import json

import pytest

from agent import run_turn
from tools import execute_tool, TOOL_FUNCTIONS


# ---------------------------------------------------------------------
# Helpers for building fake OpenAI-shaped responses
# ---------------------------------------------------------------------

def make_tool_call(call_id, name, arguments: dict):
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    tc.model_dump.return_value = {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }
    return tc


def make_response(content=None, tool_calls=None):
    response = MagicMock()
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    response.choices = [MagicMock(message=message)]
    return response


# ---------------------------------------------------------------------
# tools.py
# ---------------------------------------------------------------------

def test_execute_tool_get_weather():
    result = execute_tool("get_weather", {"location": "Sydney"})
    assert result["location"] == "Sydney"
    assert "condition" in result


def test_execute_tool_set_timer_default_label():
    with patch("tools.threading.Timer") as mock_timer_cls:
        mock_timer_cls.return_value = MagicMock()
        result = execute_tool("set_timer", {"duration_seconds": 60})
    assert result["duration_seconds"] == 60
    assert result["label"] == "unnamed timer"


def test_execute_tool_unknown_raises():
    with pytest.raises(KeyError):
        execute_tool("nonexistent_tool", {})


def test_all_registered_tools_are_callable():
    for name, fn in TOOL_FUNCTIONS.items():
        assert callable(fn), f"{name} is not callable"


# ---------------------------------------------------------------------
# tools.py — real tools (timer, web search, Spotify), all I/O mocked
# ---------------------------------------------------------------------

def test_set_timer_schedules_background_thread():
    with patch("tools.threading.Timer") as mock_timer_cls:
        mock_timer_instance = MagicMock()
        mock_timer_cls.return_value = mock_timer_instance

        result = execute_tool("set_timer", {"duration_seconds": 600, "label": "umbrella"})

        # Confirms a real threading.Timer was constructed with the right
        # duration and set to run in the background, not just a stub dict.
        mock_timer_cls.assert_called_once()
        args, _ = mock_timer_cls.call_args
        assert args[0] == 600
        assert mock_timer_instance.daemon is True
        mock_timer_instance.start.assert_called_once()
        assert result["status"] == "scheduled"
        assert result["label"] == "umbrella"


def test_search_web_missing_api_key_returns_error_not_crash(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    result = execute_tool("search_web", {"query": "test query"})
    assert "error" in result
    assert "TAVILY_API_KEY" in result["error"]


def test_search_web_success(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake_key_for_testing")

    fake_response = MagicMock()
    fake_response.json.return_value = {
        "results": [
            {"title": "Result One", "url": "https://example.com/1", "content": "Some content here."}
        ]
    }
    fake_response.raise_for_status.return_value = None

    with patch("tools.requests.post", return_value=fake_response) as mock_post:
        result = execute_tool("search_web", {"query": "sydney weather", "max_results": 1})

    mock_post.assert_called_once()
    assert result["query"] == "sydney weather"
    assert result["results"][0]["title"] == "Result One"


def test_search_web_network_failure_returns_error_not_crash(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake_key_for_testing")

    with patch("tools.requests.post", side_effect=requests_exceptions_connection_error()):
        result = execute_tool("search_web", {"query": "anything"})

    assert "error" in result


def requests_exceptions_connection_error():
    import requests
    return requests.exceptions.ConnectionError("simulated network failure")


# ---------------------------------------------------------------------
# agent.py — run_turn with a mocked client
# ---------------------------------------------------------------------

def test_run_turn_no_tool_call_returns_text():
    client = MagicMock()
    client.chat.completions.create.return_value = make_response(
        content="Just a plain answer.", tool_calls=None
    )
    messages = [{"role": "user", "content": "hi"}]

    result = run_turn(client, messages)

    assert result == "Just a plain answer."
    assert messages[-1]["role"] == "assistant"


def test_run_turn_single_tool_call_then_final_answer():
    client = MagicMock()
    tool_call = make_tool_call("call_1", "get_weather", {"location": "Sydney"})
    first_response = make_response(content=None, tool_calls=[tool_call])
    second_response = make_response(content="It's partly cloudy in Sydney.", tool_calls=None)
    client.chat.completions.create.side_effect = [first_response, second_response]

    messages = [{"role": "user", "content": "what's the weather in Sydney"}]
    result = run_turn(client, messages)

    assert result == "It's partly cloudy in Sydney."
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert "Sydney" in tool_messages[0]["content"]


def test_run_turn_two_sequential_tool_calls_different_tools():
    """
    Multi-step chain: get_weather -> (model reasons about the result) ->
    set_timer -> final answer. Proves run_turn() correctly loops through
    more than one hop with *different* tools, not just repeats of one —
    same pattern as "check the weather, then set a timer if it's not
    sunny" tried manually against the real API.
    """
    client = MagicMock()
    weather_call = make_tool_call("call_1", "get_weather", {"location": "Sydney"})
    timer_call = make_tool_call("call_2", "set_timer", {"duration_seconds": 600, "label": "bring an umbrella"})

    first_response = make_response(content=None, tool_calls=[weather_call])
    second_response = make_response(content=None, tool_calls=[timer_call])
    third_response = make_response(
        content="It's partly cloudy in Sydney, so I set a 10-minute umbrella reminder.",
        tool_calls=None,
    )
    client.chat.completions.create.side_effect = [first_response, second_response, third_response]

    messages = [{"role": "user", "content": "check the weather in Sydney and set a reminder if it's not sunny"}]
    result = run_turn(client, messages)

    assert "umbrella" in result.lower()
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    # Two distinct tool hops should have run, in order: weather first, timer second.
    assert len(tool_messages) == 2
    assert "Sydney" in tool_messages[0]["content"]
    assert "600" in tool_messages[1]["content"]
    assert client.chat.completions.create.call_count == 3


def test_run_turn_unknown_tool_surfaces_error_not_crash():
    client = MagicMock()
    bad_call = make_tool_call("call_1", "totally_fake_tool", {})
    first_response = make_response(content=None, tool_calls=[bad_call])
    second_response = make_response(content="Sorry, I couldn't do that.", tool_calls=None)
    client.chat.completions.create.side_effect = [first_response, second_response]

    messages = [{"role": "user", "content": "do the impossible thing"}]
    result = run_turn(client, messages)

    assert result == "Sorry, I couldn't do that."
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert "error" in tool_messages[0]["content"]


def test_run_turn_respects_max_hops():
    client = MagicMock()
    looping_call = make_tool_call("call_1", "get_weather", {"location": "Nowhere"})
    client.chat.completions.create.return_value = make_response(
        content=None, tool_calls=[looping_call]
    )

    messages = [{"role": "user", "content": "trigger a loop"}]
    result = run_turn(client, messages)

    assert "gave up" in result.lower()