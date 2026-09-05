"""
Automated tests for jarvis.py — Phase 1 (Ollama backend).

These mock requests.post so they run instantly, for free, with no
dependency on Ollama actually being installed or running. They test
*your* logic (history trimming, error handling, backend routing) —
not Ollama itself.

Run with:
    pip install pytest
    pytest test_jarvis.py -v
"""

import os
import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, os.path.dirname(__file__))
import jarvis


# ---------------------------------------------------------------------------
# trim_history()
# ---------------------------------------------------------------------------

def test_trim_history_leaves_short_lists_untouched():
    messages = [{"role": "user", "content": "hi"}] * 5
    assert jarvis.trim_history(messages) == messages


def test_trim_history_caps_long_lists():
    messages = [{"role": "user", "content": str(i)} for i in range(100)]
    trimmed = jarvis.trim_history(messages)
    assert len(trimmed) == jarvis.MAX_HISTORY_MESSAGES
    assert trimmed[-1]["content"] == "99"  # keeps most recent, not oldest


# ---------------------------------------------------------------------------
# call_ollama() — success path
# ---------------------------------------------------------------------------

def test_call_ollama_returns_text_on_success():
    fake_response = MagicMock()
    fake_response.json.return_value = {"message": {"role": "assistant", "content": "Hello there!"}}
    fake_response.raise_for_status.return_value = None

    with patch("jarvis.requests.post", return_value=fake_response) as mock_post:
        result = jarvis.call_ollama([{"role": "user", "content": "hi"}])

    assert result == "Hello there!"
    _, kwargs = mock_post.call_args
    assert kwargs["json"]["model"] == jarvis.OLLAMA_MODEL
    # system prompt must be prepended
    assert kwargs["json"]["messages"][0]["role"] == "system"
    assert kwargs["json"]["messages"][-1] == {"role": "user", "content": "hi"}


def test_system_prompt_includes_current_date():
    """Regression test for the 'Jarvis gets the date wrong' bug —
    local models have no innate sense of 'now', so the real date/time
    must be injected into the system prompt on every call."""
    prompt = jarvis.get_system_prompt()
    year = str(datetime.now().year)
    assert year in prompt
    assert "Current date and time" in prompt


# ---------------------------------------------------------------------------
# call_ollama() — failure paths (this is the important part)
# ---------------------------------------------------------------------------

def test_call_ollama_handles_connection_error_gracefully():
    with patch("jarvis.requests.post", side_effect=requests.exceptions.ConnectionError):
        result = jarvis.call_ollama([{"role": "user", "content": "hi"}])
    assert "ollama" in result.lower()
    assert "running" in result.lower() or "serve" in result.lower()


def test_call_ollama_handles_timeout_gracefully():
    with patch("jarvis.requests.post", side_effect=requests.exceptions.Timeout):
        result = jarvis.call_ollama([{"role": "user", "content": "hi"}])
    assert "too long" in result.lower() or "timeout" in result.lower() or "loading" in result.lower()


def test_call_ollama_handles_missing_model_404():
    fake_response = MagicMock()
    fake_response.status_code = 404
    http_err = requests.exceptions.HTTPError(response=fake_response)

    fake_post_response = MagicMock()
    fake_post_response.raise_for_status.side_effect = http_err

    with patch("jarvis.requests.post", return_value=fake_post_response):
        result = jarvis.call_ollama([{"role": "user", "content": "hi"}])
    assert "pull" in result.lower()
    assert jarvis.OLLAMA_MODEL in result


def test_call_ollama_never_raises():
    """The whole point of wrapping the call: a broken request should
    never crash the CLI loop."""
    with patch("jarvis.requests.post", side_effect=requests.exceptions.ConnectionError):
        try:
            jarvis.call_ollama([{"role": "user", "content": "hi"}])
        except Exception as e:
            pytest.fail(f"call_ollama raised instead of handling it: {e}")


def test_call_ollama_handles_malformed_response():
    fake_response = MagicMock()
    fake_response.json.return_value = {"unexpected": "shape"}
    fake_response.raise_for_status.return_value = None

    with patch("jarvis.requests.post", return_value=fake_response):
        result = jarvis.call_ollama([{"role": "user", "content": "hi"}])
    assert result.startswith("[")  # graceful error marker, not a crash


# ---------------------------------------------------------------------------
# call_llm() backend routing
# ---------------------------------------------------------------------------

def test_call_llm_routes_to_ollama_by_default(monkeypatch):
    monkeypatch.setattr(jarvis, "LLM_BACKEND", "ollama")
    with patch("jarvis.call_ollama", return_value="ollama reply") as mock_ollama:
        result = jarvis.call_llm([{"role": "user", "content": "hi"}])
    assert result == "ollama reply"
    mock_ollama.assert_called_once()


def test_call_llm_routes_to_anthropic_when_configured(monkeypatch):
    monkeypatch.setattr(jarvis, "LLM_BACKEND", "anthropic")
    with patch("jarvis.call_anthropic", return_value="claude reply") as mock_claude:
        result = jarvis.call_llm([{"role": "user", "content": "hi"}])
    assert result == "claude reply"
    mock_claude.assert_called_once()


# ---------------------------------------------------------------------------
# check_ollama_available()
# ---------------------------------------------------------------------------

def test_check_ollama_available_true_when_reachable():
    with patch("jarvis.requests.get", return_value=MagicMock()):
        assert jarvis.check_ollama_available() is True


def test_check_ollama_available_false_when_unreachable():
    with patch("jarvis.requests.get", side_effect=requests.exceptions.ConnectionError):
        assert jarvis.check_ollama_available() is False


# ---------------------------------------------------------------------------
# main() loop behavior
# ---------------------------------------------------------------------------

def test_main_exits_cleanly_on_quit(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "quit")
    with patch("jarvis.check_ollama_available", return_value=True):
        jarvis.main()
    captured = capsys.readouterr()
    assert "Goodbye" in captured.out


def test_main_handles_empty_input_without_calling_llm(monkeypatch):
    inputs = iter(["", "   ", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    with patch("jarvis.check_ollama_available", return_value=True), \
         patch("jarvis.call_llm") as mock_call_llm:
        jarvis.main()
    mock_call_llm.assert_not_called()


def test_main_warns_when_ollama_unreachable(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "quit")
    with patch("jarvis.check_ollama_available", return_value=False):
        jarvis.main()
    captured = capsys.readouterr()
    assert "WARNING" in captured.out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
