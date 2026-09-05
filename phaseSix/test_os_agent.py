"""
test_os_agent.py — Phase 6 tests for the confirmation gate in
os_agent.py, specifically the password requirement added after manual
testing flagged that a plain "yes" is too easy to satisfy by accident.

Mocks getpass/input (never actually blocks on real terminal input) and
mocks tools.confirm_pending_action/cancel_pending_action (already
covered directly by test_tools.py) so these tests isolate exactly one
thing: does _prompt_for_confirmation()/_handle_tool_result() correctly
gate on the configured password.

Requires phaseFour/llm_client.py to be importable (see os_agent.py's
PHASE_FOUR_PATH) — same precondition os_agent.py itself has.
"""

import os_agent


def test_non_confirmation_results_pass_through_unchanged():
    result = {"status": "deleted", "path": "x"}
    assert os_agent._handle_tool_result(result) is result


def test_password_configured_correct_password_confirms(monkeypatch):
    monkeypatch.setattr(os_agent, "CONFIRM_PASSWORD", "secret123")
    monkeypatch.setattr(os_agent.getpass, "getpass", lambda prompt="": "secret123")

    calls = []
    monkeypatch.setattr(
        os_agent, "confirm_pending_action",
        lambda token: calls.append(("confirm", token)) or {"status": "deleted"},
    )
    monkeypatch.setattr(
        os_agent, "cancel_pending_action",
        lambda token: calls.append(("cancel", token)) or {"status": "cancelled"},
    )

    result = os_agent._handle_tool_result(
        {"status": "confirmation_required", "token": "abc123", "message": "delete x"}
    )

    assert result == {"status": "deleted"}
    assert calls == [("confirm", "abc123")]


def test_password_configured_wrong_password_cancels(monkeypatch):
    monkeypatch.setattr(os_agent, "CONFIRM_PASSWORD", "secret123")
    monkeypatch.setattr(os_agent.getpass, "getpass", lambda prompt="": "wrongpass")

    calls = []
    monkeypatch.setattr(os_agent, "confirm_pending_action", lambda token: calls.append("confirm"))
    monkeypatch.setattr(
        os_agent, "cancel_pending_action",
        lambda token: calls.append("cancel") or {"status": "cancelled"},
    )

    result = os_agent._handle_tool_result(
        {"status": "confirmation_required", "token": "abc123", "message": "delete x"}
    )

    assert result == {"status": "cancelled"}
    assert calls == ["cancel"]


def test_password_configured_empty_input_cancels(monkeypatch):
    monkeypatch.setattr(os_agent, "CONFIRM_PASSWORD", "secret123")
    monkeypatch.setattr(os_agent.getpass, "getpass", lambda prompt="": "")

    calls = []
    monkeypatch.setattr(os_agent, "confirm_pending_action", lambda token: calls.append("confirm"))
    monkeypatch.setattr(
        os_agent, "cancel_pending_action",
        lambda token: calls.append("cancel") or {"status": "cancelled"},
    )

    os_agent._handle_tool_result({"status": "confirmation_required", "token": "t", "message": "m"})

    assert calls == ["cancel"]


def test_no_password_configured_falls_back_to_yes_no(monkeypatch):
    monkeypatch.setattr(os_agent, "CONFIRM_PASSWORD", None)
    monkeypatch.setattr("builtins.input", lambda prompt="": "yes")

    calls = []
    monkeypatch.setattr(
        os_agent, "confirm_pending_action",
        lambda token: calls.append("confirm") or {"status": "deleted"},
    )
    monkeypatch.setattr(os_agent, "cancel_pending_action", lambda token: calls.append("cancel"))

    result = os_agent._handle_tool_result({"status": "confirmation_required", "token": "t", "message": "m"})

    assert result == {"status": "deleted"}
    assert calls == ["confirm"]


def test_no_password_configured_anything_but_yes_cancels(monkeypatch):
    monkeypatch.setattr(os_agent, "CONFIRM_PASSWORD", None)
    monkeypatch.setattr("builtins.input", lambda prompt="": "sure")

    calls = []
    monkeypatch.setattr(os_agent, "confirm_pending_action", lambda token: calls.append("confirm"))
    monkeypatch.setattr(
        os_agent, "cancel_pending_action",
        lambda token: calls.append("cancel") or {"status": "cancelled"},
    )

    os_agent._handle_tool_result({"status": "confirmation_required", "token": "t", "message": "m"})

    assert calls == ["cancel"]


def test_getpass_failure_fails_closed(monkeypatch):
    """A broken/no-tty terminal must never be treated as approval."""
    monkeypatch.setattr(os_agent, "CONFIRM_PASSWORD", "secret123")

    def _raise(prompt=""):
        raise Exception("no tty available")

    monkeypatch.setattr(os_agent.getpass, "getpass", _raise)

    calls = []
    monkeypatch.setattr(os_agent, "confirm_pending_action", lambda token: calls.append("confirm"))
    monkeypatch.setattr(
        os_agent, "cancel_pending_action",
        lambda token: calls.append("cancel") or {"status": "cancelled"},
    )

    os_agent._handle_tool_result({"status": "confirmation_required", "token": "t", "message": "m"})

    assert calls == ["cancel"]


def test_error_outcome_is_printed_directly_to_terminal(monkeypatch, capsys):
    """The human should see the real reason a confirmation failed (e.g.
    'expired') printed directly, not only whatever the model chooses to
    say about it in its reply."""
    monkeypatch.setattr(os_agent, "CONFIRM_PASSWORD", "secret123")
    monkeypatch.setattr(os_agent.getpass, "getpass", lambda prompt="": "secret123")
    monkeypatch.setattr(
        os_agent, "confirm_pending_action",
        lambda token: {"error": "That confirmation expired after sitting unanswered for 300 seconds."},
    )

    os_agent._handle_tool_result({"status": "confirmation_required", "token": "t", "message": "delete x"})

    captured = capsys.readouterr()
    assert "[NOTE]" in captured.out
    assert "expired" in captured.out.lower()


def test_success_outcome_is_printed_directly_to_terminal(monkeypatch, capsys):
    monkeypatch.setattr(os_agent, "CONFIRM_PASSWORD", "secret123")
    monkeypatch.setattr(os_agent.getpass, "getpass", lambda prompt="": "secret123")
    monkeypatch.setattr(
        os_agent, "confirm_pending_action",
        lambda token: {"status": "deleted", "path": "C:\\some\\file.txt"},
    )

    os_agent._handle_tool_result({"status": "confirmation_required", "token": "t", "message": "delete x"})

    captured = capsys.readouterr()
    assert "[NOTE]" in captured.out
    assert "deleted" in captured.out.lower()


# ---------------------------------------------------------------------------
# _summarize_for_log — keeps large text fields (screen OCR, file content)
# out of the log file while leaving the model's own view untouched
# ---------------------------------------------------------------------------

def test_summarize_for_log_truncates_long_text_field():
    long_text = "x" * 1000
    result = {"text": long_text, "region": "full screen"}

    summary = os_agent._summarize_for_log(result)

    assert len(summary["text"]) < len(long_text)
    assert summary["text"].startswith("x" * 50)
    assert "truncated for log" in summary["text"]
    assert summary["region"] == "full screen"  # untouched, not a text/content field


def test_summarize_for_log_leaves_short_text_untouched():
    result = {"text": "short", "content": "also short"}

    summary = os_agent._summarize_for_log(result)

    assert summary == result


def test_summarize_for_log_does_not_mutate_the_original_result():
    long_text = "y" * 1000
    result = {"text": long_text}

    os_agent._summarize_for_log(result)

    assert result["text"] == long_text  # the model's actual view is untouched


def test_summarize_for_log_handles_missing_text_fields():
    result = {"status": "deleted", "path": "x"}
    assert os_agent._summarize_for_log(result) == result
