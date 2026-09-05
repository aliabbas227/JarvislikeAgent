"""
test_orchestrator.py -- Phase 7 tests for orchestrator.think(),
orchestrator.run_turn(), and orchestrator.run_conversation().

think() covers the DeepSeek-backed tool-calling handoff itself -- that
it lazily builds and caches Phase 4's client and delegates to
_run_tool_calling_turn() (this file's own merged-tool loop as of
milestone 4 -- see its docstring for why it's no longer a direct reuse
of either Phase 4's or Phase 6's own loop). The merged tool registry
(TOOL_SCHEMAS/execute_tool, combining Phase 4's and Phase 6's tools.py)
and the confirmation gate (_prompt_for_confirmation/_handle_tool_result,
reused from Phase 6's os_agent.py and adapted for voice) each get their
own test section below, largely mirroring phaseSix/test_os_agent.py's
own suite for the parts that are direct reuse.

run_turn() covers the per-recording logic: record -> transcribe ->
exit-phrase check -> think() -> speak, and how each failure mode maps
to a TurnOutcome -- tested by faking think() itself rather than
DeepSeek, since think()'s own contract (mutates history in place,
returns the final text) is what run_turn() actually depends on.
run_conversation() covers the follow-up-window chaining logic on top
of that -- tested purely by faking run_turn() itself, so it doesn't
need any of run_turn's own mocks (record_until_silence,
transcribe_audio, think, etc.) at all.

The outer run_orchestrator() wake/mic hand-off loop is thin glue over
WakeWordGate (already covered by test_wake_gate.py) and run_conversation
(covered here) and isn't re-tested in detail.

Requires orchestrator.py to be importable, i.e. phaseOne/jarvis.py,
phaseTwo/voice_io.py, phaseThree/wake_word.py, phaseFour's
llm_client.py/tools.py, phaseFive's memory.py/memory_chat.py, and
phaseSix's tools.py present at the sibling paths orchestrator.py
expects (see its PHASE_*_PATH constants) -- and the real `openai`,
`chromadb` packages installed (only building real clients/stores needs
real credentials/data, which nothing here does since think()/
_get_memory_store() are always faked below). openwakeword itself is
stubbed via sys.modules so the real package doesn't need to be
installed just to run this suite -- same "lazy import, fake the
module" pattern Phase 6's own test suite used for
PIL/pytesseract/pyautogui (neither of those needs stubbing here either
-- phaseSix's tools.py only imports them lazily inside the functions
that use them, so loading the module itself never touches them).
"""

import sys
import types

import pytest


@pytest.fixture(autouse=True, scope="module")
def stub_openwakeword():
    if "openwakeword" not in sys.modules:
        fake_pkg = types.ModuleType("openwakeword")
        fake_model_module = types.ModuleType("openwakeword.model")

        class _FakeModel:
            def __init__(self, *args, **kwargs):
                pass

            def predict(self, audio):
                return {}

        fake_model_module.Model = _FakeModel
        fake_pkg.model = fake_model_module
        sys.modules["openwakeword"] = fake_pkg
        sys.modules["openwakeword.model"] = fake_model_module
    yield


@pytest.fixture
def orch(stub_openwakeword):
    import orchestrator as orch_module
    return orch_module


@pytest.fixture(autouse=True)
def no_memory_by_default(orch, monkeypatch):
    """Memory is off by default in this suite -- forces
    _get_memory_store()'s own sticky-failure short-circuit (see its
    docstring) so it returns None without ever constructing a real
    MemoryStore. Prevents every existing (memory-agnostic) test from
    accidentally hitting real chromadb / real disk I/O / a real
    embedding-model load just by exercising a normal turn. Tests that
    actually want a memory store override this by monkeypatching
    _get_memory_store directly (a fake store) or by resetting
    _memory_store/_memory_store_failed to exercise the real caching
    logic (see the _get_memory_store() tests below)."""
    monkeypatch.setattr(orch, "_memory_store", None)
    monkeypatch.setattr(orch, "_memory_store_failed", True)


# ---------------------------------------------------------------------------
# think() -- the DeepSeek/tool-calling handoff
# ---------------------------------------------------------------------------

def test_think_builds_client_lazily_and_delegates_to_tool_loop(orch, monkeypatch):
    monkeypatch.setattr(orch, "_deepseek_client", None)

    built = []
    fake_client = object()
    monkeypatch.setattr(orch, "build_client", lambda: built.append(1) or fake_client)

    received = {}

    def fake_run_tool_calling_turn(client, messages):
        received["client"] = client
        received["messages"] = messages
        messages.append({"role": "assistant", "content": "final answer"})
        return "final answer"

    monkeypatch.setattr(orch, "_run_tool_calling_turn", fake_run_tool_calling_turn)

    history = [{"role": "user", "content": "hi"}]
    reply = orch.think(history)

    assert reply == "final answer"
    assert received["client"] is fake_client
    assert received["messages"] is history  # mutated in place, not copied
    assert history[-1] == {"role": "assistant", "content": "final answer"}
    assert built == [1]


def test_think_only_builds_client_once(orch, monkeypatch):
    """A second think() call should reuse the cached client rather than
    calling build_client() again -- same one-time-cost shape as
    build_model() for the wake-word model."""
    monkeypatch.setattr(orch, "_deepseek_client", None)

    build_calls = []
    monkeypatch.setattr(orch, "build_client", lambda: build_calls.append(1) or object())
    monkeypatch.setattr(orch, "_run_tool_calling_turn", lambda client, messages: "ok")

    orch.think([])
    orch.think([])

    assert len(build_calls) == 1


# ---------------------------------------------------------------------------
# Merged tool registry -- milestone 4's TOOL_SCHEMAS/execute_tool
# ---------------------------------------------------------------------------
# Phase 4's and Phase 6's tools.py are both loaded (see
# _load_module_from_path()) and merged into one registry. These tests
# cover the merge itself, not either phase's own tool implementations
# (get_weather's stub shape, delete_file's validation, etc.) -- those
# are already covered by phaseFour/test_agent.py and
# phaseSix/test_tools.py respectively.

def test_tool_schemas_include_both_phase_four_and_phase_six_tools(orch):
    names = {schema["function"]["name"] for schema in orch.TOOL_SCHEMAS}
    assert {"get_weather", "set_timer", "search_web"} <= names  # Phase 4
    assert {"list_directory", "read_file", "delete_file", "click"} <= names  # Phase 6


def test_tool_schemas_has_no_duplicate_names(orch):
    names = [schema["function"]["name"] for schema in orch.TOOL_SCHEMAS]
    assert len(names) == len(set(names))


def test_execute_tool_dispatches_to_phase_four_tool(orch):
    result = orch.execute_tool("get_weather", {"location": "Sydney"})
    assert result["location"] == "Sydney"


def test_execute_tool_dispatches_to_phase_six_tool(orch, tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    result = orch.execute_tool("list_directory", {"path": str(tmp_path)})
    assert result["entries"][0]["name"] == "a.txt"


def test_execute_tool_unknown_name_raises_keyerror(orch):
    with pytest.raises(KeyError):
        orch.execute_tool("not_a_real_tool", {})


def test_confirm_and_cancel_are_not_llm_reachable(orch):
    """Mirrors phaseSix/test_tools.py's own guardrail test: the model
    must never have a tool-calling path to confirm_pending_action or
    cancel_pending_action, in the MERGED registry either."""
    names = {schema["function"]["name"] for schema in orch.TOOL_SCHEMAS}
    assert "confirm_pending_action" not in names
    assert "cancel_pending_action" not in names


# ---------------------------------------------------------------------------
# register_tools() -- the Phase 8 extension seam
# ---------------------------------------------------------------------------
# Added so Phase 8 can fix two wrong tools and add a missing one
# without editing Phase 4's finished tools.py. These tests cover the
# seam itself; what Phase 8 registers through it is Phase 8's own
# business and is tested in phaseEight/test_app.py.

@pytest.fixture
def restore_registry(orch):
    """register_tools mutates module-level state -- snapshot and
    restore, or an override leaks into every test that runs after."""
    saved_schemas = list(orch.TOOL_SCHEMAS)
    saved_functions = dict(orch._TOOL_FUNCTIONS)
    yield
    orch.TOOL_SCHEMAS = saved_schemas
    orch._TOOL_FUNCTIONS.clear()
    orch._TOOL_FUNCTIONS.update(saved_functions)


def _schema(name):
    return {"type": "function", "function": {"name": name, "description": name, "parameters": {}}}


def test_register_tools_adds_a_new_tool(orch, restore_registry):
    orch.register_tools([_schema("brand_new")], {"brand_new": lambda: {"ok": True}})
    assert orch.execute_tool("brand_new", {}) == {"ok": True}
    assert any(s["function"]["name"] == "brand_new" for s in orch.TOOL_SCHEMAS)


def test_register_tools_replaces_an_existing_tool_by_name(orch, restore_registry):
    orch.register_tools([_schema("get_weather")], {"get_weather": lambda **kw: {"replaced": True}})
    assert orch.execute_tool("get_weather", {"location": "Sydney"}) == {"replaced": True}


def test_register_tools_replaces_the_schema_rather_than_appending(orch, restore_registry):
    """An override that left the old schema in place would show the
    model two contradictory descriptions of the same tool name."""
    orch.register_tools([_schema("get_weather")], {"get_weather": lambda **kw: {}})
    matching = [s for s in orch.TOOL_SCHEMAS if s["function"]["name"] == "get_weather"]
    assert len(matching) == 1
    assert matching[0]["function"]["description"] == "get_weather"


def test_register_tools_leaves_untouched_tools_alone(orch, restore_registry):
    before = set(orch._TOOL_FUNCTIONS)
    orch.register_tools([_schema("get_weather")], {"get_weather": lambda **kw: {}})
    assert set(orch._TOOL_FUNCTIONS) == before


def test_register_tools_rejects_a_schema_with_no_function(orch, restore_registry):
    """A tool the model can call and nothing can answer. Fails here
    rather than as a KeyError mid-turn, mid-conversation."""
    with pytest.raises(ValueError, match="same names"):
        orch.register_tools([_schema("orphan")], {})


def test_register_tools_rejects_a_function_with_no_schema(orch, restore_registry):
    """Dead code the model can never reach -- silent on its own."""
    with pytest.raises(ValueError, match="same names"):
        orch.register_tools([], {"unreachable": lambda: None})


def test_register_tools_leaves_registry_unchanged_when_it_rejects(orch, restore_registry):
    before_schemas = list(orch.TOOL_SCHEMAS)
    before_functions = dict(orch._TOOL_FUNCTIONS)
    with pytest.raises(ValueError):
        orch.register_tools([_schema("orphan")], {})
    assert orch.TOOL_SCHEMAS == before_schemas
    assert orch._TOOL_FUNCTIONS == before_functions


# ---------------------------------------------------------------------------
# register_speaker() -- the Phase 8 barge-in seam
# ---------------------------------------------------------------------------

@pytest.fixture
def restore_speaker(orch):
    saved = orch._speaker
    yield
    orch._speaker = saved


def test_speak_safe_uses_phase_twos_tts_by_default(orch, restore_speaker, monkeypatch):
    """Phase 7 on its own must speak exactly as it always has."""
    orch._speaker = None
    said = []
    monkeypatch.setattr(orch, "speak_text", lambda text: said.append(text))
    orch.speak_safe("hello")
    assert said == ["hello"]


def test_a_registered_speaker_takes_over(orch, restore_speaker, monkeypatch):
    said = []
    monkeypatch.setattr(orch, "speak_text", lambda text: said.append(("phase2", text)))
    orch.register_speaker(lambda text: said.append(("phase8", text)))
    orch.speak_safe("hello")
    assert said == [("phase8", "hello")]


def test_a_registered_speaker_still_gets_the_error_fallback(orch, restore_speaker, capsys):
    """The reason speak_safe catches SpeechError -- a dead audio driver
    must never kill the loop -- applies at least as much to a speaker
    with a mic and a transcription model behind it."""
    def broken(text):
        raise orch.SpeechError("audio device died")

    orch.register_speaker(broken)
    orch.speak_safe("hello")  # must not raise
    assert "text fallback" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# register_confirmation_handler() / _resolve_confirmation()
# ---------------------------------------------------------------------------
# Phase 8 brings destructive tools of its own, whose tokens Phase 6's
# confirm_pending_action() cannot redeem. Routing is by token prefix so
# the gate's behavior never depends on registry ordering or on matching
# error strings.

@pytest.fixture
def restore_handlers(orch):
    saved = dict(orch._confirmation_handlers)
    yield
    orch._confirmation_handlers.clear()
    orch._confirmation_handlers.update(saved)


def test_unregistered_tokens_go_to_phase_six(orch):
    """Phase 6's tokens are bare hex and it stays the default, so Phase
    7 running on its own is completely unaffected by this seam."""
    confirm, cancel = orch._resolve_confirmation("abc12345")
    assert confirm is orch.confirm_pending_action
    assert cancel is orch.cancel_pending_action


def test_a_registered_prefix_claims_its_tokens(orch, restore_handlers):
    sentinel_confirm, sentinel_cancel = object(), object()
    orch.register_confirmation_handler("zz-", sentinel_confirm, sentinel_cancel)
    assert orch._resolve_confirmation("zz-abc123") == (sentinel_confirm, sentinel_cancel)


def test_registering_a_prefix_does_not_capture_other_tokens(orch, restore_handlers):
    orch.register_confirmation_handler("zz-", object(), object())
    assert orch._resolve_confirmation("abc12345")[0] is orch.confirm_pending_action


def test_an_empty_prefix_is_rejected(orch, restore_handlers):
    """An empty prefix matches every token, silently stealing Phase 6's."""
    with pytest.raises(ValueError):
        orch.register_confirmation_handler("", object(), object())


def test_handle_tool_result_routes_to_the_registered_gate(orch, restore_handlers, monkeypatch):
    """The seam only matters if _handle_tool_result actually uses it."""
    confirmed = []
    orch.register_confirmation_handler(
        "zz-",
        lambda token: confirmed.append(token) or {"status": "done"},
        lambda token: {"status": "cancelled"},
    )
    monkeypatch.setattr(orch, "_prompt_for_confirmation", lambda message: True)
    outcome = orch._handle_tool_result(
        {"status": "confirmation_required", "token": "zz-999", "message": "do it?"}
    )
    assert outcome == {"status": "done"}
    assert confirmed == ["zz-999"]


def test_handle_tool_result_routes_a_decline_to_the_registered_cancel(orch, restore_handlers, monkeypatch):
    cancelled = []
    orch.register_confirmation_handler(
        "zz-",
        lambda token: {"status": "done"},
        lambda token: cancelled.append(token) or {"status": "cancelled"},
    )
    monkeypatch.setattr(orch, "_prompt_for_confirmation", lambda message: False)
    orch._handle_tool_result(
        {"status": "confirmation_required", "token": "zz-999", "message": "do it?"}
    )
    assert cancelled == ["zz-999"]


# ---------------------------------------------------------------------------
# _trim_history_preserving_system()
# ---------------------------------------------------------------------------
# Regression tests for a real bug found during Phase 8's accuracy work:
# jarvis.trim_history() is a bare messages[-40:], so once a session
# passed 40 messages it dropped TOOL_SYSTEM_PROMPT off the front along
# with the oldest turn, and every call after that went to DeepSeek with
# no system message at all. The session kept working, just
# progressively worse -- which reads as the model getting dumber, not
# as a bug.

def _long_history(orch, n):
    return [{"role": "system", "content": "SYSTEM"}] + [
        {"role": "user", "content": f"m{i}"} for i in range(n)
    ]


def _limit(orch):
    return orch.MAX_HISTORY_MESSAGES


def test_trim_keeps_the_system_message_past_the_limit(orch):
    trimmed = orch._trim_history_preserving_system(_long_history(orch, _limit(orch) * 2))
    assert trimmed[0]["content"] == "SYSTEM"


def test_a_plain_slice_would_have_dropped_it(orch):
    """Pins the actual bug rather than just the fix: a bare tail slice,
    which is what Phase 1's trim_history does, loses the system message."""
    history = _long_history(orch, _limit(orch) * 2)
    assert history[-_limit(orch):][0]["content"] != "SYSTEM"


def test_trim_still_drops_the_oldest_conversational_messages(orch):
    trimmed = orch._trim_history_preserving_system(_long_history(orch, _limit(orch) * 2))
    contents = [m["content"] for m in trimmed]
    assert "m0" not in contents
    assert f"m{_limit(orch) * 2 - 1}" in contents


def test_trim_keeps_exactly_the_limit_plus_the_system_message(orch):
    trimmed = orch._trim_history_preserving_system(_long_history(orch, _limit(orch) * 2))
    assert len(trimmed) == _limit(orch) + 1


def test_trim_leaves_a_short_history_untouched(orch):
    history = _long_history(orch, 3)
    assert orch._trim_history_preserving_system(history) == history


def test_trim_without_a_system_message_still_trims(orch):
    """run_conversation can in principle be called with a history that
    doesn't start with a system message (Phase 7's own milestone 1 did
    exactly that). Don't pin an arbitrary first message in place."""
    history = [{"role": "user", "content": f"m{i}"} for i in range(_limit(orch) * 2)]
    trimmed = orch._trim_history_preserving_system(history)
    assert len(trimmed) == _limit(orch)
    assert trimmed[0]["content"] != "m0"


def test_trim_handles_empty_history(orch):
    assert orch._trim_history_preserving_system([]) == []


# ---------------------------------------------------------------------------
# Confirmation gate -- _prompt_for_confirmation() / _handle_tool_result()
# ---------------------------------------------------------------------------
# Reused directly from Phase 6's os_agent.py, adapted for voice (a
# spoken heads-up before blocking on the same real terminal prompt).
# Mirrors phaseSix/test_os_agent.py's own suite for the password-gate
# behavior, plus one new test for the voice-specific heads-up.

def test_prompt_for_confirmation_speaks_a_heads_up_before_blocking(orch, monkeypatch):
    monkeypatch.setattr(orch, "CONFIRM_PASSWORD", "secret123")
    monkeypatch.setattr(orch.getpass, "getpass", lambda prompt="": "secret123")
    spoken = []
    monkeypatch.setattr(orch, "speak_safe", lambda text: spoken.append(text))

    orch._prompt_for_confirmation("About to delete x")

    assert len(spoken) == 1
    assert "terminal" in spoken[0].lower()


def test_prompt_for_confirmation_correct_password_approves(orch, monkeypatch):
    monkeypatch.setattr(orch, "CONFIRM_PASSWORD", "secret123")
    monkeypatch.setattr(orch.getpass, "getpass", lambda prompt="": "secret123")
    monkeypatch.setattr(orch, "speak_safe", lambda text: None)

    assert orch._prompt_for_confirmation("m") is True


def test_prompt_for_confirmation_wrong_password_denies(orch, monkeypatch):
    monkeypatch.setattr(orch, "CONFIRM_PASSWORD", "secret123")
    monkeypatch.setattr(orch.getpass, "getpass", lambda prompt="": "wrongpass")
    monkeypatch.setattr(orch, "speak_safe", lambda text: None)

    assert orch._prompt_for_confirmation("m") is False


def test_prompt_for_confirmation_no_password_configured_yes_approves(orch, monkeypatch):
    monkeypatch.setattr(orch, "CONFIRM_PASSWORD", None)
    monkeypatch.setattr("builtins.input", lambda prompt="": "yes")
    monkeypatch.setattr(orch, "speak_safe", lambda text: None)

    assert orch._prompt_for_confirmation("m") is True


def test_prompt_for_confirmation_no_password_configured_anything_else_denies(orch, monkeypatch):
    monkeypatch.setattr(orch, "CONFIRM_PASSWORD", None)
    monkeypatch.setattr("builtins.input", lambda prompt="": "sure")
    monkeypatch.setattr(orch, "speak_safe", lambda text: None)

    assert orch._prompt_for_confirmation("m") is False


def test_prompt_for_confirmation_getpass_failure_fails_closed(orch, monkeypatch):
    """A broken/no-tty terminal must never be treated as approval."""
    monkeypatch.setattr(orch, "CONFIRM_PASSWORD", "secret123")

    def _raise(prompt=""):
        raise Exception("no tty available")

    monkeypatch.setattr(orch.getpass, "getpass", _raise)
    monkeypatch.setattr(orch, "speak_safe", lambda text: None)

    assert orch._prompt_for_confirmation("m") is False


def test_handle_tool_result_passes_through_non_confirmation_results(orch):
    result = {"status": "deleted", "path": "x"}
    assert orch._handle_tool_result(result) is result


def test_handle_tool_result_approved_confirms_and_returns_outcome(orch, monkeypatch):
    monkeypatch.setattr(orch, "_prompt_for_confirmation", lambda message: True)
    calls = []
    monkeypatch.setattr(
        orch, "confirm_pending_action",
        lambda token: calls.append(("confirm", token)) or {"status": "deleted"},
    )
    monkeypatch.setattr(orch, "cancel_pending_action", lambda token: calls.append(("cancel", token)))

    result = orch._handle_tool_result({"status": "confirmation_required", "token": "abc", "message": "delete x"})

    assert result == {"status": "deleted"}
    assert calls == [("confirm", "abc")]


def test_handle_tool_result_denied_cancels(orch, monkeypatch):
    monkeypatch.setattr(orch, "_prompt_for_confirmation", lambda message: False)
    calls = []
    monkeypatch.setattr(orch, "confirm_pending_action", lambda token: calls.append("confirm"))
    monkeypatch.setattr(
        orch, "cancel_pending_action",
        lambda token: calls.append("cancel") or {"status": "cancelled"},
    )

    result = orch._handle_tool_result({"status": "confirmation_required", "token": "abc", "message": "delete x"})

    assert result == {"status": "cancelled"}
    assert calls == ["cancel"]


def test_handle_tool_result_prints_outcome_to_terminal(orch, monkeypatch, capsys):
    monkeypatch.setattr(orch, "_prompt_for_confirmation", lambda message: True)
    monkeypatch.setattr(orch, "confirm_pending_action", lambda token: {"status": "deleted", "path": "C:\\x"})

    orch._handle_tool_result({"status": "confirmation_required", "token": "t", "message": "delete x"})

    captured = capsys.readouterr()
    assert "[NOTE]" in captured.out
    assert "deleted" in captured.out.lower()


def test_handle_tool_result_prints_error_outcome_to_terminal(orch, monkeypatch, capsys):
    monkeypatch.setattr(orch, "_prompt_for_confirmation", lambda message: True)
    monkeypatch.setattr(
        orch, "confirm_pending_action",
        lambda token: {"error": "That confirmation expired after sitting unanswered for 300 seconds."},
    )

    orch._handle_tool_result({"status": "confirmation_required", "token": "t", "message": "delete x"})

    captured = capsys.readouterr()
    assert "[NOTE]" in captured.out
    assert "expired" in captured.out.lower()


# ---------------------------------------------------------------------------
# _summarize_for_log() -- reused as-is from Phase 6's os_agent.py
# ---------------------------------------------------------------------------

def test_summarize_for_log_truncates_long_text_field(orch):
    long_text = "x" * 1000
    summary = orch._summarize_for_log({"text": long_text, "region": "full screen"})

    assert len(summary["text"]) < len(long_text)
    assert "truncated for log" in summary["text"]
    assert summary["region"] == "full screen"


def test_summarize_for_log_leaves_short_text_untouched(orch):
    result = {"text": "short", "content": "also short"}
    assert orch._summarize_for_log(result) == result


def test_summarize_for_log_does_not_mutate_the_original_result(orch):
    long_text = "y" * 1000
    result = {"text": long_text}
    orch._summarize_for_log(result)
    assert result["text"] == long_text


# ---------------------------------------------------------------------------
# _run_tool_calling_turn() -- this file's own merged-tool loop
# ---------------------------------------------------------------------------
# Not re-testing hop-counting/dispatch mechanics already proven by
# phaseFour/test_agent.py and phaseSix/os_agent's own precedent -- just
# the two things genuinely new to the merge: a Phase-6-only tool is
# reachable through this loop, and a confirmation_required result gets
# intercepted before it's fed back as the tool's result.

def _make_tool_call(call_id, name, arguments: dict):
    from unittest.mock import MagicMock
    import json as _json

    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = _json.dumps(arguments)
    tc.model_dump.return_value = {"id": call_id, "type": "function", "function": {"name": name, "arguments": _json.dumps(arguments)}}
    return tc


def _make_response(content=None, tool_calls=None):
    from unittest.mock import MagicMock

    response = MagicMock()
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    response.choices = [MagicMock(message=message)]
    return response


def test_run_tool_calling_turn_reaches_a_phase_six_tool(orch, monkeypatch):
    """get_active_window is real (no mock beneath it) -- proves a
    Phase-6-only tool is genuinely reachable through this loop, not
    just present in TOOL_SCHEMAS."""
    from unittest.mock import MagicMock

    client = MagicMock()
    tool_call = _make_tool_call("call_1", "get_active_window", {})
    first_response = _make_response(content=None, tool_calls=[tool_call])
    second_response = _make_response(content="Got it.", tool_calls=None)

    calls = {"n": 0}

    def fake_call_llm_with_tools(client, messages, schemas):
        calls["n"] += 1
        return first_response if calls["n"] == 1 else second_response

    monkeypatch.setattr(orch, "call_llm_with_tools", fake_call_llm_with_tools)

    messages = [{"role": "user", "content": "what's focused"}]
    result = orch._run_tool_calling_turn(client, messages)

    assert result == "Got it."
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert "title" in tool_messages[0]["content"]


def test_run_tool_calling_turn_intercepts_confirmation_required_result(orch, monkeypatch):
    from unittest.mock import MagicMock

    client = MagicMock()
    tool_call = _make_tool_call("call_1", "delete_file", {"path": "x.txt"})
    first_response = _make_response(content=None, tool_calls=[tool_call])
    second_response = _make_response(content="Deleted it.", tool_calls=None)

    calls = {"n": 0}

    def fake_call_llm_with_tools(client, messages, schemas):
        calls["n"] += 1
        return first_response if calls["n"] == 1 else second_response

    monkeypatch.setattr(orch, "call_llm_with_tools", fake_call_llm_with_tools)
    monkeypatch.setattr(
        orch, "execute_tool",
        lambda name, args: {"status": "confirmation_required", "token": "tok1", "message": "delete x.txt"},
    )

    handled = []

    def fake_handle(result):
        handled.append(result)
        return {"status": "deleted", "path": "x.txt"}

    monkeypatch.setattr(orch, "_handle_tool_result", fake_handle)

    messages = [{"role": "user", "content": "delete x.txt"}]
    result = orch._run_tool_calling_turn(client, messages)

    assert result == "Deleted it."
    assert handled == [{"status": "confirmation_required", "token": "tok1", "message": "delete x.txt"}]
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert "deleted" in tool_messages[0]["content"]


# ---------------------------------------------------------------------------
# _get_memory_store() -- milestone 3's memory-store handoff
# ---------------------------------------------------------------------------
# The no_memory_by_default autouse fixture forces _memory_store_failed
# so every OTHER test gets None without touching real chromadb -- these
# tests reset that state deliberately to exercise the real function.

def test_get_memory_store_builds_once_and_caches(orch, monkeypatch):
    monkeypatch.setattr(orch, "_memory_store", None)
    monkeypatch.setattr(orch, "_memory_store_failed", False)

    build_calls = []
    fake_store = object()
    monkeypatch.setattr(orch, "MemoryStore", lambda: build_calls.append(1) or fake_store)

    store1 = orch._get_memory_store()
    store2 = orch._get_memory_store()

    assert store1 is fake_store
    assert store2 is fake_store
    assert len(build_calls) == 1


def test_get_memory_store_returns_none_on_failure_without_raising(orch, monkeypatch):
    monkeypatch.setattr(orch, "_memory_store", None)
    monkeypatch.setattr(orch, "_memory_store_failed", False)

    def _raise():
        raise orch.MemoryError("chromadb is not installed")

    monkeypatch.setattr(orch, "MemoryStore", _raise)

    assert orch._get_memory_store() is None


def test_get_memory_store_does_not_retry_after_a_failure(orch, monkeypatch):
    monkeypatch.setattr(orch, "_memory_store", None)
    monkeypatch.setattr(orch, "_memory_store_failed", False)

    build_calls = []

    def _raise():
        build_calls.append(1)
        raise orch.MemoryError("store directory could not be opened")

    monkeypatch.setattr(orch, "MemoryStore", _raise)

    assert orch._get_memory_store() is None
    assert orch._get_memory_store() is None
    assert len(build_calls) == 1  # second call short-circuited, didn't retry


# ---------------------------------------------------------------------------
# _handle_remember_command() / "remember" turns
# ---------------------------------------------------------------------------

def test_remember_command_stores_fact_and_confirms(orch, monkeypatch):
    monkeypatch.setattr(orch, "record_until_silence", lambda: "/tmp/fake.wav")
    monkeypatch.setattr(orch, "transcribe_audio", lambda path: "remember my name is Sam")
    monkeypatch.setattr(orch.os, "remove", lambda path: None)

    add_calls = []

    class FakeStore:
        def add_memory(self, text):
            add_calls.append(text)

    monkeypatch.setattr(orch, "_get_memory_store", lambda: FakeStore())

    llm_calls = []
    monkeypatch.setattr(orch, "think", lambda messages: llm_calls.append(messages) or "should not run")
    spoken = []
    monkeypatch.setattr(orch, "speak_safe", lambda text: spoken.append(text))

    history, outcome = orch.run_turn([])

    assert outcome == orch.TurnOutcome.CONTINUE
    assert add_calls == ["my name is Sam"]  # "remember " prefix stripped
    assert spoken == ["Got it, I'll remember that."]
    assert llm_calls == []  # never reaches the LLM
    assert history == []  # remember turns aren't added to conversation history


def test_remember_command_without_a_store_apologizes(orch, monkeypatch):
    monkeypatch.setattr(orch, "record_until_silence", lambda: "/tmp/fake.wav")
    monkeypatch.setattr(orch, "transcribe_audio", lambda path: "remember I like coffee")
    monkeypatch.setattr(orch.os, "remove", lambda path: None)
    monkeypatch.setattr(orch, "_get_memory_store", lambda: None)

    spoken = []
    monkeypatch.setattr(orch, "speak_safe", lambda text: spoken.append(text))

    history, outcome = orch.run_turn([])

    assert outcome == orch.TurnOutcome.CONTINUE
    assert spoken == ["Sorry, I can't save memories right now."]


def test_remember_command_store_failure_apologizes(orch, monkeypatch):
    monkeypatch.setattr(orch, "record_until_silence", lambda: "/tmp/fake.wav")
    monkeypatch.setattr(orch, "transcribe_audio", lambda path: "remember I like coffee")
    monkeypatch.setattr(orch.os, "remove", lambda path: None)

    class FailingStore:
        def add_memory(self, text):
            raise orch.MemoryError("disk full")

    monkeypatch.setattr(orch, "_get_memory_store", lambda: FailingStore())

    spoken = []
    monkeypatch.setattr(orch, "speak_safe", lambda text: spoken.append(text))

    history, outcome = orch.run_turn([])

    assert outcome == orch.TurnOutcome.CONTINUE
    assert spoken == ["Sorry, I couldn't save that."]


# ---------------------------------------------------------------------------
# _augment_with_memories() / retrieval on normal turns
# ---------------------------------------------------------------------------

def test_normal_turn_augments_history_with_retrieved_memories(orch, monkeypatch):
    monkeypatch.setattr(orch, "record_until_silence", lambda: "/tmp/fake.wav")
    monkeypatch.setattr(orch, "transcribe_audio", lambda path: "what's my name")
    monkeypatch.setattr(orch.os, "remove", lambda path: None)
    monkeypatch.setattr(orch, "_trim_history_preserving_system", lambda messages: messages)

    class FakeStore:
        def search_memories(self, query, n_results=None):
            return [{"id": "1", "text": "my name is Sam", "metadata": {}, "distance": 0.1}]

    monkeypatch.setattr(orch, "_get_memory_store", lambda: FakeStore())

    def fake_think(messages):
        messages.append({"role": "assistant", "content": "You're Sam."})
        return "You're Sam."

    monkeypatch.setattr(orch, "think", fake_think)
    monkeypatch.setattr(orch, "speak_safe", lambda text: None)

    history, outcome = orch.run_turn([])

    assert outcome == orch.TurnOutcome.CONTINUE
    user_message = history[-2]
    assert user_message["role"] == "user"
    assert "my name is Sam" in user_message["content"]
    assert "what's my name" in user_message["content"]  # original message still present


def test_normal_turn_without_a_store_sends_message_unaugmented(orch, monkeypatch):
    """_get_memory_store() returning None (no store available) must not
    change a normal turn's behavior at all -- same as before memory
    existed."""
    monkeypatch.setattr(orch, "record_until_silence", lambda: "/tmp/fake.wav")
    monkeypatch.setattr(orch, "transcribe_audio", lambda path: "what time is it")
    monkeypatch.setattr(orch.os, "remove", lambda path: None)
    monkeypatch.setattr(orch, "_trim_history_preserving_system", lambda messages: messages)
    monkeypatch.setattr(orch, "_get_memory_store", lambda: None)

    def fake_think(messages):
        messages.append({"role": "assistant", "content": "It's three o'clock."})
        return "It's three o'clock."

    monkeypatch.setattr(orch, "think", fake_think)
    monkeypatch.setattr(orch, "speak_safe", lambda text: None)

    history, outcome = orch.run_turn([])

    assert outcome == orch.TurnOutcome.CONTINUE
    assert history[-2] == {"role": "user", "content": "what time is it"}


def test_normal_turn_search_failure_falls_back_to_unaugmented_message(orch, monkeypatch):
    monkeypatch.setattr(orch, "record_until_silence", lambda: "/tmp/fake.wav")
    monkeypatch.setattr(orch, "transcribe_audio", lambda path: "what time is it")
    monkeypatch.setattr(orch.os, "remove", lambda path: None)
    monkeypatch.setattr(orch, "_trim_history_preserving_system", lambda messages: messages)

    class FailingStore:
        def search_memories(self, query, n_results=None):
            raise orch.MemoryError("query failed")

    monkeypatch.setattr(orch, "_get_memory_store", lambda: FailingStore())

    def fake_think(messages):
        messages.append({"role": "assistant", "content": "It's three o'clock."})
        return "It's three o'clock."

    monkeypatch.setattr(orch, "think", fake_think)
    monkeypatch.setattr(orch, "speak_safe", lambda text: None)

    history, outcome = orch.run_turn([])

    assert outcome == orch.TurnOutcome.CONTINUE
    assert history[-2] == {"role": "user", "content": "what time is it"}


# ---------------------------------------------------------------------------
# _contains_exit_phrase() -- direct unit tests
# ---------------------------------------------------------------------------
# Covers the exact-fragment match (original behavior) and the newer
# trailing-clause match (found via manual testing: "goodbye" wasn't
# recognized when it trailed another phrase in the same breath, e.g.
# "okay, goodbye" -- see the function's own docstring). Direct tests
# here for speed/precision on the matching logic itself; one
# run_turn()-level integration test below confirms the fix end-to-end,
# matching this file's existing exit-phrase testing style.

@pytest.mark.parametrize(
    "text",
    [
        "goodbye",
        "Goodbye.",
        "quit",
        "exit",
        "stop",
        "shutdown",
        "shut down",
        "good bye",
    ],
)
def test_contains_exit_phrase_exact_matches(orch, text):
    assert orch._contains_exit_phrase(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "okay, goodbye",
        "alright then, goodbye",
        "ok goodbye",
        "well, good bye",
    ],
)
def test_contains_exit_phrase_recognizes_trailing_goodbye(orch, text):
    """The specific bug reported: goodbye/good bye trailing another
    phrase in the same breath, no sentence-ending punctuation between
    them."""
    assert orch._contains_exit_phrase(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "did I quit too early",
        "make it stop",
        "where's the exit",
        "when does the store shut down",
        "why did you quit",
        "please don't stop",
    ],
)
def test_contains_exit_phrase_does_not_match_unrelated_trailing_words(orch, text):
    """The tradeoff this feature is deliberately NOT taking: 'stop',
    'exit', 'quit', and 'shutdown'/'shut down' stay exact-match-only
    (see TRAILING_SAFE_EXIT_PHRASES's comment) specifically because
    they're common as the last word of an ordinary, non-exit request."""
    assert orch._contains_exit_phrase(text) is False


def test_contains_exit_phrase_known_tradeoff_not_goodbye(orch):
    """Documents the accepted false-positive tradeoff from this fix,
    rather than letting it be silently discovered later -- see the
    function's docstring."""
    assert orch._contains_exit_phrase("not goodbye") is True


# ---------------------------------------------------------------------------
# run_turn()
# ---------------------------------------------------------------------------

def test_exit_phrase_recognized_when_trailing_another_phrase(orch, monkeypatch):
    """Integration-level regression test for the reported bug, matching
    this file's existing run_turn()-level exit-phrase test style."""
    monkeypatch.setattr(orch, "record_until_silence", lambda: "/tmp/fake.wav")
    monkeypatch.setattr(orch, "transcribe_audio", lambda path: "Alright, goodbye")
    monkeypatch.setattr(orch.os, "remove", lambda path: None)

    llm_calls = []
    monkeypatch.setattr(orch, "think", lambda messages: llm_calls.append(messages) or "should not run")
    spoken = []
    monkeypatch.setattr(orch, "speak_safe", lambda text: spoken.append(text))

    history, outcome = orch.run_turn([])
    assert outcome == orch.TurnOutcome.EXIT
    assert llm_calls == []
    assert spoken == ["Goodbye."]


def test_exit_phrase_ends_turn_without_calling_llm(orch, monkeypatch):
    monkeypatch.setattr(orch, "record_until_silence", lambda: "/tmp/fake.wav")
    monkeypatch.setattr(orch, "transcribe_audio", lambda path: "quit")
    monkeypatch.setattr(orch.os, "remove", lambda path: None)

    llm_calls = []
    monkeypatch.setattr(orch, "think", lambda messages: llm_calls.append(messages) or "should not run")
    spoken = []
    monkeypatch.setattr(orch, "speak_safe", lambda text: spoken.append(text))

    history, outcome = orch.run_turn([])
    assert outcome == orch.TurnOutcome.EXIT
    assert llm_calls == []
    assert spoken == ["Goodbye."]


def test_exit_phrase_matching_is_exact_not_substring(orch, monkeypatch):
    """'did I quit too early' should NOT trigger exit -- guards the same
    substring-match footgun voice_chat.py's comment already flags."""
    monkeypatch.setattr(orch, "record_until_silence", lambda: "/tmp/fake.wav")
    monkeypatch.setattr(orch, "transcribe_audio", lambda path: "did I quit too early")
    monkeypatch.setattr(orch.os, "remove", lambda path: None)
    monkeypatch.setattr(orch, "_trim_history_preserving_system", lambda messages: messages)

    def fake_think(messages):
        messages.append({"role": "assistant", "content": "No, you're fine."})
        return "No, you're fine."

    monkeypatch.setattr(orch, "think", fake_think)
    monkeypatch.setattr(orch, "speak_safe", lambda text: None)

    history, outcome = orch.run_turn([])
    assert outcome == orch.TurnOutcome.CONTINUE


def test_exit_phrase_matches_even_when_repeated_with_punctuation(orch, monkeypatch):
    """Regression test for the real incident: Whisper transcribed a
    repeated, nervous goodbye as 'Goodbye. Good bye. Good bye.' -- the
    old whole-utterance check missed this entirely."""
    monkeypatch.setattr(orch, "record_until_silence", lambda: "/tmp/fake.wav")
    monkeypatch.setattr(orch, "transcribe_audio", lambda path: "Goodbye. Good bye. Good bye.")
    monkeypatch.setattr(orch.os, "remove", lambda path: None)

    llm_calls = []
    monkeypatch.setattr(orch, "think", lambda messages: llm_calls.append(messages) or "should not run")
    monkeypatch.setattr(orch, "speak_safe", lambda text: None)

    history, outcome = orch.run_turn([])
    assert outcome == orch.TurnOutcome.EXIT
    assert llm_calls == []


def test_exit_phrase_matches_single_clean_word(orch, monkeypatch):
    monkeypatch.setattr(orch, "record_until_silence", lambda: "/tmp/fake.wav")
    monkeypatch.setattr(orch, "transcribe_audio", lambda path: "Exit")
    monkeypatch.setattr(orch.os, "remove", lambda path: None)
    monkeypatch.setattr(orch, "speak_safe", lambda text: None)

    history, outcome = orch.run_turn([])
    assert outcome == orch.TurnOutcome.EXIT


def test_normal_turn_calls_llm_and_updates_history(orch, monkeypatch):
    """think() (Phase 4's agent.run_turn) appends the assistant reply to
    history itself -- the fake here replicates that contract, since
    run_turn() must not append it a second time."""
    monkeypatch.setattr(orch, "record_until_silence", lambda: "/tmp/fake.wav")
    monkeypatch.setattr(orch, "transcribe_audio", lambda path: "what time is it")
    monkeypatch.setattr(orch.os, "remove", lambda path: None)
    monkeypatch.setattr(orch, "_trim_history_preserving_system", lambda messages: messages)

    def fake_think(messages):
        messages.append({"role": "assistant", "content": "It's three o'clock."})
        return "It's three o'clock."

    monkeypatch.setattr(orch, "think", fake_think)

    spoken = []
    monkeypatch.setattr(orch, "speak_safe", lambda text: spoken.append(text))

    history, outcome = orch.run_turn([])
    assert outcome == orch.TurnOutcome.CONTINUE
    assert history[-2] == {"role": "user", "content": "what time is it"}
    assert history[-1] == {"role": "assistant", "content": "It's three o'clock."}
    assert spoken == ["It's three o'clock."]


def test_recording_error_returns_no_speech_without_touching_history(orch, monkeypatch):
    def _raise():
        raise orch.AudioRecordingError("no mic")

    monkeypatch.setattr(orch, "record_until_silence", _raise)

    prior_history = [{"role": "user", "content": "prior"}]
    history, outcome = orch.run_turn(list(prior_history))
    assert outcome == orch.TurnOutcome.NO_SPEECH
    assert history == prior_history


def test_transcription_error_returns_no_speech(orch, monkeypatch):
    monkeypatch.setattr(orch, "record_until_silence", lambda: "/tmp/fake.wav")

    def _raise(path):
        raise orch.TranscriptionError("gibberish")

    monkeypatch.setattr(orch, "transcribe_audio", _raise)
    monkeypatch.setattr(orch.os, "remove", lambda path: None)

    history, outcome = orch.run_turn([])
    assert outcome == orch.TurnOutcome.NO_SPEECH
    assert history == []


def test_temp_wav_is_removed_even_if_transcription_fails(orch, monkeypatch):
    monkeypatch.setattr(orch, "record_until_silence", lambda: "/tmp/fake.wav")

    def _raise(path):
        raise orch.TranscriptionError("bad audio")

    monkeypatch.setattr(orch, "transcribe_audio", _raise)

    removed = []
    monkeypatch.setattr(orch.os, "remove", lambda path: removed.append(path))

    orch.run_turn([])
    assert removed == ["/tmp/fake.wav"]


def test_missing_temp_wav_on_cleanup_does_not_raise(orch, monkeypatch):
    """os.remove can legitimately fail (file already gone) -- run_turn
    should swallow that, same as voice_chat.py does."""
    monkeypatch.setattr(orch, "record_until_silence", lambda: "/tmp/fake.wav")
    monkeypatch.setattr(orch, "transcribe_audio", lambda path: "hello")
    monkeypatch.setattr(orch, "_trim_history_preserving_system", lambda messages: messages)

    def fake_think(messages):
        messages.append({"role": "assistant", "content": "hi there"})
        return "hi there"

    monkeypatch.setattr(orch, "think", fake_think)
    monkeypatch.setattr(orch, "speak_safe", lambda text: None)

    def _raise_oserror(path):
        raise OSError("already deleted")

    monkeypatch.setattr(orch.os, "remove", _raise_oserror)

    history, outcome = orch.run_turn([])
    assert outcome == orch.TurnOutcome.CONTINUE  # did not propagate the OSError


def test_llm_failure_speaks_apology_and_continues(orch, monkeypatch):
    monkeypatch.setattr(orch, "record_until_silence", lambda: "/tmp/fake.wav")
    monkeypatch.setattr(orch, "transcribe_audio", lambda path: "hello")
    monkeypatch.setattr(orch.os, "remove", lambda path: None)
    monkeypatch.setattr(orch, "_trim_history_preserving_system", lambda messages: messages)

    def _raise(messages):
        raise RuntimeError("LLM down")

    monkeypatch.setattr(orch, "think", _raise)

    spoken = []
    monkeypatch.setattr(orch, "speak_safe", lambda text: spoken.append(text))

    history, outcome = orch.run_turn([])
    assert outcome == orch.TurnOutcome.CONTINUE
    assert "trouble reaching my brain" in spoken[0]
    # The user's turn is still recorded even though the LLM call
    # failed -- same as voice_chat.py's behavior -- so what they said
    # isn't silently dropped from context on the next attempt.
    assert history == [{"role": "user", "content": "hello"}]


def test_pre_speech_timeout_is_passed_through_when_given(orch, monkeypatch):
    received = {}

    def fake_record(**kwargs):
        received.update(kwargs)
        return "/tmp/fake.wav"

    monkeypatch.setattr(orch, "record_until_silence", fake_record)
    monkeypatch.setattr(orch, "transcribe_audio", lambda path: "hello")
    monkeypatch.setattr(orch.os, "remove", lambda path: None)
    monkeypatch.setattr(orch, "_trim_history_preserving_system", lambda m: m)
    monkeypatch.setattr(orch, "think", lambda m: "hi")
    monkeypatch.setattr(orch, "speak_safe", lambda t: None)

    orch.run_turn([], pre_speech_timeout=5.0)
    assert received == {"pre_speech_timeout": 5.0}


def test_pre_speech_timeout_none_uses_plain_call(orch, monkeypatch):
    calls = []

    def fake_record(*args, **kwargs):
        calls.append((args, kwargs))
        return "/tmp/fake.wav"

    monkeypatch.setattr(orch, "record_until_silence", fake_record)
    monkeypatch.setattr(orch, "transcribe_audio", lambda path: "hello")
    monkeypatch.setattr(orch.os, "remove", lambda path: None)
    monkeypatch.setattr(orch, "_trim_history_preserving_system", lambda m: m)
    monkeypatch.setattr(orch, "think", lambda m: "hi")
    monkeypatch.setattr(orch, "speak_safe", lambda t: None)

    orch.run_turn([])
    assert calls == [((), {})]


def test_on_state_change_is_called_in_expected_order(orch, monkeypatch):
    monkeypatch.setattr(orch, "record_until_silence", lambda: "/tmp/fake.wav")
    monkeypatch.setattr(orch, "transcribe_audio", lambda path: "hello")
    monkeypatch.setattr(orch.os, "remove", lambda path: None)
    monkeypatch.setattr(orch, "_trim_history_preserving_system", lambda messages: messages)
    monkeypatch.setattr(orch, "think", lambda messages: "hi")
    monkeypatch.setattr(orch, "speak_safe", lambda text: None)

    states = []
    orch.run_turn([], on_state_change=states.append)

    assert states == [
        orch.OrchestratorState.LISTENING,
        orch.OrchestratorState.TRANSCRIBING,
        orch.OrchestratorState.THINKING,
        orch.OrchestratorState.SPEAKING,
    ]


def test_on_state_change_stops_at_speaking_for_exit_phrase(orch, monkeypatch):
    monkeypatch.setattr(orch, "record_until_silence", lambda: "/tmp/fake.wav")
    monkeypatch.setattr(orch, "transcribe_audio", lambda path: "goodbye")
    monkeypatch.setattr(orch.os, "remove", lambda path: None)
    monkeypatch.setattr(orch, "speak_safe", lambda text: None)

    states = []
    orch.run_turn([], on_state_change=states.append)

    assert states == [
        orch.OrchestratorState.LISTENING,
        orch.OrchestratorState.TRANSCRIBING,
        orch.OrchestratorState.SPEAKING,
    ]


# ---------------------------------------------------------------------------
# _play_wake_chime()
# ---------------------------------------------------------------------------

def test_wake_chime_disabled_skips_entirely(orch, monkeypatch):
    monkeypatch.setattr(orch, "WAKE_CHIME_ENABLED", False)

    def _fail_if_called():
        raise AssertionError("platform.system() should not be checked when the chime is disabled")

    monkeypatch.setattr(orch.platform, "system", _fail_if_called)
    orch._play_wake_chime()  # should return before ever checking the platform


def test_wake_chime_does_not_raise_on_non_windows(orch, monkeypatch):
    monkeypatch.setattr(orch, "WAKE_CHIME_ENABLED", True)
    monkeypatch.setattr(orch.platform, "system", lambda: "Linux")
    orch._play_wake_chime()  # clean no-op, no winsound import attempted


def test_wake_chime_failure_is_non_fatal(orch, monkeypatch):
    """Even on a platform reporting 'Windows', if the beep itself fails
    (missing winsound, driver issue, whatever), the turn should never
    be interrupted over a missing chime."""
    monkeypatch.setattr(orch, "WAKE_CHIME_ENABLED", True)
    monkeypatch.setattr(orch.platform, "system", lambda: "Windows")
    orch._play_wake_chime()  # winsound isn't available in this test env -- must not raise


# ---------------------------------------------------------------------------
# run_conversation() -- the follow-up window chaining logic
# ---------------------------------------------------------------------------
# Tested purely by faking run_turn() itself -- run_conversation doesn't
# know or care *why* a turn came back a given way, so there's no need
# to re-fake record_until_silence/transcribe_audio/think here.

def _fake_run_turn_sequence(outcomes, calls_out):
    """Returns a fake run_turn(history, on_state_change=None,
    pre_speech_timeout=None) that pops one outcome per call and
    records the pre_speech_timeout it was given, so tests can assert
    both the sequence of outcomes consumed and the window sizes used."""
    remaining = list(outcomes)

    def _fake(history, on_state_change=None, pre_speech_timeout=None):
        calls_out.append(pre_speech_timeout)
        return history, remaining.pop(0)

    return _fake


def test_run_conversation_falls_back_to_idle_when_first_turn_has_no_speech(orch, monkeypatch):
    calls = []
    monkeypatch.setattr(
        orch, "run_turn", _fake_run_turn_sequence([orch.TurnOutcome.NO_SPEECH], calls)
    )

    history, exited = orch.run_conversation([])
    assert exited is False
    assert calls == [None]  # only the first-turn attempt, using the full window


def test_run_conversation_uses_follow_up_window_after_a_successful_turn(orch, monkeypatch):
    calls = []
    monkeypatch.setattr(
        orch,
        "run_turn",
        _fake_run_turn_sequence([orch.TurnOutcome.CONTINUE, orch.TurnOutcome.NO_SPEECH], calls),
    )

    history, exited = orch.run_conversation([])
    assert exited is False
    assert calls == [None, orch.FOLLOWUP_LISTEN_SECONDS]


def test_run_conversation_stops_immediately_on_exit(orch, monkeypatch):
    calls = []
    monkeypatch.setattr(
        orch, "run_turn", _fake_run_turn_sequence([orch.TurnOutcome.EXIT], calls)
    )

    history, exited = orch.run_conversation([])
    assert exited is True
    assert calls == [None]


def test_run_conversation_exit_during_a_follow_up_still_reports_exited(orch, monkeypatch):
    calls = []
    monkeypatch.setattr(
        orch,
        "run_turn",
        _fake_run_turn_sequence([orch.TurnOutcome.CONTINUE, orch.TurnOutcome.EXIT], calls),
    )

    history, exited = orch.run_conversation([])
    assert exited is True
    assert calls == [None, orch.FOLLOWUP_LISTEN_SECONDS]


def test_run_conversation_can_chain_several_follow_up_turns(orch, monkeypatch):
    calls = []
    monkeypatch.setattr(
        orch,
        "run_turn",
        _fake_run_turn_sequence(
            [
                orch.TurnOutcome.CONTINUE,
                orch.TurnOutcome.CONTINUE,
                orch.TurnOutcome.CONTINUE,
                orch.TurnOutcome.NO_SPEECH,
            ],
            calls,
        ),
    )

    history, exited = orch.run_conversation([])
    assert exited is False
    assert calls == [None, orch.FOLLOWUP_LISTEN_SECONDS, orch.FOLLOWUP_LISTEN_SECONDS, orch.FOLLOWUP_LISTEN_SECONDS]


def test_run_conversation_passes_history_through_across_turns(orch, monkeypatch):
    def _fake(history, on_state_change=None, pre_speech_timeout=None):
        history = history + [{"role": "user", "content": "turn"}]
        outcome = orch.TurnOutcome.CONTINUE if len(history) < 3 else orch.TurnOutcome.NO_SPEECH
        return history, outcome

    monkeypatch.setattr(orch, "run_turn", _fake)

    history, exited = orch.run_conversation([])
    assert exited is False
    assert len(history) == 3


def test_run_conversation_disabled_stops_after_one_turn_even_on_continue(orch, monkeypatch):
    calls = []
    monkeypatch.setattr(orch, "FOLLOWUP_ENABLED", False)
    monkeypatch.setattr(
        orch,
        "run_turn",
        _fake_run_turn_sequence([orch.TurnOutcome.CONTINUE], calls),
    )

    history, exited = orch.run_conversation([])
    assert exited is False
    assert calls == [None]  # never attempted a second, follow-up-window turn


def test_run_conversation_on_state_change_is_forwarded_to_run_turn(orch, monkeypatch):
    received = []

    def _fake(history, on_state_change=None, pre_speech_timeout=None):
        if on_state_change:
            on_state_change("marker")
        return history, orch.TurnOutcome.NO_SPEECH

    monkeypatch.setattr(orch, "run_turn", _fake)

    orch.run_conversation([], on_state_change=received.append)
    assert received == ["marker"]

# ---------------------------------------------------------------------------
# _spoken_reason_for() -- saying WHY confirmation is needed
# ---------------------------------------------------------------------------
# The gate used to say only "That needs your confirmation", which told
# the person nothing about what they were being asked to approve --
# a poor way to ask someone to make a security decision. It now names
# the kind of action and its consequence, and deliberately still never
# says the target: no paths, no coordinates, no keystrokes. That
# boundary is Phase 7's own (see the README's "Confirmation design"),
# and the full detail prints to the terminal where the human is
# confirming anyway.

@pytest.mark.parametrize("message,expected_words", [
    ("About to permanently delete 'C:/x/notes.txt'.", ["delete", "cannot be undone"]),
    ("About to overwrite the existing contents of 'C:/x/a.yaml'.", ["replace", "recovered"]),
    ("About to close the window 'Spotify'.", ["close a window", "unsaved"]),
    ("About to terminate 'chrome.exe' (pid 4821).", ["force a program to close"]),
    ("About to SHUTDOWN this machine.", ["shut the machine down"]),
    ("About to SLEEP this machine.", ["sleep"]),
    ("About to click at (840, 512) in window 'Notepad'.", ["mouse click"]),
    ("About to press hotkey alt+f4.", ["keyboard shortcut"]),
])
def test_the_spoken_reason_names_the_consequence(orch, message, expected_words):
    spoken = orch._spoken_reason_for(message).lower()
    for word in expected_words:
        assert word.lower() in spoken


@pytest.mark.parametrize("message", [
    "About to permanently delete 'C:/Users/Example/Documents/notes.txt'. Cannot be undone.",
    "About to click at (840, 512) in window 'Notepad'.",
    "About to terminate 'chrome.exe' (pid 4821).",
    "About to press hotkey alt+f4 in window 'Notepad'.",
])
def test_the_spoken_reason_never_reads_the_target_aloud(orch, message):
    """The load-bearing half. Reading an absolute path or click
    coordinates out loud is neither a good security property nor a good
    listening experience -- it goes to the terminal, not the room."""
    spoken = orch._spoken_reason_for(message)
    for leak in ("/", "\\", "(", "pid", ".txt", "alt+f4", "Notepad", "840"):
        assert leak not in spoken


def test_an_unrecognised_message_still_warns(orch):
    """A tool whose wording changes must degrade to a generic warning,
    never to silence and never to reading the message out."""
    spoken = orch._spoken_reason_for("Some future tool with wording nobody predicted.")
    assert "destructive" in spoken.lower()
    assert "confirmation at the terminal" in spoken.lower()
    assert "nobody predicted" not in spoken


def test_every_reason_directs_the_person_to_the_terminal(orch):
    """Confirming is keyboard-only by design, so every spoken warning
    has to say where to go."""
    for keyword, _ in orch._CONFIRMATION_REASONS:
        spoken = orch._spoken_reason_for(f"About to {keyword} something.")
        assert "terminal" in spoken.lower()


def test_prompt_for_confirmation_speaks_the_reason(orch, monkeypatch):
    said = []
    monkeypatch.setattr(orch, "speak_safe", lambda text: said.append(text))
    monkeypatch.setattr(orch, "CONFIRM_PASSWORD", None)
    monkeypatch.setattr("builtins.input", lambda prompt="": "no")

    orch._prompt_for_confirmation("About to permanently delete 'C:/x/notes.txt'.")
    assert said and "delete" in said[0].lower()
    assert "notes.txt" not in said[0]


# ---------------------------------------------------------------------------
# register_reply_filter() -- log and audio must say the same thing
# ---------------------------------------------------------------------------

@pytest.fixture
def restore_reply_filter(orch):
    saved = orch._reply_filter
    yield
    orch._reply_filter = saved


def test_no_filter_leaves_the_reply_alone(orch, restore_reply_filter):
    orch._reply_filter = None
    assert orch._filter_reply("hello there") == "hello there"


def test_a_registered_filter_is_applied(orch, restore_reply_filter):
    orch.register_reply_filter(lambda text: text.upper())
    assert orch._filter_reply("hello") == "HELLO"


def test_a_failing_filter_falls_back_to_the_original(orch, restore_reply_filter):
    """A cosmetic text tidy must never be able to lose a reply the model
    already spent a round-trip producing."""
    def boom(text):
        raise ValueError("bad regex")

    orch.register_reply_filter(boom)
    assert orch._filter_reply("the original reply") == "the original reply"


def test_run_turn_logs_exactly_what_it_speaks(orch, monkeypatch, restore_reply_filter):
    """The bug this seam fixes: Phase 8 rewrote text on its way to the
    speech engine, so the terminal printed one thing and the speakers
    said another."""
    spoken = []
    logged = []

    orch.register_reply_filter(lambda text: text.replace("**", ""))
    monkeypatch.setattr(orch, "record_until_silence", lambda **kw: "fake.wav")
    monkeypatch.setattr(orch, "transcribe_audio", lambda path: "how is my pc")
    monkeypatch.setattr(orch, "think", lambda history: "The **memory** is high.")
    monkeypatch.setattr(orch, "_augment_with_memories", lambda text: text)
    monkeypatch.setattr(orch, "is_remember_command", lambda text: False)
    monkeypatch.setattr(orch, "speak_safe", lambda text: spoken.append(text))
    monkeypatch.setattr(orch.os, "remove", lambda path: None)
    monkeypatch.setattr(orch.logger, "info", lambda msg, *a: logged.append(str(msg)))

    orch.run_turn([{"role": "system", "content": "x"}])

    jarvis_lines = [line for line in logged if line.startswith("Jarvis:")]
    assert jarvis_lines, "the reply was never logged"
    assert jarvis_lines[0][len("Jarvis: "):] == spoken[0]
    assert "**" not in spoken[0]


# ---------------------------------------------------------------------------
# Tool-call-aware trimming, and the raised ceilings
# ---------------------------------------------------------------------------
# Known and deliberately unfixed since milestone 2, on the grounds that
# it was "unlikely in practice at MAX_HISTORY_MESSAGES=40". Raising the
# tool-hop ceiling is what made it likely: one complex turn can now
# append more messages than the old history budget held.

def _tool_exchange(index):
    """One hop, as DeepSeek's API shapes it: an assistant message
    requesting a tool, then the tool-role message answering it."""
    return [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": f"call_{index}", "function": {"name": "x", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": f"call_{index}", "content": "result"},
    ]


def test_a_trim_never_leaves_an_orphaned_tool_message(orch):
    """DeepSeek rejects a tool-role message that does not follow the
    assistant message whose tool_calls it answers. A tail slice can land
    exactly there, and the whole turn fails."""
    history = [{"role": "system", "content": "SYSTEM"}]
    for i in range(orch.MAX_HISTORY_MESSAGES):
        history.extend(_tool_exchange(i))

    trimmed = orch._trim_history_preserving_system(history)
    conversation = trimmed[1:]
    assert conversation, "everything was trimmed away"
    assert conversation[0]["role"] != "tool", "left a tool result with nothing to answer"


def test_every_tool_message_still_follows_its_assistant(orch):
    """The property the API actually enforces, checked over the whole
    trimmed window rather than just its first message."""
    history = [{"role": "system", "content": "SYSTEM"}]
    for i in range(orch.MAX_HISTORY_MESSAGES):
        history.extend(_tool_exchange(i))

    trimmed = orch._trim_history_preserving_system(history)
    for previous, current in zip(trimmed, trimmed[1:]):
        if current.get("role") == "tool":
            assert previous.get("role") == "assistant" and previous.get("tool_calls"), \
                "a tool result was left stranded mid-window"


def test_dropping_orphans_only_removes_leading_tool_messages(orch):
    """A cut can only ever orphan results, never requests -- so nothing
    past the first non-tool message should be touched."""
    messages = [
        {"role": "tool", "content": "orphan"},
        {"role": "tool", "content": "orphan"},
        {"role": "user", "content": "keep me"},
        {"role": "tool", "content": "keep me too"},
    ]
    kept = orch._drop_orphaned_tool_messages(messages)
    assert kept == messages[2:]


def test_dropping_orphans_leaves_a_clean_window_alone(orch):
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert orch._drop_orphaned_tool_messages(messages) == messages


def test_the_hop_ceiling_is_high_enough_for_multi_step_work(orch):
    """Raised from 8 during Phase 8's polish pass -- genuinely
    multi-step requests were dying at the old cap."""
    assert orch.MAX_TOOL_HOPS >= 40


def test_history_outlasts_a_single_maximum_turn(orch):
    """The interaction that made the trim bug urgent: each hop appends
    at least two messages, so the history budget has to exceed what one
    full-length turn can produce, or a complex turn erases the
    conversation that prompted it."""
    assert orch.MAX_HISTORY_MESSAGES > orch.MAX_TOOL_HOPS * 2


def test_a_turn_stops_when_it_runs_out_of_time(orch, monkeypatch):
    """The time budget is what actually bounds a confused loop -- a
    bigger hop ceiling alone would just make a runaway take longer.

    The clock is driven rather than slept through, so this stays fast
    and deterministic.
    """
    monkeypatch.setattr(orch, "TURN_TIME_BUDGET_SECONDS", 10.0)
    ticks = iter([0.0, 999.0])  # start, then well past the budget
    monkeypatch.setattr(orch.time, "monotonic", lambda: next(ticks))

    calls = []
    monkeypatch.setattr(orch, "call_llm_with_tools", lambda *a, **kw: calls.append(1))

    reply = orch._run_tool_calling_turn(object(), [])
    assert "ran out of time" in reply.lower()
    assert calls == [], "should stop before making another API call"


def test_the_time_budget_does_not_fire_on_a_normal_turn(orch, monkeypatch):
    monkeypatch.setattr(orch, "TURN_TIME_BUDGET_SECONDS", 300.0)

    class _Message:
        content = "done"
        tool_calls = None

    class _Response:
        choices = [type("C", (), {"message": _Message()})()]

    monkeypatch.setattr(orch, "call_llm_with_tools", lambda *a, **kw: _Response())
    assert orch._run_tool_calling_turn(object(), []) == "done"


def test_giving_up_explains_itself_in_plain_language(orch, monkeypatch):
    """The old text was "(gave up after too many tool calls in a row --
    check MAX_TOOL_HOPS)", which is a log line, not something to say to
    a person out loud."""
    monkeypatch.setattr(orch, "MAX_TOOL_HOPS", 1)

    class _Message:
        content = None
        tool_calls = [type("TC", (), {
            "id": "call_1",
            "function": type("F", (), {"name": "get_current_time", "arguments": "{}"})(),
            "model_dump": lambda self: {"id": "call_1"},
        })()]

    class _Response:
        choices = [type("C", (), {"message": _Message()})()]

    monkeypatch.setattr(orch, "call_llm_with_tools", lambda *a, **kw: _Response())
    monkeypatch.setattr(orch, "execute_tool", lambda name, args: {"ok": True})

    reply = orch._run_tool_calling_turn(object(), [])
    assert "MAX_TOOL_HOPS" not in reply
    assert "smaller" in reply.lower()


# ---------------------------------------------------------------------------
# register_turn_context() -- keeping the tool schemas cacheable
# ---------------------------------------------------------------------------

@pytest.fixture
def restore_turn_context(orch):
    saved = orch._turn_context_provider
    yield
    orch._turn_context_provider = saved


def test_no_provider_means_no_context(orch, restore_turn_context):
    orch._turn_context_provider = None
    assert orch._turn_context() == ""


def test_a_registered_provider_supplies_context(orch, restore_turn_context):
    orch.register_turn_context(lambda: "[Context: it is Tuesday]")
    assert orch._turn_context() == "[Context: it is Tuesday]"


def test_a_failing_provider_does_not_cost_the_turn(orch, restore_turn_context):
    def boom():
        raise RuntimeError("clock exploded")

    orch.register_turn_context(boom)
    assert orch._turn_context() == ""


def test_the_context_is_prepended_to_the_user_message(orch, monkeypatch, restore_turn_context):
    """It has to land in the user turn, AFTER the tool schemas in the
    serialized request -- that is the whole point. In the system prompt
    it invalidated the cached prefix and re-charged all 35 tool schemas
    every conversation."""
    orch.register_turn_context(lambda: "[Context: it is Tuesday]")
    monkeypatch.setattr(orch, "record_until_silence", lambda **kw: "fake.wav")
    monkeypatch.setattr(orch, "transcribe_audio", lambda path: "what time is it")
    monkeypatch.setattr(orch, "_augment_with_memories", lambda text: text)
    monkeypatch.setattr(orch, "is_remember_command", lambda text: False)
    monkeypatch.setattr(orch, "think", lambda history: "It is Tuesday sir.")
    monkeypatch.setattr(orch, "speak_safe", lambda text: None)
    monkeypatch.setattr(orch.os, "remove", lambda path: None)

    history, _ = orch.run_turn([{"role": "system", "content": "x"}])
    user_messages = [m for m in history if m["role"] == "user"]
    assert user_messages[0]["content"].startswith("[Context: it is Tuesday]")
    assert "what time is it" in user_messages[0]["content"]


def test_memory_augmentation_still_reaches_the_model(orch, monkeypatch, restore_turn_context):
    """The context line is prepended around memory augmentation, not
    instead of it."""
    orch.register_turn_context(lambda: "[Context: it is Tuesday]")
    monkeypatch.setattr(orch, "record_until_silence", lambda **kw: "fake.wav")
    monkeypatch.setattr(orch, "transcribe_audio", lambda path: "what did I say")
    monkeypatch.setattr(orch, "_augment_with_memories", lambda text: f"REMEMBERED\n{text}")
    monkeypatch.setattr(orch, "is_remember_command", lambda text: False)
    monkeypatch.setattr(orch, "think", lambda history: "ok sir.")
    monkeypatch.setattr(orch, "speak_safe", lambda text: None)
    monkeypatch.setattr(orch.os, "remove", lambda path: None)

    history, _ = orch.run_turn([{"role": "system", "content": "x"}])
    content = [m for m in history if m["role"] == "user"][0]["content"]
    assert "REMEMBERED" in content and "what did I say" in content


# ---------------------------------------------------------------------------
# Per-turn crash recovery (reliability pass)
#
# run_turn() already guarded its own three recoverable steps. What was
# unguarded was the loop AROUND it -- the wake gate, the chime, the
# system-prompt provider -- where a single unhandled exception killed
# the whole process. These pin the recovery, and equally importantly
# pin the two things recovery must NOT break: Ctrl-C must still stop
# Jarvis immediately, and a permanent fault must not spin forever.
# ---------------------------------------------------------------------------

class _RecordingGate:
    """Stand-in for WakeWordGate that can be told to fail on demand."""

    def __init__(self, fail_on=()):
        self.fail_on = set(fail_on)
        self.starts = 0
        self.stops = 0

    def start(self):
        self.starts += 1
        if self.starts in self.fail_on:
            raise RuntimeError(f"microphone exploded on start {self.starts}")

    def wait_for_wake(self):
        return True

    def stop(self):
        self.stops += 1


def _orchestrator_harness(orch, monkeypatch, gate, conversation_results):
    """Wires run_orchestrator up to a fake gate and a scripted
    run_conversation, with the model build and chime stubbed out."""
    monkeypatch.setattr(orch, "build_model", lambda *a, **k: object())
    monkeypatch.setattr(orch, "WakeWordGate", lambda *a, **k: gate)
    monkeypatch.setattr(orch, "_play_wake_chime", lambda: None)
    monkeypatch.setattr(orch.time, "sleep", lambda s: None)

    results = list(conversation_results)

    def fake_run_conversation(history, on_state_change=None):
        if not results:
            raise AssertionError("run_conversation called more times than scripted")
        outcome = results.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return history, outcome

    monkeypatch.setattr(orch, "run_conversation", fake_run_conversation)
    return gate


def test_a_failed_turn_does_not_kill_the_orchestrator(orch, monkeypatch):
    gate = _RecordingGate()
    _orchestrator_harness(
        orch, monkeypatch, gate,
        [RuntimeError("audio driver hiccup"), True],  # fail, then a clean exit
    )

    orch.run_orchestrator()  # must return normally, not raise

    assert gate.starts == 2, "should have gone back to listening after the failure"


def test_a_failure_in_the_wake_gate_itself_is_recovered(orch, monkeypatch):
    # The gate is where the "no usable input device" error actually
    # surfaces, and it sits outside run_turn's own guards entirely.
    gate = _RecordingGate(fail_on={1})
    _orchestrator_harness(orch, monkeypatch, gate, [True])

    orch.run_orchestrator()

    assert gate.starts == 2


def test_a_recovered_turn_releases_the_microphone_before_retrying(orch, monkeypatch):
    gate = _RecordingGate()
    _orchestrator_harness(orch, monkeypatch, gate, [RuntimeError("boom"), True])

    orch.run_orchestrator()

    # A half-open stream is the likeliest reason the next start() would
    # fail too, turning one fault into a run of them.
    assert gate.stops >= 2


def test_a_permanent_fault_gives_up_instead_of_looping(orch, monkeypatch):
    gate = _RecordingGate()
    _orchestrator_harness(
        orch, monkeypatch, gate,
        [RuntimeError("no microphone")] * orch.MAX_CONSECUTIVE_TURN_FAILURES,
    )

    orch.run_orchestrator()

    assert gate.starts == orch.MAX_CONSECUTIVE_TURN_FAILURES


def test_the_failure_count_resets_after_a_good_turn(orch, monkeypatch):
    # A transient fault every other turn must never accumulate into a
    # shutdown -- otherwise a flaky mic ends the session eventually
    # even though Jarvis is working fine in between.
    gate = _RecordingGate()
    script = []
    for _ in range(orch.MAX_CONSECUTIVE_TURN_FAILURES + 2):
        script.extend([RuntimeError("blip"), False])
    script.append(True)
    _orchestrator_harness(orch, monkeypatch, gate, script)

    orch.run_orchestrator()

    assert gate.starts == len(script), "should have survived every alternating failure"


def test_ctrl_c_still_stops_jarvis_immediately(orch, monkeypatch):
    # KeyboardInterrupt inherits from BaseException, not Exception, so
    # the recovery handler must not absorb it and retry.
    gate = _RecordingGate()
    _orchestrator_harness(orch, monkeypatch, gate, [KeyboardInterrupt(), True])

    orch.run_orchestrator()

    assert gate.starts == 1, "Ctrl-C must not be treated as a failed turn"


def test_the_microphone_is_released_on_the_way_out(orch, monkeypatch):
    gate = _RecordingGate()
    _orchestrator_harness(orch, monkeypatch, gate, [True])

    orch.run_orchestrator()

    assert gate.stops >= 1


# ---------------------------------------------------------------------------
# LLM timeout (reliability pass)
# ---------------------------------------------------------------------------

def test_the_llm_client_is_given_a_real_timeout(orch, monkeypatch):
    captured = {}

    class _FakeClient:
        def with_options(self, **kwargs):
            captured.update(kwargs)
            return self

    monkeypatch.setattr(orch, "build_client", lambda: _FakeClient())
    monkeypatch.setattr(orch, "_deepseek_client", None)

    orch._get_deepseek_client()

    assert captured["timeout"] == orch.LLM_TIMEOUT_SECONDS
    assert captured["max_retries"] == orch.LLM_MAX_RETRIES


def test_the_timeout_is_far_below_the_sdk_default(orch):
    # The SDK would otherwise give this read=600s with 2 retries --
    # ~30 minutes of a silent, unresponsive Jarvis on one hung call.
    worst_case = orch.LLM_TIMEOUT_SECONDS * (orch.LLM_MAX_RETRIES + 1)
    assert worst_case <= 180, f"worst-case LLM stall is {worst_case}s"


def test_the_timeout_still_allows_a_real_tool_heavy_hop(orch):
    # A hop carrying 39 tool schemas and a long history legitimately
    # takes 20-30s; a tighter timeout would cancel work about to
    # succeed.
    assert orch.LLM_TIMEOUT_SECONDS >= 45


def test_a_client_that_cannot_take_options_still_works(orch, monkeypatch):
    class _OldClient:
        def with_options(self, **kwargs):
            raise TypeError("unexpected keyword")

    monkeypatch.setattr(orch, "build_client", lambda: _OldClient())
    monkeypatch.setattr(orch, "_deepseek_client", None)

    client = orch._get_deepseek_client()

    assert isinstance(client, _OldClient), "must degrade, not crash"
