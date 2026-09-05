"""
test_app.py -- Phase 8 milestone 1 tests for the wiring in app.py:
the date-aware system prompt, and the registration that layers Phase
8's tools over Phase 4's.

Requires phaseSeven/orchestrator.py to be importable (and therefore
Phases 1-6's modules at the sibling paths it expects), same as Phase
7's own suite. openwakeword is stubbed via sys.modules for the same
reason it is there -- the real package isn't needed just to run tests.

The registration tests mutate orchestrator's module-level registry,
which is process-global, so every one of them restores it afterward.
Skipping that would leak an overridden get_weather into any test that
ran later.
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
def app_module(stub_openwakeword):
    import app
    return app


@pytest.fixture
def clean_registry(app_module):
    """Snapshots orchestrator's tool registry and puts it back after the
    test, since register_tools mutates module-level state."""
    o = app_module.o
    saved_schemas = list(o.TOOL_SCHEMAS)
    saved_functions = dict(o._TOOL_FUNCTIONS)
    yield o
    o.TOOL_SCHEMAS = saved_schemas
    o._TOOL_FUNCTIONS.clear()
    o._TOOL_FUNCTIONS.update(saved_functions)


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------

def test_system_prompt_keeps_all_of_phase_sevens(app_module):
    """Phase 8 appends; it does not replace. Everything Phase 7's
    prompt says about the OS tools and the confirmation gate is still
    correct and load-bearing."""
    assert app_module.o.TOOL_SYSTEM_PROMPT in app_module.build_system_prompt()


def test_system_prompt_adds_phase_eight_guidance(app_module):
    assert app_module.PHASE_EIGHT_PROMPT_ADDITIONS in app_module.build_system_prompt()


def test_the_turn_context_states_the_current_date_and_time(app_module, monkeypatch):
    monkeypatch.setattr(
        app_module.tools_extra,
        "get_current_time",
        lambda: {"spoken": "Tuesday, September 1, 2026 at 3:20 AM", "timezone": "AUS Eastern Standard Time"},
    )
    context = app_module.build_turn_context()
    assert "Tuesday, September 1, 2026 at 3:20 AM" in context
    assert "AUS Eastern Standard Time" in context


def test_the_clock_is_not_in_the_system_prompt(app_module):
    """It moved to the turn context deliberately. A system prompt that
    changes every minute invalidates the cached prefix and re-charges
    all 35 tool schemas on every conversation -- measured at 4,263
    full-price tokens per conversation versus about 28 after the move."""
    assert "current date and time" not in app_module.build_system_prompt()


def test_the_turn_context_claims_authority(app_module):
    """A line in the user turn carries less weight than one in the
    system prompt, and the whole point of stating the time is that the
    model should trust it over its training data."""
    assert "authoritative" in app_module.build_turn_context().lower()


def test_the_turn_context_is_rebuilt_from_the_clock_each_call(app_module, monkeypatch):
    """The whole reason this is a provider function rather than a
    constant: run for two days and a date captured at startup is wrong,
    and the model has no reason to distrust it."""
    times = iter([
        {"spoken": "Monday, August 31, 2026 at 11:59 PM", "timezone": "TZ"},
        {"spoken": "Tuesday, September 1, 2026 at 12:01 AM", "timezone": "TZ"},
    ])
    monkeypatch.setattr(app_module.tools_extra, "get_current_time", lambda: next(times))
    assert "August 31" in app_module.build_turn_context()
    assert "September 1" in app_module.build_turn_context()


def test_the_system_prompt_is_byte_identical_across_calls(app_module):
    """The cacheable-prefix property this whole change exists for: if
    the system prompt varies at all between conversations, the tool
    schemas behind it get re-charged at full price."""
    assert app_module.build_system_prompt() == app_module.build_system_prompt()


def test_system_prompt_tells_the_model_not_to_answer_time_from_memory(app_module):
    """Regression guard on the actual diagnostic finding: the model
    answered time questions by calling adjacent tools and narrating a
    plausible answer, rather than saying it couldn't tell."""
    prompt = app_module.build_system_prompt().lower()
    assert "get_current_time" in prompt
    assert "never answer" in prompt


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def test_registration_adds_the_clock_tool(app_module, clean_registry):
    o = clean_registry
    assert "get_current_time" not in o._TOOL_FUNCTIONS  # the bug, before the fix
    o.register_tools(app_module.tools_extra.TOOL_SCHEMAS, app_module.tools_extra.TOOL_FUNCTIONS)
    assert "get_current_time" in o._TOOL_FUNCTIONS
    assert any(s["function"]["name"] == "get_current_time" for s in o.TOOL_SCHEMAS)


def test_registration_replaces_phase_fours_stub_weather(app_module, clean_registry):
    o = clean_registry
    o.register_tools(app_module.tools_extra.TOOL_SCHEMAS, app_module.tools_extra.TOOL_FUNCTIONS)
    assert o._TOOL_FUNCTIONS["get_weather"] is app_module.tools_extra.get_weather


def test_registration_replaces_the_schema_too_not_just_the_function(app_module, clean_registry):
    """An override has to swap the description as well. Leaving Phase
    4's 'Get the current weather for a given location.' in place would
    keep pointing the model at a tool contract that no longer matches
    the implementation."""
    o = clean_registry
    o.register_tools(app_module.tools_extra.TOOL_SCHEMAS, app_module.tools_extra.TOOL_FUNCTIONS)
    weather_schemas = [s for s in o.TOOL_SCHEMAS if s["function"]["name"] == "get_weather"]
    assert len(weather_schemas) == 1
    assert "Open-Meteo" in weather_schemas[0]["function"]["description"]


def test_registration_leaves_phase_six_os_tools_alone(app_module, clean_registry):
    """Phase 8 overrides two Phase 4 tools and adds one. Nothing about
    the OS-control surface or its confirmation gate should move."""
    o = clean_registry
    before = {n for n in o._TOOL_FUNCTIONS if n not in ("get_weather", "search_web")}
    o.register_tools(app_module.tools_extra.TOOL_SCHEMAS, app_module.tools_extra.TOOL_FUNCTIONS)
    after = {n for n in o._TOOL_FUNCTIONS if n not in ("get_weather", "search_web", "get_current_time")}
    assert before == after


def test_registration_does_not_duplicate_any_tool(app_module, clean_registry):
    o = clean_registry
    o.register_tools(app_module.tools_extra.TOOL_SCHEMAS, app_module.tools_extra.TOOL_FUNCTIONS)
    names = [s["function"]["name"] for s in o.TOOL_SCHEMAS]
    assert len(names) == len(set(names))


def test_registered_tools_are_reachable_through_execute_tool(app_module, clean_registry):
    """The registry and the dispatch path are separate things -- this
    is the one that proves a registered tool can actually be called."""
    o = clean_registry
    o.register_tools(app_module.tools_extra.TOOL_SCHEMAS, app_module.tools_extra.TOOL_FUNCTIONS)
    result = o.execute_tool("get_current_time", {})
    assert "weekday" in result


def test_main_registers_before_running(app_module, clean_registry, monkeypatch):
    o = clean_registry
    seen = {}

    def fake_run(system_prompt_provider=None):
        seen["tools"] = set(o._TOOL_FUNCTIONS)
        seen["provider"] = system_prompt_provider

    monkeypatch.setattr(o, "run_orchestrator", fake_run)
    app_module.main()
    assert "get_current_time" in seen["tools"]
    assert seen["provider"] is app_module.build_system_prompt


# ---------------------------------------------------------------------------
# Milestone 2 wiring: every tool module, and the confirmation handler
# ---------------------------------------------------------------------------

def test_every_phase_eight_module_gets_registered(app_module, clean_registry):
    o = clean_registry
    app_module.register_everything()
    for module in app_module.TOOL_MODULES:
        for name in module.TOOL_FUNCTIONS:
            assert name in o._TOOL_FUNCTIONS, f"{name} was not registered"


def test_registration_produces_no_duplicate_tool_names(app_module, clean_registry):
    o = clean_registry
    app_module.register_everything()
    names = [s["function"]["name"] for s in o.TOOL_SCHEMAS]
    assert len(names) == len(set(names))


def test_schemas_and_functions_stay_in_sync_after_full_registration(app_module, clean_registry):
    o = clean_registry
    app_module.register_everything()
    assert {s["function"]["name"] for s in o.TOOL_SCHEMAS} == set(o._TOOL_FUNCTIONS)


def test_the_confirmation_gate_is_never_exposed_as_a_tool(app_module, clean_registry):
    """The single most important assertion in this file. Phase 8 adds
    several destructive tools; if confirm or cancel were ever reachable
    through the registry, the model could approve its own proposals and
    the entire two-phase design would be decorative."""
    o = clean_registry
    app_module.register_everything()
    names = set(o._TOOL_FUNCTIONS)
    assert not names & {"confirm_pending_action", "cancel_pending_action", "propose"}


def test_phase_eight_tokens_route_to_phase_eights_gate(app_module, clean_registry):
    o = clean_registry
    app_module.register_everything()
    confirm, cancel = o._resolve_confirmation(app_module.gate.TOKEN_PREFIX + "abcd1234")
    assert confirm is app_module.gate.confirm_pending_action
    assert cancel is app_module.gate.cancel_pending_action


def test_phase_six_tokens_still_route_to_phase_sixs_gate(app_module, clean_registry):
    """Registering Phase 8's handler must not hijack Phase 6's tokens --
    delete_file and the click/hotkey gate still belong to Phase 6."""
    o = clean_registry
    app_module.register_everything()
    confirm, _ = o._resolve_confirmation("abcd1234")
    assert confirm is o.confirm_pending_action


def test_a_gated_phase_eight_tool_survives_the_full_round_trip(app_module, clean_registry, monkeypatch, tmp_path):
    """End-to-end through the orchestrator's own gate handling: a
    destructive Phase 8 tool proposes, the human declines, and nothing
    happens. Proves the routing seam actually connects -- without it,
    the token would reach Phase 6's registry and be rejected as
    invalid."""
    o = clean_registry
    app_module.register_everything()

    source, destination = tmp_path / "a.txt", tmp_path / "b.txt"
    source.write_text("new")
    destination.write_text("original")

    monkeypatch.setattr(o, "_prompt_for_confirmation", lambda message: False)
    proposal = o.execute_tool("move_path", {"source": str(source), "destination": str(destination)})
    outcome = o._handle_tool_result(proposal)

    assert outcome["status"] == "cancelled"
    assert destination.read_text() == "original"


def test_approving_a_gated_phase_eight_tool_completes_it(app_module, clean_registry, monkeypatch, tmp_path):
    o = clean_registry
    app_module.register_everything()

    source, destination = tmp_path / "a.txt", tmp_path / "b.txt"
    source.write_text("new")
    destination.write_text("original")

    monkeypatch.setattr(o, "_prompt_for_confirmation", lambda message: True)
    proposal = o.execute_tool("move_path", {"source": str(source), "destination": str(destination)})
    outcome = o._handle_tool_result(proposal)

    assert outcome["status"] == "moved"
    assert destination.read_text() == "new"


def test_system_prompt_covers_the_new_tools(app_module):
    prompt = app_module.build_system_prompt()
    for mentioned in ("list_windows", "focus_window", "find_files", "read_document", "close_window"):
        assert mentioned in prompt


def test_the_spoken_output_rule_is_the_last_thing_in_the_prompt(app_module):
    """Regression guard, from a bug caught by live verification: with
    Phase 8's additions appended after Phase 7's prompt, the "no
    markdown" rule stopped being the last thing read and the model
    started answering list-shaped questions with markdown bullets --
    which TTS reads aloud as literal punctuation. Position is the fix,
    so position is what's tested."""
    assert app_module.build_system_prompt().endswith(app_module._SPOKEN_OUTPUT_REMINDER)


def test_the_spoken_output_rule_names_the_formats_it_forbids(app_module):
    reminder = app_module._SPOKEN_OUTPUT_REMINDER.lower()
    for banned in ("markdown", "bullet", "numbered", "asterisk"):
        assert banned in reminder


# ---------------------------------------------------------------------------
# Milestone 4: character
# ---------------------------------------------------------------------------

def test_the_character_opens_the_prompt(app_module):
    """Identity belongs at the top. Phase 7's prompt opens by calling
    itself "a helpful personal assistant", which is the generic register
    this milestone replaces -- so the character has to come first."""
    assert app_module.build_system_prompt().startswith(app_module.JARVIS_CHARACTER)


def test_phase_sevens_tool_and_safety_guidance_survives_the_character(app_module):
    """Personality must not displace the confirmation-gate framing or
    the OS-tool guidance. It is added around them, never instead."""
    assert app_module.o.TOOL_SYSTEM_PROMPT in app_module.build_system_prompt()


def test_the_spoken_output_rule_is_still_last(app_module):
    """Milestone 2's lesson: appending silently demotes whatever used to
    be last. Adding a character must not push the no-markdown rule out
    of the final position."""
    assert app_module.build_system_prompt().endswith(app_module._SPOKEN_OUTPUT_REMINDER)


def test_the_character_is_re_anchored_near_the_end(app_module):
    """Several hundred words of tool guidance sit between the opening
    character statement and the reply."""
    assert "Stay in character" in app_module.build_system_prompt()


def test_the_character_defines_behaviour_not_just_adjectives(app_module):
    """"Be witty" licenses padding every answer with jokes. The rules
    that keep a personality usable are the restrictive ones."""
    character = app_module.JARVIS_CHARACTER.lower()
    assert "sir" in character
    assert "brief" in character
    assert "never" in character


def test_the_character_still_pushes_back_on_unwise_requests(app_module):
    """The point of the character is not deference. This thing can
    delete files and shut the machine down, so one that flatters rather
    than warns would be worse than no character at all."""
    character = app_module.JARVIS_CHARACTER.lower()
    assert "disapproval" in character or "unwise" in character
    assert "comply" in character


# ---------------------------------------------------------------------------
# Device pinning happens before the wake gate opens the microphone
# ---------------------------------------------------------------------------

def test_main_pins_the_audio_devices(app_module, monkeypatch):
    called = []
    monkeypatch.setattr(app_module.audio_device, "pin_devices", lambda *a, **k: called.append(True))
    monkeypatch.setattr(app_module, "register_everything", lambda: None)
    monkeypatch.setattr(app_module.o, "run_orchestrator", lambda **k: None)

    app_module.main()

    assert called, "main() must pin the audio devices"


def test_pinning_happens_before_the_orchestrator_starts(app_module, monkeypatch):
    # Ordering is the whole point: the wake gate opens the microphone
    # as its first act inside run_orchestrator(), so pinning afterwards
    # would be too late for the component that most needs it.
    order = []
    monkeypatch.setattr(app_module.audio_device, "pin_devices", lambda *a, **k: order.append("pin"))
    monkeypatch.setattr(app_module, "register_everything", lambda: None)
    monkeypatch.setattr(app_module.o, "run_orchestrator", lambda **k: order.append("run"))

    app_module.main()

    assert order == ["pin", "run"]
