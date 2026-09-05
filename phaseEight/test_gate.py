"""
test_gate.py -- Phase 8 milestone 2 tests for the confirmation registry.

This is the most safety-sensitive code in the phase, so the tests are
mostly about what must NOT happen: an action running without
confirmation, a token working twice, an expired token still firing, or
the model getting any path to the executor.
"""

import pytest

import gate


@pytest.fixture(autouse=True)
def clean_registry():
    """The registry is module-level state shared across the process."""
    gate._pending_actions.clear()
    yield
    gate._pending_actions.clear()


def _spy():
    """Returns (calls_list, callable) so a test can assert the executor
    was or was not run."""
    calls = []

    def execute():
        calls.append(True)
        return {"status": "done"}

    return calls, execute


# ---------------------------------------------------------------------------
# propose
# ---------------------------------------------------------------------------

def test_propose_does_not_execute_anything():
    """The entire point of the phase's safety model in one assertion."""
    calls, execute = _spy()
    gate.propose("thing", "About to do a thing.", execute)
    assert calls == []


def test_propose_returns_the_shape_the_orchestrator_expects():
    """orchestrator._handle_tool_result() keys off exactly these
    fields. A different shape means the gate silently doesn't engage
    and the result goes straight to the model as if it were an outcome."""
    result = gate.propose("thing", "About to do a thing.", lambda: {})
    assert result["status"] == "confirmation_required"
    assert result["token"]
    assert result["message"] == "About to do a thing."


def test_propose_namespaces_its_tokens():
    """Routing in orchestrator._resolve_confirmation() is by prefix --
    an unprefixed token would be handed to Phase 6's registry, which
    would reject it."""
    assert gate.propose("thing", "msg", lambda: {})["token"].startswith(gate.TOKEN_PREFIX)


def test_propose_includes_details_in_the_result():
    result = gate.propose("kill", "msg", lambda: {}, name="thing.exe", pid=42)
    assert result["name"] == "thing.exe"
    assert result["pid"] == 42


def test_propose_does_not_leak_the_executor_to_the_caller():
    """The result dict is JSON-serialized into the model's context. A
    callable in there would both break serialization and represent the
    model being handed the thing it must never reach."""
    result = gate.propose("thing", "msg", lambda: {})
    assert not any(callable(v) for v in result.values())


def test_each_proposal_gets_a_distinct_token():
    tokens = {gate.propose("t", "m", lambda: {})["token"] for _ in range(50)}
    assert len(tokens) == 50


# ---------------------------------------------------------------------------
# confirm
# ---------------------------------------------------------------------------

def test_confirm_runs_the_executor_and_returns_its_outcome():
    calls, execute = _spy()
    token = gate.propose("thing", "msg", execute)["token"]
    assert gate.confirm_pending_action(token) == {"status": "done"}
    assert calls == [True]


def test_confirm_runs_the_exact_action_that_was_proposed():
    """Holding a closure rather than a description means there is no
    re-interpretation step between what the human read and what fires."""
    seen = []
    token = gate.propose("del", "msg", lambda: seen.append("/real/path") or {"status": "ok"})["token"]
    gate.confirm_pending_action(token)
    assert seen == ["/real/path"]


def test_a_token_cannot_be_used_twice():
    calls, execute = _spy()
    token = gate.propose("thing", "msg", execute)["token"]
    gate.confirm_pending_action(token)
    assert "error" in gate.confirm_pending_action(token)
    assert calls == [True]


def test_an_unknown_token_does_nothing():
    assert "error" in gate.confirm_pending_action("p8-deadbeef")


def test_an_expired_token_does_not_execute():
    calls, execute = _spy()
    token = gate.propose("thing", "msg", execute)["token"]
    gate._pending_actions[token]["created_at"] -= gate.PENDING_TTL_SECONDS + 1
    assert "expired" in gate.confirm_pending_action(token)["error"]
    assert calls == []


def test_expired_and_invalid_are_distinguishable():
    """Reused verbatim from Phase 6's reasoning: a human who waited too
    long and a human with a typo need different messages, or the first
    has no way to know what to do next."""
    token = gate.propose("thing", "msg", lambda: {})["token"]
    gate._pending_actions[token]["created_at"] -= gate.PENDING_TTL_SECONDS + 1
    expired = gate.confirm_pending_action(token)["error"]
    invalid = gate.confirm_pending_action("p8-nothing")["error"]
    assert expired != invalid


def test_a_token_is_consumed_even_if_the_action_raises():
    """A confirmed action that blew up must not stay redeemable. The
    human approved one attempt, not a retry loop."""
    def boom():
        raise OSError("disk on fire")

    token = gate.propose("thing", "msg", boom)["token"]
    assert "error" in gate.confirm_pending_action(token)
    assert gate.pending_count() == 0
    assert "invalid or was already used" in gate.confirm_pending_action(token)["error"]


def test_an_exception_in_the_action_is_reported_not_raised():
    """A tool that raises would take down the whole voice turn; the
    convention everywhere in this project is an error dict."""
    token = gate.propose("thing", "msg", lambda: 1 / 0)["token"]
    assert "error" in gate.confirm_pending_action(token)


# ---------------------------------------------------------------------------
# cancel
# ---------------------------------------------------------------------------

def test_cancel_does_not_execute():
    calls, execute = _spy()
    token = gate.propose("thing", "msg", execute)["token"]
    assert gate.cancel_pending_action(token)["status"] == "cancelled"
    assert calls == []


def test_cancel_consumes_the_token():
    token = gate.propose("thing", "msg", lambda: {})["token"]
    gate.cancel_pending_action(token)
    assert "error" in gate.confirm_pending_action(token)


def test_cancelling_nothing_is_not_an_error():
    """A cancel that finds nothing pending still achieved what the
    human wanted."""
    assert gate.cancel_pending_action("p8-nothing")["status"] == "cancelled"


# ---------------------------------------------------------------------------
# expiry housekeeping
# ---------------------------------------------------------------------------

def test_proposing_prunes_expired_actions():
    stale = gate.propose("old", "msg", lambda: {})["token"]
    gate._pending_actions[stale]["created_at"] -= gate.PENDING_TTL_SECONDS + 1
    gate.propose("new", "msg", lambda: {})
    assert stale not in gate._pending_actions


def test_pruning_does_not_touch_live_actions():
    live = gate.propose("live", "msg", lambda: {})["token"]
    gate.propose("other", "msg", lambda: {})
    assert live in gate._pending_actions
