"""
gate.py -- Phase 8's two-phase confirmation registry.

Phase 6 built this pattern and Phase 7 wired it into voice. Phase 8
needs it again for its own destructive tools (killing a process,
closing a window, putting the machine to sleep, overwriting a file
during a move or copy), and cannot simply reuse Phase 6's registry:
Phase 6's confirm_pending_action() dispatches on a hardcoded
if/elif chain over its own four action types (delete, overwrite,
click, hotkey) and has no idea what "kill this process" means.

So this is the same guarantee, implemented once more, generically. The
shape of the contract is deliberately identical to Phase 6's, because
orchestrator._handle_tool_result() already understands that shape and
should not have to learn a second one:

    propose()  -> {"status": "confirmation_required", "token": ..., "message": ...}
    confirm()  -> {"status": "<past-tense verb>", ...} or {"error": ...}
    cancel()   -> {"status": "cancelled", ...}

The property that actually matters is preserved exactly: **the thing
that does the damage is a Python callable held in this module's own
memory, keyed by a token.** The model never sees it, never receives a
path to it, and cannot call it. It can only ever get as far as a
proposal, which a human has to approve out-of-band -- at a real
terminal, with a real password, on a channel the model and the
microphone both cannot reach. Phase 5's README documented a memory
shaped like a prompt injection hijacking a whole session; Phase 6 and
Phase 7 both named poisoned-memory-to-destructive-tool-call as the
realistic threat. Phase 8 adds more destructive tools to that same
loop, so it adds no new way around the gate.

Tokens are prefixed (see TOKEN_PREFIX) so the orchestrator can route a
confirmation to the registry that actually owns it -- Phase 6's tokens
are bare hex and stay that way, untouched.
"""

import time
import uuid

# Matches Phase 6's PENDING_TTL_SECONDS exactly. Not independently
# re-derived: a person confirming a Phase 8 action at the terminal is
# doing literally the same thing, at the same prompt, under the same
# time pressure as one confirming a Phase 6 action. Two different
# timeouts for one experience would be a bug waiting to be noticed.
PENDING_TTL_SECONDS = 300

# Namespaces Phase 8's tokens so orchestrator._resolve_confirmation()
# can tell them from Phase 6's bare-hex ones and route accordingly.
TOKEN_PREFIX = "p8-"

# token -> {"action": str, "message": str, "execute": callable,
#           "details": dict, "created_at": float}
_pending_actions = {}


def _prune_expired() -> None:
    now = time.time()
    for token in [
        t for t, a in _pending_actions.items()
        if now - a["created_at"] > PENDING_TTL_SECONDS
    ]:
        del _pending_actions[token]


def propose(action: str, message: str, execute, **details) -> dict:
    """
    Registers a destructive action and returns the proposal dict the
    model sees. Nothing happens until confirm() is called.

    `execute` is a zero-argument callable returning the outcome dict.
    Holding a closure rather than a description-to-be-re-parsed means
    the action that eventually runs is the exact one that was proposed
    -- there is no second interpretation step between the human reading
    the message and the thing firing, and so no gap for the arguments
    to drift.

    `message` is what the human reads at the terminal. It is the entire
    basis for their decision, so it must name the concrete target --
    the actual process, window, or path -- never a summary.
    """
    _prune_expired()
    token = TOKEN_PREFIX + uuid.uuid4().hex[:8]
    _pending_actions[token] = {
        "action": action,
        "message": message,
        "execute": execute,
        "details": details,
        "created_at": time.time(),
    }
    return {
        "status": "confirmation_required",
        "token": token,
        "message": message,
        **details,
    }


def confirm_pending_action(token: str) -> dict:
    """
    Executes a previously-proposed action. NEVER reachable from the
    tool-calling loop -- deliberately absent from every TOOL_FUNCTIONS
    and TOOL_SCHEMAS in this phase, exactly as Phase 6 keeps its own
    out of its LLM-facing surface.

    Deliberately does NOT prune before the lookup, reusing Phase 6's
    own reasoning verbatim: pruning first would erase the difference
    between "this token never existed or was already used" and "this
    token existed but you took too long." A human who waited past the
    TTL and got a generic "invalid token" cannot tell that from a typo;
    one told specifically that it expired knows to just ask again.
    """
    action = _pending_actions.get(token)
    if action is None:
        return {"error": "That confirmation token is invalid or was already used. Nothing was changed."}

    if time.time() - action["created_at"] > PENDING_TTL_SECONDS:
        del _pending_actions[token]
        return {
            "error": (
                f"That confirmation expired after sitting unanswered for "
                f"{PENDING_TTL_SECONDS} seconds. Nothing was changed. "
                "Ask again to get a fresh confirmation."
            )
        }

    # Removed before execution, not after: a token that fired and then
    # raised must not remain in the registry for a second attempt. The
    # human confirmed one action, not a retry loop.
    del _pending_actions[token]

    try:
        return action["execute"]()
    except Exception as e:
        return {"error": f"Action failed: {e}"}


def cancel_pending_action(token: str) -> dict:
    """Discards a proposed action without running it. Same shape as
    Phase 6's, including tolerating an unknown token -- a cancel that
    finds nothing has still achieved what the human wanted, so it is
    not reported as an error."""
    action = _pending_actions.pop(token, None)
    if action is None:
        return {"status": "cancelled", "note": "Nothing was pending. Nothing was changed."}
    return {"status": "cancelled", "action": action["action"], **action["details"]}


def pending_count() -> int:
    """Test/debug helper. Not a tool -- never exposed to the model."""
    return len(_pending_actions)
