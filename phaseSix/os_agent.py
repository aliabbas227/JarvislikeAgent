"""
os_agent.py — Phase 6 entry point: CLI agent with OS-control tools.

Reuses Phase 4's llm_client.py (build_client / call_llm_with_tools,
DeepSeek backend) the same deliberate way Phase 2's voice_chat.py and
Phase 5's memory_chat.py reused Phase 1's jarvis.py — proven, tested
infrastructure, not something this phase needs to relearn. A local
model was flagged in Phase 4 as unreliable for tool-calling in
general; that concern only gets sharper once one of the tools can
delete a file, so there's no case for reverting to Ollama here.

Phase 6's actual new thing is tools.py's OS-control tools and, more
importantly, the confirmation gate below. This module does NOT import
phaseFour/tools.py, phaseFour/agent.py's loop, phaseTwo/voice_io.py,
phaseThree/wake_word.py, or phaseFive/memory.py — standalone-phase
discipline holds. Wiring all of these together is Phase 7's job.

Adjust PHASE_FOUR_PATH if your folder layout differs from
PROJECTS/JARVIS/phaseFour, phaseSix/.
"""

import json
import getpass
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("os_agent")

from tools import TOOL_SCHEMAS, execute_tool, confirm_pending_action, cancel_pending_action

PHASE_FOUR_PATH = Path(__file__).resolve().parent.parent / "phaseFour"
sys.path.insert(0, str(PHASE_FOUR_PATH))

try:
    from llm_client import build_client, call_llm_with_tools
except ImportError:
    logger.error(
        f"Could not import Phase 4's llm_client.py from {PHASE_FOUR_PATH}. "
        "Update PHASE_FOUR_PATH at the top of os_agent.py to point at your "
        "phaseFour folder."
    )
    raise

MAX_TOOL_HOPS = 8  # safety cap so a confused model can't loop forever.
# Raised from 4 after repeated manual testing showed multi-step UI tasks
# (open an app, check focus, type, press a hotkey, check again...)
# routinely need more than 4 round-trips even when everything's working
# correctly -- 4 was cutting off legitimate sequences, not catching
# actual runaway loops. Still bounded, just less trigger-happy.

# Long text fields (screen OCR output, file contents) get truncated to
# this many characters before being written to the log -- a single tool
# call should never dump an entire screen capture or file's contents
# into a log file that might persist, get shared for debugging, or end
# up in version control. The model still gets the full, untruncated
# result; this only affects what lands in the log.
LOG_TEXT_TRUNCATE_CHARS = 300

# Optional. If set, confirming a destructive action requires typing this
# exact password (hidden input, via getpass) instead of just "yes" --
# see _prompt_for_confirmation() below for why this exists.
CONFIRM_PASSWORD = os.getenv("CONFIRM_PASSWORD")

if not CONFIRM_PASSWORD:
    logger.warning(
        "CONFIRM_PASSWORD is not set in .env -- destructive actions will fall back "
        "to a plain yes/no prompt. Set CONFIRM_PASSWORD for a stronger gate before "
        "using this unattended or once voice input is wired in (Phase 7)."
    )


def _prompt_for_confirmation(message: str) -> bool:
    """Prompts a real human at the terminal and returns True only if
    they approve. If CONFIRM_PASSWORD is configured, this requires
    typing that exact password (hidden, via getpass) rather than just
    the word "yes" -- raises the bar against anything other than a
    deliberate, present human approving. A plain "yes" is trivially
    easy to satisfy by accident (a stray word picked up by a future
    voice-input phase, background noise, someone else in the room,
    a script that happens to feed stdin) in a way a specific password
    isn't. Falls back to the plain yes/no prompt if no password is
    configured, per the startup warning above.

    Fails closed (returns False, i.e. cancels) on any input error --
    getpass can raise in some terminals/environments without a real
    tty, and a destructive action should never proceed just because
    the confirmation prompt itself broke."""
    print(f"\n[CONFIRMATION NEEDED] {message}")

    if CONFIRM_PASSWORD:
        try:
            answer = getpass.getpass("Enter confirmation password to proceed (hidden): ")
        except Exception as e:
            logger.warning(f"Could not read confirmation password, cancelling: {e}")
            return False
        return answer == CONFIRM_PASSWORD

    answer = input("Type 'yes' to proceed, anything else to cancel: ").strip().lower()
    return answer == "yes"


def _handle_tool_result(result: dict) -> dict:
    """Intercepts any tool result that proposes a destructive action and
    forces a real, out-of-band human confirmation before anything
    happens -- this is the actual guardrail, not the model's judgment.
    The model never sees a confirmation token, a password, or any path
    to call confirm_pending_action itself; it only ever sees the final
    outcome (deleted / overwritten / cancelled / error)."""
    if result.get("status") != "confirmation_required":
        return result

    approved = _prompt_for_confirmation(result.get("message", "A destructive action was proposed."))
    token = result["token"]
    outcome = confirm_pending_action(token) if approved else cancel_pending_action(token)

    # Print the real outcome directly to the terminal, don't rely on the
    # model to relay it accurately in its reply -- a human who just typed
    # "yes" and then reads a vague, unrelated-sounding model response
    # (e.g. after an expired-token error) has no way to know what
    # actually happened without this. See PHASE6 manual testing notes:
    # an expired confirmation correctly blocked the deletion, but
    # nothing at the time explained *why* to the human in the moment.
    if "error" in outcome:
        print(f"[NOTE] {outcome['error']}")
    else:
        print(f"[NOTE] {outcome.get('status', 'done')}: {outcome.get('path', '')}")

    return outcome


def _summarize_for_log(result: dict) -> dict:
    """Returns a copy of `result` safe to write to the log, with any long
    text fields (read_screen's OCR output, read_file's file contents)
    truncated. See LOG_TEXT_TRUNCATE_CHARS above for why."""
    summary = dict(result)
    for key in ("text", "content"):
        value = summary.get(key)
        if isinstance(value, str) and len(value) > LOG_TEXT_TRUNCATE_CHARS:
            summary[key] = (
                value[:LOG_TEXT_TRUNCATE_CHARS]
                + f"... [{len(value)} chars total, truncated for log]"
            )
    return summary


def run_turn(client, messages: list) -> str:
    """Run one user turn to completion, same shape as Phase 4's
    run_turn — except every tool result is routed through
    _handle_tool_result() so a destructive proposal always stops for a
    real human before the model's turn is allowed to continue."""
    for _ in range(MAX_TOOL_HOPS):
        response = call_llm_with_tools(client, messages, TOOL_SCHEMAS)
        message = response.choices[0].message

        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            messages.append({"role": "assistant", "content": message.content})
            return message.content

        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [tc.model_dump() for tc in tool_calls],
            }
        )

        for tc in tool_calls:
            name = tc.function.name
            try:
                arguments = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                arguments = {}

            try:
                result = execute_tool(name, arguments)
                result = _handle_tool_result(result)
            except (KeyError, TypeError) as e:
                result = {"error": str(e)}

            # Always log the real result, even on the "boring" path — a
            # tool result the model then mis-describes in its reply (e.g.
            # claiming confirmation is still needed after a successful
            # delete) is otherwise invisible. Same reasoning Phase 5's
            # memory_chat.py used for always logging retrieval results.
            # Long text fields are truncated first (see _summarize_for_log).
            logger.info(f"Tool result for {name}: {_summarize_for_log(result)}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )

    return "(gave up after too many tool calls in a row — check MAX_TOOL_HOPS)"


def main():
    client = build_client()

    system_prompt = (
        "You are Jarvis, a helpful assistant with access to OS-control tools: "
        "listing directories, reading files, writing/creating files, reading the "
        "screen (OCR), launching applications, deleting files, moving the mouse, "
        "typing text, clicking, and pressing keyboard shortcuts. Deleting a file, "
        "overwriting an existing file, clicking, and pressing a hotkey all require "
        "separate human confirmation that you cannot see or influence — treat "
        "'deleted'/'overwritten'/'clicked'/'pressed' or 'cancelled' outcomes as "
        "normal, not errors. Moving the mouse and typing text happen immediately "
        "without confirmation. "
        "read_screen is OCR text extraction, NOT vision — it can read words on "
        "screen but cannot locate icons, buttons, or other non-text UI elements, "
        "and does not return coordinates for anything it reads. Do not try to "
        "find a button by guessing screen regions and reading them repeatedly; "
        "for standard window actions (closing, minimizing, switching apps), "
        "prefer a keyboard shortcut via press_hotkey (e.g. alt+f4 to close the "
        "active window) over hunting for a click target you cannot actually see. "
        "If something you expect isn't visible with a normal read_screen call, "
        "it may be on a second monitor — try read_screen with all_screens=true "
        "before assuming it isn't open. Before sending a hotkey or click meant "
        "for a specific app (especially one just launched via open_application), "
        "call get_active_window first to confirm that app actually has focus — "
        "a newly-launched window is not guaranteed to be focused, and a hotkey "
        "like alt+f4 applies to whatever IS focused, not whatever you intended. "
        "click and press_hotkey both REQUIRE a target_window_contains argument "
        "naming the window you believe is focused (e.g. 'Notepad') — this is "
        "checked against reality before proposing and again right before firing. "
        "This matters most right after the human sends a message (e.g. 'go "
        "ahead' after you said you'd do something but didn't call the tool yet) "
        "— typing that message can shift OS focus back to this terminal, so "
        "always call get_active_window fresh in that situation rather than "
        "assuming the app you launched earlier is still focused. Keep replies "
        "concise."
    )
    messages = [{"role": "system", "content": system_prompt}]

    print("Jarvis Phase 6 agent (OS control, DeepSeek backend). Type 'exit' to quit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nJarvis: Shutting down. Goodbye.")
            break

        if user_input.lower() in ("exit", "quit"):
            print("Jarvis: Goodbye.")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        reply = run_turn(client, messages)
        print(f"Jarvis: {reply}\n")


if __name__ == "__main__":
    # Wrapped so a crash before/inside main() always prints *something*
    # human-readable, rather than potentially exiting silently in some
    # terminal setups (observed: `python os_agent.py` returning straight
    # to the prompt with no output and no visible traceback).
    try:
        main()
    except Exception:
        import traceback
        print("\n[FATAL] os_agent.py crashed on startup. Full traceback:\n", file=sys.stderr)
        traceback.print_exc()
        input("\nPress Enter to close...")
        raise