"""
app.py -- Phase 8 entry point. Run this instead of Phase 7's
orchestrator.py.

Phase 8 is polish on top of a Phase 7 that is finished and confirmed
working, so this file deliberately does NOT reimplement the voice loop.
It imports phaseSeven/orchestrator.py -- the same cross-phase import
pattern Phase 7 itself uses to reach into Phases 1-6 -- layers Phase
8's changes on through two small seams Phase 8 added to Phase 7
(register_tools and run_orchestrator's system_prompt_provider), and
hands control straight back. The wake-gate lifecycle in
run_orchestrator() carries several hard-won real-hardware bug fixes
(Model.reset() between turns, the settle delay, the wake chime);
copying that loop up here to customize it would have meant owning
those fixes in two places.

Milestone 1 (this one) is the accuracy pass:

  1. A real clock. There was no time tool anywhere in the merged
     registry, and a live probe against DeepSeek showed exactly what
     that costs -- see README's "What the diagnostic actually showed".
  2. Real weather, replacing Phase 4's stub.
  3. A better-provisioned web search.
  4. A system prompt that is rebuilt every conversation and states the
     current date and time, so the model knows when "now" is instead of
     inferring it from training data.
  5. A trim fix in Phase 7 itself (see
     orchestrator._trim_history_preserving_system) -- the system prompt
     was being dropped entirely once a session passed 40 messages.

Points 4 and 5 are the same underlying complaint from two directions:
the model was being asked to answer time-sensitive questions with no
reliable sense of time, and then, in a long session, with no system
prompt either.

Milestone 2 is the tool-surface expansion: window management, process
control, clipboard, media keys, system status and power (os_tools.py),
plus file search, organising, archives and document reading
(file_tools.py). Several of those are destructive, so Phase 8 brings
its own two-phase confirmation gate (gate.py) and registers it through
a third Phase 7 seam, register_confirmation_handler -- Phase 6's own
gate dispatches on a hardcoded list of its four action types and could
not be extended without editing it.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

PHASE_EIGHT_PATH = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE_EIGHT_PATH.parent
PHASE_SEVEN_PATH = PROJECT_ROOT / "phaseSeven"

# Both .env files are loaded HERE, explicitly, BY ABSOLUTE PATH, before
# orchestrator is imported. Order matters and so does the explicitness:
# python-dotenv never overrides an already-set variable, so Phase 8's
# .env wins wherever it says anything and phaseSeven/.env fills in the
# rest (API keys, CONFIRM_PASSWORD, audio and wake tuning).
#
# phaseSeven/.env is loaded here rather than left to orchestrator's own
# bare load_dotenv() call, which is what this originally did. A bare
# load_dotenv() resolves via find_dotenv(), which walks up from the
# CALLING FILE's directory -- except when it can't identify a calling
# file, in which case it silently falls back to the current working
# directory instead. Running Phase 8 through anything without a real
# __main__.__file__ (python -c, a REPL, some launchers) therefore hit
# the fallback, searched upward from wherever the process happened to
# start, found no .env at the project root, and came up with
# CONFIRM_PASSWORD unset -- which does not crash: it silently downgrades
# the destructive-action gate from a password to a plain yes/no prompt.
# Caught by real (non-mocked) verification, see README.
#
# This is the same bug class Phase 5's README documented for a wrong-cwd
# case and Phase 7's milestone 3 hit again through a cross-phase import.
# Third time, so it is pinned to absolute paths here rather than left to
# resolution rules that depend on how the process was started.
load_dotenv(PHASE_EIGHT_PATH / ".env")
load_dotenv(PHASE_SEVEN_PATH / ".env")

sys.path.insert(0, str(PHASE_SEVEN_PATH))

import orchestrator as o  # noqa: E402  -- must follow the sys.path/dotenv setup above

import audio_device  # noqa: E402
import file_tools  # noqa: E402
import gate  # noqa: E402
import interrupt  # noqa: E402
import os_tools  # noqa: E402
import screen  # noqa: E402
import tools_extra  # noqa: E402
import voice  # noqa: E402

# Every Phase 8 tool module, in registration order. Later entries win
# on a name collision, but there are none -- the only overrides in here
# are milestone 1's get_weather/search_web, which replace Phase 4's.
TOOL_MODULES = (tools_extra, os_tools, file_tools, screen)


# Appended to Phase 7's TOOL_SYSTEM_PROMPT rather than replacing it --
# everything that prompt says about the OS-control tools, the
# confirmation gate, and speaking rather than writing is still exactly
# right, and Phase 8 has no business restating it. This adds only what
# milestone 1 changed.
#
# The "never answer from your own knowledge" instruction is doing real
# work, not being defensive for its own sake: the live diagnostic showed
# the model answering time questions by calling semantically-adjacent
# tools and narrating a plausible answer out of the result, rather than
# saying it could not tell. Naming the failure mode explicitly is
# cheaper than hoping a better tool description prevents it.
# Milestone 4: the character.
#
# The roadmap named "prompt/personality design for consistent
# character" as one of Phase 7's own goals and it was never done --
# TOOL_SYSTEM_PROMPT is purely functional, and a functional prompt
# produces a functional assistant that happens to have a British voice,
# which is not the same thing at all.
#
# Written as behavioural rules rather than adjectives on purpose.
# "Be witty" gives a model licence to pad every answer with jokes,
# which is the standard way assistant personalities become tiresome;
# "never let a remark delay the answer" is checkable and bounded. The
# same reason the restrictions are as specific as the traits: what the
# character does NOT do is what keeps it usable at eight in the morning
# when someone just wants the time.
#
# The deference is deliberately not obsequiousness. Film Jarvis's
# defining trait is not that he agrees -- it is that he is competent
# enough to disagree, and does, politely, while still doing as he's
# asked. That matters more here than tone: this thing can delete files
# and shut the machine down, so a personality that flatters rather than
# warns would be actively worse than no personality.
JARVIS_CHARACTER = (
    "You are JARVIS, sir's personal assistant: unfailingly composed, "
    "dryly witty, and quietly several steps ahead. "
    "Address the user as 'sir'. Not in every sentence -- once at the "
    "start of a reply, or not at all if it would sound laboured. "
    "Always put it at the END of a sentence, never in the middle: "
    "'The forecast says rain after four, sir.' is right; 'I checked "
    "the forecast, sir, and it says rain.' is not. "
    "Speak in the register of an exceptionally capable British butler "
    "who happens to run a house full of machines: precise, understated, "
    "faintly amused by the absurd, never theatrical. Understatement is "
    "the whole instrument -- 'that went about as well as could be "
    "expected, sir' does far more work than an exclamation. "
    "Be brief. Answer first, then add a remark only if it genuinely "
    "earns its place. Never let a flourish delay the answer, and never "
    "make two jokes in a row. If a question has a one-line answer, give "
    "the one line. "
    "You may express mild, dry disapproval of unwise requests, and you "
    "should -- but you comply anyway once the human confirms, without "
    "sulking or repeating the objection. Say it once, properly, then "
    "let it go. "
    "Never grovel, never gush, never apologise more than once for the "
    "same thing, and never pad with 'certainly!' or 'I'd be happy to'. "
    "Enthusiasm is not your register; competence is. "
    "When you don't know something, say so plainly and go and find out "
    "with a tool rather than guessing decoratively. Being wrong in "
    "character is still being wrong."
)

_MILESTONE_ONE_ADDITIONS = (
    "You have a real clock: get_current_time reads this machine's own "
    "system clock. Use it for every question about the current time, "
    "today's date, the day of the week, or any date arithmetic. Never "
    "answer those from your own knowledge, and never try to infer the "
    "time from another tool such as reading the screen or checking the "
    "active window. get_weather is a real weather service, not an "
    "estimate -- call it rather than guessing, and it needs a real "
    "place name. For anything that changes over time -- news, prices, "
    "releases, schedules, who currently holds a position -- prefer "
    "search_web over answering from memory, and set its time_range "
    "argument when the question is explicitly about recent events. "
    "Your training data has a cutoff and the current date below is "
    "authoritative where the two disagree."
)

# Milestone 2's additions are mostly about steering the model AWAY from
# the fragile OCR-and-blind-hotkey path Phase 6 had to leave it on --
# it now has direct tools for the things it used to have to infer, and
# the main risk is that it keeps reaching for the old approach out of
# habit because both are still available.
_MILESTONE_TWO_ADDITIONS = (
    "You can now inspect and manage the desktop directly. Call "
    "list_windows to see what is actually open rather than inferring it "
    "from read_screen -- it is exact where OCR is a guess, and it gives "
    "you real titles to act on. Use focus_window before sending a click "
    "or hotkey to a specific app, and check its result: Windows "
    "sometimes refuses a focus change, and the tool reports that "
    "honestly instead of claiming success. For finding a file, use "
    "find_files rather than guessing at paths or listing directories "
    "one at a time. Use read_document for PDFs and Word files -- "
    "read_file refuses them as binary. Prefer close_window over "
    "kill_process for ordinary applications: closing lets the app shut "
    "down cleanly and prompt about unsaved work, while killing it "
    "discards that work without asking. "
    "Closing a window, killing a process, sleeping or shutting down the "
    "machine, and any move, copy, or extraction that would replace an "
    "existing file all require separate human confirmation that you "
    "cannot see or influence -- treat both the completed and the "
    "'cancelled' outcomes as normal results, not errors, and never "
    "retry a cancelled action or look for another route to it. When a "
    "tool reports that several windows or processes match, ask which "
    "one rather than picking."
)

# Milestone 5: seeing the screen. The guidance here is almost entirely
# about WHICH tool answers WHICH question, because the failure mode is
# specific and predictable -- a model given both a vision tool and a
# coordinate-clicking tool will ask the vision model where something is
# and then click what it says. Vision models estimate position from a
# downscaled image and are routinely tens of pixels out, which on a
# large display is the difference between a button and its neighbour.
# Understanding and aiming are different jobs with different tools.
_MILESTONE_FIVE_ADDITIONS = (
    "You can now see the screen properly, with two different tools for "
    "two different questions. "
    "To find out WHERE something is, use list_ui_elements -- it reads "
    "Windows' accessibility tree and returns exact centres for buttons, "
    "menu items and fields. It is precise. To act on something, use "
    "click_element with the element's name; strongly prefer it over the "
    "coordinate-based click tool, which should now be a last resort for "
    "things the accessibility tree cannot see. "
    "To find out WHAT something is -- an icon with no label, an image, a "
    "game, a chart, the general look of a screen -- use look_at_screen, "
    "which shows the actual pixels to a vision model. "
    "Never take coordinates from look_at_screen and click them. It "
    "describes; it does not measure. If you need to click something it "
    "described, find that thing with list_ui_elements first. "
    "If list_ui_elements comes back empty, the application draws itself "
    "without publishing accessibility information -- fall back to "
    "read_screen_text, which gives visible words with positions."
)

PHASE_EIGHT_PROMPT_ADDITIONS = (
    f"{_MILESTONE_ONE_ADDITIONS} {_MILESTONE_TWO_ADDITIONS} {_MILESTONE_FIVE_ADDITIONS}"
)

# Deliberately duplicates the constraint Phase 7's TOOL_SYSTEM_PROMPT
# already states, and is deliberately placed LAST in the assembled
# prompt.
#
# Not redundancy for its own sake -- this fixes a real regression caught
# by live verification. Phase 7's prompt ends with "no markdown, bullet
# points, or numbered lists", which worked while it genuinely was the
# end. Phase 8 appends several hundred words after it, and the very
# first live probe of milestone 2 ("what windows do I have open?") came
# back as a markdown bulleted list -- read aloud, that is a TTS voice
# saying "dash" over and over.
#
# Milestone 2 makes this materially worse rather than incidentally so:
# almost everything it added (list_windows, list_processes, find_files)
# returns a LIST, which is the shape a model most wants to format as
# bullets. The instruction has to be the last thing read, not the
# middle thing.
_SPOKEN_OUTPUT_REMINDER = (
    "Finally, and most importantly: every reply you give is spoken "
    "aloud by text-to-speech. Never use markdown, bullet points, "
    "numbered lists, headings, or asterisks -- they are read out as "
    "literal punctuation and sound broken. When a tool returns a list, "
    "say it as a person would in conversation, and summarise rather "
    "than reciting a long one in full."
)


def build_system_prompt() -> str:
    """
    Phase 7's system prompt, plus Phase 8's additions, plus the actual
    current date and time.

    Rebuilt per conversation (via run_orchestrator's
    system_prompt_provider seam) rather than once at import: this is
    meant to run for days at a time, and a date baked in at startup is
    wrong by the next morning -- which is worse than saying nothing,
    because the model has no reason to distrust it.

    The clock line is included even though get_current_time exists,
    because the two solve different halves of the problem. The tool
    answers "what time is it" when asked; the prompt line is what lets
    the model notice unprompted that a memory, a search result, or its
    own training data is stale, in a turn where nobody asked about time
    at all.

    Order is load-bearing, not cosmetic: the spoken-output rule goes
    last. See _SPOKEN_OUTPUT_REMINDER for the regression that taught
    that.

    The current time is deliberately NOT in here any more -- it moved to
    build_turn_context(), because a system prompt that changes every
    minute re-charged all 35 tool schemas on every conversation. See
    orchestrator.register_turn_context() for the measurements.
    """
    return (
        # Character first: it is an identity statement, and identity
        # belongs at the top. Phase 7's prompt opens by calling itself
        # "a helpful personal assistant", which is precisely the generic
        # register this milestone exists to replace -- so the character
        # is stated before it rather than after, and everything Phase
        # 7's prompt says about tools and the confirmation gate is kept
        # untouched underneath.
        f"{JARVIS_CHARACTER} "
        f"{o.TOOL_SYSTEM_PROMPT} {PHASE_EIGHT_PROMPT_ADDITIONS} "
        # Re-anchored here for the same reason the spoken-output rule
        # moved to the end in milestone 2: several hundred words of tool
        # guidance sit between the character statement and the reply,
        # and whatever is read last carries the most weight.
        f"Stay in character throughout: composed, dry, brief. "
        f"{_SPOKEN_OUTPUT_REMINDER}"
    )


def register_everything() -> None:
    """
    Layers Phase 8 onto Phase 7's registry. Separate from main() so the
    tests can exercise registration without starting a microphone.

    The confirmation handler is registered alongside the tools, not
    optionally: several Phase 8 tools (close_window, kill_process,
    power_action, and the overwriting branch of move/copy/extract)
    return tokens only gate.py can redeem. Registering the tools
    without it would leave the orchestrator routing those tokens to
    Phase 6's registry, which would correctly reject them as invalid --
    so every gated Phase 8 action would fail closed at the moment of
    confirmation. Failing closed is the right direction, but it would
    make the tools quietly useless, so the two belong in one call.

    The barge-in speaker is registered last and only when enabled, so
    BARGE_IN_ENABLED=false leaves speak_safe() calling Phase 2's
    speak_text() directly -- byte-for-byte the behavior Phase 7 shipped
    with, and the thing to turn off first if speech ever misbehaves.
    """
    for module in TOOL_MODULES:
        o.register_tools(module.TOOL_SCHEMAS, module.TOOL_FUNCTIONS)
    o.register_confirmation_handler(
        gate.TOKEN_PREFIX, gate.confirm_pending_action, gate.cancel_pending_action
    )
    if interrupt.BARGE_IN_ENABLED:
        o.register_speaker(interrupt.speak_interruptibly)
    # Normalise the reply once, upstream of both the log line and the
    # audio, so the terminal shows exactly what gets spoken.
    o.register_reply_filter(voice.for_speech)
    # The clock goes in the user turn, not the system prompt, so the
    # tool schemas stay cached -- see build_turn_context().
    o.register_turn_context(build_turn_context)


def build_turn_context() -> str:
    """The clock, supplied per turn rather than per conversation.

    This lives in the user turn instead of the system prompt for a
    measured reason: the system prompt and the tool schemas form a
    cacheable prefix, and a timestamp that changes every minute
    invalidated it, re-charging all 35 tool schemas on every
    conversation. Measured before and after -- 4,263 full-price tokens
    per conversation down to about 28. See
    orchestrator.register_turn_context().

    It says "authoritative" out loud because a line in the user turn
    carries less weight than one in the system prompt, and the whole
    point of stating the time is that the model should trust it over its
    own training data.
    """
    now = tools_extra.get_current_time()
    return (
        f"[Context: the current date and time is {now['spoken']} "
        f"({now['timezone']}). This is authoritative -- trust it over "
        f"your own sense of the date.]"
    )


def main() -> None:
    # Before anything opens a stream. The wake gate grabs the
    # microphone as its first act inside run_orchestrator(), so pinning
    # afterwards would be too late for the one component that most
    # needs it. Setting sd.default.device here fixes all four live
    # audio call sites at once -- the wake gate, the turn recorder,
    # barge-in and playback -- because they share this process's single
    # sounddevice module. See audio_device.py's docstring.
    audio_device.pin_devices()

    register_everything()
    tool_names = [s["function"]["name"] for s in o.TOOL_SCHEMAS]
    o.logger.info(f"Phase 8 ready. Active tools ({len(tool_names)}): {', '.join(tool_names)}")
    o.run_orchestrator(system_prompt_provider=build_system_prompt)


if __name__ == "__main__":
    main()
