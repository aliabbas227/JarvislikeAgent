"""
orchestrator.py -- Phase 7: wake word -> listen -> transcribe -> LLM
(with tools) -> speak, in one continuous loop.

Staged build (per project discipline, extended into Phase 7 itself):
    Milestone 1: wake word + voice + LLM, no tools, no memory, no OS
        control.
    Milestone 2: fold in Phase 4's tool-calling loop.
    Milestone 3: fold in Phase 5's memory retrieval/storage.
    Milestone 4 (this file): fold in Phase 6's OS-control tools, with
        the confirmation gate redesigned for a mixed voice/keyboard
        flow (destructive actions stay keyboard-confirmed -- see
        below).

State machine:
    IDLE (wake-word listening)
      -> LISTENING (recording after wake, or during a follow-up window)
      -> TRANSCRIBING
      -> THINKING (LLM call)
      -> SPEAKING
      -> back to LISTENING (follow-up window) or IDLE

After Jarvis finishes speaking, it opens a short "follow-up window"
(FOLLOWUP_LISTEN_SECONDS) where you can keep talking without repeating
the wake word -- standard voice-assistant UX. The first attempt in
that window that doesn't produce usable speech (silence timeout, a
mic hiccup, a failed transcription) ends the window and falls back to
full wake-word listening, rather than retrying indefinitely -- a
follow-up window that never gives up is indistinguishable from just
never using a wake word at all, and risks looping on background noise.
Set FOLLOWUP_ENABLED=false to disable it and always require the wake
word.

Deliberately synchronous, not asyncio. Phase 4's tools.py flagged
asyncio as needed "once wake-word + voice + tools + memory all need to
run concurrently." Milestone 1 has no concurrency requirement -- one
thing happens at a time, no barge-in/interruption support -- so a
plain state machine is the honest, simple version for now.

Milestone 1 deliberately did NOT seed history with a manual
system-role message, because jarvis.call_llm() (Phase 1,
Ollama/Anthropic) builds its own system prompt internally, and the
Anthropic backend's Messages API rejects a "system" role inside
`messages` outright.

Milestone 2 changes this. The THINKING step (see think() below) no
longer calls Phase 1's call_llm() at all -- it now runs Phase 4's
tool-calling loop (agent.run_turn(), reused as-is) against DeepSeek,
matching the project's own documented policy ("DeepSeek for
tool-calling phases -- Ollama was rejected in Phase 4 for unreliable
tool-calling," see CLAUDE.md). DeepSeek's OpenAI-compatible API
expects a system message inside `messages` -- Phase 4's own agent.py
seeds one at the start of every conversation -- so history now starts
with one too (see TOOL_SYSTEM_PROMPT and run_orchestrator() below).
Phase 1's call_llm() and jarvis.get_system_prompt() are no longer used
by this file; trim_history() still is (see think()'s docstring for the
one known sharp edge that comes with reusing it here unchanged).

Milestone 3 folds in Phase 5's persistent memory (see
_get_memory_store(), the "remember" handling in run_turn(), and
memory_chat.py's own is_remember_command()/strip_remember_prefix()/
build_augmented_message(), all reused as-is). Storage stays explicit
("remember <fact>", said out loud) rather than automatic, matching
Phase 5's own design decision -- see its README for why. Retrieval is
automatic on every other turn: the most relevant stored memories are
folded into the user message as a labeled preamble before it's handed
to think(), same augmentation Phase 5's memory_chat.py already used,
just now living inside a DeepSeek/tool-calling message instead of a
Phase-1 one -- format-agnostic since it only ever touches a
{"role": "user", "content": ...} message's content.

Deliberately NOT wired into voice: Phase 5's "list"/"forget" commands.
Both are fine as typed text (short ids are easy to read and type) but
awkward over voice -- reading a whole memory store aloud is a poor
experience, and nobody reliably says an 8-character hex id out loud in
a way Whisper transcribes back correctly. `memory_chat.py` (Phase 5's
own text CLI) remains the way to inspect or prune the store directly
if a bad memory needs removing -- same "out-of-band from voice"
precedent this milestone now leans on for a much higher-stakes case,
below.

Milestone 4 folds in Phase 6's OS-control tools (list/read files,
launch apps, read the screen via OCR, move the mouse, type, and --
gated -- delete a file, overwrite one, click, or press a hotkey). This
is the first milestone where THINKING is no longer a straight reuse of
one earlier phase's tool-calling loop: Phase 4's tools.py and Phase
6's tools.py are BOTH modules literally named `tools.py`, so a bare
`import tools` with both phase paths on sys.path would resolve to
whichever was inserted last, silently shadowing the other -- flagged
as a known future problem back in milestone 2's own note, now real.
Resolved by loading each by explicit file path via `importlib` (see
_load_module_from_path()) instead of relying on sys.path + bare
imports at all, and merging their TOOL_SCHEMAS/TOOL_FUNCTIONS into one
combined registry. Neither phase's own tool-calling loop (Phase 4's
agent.run_turn(), Phase 6's os_agent.run_turn()) can be reused as-is
anymore either, since each is hardwired to its own single tools
module and neither has a hook for the other's confirmation gate --
_run_tool_calling_turn() below is this file's own loop, adapted from
both, reused as closely as the merge allows rather than as a black-box
import. This is expected, not a departure: Phase 7 is exactly where
standalone-phase discipline was always meant to end.

The confirmation gate itself IS reused directly, unchanged, from Phase
6's tools.py (confirm_pending_action/cancel_pending_action, the
in-memory pending-action registry, the TTL, the focus-shift/
out-of-bounds checks for click/hotkey) -- none of that logic is
voice-specific and none of it needed to change. What's new here is
_prompt_for_confirmation(): the human-facing side of the gate, which
still requires the exact same typed password at the exact same real
terminal (see CONFIRM_PASSWORD below) as Phase 6's CLI did, PLUS a
short spoken heads-up first, so a person doesn't sit listening to
silence wondering whether Jarvis is stuck -- matching this project's
own already-decided design (see Milestone 1's original "Confirmation
design" note): voice can propose a destructive action, but confirming
it always goes through a separate, non-voice channel, specifically
because voice introduces new ways a confirmation could be corrupted
(misheard "yes", background noise, Jarvis's own TTS output being
picked back up by the mic) that a keyboard doesn't. This matters more
now than it did in milestone 1's note, not less: Phase 5's own README
already flagged that a memory shaped like a prompt-injection payload
can hijack the model's behavior for a session, and Phase 6's own
README names the exact realistic consequence once memory and
OS-control tools share a loop -- a poisoned "memory" as a path to an
*unintended* destructive tool call. The gate does not trust the model
to police this; it lives in code, entirely outside the model's reach,
same as it always did in Phase 6.
"""

import getpass
import importlib.util
import json
import logging
import os
import platform
import re
import sys
import time
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("orchestrator")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE_ONE_PATH = PROJECT_ROOT / "phaseOne"
PHASE_TWO_PATH = PROJECT_ROOT / "phaseTwo"
PHASE_THREE_PATH = PROJECT_ROOT / "phaseThree"
PHASE_FOUR_PATH = PROJECT_ROOT / "phaseFour"
PHASE_FIVE_PATH = PROJECT_ROOT / "phaseFive"
PHASE_SIX_PATH = PROJECT_ROOT / "phaseSix"

# Deliberately NOT adding PHASE_SIX_PATH here. phaseFour and phaseSix
# each have their own tools.py -- a bare `import tools` with both paths
# on sys.path would resolve to whichever was inserted last, silently
# shadowing the other. Both are loaded by explicit file path via
# importlib instead (see _load_module_from_path() below), which needs
# no sys.path entry at all. phaseSix's tools.py has no other local
# imports of its own to resolve, so this is the only phaseSix file
# this module ever needs.
for _phase_path in (PHASE_ONE_PATH, PHASE_TWO_PATH, PHASE_THREE_PATH, PHASE_FOUR_PATH, PHASE_FIVE_PATH):
    sys.path.insert(0, str(_phase_path))

# Phase 1's trim_history() is deliberately no longer used. It is a bare
# messages[-40:] with no awareness of tool_calls/tool-role pairing, and
# Phase 8 raised the tool-hop ceiling to the point where one turn can
# append more messages than that whole budget -- see
# _trim_history_preserving_system() and _drop_orphaned_tool_messages()
# below, which replace it. Phase 1's own default is untouched: its text
# CLI has no tool calls and no reason to carry 200 messages.

try:
    from voice_io import (
        AudioRecordingError,
        SpeechError,
        TranscriptionError,
        record_until_silence,
        speak_text,
        transcribe_audio,
    )
except ImportError:
    logger.error(f"Could not import Phase 2's voice_io.py from {PHASE_TWO_PATH}.")
    raise

try:
    from wake_word import MODEL_NAME as WAKE_MODEL_NAME
    from wake_word import THRESHOLD as WAKE_THRESHOLD
    from wake_word import build_model
except ImportError:
    logger.error(f"Could not import Phase 3's wake_word.py from {PHASE_THREE_PATH}.")
    raise

try:
    from llm_client import build_client, call_llm_with_tools
except ImportError:
    logger.error(f"Could not import Phase 4's llm_client.py from {PHASE_FOUR_PATH}.")
    raise

# Confirmed via manual testing that this actually breaks without it:
# memory.py calls load_dotenv() at its own module level, and
# python-dotenv resolves that relative to memory.py's OWN file
# location (phaseFive/), not this process's cwd or orchestrator.py's
# location -- so importing memory.py here loads phaseFive/.env, which
# sets CHROMA_DB_PATH=./chroma_store (a RELATIVE path). That path then
# resolves against THIS process's cwd (phaseSeven, per this project's
# own "cd phaseSeven; python orchestrator.py" convention), silently
# creating/opening an empty, wrong store there instead of the real,
# populated phaseFive/chroma_store -- the exact same bug class Phase
# 5's own README already documented once for a plain wrong-cwd case,
# just resurfacing here through a cross-phase import instead. Setting
# this explicitly and first wins over memory.py's own load_dotenv()
# call (python-dotenv never overrides an already-set env var by
# default) -- os.environ.setdefault so an explicit CHROMA_DB_PATH in
# this phase's own .env (if anyone ever wants a separate store) still
# takes priority.
os.environ.setdefault("CHROMA_DB_PATH", str(PHASE_FIVE_PATH / "chroma_store"))

try:
    from memory import MemoryError, MemoryStore
    from memory_chat import (
        RECALL_MAX_RESULTS,
        build_augmented_message,
        is_remember_command,
        strip_remember_prefix,
    )
except ImportError:
    logger.error(f"Could not import Phase 5's memory.py/memory_chat.py from {PHASE_FIVE_PATH}.")
    raise


def _load_module_from_path(module_name: str, file_path: Path):
    """Loads a .py file as a module under an explicit name without ever
    touching sys.path -- see the module docstring's "Milestone 4 folds
    in" paragraph for why this exists (phaseFour's and phaseSix's
    tools.py share a filename and would otherwise silently shadow one
    another). Registers the module in sys.modules under module_name so
    it behaves like a normally-imported module (picklability, repr,
    etc. all work), but under a name that can never collide with the
    real "tools" the bare-import mechanism would resolve on its own."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


try:
    _phase_four_tools = _load_module_from_path("phase_four_tools", PHASE_FOUR_PATH / "tools.py")
    _phase_six_tools = _load_module_from_path("phase_six_tools", PHASE_SIX_PATH / "tools.py")
except Exception:
    logger.error(
        f"Could not load Phase 4's tools.py ({PHASE_FOUR_PATH}) or "
        f"Phase 6's tools.py ({PHASE_SIX_PATH})."
    )
    raise

# Merged tool surface: Phase 4's (get_weather, set_timer, search_web)
# plus Phase 6's (list_directory, read_file, read_screen, write_file,
# open_application, delete_file, move_mouse, type_text, click,
# press_hotkey, get_active_window). No name collisions between the two
# sets. confirm_pending_action/cancel_pending_action are Phase 6's own
# safety boundary -- deliberately NOT merged into TOOL_SCHEMAS/
# TOOL_FUNCTIONS below, exactly as Phase 6's own tools.py keeps them
# out of its LLM-facing surface. The model must never have a
# tool-calling path to either.
TOOL_SCHEMAS = _phase_four_tools.TOOL_SCHEMAS + _phase_six_tools.TOOL_SCHEMAS
_TOOL_FUNCTIONS = {**_phase_four_tools.TOOL_FUNCTIONS, **_phase_six_tools.TOOL_FUNCTIONS}
confirm_pending_action = _phase_six_tools.confirm_pending_action
cancel_pending_action = _phase_six_tools.cancel_pending_action


def execute_tool(name: str, arguments: dict) -> dict:
    if name not in _TOOL_FUNCTIONS:
        raise KeyError(f"Unknown tool: {name}")
    return _TOOL_FUNCTIONS[name](**arguments)


def register_tools(schemas: list, functions: dict) -> None:
    """
    Extension seam for Phase 8. Adds tools to the merged registry, or
    replaces existing ones by name.

    Added because Phase 8 needs to fix two tools that are wrong
    (Phase 4's stub get_weather, and the missing clock the model was
    substituting other tools for) without editing Phase 4's own
    tools.py, which is a finished standalone mini-project whose CLI
    still has to work as it always did. Registering over the top keeps
    that boundary intact and the change reversible.

    A name present in `schemas` replaces any existing schema with the
    same name, so an override swaps out the old tool's description too
    -- not just its implementation. Both halves are checked against each
    other and a mismatch raises immediately: a schema with no function
    behind it is a tool the model can call and nothing can answer, and
    a function with no schema is dead code the model can never reach.
    Neither fails loudly on its own at runtime, so they fail loudly
    here instead.
    """
    global TOOL_SCHEMAS

    schema_names = {s["function"]["name"] for s in schemas}
    if schema_names != set(functions):
        raise ValueError(
            "register_tools: schemas and functions must cover the same names. "
            f"Only in schemas: {sorted(schema_names - set(functions))}; "
            f"only in functions: {sorted(set(functions) - schema_names)}."
        )

    TOOL_SCHEMAS = [
        s for s in TOOL_SCHEMAS if s["function"]["name"] not in schema_names
    ] + list(schemas)
    _TOOL_FUNCTIONS.update(functions)


from wake_gate import WakeWordGate

# Exact-match per fragment (see _contains_exit_phrase below) -- same
# not-a-substring-of-anything reasoning Phase 2's voice_chat.py used,
# just applied per sentence instead of to the whole utterance. "good
# bye" is included alongside "goodbye" because real Whisper output for
# that word varies -- confirmed by manual testing (see README).
EXIT_PHRASES = {"quit", "exit", "stop", "goodbye", "good bye", "shut down", "shutdown"}

# Subset of EXIT_PHRASES also recognized when trailing other words in
# the same utterance, e.g. "okay, goodbye" or "alright then, goodbye"
# -- found via manual testing (see _contains_exit_phrase's docstring).
# Deliberately NOT all of EXIT_PHRASES: "stop" and "exit" are far too
# common as the last word of an ordinary, non-exit request ("make it
# stop", "where's the exit"), and "shut down"/"shutdown" show up in
# ordinary questions too ("when does the store shut down"). "quit" is
# judged the same way, conservatively, for consistency -- none of
# these were reported as broken, only "goodbye" was.
TRAILING_SAFE_EXIT_PHRASES = {"goodbye", "good bye"}

WAKE_COOLDOWN_SECONDS = float(os.getenv("WAKE_COOLDOWN_SECONDS", "2.0"))

# How many audio callbacks to discard right after (re)opening the
# wake-word stream, before scoring resumes -- see WakeWordGate.start()'s
# docstring. Found via manual testing: restarting the gate between turns
# could register a spurious wake almost immediately. Default (4, ~320ms
# at the gate's default 80ms/frame chunk size) is short enough that a
# real "hey jarvis" said right after a turn ends still works.
WAKE_STARTUP_SKIP_FRAMES = int(os.getenv("WAKE_STARTUP_SKIP_FRAMES", "4"))

# Real pause before reopening the wake-word stream (skipped on the very
# first start() of the process) -- see WakeWordGate.start()'s
# docstring. Confirmed via diagnostic logging that startup_skip_frames
# alone wasn't enough: a spurious wake fired with a genuinely high
# confidence score (0.99) at the very first scored frame after restart,
# pointing at stale audio bleeding in from whichever stream (the turn
# recorder's, or this gate's own) had just closed on the same device.
WAKE_RESTART_SETTLE_SECONDS = float(os.getenv("WAKE_RESTART_SETTLE_SECONDS", "0.3"))

# Whether to play a short chime immediately after wake-word detection,
# before recording starts. Added after manual testing showed that with
# no acknowledgment at all, a person has no way to know Jarvis heard
# the wake word and ends up repeating themselves mid-recording -- which
# both makes the transcript noisier and, combined with the old
# whole-utterance exit check, was the actual root cause of "goodbye"
# not working (see _contains_exit_phrase's docstring).
WAKE_CHIME_ENABLED = os.getenv("WAKE_CHIME_ENABLED", "true").lower() not in ("false", "0", "no")

# How long the follow-up window (see module docstring) waits for you to
# say something before giving up. Deliberately shorter than
# voice_io.py's own MAX_RECORD_SECONDS default (30s) -- there's no
# reason to hold the mic open anywhere near that long just to notice
# nobody's saying anything after a reply.
FOLLOWUP_LISTEN_SECONDS = float(os.getenv("FOLLOWUP_LISTEN_SECONDS", "6.0"))
FOLLOWUP_ENABLED = os.getenv("FOLLOWUP_ENABLED", "true").lower() not in ("false", "0", "no")

# Raised from Phase 4's MAX_TOOL_HOPS=4 to Phase 6's MAX_TOOL_HOPS=8,
# reused here for the same reason Phase 6 raised it: repeated manual
# testing showed multi-step UI tasks (open an app, check focus, type,
# press a hotkey, check again...) routinely need more than 4 round-trips
# even when everything's working correctly. Still bounded, just less
# trigger-happy about giving up on legitimate sequences.
MAX_TOOL_HOPS = int(os.getenv("MAX_TOOL_HOPS", "40"))

# Wall-clock ceiling for a single turn's tool-calling loop, checked
# between hops.
#
# Raised from 8 to 40 during Phase 8's polish pass, because genuinely
# multi-step requests ("do these four things") were dying at the old
# ceiling with "gave up after too many tool calls in a row" -- the cap
# was a get-it-working-first compromise, not a considered limit.
#
# A bigger number alone would have been the wrong fix, though. The cap
# exists because a model that misreads a tool result can loop, and at 40
# hops a confused loop is 40 API calls and several minutes of Jarvis
# talking to itself before anything stops it. The time budget is what
# actually bounds that: real work finishes well inside it, while a loop
# hits the wall quickly regardless of how many hops are left. Two
# different limits for two different failure modes -- a long legitimate
# task, and a short pathological one.
TURN_TIME_BUDGET_SECONDS = float(os.getenv("TURN_TIME_BUDGET_SECONDS", "180"))

# How many conversational messages to keep, on top of the system
# message. Raised from Phase 1's 40 during the same pass: at 40, a
# single 40-hop turn could append more messages than the entire history
# was allowed to hold, so a complex turn would erase the conversation
# that prompted it. Phase 1's own default is untouched -- its text CLI
# has no tool calls and no reason to carry 200 messages.
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "200"))

# How long one LLM request may take before it is abandoned, and how
# many times it may be retried -- see _get_deepseek_client() for why
# the SDK's own defaults (read=600s, 2 retries) are wrong here.
#
# 60s is not arbitrary: a real hop carrying 39 tool schemas and a long
# history can legitimately take 20-30s, so anything much tighter would
# start cancelling work that was about to succeed. One retry rather
# than two because the worst case is what matters for a voice UX --
# 60s x 2 attempts is ~120s of silence, against ~1800s today.
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "1"))

# How many turns in a row may fail before the orchestrator gives up
# instead of looping. See run_orchestrator's per-turn recovery: a
# transient fault (mic glitch, network blip) should cost one turn, but
# a permanent one (no microphone at all) must not spin forever.
MAX_CONSECUTIVE_TURN_FAILURES = int(os.getenv("MAX_CONSECUTIVE_TURN_FAILURES", "5"))

# Pause after a failed turn, so a fast, repeating fault cannot become a
# hot loop that fills the log and pins a core.
TURN_FAILURE_BACKOFF_SECONDS = float(os.getenv("TURN_FAILURE_BACKOFF_SECONDS", "2.0"))

# Long text fields (screen OCR output, file contents) get truncated to
# this many characters before being written to the log -- reused as-is
# from Phase 6's os_agent.py. A single tool call should never dump an
# entire screen capture or file's contents into a log file that might
# persist, get shared for debugging, or end up in version control. The
# model still gets the full, untruncated result; this only affects
# what lands in the log.
LOG_TEXT_TRUNCATE_CHARS = int(os.getenv("LOG_TEXT_TRUNCATE_CHARS", "300"))

# Optional. If set, confirming a destructive action (delete_file,
# write_file(overwrite=True), click, press_hotkey) requires typing this
# exact password at the terminal (hidden via getpass) instead of just
# "yes" -- reused as-is from Phase 6's guardrail design, see the module
# docstring's "Milestone 4 folds in" paragraph for why this matters
# even more now that voice can be the thing proposing the action.
CONFIRM_PASSWORD = os.getenv("CONFIRM_PASSWORD")

if not CONFIRM_PASSWORD:
    logger.warning(
        "CONFIRM_PASSWORD is not set -- destructive actions will fall back to a "
        "plain yes/no prompt at the terminal. Set CONFIRM_PASSWORD in .env for a "
        "stronger gate, especially with voice input now able to propose "
        "destructive actions (see README's Confirmation design note)."
    )

# Seeded once at the start of history (see run_orchestrator) so
# DeepSeek's tool-calling loop has a system message to work with --
# see the module docstring's "Milestone 2 changes this" note. Adapted
# from Phase 4's agent.py system prompt (same intent, same wording
# where it still applies) with one voice-specific addition: replies
# get spoken aloud by TTS, so markdown/lists read badly and are asked
# for by name to be avoided. Milestone 4 folds in Phase 6's os_agent.py
# system prompt guidance on the OS-control tools -- read_screen being
# OCR (not vision), preferring press_hotkey over guessing at click
# coordinates, and target_window_contains -- largely verbatim, since
# none of that guidance is voice-specific.
TOOL_SYSTEM_PROMPT = (
    "You are Jarvis, a helpful personal assistant with access to tools, "
    "including OS-control tools: listing directories, reading files, "
    "writing/creating files, reading the screen (OCR), launching "
    "applications, deleting files, moving the mouse, typing text, "
    "clicking, and pressing keyboard shortcuts. Deleting a file, "
    "overwriting an existing file, clicking, and pressing a hotkey all "
    "require separate human confirmation that you cannot see or "
    "influence -- treat 'deleted'/'overwritten'/'clicked'/'pressed' or "
    "'cancelled' outcomes as normal, not errors. Moving the mouse and "
    "typing text happen immediately without confirmation. "
    "read_screen is OCR text extraction, NOT vision -- it can read "
    "words on screen but cannot locate icons, buttons, or other "
    "non-text UI elements, and does not return coordinates for "
    "anything it reads. Do not try to find a button by guessing screen "
    "regions and reading them repeatedly; for standard window actions "
    "(closing, minimizing, switching apps), prefer a keyboard shortcut "
    "via press_hotkey (e.g. alt+f4 to close the active window) over "
    "hunting for a click target you cannot actually see. If something "
    "you expect isn't visible with a normal read_screen call, it may be "
    "on a second monitor -- try read_screen with all_screens=true "
    "before assuming it isn't open. Before sending a hotkey or click "
    "meant for a specific app (especially one just launched via "
    "open_application), call get_active_window first to confirm that "
    "app actually has focus. click and press_hotkey both REQUIRE a "
    "target_window_contains argument naming the window you believe is "
    "focused (e.g. 'Notepad') -- this is checked against reality "
    "before proposing and again right before firing. Use tools when "
    "they'd give a better answer than guessing. Your replies are "
    "spoken aloud through text-to-speech, so keep them concise and "
    "conversational -- no markdown, bullet points, or numbered lists."
)


class OrchestratorState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"


class TurnOutcome(Enum):
    """
    What run_turn() accomplished, so run_conversation() (below) knows
    whether to keep the follow-up window open, end the whole session,
    or fall back to full wake-word listening.
    """
    CONTINUE = "continue"     # a normal turn completed -- keep the follow-up window open
    EXIT = "exit"              # the user said an exit phrase -- end the whole session
    NO_SPEECH = "no_speech"    # nothing usable was captured (recording error, failed
                               # transcription, or -- most commonly -- the follow-up
                               # window simply elapsed with no one talking). This is
                               # an expected, everyday outcome, not an error.


_turn_context_provider = None  # see register_turn_context()


def register_turn_context(provider) -> None:
    """
    Sixth extension seam for Phase 8. Supplies a short context line
    prepended to each user message, rather than baked into the system
    prompt.

    Exists for a measured reason. DeepSeek caches request prefixes, and
    a cache hit costs roughly a tenth of a miss. Measured on this
    project's own registry: a warm request is 5,376 cached tokens
    against 39 uncharged-at-full-price ones -- but ANY change to the
    system prompt drops that to 1,152 cached and 4,263 at full price,
    because the tool schemas serialize after the system message and a
    changed prefix invalidates everything behind it.

    The system prompt used to state the current time to the minute, so
    every new conversation changed it, and every new conversation
    therefore re-paid for all 35 tool schemas. Moving that one line
    after the tools -- into the user turn -- keeps the whole system
    prompt and the entire tool registry as a stable prefix. Measured
    afterwards: 5,376 cached / ~28 full-price per conversation, holding
    across day changes.

    Content the model must treat as authoritative should say so, since
    a user-turn line carries less weight than a system-prompt one.

    NOTE: this is the sixth seam bolted onto this module, which is the
    layering telling us something. Phase 8 should own this loop outright
    in the eventual application rewrite rather than reaching into Phase
    7 six different ways.
    """
    global _turn_context_provider
    _turn_context_provider = provider


def _turn_context() -> str:
    """Current per-turn context, or "" if none is registered. Never
    raises -- a missing context line must not cost a turn."""
    if _turn_context_provider is None:
        return ""
    try:
        return _turn_context_provider() or ""
    except Exception as e:
        logger.warning(f"Turn-context provider failed, continuing without it: {e}")
        return ""


_reply_filter = None  # see register_reply_filter()


def register_reply_filter(reply_filter) -> None:
    """
    Fifth extension seam for Phase 8. Normalises the assistant's reply
    once, before it is logged OR spoken.

    Added to fix a real mismatch: Phase 8 rewrites text on its way to
    the speech engine (moving "sir" to the end of its sentence, stripping
    markdown that would otherwise be read out as punctuation), but that
    happened at render time, downstream of this file. The terminal
    therefore printed what the model wrote while the speakers said
    something slightly different -- which is exactly the kind of small
    inconsistency that wastes an hour when something later goes wrong
    and the log is the only record of what happened.

    Applying it here means the log line, the audio, and anything else
    downstream all agree on one string.
    """
    global _reply_filter
    _reply_filter = reply_filter


def _filter_reply(reply: str) -> str:
    """Applies the registered reply filter, if any. Never raises -- a
    cosmetic text tidy must not be able to lose a reply that the model
    already spent a round-trip producing."""
    if _reply_filter is None or not reply:
        return reply
    try:
        return _reply_filter(reply)
    except Exception as e:
        logger.warning(f"Reply filter failed, speaking the original text: {e}")
        return reply


_speaker = None  # see register_speaker()


def register_speaker(speak_function) -> None:
    """
    Fourth extension seam for Phase 8 (see register_tools,
    run_orchestrator's system_prompt_provider, and
    register_confirmation_handler for the others). Replaces what
    speak_safe() uses to actually produce audio.

    Phase 8's barge-in milestone needs to speak a reply in interruptible
    pieces while a microphone listens for an interrupt word -- which is
    a different way of speaking, not a different thing being said. Every
    caller of speak_safe() still just calls speak_safe(); only the
    mechanism underneath changes.

    The error handling in speak_safe() stays wrapped around whatever is
    registered, deliberately: the reason it exists -- a dead audio
    driver must never kill the loop -- applies at least as much to a
    speaker with a mic and a transcription model behind it as it did to
    a plain pyttsx3 call.

    Left as None here, so Phase 7 on its own speaks exactly as it always
    has.
    """
    global _speaker
    _speaker = speak_function


def speak_safe(text: str) -> None:
    """Speaks text; falls back to printing if TTS itself is broken, so
    a dead audio driver never kills the whole loop. Same pattern as
    Phase 2's voice_chat.py."""
    try:
        (_speaker or speak_text)(text)
    except SpeechError as e:
        logger.warning(f"TTS failed, printing instead: {e}")
        print(f"Jarvis (text fallback): {text}")


def _contains_exit_phrase(text: str) -> bool:
    """
    Checks each sentence-like fragment of the transcript against
    EXIT_PHRASES independently, instead of requiring the whole
    utterance to match exactly.

    Found via manual testing: a nervous "Goodbye. Good bye. Good
    bye." -- said because the person had no feedback that Jarvis had
    heard the wake word, so they repeated themselves -- never matched
    as a single string under the original whole-utterance check
    (inherited from Phase 2's voice_chat.py), even though every
    fragment in it clearly is a genuine exit phrase. It fell through
    to the LLM instead, which happened to reply "Goodbye." on its own
    -- looking superficially like it worked, while the session kept
    running.

    Splitting on sentence-ending punctuation and checking each
    fragment exactly keeps the original safety property intact -- a
    full, unrelated sentence like "did I quit too early" still won't
    match, since neither the whole thing nor any of its
    punctuation-delimited fragments is itself in EXIT_PHRASES -- while
    tolerating repetition and Whisper noise around a genuine exit
    phrase.

    Second manual-testing finding: that exact-fragment check still
    missed "goodbye" when it trailed another phrase in the same breath
    with no sentence-ending punctuation between them -- "okay, goodbye"
    or "alright then, goodbye" transcribe as ONE fragment ("okay,
    goodbye"), which doesn't equal "goodbye" exactly, so it fell
    through to the LLM the same way the original bug did. Fixed by also
    checking whether a fragment *ends* with one of
    TRAILING_SAFE_EXIT_PHRASES (preceded by whitespace, a comma, or the
    start of the fragment) -- deliberately restricted to that smaller
    set rather than all of EXIT_PHRASES; see its own comment for why
    "stop"/"exit"/"shutdown" stay exact-match-only. Known, accepted
    tradeoff: a fragment like "not goodbye" would now also match --
    judged rare enough in practice not to be worth guarding against
    (e.g. with negation detection) given the much more common case this
    fixes.
    """
    fragments = [f.strip() for f in re.split(r"[.!?]+", text.lower()) if f.strip()]

    for fragment in fragments:
        if fragment in EXIT_PHRASES:
            return True
        if any(
            re.search(rf"(?:^|[\s,]){re.escape(phrase)}$", fragment)
            for phrase in TRAILING_SAFE_EXIT_PHRASES
        ):
            return True

    return False


def _play_wake_chime() -> None:
    """
    Short beep played immediately after a wake-word detection, before
    recording starts. See WAKE_CHIME_ENABLED above for why this
    exists. Windows-only for now (winsound is stdlib on Windows,
    matching this project's Windows-first hardware); on any other
    platform, or if it's disabled, this is a clean no-op rather than
    failing the turn over a missing beep.
    """
    if not WAKE_CHIME_ENABLED:
        return
    if platform.system() != "Windows":
        return
    try:
        import winsound

        winsound.Beep(880, 120)
    except Exception as e:
        logger.debug(f"Wake chime failed (non-fatal): {e}")


_deepseek_client = None  # built lazily, once -- see _get_deepseek_client()


def _get_deepseek_client():
    """Builds Phase 4's DeepSeek client on first use and caches it for
    the life of the process -- same one-time-cost shape as build_model()
    for the wake-word model. Kept lazy (rather than built at import
    time) so importing this module -- e.g. for tests -- never requires
    DEEPSEEK_API_KEY to be set; only actually running a real turn
    does.

    Phase 4's build_client() is used unmodified and then given a
    timeout here, rather than edited -- that phase is finished, its own
    CLI works, and a voice loop's needs are not a text CLI's. The
    default it would otherwise inherit from the OpenAI SDK is
    read=600s with 2 retries: up to ~30 minutes wedged on one hung
    call, during which Jarvis is silent, unresponsive and
    indistinguishable from crashed. TURN_TIME_BUDGET_SECONDS cannot
    save this, because it is only checked BETWEEN hops -- a single
    stuck request never reaches the check."""
    global _deepseek_client
    if _deepseek_client is None:
        client = build_client()
        try:
            client = client.with_options(
                timeout=LLM_TIMEOUT_SECONDS, max_retries=LLM_MAX_RETRIES
            )
        except Exception as e:  # pragma: no cover -- SDK shape changed
            logger.warning(
                f"Could not set an LLM timeout ({e}); falling back to the "
                "SDK default. A hung call may block for several minutes."
            )
        _deepseek_client = client
    return _deepseek_client


def _prompt_for_confirmation(message: str) -> bool:
    """
    Human-facing side of Phase 6's confirmation gate, adapted for
    voice. Speaks a short heads-up first (so a person doesn't sit
    listening to silence wondering whether Jarvis is stuck), then
    blocks on a REAL terminal prompt -- never voice/the mic -- exactly
    like Phase 6's own os_agent.py did. This is the load-bearing part
    of the module docstring's "Confirmation design": voice can propose
    a destructive action, but confirming it always goes through a
    separate, non-voice channel.

    If CONFIRM_PASSWORD is configured, this requires typing that exact
    password (hidden, via getpass) rather than the word "yes" -- a
    plain "yes" is trivially easy to satisfy by accident (a
    mis-transcription, background noise, Jarvis's own TTS reply being
    picked back up by the mic), which matters far more here than it did
    for Phase 6's typed-only CLI. Falls back to a plain yes/no prompt
    if unconfigured (see the startup warning above).

    Fails closed (returns False, i.e. cancels) on any input error --
    getpass can raise in some terminals/environments without a real
    tty, and a destructive action should never proceed just because
    the confirmation prompt itself broke.
    """
    speak_safe(_spoken_reason_for(message))
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


# Confirmation registries, keyed by the token prefix that identifies
# them. Phase 6's is the default and its tokens are bare hex, so it
# isn't listed here -- see _resolve_confirmation() below.
_confirmation_handlers = {}


def register_confirmation_handler(token_prefix: str, confirm, cancel) -> None:
    """
    Third extension seam for Phase 8 (see register_tools and
    run_orchestrator's system_prompt_provider for the first two).

    Phase 8 adds destructive tools of its own -- killing a process,
    closing a window, sleeping the machine -- and those cannot go
    through Phase 6's confirm_pending_action(), which dispatches on a
    hardcoded if/elif chain over its own four action types and has no
    idea what any of them mean. It therefore brings its own pending-
    action registry, and this is how a confirmation gets routed back to
    the registry that actually issued the token.

    Routing is by token prefix rather than by trying each registry in
    turn, deliberately: "ask every registry until one doesn't say
    invalid token" would mean the gate's behavior depended on registry
    ordering and on error-string matching, which is not a property to
    want anywhere near this particular boundary.

    What does NOT change, for any registry registered here: confirm and
    cancel stay out of TOOL_SCHEMAS/TOOL_FUNCTIONS entirely. The model
    gets no tool-calling path to either, ever.
    """
    if not token_prefix:
        raise ValueError("register_confirmation_handler: token_prefix must be non-empty.")
    _confirmation_handlers[token_prefix] = (confirm, cancel)


def _resolve_confirmation(token: str):
    """Returns the (confirm, cancel) pair owning `token`. Falls back to
    Phase 6's, whose tokens are bare hex with no prefix and which is
    still the only registry present when Phase 7 runs on its own."""
    for prefix, handlers in _confirmation_handlers.items():
        if token.startswith(prefix):
            return handlers
    return confirm_pending_action, cancel_pending_action


# What each kind of gated action actually costs, said in general terms.
# Matched against the proposal message in order, first hit wins.
#
# Deliberately names the KIND of action and its consequence, never the
# specific target: no absolute paths, no click coordinates, no
# keystrokes. That boundary is Phase 7's own (see the README's
# "Confirmation design" note) -- reading a file path aloud is neither a
# good security property nor a good listening experience, and the full
# detail is printed to the terminal a foot away, which is where the
# human is confirming anyway.
#
# What changed is that saying only "that needs your confirmation" told
# the person nothing about what they were being asked to approve, which
# is a poor way to ask someone to make a security decision.
_CONFIRMATION_REASONS = (
    ("delete", "That would permanently delete a file, and it cannot be undone."),
    ("overwrite", "That would replace the contents of a file that already exists, "
                  "and the old contents cannot be recovered."),
    ("replac", "That would replace a file that already exists, and the old contents "
               "cannot be recovered."),
    ("close the window", "That would close a window, and any unsaved work in it could be lost."),
    ("terminate", "That would force a program to close, and any unsaved work in it "
                  "would be lost."),
    ("shutdown", "That would shut the machine down and close everything, including me."),
    ("restart", "That would restart the machine and close everything, including me."),
    ("sleep", "That would put the machine to sleep."),
    ("click", "That would send a mouse click, which can do whatever the thing under it does."),
    ("hotkey", "That would send a keyboard shortcut, which can act on whatever has focus."),
    ("press", "That would send a keystroke, which can act on whatever has focus."),
)

_GENERIC_CONFIRMATION_REASON = (
    "That is a destructive action that cannot be undone."
)


def _spoken_reason_for(message: str) -> str:
    """Turns a proposal message into something worth saying out loud.

    Falls back to a generic warning if nothing matches, so a tool whose
    wording changes degrades to today's behavior rather than going
    silent or, worse, reading a path aloud by accident.
    """
    lowered = (message or "").lower()
    for keyword, reason in _CONFIRMATION_REASONS:
        if keyword in lowered:
            return f"{reason} I'll need your confirmation at the terminal."
    return f"{_GENERIC_CONFIRMATION_REASON} I'll need your confirmation at the terminal."


def _handle_tool_result(result: dict) -> dict:
    """
    Reused as-is from Phase 6's os_agent.py (same function, same
    behavior): intercepts any tool result that proposes a destructive
    action and forces a real, out-of-band human confirmation before
    anything happens. The model never sees a confirmation token, a
    password, or any path to call confirm_pending_action itself; it
    only ever sees the final outcome (deleted/overwritten/clicked/
    pressed/cancelled/error).

    Also prints the real outcome directly to the terminal, same as
    Phase 6 -- don't rely on the model to relay it accurately in its
    reply. Confirmed necessary by Phase 6's own manual testing: the
    model's prose description of a tool result is not always
    trustworthy even when the result itself is correct (see Phase 6's
    README).
    """
    if result.get("status") != "confirmation_required":
        return result

    approved = _prompt_for_confirmation(result.get("message", "A destructive action was proposed."))
    token = result["token"]
    confirm, cancel = _resolve_confirmation(token)
    outcome = confirm(token) if approved else cancel(token)

    if "error" in outcome:
        print(f"[NOTE] {outcome['error']}")
    else:
        print(f"[NOTE] {outcome.get('status', 'done')}: {outcome.get('path', '')}")

    return outcome


def _summarize_for_log(result: dict) -> dict:
    """Reused as-is from Phase 6's os_agent.py: returns a copy of
    `result` safe to write to the log, with any long text fields
    (read_screen's OCR output, read_file's file contents) truncated to
    LOG_TEXT_TRUNCATE_CHARS. The model still gets the full,
    untruncated result; this only affects what lands in the log."""
    summary = dict(result)
    for key in ("text", "content"):
        value = summary.get(key)
        if isinstance(value, str) and len(value) > LOG_TEXT_TRUNCATE_CHARS:
            summary[key] = (
                value[:LOG_TEXT_TRUNCATE_CHARS]
                + f"... [{len(value)} chars total, truncated for log]"
            )
    return summary


def _log_usage(response, hop: int) -> None:
    """Logs what each LLM call actually cost, in tokens.

    Nothing tracked spend before this, which made "is the tool registry
    expensive?" unanswerable without guessing -- and the guess was
    wrong. The 35 tool schemas look like a large per-call cost, and
    measurement showed they are almost entirely served from DeepSeek's
    prefix cache at roughly a tenth the price. The same measurement then
    found the real problem: a timestamp in the system prompt was
    invalidating that cache once per conversation (see
    register_turn_context()).

    Cache hits are broken out separately because the hit/miss split is
    the number that actually matters -- total prompt tokens barely moved
    across that fix, while the full-price share fell by about 98%.

    DEBUG rather than INFO: this is one line per hop, and a 40-hop turn
    would otherwise bury the conversation in accounting.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    hit = getattr(usage, "prompt_cache_hit_tokens", None)
    miss = getattr(usage, "prompt_cache_miss_tokens", None)
    cache = f", cached {hit}/full-price {miss}" if hit is not None else ""
    logger.debug(
        f"hop {hop + 1}: {usage.prompt_tokens} prompt + "
        f"{usage.completion_tokens} completion{cache}"
    )


def _run_tool_calling_turn(client, messages: list) -> str:
    """
    This file's own tool-calling loop -- see the module docstring's
    "Milestone 4 folds in" paragraph for why neither Phase 4's
    agent.run_turn() nor Phase 6's os_agent.run_turn() could be reused
    as a black box anymore. Adapted from both: Phase 4's loop shape
    (call the LLM with tool schemas, execute any tool calls, feed
    results back, repeat until a plain-text answer or MAX_TOOL_HOPS),
    plus Phase 6's confirmation-result interception
    (_handle_tool_result) and log-truncation (_summarize_for_log) on
    top -- now running against the MERGED tool registry
    (TOOL_SCHEMAS/execute_tool above) instead of either phase's tools
    alone.

    Same mutate-in-place contract as think() itself: appends the
    assistant's final reply (and any intermediate assistant/tool-role
    messages) to `messages` directly and returns just the final text.
    """
    started = time.monotonic()

    for hop in range(MAX_TOOL_HOPS):
        elapsed = time.monotonic() - started
        if elapsed > TURN_TIME_BUDGET_SECONDS:
            logger.warning(
                f"Turn exceeded its {TURN_TIME_BUDGET_SECONDS:.0f}s budget after "
                f"{hop} tool calls -- stopping."
            )
            return (
                "I ran out of time on that one before I could finish. "
                "Try asking for a smaller piece of it."
            )

        response = call_llm_with_tools(client, messages, TOOL_SCHEMAS)
        _log_usage(response, hop)
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

            logger.info(f"Tool result for {name}: {_summarize_for_log(result)}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )

    logger.warning(f"Hit MAX_TOOL_HOPS ({MAX_TOOL_HOPS}) without a final answer.")
    return (
        "That turned into more steps than I could finish in one go. "
        "Try breaking it into a couple of smaller requests."
    )


def _drop_orphaned_tool_messages(messages: list) -> list:
    """Drops tool-role messages left stranded at the front of a trimmed
    window.

    DeepSeek's API requires every tool-role message to follow the
    assistant message whose tool_calls it answers. Slicing a history to
    its last N entries can land the cut between an assistant's tool_calls
    and their results, leaving responses with nothing to respond to --
    which the API rejects outright, failing the turn.

    Dropping leading tool messages is the whole fix, and it is enough:
    a cut can only ever orphan results, never requests. If the window
    happens to start with an assistant message that has tool_calls, its
    results are by definition still behind it and the pairing holds.

    Known and deliberately unfixed since Phase 7's milestone 2, on the
    grounds that it was "unlikely in practice at MAX_HISTORY_MESSAGES=40
    for a hobby session". Raising the tool-hop ceiling is exactly what
    makes it likely -- one complex turn can now append more messages
    than the entire old history cap -- so it stopped being theoretical
    and had to be fixed before the ceiling could move.
    """
    start = 0
    while start < len(messages) and messages[start].get("role") == "tool":
        start += 1
    return messages[start:]


def _trim_history_preserving_system(history: list) -> list:
    """
    trim_history() with the system message pinned in place.

    Found during Phase 8's accuracy work, not by the test suite.
    jarvis.trim_history() is a bare `messages[-MAX_HISTORY_MESSAGES:]`
    -- it has no idea the first message is special, so the moment a
    session passes 40 messages it drops TOOL_SYSTEM_PROMPT off the
    front along with the oldest turn. Every call after that point goes
    to DeepSeek with no system message at all: no "your replies are
    spoken aloud, avoid markdown", no read_screen-is-OCR-not-vision
    guidance, no confirmation-gate framing, none of the tool guidance
    the prompt exists to carry. The session keeps working, just
    progressively worse, in a way that looks like the model randomly
    got dumber rather than like a bug.

    Trimming the conversation and keeping the system message is what
    the original call always meant; the message count is now
    MAX_HISTORY_MESSAGES conversational messages plus the system one,
    rather than 40 total with the system message steadily closer to
    falling off the edge.

    The pre-existing sharp edge trim_history already had is untouched
    and still real: it has no concept of tool_calls/tool-role message
    pairing, so a trim landing between an assistant message requesting
    a tool call and the tool-role message answering it still produces
    a history DeepSeek rejects (see think()'s docstring).
    """
    if not history:
        return history

    system = history[0] if history[0].get("role") == "system" else None
    rest = history[1:] if system else list(history)

    if len(rest) > MAX_HISTORY_MESSAGES:
        rest = _drop_orphaned_tool_messages(rest[-MAX_HISTORY_MESSAGES:])

    return ([system] + rest) if system else rest


def think(history: list) -> str:
    """
    THINKING step: runs this file's own merged tool-calling loop
    (_run_tool_calling_turn(), see its docstring for why it's no
    longer a direct reuse of either Phase 4's or Phase 6's own loop)
    against DeepSeek over `history`.

    Mutates `history` in place -- it appends the assistant's final
    reply itself (and, along the way, any intermediate assistant/
    tool-role messages a tool call produced) -- so callers must NOT
    also append the reply themselves afterward. Returns just the final
    text reply, ready to speak.

    Known sharp edge, carried over deliberately rather than solved
    here (small-and-ugly-first): jarvis.trim_history() (still reused
    from Phase 1 for its truncate-oldest-messages behavior) has no
    awareness of tool_calls/tool-role message pairing. If a trim ever
    lands between an assistant message that requested a tool call and
    the tool-role message answering it, DeepSeek's API will reject the
    resulting history. Unlikely in practice at MAX_HISTORY_MESSAGES=40
    for a hobby session, but worth knowing if a "history got rejected"
    error ever shows up in the wild.
    """
    client = _get_deepseek_client()
    return _run_tool_calling_turn(client, history)


_memory_store = None       # built lazily, once -- see _get_memory_store()
_memory_store_failed = False  # sticky: don't retry every turn once it's failed


def _get_memory_store():
    """Builds Phase 5's MemoryStore on first use and caches it for the
    life of the process, same one-time-cost shape as _get_deepseek_client().

    Unlike the wake model and the DeepSeek client, a failure here is
    NOT allowed to crash the whole orchestrator: memory is an
    enhancement (recall context, "remember" storage) on top of a voice
    assistant that's still fully useful without it, whereas a missing
    wake model or LLM credential means nothing can happen at all.
    Returns None if construction fails (e.g. chromadb isn't installed,
    or the store directory can't be opened) -- logged once, then
    remembered (via _memory_store_failed) so every subsequent turn
    doesn't retry a failure that isn't going to spontaneously fix
    itself. Callers treat None as "proceed without memory this turn,"
    not as an error to propagate."""
    global _memory_store, _memory_store_failed
    if _memory_store_failed:
        return None
    if _memory_store is None:
        try:
            _memory_store = MemoryStore()
        except MemoryError as e:
            logger.error(f"Memory store unavailable, continuing without persistent memory: {e}")
            _memory_store_failed = True
            return None
    return _memory_store


def _handle_remember_command(user_text: str, history: list, _set_state) -> tuple:
    """
    Handles a spoken "remember <fact>" command: stores the fact (if the
    memory store is available) and speaks a confirmation, without ever
    reaching think()/the LLM for this turn -- same behavior as Phase
    5's memory_chat.py, which also treats "remember" as a store-and-continue
    command rather than a normal chat message.

    Deliberately does NOT append anything to `history` for this turn,
    matching memory_chat.py again -- a "remember" command isn't part of
    the visible conversation, just a side effect on the memory store.
    """
    _set_state(OrchestratorState.SPEAKING)
    store = _get_memory_store()
    if store is None:
        speak_safe("Sorry, I can't save memories right now.")
        return history, TurnOutcome.CONTINUE

    fact = strip_remember_prefix(user_text)
    try:
        store.add_memory(fact)
        logger.info(f"Stored memory: {fact!r}")
        speak_safe("Got it, I'll remember that.")
    except MemoryError as e:
        logger.warning(f"Couldn't save memory: {e}")
        speak_safe("Sorry, I couldn't save that.")

    return history, TurnOutcome.CONTINUE


def _augment_with_memories(user_text: str) -> str:
    """
    Retrieves the most relevant stored memories for user_text and folds
    them into it as a labeled preamble, via Phase 5's own
    build_augmented_message() (reused as-is) -- same RAG shape as
    memory_chat.py, just returning a string here instead of building a
    whole message dict, so run_turn() can slot it into whichever
    message format the caller (think()) expects.

    Falls back to returning user_text unchanged -- no augmentation --
    if the memory store isn't available or the search itself fails;
    either way this never raises, so a memory hiccup can't take down
    an otherwise-normal turn. Always logs what was retrieved (even
    "none"), matching memory_chat.py's own discipline here: a silent
    retrieval failure was a real bug there once (see Phase 5's
    README), and staying visible is what caught it.
    """
    store = _get_memory_store()
    retrieved = []
    if store is not None:
        try:
            retrieved = store.search_memories(user_text, n_results=RECALL_MAX_RESULTS)
        except MemoryError as e:
            logger.warning(f"Memory search failed, continuing without it: {e}")
            retrieved = []

    if retrieved:
        logger.info(
            "Retrieved memories: "
            + "; ".join(f"{r['text']!r} (d={r['distance']:.3f})" for r in retrieved)
        )
    else:
        logger.info("Retrieved memories: none.")

    return build_augmented_message(user_text, retrieved)


def run_turn(history: list, on_state_change=None, pre_speech_timeout: float = None) -> tuple:
    """
    Runs one full voice turn: record -> transcribe -> (exit check) ->
    (remember-command check) -> (memory retrieval + augmentation) ->
    LLM -> speak.

    pre_speech_timeout, if given, overrides voice_io.py's default
    MAX_RECORD_SECONDS for how long to wait for speech to START in
    this recording -- used to give the follow-up window (see
    run_conversation below) a shorter "did anyone start talking"
    timeout than the first, wake-word-triggered recording in a
    conversation. Once speech actually starts, voice_io.py falls back
    to its own normal, longer max_seconds cap regardless -- this only
    ever shortens the *waiting* phase, never how long you're allowed to
    keep talking once you've started (see voice_io.record_until_silence's
    docstring). Left as None for the first turn, which keeps the
    original single, longer window throughout.

    on_state_change, if given, is called with each OrchestratorState as
    the turn progresses -- purely for observability/logging by the
    caller (per the roadmap's "instrument it well" advice for this
    phase); run_turn's own control flow doesn't depend on it.

    Returns (updated_history, TurnOutcome). See TurnOutcome's
    docstring for what each value means to the caller. Recoverable
    failures (a bad recording, a failed transcription, nothing said
    before the timeout, an LLM hiccup) never raise -- they come back
    as NO_SPEECH or CONTINUE so a single bad turn can't crash the
    whole session.
    """
    def _set_state(state: OrchestratorState) -> None:
        if on_state_change:
            on_state_change(state)

    _set_state(OrchestratorState.LISTENING)
    try:
        if pre_speech_timeout is not None:
            wav_path = record_until_silence(pre_speech_timeout=pre_speech_timeout)
        else:
            wav_path = record_until_silence()
    except AudioRecordingError as e:
        # Deliberately not distinguishing "no speech before timeout"
        # from a genuine mic failure here -- voice_io.py raises the
        # same AudioRecordingError for both, and either way the right
        # response at this level is the same: report nothing was
        # captured and let the caller decide what "nothing captured"
        # means (end of a follow-up window vs. a real problem worth a
        # louder log elsewhere).
        logger.info(f"Nothing captured: {e}")
        return history, TurnOutcome.NO_SPEECH

    _set_state(OrchestratorState.TRANSCRIBING)
    try:
        user_text = transcribe_audio(wav_path)
    except TranscriptionError as e:
        logger.warning(f"Couldn't transcribe that: {e}.")
        return history, TurnOutcome.NO_SPEECH
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass

    logger.info(f"You said: {user_text}")

    if _contains_exit_phrase(user_text):
        logger.info("Exit command recognized. Shutting down.")
        _set_state(OrchestratorState.SPEAKING)
        speak_safe("Goodbye.")
        return history, TurnOutcome.EXIT

    if is_remember_command(user_text):
        return _handle_remember_command(user_text, history, _set_state)

    # Turn context (the clock, today) goes in the user turn rather than
    # the system prompt, so the system prompt and the tool schemas stay
    # a stable, cacheable prefix -- see register_turn_context().
    content = _augment_with_memories(user_text)
    context = _turn_context()
    if context:
        content = f"{context}\n\n{content}"

    history.append({"role": "user", "content": content})
    history = _trim_history_preserving_system(history)

    _set_state(OrchestratorState.THINKING)
    try:
        reply = think(history)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        _set_state(OrchestratorState.SPEAKING)
        speak_safe("Sorry, I had trouble reaching my brain just now.")
        return history, TurnOutcome.CONTINUE

    # Normalise once, so the log line and the audio say the same thing
    # -- see register_reply_filter(). Deliberately before the log, not
    # after it.
    reply = _filter_reply(reply)

    logger.info(f"Jarvis: {reply}")
    # think() (Phase 4's agent.run_turn) already appended the
    # assistant's reply -- and any intermediate tool-call/tool-role
    # messages -- to history in place. Appending it again here would
    # duplicate it.

    _set_state(OrchestratorState.SPEAKING)
    speak_safe(reply)
    return history, TurnOutcome.CONTINUE


def run_conversation(history: list, on_state_change=None) -> tuple:
    """
    Runs turns back-to-back without re-requiring the wake word between
    them -- the follow-up window described in the module docstring.

    The first turn uses run_turn's normal, longer window
    (pre_speech_timeout=None). Each turn after a CONTINUE outcome uses
    the shorter FOLLOWUP_LISTEN_SECONDS pre-speech timeout instead, so
    a person doesn't have to repeat the wake word for a quick
    back-and-forth, but an idle mic also doesn't sit open indefinitely
    once the conversation is actually over. Once someone actually
    starts talking within that window, the normal, longer recording
    length takes back over (see run_turn/voice_io.record_until_silence)
    -- this window only shortens how long Jarvis waits for you to
    START a follow-up, never how long you're allowed to keep talking
    once you have.

    Stops and returns as soon as either:
      - the user says an exit phrase (TurnOutcome.EXIT), or
      - a turn comes back as TurnOutcome.NO_SPEECH -- which, for any
        turn after the first, means the follow-up window elapsed with
        nobody talking. That's treated as "the conversation is over,"
        not a failure -- the caller falls back to full wake-word
        listening.

    If FOLLOWUP_ENABLED is False, this always runs exactly one turn
    and returns -- equivalent to every conversation requiring a fresh
    wake word, matching the original milestone-1 behavior.

    Returns (updated_history, exited) -- exited is True only when the
    user actually asked to end the session; False means "back to
    wake-word listening," which is the normal way most conversations
    end.
    """
    pre_speech_timeout = None  # first turn: run_turn's own default window
    while True:
        if pre_speech_timeout is not None:
            logger.info(f"Listening for a follow-up ({FOLLOWUP_LISTEN_SECONDS:.0f}s, no wake word needed)...")

        history, outcome = run_turn(
            history, on_state_change=on_state_change, pre_speech_timeout=pre_speech_timeout
        )

        if outcome == TurnOutcome.EXIT:
            return history, True
        if outcome == TurnOutcome.NO_SPEECH:
            return history, False
        if not FOLLOWUP_ENABLED:
            return history, False

        pre_speech_timeout = FOLLOWUP_LISTEN_SECONDS


def run_orchestrator(system_prompt_provider=None) -> None:
    """
    system_prompt_provider, if given, is called for a fresh system
    prompt string at the start of every wake-triggered conversation,
    replacing history[0] in place.

    Second extension seam for Phase 8 (see register_tools above for the
    first). A system prompt that never changes cannot state anything
    that goes stale, which rules out telling the model what today's
    date is -- and a model that doesn't know the date can't tell
    whether its own training data is current, can't form a
    date-anchored search query, and can't do arithmetic on "how long
    until". Rebuilding it per conversation rather than once per process
    matters specifically because this thing is meant to sit running
    indefinitely; a date baked in at import time is wrong by the next
    morning.

    Left as None here, which keeps Phase 7's own behavior byte-for-byte
    what it was: one fixed TOOL_SYSTEM_PROMPT for the life of the
    process.
    """
    model = build_model()
    gate = WakeWordGate(
        model=model,
        model_name=WAKE_MODEL_NAME,
        threshold=WAKE_THRESHOLD,
        cooldown_seconds=WAKE_COOLDOWN_SECONDS,
        startup_skip_frames=WAKE_STARTUP_SKIP_FRAMES,
        restart_settle_seconds=WAKE_RESTART_SETTLE_SECONDS,
    )
    history: list = [{"role": "system", "content": TOOL_SYSTEM_PROMPT}]

    def log_state(state: OrchestratorState) -> None:
        logger.info(f"State -> {state.value}")

    logger.info(f"Jarvis Phase 7 (milestone 4) ready. Say '{WAKE_MODEL_NAME}' to start a turn.")

    # Turns that failed in a row. Reset by any turn that completes, so
    # this counts a *sustained* fault, not a total across the session.
    consecutive_failures = 0

    try:
        while True:
            # Per-iteration recovery. run_turn() already guards its own
            # three recoverable steps (recording, transcription, the LLM
            # call), but nothing guarded the loop AROUND it: the wake
            # gate, the chime, the system-prompt provider and speaking.
            # A single unhandled exception there -- a microphone
            # unplugged mid-session, an audio driver hiccup, edge-tts
            # losing the network mid-sentence -- killed the process
            # outright, leaving a dead terminal and no assistant. For
            # something meant to sit running for hours that was the
            # single biggest reliability gap, and it is exactly where
            # the "no usable input device" failure surfaces.
            #
            # KeyboardInterrupt is deliberately NOT caught here: it
            # inherits from BaseException, not Exception, so Ctrl-C
            # still falls straight through to the handler below and
            # stops Jarvis immediately rather than being absorbed as a
            # failed turn and retried.
            try:
                log_state(OrchestratorState.IDLE)
                gate.start()
                gate.wait_for_wake()
                gate.stop()  # release the mic before the turn recorder opens it
                _play_wake_chime()

                if system_prompt_provider is not None and history and history[0].get("role") == "system":
                    history[0] = {"role": "system", "content": system_prompt_provider()}

                history, exited = run_conversation(history, on_state_change=log_state)
                consecutive_failures = 0
                if exited:
                    break
            except Exception as e:
                consecutive_failures += 1
                # Full traceback for the FIRST failure of a run, one
                # line for repeats of the same fault. A permanently
                # broken microphone otherwise prints five identical
                # 15-line tracebacks on the way out, which buries the
                # one line that actually says what to do about it.
                if consecutive_failures == 1:
                    logger.exception(
                        f"Turn failed (1/{MAX_CONSECUTIVE_TURN_FAILURES}). Recovering."
                    )
                else:
                    logger.error(
                        f"Turn failed ({consecutive_failures}/"
                        f"{MAX_CONSECUTIVE_TURN_FAILURES}), same fault: "
                        f"{type(e).__name__}: {e}"
                    )
                # Release the mic before retrying. A half-open stream is
                # the most likely reason the next start() would fail
                # too, which would turn one fault into a run of them.
                try:
                    gate.stop()
                except Exception:
                    logger.debug("Could not stop the wake gate while recovering.", exc_info=True)

                if consecutive_failures >= MAX_CONSECUTIVE_TURN_FAILURES:
                    logger.error(
                        f"{consecutive_failures} turns failed in a row. Stopping rather "
                        "than looping -- this looks permanent, not transient. Check the "
                        "traceback above; if it is about the microphone, run "
                        "phaseEight/audio_device.py to see what devices are actually usable."
                    )
                    break
                time.sleep(TURN_FAILURE_BACKOFF_SECONDS)
    except KeyboardInterrupt:
        logger.info("Orchestrator stopped.")
    finally:
        gate.stop()


if __name__ == "__main__":
    run_orchestrator()