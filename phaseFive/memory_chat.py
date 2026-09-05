"""
memory_chat.py — Phase 5: text chat with persistent semantic memory (RAG).

Loop: read input -> if it's an explicit "remember" command, store it and
loop again -> otherwise, retrieve the most relevant past memories for
this message, fold them into the prompt sent to the LLM, print the
reply, repeat.

Standalone-phase discipline: reuses Phase 1's jarvis.call_llm() the same
deliberate way Phase 2's voice_chat.py does — proven infrastructure,
not something this phase needs to relearn. Does NOT import
phaseTwo/voice_io.py, phaseThree/wake_word.py, or
phaseFour/agent.py's tool-calling loop — text-only, no tools, on
purpose. Voice + memory + tools all combining is Phase 7's job.

The `from jarvis import ...` is done lazily, inside run_chat_loop()
rather than at module import time. voice_chat.py (Phase 2) imports it
at the top instead — the difference here is deliberate: it keeps the
pure-logic helpers below (build_augmented_message, is_remember_command,
strip_remember_prefix) importable and unit-testable without phaseOne
actually being present on disk, matching voice_io.py's "lazy import
for testability" pattern rather than voice_chat.py's.

Adjust PHASE_ONE_PATH if your folder layout differs from
PROJECTS/JARVIS/phaseOne, phaseFive/.
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("memory_chat")

from memory import MemoryError, MemoryStore

PHASE_ONE_PATH = Path(__file__).resolve().parent.parent / "phaseOne"

# Order matters: "remember that " must be checked before the shorter
# "remember " prefix, or strip_remember_prefix would match "remember "
# first and leave a dangling "that " on the front of the stored fact.
REMEMBER_PREFIXES = ("remember that ", "remember:", "remember ")

# Added after manual testing showed a real gap: there was no way to
# remove a bad memory (e.g. an accidental prompt-injection-shaped one —
# see README) short of writing a throwaway Python snippet. "list" shows
# short ids; "forget <id-prefix>" deletes by (possibly partial) id.
LIST_COMMANDS = {"list", "list memories", "memories"}
FORGET_PREFIX = "forget "

# NOTE on distance filtering: an earlier version of this file hard-coded
# a cosine-distance cutoff of 1.1 to filter out irrelevant memories.
# That was wrong on two counts: (1) Chroma's *default* distance metric
# is squared L2, not cosine, so the scale was never what the comment
# claimed; (2) even fixed, a single global cutoff punishes compound
# queries — "what's my name and what am I building?" sits further from
# either single-topic memory than a single-topic query does, so a real
# match can legitimately score worse than a tight cutoff allows. Result:
# a correctly-stored, correctly-matchable memory got silently dropped
# and never reached the LLM at all — no error, just wrong answers.
#
# Fix: don't hard-filter by distance. Trust the prompt instruction
# ("only use them if they're actually relevant") to let the model
# disregard genuinely unrelated results instead. Set RECALL_MAX_RESULTS
# in .env if you want to cap how many candidates get considered — that's
# a cheap, safe knob; a distance cutoff on an unverified metric is not.
RECALL_MAX_RESULTS = int(os.getenv("RECALL_MAX_RESULTS", 5))


def strip_remember_prefix(text: str) -> str:
    lowered = text.lower()
    for prefix in REMEMBER_PREFIXES:
        if lowered.startswith(prefix):
            return text[len(prefix):].strip()
    return text.strip()


def is_remember_command(text: str) -> bool:
    return text.lower().startswith(REMEMBER_PREFIXES)


def format_memory_line(mem: dict) -> str:
    """Short 8-char id prefix + text, for 'list' output and forget
    confirmations. Full UUIDs are unpleasant to type by hand, so
    find_memory_by_prefix() below accepts this shortened form."""
    return f"[{mem['id'][:8]}] {mem['text']}"


def find_memory_by_prefix(store: MemoryStore, id_prefix: str) -> list:
    """Returns every stored memory whose id starts with id_prefix
    (case-insensitive). Used by the 'forget' command so the user can
    type a short prefix from 'list' output instead of a full UUID.
    Returns a list (not a single match) so the caller can detect and
    reject ambiguous prefixes rather than silently deleting the wrong
    one."""
    id_prefix = id_prefix.strip().lower()
    if not id_prefix:
        return []
    return [m for m in store.get_all_memories() if m["id"].lower().startswith(id_prefix)]


def build_augmented_message(user_input: str, retrieved: list) -> str:
    """
    Phase 1's call_llm() always builds its own system prompt internally
    (see jarvis.get_system_prompt()) — it ignores any system-role
    message passed to it, and adding an extra system-role entry to the
    messages list would actually break the Anthropic backend (its
    Messages API only accepts user/assistant roles in `messages`;
    system is a separate top-level param). So retrieved memories can't
    be injected as a system message without reaching into Phase 1's
    internals, which would cross the standalone-phase boundary.

    Workaround: fold memories into the user message itself, as a
    labeled preamble. This augmented text is what gets sent to the LLM,
    but `history` (see run_chat_loop) stores the original, un-augmented
    user_input instead — so the memory block doesn't get repeated back
    to the model on every subsequent turn.
    """
    if not retrieved:
        return user_input

    memory_lines = "\n".join(f"- {m['text']}" for m in retrieved)
    return (
        "[The following facts were previously shared with you and may "
        "be relevant — only use them if they're actually relevant to "
        "this message]\n"
        f"{memory_lines}\n\n"
        f"[User's message]\n{user_input}"
    )


def run_chat_loop() -> None:
    sys.path.insert(0, str(PHASE_ONE_PATH))
    try:
        from jarvis import call_llm, trim_history
    except ImportError:
        logger.error(
            f"Could not import Phase 1's jarvis.py from {PHASE_ONE_PATH}. "
            "Update PHASE_ONE_PATH at the top of memory_chat.py to point "
            "at your phaseOne folder."
        )
        raise

    try:
        store = MemoryStore()
    except MemoryError as e:
        logger.error(f"Could not start memory store: {e}")
        return

    logger.info(
        f"Memory chat ready ({store.count()} memories stored). "
        "Say 'remember <fact>' to store something, 'list' to see everything "
        "stored, 'forget <id>' to delete one. Type 'exit' or 'quit' to leave."
    )

    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nJarvis: Shutting down. Goodbye.")
            break

        if not user_input:
            continue
        lowered_input = user_input.lower()

        if lowered_input in ("exit", "quit"):
            print("Jarvis: Goodbye.")
            break

        if lowered_input in LIST_COMMANDS:
            memories = store.get_all_memories()
            memories.sort(key=lambda m: m["metadata"].get("created_at", ""))
            if not memories:
                print("Jarvis: No memories stored yet.\n")
            else:
                print("Jarvis: Stored memories:")
                for m in memories:
                    print(f"  {format_memory_line(m)}")
                print()
            continue

        if lowered_input.startswith(FORGET_PREFIX):
            id_prefix = user_input[len(FORGET_PREFIX):].strip()
            if not id_prefix:
                print("Jarvis: Usage: forget <id-prefix>  (see 'list' for ids)\n")
                continue

            matches = find_memory_by_prefix(store, id_prefix)
            if not matches:
                print(f"Jarvis: No memory found starting with '{id_prefix}'.\n")
            elif len(matches) > 1:
                print(f"Jarvis: '{id_prefix}' matches more than one memory, be more specific:")
                for m in matches:
                    print(f"  {format_memory_line(m)}")
                print()
            else:
                try:
                    store.delete_memory(matches[0]["id"])
                    print(f"Jarvis: Forgot: {matches[0]['text']!r}\n")
                except MemoryError as e:
                    print(f"Jarvis: Couldn't delete that — {e}\n")
            continue

        if is_remember_command(user_input):
            fact = strip_remember_prefix(user_input)
            try:
                store.add_memory(fact)
                print("Jarvis: Got it, I'll remember that.\n")
            except MemoryError as e:
                print(f"Jarvis: Couldn't save that — {e}\n")
            continue

        try:
            retrieved = store.search_memories(user_input, n_results=RECALL_MAX_RESULTS)
        except MemoryError as e:
            logger.warning(f"Memory search failed, continuing without it: {e}")
            retrieved = []

        # Always log what came back, even if nothing did — this is the
        # only visibility you have into whether retrieval is working at
        # all, and distance-based filtering already burned us once (see
        # the RECALL_MAX_RESULTS comment above) by failing silently.
        if retrieved:
            logger.info(
                "Retrieved memories: "
                + "; ".join(f"{r['text']!r} (d={r['distance']:.3f})" for r in retrieved)
            )
        else:
            logger.info("Retrieved memories: none.")

        augmented = build_augmented_message(user_input, retrieved)
        messages = history + [{"role": "user", "content": augmented}]
        messages = trim_history(messages)

        try:
            reply = call_llm(messages)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            print("Jarvis: Sorry, I had trouble reaching my brain just now.\n")
            continue

        print(f"Jarvis: {reply}\n")

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})
        history = trim_history(history)


if __name__ == "__main__":
    run_chat_loop()