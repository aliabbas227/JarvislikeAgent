# Jarvis — Phase 7: Wake Word → Voice → LLM (+ Tools + Memory + OS Control)

Phase 7 is where standalone-phase discipline deliberately ends — its
whole job is wiring the earlier phases together. Rather than one
big-bang integration, it's being built in staged sub-milestones, the
same "small and ugly first" way every earlier phase was:

1. Milestone 1: wake word (Phase 3) + voice I/O (Phase 2) + LLM
   (Phase 1). No tools, no memory, no OS control.
2. Milestone 2: fold in Phase 4's tool-calling loop.
3. Milestone 3: fold in Phase 5's memory retrieval/storage.
4. **Milestone 4 (this one):** fold in Phase 6's OS-control tools, with
   the confirmation gate redesigned for voice (see below).

## What this milestone does

Say the wake word → it records your next utterance → transcribes it →
checks for an exit phrase, then a spoken "remember <fact>" command,
then — for every other turn — retrieves the most relevant stored
memories and folds them into the message before running it through
this file's own tool-calling loop (`_run_tool_calling_turn()` — see
"Milestone 4 design notes" below for why it's no longer a direct reuse
of any one earlier phase's loop) against DeepSeek → speaks the reply →
then listens directly for a follow-up for a few seconds before falling
back to wake-word listening (see "Follow-up window" below). Say an
exit phrase ("quit", "exit", "stop", "goodbye", "shut down",
"shutdown") to end the session.

The LLM can now call any tool from **both** Phase 4 and Phase 6 mid-turn:
weather (stub data), `set_timer` (real, background thread), `search_web`
(real, via Tavily), listing/reading files, reading the screen via OCR,
launching applications, and — gated behind a real, out-of-band human
confirmation, never something voice or the model can satisfy on its
own — deleting a file, overwriting one, clicking, or pressing a
keyboard shortcut. It also recalls facts you've told it to remember in
past conversations, even across separate process runs.

## Files

- **`wake_gate.py`** — `WakeWordGate`: wraps a wake-word model with
  `start()`/`stop()`/`wait_for_wake()`. New this phase. Phase 3's own
  `listen_loop()` blocks forever and can't release the microphone,
  which the orchestrator needs (only one `InputStream` should own the
  mic at a time — the wake-word listener or the turn recorder, never
  both). The model itself (`build_model()`) is still Phase 3's, reused
  as-is; only the run/pause control is new.
- **`orchestrator.py`** — the state machine and main loop:
  `IDLE → LISTENING → TRANSCRIBING → THINKING → SPEAKING → IDLE`.
  Imports Phase 1's `jarvis.py` (now just for `trim_history`), Phase
  2's `voice_io.py`, Phase 3's `wake_word.py`, Phase 4's
  `llm_client.py`, Phase 5's `memory.py`/`memory_chat.py`, and — new
  this milestone — Phase 4's *and* Phase 6's `tools.py` (both, loaded
  side by side via `importlib` rather than `sys.path`, since they
  share a filename — see `_load_module_from_path()`), all directly, the
  same pattern Phases 2 and 5 already used to reach back into Phase 1.
  The THINKING step lives in its own `think()` function, which lazily
  builds and caches a DeepSeek client and hands off to
  `_run_tool_calling_turn()` — new this milestone, and no longer a
  reuse of any single earlier phase's loop (see "Milestone 4 design
  notes" below). `_get_memory_store()` (lazy-cache; a failure degrades
  gracefully to "no memory this session" instead of crashing — see its
  own docstring for why that's different from the wake model and
  DeepSeek client), `_handle_remember_command()`, and
  `_augment_with_memories()` are unchanged from milestone 3. New this
  milestone: `_prompt_for_confirmation()`/`_handle_tool_result()` (the
  voice-adapted confirmation gate) and `_summarize_for_log()`.
- **`test_wake_gate.py`**, **`test_orchestrator.py`** — fully mocked
  pytest suites. No real microphone, model weights, or `openwakeword`
  package installation required to run them.

## Milestone 1 design notes

**Synchronous, not `asyncio`, on purpose.** Phase 4's `tools.py`
flagged async as needed "once wake-word + voice + tools + memory all
need to run concurrently." Milestone 1 has no concurrency requirement
— strict turn-taking, no barge-in/interruption support — so a plain
state machine is the honest, simple version. Worth revisiting once a
later milestone needs something running *while* still listening (e.g.
a spoken "stop" during a long TTS reply).

**No manual system-role message in `history`.** Phase 2's
`voice_chat.py` seeds `history` with `{"role": "system", ...}` up
front — but `jarvis.call_llm()` already builds its own system prompt
internally on every call, and Anthropic's Messages API rejects a
`system` role inside `messages` outright. This only works in
`voice_chat.py` today because the Ollama backend tolerates the
redundant message; it would break under `LLM_BACKEND=anthropic`. This
orchestrator instead follows Phase 5's `memory_chat.py`, which already
documents and avoids this — `history` starts empty, no system entry.
**Worth fixing in `voice_chat.py` itself at some point**, independent
of Phase 7. **Superseded in milestone 2** — see below; `history` now
starts with a system message again, for a different backend that
actually needs one.

**Confirmation design, decided in milestone 1, built in milestone 4:**
destructive OS-control actions (`delete_file`, `write_file(overwrite=True)`,
`click`, `press_hotkey`) keep requiring a **typed keyboard password**,
even though the rest of the pipeline is voice-driven. Voice can
*propose* an action; confirming it still goes through a separate,
non-voice channel. This isn't a fallback or a compromise — it's
Phase 6's own "out-of-band from the model" principle extended to also
be out-of-band from voice, specifically because voice introduces new
ways for a confirmation to be corrupted that a keyboard doesn't
(misheard "yes", background noise, Jarvis's own TTS output being
picked back up by the mic) — the same category of problem bug #9 in
the Phase 6 handoff already found once, from typing alone. See
"Milestone 4 design notes" below for how this was actually implemented.

**Wake-word re-arming after a turn.** Nothing yet prevents Jarvis's
own spoken reply from being picked up by the wake-word listener once
it restarts — the gate is stopped for the whole turn (recording *and*
speaking) and only restarts once `run_turn()` returns, so this isn't
an issue yet in milestone 1. Flagging it because it stops being free
the moment a later milestone wants the wake-word listener running
*during* TTS playback (e.g. for barge-in).

## Milestone 2 design notes

**Switching brains: Phase 1's `call_llm()` → Phase 4's tool-calling
loop, on DeepSeek.** The THINKING step no longer calls
`jarvis.call_llm()` at all. It now runs Phase 4's `agent.run_turn()`
(reused as-is) against DeepSeek — matching the project's own
documented policy in the root `CLAUDE.md` ("DeepSeek for tool-calling
phases — Ollama was rejected in Phase 4 for unreliable tool-calling").
This wasn't a free choice to make independently at the orchestrator
level; it's this project's existing, already-validated answer to "what
backend do tool-calling turns use," just applied here for the first
time outside Phase 4 itself. `LLM_BACKEND`/`OLLAMA_*`/`ANTHROPIC_API_KEY`
in `.env` are consequently unused by `orchestrator.py` as of this
milestone — left in `.env.example` in case a future milestone
reintroduces a non-tool-calling path.

**`history` seeds a system message again, for a different reason than
milestone 1 removed one.** DeepSeek's OpenAI-compatible API expects a
system message in `messages`, and Phase 4's own `agent.py` seeds one
at the start of every conversation. `TOOL_SYSTEM_PROMPT` in
`orchestrator.py` is adapted from that same prompt, with one addition:
since replies get spoken aloud, it explicitly asks the model to avoid
markdown/lists, which read badly through TTS. This is safe now in a
way it wasn't in milestone 1 specifically *because* the backend
changed too — DeepSeek's API (unlike Anthropic's) doesn't reject a
`system` role inside `messages`.

**`think()` mutates `history` in place — callers must not re-append
the reply.** `agent.run_turn()` appends the assistant's final reply
(and any intermediate tool-call/tool-role messages) to `messages`
itself as it goes, rather than returning a fresh list. `run_turn()`
(the orchestrator's own, name collision aside) used to append the
reply manually after calling `call_llm()`, since that function was a
pure computation with no side effect on `history`. That line had to
come out — leaving it in would have silently duplicated every
assistant reply in history.

> **Fixed in Phase 8's polish pass.** See "Limits raised" below — the
> sharp edge described next stopped being theoretical the moment the
> tool-hop ceiling went up, and was fixed then.

**Known sharp edge, deliberately not solved yet: `trim_history()` isn't
tool-call-aware.** `jarvis.trim_history()` is still reused from Phase 1
as-is — it just keeps the last `MAX_HISTORY_MESSAGES` entries. It has
no concept of `tool_calls`/`tool`-role message pairing. If a trim ever
lands between an assistant message that requested a tool call and the
tool-role message answering it, DeepSeek's API will reject the
resulting history outright. Unlikely in practice at 40 messages for a
hobby session — flagging it rather than building a tool-aware trimmer
now, in keeping with "small and ugly first."

**Future collision, flagged now for milestone 4: two `tools.py`
modules.** Phase 4 (`get_weather`/`set_timer`/`search_web`) and Phase 6
(OS-control) each have their own `tools.py`. Milestone 4 will need
both `PHASE_FOUR_PATH` and `PHASE_SIX_PATH` on `sys.path` at once,
which means a bare `import tools` will resolve to whichever path was
inserted last, silently shadowing the other. Not a problem yet — only
Phase 4's path is on `sys.path` in this milestone — but milestone 4
will need to resolve it deliberately (e.g. import each module by its
explicit file path via `importlib`, rather than relying on `sys.path`
order).

**Tool set is exactly Phase 4's, unchanged.** This milestone is
integration only, per its own definition — no new tools were added or
modified. `get_weather` is still stub data, `set_timer` is a real
background `threading.Timer`, `search_web` is real (Tavily, requires
`TAVILY_API_KEY`).

## Milestone 3 design notes

**Storage stays explicit; retrieval is automatic.** Say "remember
<fact>" out loud to store something — nothing else gets written to
the memory store, matching Phase 5's own design decision (see its
README: auto-storing every message would flood the store with
greetings and throwaway questions and degrade retrieval quality for
everything else). Every OTHER turn automatically retrieves the most
relevant stored memories and folds them into the message before it
reaches `think()` — no voice command needed for recall, since that's
the whole point of persistent memory. Both behaviors, and the
`is_remember_command()`/`strip_remember_prefix()`/
`build_augmented_message()` functions implementing them, are reused
directly from `memory_chat.py`, unchanged.

**The augmentation pattern is format-agnostic, which is exactly why it
still works after milestone 2 changed the message format underneath
it.** Phase 5's `build_augmented_message()` folds retrieved memories
into the *user* message's `content` as a labeled preamble — Phase 5
built it that way specifically to avoid touching Phase 1's
internally-built system prompt. Milestone 2 already swapped the
THINKING step from Phase 1's plain user/assistant messages to
DeepSeek's OpenAI-style format (system role + tool_calls/tool
messages) — but `build_augmented_message()` never cared what format
surrounded it, only that it's building the `content` string of a
`{"role": "user", ...}` message. No changes needed to reuse it here.

**Memory failures degrade gracefully; wake-word/LLM failures don't.**
`_get_memory_store()` returns `None` on failure (e.g. chromadb isn't
installed) rather than crashing the orchestrator, unlike
`build_model()` or `_get_deepseek_client()`, which are allowed to take
the whole process down if they fail. Deliberate distinction: memory is
an enhancement on top of an assistant that's still fully useful
without it (recall just silently isn't available that session), while
a missing wake model or LLM credential means nothing can happen at
all. The failure is logged once and cached (`_memory_store_failed`) so
a real, unrecoverable problem doesn't get retried and logged every
single turn.

**Deliberately NOT wired into voice: `list`/`forget`.** Phase 5's
`memory_chat.py` also has commands to list every stored memory and
delete one by a short id prefix — useful for pruning a bad memory (see
Phase 5's own README for why that matters: a memory shaped like a
prompt injection hijacked its whole session once). Neither translates
well to voice: reading an entire memory store aloud is a poor
experience, and nobody reliably says an 8-character hex id out loud in
a way Whisper transcribes back correctly. `memory_chat.py` itself
remains the way to inspect or prune the shared store directly — same
"out-of-band from voice" reasoning Milestone 1 already used for Phase
6's future confirmation gate on destructive actions, just applied here
to store maintenance instead.

**The memory store is shared with Phase 5's own CLI, on purpose.**
`memory.py`'s `CHROMA_DB_PATH` default is anchored to its own file
location (`phaseFive/chroma_store`), not any particular caller's cwd —
so importing it from here, unmodified, means the orchestrator reads
and writes the SAME persistent memories `memory_chat.py` does. That's
the intended behavior for "one Jarvis, one memory," not an accident —
see the bug below for a real way this could have silently broken
instead.

## Milestone 4 design notes

**The `tools.py` name collision, resolved.** Phase 4's and Phase 6's
`tools.py` are different modules with the same filename — flagged as a
known future problem back in milestone 2's own note, now real, since
this milestone needs both simultaneously. A bare `import tools` with
both phase paths on `sys.path` resolves to whichever was inserted
last, silently shadowing the other. Resolved by loading each by
explicit file path via `importlib.util.spec_from_file_location()`
(`_load_module_from_path()`) instead — neither phase's directory needs
to be on `sys.path` for this at all, since neither `tools.py` has other
local imports of its own to resolve. Their `TOOL_SCHEMAS` (concatenated,
no name collisions between the two sets) and `TOOL_FUNCTIONS` (merged
dict) become this file's own `TOOL_SCHEMAS`/`execute_tool()`.

**Neither phase's own tool-calling loop could be reused as a black box
anymore — this milestone's `_run_tool_calling_turn()` is genuinely new
code, not an import.** Phase 4's `agent.run_turn()` is hardwired to its
own single `tools` module at import time; Phase 6's
`os_agent.run_turn()` is hardwired the same way to its own, plus has
no hook for anything outside its own confirmation gate. Once both tool
sets need to coexist in one loop, neither can be reused unchanged.
`_run_tool_calling_turn()` is adapted from both: Phase 4's loop shape
(call the LLM with tool schemas, execute, feed results back, repeat
until a plain-text answer or `MAX_TOOL_HOPS`) plus Phase 6's
confirmation-result interception and log-truncation on top, now
running against the merged registry. This is the expected shape of
Phase 7's job, not a departure from "reuse as-is" — the whole point of
this phase is wiring separate phases into one loop, and two different
phases' own tool-calling loops were never going to merge by import
alone. `MAX_TOOL_HOPS` is reused at Phase 6's value (8, raised from
Phase 4's 4) since multi-step UI tasks need more round-trips even when
everything's working correctly — same reasoning Phase 6's own README
gives.

**The confirmation gate itself is reused directly, unchanged.**
`confirm_pending_action()`/`cancel_pending_action()`, the in-memory
pending-action registry, the 5-minute TTL, the focus-shift and
click-out-of-bounds checks for `click`/`press_hotkey` — none of that
is voice-specific, and none of it needed to change; it's imported
straight from Phase 6's `tools.py` (part of the same merge above).
What's new is the human-facing side:
`_prompt_for_confirmation()`/`_handle_tool_result()`, adapted from
Phase 6's `os_agent.py`, with one addition — a short spoken heads-up
("That needs your confirmation. Please check the terminal.") before
blocking on the exact same real terminal prompt (`CONFIRM_PASSWORD` via
`getpass`, hidden input, fails closed on any read error) Phase 6's CLI
always used. The full destructive-action message (path, coordinates,
keys) is still only ever printed to the terminal, never spoken — voice
announcing a file path or click coordinates aloud isn't a good
security boundary or a good listening experience, and the whole point
of this gate is that a human reads and decides with their eyes, not
their ears. The `[NOTE] ...` print of the real outcome after
confirm/cancel is reused as-is too, for the same reason Phase 6 added
it: the model's own prose description of a tool result isn't always
trustworthy even when the result itself is correct (see Phase 6's
README).

**Why this matters more here than it did for Phase 6's own CLI.**
Phase 5's README already found that a memory shaped like a
prompt-injection payload can hijack the model's behavior for a
session. Phase 6's own README names the realistic consequence once
memory and OS-control tools share a loop: a poisoned "memory" as a
path to an *unintended* destructive tool call. This milestone is where
that stops being hypothetical — memory retrieval (milestone 3) now
feeds directly into the same loop that can call `delete_file`/`click`/
`press_hotkey`. The gate does not depend on the model behaving
correctly or the user trusting what it says; it lives entirely in
code, outside the model's reach, exactly as it did in Phase 6 —
merging the loops didn't weaken that boundary, since the gate was
never part of the loop being merged in the first place.

**Tool set is exactly the union of Phase 4's and Phase 6's, unchanged.**
No new tools were added or modified this milestone. `get_active_window`
being reachable (read-only, ungated) means the model can check window
focus before proposing a `click`/`press_hotkey` — same as Phase 6's CLI.

## Bugs found and fixed during milestone 1's first real run

Same lesson every earlier phase's manual testing already taught: the
mocked test suite passed throughout (it proves code paths run as
written, not that real-world assumptions hold) -- both of these only
showed up running against a real mic and real Whisper output.

1. **"goodbye" didn't stop the session.** Whisper transcribed a
   repeated, nervous goodbye as `"Goodbye. Good bye. Good bye."` (see
   #2 below for why it was repeated). The exit check, inherited
   as-is from Phase 2's `voice_chat.py`, required the *whole
   utterance* to exactly equal an exit phrase -- that string never
   does, even though every fragment in it clearly is one. It fell
   through to the LLM instead, which happened to reply "Goodbye." on
   its own, masking the failure. **Fixed:** `_contains_exit_phrase()`
   now splits the transcript on sentence-ending punctuation and checks
   each fragment independently. This keeps the original safety
   property -- an unrelated full sentence like "did I quit too early"
   still won't match, since neither the whole thing nor any fragment
   of it is itself in `EXIT_PHRASES` -- while tolerating repetition
   and STT noise around a genuine exit phrase. Also added `"good bye"`
   as an explicit alias alongside `"goodbye"`, since Whisper
   transcribed the two-word form.
2. **No acknowledgment on wake, causing the repetition in #1.**
   Nothing told the person Jarvis had heard the wake word, so they
   understandably repeated themselves mid-recording -- which is what
   produced the noisy, repeated transcript in the first place, and
   made that one recording run far longer than it needed to. **Fixed:**
   a short chime (`_play_wake_chime()`, Windows `winsound.Beep`, toggle
   via `WAKE_CHIME_ENABLED`) now plays immediately after wake
   detection and before recording starts.

## Bugs found and fixed during milestone 2 manual testing

Both found running the tool-calling pipeline against real hardware --
neither showed up in the mocked suite, same pattern as milestone 1.

1. **Follow-up window cut people off mid-sentence.** The follow-up
   window passed `FOLLOWUP_LISTEN_SECONDS` (6s) to
   `voice_io.record_until_silence()` as `max_seconds` -- but
   `max_seconds` is a hard cap on the *entire* recording from the
   moment it starts, not a "how long to wait for you to start talking"
   budget. If you started speaking anywhere other than right at the
   beginning of that 6s window, the recording still hard-stopped 6s
   after it opened, cutting you off mid-reply. **Fixed:**
   `record_until_silence()` now takes a separate `pre_speech_timeout`
   parameter -- it governs only the wait *before* speech is first
   detected; once speech starts, the normal (longer) `max_seconds` cap
   and silence-based stop take back over, same as any other turn. The
   orchestrator's own `record_max_seconds` parameter (on `run_turn()` /
   `run_conversation()`) was renamed to `pre_speech_timeout` throughout
   to match, since its old name was actively misleading once the
   semantics changed. See `phaseTwo/voice_io.py`'s
   `record_until_silence()` docstring and its two new regression tests
   in `phaseTwo/test_voice_io.py`.
2. **Falling back to full wake-word listening after a silent follow-up
   window sometimes re-triggered instantly, with no wake word said.**
   Took three real-hardware attempts to actually root-cause -- logging
   the exact reasoning here since the first two looked plausible and
   still turned out to be wrong.
   - **Attempt 1:** guessed a transient in a freshly (re)opened
     `InputStream`'s first callback(s). Fixed by discarding the first
     `startup_skip_frames` callbacks after `start()`. Did not fix it --
     still reported happening on the next real-hardware pass.
   - **Attempt 2:** added diagnostic logging (score + frame count +
     elapsed time at trigger) instead of guessing a third time. The
     next occurrence: `score=0.990` at frame #5 (the very first frame
     scored after the skip window), 0.48s after the stream reopened,
     right after the just-closed follow-up recording's own stream. Read
     that as stale audio bleeding in from whichever `InputStream` had
     just closed on the same device, and added `restart_settle_seconds`
     -- a real pause *before* reopening the device at all. Still did
     not fix it: the next real occurrences (two more, in the same
     session) showed the *exact* same signature -- frame #5, ~0.45s
     since reopen, 0.98-0.99 confidence -- every single time. That
     precision was the actual tell: real ambient noise wouldn't line up
     identically on every occurrence.
   - **Root cause (attempt 3):** openWakeWord's `Model` object is
     constructed once in `orchestrator.py` and reused for the entire
     process -- restarting the `InputStream` never touched it.
     `Model.predict()` keeps its own internal `prediction_buffer` (a
     smoothed rolling window of recent scores) and preprocessor buffer;
     `Model.reset()` exists specifically to clear both. After a genuine
     "hey jarvis" detection, that buffer is full of high scores, and
     nothing was ever clearing it -- so the model kept "remembering"
     the old detection the next time scoring resumed. This also
     explains the exact frame-#5 repeat: `startup_skip_frames` delays
     when `predict()` is even called again, during which the stale
     buffer sits completely untouched. **Fixed:** `WakeWordGate.start()`
     now calls `self._model.reset()` on every restart.
     `startup_skip_frames` and `restart_settle_seconds` stay in place as
     cheap, likely-inert secondary layers in case a real stream-level
     issue also exists, but `model.reset()` is the fix that actually
     matters. **Not yet re-confirmed against real hardware** -- if a
     spurious trigger recurs, check the log for a `reset` happening
     before it rather than reaching for either of the other two knobs.
3. **"goodbye" still wasn't recognized when it trailed another phrase
   in the same breath.** Milestone 1's fix (splitting on
   sentence-ending punctuation, checking each fragment for an *exact*
   match) fixed the repeated-"goodbye" case, but "okay, goodbye" or
   "alright then, goodbye" -- said as one breath, no `.`/`!`/`?`
   between the lead-in and the farewell -- transcribe as a single
   fragment ("okay, goodbye") that still doesn't equal "goodbye"
   exactly, so it fell through to the LLM the same way the original
   bug did. **Fixed:** `_contains_exit_phrase()` now also checks
   whether a fragment *ends* with a phrase from a new, deliberately
   smaller set, `TRAILING_SAFE_EXIT_PHRASES = {"goodbye", "good bye"}`
   -- not extended to the rest of `EXIT_PHRASES`, since "stop" and
   "exit" are far too common as the last word of an ordinary, non-exit
   request ("make it stop", "where's the exit"), and "shut down"/
   "shutdown" show up in ordinary questions too ("when does the store
   shut down"); "quit" is judged the same way, conservatively, for
   consistency. Known, accepted tradeoff: a fragment like "not
   goodbye" would now also match -- judged rare enough in practice not
   to be worth guarding against (e.g. with negation detection) given
   the much more common case this fixes. See
   `_contains_exit_phrase()`'s docstring and its dedicated unit tests
   in `test_orchestrator.py` (including explicit tests that "make it
   stop" / "where's the exit" / "why did you quit" / "when does the
   store shut down" still correctly do NOT trigger exit).

## Bugs found and fixed during milestone 3 manual testing

Caught by manual verification against the real (non-mocked) memory
store, not the mocked test suite -- which wouldn't have caught this at
all, since it never touches the real `chromadb`/`python-dotenv`
machinery. Same lesson every earlier phase's real-hardware testing has
already taught, once again: exercise the real path at least once, even
when the mocked suite is green.

1. **The "shared memory store" design silently broke, some of the
   time, depending on how the orchestrator was invoked.** First real
   check (`python -c "..."` from inside `phaseSeven\`) showed the
   memory store correctly resolving to `phaseFive/chroma_store` with
   its real, existing memories (`count() == 4`). A second check,
   running an actual `.py` script file instead of an inline `-c`
   string, showed `count() == 0` -- a completely different, empty
   store. Root cause: `memory.py` calls `load_dotenv()` at its own
   module level, and python-dotenv resolves that relative to
   **memory.py's own file location** (`phaseFive/`), not this
   process's cwd or `orchestrator.py`'s location -- confirmed by
   checking `find_dotenv()`'s actual resolved path directly rather than
   guessing. So importing `memory.py` from here loads `phaseFive/.env`,
   which sets `CHROMA_DB_PATH=./chroma_store` -- a **relative** path.
   That path then resolves against whatever this process's cwd happens
   to be (`phaseSeven`, per this project's own "cd phaseSeven; python
   orchestrator.py" convention), silently creating/opening an empty,
   wrong store there instead of the real, populated
   `phaseFive/chroma_store`. This is the exact same bug class Phase 5's
   own README already documented once, for a plain wrong-cwd case with
   its own CLI -- resurfacing here through a cross-phase import
   instead, in a way the mocked test suite structurally cannot catch
   (it never constructs a real `MemoryStore` or triggers a real
   `load_dotenv()` call). **Fixed:** `orchestrator.py` now sets
   `os.environ.setdefault("CHROMA_DB_PATH", str(PHASE_FIVE_PATH /
   "chroma_store"))` explicitly, before `memory.py` is ever imported --
   `setdefault` so an explicit `CHROMA_DB_PATH` in this phase's own
   `.env` (if anyone ever wants a separate, non-shared store) still
   wins; python-dotenv never overrides an already-set env var by
   default, so this reliably wins over `memory.py`'s own
   `load_dotenv()` call regardless of cwd or invocation style.
   Re-verified from both `phaseSeven\` and the project root after the
   fix -- both now correctly resolve to the shared store.

## Follow-up window (added after initial testing)

After Jarvis replies, it now listens directly for up to
`FOLLOWUP_LISTEN_SECONDS` (default 6s) without needing the wake word
again -- so a quick back-and-forth doesn't require re-triggering wake
detection every single turn. If you keep talking, each subsequent turn
gets the same short window; if you go quiet for that long, it falls
back to full wake-word listening -- a normal, expected end to the
conversation, not an error.

`run_turn()` now returns a `TurnOutcome` (`CONTINUE` / `EXIT` /
`NO_SPEECH`) instead of a bare bool, and a new `run_conversation()`
chains turns together on top of it -- first turn uses the normal
recording window, each turn after a `CONTINUE` uses the shorter
follow-up window, and it stops on `EXIT` or `NO_SPEECH`. The bare bool
this replaces used to conflate "the conversation is continuing" with
"go back to wake-word listening" into the same `True` — that stopped
being precise enough once there's a second kind of listening
(follow-up) to fall back *from*, not just the original wake-word
listening.

Set `FOLLOWUP_ENABLED=false` to turn this off entirely and always
require the wake word, matching the original milestone-1 behavior.

## Real-hardware verification log

All four milestones have now landed (tool calling, memory, OS
control), and each has had a dedicated real-hardware (not just mocked)
verification pass:

- **Milestone 2's real-hardware fixes re-confirmed.** The wake-word
  spurious-relisten `Model.reset()` fix and the trailing-"goodbye"
  exit-phrase fix both got a dedicated real-hardware re-test after
  landing (beyond just the mocked suite) — confirmed still working as
  expected, no regressions.
- **Milestone 3's full voice round-trip confirmed.** Said "remember
  X" out loud, then asked a question that should recall it in an
  actual voice session — worked as expected. Combined with the memory
  store's read/write path already verified directly (see its bug entry
  above), this milestone is now fully confirmed, not just
  provisionally done.
- **Milestone 4 fully confirmed, including by voice.** Non-voice path
  verified for real first (live DeepSeek, not mocked): asked "what
  window is currently focused?" — the live model correctly called the
  Phase-6 `get_active_window` tool (reached through the merged
  registry) and answered correctly; asked it to delete a real throwaway
  scratch file — the model checked it first (`read_file`), then
  proposed `delete_file`; the gate correctly stopped and printed
  `[CONFIRMATION NEEDED]` with the real path; declining (piped "no" via
  the no-`CONFIRM_PASSWORD` yes/no fallback, tested in-process only —
  the real `.env` was never touched) correctly cancelled, and the file
  was confirmed still on disk afterward. Separately confirmed the
  actual password-gated path is genuinely un-automatable:
  `getpass.getpass()` blocks on the real console directly, bypassing
  piped stdin entirely — it cannot be scripted around, which is
  exactly the intended property. **Then confirmed by voice, by the
  user, for real:** a full voice-triggered confirmation round-trip
  (spoken destructive request → spoken heads-up → typed real password
  at the terminal → spoken final outcome) worked as expected, and a
  `click`/`press_hotkey` test against a real window (needing
  `target_window_contains`) also worked as expected, confirming the
  focus-shift check still behaves correctly reached via the
  voice-triggered path instead of Phase 6's own typed CLI loop.

## Phase 8 extension seams (added after Phase 7 was complete)

Phase 8 layers on top of this orchestrator rather than forking it — the
wake-gate loop here carries several real-hardware fixes (`Model.reset()`
between turns, the settle delay, the chime, the split pre-speech
timeout) that were not worth owning in two places. Two small seams were
added for it. Both are no-ops when unused, so Phase 7's own behavior
running `orchestrator.py` directly is unchanged:

- **`register_tools(schemas, functions)`** — adds tools to the merged
  registry or replaces existing ones by name, so Phase 8 could fix
  Phase 4's stub `get_weather` and add the missing clock without
  editing Phase 4's own finished `tools.py`. Raises immediately if the
  schema names and function names disagree in either direction.
- **`run_orchestrator(system_prompt_provider=...)`** — called for a
  fresh system prompt at the start of every wake-triggered
  conversation, so Phase 8 can state the current date without it going
  stale in a long-running process.
- **`register_confirmation_handler(token_prefix, confirm, cancel)`** —
  routes a confirmation token to the registry that issued it. Phase 8
  adds destructive tools of its own (closing a window, killing a
  process, sleeping the machine) and cannot go through Phase 6's
  `confirm_pending_action()`, which dispatches on a hardcoded if/elif
  chain over its own four action types. Routing is by token prefix
  rather than by trying each registry in turn — the latter would make
  the gate depend on registry ordering and error-string matching.
  Phase 6's tokens are bare hex, remain the default, and are
  unaffected. What does not change for any registered registry:
  `confirm`/`cancel` stay out of `TOOL_SCHEMAS`/`TOOL_FUNCTIONS`
  entirely, so the model never gets a tool-calling path to either.
- **`register_speaker(speak_function)`** — replaces what `speak_safe()`
  uses to produce audio. Phase 8's barge-in speaks a reply in
  interruptible chunks while a microphone listens for "skip", which is
  a different way of *speaking*, not a different thing being said — so
  every caller still just calls `speak_safe()`. The `SpeechError`
  fallback stays wrapped around whatever is registered, deliberately:
  the reason it exists (a dead audio driver must never kill the loop)
  applies at least as much to a speaker with a mic and a transcription
  model behind it.
- **`register_reply_filter(fn)`** — normalises the assistant's reply
  once, *before* it is logged or spoken. Added to fix a real mismatch:
  Phase 8 rewrites text on its way to the speech engine (moving "sir"
  to the end of its sentence, stripping markdown that would otherwise
  be read out as literal characters), but that happened at render time,
  downstream of the log line. The terminal printed what the model wrote
  while the speakers said something slightly different — the kind of
  small inconsistency that wastes an hour when the log is the only
  record of what happened. A filter that raises falls back to the
  original text: a cosmetic tidy must not be able to lose a reply.

**The confirmation gate now says WHY out loud.** It used to speak only
"That needs your confirmation. Please check the terminal", which told
the person nothing about what they were being asked to approve — a poor
way to ask someone to make a security decision.
`_spoken_reason_for()` maps the proposal to a general-terms warning
("That would permanently delete a file, and it cannot be undone. I'll
need your confirmation at the terminal."). It deliberately still names
only the KIND of action and its consequence, **never the target** — no
absolute paths, no click coordinates, no keystrokes. That boundary is
unchanged and is the same one documented under "Confirmation design"
above: the full detail prints to the terminal a foot away, which is
where the human is confirming anyway. An unrecognised message degrades
to a generic destructive-action warning rather than to silence or to
reading the message aloud.

**One real Phase 7 bug was fixed at the same time**, found during Phase
8's accuracy work and not by this suite: `jarvis.trim_history()` is a
bare `messages[-40:]`, so once a session passed 40 messages it dropped
`TOOL_SYSTEM_PROMPT` off the front along with the oldest turn — and
every call after that went to DeepSeek **with no system message at
all**. No spoken-output guidance, no read_screen-is-OCR guidance, no
confirmation-gate framing. The session keeps working, just
progressively worse, which presents as the model getting dumber rather
than as a bug. Fixed by `_trim_history_preserving_system()`; the
message budget is now `MAX_HISTORY_MESSAGES` conversational messages
*plus* the system one. See `phaseEight/README.md` for the full
write-up.

## Limits raised (Phase 8 polish pass)

These were "get something working first" compromises, not considered
limits, and complex multi-step requests were dying on them.

| | Was | Now |
|---|---|---|
| `MAX_TOOL_HOPS` | 8 | **40** |
| turn wall-clock budget | none | **180s** |
| `MAX_HISTORY_MESSAGES` | 40 (Phase 1's) | **200** (Phase 7's own) |

**A bigger hop number alone would have been the wrong fix**, for two
reasons.

First, the cap exists because a model that misreads a tool result can
loop, and at 40 hops that is 40 API calls and several minutes of Jarvis
talking to itself before anything stops it. `TURN_TIME_BUDGET_SECONDS`
is what actually bounds that: real work finishes well inside it, a loop
hits the wall quickly regardless of hops remaining. Two limits for two
different failure modes — a long legitimate task, and a short
pathological one.

Second, and more seriously: **each hop appends at least two messages**
(the assistant's tool-call, plus one result per tool). At 40 hops a
single turn can append ~80 messages — twice the entire old 40-message
history budget. So a complex turn would have erased the conversation
that prompted it, and the next trim would have landed mid-tool-sequence,
which is exactly the `trim_history()` sharp edge flagged in milestone 2
above as "unlikely in practice at MAX_HISTORY_MESSAGES=40". Raising the
ceiling is precisely what made it likely.

So the trimmer had to be fixed first, not afterwards:

- `_drop_orphaned_tool_messages()` removes tool-role messages stranded
  at the front of a trimmed window. DeepSeek requires every tool result
  to follow the assistant message whose `tool_calls` it answers, and a
  tail slice can land exactly between them. Dropping leading tool
  messages is the whole fix and is sufficient: a cut can only orphan
  *results*, never *requests*.
- `MAX_HISTORY_MESSAGES` is now Phase 7's own, configurable, and
  defaults to 200 — comfortably more than one full-length turn can
  produce. **Phase 1's own default is untouched**: its text CLI has no
  tool calls and no reason to carry 200 messages. Phase 1's
  `trim_history()` is consequently no longer used here.

The give-up messages were also rewritten. `"(gave up after too many
tool calls in a row -- check MAX_TOOL_HOPS)"` is a log line, not
something to say to a person out loud; it now suggests breaking the
request up.

**Verified live:** a six-part request ("tell me the time, check the
weather in Sydney, disk space, what windows are open, top memory
processes, and search the web") completed in **one turn, 6 tool calls,
15 seconds** — the class of request that previously died at hop 8.

## Not yet covered / known gaps

What's left is genuinely "polish beyond the roadmap's Phase 7 scope,"
not a missing capability from any of Phases 1-6:

- ~~Any interruption/barge-in support.~~ **Built in Phase 8's milestone
  3** — see `phaseEight/README.md`. It needed a fourth seam here
  (`register_speaker`) but no change to `run_turn`/`run_conversation`:
  a reply cut off mid-way looks the same to the conversation loop as
  one that finished early, so the follow-up window opens exactly as it
  already did.
- Multi-turn conversations across separate wake events don't currently
  reset `history` — it's carried forward for the life of the process,
  same as Phase 1/2/5's chat loops. Long-lived in-process `history` and
  persistent cross-session memory can end up duplicating the same
  information two different ways. Not a bug, just an unresolved design
  question worth revisiting if it causes confusing behavior in
  practice.
- `list`/`forget` (Phase 5) are deliberately not wired into voice at
  all — see "Milestone 3 design notes" above for why.
- **`read_screen` needs Tesseract installed separately from the pip
  package** (see phaseSix/README.md) — confirmed present on this
  machine at setup time, but worth knowing if this ever runs somewhere
  else.
- **No CONFIRM_PASSWORD strength/rotation policy.** It's a plain string
  compared with `==` — fine for a single-user local assistant, but
  worth knowing if this project's threat model ever changes (shared
  machine, remote access, etc.).
- **Prompt/personality design for consistent character** — the roadmap
  names this as one of Phase 7's own goals; `TOOL_SYSTEM_PROMPT` is
  currently purely functional ("helpful assistant with tools"), not an
  actual designed character.

## Running it

```
pip install -r requirements.txt
cp .env.example .env   # fill in as needed
python orchestrator.py
```

## Testing it

```
pytest
```

Both suites are fully mocked — no real audio device, model weights, or
the `openwakeword` package itself required to run them (it's stubbed
via `sys.modules`, same pattern Phase 6's tests used for
`PIL`/`pytesseract`/`pyautogui`). The real `openai` package does need
to be installed (it's imported at module level by Phase 4's
`llm_client.py`), but no `DEEPSEEK_API_KEY` is needed to run the
suite — `think()` itself is always faked in tests, so the real
DeepSeek client is never actually built. Same story for `chromadb`
(Phase 5, milestone 3): it needs to be installed (`memory.py` imports
it, lazily, inside `MemoryStore.__init__`), but an autouse fixture
(`no_memory_by_default`) forces `_get_memory_store()`'s own
sticky-failure short-circuit for every test by default, so nothing
here ever constructs a real `MemoryStore` or touches real chromadb —
tests that need a store use a small fake instead (see the
`_get_memory_store()`/remember-command/retrieval test sections). This
matters more than it might look: the mocked suite structurally cannot
catch the cross-phase `load_dotenv()` bug logged above, since it never
exercises the real `memory.py` import path at all — that one only
showed up by actually running the real thing.

**Milestone 4 (Phase 6 OS-control tools)**: `Pillow`/`pytesseract`/
`pyautogui` need to be installed (Phase 6's `tools.py` is loaded for
real -- via `importlib`, not stubbed -- so importing it for real
touches these at import time... except it doesn't: exactly like
`memory.py`'s lazy `chromadb` import, `tools.py` only imports
`pytesseract`/`PIL`/`pyautogui` lazily inside the specific functions
that use them, so loading the module itself, and everything the test
suite exercises, needs none of them installed to *run* the mocked
suite -- they only matter for a real `read_screen`/`move_mouse`/
`type_text`/`click`/`press_hotkey` call, none of which the mocked
tests make). `CONFIRM_PASSWORD` also doesn't need to be set to run the
suite -- `_prompt_for_confirmation` is monkeypatched directly in the
tests that exercise `_handle_tool_result`'s dispatch, and set/unset
explicitly per test where the password-gate logic itself is under
test. As with the memory-store bug above: this suite proves the
merge/gate *logic* is correct, not that a real voice-triggered
confirmation round-trip actually works end-to-end against a real
terminal -- see "Not yet covered" above.