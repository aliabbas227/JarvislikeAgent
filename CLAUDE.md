# Jarvis — Personal AI Assistant

## Why
Building a locally-run, voice-activated personal assistant (Iron
Man's Jarvis, roughly) as a learning project, incrementally, one
phase at a time.

## What
Windows 11 desktop, i5-10600KF, RTX 3060 (12GB), 16GB RAM. Each phase
(`phaseOne/` through `phaseSeven/`) is its own folder with its own
venv — see "Working rules" below for why. `jarvis-agent-roadmap.md`
at the project root is the original phase-by-phase plan this project
follows.

- phaseOne — text CLI chatbot (Ollama or Anthropic backend)
- phaseTwo — voice I/O (Whisper STT, pyttsx3 TTS)
- phaseThree — wake word (openWakeWord, "hey_jarvis")
- phaseFour — tool-calling agent (DeepSeek backend, `deepseek-chat`)
- phaseFive — persistent memory (ChromaDB + RAG)
- phaseSix — OS control tools (file ops, screen OCR, mouse/keyboard),
  with a two-phase confirmation gate on anything destructive
- phaseSeven — orchestration: wiring everything above into one
  continuous voice loop. **All four milestones complete and confirmed
  against real hardware — see "Current status" below.**
- phaseEight — polish on top of a finished Phase 7. **Not a standalone
  mini-project**: `app.py` imports `phaseSeven/orchestrator.py` and
  extends it through two small seams (`register_tools`,
  `run_orchestrator(system_prompt_provider=...)`) rather than forking
  the voice loop. Run `phaseEight/app.py`, not `orchestrator.py`.

## How

### Commands
Each phase has its own venv — always activate the right one before
running anything:
```
cd phaseEight
.\venv\Scripts\Activate.ps1
(Get-Command python).Path   # verify it actually points into phaseEight\venv
pytest                       # run that phase's test suite
python app.py                # run Jarvis for real (needs mic/speakers)
```
`phaseEight/app.py` is the current entry point — it imports and extends
Phase 7's orchestrator. `phaseSeven/python orchestrator.py` still runs
standalone (without Phase 8's accuracy layer) if you need to isolate a
problem to Phase 7.
A wrong-venv mixup has already caused real bugs here (wrong
`openwakeword` install location) — always verify the interpreter
path, don't trust the `(venv)` prompt prefix alone.

### Working rules
- **Standalone-phase discipline (phases 1–6):** each phase is a
  self-contained, working, "small and ugly first" mini-project with
  its own venv. Cross-phase imports only where explicitly justified
  (documented in that phase's own comments/handoff, e.g. Phase 2
  reusing Phase 1's `call_llm()`). **Phase 7 is deliberately where
  this discipline ends** — its whole job is wiring 1–6 together, and
  Phase 8 layers on top of Phase 7 the same way.
- **A new venv needs openWakeWord's weights downloaded separately.**
  The pip package doesn't ship them, and `build_model()` looks inside
  that venv's own `openwakeword/resources/models/`. Run
  `python -c "import openwakeword.utils as u; u.download_models()"`
  once per venv. Hit for real in Phase 8; the mocked suites can't catch
  it, since they stub `openwakeword` out entirely.
- **Audio devices are pinned once, in `app.py`, via
  `audio_device.pin_devices()`.** Never add a `device=` argument to an
  individual `sd.InputStream()` — `sd.default.device` is shared
  process-wide, so one assignment covers the wake gate, the recorder,
  barge-in and playback, and pinning them individually is how they end
  up on different devices. Prefer non-WDM-KS host APIs: WDM-KS
  supports the wake gate's callback but not `voice_io.py`'s blocking
  read, so it wakes and then fails every turn.
- **Every cross-phase `.env` load gets an absolute path.** A bare
  `load_dotenv()` resolves relative to the calling file, and silently
  falls back to the cwd when it can't identify one. This bug class has
  hit this project three times now (Phase 5's README, Phase 7's
  milestone 3, Phase 8's milestone 1) and is structurally invisible to
  the mocked suites every time.
- **Every phase/milestone ends with a handoff doc or README** (design
  decisions, bugs found via manual testing, what's deferred). Read the
  relevant one before touching that phase's code.
- **Mocked pytest tests are necessary but not sufficient.** Every
  phase so far has had real bugs that only showed up running against
  real hardware/APIs, invisible to the mocked suite. Manual,
  real-hardware testing is expected and its findings get logged as
  "bugs found" in that phase's README, same style as Phase 6's
  handoff doc.
- **Delivery discipline:** only touch files that need to change. Run
  the test suite before considering something done. Flag judgment
  calls explicitly rather than silently deciding.
- Backend: DeepSeek (`deepseek-chat`) for tool-calling phases —
  Ollama was rejected in Phase 4 for unreliable tool-calling.
- Config lives in each phase's `.env` (see that phase's
  `.env.example`), never hardcoded.

## Current status (update this section as work progresses)
**Phase 7, Milestone 1: done and confirmed working against real
hardware.** Wake word → voice → LLM loop, with a follow-up window (no
wake word needed for quick back-and-forth) and a wake chime.

**Phase 7, Milestone 2: code complete, manually tested against real
hardware once, two bugs found and fixed, mocked suite passing
(41 phaseSeven + 15 phaseTwo).** Folded Phase 4's tool-calling loop
into the orchestrator — the THINKING step now runs `agent.run_turn()`
against DeepSeek (reused as-is from Phase 4) instead of Phase 1's
`call_llm()`, so voice turns can call Phase 4's tools (`get_weather`
stub, real `set_timer`, real `search_web` via Tavily). `LLM_BACKEND`/
Ollama/Anthropic are consequently unused by `orchestrator.py` now.
Memory and OS control still not wired in. Real-hardware testing
found: (1) the follow-up window's short timeout was cutting people off
mid-sentence — fixed by splitting it into a short "wait for speech to
start" timeout vs. the normal recording length once speech starts; (2)
falling back to wake-word listening after a silent follow-up could
spuriously re-trigger instantly, no wake word said — took three
real-hardware attempts to root-cause. The actual cause: openWakeWord's
`Model` object is constructed once and reused for the whole process,
and its internal score-smoothing buffer was never cleared between
gate restarts, so it kept "remembering" a previous genuine detection.
Fixed by calling `Model.reset()` on every `WakeWordGate.start()` — two
earlier attempts (discarding initial audio callbacks, then adding a
pause before reopening the stream) targeted the audio/driver layer and
didn't help, since the stale state actually lived in the model object,
not the stream. A third, smaller bug also surfaced: "goodbye" trailing
another phrase in the same breath ("okay, goodbye") wasn't recognized
as an exit command — fixed with a deliberately narrow trailing-clause
match limited to "goodbye"/"good bye" only (not "stop"/"exit"/etc.,
which are too common as the last word of an unrelated request). See
`phaseSeven/README.md`'s milestone 2 bug-log entries for the full
reasoning on both. Neither fix has had an explicit dedicated
re-confirmation pass since landing.

**Phase 7, Milestone 3: code complete, mocked suite passing (75
phaseSeven), one real bug found and fixed via manual (non-mocked)
verification.** Folded Phase 5's persistent memory into the
orchestrator — every turn now retrieves relevant stored memories and
folds them into the message before it reaches `think()` (Phase 5's
own `build_augmented_message()`, reused as-is), and a spoken "remember
<fact>" command stores a new one, matching Phase 5's own
explicit-storage design. `list`/`forget` deliberately not wired into
voice — awkward over voice (reading a whole store aloud, saying a hex
id out loud), `memory_chat.py` remains the way to do that. Real bug
found: `memory.py` calls `load_dotenv()` at its own module level,
which python-dotenv resolves relative to memory.py's own file location
(phaseFive), not the importing process's cwd — so importing it from
orchestrator.py silently loaded `phaseFive/.env`'s
`CHROMA_DB_PATH=./chroma_store` (a relative path), which then resolved
against phaseSeven's cwd into an empty, wrong store instead of the
real, shared one. Same bug class Phase 5's own README already
documented once for a wrong-cwd case, resurfacing through a
cross-phase import this time, and undetectable by the mocked suite
(which never touches the real `memory.py` import path). Fixed by
`orchestrator.py` explicitly setting `CHROMA_DB_PATH` (via
`os.environ.setdefault`, so an explicit override still wins) before
importing `memory.py` at all. Re-verified from two different working
directories after the fix. **Full voice round-trip since confirmed
working** (said "remember X" out loud, then asked a question that
should recall it, in an actual voice session) — this milestone is now
fully confirmed, not just provisionally done.

**Phase 7, Milestone 4: code complete, mocked suite passing (97
phaseSeven), fully confirmed working — both non-voice (live DeepSeek)
and by real voice round-trip.** Folded
Phase 6's OS-control tools into the orchestrator. Since Phase 4's and
Phase 6's `tools.py` share a filename, both are now loaded side by
side via `importlib` (`_load_module_from_path()`) rather than
`sys.path`, and merged into one `TOOL_SCHEMAS`/`execute_tool()`
registry — the collision flagged as a future risk back in milestone
2's own note, now real and resolved. Neither Phase 4's `agent.run_turn()`
nor Phase 6's `os_agent.run_turn()` could be reused as a black box
anymore once both tool sets needed to coexist (each is hardwired to
its own single tools module); `orchestrator.py` now has its own
`_run_tool_calling_turn()`, adapted from both, running the merged
registry. The confirmation gate itself
(`confirm_pending_action`/`cancel_pending_action`, the pending-action
registry, the TTL, the focus-shift/out-of-bounds checks) is reused
directly, unchanged, from Phase 6's `tools.py` — none of that needed
to change for voice. What's new is the human-facing side
(`_prompt_for_confirmation`/`_handle_tool_result`, adapted from Phase
6's `os_agent.py`): a short spoken heads-up, then the exact same real
terminal password prompt (`CONFIRM_PASSWORD` via `getpass`, fails
closed) Phase 6's CLI always used — matching this project's own
already-decided design (see `phaseSeven/README.md`'s "Confirmation
design" note): voice can propose a destructive action, but confirming
it always goes through a separate, non-voice channel. This matters
more now than when that note was written: Phase 5's memory retrieval
now feeds directly into the same loop that can call
`delete_file`/`click`/`press_hotkey`, which is exactly the poisoned-memory-to-
destructive-tool-call path Phase 5/6's own READMEs already flagged as
a realistic risk — the gate lives entirely in code, outside the
model's reach, unaffected by merging the loops.

**Verified for real (live DeepSeek, not mocked), everything downstream
of transcription:** asked "what window is currently focused?" — the
live model correctly called the Phase-6 `get_active_window` tool
through the merged registry and answered correctly. Asked it to delete
a real throwaway file — the model called `read_file` to check it
first, then proposed `delete_file`; the gate correctly stopped and
printed the real confirmation prompt; declining (via the no-password
yes/no fallback, tested in-process only — the real `.env` and its
`CONFIRM_PASSWORD` were never touched) correctly cancelled, and the
file was confirmed still on disk afterward. Separately confirmed the
actual password gate is genuinely un-automatable: `getpass.getpass()`
blocks on the real console directly (bypasses piped stdin entirely),
both in isolation and inside a real orchestrator run — it cannot be
scripted around. **Then confirmed by voice, by the user, for real:** a
full voice-triggered confirmation round-trip (spoken destructive
request → spoken heads-up → typed real password at the terminal →
spoken final outcome) worked as expected, and a `click`/`press_hotkey`
test against a real window (`target_window_contains`) also worked as
expected — the focus-shift check still behaves correctly reached via
the voice-triggered path instead of Phase 6's own typed CLI loop. See
`phaseSeven/README.md`'s "Real-hardware verification log" for the full
detail on all three milestones' confirmations.

**This completes all four originally-planned Phase 7 milestones, each
now confirmed against real hardware, not just mocked tests** — tool
calling, memory, and OS control are all wired into one voice loop and
verified working end-to-end, including the security-sensitive
confirmation gate.

**Phase 8, Milestone 1 (accuracy): code complete, mocked suites passing
(36 phaseEight + 111 phaseSeven), confirmed against the live DeepSeek
API, two real bugs found and fixed. Not yet voice-tested.** Prompted by
three complaints: wrong time, outdated search, suspected tool
conflicts. A live (non-mocked) probe found the third was the right
diagnosis and worse than expected — **there was no clock in the tool
registry at all**, so asked the time the live model called
`get_active_window`, and asked the date it called
`get_weather(location="Today")` plus `read_screen`. Compounding it,
Phase 4's `get_weather` is stub data (hardcoded "partly cloudy", 21°C)
whose disclaimer the model doesn't reliably relay — so "what's today's
date" was being answered out of fabricated weather for a place called
"Today". Search was the useful negative result: **Tavily was returning
genuinely current data all along**; what was thin was what Phase 4
asked for (3 results, 300-char snippets, no synthesized answer, no
recency control) and the fact that the model had no idea what today's
date was, so it could neither judge freshness nor write a date-anchored
query. Fixed by a new `get_current_time` tool, a real Open-Meteo
`get_weather`, a better-provisioned `search_web` (`include_answer`,
`search_depth`, optional `time_range`, more/longer results, Bearer
auth), and a system prompt rebuilt every conversation that states the
current date. All three overrides happen at the registry level —
**Phase 4's `tools.py` was not edited**, keeping that finished phase's
own CLI working as it always did.

Two real bugs found, neither catchable by the mocked suites. (1) In
Phase 7: `trim_history()` is a bare `messages[-40:]`, so past 40
messages it dropped `TOOL_SYSTEM_PROMPT` entirely and every subsequent
call went to DeepSeek with **no system message at all** — presenting as
the model getting dumber, not as a bug. Fixed with
`_trim_history_preserving_system()`. (2) In Phase 8's own wiring:
relying on `orchestrator`'s bare `load_dotenv()` to find
`phaseSeven/.env` left `CONFIRM_PASSWORD` unset under some launch
styles, **silently downgrading the destructive-action gate from a
password to a plain yes/no prompt** — the third appearance of this same
`load_dotenv()`-resolution bug class in this project (Phase 5's README,
Phase 7's milestone 3, now here). Rule going forward: **every
cross-phase `.env` load gets an absolute path.** Fixed and re-verified
from three different working directories.

Verified live: all three probe questions now route to the correct tool
and answer correctly, with markdown-free spoken output intact. **Still
outstanding: a real voice round-trip** — everything downstream of
transcription is confirmed, but no microphone has been in the loop yet.

**Phase 8, Milestone 2 (tool expansion): code complete, mocked suites
passing (139 phaseEight + 117 phaseSeven), confirmed against live
DeepSeek, one real bug found and fixed. Not yet voice-tested.** Grew
the tool surface from 15 to 35, in the two areas the user picked:
deeper OS/window control (`os_tools.py` — `list_windows`,
`focus_window`, `move_window`, `close_window`, `list_processes`,
`kill_process`, clipboard, `media_control`, `take_screenshot`,
`get_system_status`, `lock_screen`, `power_action`) and file/system
work (`file_tools.py` — `find_files`, `create_directory`, `move_path`,
`copy_path`, `extract_archive`, `get_disk_usage`, `read_document` for
PDF/docx). All local, no new API keys; Windows-first via ctypes against
`user32` rather than pywin32, matching Phase 6's own choice. New pip
deps: `psutil`, `pypdf`, `python-docx`.

Phase 8 needed **its own confirmation gate** (`gate.py`): Phase 6's
`confirm_pending_action()` dispatches on a hardcoded if/elif chain over
its own four action types and has no idea what "kill this process"
means. Same guarantee, implemented generically — it holds a Python
closure keyed by a token, in module memory, which the model never sees
and cannot call. Tokens are prefixed `p8-` and routed by
`orchestrator.register_confirmation_handler` (a third Phase 7 seam);
Phase 6's tokens are untouched and still route to Phase 6. Guardrail
split stated explicitly: `close_window`/`kill_process`/`power_action`
always gated; `move_path`/`copy_path`/`extract_archive` gated only when
they would replace something existing (exactly Phase 6's `write_file`
rule); `lock_screen` and `media_control` deliberately ungated with
written reasoning. Two identifier-drift guards beyond the gate — window
handles re-verified by title at confirm time (Windows reuses handles),
and PIDs pinned by process creation timestamp (PID reuse). Protected
system processes and Zip-Slip archive entries are refused outright
rather than gated. Ambiguous matches ("2 windows match 'firefox'") are
refused with the candidates listed, never guessed.

Real bug found by live verification: **the "no markdown" instruction
stopped working.** Phase 7's prompt ends with it, which worked while it
genuinely was the end — Phase 8 appends several hundred words after it,
and the first live probe came back as a markdown bulleted list, which
TTS reads aloud as literal punctuation. Milestone 2 makes this worse by
its nature, since most of what it added returns lists. Fixed by
restating the rule and placing it last in the assembled prompt; a test
pins the position. **General lesson: appending to a system prompt
silently demotes whatever used to be last.**

Verified live with all confirmations auto-declined: from 35 tools the
model picked correctly on every probe, and "close the Spotify window"
correctly chained `list_windows` → `get_active_window` →
`close_window`, engaged the gate, honoured the decline, and left
Spotify open.

**Milestones 1 and 2 are now confirmed by voice, by the user, against
real hardware** — clock, weather, window listing, system status and the
gated `close_window` round-trip (spoken heads-up → real
`CONFIRM_PASSWORD` at the terminal → spoken outcome) all worked, replies
came back as spoken prose rather than markdown read aloud, and tool
selection stayed accurate across all 35 tools. No new bugs surfaced.
Also fixed pre-flight: a fresh venv has openWakeWord's library but not
its weights, which crashed `app.py` at startup until
`download_models()` was run (see Working rules above).

**Phase 8, Milestone 3 (barge-in): code complete, mocked suites passing
(171 phaseEight + 120 phaseSeven), verified against real hardware
except for a person actually saying the word.** Say "skip" while Jarvis
is talking and it stops. This is the concurrency milestone Phase 7's
own docstring anticipated. Two design decisions were settled by
measurement, both against an obvious-looking answer that was wrong:

1. **Cross-thread `pyttsx3.stop()` silently does nothing.** Probed
   directly: no exception, clean return, no effect, and it left the
   speaking thread wedged 13.5s into a 10s phrase. So all pyttsx3 work
   stays on one thread — the reply is split into sentences and the
   interrupt flag is checked between them. Interrupting means "don't
   start the next sentence". Measured cost: 8.94s vs 8.50s for three
   sentences, ~0.2s per boundary. (A first measurement appeared to show
   chunking being *faster*; that was silence — `pyttsx3.init()` returns
   a CACHED engine when one is alive, and the reused engine produced no
   audio. Phase 2's `speak_text()` already avoids this, which is why
   `interrupt.py` calls it rather than driving pyttsx3 itself.)
2. **The loudness gate alone is not enough — echo really does get
   through.** With the threshold at 2.5x normal and a *headset* mic, a
   12s reply produced **7 occasions where Jarvis's own voice crossed
   the gate and reached Whisper**. All 7 were correctly rejected,
   because detection also requires the actual word. Pure voice-activity
   barge-in would have interrupted itself every reply.

Detection is therefore: loudness gate → short-utterance cap → Whisper →
transcript must be an interrupt phrase, with a word-count cap and
whole-word matching so "I'll skip the rest of the forecast" doesn't
interrupt Jarvis (verified for real with that exact sentence).
Interrupt words are deliberately NOT Phase 7's `stop`/`quit` exit
phrases — overloading them would make the most destructive command in
the loop depend on timing the person can't see. Needed a fourth Phase 7
seam (`register_speaker`) but **no change to `run_turn`/
`run_conversation`**: a cut-off reply looks the same to the loop as one
that finished early, so the follow-up window opens as it already did.
Fails soft throughout (dead mic → no barge-in, never a crashed loop),
and `BARGE_IN_ENABLED=false` restores Phase 7's speech exactly.

**First real voice test: "skip" did not register at all. Root-caused by
instrumenting; three code faults plus one hardware/config finding that
outranks them.**

1. **A fixed threshold this machine could never reach.** The detector
   demanded 2.5x `SILENCE_THRESHOLD` (1250 RMS); the mic measured 20 at
   rest and 143 while audio played, so *no audio ever reached the
   detector at all*. The constant was tuned for the open-mic case,
   which makes it far too high for a split-device setup and far too low
   for a genuinely echoing room. Fixed by deriving the threshold live
   from a low percentile of recent mic levels, times a ratio, with a
   floor.
2. **The buffer-drop heuristic discarded exactly the target case.**
   Audio loud for 2s straight was binned as "continuous, therefore
   Jarvis" — but a person talking *over* Jarvis is continuous loud
   audio. Now a flush interval, not a bin.
3. **The background estimate deadlocked in a noisy room** (found by
   simulating an open mic). It was fed only chunks *below* the current
   threshold, so when a room's floor sits above the minimum nothing is
   ever recorded as background, the threshold stays pinned, and
   everything counts as speech forever. Fixed by feeding every level in
   and taking a low percentile.

Echo rejection also changed: the listener is now told what Jarvis is
saying right now (`now_speaking`), so an interrupt word appearing in
Jarvis's *current sentence* is treated as echo. That replaced a blunt
four-word cap, which only worked if the person's voice arrived cleanly
alone — when two voices hit one mic, the clip contains both and length
says nothing about who spoke.

**The bigger finding: the default input device moves.** Across three
runs minutes apart, Windows' default input was a controller
headset (peak 143), the same headset (peak 20), then a powered-off
wireless headset (**peak 1**), while output drifted to the monitor over
HDMI. Every phase opens `sd.InputStream()` with no device argument, so
the whole pipeline silently follows that default — wake word and turn
recording included. It presents as "Jarvis stopped hearing me" with no
error. **For an always-on open mic the microphone must be pinned as the
Windows default**, which fixes every phase at once; pinning only Phase 8
would leave the wake word and recorder on a different device, which is
its own bug.

**Open-mic ceiling, measured not guessed:** mixing a real rendered
"skip" over a real rendered reply and sweeping the ratio, detection
needs the person **~3x louder (~10 dB)** than Jarvis at the mic (0.5x,
1x, 2x all missed; 3x and 5x detected). The limit is Whisper locking
onto the dominant voice, and tuning cannot move it much. Robust
open-mic barge-in needs a different detector — openWakeWord (already a
dependency, already proven here, but only has a `hey_jarvis` model, so
the trigger word would change) or real acoustic echo cancellation.

Added `calibrate_barge_in.py` (companion to
`phaseSeven/calibrate_mic.py`) because the original failure was
invisible: "didn't hear the word", "heard and rejected it", and "heard
nothing at all" were indistinguishable from outside. Also added a real
`phaseEight/.env` — previously only `.env.example` existed, entirely
commented out, so none of the documented tuning knobs were actually
settable.

**Detector replaced: barge-in now uses openWakeWord, not Whisper. The
interrupt phrase is the wake word ("hey jarvis"), not "skip".** After
all six faults above were fixed, it still failed in simulation at this
machine's own measured levels — a clip that transcribed perfectly as
`'Skip.'` on its own came back as `'Quick TENSCHES Okoi'` from inside
the listener, and widening the window to 1.9s did not help. That was
not another bug: Whisper is an utterance transcriber, and one short
quiet word over the top of other speech is precisely its weak spot.

Measured on identical audio at identical levels:

| Detector | wake word over Jarvis's voice | Jarvis's voice alone |
|---|---|---|
| Whisper | garbage | correctly rejected |
| **openWakeWord** | **0.999 detected** | **0.000, no false positive** |

openWakeWord was already a dependency and already proven in Phase 3.
The rewrite deleted the entire Whisper path — no transcription, no
loudness thresholds, no pre-roll/hangover, no normalization, no
adaptive background — and with it the `INTERRUPT_MIN_THRESHOLD`/
`INTERRUPT_ADAPTIVE_RATIO`/`INTERRUPT_MAX_WORDS` knobs. The listener
builds its **own** Model rather than sharing the wake gate's, and calls
`reset()` per session, both because of the stale-prediction-buffer bug
Phase 7's milestone 2 spent three passes on. `calibrate_barge_in.py`
was rewritten to report real model scores instead of loudness.

**Verified end to end** with real TTS, a real mic stream and the real
model: control run spoke fully (16.4s, no false trigger); with "hey
jarvis" mixed into the mic stream it detected at 0.993 and stopped
after 1 of 4 chunks (4.3s). **Then confirmed by voice, by the user:
saying "hey Jarvis" out loud during a reply stops it.** Remaining gap
is open speakers rather than a headset.

**Phase 8, Milestone 4 (personality + voice): code complete, mocked
suites passing (177 phaseEight + 120 phaseSeven), confirmed live
against DeepSeek and the real voice pipeline.** Two requests that
turned out to be one problem: the film-Jarvis character, and a voice
that isn't a 2013 speech synthesiser. This also closes the last
outstanding item from the roadmap's own **Phase 7** goals
("prompt/personality design for consistent character"), carried
unaddressed since Phase 7 shipped.

**Voice:** `en-GB-RyanNeural` at `+15%` via `edge-tts` (free, no API
key, no account), played through sounddevice, in new
`phaseEight/voice.py`.
Chosen by rendering six candidates saying the same line and listening
side by side, not from descriptions. Worth knowing: pyttsx3 could only
see three SAPI5 voices, all ~2013 vintage; the registry also holds
better OneCore voices pyttsx3 cannot reach, but none neural and none
British. Cloud was judged acceptable because **DeepSeek already
requires internet** — if it is down Jarvis cannot think anyway, so the
voice adds no new failure mode. Piper (`en_GB-alan-medium`, fully
offline) was rendered and compared and stays the drop-in if the project
ever goes fully local, at 63MB. Any failure falls back to Phase 2's
`speak_text()`, logged once per process rather than per sentence.

**It made barge-in instant, unplanned.** Milestone 3 could only stop at
a sentence boundary because pyttsx3 cannot be stopped cross-thread.
sounddevice can: measured, `sd.stop()` from another thread ended a
14.0s clip at **1.74s**. The listener now cuts playback mid-word via an
`on_trigger` hook, closing milestone 3's known gap as a side effect of
changing engine. Speech is still chunked, but now for latency, not
interruption — rendering runs one sentence ahead of playback (measured
3-9x faster than real time), so the first word costs ~0.9s and there
are no gaps after it.

**Character:** written as behavioural rules rather than adjectives —
"be witty" licenses padding every reply with jokes, while "never let a
flourish delay the answer" and "never make two jokes in a row" are
bounded and checkable. The deference is deliberately not
obsequiousness: this thing can delete files and shut the machine down,
so a personality that flatters rather than warns would be worse than
none. Placement follows milestone 2's lesson — character opens the
prompt (identity belongs first, and Phase 7's prompt opens by calling
itself "a helpful personal assistant", the exact register being
replaced) and is re-anchored in one line just before the end, with the
no-markdown rule still holding the final position and all of Phase 7's
tool/gate guidance untouched underneath.

Verified live: *"It is 8:24 PM on Tuesday, the first of September,
sir."* / *"The machine is in fine spirits, sir... All quite
unremarkable, which is very much the desired state."* / and, asked to
delete every file in Documents, a refusal with a real reason and a
workable alternative. No markdown leaked into any reply.

**Two things were then corrected by ear, and both are worth recording
as method rather than detail.**

(1) *The speaking rate had been guessed.* The first version shipped
`-5%` — a hedge between the two samples that had actually been
compared (`+0%` and `-8%`), and a value the user had never heard. It
read as sluggish, and the first question back was "did you pick the
slower voice or the normal one", which is exactly the confusion a
silent compromise produces. Re-rendered at `+0%`/`+8%`/`+15%`, chosen
by ear: **`+15%`**. This prompted a standing instruction — see the
memory note `ask-dont-assume`: surface real decisions rather than
picking a reasonable default.

(2) *"sir": the prompt lost, so code won.* Instructed explicitly to
write "sir" with no comma and at the end of a sentence, DeepSeek still
produced ", sir" in **4 of 5 live replies** - it treats that as correct
English and reverts. It is a speech problem, not a writing one:
edge-tts renders a comma as an audible pause, so "fine spirits, sir"
becomes "fine spirits [pause] sir", and a mid-sentence ", sir," gets
two pauses around it. Rendered both forms and compared by ear, the
end-of-sentence version was clearly cleaner, so the rule is now
**"sir" always lands at the end of its sentence, never with a comma
before it** - enforced deterministically in `voice.for_speech()`, which
**moves** the address rather than deleting it (deleting would silently
cost a reply its only use of it and erode the character over a
session). A comma-only pattern was not enough: it missed ", sir -",
which the model produced in 2 of 5 real replies, so the pattern now
accepts any sentence-continuing punctuation. Re-verified live: 0
problems in 6 replies. **General lesson: a mechanical rule the model
keeps breaking belongs in code, not in an argument with the prompt -
with the prompt still asking for the same thing, so the two agree
rather than fight.**

**Three further fixes after using it (milestone 4 follow-ups):**

(3) *Markdown was being read aloud.* edge-tts does not interpret
markdown, it reads the characters, so "that would be *hard*" was spoken
with the asterisks pronounced and bullets became "dash" utterances. The
prompt forbids markdown and mostly works, but it leaks exactly where a
model reaches for emphasis hardest. `voice.strip_markdown()` removes
the markers and never the words between them; a bare `*` with spaces
around it is left alone, since that is multiplication.

(4) *The terminal and the speakers disagreed.* `for_speech()` ran at
render time, downstream of Phase 7's log line, so the terminal printed
what the model wrote while the audio said the normalised version. Fixed
with a fifth Phase 7 seam, `register_reply_filter`, applying the
normalisation once upstream of both. A filter that raises falls back to
the original text -- a cosmetic tidy must not be able to lose a reply.

(5) *The confirmation gate now says WHY.* It used to speak only "That
needs your confirmation. Please check the terminal", which is a poor
way to ask someone to approve something they cannot see.
`_spoken_reason_for()` maps a proposal to a general-terms warning
("That would close a window, and any unsaved work in it could be lost.
I'll need your confirmation at the terminal."). It still deliberately
**never says the target** -- no paths, no coordinates, no keystrokes --
which is Phase 7's existing documented boundary, unchanged; the full
detail prints to the terminal, which is where the confirming happens.
Unrecognised wording degrades to a generic destructive warning rather
than to silence or to reading the message aloud.

Emphasis is deliberately left flattened rather than voiced: making tone
change would need SSML, and edge-tts builds its own SSML and escapes
the text it is given, so it is likely unreachable without changing
engine.

**Limits raised (polish pass, after complex multi-step requests kept
failing).** `MAX_TOOL_HOPS` 8 -> **40**, plus a new
`TURN_TIME_BUDGET_SECONDS` (180s) checked between hops;
`MAX_HISTORY_MESSAGES` 40 -> **200**, now Phase 7's own and
configurable (Phase 1's default untouched -- its text CLI has no tool
calls); `MAX_RECORD_SECONDS` 30 -> 90; `read_file` 5k -> 20k chars,
`read_document` 8k -> 30k, `get_clipboard` 5k -> 20k, `find_files` 25 ->
100 results and 20s -> 60s, `list_processes` 20 -> 60, `search_web` 5 ->
10 results.

**A bigger hop number alone would have been wrong, for two reasons.**
(1) The cap exists because a confused model can loop, and at 40 hops
that is 40 API calls before anything stops it -- the time budget is what
actually bounds that, so there are now two limits for two different
failure modes. (2) **Each hop appends at least two messages**, so a
40-hop turn can append ~80 -- twice the old 40-message history budget.
A complex turn would have erased the conversation that prompted it, and
the next trim would land mid-tool-sequence: exactly the
tool-call-unaware `trim_history()` sharp edge flagged since milestone 2
as "unlikely in practice at MAX_HISTORY_MESSAGES=40". Raising the
ceiling is what made it likely, so **the trimmer was fixed first**
(`_drop_orphaned_tool_messages()` -- a cut can only orphan results,
never requests, so dropping leading tool-role messages is the whole
fix). Safety limits were deliberately NOT touched: the confirmation
gate and its TTL, protected system processes, the Zip-Slip check, and
pyautogui's failsafe.

Verified live: a six-part request completed in one turn, 6 tool calls,
15s.

**Pre-existing failure found, not caused by this work:
`phaseSix/test_tools.py` has 17 failures** -- `click()` and
`press_hotkey()` gained a required `target_window_contains` argument
(Phase 6's own safety anchor, added "after a real incident") and the
suite still calls `tools.click(50, 60)`. Phase 6's tests were never
re-run after that change. Not fixed here; flagged for a decision.

**Phase 8, Milestone 5 (seeing the screen): code complete, mocked
suites passing (240 phaseEight + 153 phaseSeven), confirmed live.**
35 tools -> 39. New `phaseEight/screen.py`.

The request was "it should see what I see, not just the text, so moving
the mouse is actually useful instead of a gimmick". That contains two
different problems and the design keeps them apart:

  "what am I looking at?"      -> look_at_screen   (local vision model)
  "where is the Save button?"  -> list_ui_elements (accessibility tree, EXACT)
  "click Save"                 -> click_element    (by name)
  "what does that say?"        -> read_screen_text (OCR WITH positions)

**A vision model alone would not have worked.** VLMs estimate
coordinates from a downscaled image and are routinely tens of pixels
out, which on a large display is the difference between a button and
its neighbour. Grounding comes from **Windows UI Automation** -- the
same accessibility tree screen readers use, giving exact rectangles in
0.1-0.2s with no model involved. The system prompt states the division
explicitly, because a model given both tools will otherwise ask the
vision model where something is and click what it says.

`click_element` stays gated exactly like Phase 6's coordinate click --
knowing the target by name makes the click accurate, not safe -- but
the confirmation now reads "About to left-click the Button 'Close Tab'
in 'notes.txt - Notepad'" instead of coordinates. The element is
**re-resolved at confirmation time** (same class of check as
close_window's stale handle and kill_process's PID reuse).

Vision is **qwen2.5vl:7b via local Ollama**, on purpose: a screenshot is
the most sensitive thing this assistant captures. Measured on the 3060:
cold ~41s (model load), warm 2.6-4.5s; image width barely matters
(672px and 1280px both ~4s).

**Two bugs found.** (1) `list_windows` had been reporting **phantom
windows** since milestone 2 -- suspended UWP apps leave a DWM-*cloaked*
`ApplicationFrameWindow` that `IsWindowVisible()` calls visible, so the
model had been telling the user about "two Settings windows" that were
not open, and the accessibility tree could not inspect them. Fixed with
a `DWMWA_CLOAKED` check; the two window lists now agree exactly.
(2) `OLLAMA_URL` is shared with Phase 1, which sets it to a full
endpoint, so appending `/api/generate` produced a 404 -- only scheme and
host are read now.

Verified live: "what is on my screen" -> look_at_screen described VS
Code correctly; "what buttons are in Notepad" -> list_windows +
list_ui_elements returned real names; "click the Close Tab button" ->
list_ui_elements -> get_active_window -> focus_window -> click_element,
gated by name.

**Token cost fixed, and my earlier claim about it was wrong.** I had
listed "35 tool schemas on every call" as a flaw; measurement showed
DeepSeek serves them almost entirely from its prefix cache (5,376 hit /
39 miss on a warm call). The **real** problem was that the system prompt
stated the time to the minute, so every new conversation changed the
prefix and re-charged all 35 schemas at full price (4,263 miss tokens).
Moving the clock into the user turn via a sixth seam
(`register_turn_context`) fixed it: measured **4,263 -> ~59 full-price
tokens per conversation**, holding across day changes, with all tools
still available. Tool subsetting was therefore NOT done -- it would have
cost capability to solve a problem that measurement says does not exist.
`_log_usage()` now logs per-hop token and cache split at DEBUG.

**NOTE: that is now six seams bolted onto Phase 7** (`register_tools`,
`system_prompt_provider`, `register_confirmation_handler`,
`register_speaker`, `register_reply_filter`, `register_turn_context`).
Each is individually justified and collectively it is the layering
saying Phase 8 should own the loop. The user has said the
phaseOne..phaseEight folder split is temporary and this will become a
proper application -- that rewrite is where this gets resolved, and new
work should prefer designs that port cleanly rather than leaning on the
phase layout.

**All five Phase 8 milestones are built.**

**Reliability pass (not a milestone): code complete, mocked suites
passing (270 phaseEight + 164 phaseSeven), two of three fixes confirmed
against real hardware.** Prompted by scoping an always-on use case
(voice control while gaming) and finding that every part of it assumes
a process that stays alive, which this could not do.

(1) **One `try` wrapped the entire main loop.** `run_turn()` guards its
own three recoverable steps, but nothing guarded the loop *around* it --
the wake gate, the chime, the system-prompt provider, speaking. A mic
unplugged mid-session or edge-tts losing the network killed the process
outright. Now each iteration has its own `except Exception` that logs,
releases the mic, backs off and returns to listening. Deliberately does
NOT catch `KeyboardInterrupt` (it is `BaseException`, so Ctrl-C still
stops Jarvis immediately rather than being retried as a failed turn),
and deliberately does NOT retry forever --
`MAX_CONSECUTIVE_TURN_FAILURES=5`, with the counter reset by any
completed turn so a flaky device never accumulates into a shutdown.

(2) **New `phaseEight/audio_device.py`** — pins `sd.default.device`
once, in `app.py`, before anything opens a stream. **One assignment
fixes all four live audio call sites** (`voice_io` recording,
`wake_gate` waking, `interrupt` barge-in, `voice` playback) because
they share the process's single `sounddevice` module and none pass
`device=`. No edits to phases 2/3/7: pinning only Phase 8 would leave
the wake word on a *different* device from barge-in, and threading an
argument through three finished phases would break their standalone
discipline. Config is a **name fragment, not an index** (indices
renumber on connect); an ambiguous fragment is refused with candidates
listed, never guessed — same identifier-drift reasoning as
`close_window`'s handle re-verification. Two measured findings changed
the design: **`sd.check_input_settings()` is not sufficient** (it
passes for WDM-KS devices whose blocking read then fails with
`PaErrorCode -9999`; `voice_io.py` reads blocking while `wake_gate.py`
uses a callback, so a WDM-KS mic *wakes and then fails every turn*), so
`_usable()` does a real open and real read and `HOST_API_PREFERENCE`
ranks WDM-KS last; and the live failure was worse than drift —
`sd.default.device` was `[-1, 1]` with every host API reporting
`default_input_device = -1`, so Jarvis died at the wake gate before a
single turn. `python audio_device.py` is a diagnostic entry point
(companion to `calibrate_mic.py`/`calibrate_barge_in.py` — those tune a
mic that works, this says whether you have one).

(3) **A hung LLM call could block ~30 minutes.** `build_client()` was
used bare, inheriting the SDK's `read=600s` with 2 retries.
`TURN_TIME_BUDGET_SECONDS` does not cover it — that is only checked
*between* hops. Now `timeout=60`, `max_retries=1` via `.with_options()`
in Phase 7 rather than editing Phase 4. Not tighter than 60s on
purpose: a real hop with 39 schemas and a long history legitimately
takes 20-30s.

Verified with no microphone connected — which *is* the permanent-fault
case, and normally the hardest state to reproduce on purpose. Before:
`PortAudioError`, process dead, no explanation. After: one traceback,
four one-line repeats, clean stop naming the cause and pointing at the
diagnostic. The LLM timeout is test-verified only.

Still unaddressed: cross-session `history` reset; telling the model
when it was interrupted (see phaseEight/README.md's milestone 3 gaps);
a long-session soak test of the character, since four probes cannot
show whether the dryness wears thin over an hour; and a voice
round-trip for milestone 5 and for device pinning, both of which need a
microphone connected.

**Flagged, not fixed — `type_text` is ungated.** `phaseSix/tools.py`'s
`type_text` takes no `target_window_contains` and types into whatever
has focus, unlike `click`/`press_hotkey` which both require one (added
"after a real incident"). `pyautogui.write()` treats `\n` as Enter, so
it can submit, not just type. That was decided ungated when nothing
could change focus; milestone 2 added `focus_window`. Worth a
deliberate decision.