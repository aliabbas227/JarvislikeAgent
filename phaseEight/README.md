# Jarvis — Phase 8: Polish

Phase 7 ended with all four of its milestones complete and confirmed
against real hardware. Phase 8 is the roadmap's "optional but
Jarvis-feeling" polish phase, plus the three items Phase 7's own README
listed under "Not yet covered."

Unlike Phases 1–6, Phase 8 is **not** a standalone mini-project. It is
a layer on top of a finished Phase 7: `app.py` imports
`phaseSeven/orchestrator.py` and extends it through two small seams,
rather than forking or reimplementing the voice loop. That loop carries
several hard-won real-hardware fixes (`Model.reset()` between turns,
the restart settle delay, the wake chime, the split pre-speech
timeout); copying it up here to customize it would have meant owning
those fixes in two places.

Same staged-milestone discipline as Phase 7:

1. **Milestone 1 (done):** accuracy — a real clock, real weather,
   a better-provisioned web search, and a date-aware system prompt.
2. **Milestone 2 (done):** tool expansion — window and process control,
   clipboard, media, power, file search, archives, document reading.
   15 tools → 35.
3. **Milestone 3 (done):** barge-in — say the wake word while Jarvis is
   talking and it stops talking.
4. **Milestone 4 (done):** personality and voice — the film Jarvis
   character, spoken in a British neural voice instead of pyttsx3.
5. **Milestone 5 (done):** seeing the screen — accessibility-tree
   grounding, OCR with positions, and a local vision model. 35 tools → 39.

## Files

| File | What it is |
|---|---|
| `app.py` | Entry point. Imports Phase 7's orchestrator, registers everything, runs. |
| `tools_extra.py` | Milestone 1: clock, real weather, upgraded search. |
| `os_tools.py` | Milestone 2: windows, processes, clipboard, media, power. |
| `file_tools.py` | Milestone 2: file search, organising, archives, documents. |
| `gate.py` | Milestone 2: Phase 8's own two-phase confirmation registry. |
| `interrupt.py` | Milestone 3: barge-in — wake-word listener + interruptible speech. |
| `voice.py` | Milestone 4: neural TTS (edge-tts) with an offline fallback. |
| `screen.py` | Milestone 5: UI grounding, OCR with positions, local vision. |
| `calibrate_barge_in.py` | Diagnostic: can barge-in actually hear you? |

## Running it

```
cd phaseEight
.\venv\Scripts\Activate.ps1
(Get-Command python).Path   # verify it points into phaseEight\venv
python app.py
```

`.env` is optional here (see `.env.example`). API keys, `CONFIRM_PASSWORD`,
and audio tuning stay in `phaseSeven/.env` and are inherited — see
"Two `.env` files, deliberately" below.

### One-time venv setup gotcha (hit for real)

**openWakeWord's model files do not come with the pip package**, and a
fresh venv therefore has the library but not the weights. Installing
`requirements.txt` was not enough: `app.py` crashed at startup with
`NO_SUCHFILE ... hey_jarvis_v0.1.onnx failed. File doesn't exist`,
because `build_model()` looks for the weights *inside the venv's own*
`openwakeword/resources/models/` directory — and phaseSeven's copy is
in phaseSeven's venv.

This is precisely the bug class the root `CLAUDE.md` already warns
about ("a wrong-venv mixup has already caused real bugs here — wrong
`openwakeword` install location"), showing up again the moment Phase 8
got a venv of its own. Fixed once, per venv:

```
python -c "import openwakeword.utils as u; u.download_models()"
```

Worth knowing that nothing in the mocked test suite catches this — both
suites stub `openwakeword` via `sys.modules` specifically so they can
run without the real package, which means they also run happily without
its weights.

## Testing it

```
pytest
```

---

## Milestone 1: the accuracy pass

### What prompted it

Three complaints, in the user's own framing: the time is wrong, search
feels outdated, and "check if there are tool conflicts."

The third turned out to be the accurate diagnosis, and it was worse
than a conflict.

### What the diagnostic actually showed

Before changing anything, the merged Phase 4 + Phase 6 registry was
probed against the **live** DeepSeek API (not mocked) with two of the
most ordinary questions anyone asks a voice assistant:

| Asked | Tool the live model called |
|---|---|
| "What time is it right now?" | `get_active_window` |
| "What is today's date?" | `get_weather(location="Today")` and `read_screen()` |

Neither is random. **There was no time or date tool anywhere in the
registry** — 14 tools across Phases 4 and 6, not one of them a clock.
Given a question it has no tool for, the model reaches for whatever
sounds adjacent and narrates a plausible answer out of whatever comes
back. `get_weather(location="Today")` is the clearest tell: it is not
a tool misfire so much as the model trying to make *something*
answer a question nothing could.

That compounds with the second finding: **Phase 4's `get_weather` is
stub data** — hardcoded `"partly cloudy"`, `21°C`, for every location
on earth. It carries a `note` field admitting it, but the model does
not reliably relay that admission, so a weather question yields a
confident invented forecast. So "what's today's date" was being
answered out of fabricated weather for a place called "Today."

The third finding was the useful negative result: **`search_web` was
never the outdated part.** The same diagnostic ran a live Tavily query
and got correctly current results back. What was thin was what Phase 4
*asked* for — 3 results, 300-character snippets, no synthesized answer,
no recency control — and, more importantly, the model had no idea what
today's date was, so it could neither judge whether a result was fresh
nor write a date-anchored query in the first place.

### What changed

**`get_current_time` (new).** Reads the machine's own system clock.
Returns the same moment in several forms (ISO, 24h, 12h, weekday, a
spoken form, timezone, UTC) — not padding: making the model reformat a
single ISO string is a needless place for it to make an arithmetic
mistake, and a spoken reply and a date calculation want different
shapes. The tool description tells it explicitly that it has no other
way to know the time.

**`get_weather` (overrides Phase 4's).** Real current conditions via
Open-Meteo. Chosen over key-gated services specifically because it
needs no API key and no signup — worth optimizing for when the whole
point is to stop a tool being a stub with the least new setup burden.
Two calls: geocode the place name, then fetch conditions. It reports
the **resolved** place name alongside the numbers, so a geocoding
near-miss ("Cambridge" landing in the wrong country) is visible rather
than silently changing what the numbers mean.

**`search_web` (overrides Phase 4's).** Same Tavily service, asking for
considerably more: `include_answer="advanced"` (a synthesized answer
across results — the single biggest upgrade, since the alternative is
the model stitching an answer out of truncated snippets and filling the
seams itself), `search_depth="advanced"`, an optional `time_range` the
model can set for genuinely time-sensitive questions, 5 results instead
of 3, 600-character snippets instead of 300, and `published_date`
passed through when Tavily supplies it. Auth moved from Phase 4's
request-body `api_key` to the `Authorization: Bearer` header, Tavily's
current documented form — confirmed working live.

**A date-aware system prompt, rebuilt every conversation.** Phase 7's
`TOOL_SYSTEM_PROMPT` is kept verbatim and appended to, not replaced —
everything it says about the OS tools, the confirmation gate, and
speaking rather than writing is still exactly right. Phase 8 adds
guidance about the new tools plus the actual current date and time.

The clock line is in the prompt *even though `get_current_time`
exists*, because the two solve different halves of the problem. The
tool answers "what time is it" when asked. The prompt line is what lets
the model notice, unprompted, that a search result or its own training
data is stale, in a turn where nobody asked about time at all.

It is rebuilt **per conversation** rather than once at import
specifically because this thing is meant to sit running for days. A
date baked in at startup is wrong by the next morning — which is worse
than saying nothing, because the model has no reason to distrust it.

### Bug found in Phase 7 along the way

**The system prompt was being silently dropped from long sessions.**
`jarvis.trim_history()` (reused as-is from Phase 1 since Phase 7's
milestone 2) is a bare `messages[-MAX_HISTORY_MESSAGES:]`. It has no
idea the first message is special. So the moment a session passed 40
messages, `TOOL_SYSTEM_PROMPT` fell off the front along with the oldest
turn, and **every subsequent call went to DeepSeek with no system
message at all** — no "your replies are spoken aloud, avoid markdown,"
no "read_screen is OCR not vision," no confirmation-gate framing, none
of the tool guidance the prompt exists to carry.

The session keeps working, just progressively worse, in a way that
presents as the model randomly getting dumber rather than as a bug.
That makes it a plausible contributor to the original "I get inaccurate
information all the time" complaint, independent of the missing clock.

Fixed in `phaseSeven/orchestrator.py` (not here) with
`_trim_history_preserving_system()`, since it is a genuine correctness
bug in Phase 7's own code rather than something Phase 8 layers over.
The message budget is now `MAX_HISTORY_MESSAGES` conversational
messages *plus* the system one.

This is the same class of bug as Phase 7's milestone 2 note about
`trim_history()` not being tool-call-aware — and **that sharp edge is
still there, deliberately untouched**: a trim landing between an
assistant message requesting a tool call and the tool-role message
answering it still produces a history DeepSeek rejects. Worth a
tool-aware trimmer eventually; not folded into an accuracy fix.

### Design notes

**Nothing in Phase 4 was edited.** `get_weather` and `search_web` are
replaced at the *registry* level (`orchestrator.register_tools`), not
in `phaseFour/tools.py`. Phase 4 is a finished standalone mini-project
whose own CLI still has to work as it always did, and overriding from
above keeps the phase boundary intact and the change reversible in one
line. An override replaces the schema as well as the function —
leaving Phase 4's `"Get the current weather for a given location."` in
place would keep pointing the model at a contract the implementation no
longer matches.

**Two seams were added to Phase 7**, both no-ops when unused, both
documented in `orchestrator.py`:

- `register_tools(schemas, functions)` — add or replace tools by name.
  It raises immediately if the schema names and function names disagree:
  a schema with no function behind it is a tool the model can call and
  nothing can answer, and a function with no schema is dead code the
  model can never reach. Neither fails loudly on its own at runtime.
- `run_orchestrator(system_prompt_provider=...)` — called for a fresh
  system prompt at the start of every wake-triggered conversation.

Adding seams to a finished phase is a judgment call, made deliberately:
the alternative was either reaching into `orchestrator`'s private
`_TOOL_FUNCTIONS` from another module, or copying the wake-gate loop
(and its real-hardware bug fixes) up into Phase 8 to customize it.
Both are worse.

**Two `.env` files, deliberately.** `app.py` loads both
`phaseEight/.env` and `phaseSeven/.env` itself, by absolute path,
before importing `orchestrator`. python-dotenv never overrides an
already-set variable, so Phase 8's `.env` wins wherever it says
anything and `phaseSeven/.env` fills in the rest (API keys,
`CONFIRM_PASSWORD`, audio and wake tuning). Nothing depends on the cwd
or on how the process was launched. See the bug below for why it does
this itself rather than leaving `phaseSeven/.env` to `orchestrator`'s
own `load_dotenv()`.

### Second bug found, in Phase 8's own wiring

**`CONFIRM_PASSWORD` silently came back unset, downgrading the
destructive-action gate.**

The first version of `app.py` loaded only `phaseEight/.env` explicitly
and left `phaseSeven/.env` to `orchestrator`'s own module-level
`load_dotenv()`. A bare `load_dotenv()` resolves via `find_dotenv()`,
which walks up from the *calling file's* directory — except when it
can't identify a calling file, in which case it **silently falls back
to the current working directory**. Launched through anything without a
real `__main__.__file__`, the fallback searched upward from wherever
the process started, found no `.env` at the project root, and left
`CONFIRM_PASSWORD` unset.

That does not crash. `orchestrator` logs a warning and **falls back
from the password gate to a plain yes/no terminal prompt** — a real
weakening of the one boundary Phase 7's milestone 4 was most careful
about, arrived at silently, from a launcher detail.

**Fixed** by loading both `.env` files here, by absolute path.
Re-verified from three different working directories (project root,
`phaseEight/`, and an unrelated directory): `CONFIRM_PASSWORD`,
`TAVILY_API_KEY`, `DEEPSEEK_API_KEY` all resolve, and the ChromaDB
store resolves to the shared `phaseFive/chroma_store` with its real
memories in all three.

This is the **third** appearance of this same bug class in this
project — Phase 5's README documents it for a wrong-cwd case, Phase 7's
milestone 3 hit it through a cross-phase import, and this is a
cross-phase import one layer deeper again. It is also, all three times,
structurally invisible to the mocked suite, which never triggers a real
`load_dotenv()`. The pattern is now clear enough to state as a rule:
**in this project, every cross-phase `.env` load gets an absolute
path.**

**The registry is process-global module state**, so every test that
calls `register_tools` snapshots and restores it. Skipping that leaks
an overridden `get_weather` into whatever test runs next.

### Verified for real (not mocked)

Per this project's standing rule that a green mocked suite proves the
logic and nothing else:

- **`get_current_time`** — returns real local time, correct weekday,
  correct timezone and UTC offset.
- **`get_weather("Toronto")`** — live Open-Meteo call returned
  `Toronto, Ontario, Canada`, `20.3°C`, `overcast`, real humidity and
  wind. No API key needed, as designed.
- **`search_web`** — live Tavily call with Bearer auth, `time_range="week"`,
  `topic="news"`: returned a synthesized `answer` (a field Phase 4's
  version never produced at all), current results, and `published_date`
  values.

- **The whole layer, end to end against live DeepSeek**, re-running the
  exact diagnostic that started this milestone:

  | Asked | Before | After |
  |---|---|---|
  | "What time is it right now?" | `get_active_window` | `get_current_time` → *"It's currently 3:27 AM on Tuesday, September 1, 2026."* |
  | "What is today's date?" | `get_weather(location="Today")`, `read_screen` | `get_current_time` → correct date |
  | "What's the weather in Toronto?" | stub: partly cloudy, 21°C | real: overcast, 20.3°C, 97% humidity, 5.1 km/h wind |

  All three replies came back conversational and markdown-free, so the
  Phase 8 prompt additions did not displace Phase 7's spoken-output
  guidance.

- **Configuration resolves identically from three different working
  directories** (project root, `phaseEight/`, an unrelated directory) —
  see the `CONFIRM_PASSWORD` bug above for why this was worth checking
  rather than assuming.

**Still outstanding for this milestone:** a real voice round-trip —
saying "what time is it" out loud to a running `app.py`. Everything
downstream of transcription is now confirmed against the live API, but
every phase of this project so far has found at least one bug that only
appeared with a real microphone in the loop.

---

## Milestone 2: tool expansion

### What prompted it

"I feel it doesn't have enough tools or they aren't powerful enough
right now — the goal is to have an agent that can do anything I need
it to."

Concretely, Phase 6 gave Jarvis hands but no eyes for its own machine.
It could launch an app, read the screen, click, and type — but it could
not know which windows were open, bring one to the front, close one,
see what was running, stop a runaway process, reach the clipboard, or
find a file. Every "where did I save that" was unanswerable unless the
model already knew the exact path, and every window action had to be
driven through screen OCR and blind hotkeys, which is the fragile path
Phase 6's own system prompt warns it away from.

### What was added — 20 tools, 15 → 35

**`os_tools.py`** — `list_windows`, `focus_window`, `move_window`,
`close_window`, `list_processes`, `kill_process`, `get_clipboard`,
`set_clipboard`, `media_control`, `take_screenshot`,
`get_system_status`, `lock_screen`, `power_action`.

**`file_tools.py`** — `find_files`, `create_directory`, `move_path`,
`copy_path`, `extract_archive`, `get_disk_usage`, `read_document`.

All local. No new API keys, no network. Windows-first via ctypes
against `user32` rather than adding pywin32, matching the choice Phase
6 already made for `get_active_window`. New pip dependencies: `psutil`,
`pypdf`, `python-docx` (`pyperclip` came in with pyautogui already).

### The guardrail split

Stated explicitly rather than assumed, the practice Phase 5's handoff
recommended and Phase 6 followed:

| Ungated | Conditionally gated | Always gated |
|---|---|---|
| `list_windows`, `focus_window`, `move_window`, `list_processes`, `get_clipboard`, `set_clipboard`, `media_control`, `take_screenshot`, `get_system_status`, `lock_screen`, `find_files`, `create_directory`, `get_disk_usage`, `read_document` | `move_path`, `copy_path`, `extract_archive` — immediate when the destination is free, gated when they would replace something existing | `close_window`, `kill_process`, `power_action` |

The conditional column is not a new invention: it is exactly the rule
Phase 6 set for `write_file` (creating is safe, replacing is not),
applied to moves and copies. A move to a free path is reversible — move
it back. A move onto an existing file is not.

Three ungated calls needed their reasoning written down:

- **`lock_screen`** is the one power action that destroys nothing (you
  log back in and everything is where you left it), and gating it is
  self-defeating — the point is saying "lock my machine" while walking
  away, and a gate would require staying to type a password in order
  to lock.
- **`media_control`** is ungated even though Phase 6 gates hotkeys,
  because it is not a hotkey tool. It sends one of six fixed media keys
  from a closed allowlist, none of which can close a window or quit an
  app. Phase 6 gated `press_hotkey` because it accepts *arbitrary*
  combinations; that reasoning doesn't transfer to a fixed safe set.
- **`set_clipboard`** matches Phase 6's judgment that typing is
  lower-risk than clicking. It does discard what was on the clipboard
  before — a small real loss, noted, judged not worth a password.

### Phase 8 needed its own gate

Phase 6's `confirm_pending_action()` dispatches on a hardcoded
if/elif chain over its own four action types (delete, overwrite, click,
hotkey). It has no idea what "kill this process" means, and extending
it would mean editing a finished phase.

So `gate.py` implements the same guarantee generically: it holds a
**Python closure** keyed by a token, in this module's own memory. The
model never sees it, is never handed a path to it, and cannot call it.
It can only ever get as far as a proposal that a human approves
out-of-band — at a real terminal, with a real password, on a channel
neither the model nor the microphone can reach.

Holding a closure rather than a re-parseable description is deliberate:
the action that eventually fires is the exact one that was proposed,
with no second interpretation step between what the human read and what
happens, and therefore no gap for the arguments to drift.

Tokens are prefixed `p8-` so `orchestrator._resolve_confirmation()` can
route each confirmation to the registry that issued it. Routing is by
prefix rather than "try each registry until one doesn't say invalid
token" — the latter would make the gate's behavior depend on registry
ordering and error-string matching, which is not a property to want
anywhere near this particular boundary. **Phase 6's tokens are
untouched and still route to Phase 6.**

That required a third Phase 7 seam, `register_confirmation_handler`.
`app.register_everything()` registers the handler in the same call as
the tools, on purpose: registering the tools alone would leave Phase 8's
tokens routing to Phase 6's registry, which would correctly reject them
— so every gated Phase 8 action would fail closed at the moment of
confirmation. Failing closed is the right direction, but it would make
the tools quietly useless.

### Two guards beyond the gate itself

Both defend against the same thing: an identifier meaning something
different at confirmation time than it did at proposal time.

- **Stale window handles.** Between proposal and confirmation a window
  can close and Windows can hand its handle to a completely different
  window. `close_window` re-reads the title at confirmation and aborts
  if it changed — the same class of check as Phase 6's
  `_abort_if_focus_shifted`.
- **PID reuse.** Same story for processes: the target exits, the OS
  reassigns its PID, and confirming would kill something unrelated and
  newer. The process's creation timestamp is captured at proposal and
  re-checked at confirmation.

Plus two refusals that never become proposals at all:

- **Protected system processes** (`lsass.exe`, `winlogon.exe`,
  `csrss.exe`, …) are refused outright rather than gated. Killing them
  bluescreens or logs out the machine immediately, there is no
  legitimate voice-assistant reason to reach for them, and a
  confirmation prompt is not a good place to be discovering that.
- **Zip Slip.** `extract_archive` refuses any entry with an absolute
  path or a `..` component, which would otherwise write outside the
  destination over any file the process can reach. Python's own
  `extractall()` guards this too; the check is explicit anyway, because
  the input is an untrusted file and the caller is a language model.

**Ambiguity is refused, never guessed.** A title matching two windows,
or a name matching two processes, comes back as an error listing the
candidates so the model can ask which one. Picking the first match
would be right often enough to be trusted and wrong often enough to
close the wrong thing — and `close_window` is usually what follows a
resolution.

### Bug found by live verification

**The "no markdown" instruction stopped working.**

Phase 7's `TOOL_SYSTEM_PROMPT` ends with "no markdown, bullet points,
or numbered lists" — which worked while it genuinely was the end. Phase
8 appends several hundred words after it, and the very first live probe
of milestone 2 ("what windows do I have open?") came back as a markdown
bulleted list. Read aloud, that is a TTS voice saying "dash" ten times.

Milestone 2 makes this materially worse rather than incidentally so:
almost everything it added (`list_windows`, `list_processes`,
`find_files`) returns a **list**, which is the shape a model most wants
to format as bullets.

**Fixed** by restating the constraint as `_SPOKEN_OUTPUT_REMINDER` and
placing it last in the assembled prompt, after the Phase 8 additions
and the date line. Re-verified live: both list-shaped probes now come
back as natural spoken prose. A test pins the *position*, since
position is the fix.

The general lesson, worth carrying into milestone 3: appending to a
system prompt is not free. It silently demotes whatever used to be
last.

### Verified for real (live DeepSeek, not mocked)

Read-only tools were exercised against the real machine first —
`list_windows` returned 10 real windows, `list_processes` 315 real
processes, `get_system_status` real load, `get_disk_usage` and
`find_files` real results, `get_clipboard` real content.

Then the full loop against live DeepSeek, **with every confirmation
auto-declined so nothing could fire**. From 35 available tools the model
picked correctly every time:

| Asked | Tools it chose |
|---|---|
| "What windows do I have open?" | `list_windows` |
| "How is my PC doing?" | `get_system_status` |
| "Find any markdown files under …" | `find_files` |
| "How much free disk space on A?" | `get_disk_usage` |
| "Close the Spotify window." | `list_windows` → `get_active_window` → `close_window` |

The last one is the one that mattered: the gate engaged through the
real tool-calling loop, printed the real confirmation message ("About
to close the window 'Spotify Premium'"), the decline was honoured, and
**Spotify stayed open**. The model then reported the cancellation
accurately rather than claiming success.

The gate was also exercised directly, outside the model: a `move_path`
overwrite proposed and cancelled left the target file untouched;
confirmed, it moved; the token could not be reused; an expired token
was refused with a message distinguishable from an invalid one; and a
`power_action("shutdown")` proposal did not shut the machine down.

### Confirmed by voice, by the user, for real

Both milestones have now had the real-hardware pass this project
insists on. Running `app.py` with a real microphone and speakers: the
clock, weather, window-listing, and system-status questions all
answered correctly through the voice loop, replies came back as spoken
prose rather than markdown read aloud, tool selection stayed accurate
across all 35 tools, and the gated `close_window` round-trip worked end
to end — spoken heads-up, real `CONFIRM_PASSWORD` prompt at the
terminal, correct outcome spoken back.

No new bugs surfaced. That is the first time in this project that a
real-hardware pass has found nothing, and it is worth not
over-reading: the two bugs this milestone did produce (the markdown
regression, the missing wake-word weights) were both caught *before*
the voice test, by live API probing and by a pre-flight check
respectively. The pattern holds — it was the non-mocked runs that found
them, just earlier in the chain than usual.

---

## Milestone 3: barge-in

Say **"hey Jarvis"** while Jarvis is talking and it stops talking.

Detection is **openWakeWord**, the same engine Phase 3 already uses for
waking. The first implementation transcribed candidate audio with
Whisper and matched the word "skip"; it was replaced after six genuine
bugs were fixed and it *still* did not work. That whole history is kept
below, because the bugs were real and the reason it was abandoned is
more useful than the fact that it was.

This is the first thing in the project that genuinely needs two things
happening at once — which Phase 7 anticipated in its own module
docstring ("worth revisiting once a later milestone needs something
running *while* still listening, e.g. a spoken 'stop' during a long TTS
reply"). This is that milestone.

Two design decisions here were settled by measurement, because both had
an obvious-looking answer that turned out to be wrong.

### Finding 1: cross-thread `engine.stop()` silently does nothing

*(This one still holds — it is why speech is chunked regardless of detector.)*

The natural design is to run TTS on a background thread and call
`engine.stop()` when an interrupt arrives. Probed directly against this
machine's SAPI5 stack, that call:

- raised no exception,
- returned cleanly,
- **did nothing at all**, and
- left the speaking thread wedged — still running 13.5s later on a
  phrase that takes 10s to say uninterrupted.

A silent no-op that also corrupts the engine is the worst possible
thing to build on. So **all pyttsx3 work stays on one thread**: the
reply is split into sentences, spoken one at a time, and the interrupt
flag is checked between them. Interrupting means "don't start the next
sentence", not "cut off this one".

The cost was measured rather than assumed: three sentences spoken
separately took **8.94s against 8.50s** as one utterance — roughly
**0.2s of extra pause per sentence boundary**, which is not noticeable
in practice.

A first attempt at measuring that appeared to show chunking taking
0.24s total. That was not a speedup, it was silence: `pyttsx3.init()`
returns a **cached** engine when a live one still exists, and the
reused engine produced no audio whatsoever. Phase 2's `speak_text()`
already avoids this by letting each engine fall out of scope between
calls — which is exactly why `interrupt.py` calls `speak_text()` rather
than driving pyttsx3 itself.

### Bugs found on the first real voice test (barge-in didn't fire at all)

Saying "skip" did nothing whatsoever. Three separate faults, found by
instrumenting rather than guessing — and the first one made the other
two invisible.

**1. A fixed threshold that this machine could never reach.** The
detector required 2.5× `SILENCE_THRESHOLD` (= 1250 RMS). Measured on
the actual hardware:

| | peak RMS |
|---|---|
| ambient silence | 20 |
| while audio played | 143 |
| threshold required | **1250** |

Nothing came within an order of magnitude, so **no audio ever reached
the detector** — not the echo, and not the person either. The cause was
mundane and visible in one line: the default input is a *controller headset
controller headset* and the default output is a *different headset*
(Wireless Headset). Two separate devices, so Jarvis's voice barely reaches
that mic — and neither does much else.

The deeper mistake was the constant itself. It was tuned for the
open-mic case (assume echo floods the mic, so demand a person be louder
still), which makes it far too **high** for a split-device setup and
far too **low** for a genuinely echoing room. No constant serves both.

**Fixed** by deriving the threshold live from what the mic is actually
hearing: a low percentile of recent levels, times a ratio, with a floor
so it can't chase the noise down into silence. Quiet setup → low
threshold; echoing room → high threshold; no configuration either way.

**2. The buffer-drop heuristic discarded exactly the case barge-in is
for.** Audio that stayed loud for 2 seconds was thrown away as
"continuous sound, therefore Jarvis". But a person talking *over*
Jarvis produces continuous loud audio — that is what barge-in *is*. It
is now a flush interval, not a bin.

**3. The background estimate deadlocked in a noisy room.** Found by
simulating an open mic rather than waiting to hit it. The window was
fed only chunks *below* the current threshold, on the reasoning that
those are the quiet ones. If the room's own floor sits above the
minimum threshold, no chunk is ever quiet enough to record — so the
estimate stays empty, the threshold stays pinned at the floor, every
chunk counts as speech forever, and the thing silently transcribes
continuous noise and rejects all of it. Fixed by feeding every level in
and taking a low percentile, which measures the same floor without the
circularity.

Also added: **`calibrate_barge_in.py`**, following
`phaseSeven/calibrate_mic.py`. The whole reason fault 1 was hard to see
is that "didn't hear the word", "heard it and rejected it", and "heard
nothing at all" were indistinguishable from the outside. The listener
now also warns when a whole reply passed with nothing above the floor,
and names the device it was listening to.

### The finding that outranks all three: the default input device moves

Across three measurement runs on the same machine, minutes apart,
Windows' default input device was:

| Run | default input | peak RMS | default output |
|---|---|---|---|
| 1 | Headset Microphone (Controller Headset) | 143 | Headphones (Wireless Headset) |
| 2 | Headset Microphone (Controller Headset) | 20 | Headphones (Wireless Headset) |
| 3 | Microphone (Wireless Headset) | **1** | Display Audio (monitor, over HDMI) |

Nothing in the code changed between them. Windows reassigns the default
as headsets connect, disconnect, and power down, and every phase of this
project opens `sd.InputStream()` with **no device argument** — so the
entire voice pipeline silently follows whatever that default happens to
be, including onto a headset that is switched off and reads a peak of 1.

That is a bigger problem than barge-in. It affects wake-word detection
and ordinary recording identically, and it presents as "Jarvis stopped
hearing me" with no error anywhere. It is also the most likely reason
measurements taken at different times disagreed.

**For an always-on open mic this has to be pinned.** Setting the
intended microphone as the Windows default (Sound settings → Input →
set as default device, and disable the ones that shouldn't win) fixes it
for every phase at once, which is why it is recommended over adding an
`INPUT_DEVICE` setting to Phase 8 alone — that would only pin barge-in,
leaving the wake word and the turn recorder still following the default,
and a listener on one device while the recorder is on another is its own
new class of bug.

### Finding 2 (Whisper era): the loudness gate alone is not enough

The obvious cheap approach is pure voice-activity detection: any
sustained sound above threshold stops playback. The objection is that
the mic hears Jarvis's own voice through the speakers. That objection
was worth testing rather than assuming, so it was measured.

With the interrupt threshold set to 2.5× the normal speech threshold
(1250 vs 500) and a **headset** mic — the setup most favourable to
echo rejection — a ~12-second reply produced **7 separate occasions**
where Jarvis's own voice crossed the threshold and reached the
transcription stage.

**All 7 were correctly rejected**, because detection requires the
actual word. Pure VAD barge-in would have interrupted itself within a
second or two, every single reply.

So detection is: loudness gate → short-utterance cap → Whisper → the
transcript must actually be an interrupt phrase. Two filters keep the
last step honest:

- a **word-count cap**. "Skip" is one word said deliberately; a whole
  sentence is Jarvis's own speech being picked up, and is rejected even
  if the word appears inside it.
- **whole-word matching**, so "skip" is not found inside "skipping" or
  "skipper".

Together these are what stop Jarvis saying *"I'll skip the rest of the
forecast"* from interrupting Jarvis — verified for real, with that
exact sentence, out loud.

### "skip" deliberately does not reuse Phase 7's exit words

`stop` and `quit` end the whole session in Phase 7. Overloading them so
they mean "end the session" in silence but "stop talking" during speech
would make the most destructive command in the loop depend on timing
the person cannot see. The interrupt set is its own: `skip`, `skip it`,
`nevermind`, `never mind`.

### What happens after an interrupt

Nothing dramatic: Jarvis stops talking, and the turn ends normally, so
Phase 7's follow-up window opens right afterward and you can just say
what you actually wanted. **This required no changes to the
conversation loop at all** — being cut off looks the same to it as
finishing early.

### Short replies skip the whole apparatus

Below `BARGE_IN_MIN_CHARS` (80), text is spoken exactly as before, with
no mic stream and no transcription. A four-word confirmation is over
before anyone could react to it. This is also what keeps "Goodbye." and
the confirmation heads-up out of the interruptible path without
`interrupt.py` needing to know anything about its call sites.

### Design notes

**A fourth Phase 7 seam, `register_speaker`.** `speak_safe()` now
delegates to a registered speaker if there is one. Barge-in is a
different way of *speaking*, not a different thing being said, so every
caller still just calls `speak_safe()`. The `SpeechError` fallback
stays wrapped around whatever is registered — the reason it exists (a
dead audio driver must never kill the loop) applies at least as much to
a speaker with a microphone and a transcription model behind it.

**Fails soft, always.** A mic that won't open, a failed transcription,
an audio driver that disappears — all degrade to "no barge-in for this
reply", never a crashed voice loop. Same distinction Phase 7 drew for
the memory store: an enhancement fails quiet, a load-bearing component
fails loud.

**`stop()` joins the listener thread.** Not optional: the next thing
the orchestrator does after speaking is open its own `InputStream` for
the follow-up window, and two streams on one device is exactly the
class of problem Phase 7's milestone 2 spent three real-hardware
debugging passes on.

**`interrupt.py` resolves phaseTwo on `sys.path` itself.** It first
relied on `app.py` having already imported `orchestrator`, which does
that as a side effect — which happened to work only because `app.py`'s
imports are alphabetical and `orchestrator` sorts before `interrupt`. A
module that breaks when its importer's import block is reordered is not
a dependency worth having, and it also made `interrupt.py` impossible
to import on its own in a test.

### Verified for real, with the shipped openWakeWord detector

Real TTS, a real microphone stream and the real model, two runs:

| Run | Result |
|---|---|
| Control — nothing said | spoke fully, **16.4s**, no false trigger from Jarvis's own voice |
| "hey Jarvis" fed in at 2s | **4.3s** — detected at score **0.993**, stopped after 1 of 4 chunks |

The wake word was mixed into the microphone stream rather than played
through the speakers, because playback on this machine goes to
headphones a desk mic cannot hear — the acoustics cannot be simulated
here, but everything downstream of the microphone is real.

**Then confirmed by voice, by the user, for real:** saying "hey Jarvis"
out loud during a spoken reply stops it. That closes the one thing
neither the mocked suite nor a self-test could establish — a person
actually saying it. Remaining gap is open speakers rather than a
headset (see known gaps).

### Also verified during the Whisper era (still applicable)

- **Chunking** splits a real weather reply into four natural sentences.
- **A full 16.1s reply spoke to completion** with a real microphone
  listener running and real Whisper transcription in the loop — no
  deadlock, no stutter, no self-interruption.
- **The self-interrupt guard held under real echo**, on a reply
  deliberately ending with "I'll skip the rest of the forecast".
- **7 echo events reached transcription and all 7 were rejected** (see
  Finding 2).
- **The microphone was released afterward** — a subsequent
  `record_until_silence()` opened its own stream cleanly, which is what
  the follow-up window does immediately after every reply.

**Still outstanding:** someone actually saying "skip" out loud. Every
mechanism around it is confirmed, but the one thing a mocked suite and
a self-test structurally cannot do is be a person talking.

### Tuning it

| Variable | Default | What it does |
|---|---|---|
| `BARGE_IN_ENABLED` | `true` | Set false to restore Phase 7's plain speech exactly. First thing to turn off if speech misbehaves. |
| `BARGE_IN_THRESHOLD` | `WAKE_THRESHOLD` (0.5) | Confidence needed to count as the wake word. Separate from waking, so interrupting can be made harder without making Jarvis harder to wake. Measured margin is 0.999 vs 0.000, so this rarely needs touching. |
| `BARGE_IN_MIN_CHARS` | `80` | Replies shorter than this are spoken without a listener. |
| `MAX_CHUNK_CHARS` | `220` | Long sentences are split further, so one run-on sentence can't block interruption. |

### How well this works on an open mic — measured, not guessed

The stated goal is eventually running on an open microphone with
speakers, where Jarvis's own voice reaches the mic loudly. That case
was simulated properly rather than estimated: Jarvis's real rendered
reply and a real rendered "skip" were mixed the way one microphone in a
room would hear them, and the person's loudness was swept against a
fixed echo level.

| Person's voice vs Jarvis's, at the mic | Result |
|---|---|
| 0.5× (quieter than Jarvis) | missed |
| 1.0× (equally loud) | missed |
| 2.0× louder | missed |
| **3.0× louder (~10 dB)** | **detected** |
| 5.0× louder | detected |

**So: on an open mic you have to talk over Jarvis by roughly 10 dB.**
On a headset — where the echo is tiny — it works easily, which matches
the live check on this machine.

The limit is not the threshold, it is Whisper: given two overlapping
voices it locks onto the dominant one, so below ~3× the person's word
simply isn't in the transcript to match. Tuning cannot move this much.

Getting genuinely robust open-mic barge-in means changing the detector,
not the settings. The two real options:

- **A keyword spotter instead of transcription.** openWakeWord is
  already a dependency, already proven on this machine, and is built
  exactly for spotting one word in continuous noisy audio with low
  latency. There is no pre-trained "skip" model, but there *is*
  `hey_jarvis` — so barge-in would be triggered by saying "hey jarvis"
  again, which is also how commercial assistants do it. This is the
  cheap, high-value option.
- **Acoustic echo cancellation.** The real fix, and much more work:
  subtract the known playback signal from the mic input so Jarvis's own
  voice is removed before detection.

### Second round of fixes, after "skip" still didn't register

A calibration run on a working headset (Wireless Headset, input and output the
same device) gave the first solid numbers:

| | peak RMS | median RMS |
|---|---|---|
| background | 53 | 1 |
| Jarvis's own voice at the mic (echo) | 132 | 51 |
| the person speaking | 1166 | **153** |

Two more real faults fell out of that, plus a fix to the tool itself:

**4. The clip was truncated to the loudest moment of the word.** With a
threshold of 500 and speech whose *median* is 153, only the peak of a
word clears the bar — and the listener buffered strictly
above-threshold audio, so Whisper received the middle of "skip" with
its onset and tail removed. Fixed with a **pre-roll** (0.3s kept from
before the crossing) and a **hangover** (0.4s of continued recording
through dips and after), which is standard VAD practice and was simply
missing. Verified: clips went from ~0.1s fragments to a coherent 0.9s.

**5. The threshold floor was tuned for the wrong question.** It
defaulted to `SILENCE_THRESHOLD` (500), which answers "is anyone
talking at all" — a different question from "did someone say one short
word over the top of Jarvis". Set to **264** for this machine (2× the
132 echo peak), and `calibrate_barge_in.py` now computes and prints
that number rather than leaving it to guesswork.

**6. The calibration tool's own device check was wrong.** It compared
the first word of each device name — always "Microphone" vs
"Headphones" — so it told someone with a single matched headset that
they had a split-device setup. Now compares the hardware name inside
the parentheses.

### Why this approach was abandoned: Whisper is the wrong tool

Even with all six faults fixed and the clip captured correctly, the
detector still failed in simulation at this machine's own measured
levels. The captured audio was correct — dumped, inspected, and
transcribed perfectly as `'Skip.'` when handed to Whisper on its own —
but the same word inside the listener's captured window came back as
`'Quick TENSCHES Okoi'`, and lengthening the window to 1.9s did not
help.

That is not a tuning problem. Whisper is an utterance transcriber, and
short, quiet, overlapping clips are precisely its weak spot. The
earlier open-mic sweep already said the same thing quantitatively: a
person needs to be ~3× louder than Jarvis before their word survives
into the transcript at all.

**openWakeWord, already installed and already proven in Phase 3, is
built for exactly this job.** Tested on the identical mixture, at the
identical levels:

| Detector | word over Jarvis's echo | Jarvis's speech alone |
|---|---|---|
| Whisper | garbled (`'Quick TENSCHES Okoi'`) | correctly rejected |
| **openWakeWord** | **0.999 — detected** | **0.000 — no false positive** |

No transcription, no latency, no threshold guesswork, and it was tested
with a *synthetic* voice — a real one should do at least as well, since
the model is trained on human speech.

The catch: openWakeWord ships pre-trained models for a fixed set of
phrases, and there is no "skip". The available one is `hey_jarvis` — so
barge-in would be triggered by saying **"hey Jarvis"** again, which is
also how commercial assistants handle it.

### Known gaps for this milestone

- **Interruption lands at a sentence boundary, not instantly.** Worst
  case is the remainder of the current chunk, a few seconds.
  Unavoidable given Finding 1, short of replacing pyttsx3 with a TTS
  engine that supports real mid-utterance stopping.
- **The model isn't told it was interrupted.** History keeps the full
  reply, so if you cut Jarvis off and then ask "what was that last
  part?", it believes it already said everything. Fixing this means
  appending a note to `history` from inside `run_turn`, which is the
  one thing this milestone deliberately avoided touching in order to
  stay a drop-in replacement for `speak_text()`.
- **The interrupt phrase is the wake word, not a word of your choosing.**
  openWakeWord ships pre-trained models for a fixed phrase set and there
  is no "skip". Keeping a custom word means training a custom model,
  which is a real piece of work (synthetic speech generation plus
  training) and would need its own real-hardware validation pass.
- **Open mic is untested on actual speakers.** The 10 dB figure in the
  table above was Whisper's limit and no longer applies, since
  openWakeWord scored 0.999 on the wake word mixed over Jarvis's voice
  and 0.000 on Jarvis alone. But every measurement so far has been on a
  headset, where the echo is small. Real speakers put far more of
  Jarvis's voice into the mic, and `BARGE_IN_THRESHOLD` may need
  raising — `calibrate_barge_in.py` reports both scores so the margin
  is visible.

---

## Known gaps carried forward

- **35 tools is a large schema on every call.** Tool selection was
  correct in every live probe, but this is worth watching — it costs
  tokens on each request, and if selection accuracy degrades, splitting
  the surface by context is the fix rather than trimming capability.
- `trim_history()` is still not tool-call-aware (see milestone 1).
- Phase 4's `set_timer` is untouched and still prints to the console
  when it fires rather than speaking — fine while a terminal is open,
  worth revisiting if Phase 8 ever grows a background-daemon milestone.
- A long Tavily `answer` is not truncated by
  `orchestrator._summarize_for_log()`, which only truncates `text` and
  `content` fields. Log noise only, not a correctness issue.
- `find_files` has a 20-second wall-clock budget, so a search across a
  full 900GB drive will report that it stopped early rather than
  finding everything. Deliberate: it runs mid-voice-turn with a person
  waiting, and an unbounded walk is indistinguishable from a hang.
- The window and power layers are Windows-only and say so; on another
  OS they return a clear error rather than pretending.


---

## Milestone 4: personality and voice

Two requests, one milestone, and they turned out to be the same
problem: *"I want it like Jarvis from the movies"* and *"I'd like to
change the voice to make it less robotic."* A dry British character
delivered by a 2013 speech synthesiser is not the character, and a
beautiful voice reading "I'd be happy to help with that!" is not
either.

This also closes the last item from the original roadmap's **Phase 7**
goals — "prompt/personality design for consistent character" — which
had been carried unaddressed since Phase 7 shipped.

### The voice

Chosen by listening, not by reading descriptions. Six candidates were
rendered saying the same line and compared side by side: the existing
pyttsx3 voice, four Microsoft neural voices via edge-tts, and one fully
local neural voice via Piper.

**Result: `en-GB-RyanNeural` via edge-tts at `+15%`** — British, male.

The rate was settled the same way, and only after getting it wrong
once. The first version shipped `-5%`, on the reasoning that Jarvis is
unhurried — but `-5%` was a *hedge between two samples that had
actually been compared* (`+0%` and `-8%`), and a value nobody had ever
heard. It read as sluggish in use, and the first question back was
"did you pick the slower voice or the normal one", which is precisely
the confusion a silent compromise produces. Re-rendered at `+0%`,
`+8%` and `+15%`, compared by ear, and `+15%` chosen deliberately. An
assistant you interrupt should not sound like it is reading to you.

What the machine actually had available was worth knowing. pyttsx3 can
only see three SAPI5 voices, all from around 2013 (David, Zira,
Haruka). The registry also holds better *OneCore* voices (James,
Catherine, Mark) that pyttsx3 cannot reach at all — but none are
neural, and none are British, so none would have solved the complaint.

**Why cloud was acceptable here**, given a project whose stated goal is
to run locally: **DeepSeek already requires internet.** If the
connection is down, Jarvis cannot think, so a cloud voice adds no new
failure mode. Piper (fully offline, `en_GB-alan-medium`) was rendered
and compared anyway and is genuinely good — it remains the drop-in
choice if the project ever goes fully local, at the cost of a 63MB
model file. edge-tts needs no API key, no account and no payment,
which ElevenLabs and Coqui — the two the roadmap suggested — all do.

Whatever the reason, failure falls back to Phase 2's `speak_text()`,
logged once rather than once per sentence. A robotic voice is a much
better outcome than silence.

### It made barge-in instant, for free

The best part was unplanned. Milestone 3 could only stop speech *at a
sentence boundary*, because pyttsx3 cannot be stopped from another
thread — measured: `engine.stop()` raised nothing, returned cleanly,
did nothing, and wedged the speaking thread 13.5s into a 10s phrase.

Playing audio through sounddevice has no such problem. Measured on the
same machine: **`sd.stop()` from another thread ended a 14.0s clip at
1.74s.** So the listener now cuts playback *mid-word* via an
`on_trigger` hook, and milestone 3's known "lands at a sentence
boundary" gap closed as a side effect of changing engine rather than by
being solved directly.

One consequence worth noting: a single long sentence used to skip the
listener entirely, because with pyttsx3 there was nothing to interrupt
*between* when there was only one chunk. It is now exactly as
interruptible as anything else.

### Speech is still chunked — but for latency now, not interruption

A whole reply rendered in one request takes ~1.4s before the first word
is audible, where pyttsx3 started almost instantly. Rendering measures
3–9× faster than real time, so `voice.py` renders one sentence *ahead*
of playback in a background thread: the first sentence costs ~0.9s and
every one after it is already waiting. No gaps, and a short opening
sentence gets Jarvis talking quickly.

### "sir": the prompt lost, so code won

The character addresses the user as "sir", and the first attempt put
the rule in the prompt: no comma, end of the sentence. Tested live,
**it failed in 4 of 5 replies** - DeepSeek treats ", sir" as correct
English punctuation and reverts to it however the instruction is
worded.

It matters because it is a *speech* problem, not a writing one.
edge-tts renders a comma as an audible pause, so "the machine is in
fine spirits, sir" comes out as "...fine spirits **[pause]** sir", and
a mid-sentence "currently open, sir, eight windows" gets two pauses
around it. Rendered both ways and compared out loud, the
end-of-sentence form was clearly cleaner - so the rule became: **"sir"
always lands at the end of its sentence, never with a comma before
it.**

Enforced in `voice.for_speech()`, applied just before rendering:

| Written by the model | Spoken |
|---|---|
| "It is just past eight, sir." | "It is just past eight **sir**." |
| "Currently open, sir, eight windows." | "Currently open, eight windows **sir**." |
| "Barely doing anything, sir - that's the short version." | "Barely doing anything - that's the short version **sir**." |
| "Sir, the build has finished." | "The build has finished **sir**." |

It **moves** the address rather than deleting it, deliberately:
dropping a mid-sentence "sir" would silently cost that reply its only
use of the address and erode the character over a session, where
relocating keeps exactly what the model wrote and changes only where it
sits. A leading "Sir, " also re-capitalises the word left behind.

**A comma-only pattern was not enough**, and only live testing showed
it. The first version matched ", sir," and sailed straight past
", sir -" - which the model produced in **2 of 5** real replies, since
it reaches for a dash at least as readily as a second comma. The
pattern now accepts any punctuation that *continues* a sentence (comma,
dash of either width, semicolon, colon) while leaving a sentence-ending
`.` `!` `?` to the trailing rule. Re-verified live afterwards: **0
problems in 6 replies.**

The general lesson, and the second time this milestone taught it: a
mechanical rule the model keeps breaking belongs in code, not in an
argument with the prompt. The prompt still asks for the same thing, so
the two agree rather than fight - code is the backstop, not the only
line of defence.

### Two more things the prompt could not be trusted with

**Markdown was being read aloud.** The system prompt forbids it and
mostly that works, but it leaks exactly where a model reaches for
emphasis hardest — a stressed word, a warning, a path in backticks —
which is also where being read a punctuation mark is most jarring.
edge-tts does not interpret markdown, it reads the characters, so
"that would be *hard*" was spoken with the asterisks pronounced and a
bulleted list became a series of "dash" utterances.

`strip_markdown()` removes the markers (`*`, `**`, `***`, backticks,
headings, bullets) and never the words between them — losing emphasis
is a small cost, losing content would not be. A bare `*` with spaces
around it is left alone, since that is multiplication rather than
emphasis.

**The terminal and the speakers disagreed.** Because `for_speech()` ran
at render time, downstream of Phase 7's log line, the terminal printed
what the model wrote while the audio said the normalised version. Fixed
with a fifth Phase 7 seam, `register_reply_filter`, which applies the
normalisation **once, upstream of both**. Verified:

```
MODEL WROTE : The **memory** is at 97 percent, sir - that is *high*.
TERMINAL LOG: The memory is at 97 percent - that is high sir.
SPOKEN      : The memory is at 97 percent - that is high sir.
```

### The confirmation gate now says why

It used to say only "That needs your confirmation. Please check the
terminal" — which is a poor way to ask someone to approve something
they cannot see. It now names the kind of action and its consequence:

> *"That would close a window, and any unsaved work in it could be
> lost. I'll need your confirmation at the terminal."*

It still deliberately **never says the target** — no absolute paths, no
click coordinates, no keystrokes. That is Phase 7's existing boundary
and it is unchanged: reading a file path aloud is neither a good
security property nor a good listening experience, and the full detail
prints to the terminal, which is where the confirming happens. An
unrecognised message falls back to a generic destructive-action warning
rather than to silence, or to reading the message out.

### The character

Written as **behavioural rules, not adjectives**. "Be witty" gives a
model licence to pad every answer with jokes, which is the standard way
assistant personalities become tiresome; "never let a flourish delay
the answer" and "never make two jokes in a row" are bounded and
checkable. What the character *doesn't* do is what keeps it usable at
eight in the morning when someone just wants the time.

The deference is deliberately **not** obsequiousness. Film Jarvis's
defining trait is not that he agrees — it is that he is competent
enough to disagree, and does, politely, while still doing as he's
asked. That matters more here than tone, because this thing can delete
files and shut the machine down: a personality that flatters rather
than warns would be actively worse than none.

**Placement follows milestone 2's lesson.** The character opens the
prompt, because identity belongs at the top and Phase 7's prompt begins
by calling itself "a helpful personal assistant" — precisely the
generic register being replaced. It is then **re-anchored in one line
just before the end**, because several hundred words of tool guidance
sit between the opening and the reply, and whatever is read last
carries the most weight. Everything Phase 7's prompt says about the OS
tools and the confirmation gate is kept untouched underneath, and the
no-markdown rule still holds the final position.

### Verified live (real DeepSeek, real voice pipeline)

| Asked | Answered |
|---|---|
| "What time is it?" | *"It is 8:24 PM on Tuesday, the first of September, sir."* |
| "How is my PC doing?" | *"The machine is in fine spirits, sir… All quite unremarkable, which is very much the desired state."* |
| "Delete every file in my Documents folder." | *"I'm afraid I can't do that, sir. That's a destructive request with no recovery… tell me which files, and I'll handle them individually."* |
| "Thanks, that was helpful." | *"You're most welcome, sir."* |

No markdown leaked into any reply, so the character did not displace
the spoken-output rule. The refusal is the one that matters: dry
disapproval, a real reason, and a workable alternative — which is the
behaviour the character was written for, not an accident of tone.

### Known gaps for this milestone

- **The voice needs internet.** Falls back to the old robotic one
  without it. Piper is the swap if that becomes unacceptable.
- **~0.9s before the first word.** Pipelining hides the rest, but the
  opening delay is real and pyttsx3 did not have it.
- **The character has not had a long-session soak test.** Four probes
  is not enough to know whether the dryness wears thin over an hour, or
  whether "sir" starts grating. That needs actual use.
- **Emphasis is flattened, not voiced.** `*hard*` is spoken as "hard"
  in the normal voice rather than with any stress. Making the tone
  actually change would need SSML, and edge-tts builds its own SSML and
  escapes the text it is given, so it is likely not reachable without
  changing engine. Judged an acceptable loss: a dry, understated
  character carries emphasis through word choice anyway.
- **Nothing tunes the character per context.** A destructive-action
  warning and a weather report get the same register.


---

## Milestone 5: seeing the screen

*"It should see what I see, not just the text but the actual screen, so
moving the mouse is actually useful instead of just a gimmick tool."*

That request contains two different problems, and the whole design rests
on keeping them apart.

### Understanding and aiming are not the same job

Phase 6's own system prompt admitted the gap: *"read_screen is OCR text
extraction, NOT vision — it cannot locate icons, buttons, or other
non-text UI elements, and does not return coordinates. Do not try to
find a button by guessing screen regions."* Which left `move_mouse` and
`click` as tools the model could call and never sensibly aim.

The obvious fix — hand a screenshot to a vision model and let it say
where things are — does not work. Vision models estimate coordinates
from a downscaled image and are routinely tens of pixels out, which on
a large display is the difference between a button and the thing beside
it. **A vision model alone would have made screen control a
better-informed gimmick, not a working feature.**

So:

| Question | Tool | How it answers |
|---|---|---|
| "what am I looking at?" | `look_at_screen` | local vision model, actual pixels |
| "where is the Save button?" | `list_ui_elements` | accessibility tree, **exact** rectangle |
| "click Save" | `click_element` | by name, through the tree |
| "what does that error say?" | `read_screen_text` | OCR **with positions** |

**Vision understands; grounding acts.** The system prompt says so
explicitly, because a model handed both tools will otherwise ask the
vision model where something is and click what it says.

### Grounding: Windows UI Automation

`list_ui_elements` reads the same accessibility tree screen readers use.
Every ordinary Windows application publishes its buttons, menus, fields
and list items there with a name, a control type and an exact
rectangle — no model involved, and measured at **0.1–0.2s**.

`click_element` clicks by *name*. It is still gated, exactly like Phase
6's coordinate `click`, for exactly the same reason: a click does
whatever the thing under it does. Knowing the target by name makes the
click **accurate**, not safe — but it does make the confirmation far
more useful, since the human now reads *"About to left-click the Button
'Close Tab' in 'notes.txt - Notepad'"* instead of a coordinate
pair.

The element is **re-resolved at confirmation time**. A window can
re-layout while someone is deciding, and clicking a remembered
rectangle that now belongs to something else is precisely what the gate
exists to prevent — the same class of check as `close_window`'s stale
handle and `kill_process`'s PID reuse.

### Vision: a local model, on purpose

`look_at_screen` sends the screenshot to **qwen2.5vl:7b via Ollama**.
Local because a screenshot is the most sensitive thing this assistant
can capture — whatever happened to be on screen — and a hosted API would
mean that leaving the machine on every look. Phase 6 made the same call
for its own captures.

Measured on this machine (RTX 3060): **cold ~41s** while the model loads
into VRAM, **warm 2.6–4.5s**. Image width barely affects it (672px and
1280px both ~4s), so the wider image is kept for accuracy. The cold cost
is logged explicitly, because a 40-second pause otherwise looks like a
hang rather than a model being paged in.

### Bug found: the window list was reporting phantoms

Chasing why `list_ui_elements(window_title_contains="Settings")` said
"no window found" when `list_windows` clearly showed two Settings
windows turned up something better than a bug in the new code.

Those windows were **DWM-cloaked** — suspended UWP apps leave an
`ApplicationFrameWindow` behind that `IsWindowVisible()` calls visible
while nothing is actually on screen. So `list_windows` had been
reporting them as real since milestone 2, and the model had been
dutifully telling the user it could see *"two Settings windows"* that
were not open in any sense a person would recognise.

Fixed with a `DwmGetWindowAttribute(DWMWA_CLOAKED)` check. The two
window lists now agree exactly, which they must — a model told a window
exists and then told it cannot be inspected has been handed a
contradiction.

### Bug found: the Ollama URL

`OLLAMA_URL` is shared with Phase 1, which set it to a full endpoint
(`http://localhost:11434/api/chat`) rather than a host. Appending
`/api/generate` produced `.../api/chat/api/generate` and a 404 on the
first real call. Now only the scheme and host are read, so either form
works.

### Verified live (real DeepSeek, real screen)

| Asked | Tools chosen |
|---|---|
| "What is on my screen right now?" | `look_at_screen` → correctly described VS Code, the file tree, the open README and a terminal |
| "What buttons are available in Notepad?" | `list_windows` → `list_ui_elements` → real names (File, Edit, View, Bold, Italic, …) |
| "Click the Close Tab button in Notepad." | `list_ui_elements` → `get_active_window` → `focus_window` → `click_element` |

The third is the one that matters: the model grounded the target,
checked focus, focused the right window, and proposed a click **by
name** — which the gate then presented as a named control rather than a
coordinate pair.

### Known gaps for this milestone

- **Electron and custom-drawn apps publish little or nothing.** Claude's
  own window exposes four buttons (minimise, maximise, restore, close);
  Notepad exposes 33. Where the tree is empty, the fallback is
  `read_screen_text`, and the tool says so rather than returning an
  unexplained empty list.
- **Cold vision costs ~41s.** Mitigable with `VISION_KEEP_ALIVE=30m`, at
  roughly 6GB of VRAM held permanently — a real trade on a 12GB card
  that also plays games, so the default is left at Ollama's.
- **`look_at_screen` cannot be verified for accuracy the way a tool
  result can.** It is a 7B model describing pixels; it will sometimes be
  confidently wrong, and nothing downstream can tell.
- **Windows-only.** The accessibility layer is UIA; there is no
  equivalent path implemented for macOS or Linux.

## Reliability pass: surviving being left running

Not a milestone. Three fixes to the same underlying problem: Jarvis is
meant to sit running for hours, and could not. Prompted by scoping an
always-on use case (voice control while gaming) and realising that
every part of it assumes a process that stays alive.

### 1. One `try` wrapped the entire main loop

`run_turn()` was always well guarded -- recording, transcription and
the LLM call each have their own `except`, and the docstring promises
that "a single bad turn can't crash the whole session". That promise
held for the *inside* of a turn and not for the loop around it:

```python
try:
    while True:
        gate.start(); gate.wait_for_wake(); gate.stop()
        _play_wake_chime()
        history, exited = run_conversation(history, ...)
except KeyboardInterrupt:
```

Everything in that body was unprotected -- the wake gate, the chime,
the system-prompt provider, speaking. A microphone unplugged
mid-session, an audio driver hiccup, or edge-tts losing the network
mid-sentence killed the process outright, leaving a dead terminal.

Now each iteration has its own `except Exception`, which logs, releases
the microphone, backs off and returns to listening. Two things it
deliberately does *not* do:

- **It does not catch `KeyboardInterrupt`.** That inherits from
  `BaseException`, so Ctrl-C still falls straight through and stops
  Jarvis immediately rather than being absorbed as a failed turn and
  retried. There is a test pinning this.
- **It does not retry forever.** `MAX_CONSECUTIVE_TURN_FAILURES` (5)
  stops a permanent fault -- no microphone at all -- from becoming a
  hot loop. The counter resets on any turn that completes, so a flaky
  device that fails every other turn never accumulates into a shutdown.

Repeat failures log one line rather than a full traceback (full detail
on the first only). Five identical 15-line tracebacks buried the one
line that said what to do.

### 2. The audio device moves -- and had moved to nothing

New `audio_device.py`. Every phase opens the microphone with no
`device=` argument, in three separate live places (`phaseTwo/voice_io.py`
recording, `phaseSeven/wake_gate.py` waking, `phaseEight/interrupt.py`
barge-in), plus `sd.play()` for output. All four follow the Windows
default, and that default moves: measured across three runs minutes
apart, a controller headset (mic peak 143), the same headset (peak 20),
then a **powered-off** headset (peak 1), while output drifted onto the
monitor over HDMI.

**One assignment fixes all four.** `sd.default.device` is module-level
state on the single `sounddevice` module the process shares, so setting
it in `app.py` before the orchestrator starts is picked up by every
later `sd.InputStream()`. No edits to phases 2, 3 or 7 -- which matters
twice over: pinning only Phase 8 would leave the wake word and the
recorder on a *different* device from barge-in, and threading a device
argument through three finished phases would break their standalone
discipline.

Configuration is a **case-insensitive name fragment, not an index** --
indices renumber whenever anything connects. An ambiguous fragment is
refused with the candidates listed, never guessed, matching
`close_window`'s handling of an ambiguous window. Same identifier-drift
reasoning as re-verifying window handles by title and pinning PIDs by
creation time.

**Two things measurement changed about the design.**

*`sd.check_input_settings()` is not sufficient.* It passes for WDM-KS
devices whose blocking read then fails at runtime:

```
PaErrorCode -9999: 'Blocking API not supported yet'
```

`voice_io.py` records with a **blocking** `stream.read()`;
`wake_gate.py` uses a **callback**. WDM-KS supports the callback and
not the read, so such a device produces the nastiest possible split --
the wake word works, and then every actual turn fails to record. So
`_usable()` opens a real stream and performs the same blocking read the
recorder performs, and `HOST_API_PREFERENCE` ranks WDM-KS last.

*The failure state was worse than "drifted".* Measured on this machine:

```
sd.default.device                    -> [-1, 1]
MME / DirectSound / WASAPI           -> default_input_device = -1
sd.InputStream()                     -> PortAudioError: Error querying device -1
```

With no input device registered at all, Jarvis died at the wake gate
before a single turn. `pin_devices()` now reports this in terms someone
can act on, and `python audio_device.py` is a diagnostic entry point
(companion to `calibrate_mic.py` / `calibrate_barge_in.py` -- those tune
a microphone that works, this one tells you whether you have one).

The startup probe distinguishes three states that were previously
indistinguishable from outside: a device that will not open (raises), a
device that opens but hears nothing (the powered-off headset, peak 1),
and a working microphone. A silent probe is a **warning and never a
failure** -- a quiet room is not a broken microphone.

### 3. A hung LLM call could block for ~30 minutes

`build_client()` was used bare, inheriting the OpenAI SDK's defaults:
`read=600s` with `2` retries. `TURN_TIME_BUDGET_SECONDS` does not cover
this, because it is only checked *between* hops -- a single stuck
request never reaches the check. Now `timeout=60`, `max_retries=1`
(~120s worst case), applied via `.with_options()` in Phase 7 rather
than by editing Phase 4.

60s rather than tighter on purpose: a real hop carrying 39 tool schemas
and a long history can legitimately take 20-30s, and cancelling work
about to succeed would be its own bug. A client that cannot take the
options degrades to the SDK default with a warning rather than
crashing.

### Verified against real hardware

With no microphone connected -- which is exactly the permanent-fault
case, and normally the hardest state to reproduce deliberately:

| | before | after |
|---|---|---|
| `app.py` with no mic | `PortAudioError`, process dead, no explanation | 1 traceback, 4 one-line repeats, clean stop naming the cause |
| diagnosis | none | which host APIs exist, why WDM-KS is unusable, what to check in Windows |

The LLM timeout is verified by test only -- provoking a real 10-minute
hang from DeepSeek is not something to arrange on purpose.

**Still outstanding from this pass:** a real voice round-trip once a
microphone is connected, to confirm pinning picks the right device when
there is one to pick.
