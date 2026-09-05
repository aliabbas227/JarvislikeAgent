# Jarvis

A voice-controlled personal assistant for Windows, built from scratch one
phase at a time. Say "hey Jarvis", ask for something, and it answers out
loud — and it can actually do things: search the web, read your files,
manage windows and processes, look at your screen, and remember what you
tell it.

Built as a learning project on a single desktop (i5-10600KF, RTX 3060
12GB, 16GB RAM). Speech recognition, wake word detection, vision and
memory all run **locally**; only the language model and text-to-speech
are cloud services.

```
"hey jarvis"  ──▶  wake word (openWakeWord, local)
                        │
                   record + transcribe (Whisper, local)
                        │
                   retrieve memories (ChromaDB, local)
                        │
                   think + call tools (DeepSeek)  ◀──▶  39 tools
                        │                                (gated if destructive)
                   speak (edge-tts, British neural voice)
                        │
                   ◀── interruptible: say "hey jarvis" to cut it off
```

## What it can do

**39 tools**, all reachable by voice:

| Area | Examples |
|---|---|
| Knowledge | web search with recency control, weather, current time |
| Memory | "remember I keep my invoices in Documents/2024" — recalled automatically on later turns |
| Files | find, read, move, copy, extract archives, read PDFs and Word documents, disk usage |
| Windows | list, focus, move, close; list and kill processes; system status |
| Screen | describe what's on screen (local vision model), read text with positions (OCR), list UI elements, click a control **by name** |
| Machine | clipboard, media keys, screenshots, lock, power actions |

A representative real turn: *"close the Spotify window"* → the model
chains `list_windows` → `get_active_window` → `close_window`, hits the
confirmation gate, and waits for a typed password before anything closes.

## Security model

The assistant can delete files, kill processes and shut the machine
down. Its input is an open microphone, and its context includes stored
memories and web search results — all of which are untrusted text that a
language model reads. So destructive actions do not depend on the model
behaving.

**Two-phase confirmation.** A destructive tool call never executes. It
returns a *proposal*, and the thing that would do the damage is held as
a Python closure in module memory, keyed by a token. The model never
sees that callable and has no tool that can invoke it. Confirming
requires a password typed at the **terminal** — a channel neither the
model nor the microphone can reach. `getpass` reads the real console
directly, bypassing piped stdin, so it cannot be scripted around.

Supporting rules:

- Spoken confirmations describe the *consequence* but never the target —
  no paths, coordinates or keystrokes are read aloud. Full detail goes
  to the terminal, where the decision is actually made.
- Identifiers are re-verified at confirmation time: window handles by
  title (Windows reuses handles), processes by creation timestamp (PIDs
  get reused), UI elements by re-resolving the element.
- Protected system processes and Zip-Slip archive entries are refused
  outright rather than gated.
- Ambiguous targets ("2 windows match 'firefox'") are refused with the
  candidates listed, never guessed.

## Architecture

Each phase is a self-contained mini-project with its own virtualenv,
built in order. Phases 1–6 stand alone; phase 7 wires them together and
phase 8 layers polish on top.

| Phase | What it added |
|---|---|
| `phaseOne` | Text CLI chatbot (Ollama or Anthropic backend) |
| `phaseTwo` | Voice I/O — Whisper speech-to-text, offline text-to-speech |
| `phaseThree` | Wake word detection (openWakeWord, `hey_jarvis`) |
| `phaseFour` | Tool-calling agent (DeepSeek) |
| `phaseFive` | Persistent memory (ChromaDB + retrieval) |
| `phaseSix` | OS control, with the two-phase confirmation gate |
| `phaseSeven` | Orchestration — everything above in one continuous voice loop |
| `phaseEight` | Accuracy, 39 tools, barge-in, personality and voice, screen vision, reliability |

Every phase has its own README documenting design decisions, bugs found
during real-hardware testing, and what was deliberately deferred.
`jarvis-agent-roadmap.md` is the original plan the project follows.

**The folder-per-phase layout is scaffolding, not the intended final
architecture.** It exists so each stage stayed runnable on its own.
Phase 8 extends phase 7 through six explicit seams rather than forking
it; consolidating that into one application is the next structural step.

## Setup

Requires Python 3.11+, Windows 10/11, and a working microphone.

```bash
cd phaseEight
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# openWakeWord ships its weights separately from the package
python -c "import openwakeword.utils as u; u.download_models()"

cp ../phaseSeven/.env.example ../phaseSeven/.env   # add your API keys
python app.py
```

`phaseEight/app.py` is the entry point. You will need a
[DeepSeek](https://platform.deepseek.com/) API key, and optionally a
[Tavily](https://tavily.com/) key for web search. Set `CONFIRM_PASSWORD`
in `phaseSeven/.env` — without it, the destructive-action gate falls
back to a plain yes/no prompt.

Screen vision is optional and needs [Ollama](https://ollama.com/) with
`ollama pull qwen2.5vl:7b`.

Two diagnostics worth knowing about, since audio problems are otherwise
invisible:

```bash
python audio_device.py        # which mic will be used, and is it hearing anything
python calibrate_barge_in.py  # can it detect the wake word over its own voice
```

## Things that turned out to be true

The interesting parts of this project were mostly cases where the
obvious answer was wrong and measurement said so.

- **`pyttsx3.stop()` silently does nothing across threads.** No
  exception, clean return, no effect — and it wedged the speaking thread
  13.5s into a 10s phrase. Switching to a different audio path made
  interruption work *and* dropped stop latency to 1.74s of a 14.0s clip.

- **Whisper is the wrong tool for barge-in.** A clip that transcribed
  perfectly as `'Skip.'` on its own came back as `'Quick TENSCHES Okoi'`
  when mixed under the assistant's own voice. On identical audio,
  openWakeWord scored **0.999** on the same clip and **0.000** on the
  assistant's voice alone. Utterance transcribers and keyword spotters
  are not interchangeable.

- **A vision model cannot click accurately.** VLMs estimate coordinates
  from a downscaled image and land tens of pixels off. Windows UI
  Automation returns exact rectangles in 0.1–0.2s with no model at all.
  Vision answers *"what am I looking at?"*; the accessibility tree
  answers *"where is the button?"*. Conflating them is why screen agents
  miss.

- **The tool schemas were not the token problem.** 39 schemas per call
  looked expensive but served almost entirely from cache (5,376 hit / 39
  miss). The real cost was the system prompt stating the time *to the
  minute*, which invalidated the cached prefix every conversation.
  Moving the clock into the user turn: **4,263 → ~59** full-price tokens
  per conversation.

- **Appending to a system prompt silently demotes whatever used to be
  last.** A "no markdown" instruction worked until several hundred words
  were appended after it, at which point replies came back as bulleted
  lists — which a speech synthesiser reads aloud as punctuation.

- **A mechanical rule the model keeps breaking belongs in code.** Asked
  to write "sir" without a preceding comma, the model complied in 1 of 5
  replies; it treats ", sir" as correct English and reverts. Enforcing it
  deterministically fixed it — with the prompt still asking for the same
  thing, so the two agree rather than fight.

- **`check_input_settings()` passing does not mean a microphone works.**
  It passes for WDM-KS devices whose blocking read then fails with
  `PaErrorCode -9999` — so such a device *wakes fine and then fails
  every single turn*, which is nastier than not working at all.

## Testing

```bash
cd phaseEight && .\venv\Scripts\Activate.ps1 && pytest   # 270 tests
cd phaseSeven && pytest                                   # 164 tests
```

Mocked tests are treated as necessary but not sufficient here. Every
phase has produced at least one real bug that only appeared against real
hardware or a real API and was structurally invisible to the suite —
wrong-directory `.env` resolution, stale model state across restarts,
audio devices that enumerate but cannot be read. Each phase README logs
those under its own "bugs found" section.

## Status

All eight phases are built and confirmed working against real hardware,
including full voice round-trips through the confirmation gate.

Known gaps: conversation history does not reset between sessions; the
model is not told when it was interrupted mid-reply; open-microphone
barge-in needs the speaker roughly 3x louder than the assistant
(measured), so a headset is currently assumed.

## License

MIT
