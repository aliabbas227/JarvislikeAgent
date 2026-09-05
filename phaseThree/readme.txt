# Jarvis Project — Phase 3: Wake Word / Always Listening

## Status: COMPLETE

## Goal
Detect the phrase "hey jarvis" continuously via microphone, without
clicking anything, and fire a callback. This is a **standalone**
mini-project — it does not import from or call into `phaseOne\` or
`phaseTwo\`. Wiring wake-word detection into the full voice loop is
Phase 7's job, not this one.

## Environment
- Windows 11, same machine as Phase 1/2 (i5-10600KF, RTX 3060 12GB, 16GB RAM)
- `A:\PROJECTS\JARVIS\phaseThree\`, own venv, separate from `phaseOne`/`phaseTwo`
- GPU not required for this phase — openWakeWord's models are small
  enough to run comfortably on CPU

## Files
- **`wake_word.py`** — core listener. `build_model()` loads the
  pretrained `hey_jarvis` openWakeWord model via ONNX. `listen_loop()`
  opens a `sounddevice.InputStream`, scores each audio chunk, and calls
  `on_detect(score)` when the wake word crosses `WAKE_THRESHOLD` — with a
  cooldown so one utterance doesn't fire multiple times.
- **`test_wake_word.py`** — 7 mocked pytest tests. No mic, no real
  openWakeWord model, no real `sounddevice` stream needed to run them.
- **`requirements.txt`** — `openwakeword`, `sounddevice`, `numpy`,
  `python-dotenv`
- **`.env.example`** / **`.env`** — `WAKE_MODEL`, `WAKE_THRESHOLD`

## Current `.env` configuration
```
WAKE_MODEL=hey_jarvis
WAKE_THRESHOLD=0.5
```
0.5 tested clean with zero false positives during several minutes of
normal speech and background noise near the mic. No tuning was needed —
if you change hardware/mic later, re-run the false-positive check before
trusting this threshold again.

## Environment gotchas hit and fixed
1. **`tflite_runtime` not found on Windows** — openWakeWord defaults to
   trying the tflite backend first, but `tflite-runtime` has no Windows
   wheels, so it can't be installed there at all. Fix: explicitly pass
   `inference_framework="onnx"` to `Model(...)` in `build_model()`.
2. **Pretrained model weights aren't bundled in the pip package** — first
   use needs an explicit one-time download step, run manually (not from
   inside `wake_word.py`):
   ```python
   import openwakeword
   openwakeword.utils.download_models()
   ```
   Only needs to be run once per machine/venv.
3. **Multi-trigger on a single utterance** — without a cooldown, the
   score stays above threshold for several consecutive ~80ms frames
   while you're still speaking the wake word, firing `on_detect` 5–7
   times for one "hey jarvis." Fixed with a time-based debounce
   (`cooldown_seconds`, default 2.0s) tracked via `time.monotonic()` in
   `listen_loop`.

## Manual test checklist — status
- [x] Detects "hey jarvis" reliably
- [x] One detection per utterance (debounce confirmed working)
- [x] Ctrl+C exits cleanly (no traceback spam)
- [x] No false positives during ~several minutes of normal speech /
      background noise at threshold 0.5
- [ ] Mic disconnected/unplugged mid-run — not yet explicitly tested,
      lowest priority remaining item (same deferred status as the
      equivalent item in the Phase 2 handoff)

## Explicitly deferred
- **Wiring into `voice_chat.py`** — that's Phase 7 (full orchestration:
  wake word → listen → transcribe → LLM → tools → speak → back to
  listening). Keeping phases standalone is the core discipline from the
  roadmap; don't shortcut it here.
- **Custom-trained wake model** — currently using openWakeWord's stock
  pretrained `hey_jarvis` model rather than training a custom one.
  Fine for now; revisit only if the stock model's accuracy becomes a
  real problem in practice.

## Next: Phase 4 — Tool Use / Function Calling
Per the roadmap: give the LLM the ability to actually *do* things
(weather, opening apps, web search, etc.) via function/tool calling.
This is flagged in the roadmap as the hardest conceptual leap so far —
budget real time for it. Also the natural point to revisit the
`LLM_BACKEND=anthropic` upgrade path noted in the Phase 1/2 history,
since local Llama 3.1 8B via Ollama was already flagged as unreliable
for structured tool calls.