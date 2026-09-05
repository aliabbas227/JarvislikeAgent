# Phase 2 — Voice In, Voice Out

Goal (per the roadmap): the same chatbot as Phase 1, but you speak instead
of type, and it speaks back.

## What's here

- **`voice_io.py`** — the actual new thing this phase teaches: record from
  the mic until you stop talking, transcribe with Whisper, speak with
  pyttsx3. Fully independent of the LLM — you can test and debug audio
  capture without ever calling an API.
- **`voice_chat.py`** — the loop that ties `voice_io.py` into Phase 1's
  `call_llm()` / `get_system_prompt()` / `trim_history()`. This is the one
  deliberate exception to "build standalone": the LLM backend is already
  built and tested, so this phase reuses it rather than re-implementing it.
- **`test_voice_io.py`** — 11 pytest tests, all mocked (no mic, no GPU, no
  downloaded Whisper model, no real TTS driver needed to run them).
- **`requirements.txt`**, **`.env.example`**

## Setup

```bash
cd phaseTwo
python -m venv venv
venv\Scripts\activate          # Windows

# GPU note (RTX 3060): install CUDA-enabled torch BEFORE the rest of the
# requirements, or Whisper will silently fall back to CPU. Check the
# current command for your CUDA version at https://pytorch.org/get-started/locally/
# e.g. for CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
copy .env.example .env
```

pyttsx3 on Windows uses SAPI5 and needs no extra install. sounddevice needs
PortAudio, which its wheel bundles on Windows — no separate install should
be required.

## Running it

```bash
python voice_chat.py
```

Speak after "Listening..." appears in the console. It stops recording once
you've gone quiet for `SILENCE_DURATION` seconds (1.5s by default), sends
the transcript to your Phase 1 LLM backend, and speaks the reply. Ctrl+C to
quit.

If `voice_chat.py` can't find Phase 1's `jarvis.py`, edit `PHASE_ONE_PATH`
near the top of the file to point at your `phaseOne` folder.

## Design decisions worth knowing

**Batch, not streaming.** The roadmap flags this as the key tradeoff for
this phase. This build waits for a full pause in speech, then transcribes
the whole clip — not live/streaming transcription. Simpler to build and
debug; costs you a small delay after you stop talking. Streaming STT is a
reasonable later upgrade once this is solid, not something to reach for now.

**Silence detection via RMS amplitude**, not voice-activity-detection (VAD)
models. It's a blunt instrument — a loud fridge hum or a quiet mumble can
throw it off — which is exactly why `SILENCE_THRESHOLD` is a `.env` knob and
not a hardcoded constant. Expect to tune it for your mic and room in the
first few minutes of testing.

**Local-first, same pattern as Phase 1's LLM backend switch.** Whisper
(local, free) and pyttsx3 (local, free, robotic-sounding) are the defaults.
Documented upgrade path when you want better quality:
- STT: Whisper `base` → `small`/`medium` for accuracy, or a cloud STT API
  for speed without local GPU load
- TTS: pyttsx3 → ElevenLabs or Coqui TTS for natural-sounding speech

**Lazy imports for whisper/pyttsx3.** Both are imported inside functions,
not at module load. This keeps `import voice_io` cheap and — more
importantly — is what makes `test_voice_io.py` runnable without either
package's real backend (audio driver, downloaded model) present.

## Testing

```bash
pytest test_voice_io.py -v
```

All 11 tests are mocked — safe to run on a machine with no microphone, no
GPU, and no Whisper model downloaded yet. Covers: successful record/stop,
no-speech timeout, mic-open failure, mic-read failure, successful
transcription, empty-transcription rejection, Whisper load failure,
Whisper transcription failure, successful speech, TTS init failure, and the
empty-string no-op.

`voice_chat.py` itself isn't unit-tested here — it's a thin orchestration
loop over already-tested pieces (`voice_io` and Phase 1's `jarvis.py`).
Exercise it manually per the checklist below.

## Manual test checklist

- [ ] Speak a full sentence — does it transcribe correctly and get an
      LLM reply spoken back?
- [ ] Stay silent after "Listening..." — does it time out gracefully
      (via `MAX_RECORD_SECONDS`) instead of hanging?
- [ ] Mumble something too quiet to cross `SILENCE_THRESHOLD` — does it
      re-prompt instead of crashing?
- [ ] Unplug/mute the mic mid-run — does it log a clear error and let you
      try again, rather than killing the process?
- [ ] Let a multi-turn conversation run — does context carry across turns
      the same way it did in Phase 1's text version?
- [ ] Check GPU usage during transcription (Task Manager) — confirm Whisper
      is actually using the RTX 3060, not silently falling back to CPU.

## Exit criteria for this phase

You can have a full back-and-forth spoken conversation with the Phase 1
LLM backend — no typing, no clicking a button per turn — and it survives a
bad mic read, a silent timeout, and an LLM error without crashing.

## Next: Phase 3

Wake-word detection (`Porcupine` or `openWakeWord`) so you say "Hey Jarvis"
instead of the script starting to listen the moment it launches. Per the
roadmap: build it as its own standalone mini-project first, same as this
one — don't merge it into `voice_chat.py` yet.
