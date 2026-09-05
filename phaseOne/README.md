# Jarvis — Phase 1: Text-Only CLI Chatbot

This is the foundation project from the roadmap. It's just a terminal
chatbot — no voice, no tools, no persistent memory. The point is to get
comfortable with API calls and conversation-state management before
adding anything else.

Runs against a **local Ollama model by default** — free, no API key,
no rate limits, runs entirely on your GPU.

## Setup

1. **Install Ollama**: https://ollama.com/download (Windows installer, a few clicks)

2. **Pull a model** (recommended for your hardware — RTX 3060, 16GB RAM):
   ```
   ollama pull llama3.1:8b
   ```
   This downloads ~4.9GB and fits comfortably in either the 8GB or 12GB
   RTX 3060 variant.

3. **Create a virtual environment** (recommended):
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

4. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

5. **Run it**:
   ```
   python jarvis.py
   ```

Ollama runs as a background service after install, so you shouldn't
need to manually start it — but if `jarvis.py` warns it can't reach
Ollama, run `ollama serve` in a terminal first.

Type `exit` or `quit` to leave, or just Ctrl+C.

## What's in here

- **`jarvis.py`** — the whole thing. Structure:
  - `call_ollama()` — sends the conversation to your local Ollama server
  - `call_anthropic()` — same interface, hosted Claude, for when you're ready to upgrade
  - `call_llm()` — routes to whichever backend `LLM_BACKEND` is set to; this is the seam between "free and local" and "paid and higher quality"
  - `check_ollama_available()` — pings Ollama at startup so you get a clear warning instead of a wall of connection errors
  - `trim_history()` — a crude cap on conversation length so context doesn't grow forever
  - `main()` — the input/output loop

## Upgrading later (per the roadmap: "free for now, upgrade when required")

When you hit a phase where local model quality becomes the bottleneck
(most likely Phase 4 — tool calling, where smaller local models get
unreliable), switch backends without touching any code:

```
# in .env
LLM_BACKEND=anthropic
ANTHROPIC_API_KEY=your-key-here
```

Everything else — history management, the main loop, error handling
pattern — stays identical. That's the point of the `call_llm()` seam.

## Why it's built this way

- **Conversation history is a plain Python list** of `{"role": ..., "content": ...}`
  dicts, appended to each turn and sent in full with every request. Same
  shape works for both Ollama and Anthropic's APIs.
- **Error handling is centralized** in each backend function so the main
  loop never crashes on a dropped connection, a missing model, or Ollama
  not running yet — it just prints a message and lets you keep going.
- **Startup check for Ollama** — since a local server can silently not be
  running, `main()` pings it before the loop starts and gives you a
  specific fix (`ollama serve`, or `ollama pull <model>`) instead of a
  generic error on your first message.
- **No streaming yet.** Deliberately kept simple. Streaming is a natural
  upgrade once this works.

## Exit criteria for Phase 1 (per the roadmap)

You're done with this phase when:
- [ ] You can have a multi-turn conversation and Jarvis remembers earlier turns
- [ ] You understand *why* the full message list is sent every time (the API is stateless)
- [ ] You've deliberately triggered at least one error (stop Ollama mid-session, or use a model name you haven't pulled) and watched it handle gracefully instead of crashing
- [ ] You've tweaked the system prompt and seen the personality change
- [ ] You've noticed the difference in response quality/speed vs. what you'd expect from a hosted model — this is useful signal for when to upgrade later

Once that's solid — and only then — move to Phase 2 (voice).

## A few things worth trying before moving on

- Change `SYSTEM_PROMPT` and see how much it changes Jarvis's tone.
- Try a different model (`ollama pull mistral` then set `OLLAMA_MODEL=mistral`
  in `.env`) and compare quality/speed against Llama 3.1 8B.
- Try lowering `MAX_HISTORY_MESSAGES` to something tiny (like 4) and notice
  Jarvis "forgetting" earlier context — a preview of why Phase 5 (real
  memory) is a separate, non-trivial phase.
- Run `ollama ps` in a terminal while jarvis.py is running to see the
  model actually loaded in your GPU's VRAM.
