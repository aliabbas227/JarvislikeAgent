"""
Jarvis — Phase 1: Text-Only CLI Chatbot

Goal of this phase: learn to talk to an LLM programmatically and manage
conversation state. No voice, no tools, no memory across sessions yet —
just a solid, well-structured foundation for everything else.

Runs against a local Ollama model by default (free, runs on your GPU).
Swap LLM_BACKEND to "anthropic" later without touching the rest of the
code — same trim_history/main loop, just a different call_llm().

Usage:
    1. Install Ollama: https://ollama.com/download
    2. Pull a model:   ollama pull llama3.1:8b
    3. Run this:       python jarvis.py

Requirements:
    pip install requests python-dotenv
    (pip install anthropic  -- only needed if you switch LLM_BACKEND to "anthropic")
"""

import os
import sys
from datetime import datetime
import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# "ollama" = free, local, runs on your GPU. "anthropic" = hosted, paid,
# higher quality — swap to this later for phases where reliability
# (especially tool-calling in Phase 4) matters more than cost.
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")

# --- Ollama settings ---
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

# --- Anthropic settings (used only if LLM_BACKEND == "anthropic") ---
ANTHROPIC_MODEL = "claude-sonnet-4-6"
ANTHROPIC_MAX_TOKENS = 1024

BASE_SYSTEM_PROMPT = (
    "You are Jarvis, a helpful, concise personal assistant. "
    "You are currently in phase 2 (Voice in and Voice out)."
    "of a larger build). Be direct and useful; skip unnecessary filler."
)


def get_system_prompt() -> str:
    """Build the system prompt fresh each call, with the real current
    date/time injected. Local models have no innate sense of 'now' — they
    only know what was in their training data — so without this they'll
    confidently guess wrong. This is a stand-in for real tool use
    (Phase 4), where the model would call a get_current_time() tool
    instead of having it spoon-fed in the prompt."""
    now = datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")
    return f"{BASE_SYSTEM_PROMPT}\n\nCurrent date and time: {now}."

# How many messages (user+assistant turns) to keep before trimming.
# This is a simple safeguard against unbounded context growth — not a
# real memory system (that's Phase 5).
MAX_HISTORY_MESSAGES = 40

# Local models are slower than hosted APIs, especially on the first
# request (model has to load into VRAM). Give it real time before
# giving up.
OLLAMA_TIMEOUT_SECONDS = 120


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

def check_ollama_available() -> bool:
    """Ping Ollama's base URL so we can fail with a clear, actionable
    message instead of a raw connection-refused traceback."""
    try:
        base = OLLAMA_URL.rsplit("/api/", 1)[0]
        requests.get(base, timeout=3)
        return True
    except requests.exceptions.RequestException:
        return False


def call_ollama(messages: list) -> str:
    """Send the conversation to a local Ollama server and return the
    assistant's text reply."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": get_system_prompt()}] + messages,
        "stream": False,
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]
    except requests.exceptions.ConnectionError:
        return (
            "[Can't reach Ollama. Is it running? Try 'ollama serve' in a "
            "terminal, or check that the Ollama app is open.]"
        )
    except requests.exceptions.Timeout:
        return "[Ollama took too long to respond. The model may still be loading — try again.]"
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return (
                f"[Model '{OLLAMA_MODEL}' not found. Pull it first with: "
                f"ollama pull {OLLAMA_MODEL}]"
            )
        return f"[Ollama HTTP error: {e}]"
    except (KeyError, ValueError):
        return "[Got an unexpected response from Ollama — check it's up to date.]"


# ---------------------------------------------------------------------------
# Anthropic backend (for later — kept behind the same interface)
# ---------------------------------------------------------------------------

def call_anthropic(messages: list) -> str:
    """Send the conversation to Claude and return the assistant's text
    reply. Only used when LLM_BACKEND == 'anthropic'."""
    from anthropic import Anthropic, APIError, APIConnectionError, RateLimitError

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "[ANTHROPIC_API_KEY is not set. Add it to your .env file.]"

    client = Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=ANTHROPIC_MAX_TOKENS,
            system=get_system_prompt(),
            messages=messages,
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        )
    except RateLimitError:
        return "[Jarvis is rate-limited right now — wait a moment and try again.]"
    except APIConnectionError:
        return "[Couldn't reach the API — check your internet connection.]"
    except APIError as e:
        return f"[API error: {e}]"


# ---------------------------------------------------------------------------
# Backend-agnostic core
# ---------------------------------------------------------------------------

def trim_history(messages: list) -> list:
    """Keep the conversation from growing forever. Simple truncation for
    now — a real system would summarize instead of dropping turns, but
    that's out of scope for Phase 1."""
    if len(messages) > MAX_HISTORY_MESSAGES:
        return messages[-MAX_HISTORY_MESSAGES:]
    return messages


def call_llm(messages: list) -> str:
    """Single entry point the main loop calls. Routes to whichever
    backend is configured — this is the seam you swap when you're
    ready to upgrade from free/local to hosted."""
    if LLM_BACKEND == "anthropic":
        return call_anthropic(messages)
    return call_ollama(messages)


def main() -> None:
    print(f"Jarvis (Phase 1 — text-only, backend: {LLM_BACKEND}).")
    print("Type 'exit' or 'quit' to leave.\n")

    if LLM_BACKEND == "ollama" and not check_ollama_available():
        print(
            "WARNING: Can't reach Ollama at startup. Make sure it's running "
            "('ollama serve') and that you've pulled a model "
            f"('ollama pull {OLLAMA_MODEL}'). Continuing anyway...\n"
        )

    messages = []  # each item: {"role": "user"|"assistant", "content": str}

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nJarvis: Shutting down. Goodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Jarvis: Goodbye.")
            break

        messages.append({"role": "user", "content": user_input})
        messages = trim_history(messages)

        reply = call_llm(messages)
        print(f"Jarvis: {reply}\n")

        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
