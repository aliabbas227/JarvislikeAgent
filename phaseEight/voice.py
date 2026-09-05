"""
voice.py -- Phase 8, milestone 4: a voice that doesn't sound like 2013.

Replaces Phase 2's pyttsx3/SAPI5 output with Microsoft's neural
text-to-speech (edge-tts), rendered to audio and played through
sounddevice.

The roadmap anticipated this exactly: Phase 2 lists "TTS: pyttsx3
(offline, robotic) -> later upgrade to ElevenLabs or Coqui TTS
(natural-sounding)". This is that upgrade, with a different engine --
edge-tts needs no API key, no account and no payment, which the other
two do.

## Why this also fixed barge-in

pyttsx3 could not be stopped. Measured against this machine's SAPI5
stack, engine.stop() from another thread raised nothing, returned
cleanly, did nothing, and left the speaking thread wedged 13.5s into a
10s phrase. Milestone 3 worked around it by speaking one sentence at a
time and simply not starting the next one, which meant an interruption
landed at the next sentence boundary rather than immediately.

Playing audio through sounddevice does not have that problem.
Measured on the same machine: sd.stop() called from another thread
ended a 14.0s clip at 1.74s. So interruption is now immediate --
mid-word, not mid-paragraph -- and milestone 3's known "lands at a
sentence boundary" gap is closed as a side effect of changing engine.

## Why it still speaks in sentences

Not for interruption any more, but for latency. A whole reply rendered
in one request takes about 1.4s before the first word is audible, and
pyttsx3 started almost instantly; that regression is worth avoiding.
Measured render speed is 3-9x faster than real time, so rendering runs
one sentence ahead of playback in a background thread: the first
sentence costs ~0.9s, and every sentence after it is already waiting by
the time the previous one ends. No gaps, and a short first sentence
gets Jarvis talking quickly.

## Falling back

Any failure -- no network, a refused request, a decode error -- falls
back to Phase 2's speak_text(). edge-tts is a cloud service and this
assistant is meant to keep working; a robotic voice is a far better
outcome than silence. The fallback is logged once rather than per
sentence, since a network outage would otherwise produce one warning
per sentence for the rest of the session.
"""

import asyncio
import io
import logging
import os
import queue
import re
import sys
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "phaseTwo") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "phaseTwo"))

import voice_io  # noqa: E402

logger = logging.getLogger("voice")

# en-GB-RyanNeural: British, male, measured. Chosen by listening to
# candidates side by side rather than from the description -- see
# phaseEight/README.md's milestone 4 notes.
JARVIS_VOICE = os.getenv("JARVIS_VOICE", "en-GB-RyanNeural")

# Chosen by ear, comparing +0%, +8% and +15% on the same line.
# An earlier version guessed -5% -- a hedge between two samples that
# had actually been compared, and a value nobody had heard. It read as
# sluggish in use. Picked deliberately this time.
JARVIS_VOICE_RATE = os.getenv("JARVIS_VOICE_RATE", "+15%")
JARVIS_VOICE_PITCH = os.getenv("JARVIS_VOICE_PITCH", "+0Hz")

# Set to "pyttsx3" to force Phase 2's offline voice back on, for
# working offline or if the neural voice misbehaves.
TTS_ENGINE = os.getenv("TTS_ENGINE", "edge").strip().lower()

_fallback_warned = False


def _warn_fallback_once(reason: str) -> None:
    """One warning per process, not one per sentence -- a network
    outage would otherwise fill the log with the same line."""
    global _fallback_warned
    if not _fallback_warned:
        _fallback_warned = True
        logger.warning(f"Neural voice unavailable, falling back to the offline voice: {reason}")


async def _stream_audio(text: str) -> io.BytesIO:
    buffer = io.BytesIO()
    communicate = __import__("edge_tts").Communicate(
        text, JARVIS_VOICE, rate=JARVIS_VOICE_RATE, pitch=JARVIS_VOICE_PITCH
    )
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buffer.write(chunk["data"])
    buffer.seek(0)
    return buffer


# One sentence: everything up to and including its terminal
# punctuation, or a trailing fragment with none.
_SENTENCE = re.compile(r"[^.!?]+[.!?]*", re.DOTALL)

# The three places a vocative "sir" turns up, matched as a whole word so
# nothing inside another word is touched.
#
# The mid-sentence case is not just ", sir," -- the model reaches for a
# dash at least as often. Live testing found "Barely doing anything,
# sir - that's the short version" in 2 of 5 replies, which an earlier
# comma-only pattern sailed straight past. The lookahead therefore
# accepts any punctuation that CONTINUES a sentence (another comma, a
# dash of either width, a semicolon, a colon) while leaving a
# sentence-ENDING . ! ? to the trailing rule below.
_SIR_MIDDLE = re.compile(r"\s*,\s*sir\b(?=\s*[,;:—–-])", re.IGNORECASE)
_SIR_LEADING = re.compile(r"^\s*sir\s*,\s*", re.IGNORECASE)
_SIR_TRAILING = re.compile(r",?\s*\bsir\b\s*$", re.IGNORECASE)


def _move_sir_to_end(sentence: str) -> str:
    """Rewrites one sentence so any vocative "sir" sits at its end, with
    no comma before it."""
    # Whatever separated this sentence from the previous one is kept and
    # put back, or sentences run together ("...sir.The machine...").
    leading = sentence[: len(sentence) - len(sentence.lstrip())]
    stripped = sentence.strip()

    match = re.search(r"[.!?]+$", stripped)
    punctuation = match.group(0) if match else ""
    body = stripped[: len(stripped) - len(punctuation)]

    body, mid = _SIR_MIDDLE.subn("", body)
    body, lead = _SIR_LEADING.subn("", body)
    body, trail = _SIR_TRAILING.subn("", body)
    if not (mid or lead or trail):
        return sentence

    # Removing a mid-sentence ", sir," leaves a doubled space behind.
    body = re.sub(r"[ \t]{2,}", " ", body).strip().rstrip(",").strip()
    if not body:
        # The sentence was nothing but the address ("Sir.") -- leave it.
        return sentence
    if lead and body[0].islower():
        # Removing a leading "Sir, " leaves the next word lowercase.
        body = body[0].upper() + body[1:]

    return f"{leading}{body} sir{punctuation or '.'}"


# Markdown the model still emits despite being told not to, and which a
# speech engine reads out as literal characters rather than as emphasis.
_MD_BOLD_ITALIC = re.compile(r"\*{1,3}(?=\S)(.+?)(?<=\S)\*{1,3}", re.DOTALL)
_MD_CODE = re.compile(r"`{1,3}(?=\S)(.+?)(?<=\S)`{1,3}", re.DOTALL)
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_MD_BULLET = re.compile(r"^\s{0,3}[-*+]\s+", re.MULTILINE)


def strip_markdown(text: str) -> str:
    """Removes markdown markers that a speech engine would read aloud.

    edge-tts does not interpret markdown -- it reads the characters. So
    "that is *hard*" is spoken with the asterisks pronounced rather than
    with any change of emphasis, and a bulleted list becomes a series of
    "dash" utterances.

    The system prompt already forbids markdown, and mostly that works.
    But it leaks in exactly the places a model reaches for emphasis
    hardest -- a warning, a file path in backticks, one stressed word --
    which are also the places where being read a punctuation mark is
    most jarring. Same reasoning as the "sir" rule: the prompt asks, and
    code guarantees.

    Only the markers are removed, never the words between them. Losing
    emphasis is a small cost; losing content would not be.
    """
    if not text:
        return ""
    text = _MD_BOLD_ITALIC.sub(r"\1", text)
    text = _MD_CODE.sub(r"\1", text)
    text = _MD_HEADING.sub("", text)
    text = _MD_BULLET.sub("", text)
    return text


def for_speech(text: str) -> str:
    """Normalises how "sir" is addressed, just before the text is spoken.

    Two rules, both of which exist because this is read aloud rather
    than read on a page:

      1. A vocative "sir" always lands at the END of its sentence.
      2. It never has a comma before it.

    edge-tts renders a comma as an audible pause, so "the machine is in
    fine spirits, sir" comes out as "...fine spirits [pause] sir", and a
    mid-sentence "currently open, sir, eight windows" gets two pauses
    around it. Compared side by side out loud, the end-of-sentence form
    was clearly cleaner, which is the whole reason this exists.

    Done in code rather than by prompting because prompting lost.
    Instructed explicitly to avoid the comma and put "sir" at the end,
    then tested live, DeepSeek still produced ", sir" in 4 of 5 replies
    -- it treats that as correct English punctuation and reverts to it
    however the instruction is worded. A deterministic rewrite wins
    every time and costs nothing. The prompt still asks for the same
    thing, so the two agree rather than fight; this is the backstop.

    Deliberately a MOVE and not a deletion: dropping a mid-sentence
    "sir" would silently cost the reply its only use of the address and
    quietly erode the character over a session, where relocating it
    keeps exactly what the model wrote and only changes where it sits.
    """
    if not text:
        return ""
    # Markdown first: stripping it can change where a sentence begins,
    # so the "sir" pass should see the text as it will actually be
    # spoken rather than as the model punctuated it.
    text = strip_markdown(text)
    if not re.search(r"\bsir\b", text, re.IGNORECASE):
        return text  # nothing more to do, and the common case
    return "".join(_move_sir_to_end(s) if s.strip() else s for s in _SENTENCE.findall(text))


def render(text: str):
    """Renders one piece of text to (samples, samplerate).

    Returns None on any failure rather than raising -- callers fall
    back to the offline voice, which is the whole point of not letting
    a cloud dependency become a hard one.
    """
    if not text or not text.strip():
        return None
    try:
        buffer = asyncio.run(_stream_audio(for_speech(text)))
        samples, rate = sf.read(buffer, dtype="float32")
        if samples.size == 0:
            return None
        return samples, rate
    except Exception as e:
        _warn_fallback_once(str(e))
        return None


def _play(samples, rate, should_stop: threading.Event) -> bool:
    """Plays one clip. Returns False if it was cut short.

    Waits in short slices rather than a single sd.wait() so a stop is
    noticed promptly even when nothing else calls sd.stop() for us.
    """
    try:
        sd.play(samples, rate)
    except Exception as e:
        _warn_fallback_once(str(e))
        return False

    try:
        while sd.get_stream().active:
            if should_stop is not None and should_stop.is_set():
                sd.stop()
                return False
            sd.sleep(30)
    except Exception:
        # get_stream() raises if playback already finished -- not an error.
        pass
    return not (should_stop is not None and should_stop.is_set())


def speak(text: str, chunks: list = None, should_stop: threading.Event = None,
          on_chunk=None) -> bool:
    """
    Speaks `text` in the Jarvis voice. Returns True if it finished,
    False if it was interrupted or fell back.

    `chunks`, if given, is the text already split into the pieces to
    speak (the caller usually has them). `on_chunk` is called with each
    piece just before it is spoken, so a barge-in listener can know
    what is currently in the air.

    Rendering runs one piece ahead of playback in a background thread --
    see the module docstring for why.
    """
    pieces = chunks if chunks is not None else [text]
    pieces = [p for p in pieces if p and p.strip()]
    if not pieces:
        return True

    rendered = queue.Queue(maxsize=2)

    def producer():
        for piece in pieces:
            if should_stop is not None and should_stop.is_set():
                break
            rendered.put((piece, render(piece)))
        rendered.put(None)  # sentinel

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    completed = True
    try:
        while True:
            item = rendered.get()
            if item is None:
                break
            piece, audio = item

            if should_stop is not None and should_stop.is_set():
                completed = False
                break

            if on_chunk:
                on_chunk(piece)

            if audio is None:
                # This piece failed to render -- speak it with the
                # offline voice rather than dropping it. A reply that
                # silently loses a sentence is worse than one sentence
                # in the wrong voice.
                try:
                    voice_io.speak_text(piece)
                except Exception as e:
                    logger.warning(f"Fallback speech also failed: {e}")
                continue

            samples, rate = audio
            if not _play(samples, rate, should_stop):
                completed = False
                break
    finally:
        if should_stop is not None:
            should_stop.set()  # unblock the producer if we bailed early
        # Drain so the producer thread can exit rather than blocking on
        # a full queue for the life of the process.
        try:
            while True:
                rendered.get_nowait()
        except queue.Empty:
            pass

    return completed


def is_available() -> bool:
    """Whether the neural voice should be used at all."""
    if TTS_ENGINE != "edge":
        return False
    try:
        __import__("edge_tts")
        return True
    except ImportError:
        return False
