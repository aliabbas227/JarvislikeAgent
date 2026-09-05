"""
interrupt.py -- Phase 8, milestone 3: barge-in.

Say the wake word while Jarvis is talking and it stops talking.

This is the first thing in this project that genuinely needs two things
happening at once. Phase 7's orchestrator says so in its own module
docstring -- "deliberately synchronous, not asyncio... worth revisiting
once a later milestone needs something running *while* still listening
(e.g. a spoken 'stop' during a long TTS reply)". This is that
milestone.

## How it works

Two independent halves:

  SPEAKING  -- voice.py renders the reply and plays it (see there for
               why it is still split into sentences: latency, not
               interruption).
  LISTENING -- a background thread runs openWakeWord over the
               microphone, and on a detection calls on_trigger, which
               stops playback immediately.

## Interruption is immediate, but only since milestone 4

Originally it was not, and the reason is worth keeping. pyttsx3 cannot
be stopped: measured against this machine's SAPI5 stack, engine.stop()
from another thread raised no exception, returned cleanly, did nothing
at all, and left the speaking thread wedged 13.5 seconds into a phrase
that takes 10 seconds to say. So milestone 3 worked around it by
speaking one sentence at a time and simply not starting the next one,
which meant an interruption landed at the next sentence boundary.

Milestone 4 replaced the whole output path with neural TTS played
through sounddevice (voice.py), and sounddevice playback CAN be stopped
from another thread -- measured, a 14.0s clip ended at 1.74s. So the
listener now cuts playback mid-word via on_trigger, and milestone 3's
"lands at a sentence boundary" gap closed as a side effect of changing
engine rather than by being solved directly.

The sentence-at-a-time fallback still exists, and is still exactly the
old behavior, for when the neural voice is unavailable (see _say).

## Why openWakeWord and not Whisper

The first version of this recorded candidate audio and transcribed it
with Whisper, matching the text against an interrupt word like "skip".
It was rebuilt after that approach failed on real hardware and kept
failing as each fault was fixed. The full history is in the README; the
short version is that six genuine bugs were found and fixed, and it
STILL did not work, because the remaining problem was not a bug:

  Whisper is an utterance transcriber. Barge-in needs a keyword spotter
  for one short word, spoken quietly, over the top of other speech --
  precisely its weak spot. Measured: a person has to be about 3x louder
  than Jarvis before their word survives into the transcript at all,
  and even a correctly-captured clip that transcribes as "Skip." on its
  own came back as "Quick TENSCHES Okoi" from inside the listener.

openWakeWord is built for exactly this job, and this project already
depends on it and already proved it works in Phase 3. Tested on an
identical audio mixture at this machine's own measured levels:

    the wake word over Jarvis's own voice   -> score 0.999  (detected)
    Jarvis's reply alone, no wake word      -> score 0.000  (silent)

No transcription, so no Whisper latency and no repeated CPU work during
every reply; no loudness threshold to calibrate; and it degrades far
more gracefully toward the open-microphone setup this is heading for.

The tradeoff, stated plainly: openWakeWord ships pre-trained models for
a fixed set of phrases, and "skip" is not one of them. The interrupt
phrase is therefore the wake word itself -- which is also how
commercial assistants handle barge-in.

## Its own model instance, and reset() every time

The listener builds a SECOND openWakeWord Model rather than sharing the
orchestrator's. Phase 7's milestone 2 spent three real-hardware
debugging passes discovering that a Model carries internal state
between calls: its prediction buffer stays full of high scores after a
genuine detection, so a reused model "remembers" and re-fires. Sharing
one instance between the wake gate and this listener would put that
same stale state on both sides of the same turn.

For the same reason, reset() is called at the start of every listening
session, not just at construction.

## What happens after an interrupt

Nothing dramatic: Jarvis stops talking and the turn ends normally, so
Phase 7's follow-up window opens right afterward and the person can
just say what they actually wanted. That required no changes to the
conversation loop at all -- being cut off looks the same to it as
finishing early. It also composes neatly with the phrase being the wake
word: you say "hey jarvis", it stops, and the follow-up window is
already listening for what comes next.
"""

import logging
import os
import re
import sys
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd

# Resolve Phase 2's voice_io and Phase 3's wake_word ourselves rather
# than relying on app.py having already imported orchestrator (which
# puts both on sys.path as a side effect). That happened to work only
# because app.py's imports are alphabetical and `orchestrator` sorts
# before `interrupt` -- a module that breaks when its importer's import
# block is reordered is not a dependency worth having, and it also made
# this module impossible to import on its own in a test.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _path in (_PROJECT_ROOT / "phaseTwo", _PROJECT_ROOT / "phaseThree"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import voice_io  # noqa: E402  -- must follow the sys.path setup above

import voice  # noqa: E402  -- Phase 8's neural TTS, falls back to voice_io

logger = logging.getLogger("interrupt")

BARGE_IN_ENABLED = os.getenv("BARGE_IN_ENABLED", "true").lower() not in ("false", "0", "no")

# openWakeWord's expected frame size: 1280 samples = 80ms at 16kHz.
# Same constants Phase 3's wake_word.py and Phase 7's wake_gate.py use.
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1280

# Confidence needed to treat a frame as the wake word. Defaults to the
# same value Phase 3 uses for waking, and the margin measured here is
# enormous either way -- 0.999 for a real utterance against 0.000 for
# Jarvis's own speech -- so this is not a knob that should need
# touching. Separate from WAKE_THRESHOLD so barge-in can be made
# deliberately harder to trigger than waking, without making it harder
# to wake Jarvis up in the first place.
BARGE_IN_THRESHOLD = float(os.getenv("BARGE_IN_THRESHOLD", os.getenv("WAKE_THRESHOLD", "0.5")))

# Below this length, speak normally without starting a listener at all.
# A four-word confirmation is over before anyone could react to it, and
# opening a mic stream for it is pure cost. This is also what keeps
# "Goodbye." and the confirmation heads-up out of the interruptible
# path without this module needing to know anything about its call
# sites.
BARGE_IN_MIN_CHARS = int(os.getenv("BARGE_IN_MIN_CHARS", "80"))

# Chunks longer than this get split further (on commas and semicolons)
# so a single run-on sentence can't block interruption for 15 seconds.
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "220"))


def split_into_speech_chunks(text: str) -> list:
    """
    Splits a reply into the pieces it will be spoken in -- which is also
    how often barge-in can take effect, since the interrupt flag is
    checked between them.

    Sentence boundaries first, then commas and semicolons for anything
    still too long. A single 40-word sentence spoken whole would take
    around fifteen seconds, during which the wake word would do
    nothing, and a barge-in that ignores you for fifteen seconds is not
    barge-in.
    """
    chunks = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= MAX_CHUNK_CHARS:
            chunks.append(sentence)
            continue

        # Still too long -- break on softer punctuation, accumulating
        # pieces so we don't produce a stream of two-word fragments,
        # which sound worse than one long sentence.
        current = ""
        for piece in re.split(r"(?<=[,;:])\s+", sentence):
            if current and len(current) + len(piece) + 1 > MAX_CHUNK_CHARS:
                chunks.append(current.strip())
                current = piece
            else:
                current = f"{current} {piece}".strip()
        if current:
            chunks.append(current.strip())
    return chunks


def _default_input_name() -> str:
    """Name of whatever Windows currently calls the default microphone.
    Diagnostic only, and never raises -- it exists so a "heard nothing"
    warning can say WHICH device heard nothing, which is the difference
    between a real problem and a headset being switched off.

    Worth having because this machine's default input moved three times
    during one debugging session, including onto a powered-off headset,
    while every phase opens the mic with no device argument."""
    try:
        return sd.query_devices(kind="input")["name"]
    except Exception:
        return "(unknown)"


class InterruptListener:
    """
    Listens for the wake word on its own thread while the main thread
    is speaking, using openWakeWord.

    Never raises into the caller. A microphone that cannot be opened, a
    model that fails to build, an audio driver that disappears -- all of
    it degrades to "barge-in isn't available for this reply", which is
    the same behavior as BARGE_IN_ENABLED=false and strictly better
    than taking down a working voice loop over an optional convenience.
    Same reasoning Phase 7 used for the memory store: an enhancement
    fails soft, a load-bearing component fails loud.
    """

    def __init__(self, threshold: float = None):
        self._threshold = BARGE_IN_THRESHOLD if threshold is None else threshold
        self._triggered = threading.Event()
        self._stop_requested = threading.Event()
        self._thread = None
        self._model = None
        self._model_name = None
        self._currently_speaking = ""
        self._best_score = 0.0
        # Called the instant the wake word is heard. Lets the caller
        # stop audio playback mid-word rather than waiting for the
        # current sentence to finish -- see speak_interruptibly().
        self.on_trigger = None

    @property
    def triggered(self) -> bool:
        return self._triggered.is_set()

    @property
    def best_score(self) -> float:
        """Highest confidence seen this run. Diagnostic: it separates
        "heard you and wasn't sure" from "heard nothing at all", which
        were indistinguishable the first time barge-in failed."""
        return self._best_score

    def now_speaking(self, text: str) -> None:
        """Tells the listener what Jarvis is saying right now.

        Only used for one narrow guard: openWakeWord scored Jarvis's own
        speech at 0.000 in testing, so echo is a non-issue in general --
        but a reply that literally contains the wake word ("I'm Jarvis,
        your assistant") is the one case that could plausibly trigger
        it, so detection is skipped while such a sentence is in the air.
        Scoped per chunk rather than per reply so one mention doesn't
        disable barge-in for a whole answer.
        """
        self._currently_speaking = text or ""

    def _get_model(self):
        """Builds this listener's OWN openWakeWord model, once, lazily.

        Deliberately not the orchestrator's instance. A Model carries
        internal state between calls -- Phase 7's milestone 2 spent
        three real-hardware passes discovering that its prediction
        buffer stays full of high scores after a detection and makes it
        re-fire -- so sharing one between the wake gate and this
        listener would put that stale state on both sides of a turn.
        """
        if self._model is None:
            from wake_word import MODEL_NAME, build_model

            self._model = build_model()
            self._model_name = MODEL_NAME
        return self._model

    def start(self) -> None:
        self._triggered.clear()
        self._stop_requested.clear()
        self._best_score = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signals the thread and waits for it to actually finish.

        The join is not optional. The next thing the orchestrator does
        after speaking is open its own InputStream for the follow-up
        window, and two streams on one device is exactly the class of
        problem Phase 7's milestone 2 spent three real-hardware
        debugging passes on. This one releases the mic before returning.
        """
        self._stop_requested.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                logger.warning("Interrupt listener did not stop cleanly within 3s.")
            self._thread = None

    def _run(self) -> None:
        try:
            self._listen()
        except Exception as e:
            logger.debug(f"Interrupt listener stopped on error (non-fatal): {e}")

    def _listen(self) -> None:
        try:
            model = self._get_model()
        except Exception as e:
            logger.debug(f"Barge-in unavailable, could not build wake model: {e}")
            return

        # Clear any state left over from a previous detection. Phase 7's
        # milestone 2 bug in one line: without this, the model keeps
        # "remembering" the last time it fired.
        try:
            model.reset()
        except Exception:
            pass

        try:
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE, blocksize=CHUNK_SAMPLES, channels=1, dtype="int16"
            )
            stream.start()
        except Exception as e:
            logger.debug(f"Barge-in unavailable, could not open microphone: {e}")
            return

        try:
            while not self._stop_requested.is_set():
                try:
                    frame, _ = stream.read(CHUNK_SAMPLES)
                except Exception:
                    return

                try:
                    scores = model.predict(frame[:, 0].astype(np.int16))
                except Exception as e:
                    logger.debug(f"Barge-in scoring failed (non-fatal): {e}")
                    return

                score = scores.get(self._model_name, 0.0)
                self._best_score = max(self._best_score, score)

                if score < self._threshold:
                    continue
                if self._wake_word_is_in_what_jarvis_is_saying():
                    logger.debug(
                        f"barge-in: ignoring score {score:.3f} -- Jarvis is saying the wake word itself"
                    )
                    continue

                logger.info(f"Barge-in: wake word heard (score {score:.3f}), stopping playback.")
                self._triggered.set()
                if self.on_trigger:
                    try:
                        self.on_trigger()
                    except Exception as e:
                        logger.debug(f"on_trigger failed (non-fatal): {e}")
                return
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def _wake_word_is_in_what_jarvis_is_saying(self) -> bool:
        """True when the sentence currently being spoken contains the
        wake word, so Jarvis saying its own name can't interrupt it."""
        name = (self._model_name or "").replace("_", " ").lower()
        spoken = self._currently_speaking.lower()
        return bool(name) and any(word in spoken for word in name.split() if len(word) > 3)


def _say(text: str, chunks: list, should_stop=None, on_chunk=None) -> None:
    """Speaks via the neural voice, falling back to Phase 2's offline
    one when it isn't available (no network, or TTS_ENGINE=pyttsx3).

    The fallback is not interruptible mid-sentence -- pyttsx3 cannot be
    stopped from another thread -- so it speaks chunk by chunk and
    checks between them, which is exactly milestone 3's original
    behavior. Degrading from instant interruption to sentence-boundary
    interruption is a much better failure than not speaking at all.
    """
    if voice.is_available():
        voice.speak(text, chunks=chunks, should_stop=should_stop, on_chunk=on_chunk)
        return

    if should_stop is None:
        # Nothing is listening, so chunking would only add pauses. Say
        # it in one call, exactly as Phase 2 always did.
        voice_io.speak_text(voice.for_speech(text))
        return

    for chunk in chunks or [text]:
        if should_stop.is_set():
            return
        if on_chunk:
            on_chunk(chunk)
        voice_io.speak_text(voice.for_speech(chunk))


def speak_interruptibly(text: str) -> None:
    """
    Drop-in replacement for voice_io.speak_text(), registered onto
    Phase 7's speak_safe() via orchestrator.register_speaker().

    Speaks `text` in chunks, listening between them for the wake word.
    Short replies skip the whole apparatus (see BARGE_IN_MIN_CHARS) and
    are spoken exactly as before.

    Deliberately does not report whether it was interrupted: speak_safe
    returns nothing, and the conversation loop treats a cut-off reply
    the same as a completed one, which is what makes this milestone a
    genuine drop-in rather than a rewrite of run_turn(). See the
    README's known gaps for the one place that costs something.
    """
    if not text or not text.strip():
        return

    chunks = split_into_speech_chunks(text)

    if not BARGE_IN_ENABLED or len(text) < BARGE_IN_MIN_CHARS:
        _say(text, chunks)
        return

    listener = InterruptListener()
    stop_playback = threading.Event()
    # Interruption is now immediate rather than at the next sentence:
    # sounddevice playback CAN be stopped from another thread (measured:
    # a 14.0s clip ended at 1.74s), which pyttsx3 could not. See
    # voice.py's docstring.
    listener.on_trigger = stop_playback.set

    listener.start()
    try:
        _say(text, chunks, should_stop=stop_playback, on_chunk=listener.now_speaking)
        if listener.triggered:
            logger.info("Stopped speaking on barge-in.")
    finally:
        listener.stop()
        if not listener.triggered and listener.best_score == 0.0:
            # Distinguishes "heard you and wasn't confident" from "heard
            # nothing at all". The second was invisible the first time
            # barge-in failed, and was the actual cause: the default
            # input device had moved to a powered-off headset.
            logger.debug(
                f"Barge-in scored 0.000 for this entire reply on input device "
                f"{_default_input_name()!r}. If interrupting isn't working, that mic "
                "may be the wrong device, muted, or off."
            )
