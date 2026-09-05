"""
voice_io.py — Phase 2: audio input/output primitives.

Deliberately kept separate from any LLM logic. This module answers exactly
three questions:
    1. How do I capture speech from the mic without the user pressing a button?
    2. How do I turn that into text?
    3. How do I turn text back into speech?

Design choice: BATCH, not streaming. We record until the user stops talking
(silence-detection via RMS amplitude), then transcribe the whole clip at
once. This is simpler to reason about and debug than live/streaming
transcription, at the cost of a beat of latency after you stop speaking.
Streaming STT is a reasonable Phase-2.5 upgrade once this works reliably —
don't reach for it until the simple version is solid.

Whisper and pyttsx3 are imported lazily (inside functions) rather than at
module load time. This keeps `import voice_io` cheap and makes the recording
logic testable even in environments without those packages installed.
"""

import logging
import os
import re
import tempfile
import time
import wave

import numpy as np
import sounddevice as sd

logger = logging.getLogger("voice_io")

SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 16000))
CHANNELS = 1
CHUNK_DURATION = 0.1  # seconds of audio analyzed per loop iteration
SILENCE_THRESHOLD = int(os.getenv("SILENCE_THRESHOLD", 500))  # RMS amplitude cutoff
SILENCE_DURATION = float(os.getenv("SILENCE_DURATION", 1.5))  # trailing silence to stop
MAX_RECORD_SECONDS = float(os.getenv("MAX_RECORD_SECONDS", 90))

_whisper_model = None
_tts_engine = None

# Matches Hiragana, Katakana, CJK ideographs, and Hangul. Used to pick a
# TTS voice that actually matches the reply's language — pyttsx3/SAPI5
# won't do this automatically, it just keeps using whatever voice is
# currently selected regardless of the text's script.
_CJK_PATTERN = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]")


class AudioRecordingError(Exception):
    """Raised when the microphone can't be opened, read, or captures nothing."""


class TranscriptionError(Exception):
    """Raised when Whisper can't load or can't produce usable text."""


class SpeechError(Exception):
    """Raised when TTS can't initialize or can't speak."""


def _rms(chunk: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(chunk.astype(np.float64)))))


def record_until_silence(
    sample_rate: int = SAMPLE_RATE,
    silence_threshold: int = SILENCE_THRESHOLD,
    silence_duration: float = SILENCE_DURATION,
    max_seconds: float = MAX_RECORD_SECONDS,
    pre_speech_timeout: float = None,
) -> str:
    """
    Records from the default microphone. Starts counting once speech is
    detected, stops once `silence_duration` seconds of near-silence follow.
    Hard-stops at `max_seconds` regardless (protects against a stuck-open mic
    or background noise that never drops below threshold).

    pre_speech_timeout, if given, caps only the time spent waiting for
    speech to START — once speech is first detected, `max_seconds` (the
    normal, longer hard cap) takes over instead, same as any other
    recording. Defaults to `max_seconds` itself when not given, which
    keeps this call's behavior identical to before this parameter
    existed: a single hard cap covering both "how long to wait for you
    to start talking" and "how long you're allowed to keep talking."

    Added for Phase 7's follow-up window: it wants a SHORT "did anyone
    start talking" timeout (a few seconds) but the SAME generous
    recording length as any other turn once someone actually starts.
    The old single-timeout behavior used the short window for both,
    which cut people off mid-sentence if they started talking anywhere
    other than right at the beginning of that short window.

    Returns the path to a temp WAV file the caller is responsible for
    deleting. Raises AudioRecordingError on any failure — never crashes the
    caller's loop.
    """
    if pre_speech_timeout is None:
        pre_speech_timeout = max_seconds

    chunk_frames = int(CHUNK_DURATION * sample_rate)
    frames = []
    speech_detected = False
    silence_chunks_needed = int(silence_duration / CHUNK_DURATION)
    silent_chunk_count = 0
    start_time = time.time()

    try:
        stream = sd.InputStream(samplerate=sample_rate, channels=CHANNELS, dtype="int16")
        stream.start()
    except Exception as e:
        raise AudioRecordingError(f"Could not open microphone: {e}") from e

    logger.info("Listening...")
    try:
        while True:
            deadline = max_seconds if speech_detected else pre_speech_timeout
            if time.time() - start_time > deadline:
                if speech_detected:
                    logger.warning("Max recording duration hit, stopping.")
                break
            try:
                chunk, _ = stream.read(chunk_frames)
            except Exception as e:
                raise AudioRecordingError(f"Error reading from microphone: {e}") from e

            frames.append(chunk.copy())
            level = _rms(chunk)

            if level >= silence_threshold:
                speech_detected = True
                silent_chunk_count = 0
            elif speech_detected:
                silent_chunk_count += 1
                if silent_chunk_count >= silence_chunks_needed:
                    break
    finally:
        stream.stop()
        stream.close()

    if not speech_detected:
        raise AudioRecordingError("No speech detected before timeout.")

    audio_data = np.concatenate(frames, axis=0)
    tmp_path = tempfile.mktemp(suffix=".wav")
    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # int16 = 2 bytes/sample
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

    return tmp_path


def _get_whisper_model(model_name: str = None):
    global _whisper_model
    if _whisper_model is None:
        import whisper  # local import: heavy, only needed here

        model_name = model_name or os.getenv("WHISPER_MODEL", "base")
        logger.info(f"Loading Whisper model '{model_name}' (first call only)...")
        _whisper_model = whisper.load_model(model_name)
    return _whisper_model


def transcribe_audio(wav_path: str, model_name: str = None) -> str:
    """Transcribes a WAV file to text. Raises TranscriptionError on failure
    or if the result is empty (e.g. silence that slipped past the threshold)."""
    try:
        model = _get_whisper_model(model_name)
    except Exception as e:
        raise TranscriptionError(f"Could not load Whisper model: {e}") from e

    language = os.getenv("WHISPER_LANGUAGE") or None  # None = auto-detect
    try:
        result = model.transcribe(wav_path, fp16=False, language=language)
    except Exception as e:
        raise TranscriptionError(f"Transcription failed: {e}") from e

    text = result.get("text", "").strip()
    if not text:
        raise TranscriptionError("Transcription returned empty text.")
    return text


def _new_tts_engine():
    import pyttsx3  # local import

    try:
        engine = pyttsx3.init()
        rate = os.getenv("TTS_RATE")
        if rate:
            engine.setProperty("rate", int(rate))
        return engine
    except Exception as e:
        raise SpeechError(f"Could not initialize TTS engine: {e}") from e


def _pick_voice_id(engine, text: str):
    """Picks an installed SAPI5 voice matching the text's apparent language.
    Currently only distinguishes CJK vs. everything-else-defaults-to-English
    — extend _CJK_PATTERN / add more script ranges here as you install more
    language packs. Falls back to whatever the engine already has selected
    if no clear match is found, and never raises (voice selection is a
    nice-to-have, not worth failing the whole speak_text call over)."""
    try:
        voices = list(engine.getProperty("voices") or [])

        def _matches(voice, prefix: str) -> bool:
            langs = [str(l).lower() for l in getattr(voice, "languages", [])]
            name = (getattr(voice, "name", "") or "").lower()
            return any(l.startswith(prefix) for l in langs) or prefix in name

        if _CJK_PATTERN.search(text):
            for v in voices:
                if _matches(v, "ja") or "japanese" in (getattr(v, "name", "") or "").lower():
                    return v.id
            # No Japanese voice installed — caller falls back to whatever's
            # already selected, which usually means English reading CJK
            # text badly. Better to know that than to silently mis-speak.
            logger.warning("No matching voice found for this text's script; using default voice.")
            return None

        for v in voices:
            if _matches(v, "en"):
                return v.id
        return None
    except Exception:
        # Voice selection is a nice-to-have — never let it break speak_text.
        return None


def speak_text(text: str) -> None:
    """Speaks `text` aloud via pyttsx3 (offline, SAPI5 on Windows). Silently
    no-ops on empty/whitespace-only input rather than erroring.

    A fresh engine is created per call rather than reused. pyttsx3's SAPI5
    driver on Windows has a known issue where a single engine instance
    stops producing audio after its first runAndWait() call in the same
    process — recreating it each time works around that reliably.

    Also picks a voice matching the text's script (see _pick_voice_id) —
    SAPI5 does not do this automatically on its own."""
    if not text or not text.strip():
        logger.warning("speak_text called with empty text, skipping.")
        return
    try:
        engine = _new_tts_engine()
        voice_id = _pick_voice_id(engine, text)
        if voice_id:
            engine.setProperty("voice", voice_id)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except SpeechError:
        raise
    except Exception as e:
        raise SpeechError(f"TTS playback failed: {e}") from e