"""
test_voice_io.py — fully mocked. Running this suite requires no microphone,
no GPU, no downloaded Whisper model, and no real TTS driver. That's
deliberate: Phase 1 established the pattern of testing against mocked
externals, and it matters even more here since audio hardware isn't
available in CI or on a fresh machine.
"""

import sys
import types

import numpy as np
import pytest
from unittest.mock import MagicMock

import voice_io


# ---------- record_until_silence ----------

def _make_fake_stream(chunks):
    """chunks: list of numpy arrays returned in sequence from stream.read()."""
    stream = MagicMock()
    stream.read = MagicMock(side_effect=[(c, False) for c in chunks])
    return stream


def test_record_until_silence_success(monkeypatch):
    loud = np.full((1600, 1), 5000, dtype=np.int16)
    quiet = np.zeros((1600, 1), dtype=np.int16)
    silence_chunks_needed = int(voice_io.SILENCE_DURATION / voice_io.CHUNK_DURATION)
    chunks = [loud] + [quiet] * (silence_chunks_needed + 1)
    stream = _make_fake_stream(chunks)

    monkeypatch.setattr(voice_io.sd, "InputStream", MagicMock(return_value=stream))

    import os
    path = voice_io.record_until_silence()
    try:
        assert path.endswith(".wav")
        assert os.path.exists(path)
    finally:
        os.remove(path)


def test_record_until_silence_no_speech_times_out(monkeypatch):
    quiet = np.zeros((1600, 1), dtype=np.int16)
    stream = MagicMock()
    stream.read = MagicMock(return_value=(quiet, False))
    monkeypatch.setattr(voice_io.sd, "InputStream", MagicMock(return_value=stream))

    with pytest.raises(voice_io.AudioRecordingError, match="No speech detected"):
        voice_io.record_until_silence(max_seconds=0.3)


def test_record_until_silence_pre_speech_timeout_expires_before_any_speech(monkeypatch):
    """pre_speech_timeout, not max_seconds, should govern the wait when
    no speech ever starts."""
    quiet = np.zeros((1600, 1), dtype=np.int16)
    stream = MagicMock()
    stream.read = MagicMock(return_value=(quiet, False))
    monkeypatch.setattr(voice_io.sd, "InputStream", MagicMock(return_value=stream))

    with pytest.raises(voice_io.AudioRecordingError, match="No speech detected"):
        voice_io.record_until_silence(max_seconds=30.0, pre_speech_timeout=0.3)


def test_record_until_silence_does_not_cut_off_speech_started_near_pre_speech_deadline(monkeypatch):
    """Regression test: previously, max_seconds was a single hard cap
    covering the *entire* recording, so speech that started near the
    end of a short pre_speech_timeout window (Phase 7's follow-up
    window) got cut off mid-sentence. Once speech is detected, the
    longer max_seconds cap should take back over instead -- proven here
    by checking the recording runs through its full, naturally-ending
    chunk sequence (stopped by trailing silence, not by the short
    pre-speech window) rather than being cut short."""
    import itertools

    fake_clock = itertools.count(0.0, 0.5)
    monkeypatch.setattr(voice_io.time, "time", lambda: next(fake_clock))

    loud = np.full((1600, 1), 5000, dtype=np.int16)
    quiet = np.zeros((1600, 1), dtype=np.int16)
    silence_chunks_needed = int(voice_io.SILENCE_DURATION / voice_io.CHUNK_DURATION)
    # Two quiet chunks, THEN speech (read while elapsed is still under
    # the 2.0s pre_speech_timeout), THEN enough trailing silence to stop
    # naturally -- which takes the fake clock well past 2.0s.
    chunks = [quiet, quiet, loud] + [quiet] * (silence_chunks_needed + 1)
    stream = _make_fake_stream(chunks)
    monkeypatch.setattr(voice_io.sd, "InputStream", MagicMock(return_value=stream))

    import os
    path = voice_io.record_until_silence(max_seconds=30.0, pre_speech_timeout=2.0)
    try:
        # 2 quiet + 1 loud + exactly silence_chunks_needed trailing quiet
        # chunks were consumed before the natural silence-based stop --
        # i.e. it ran well past the 2.0s pre_speech_timeout instead of
        # being cut off by it once speech had started.
        assert stream.read.call_count == 3 + silence_chunks_needed
    finally:
        os.remove(path)


def test_record_until_silence_mic_open_failure(monkeypatch):
    monkeypatch.setattr(
        voice_io.sd, "InputStream", MagicMock(side_effect=OSError("no device"))
    )
    with pytest.raises(voice_io.AudioRecordingError, match="Could not open microphone"):
        voice_io.record_until_silence()


def test_record_until_silence_read_error(monkeypatch):
    stream = MagicMock()
    stream.read = MagicMock(side_effect=OSError("device disconnected"))
    monkeypatch.setattr(voice_io.sd, "InputStream", MagicMock(return_value=stream))

    with pytest.raises(voice_io.AudioRecordingError, match="Error reading"):
        voice_io.record_until_silence()


# ---------- transcribe_audio ----------

def test_transcribe_audio_success(monkeypatch):
    fake_model = MagicMock()
    fake_model.transcribe.return_value = {"text": "  hello jarvis  "}
    fake_whisper = types.SimpleNamespace(load_model=MagicMock(return_value=fake_model))
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    voice_io._whisper_model = None

    assert voice_io.transcribe_audio("fake.wav") == "hello jarvis"


def test_transcribe_audio_empty_result(monkeypatch):
    fake_model = MagicMock()
    fake_model.transcribe.return_value = {"text": "   "}
    fake_whisper = types.SimpleNamespace(load_model=MagicMock(return_value=fake_model))
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    voice_io._whisper_model = None

    with pytest.raises(voice_io.TranscriptionError, match="empty"):
        voice_io.transcribe_audio("fake.wav")


def test_transcribe_audio_model_load_failure(monkeypatch):
    fake_whisper = types.SimpleNamespace(
        load_model=MagicMock(side_effect=RuntimeError("model not found"))
    )
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    voice_io._whisper_model = None

    with pytest.raises(voice_io.TranscriptionError, match="Could not load Whisper model"):
        voice_io.transcribe_audio("fake.wav")


def test_transcribe_audio_transcription_failure(monkeypatch):
    fake_model = MagicMock()
    fake_model.transcribe.side_effect = RuntimeError("bad audio")
    fake_whisper = types.SimpleNamespace(load_model=MagicMock(return_value=fake_model))
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    voice_io._whisper_model = None

    with pytest.raises(voice_io.TranscriptionError, match="Transcription failed"):
        voice_io.transcribe_audio("fake.wav")


# ---------- speak_text ----------

def _fake_voice(voice_id, languages, name):
    v = types.SimpleNamespace()
    v.id = voice_id
    v.languages = languages
    v.name = name
    return v


def test_speak_text_success(monkeypatch):
    fake_engine = MagicMock()
    fake_engine.getProperty.return_value = [
        _fake_voice("en-voice-id", ["en-US"], "Microsoft Zira Desktop - English"),
        _fake_voice("ja-voice-id", ["ja-JP"], "Microsoft Haruka Desktop - Japanese"),
    ]
    fake_pyttsx3 = types.SimpleNamespace(init=MagicMock(return_value=fake_engine))
    monkeypatch.setitem(sys.modules, "pyttsx3", fake_pyttsx3)
    voice_io._tts_engine = None

    voice_io.speak_text("hello there")
    fake_engine.say.assert_called_once_with("hello there")
    fake_engine.runAndWait.assert_called_once()
    fake_engine.setProperty.assert_called_once_with("voice", "en-voice-id")


def test_speak_text_picks_japanese_voice_for_cjk_text(monkeypatch):
    fake_engine = MagicMock()
    fake_engine.getProperty.return_value = [
        _fake_voice("en-voice-id", ["en-US"], "Microsoft Zira Desktop - English"),
        _fake_voice("ja-voice-id", ["ja-JP"], "Microsoft Haruka Desktop - Japanese"),
    ]
    fake_pyttsx3 = types.SimpleNamespace(init=MagicMock(return_value=fake_engine))
    monkeypatch.setitem(sys.modules, "pyttsx3", fake_pyttsx3)
    voice_io._tts_engine = None

    voice_io.speak_text("ジャービスです。")
    fake_engine.setProperty.assert_called_once_with("voice", "ja-voice-id")


def test_speak_text_no_matching_voice_falls_back_gracefully(monkeypatch):
    fake_engine = MagicMock()
    fake_engine.getProperty.return_value = [
        _fake_voice("en-voice-id", ["en-US"], "Microsoft Zira Desktop - English"),
    ]
    fake_pyttsx3 = types.SimpleNamespace(init=MagicMock(return_value=fake_engine))
    monkeypatch.setitem(sys.modules, "pyttsx3", fake_pyttsx3)
    voice_io._tts_engine = None

    # No Japanese voice installed — should still speak (with whatever voice
    # is already selected) rather than raising.
    voice_io.speak_text("ジャービスです。")
    fake_engine.say.assert_called_once_with("ジャービスです。")
    fake_engine.setProperty.assert_not_called()


def test_speak_text_init_failure(monkeypatch):
    fake_pyttsx3 = types.SimpleNamespace(init=MagicMock(side_effect=RuntimeError("no driver")))
    monkeypatch.setitem(sys.modules, "pyttsx3", fake_pyttsx3)
    voice_io._tts_engine = None

    with pytest.raises(voice_io.SpeechError, match="Could not initialize TTS engine"):
        voice_io.speak_text("hi")


def test_speak_text_empty_string_is_noop(monkeypatch):
    fake_pyttsx3 = types.SimpleNamespace(init=MagicMock())
    monkeypatch.setitem(sys.modules, "pyttsx3", fake_pyttsx3)
    voice_io._tts_engine = None

    voice_io.speak_text("   ")
    fake_pyttsx3.init.assert_not_called()