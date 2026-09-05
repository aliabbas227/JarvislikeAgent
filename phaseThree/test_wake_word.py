"""
Mocked tests for wake_word.py.

No real mic, no real openWakeWord model, no real sounddevice stream is
used — everything is patched. This matches the Phase 2 testing pattern
(test_voice_io.py): tests should run in CI or on a machine with no audio
hardware at all.
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import wake_word


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fake_model(score_sequence):
    """
    Returns a MagicMock standing in for an openwakeword Model instance.
    .predict() is configured to return successive scores from
    score_sequence on each call (one call per audio chunk).
    """
    model = MagicMock()
    scores = iter(score_sequence)

    def fake_predict(audio):
        try:
            score = next(scores)
        except StopIteration:
            score = 0.0
        return {wake_word.MODEL_NAME: score}

    model.predict.side_effect = fake_predict
    return model


def run_callback_sequence(model, on_detect, score_sequence, cooldown_seconds=2.0,
                           time_values=None):
    """
    Drives wake_word.listen_loop's inner `callback` directly, without
    opening a real sounddevice.InputStream. We reach into listen_loop by
    calling it with a stream context manager mocked out, then invoking
    the callback it registers.
    """
    captured = {}

    class FakeStream:
        def __init__(self, *args, **kwargs):
            captured["callback"] = kwargs["callback"]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    fake_indata_frames = [
        np.zeros((wake_word.CHUNK_SAMPLES, 1), dtype=np.int16)
        for _ in score_sequence
    ]

    with patch.object(wake_word.sd, "InputStream", FakeStream), \
         patch.object(wake_word.sd, "sleep", side_effect=StopIteration):
        try:
            wake_word.listen_loop(model, on_detect, cooldown_seconds=cooldown_seconds)
        except StopIteration:
            pass  # listen_loop's `while True: sd.sleep(1000)` — we bail out immediately

    callback = captured["callback"]

    if time_values is None:
        # Default: evenly spaced, 0.1s apart. Offset from 0 so we never
        # collide with listen_loop's initial last_trigger_time=0 sentinel
        # (real time.monotonic() is never actually 0 at process start).
        time_values = [100.0 + i * 0.1 for i in range(len(fake_indata_frames))]

    for indata, t in zip(fake_indata_frames, time_values):
        with patch.object(wake_word.time, "monotonic", return_value=t):
            callback(indata, wake_word.CHUNK_SAMPLES, None, None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_below_threshold_never_fires():
    on_detect = MagicMock()
    model = make_fake_model([0.1, 0.2, 0.3, 0.49])

    run_callback_sequence(model, on_detect, score_sequence=[0.1, 0.2, 0.3, 0.49])

    on_detect.assert_not_called()


def test_above_threshold_fires_once():
    on_detect = MagicMock()
    scores = [0.6]
    model = make_fake_model(scores)

    run_callback_sequence(model, on_detect, score_sequence=scores)

    on_detect.assert_called_once()
    called_score = on_detect.call_args[0][0]
    assert called_score == pytest.approx(0.6)


def test_sustained_high_score_debounced_within_cooldown():
    """
    Simulates the real-world bug we hit: one utterance keeps the score
    above threshold for several consecutive frames. Without debounce this
    fires repeatedly; with the cooldown it should fire exactly once.
    """
    on_detect = MagicMock()
    scores = [0.9, 0.95, 1.0, 1.0, 0.98, 0.9]
    model = make_fake_model(scores)

    # All frames arrive within 1 second — well inside the 2s cooldown
    time_values = [100.0, 100.15, 100.3, 100.45, 100.6, 100.75]

    run_callback_sequence(
        model, on_detect, score_sequence=scores,
        cooldown_seconds=2.0, time_values=time_values,
    )

    on_detect.assert_called_once()


def test_second_utterance_after_cooldown_fires_again():
    on_detect = MagicMock()
    scores = [0.9, 0.9]  # two separate detections, far apart in time
    model = make_fake_model(scores)

    # First at t=100, second 5s later — outside the 2s cooldown window
    time_values = [100.0, 105.0]

    run_callback_sequence(
        model, on_detect, score_sequence=scores,
        cooldown_seconds=2.0, time_values=time_values,
    )

    assert on_detect.call_count == 2


def test_callback_handles_dropped_frame_status_gracefully(capsys):
    on_detect = MagicMock()
    scores = [0.1]
    model = make_fake_model(scores)

    captured = {}

    class FakeStream:
        def __init__(self, *args, **kwargs):
            captured["callback"] = kwargs["callback"]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    with patch.object(wake_word.sd, "InputStream", FakeStream), \
         patch.object(wake_word.sd, "sleep", side_effect=StopIteration):
        try:
            wake_word.listen_loop(model, on_detect)
        except StopIteration:
            pass

    callback = captured["callback"]
    fake_status = "input overflow"  # sounddevice passes a CallbackFlags-like status
    fake_indata = np.zeros((wake_word.CHUNK_SAMPLES, 1), dtype=np.int16)

    # Should not raise even though status is non-empty
    callback(fake_indata, wake_word.CHUNK_SAMPLES, None, fake_status)

    out = capsys.readouterr().out
    assert "Audio status" in out
    on_detect.assert_not_called()  # score was 0.1, below default threshold


def test_build_model_uses_onnx_inference_framework():
    with patch.object(wake_word, "Model") as MockModel:
        wake_word.build_model()
        MockModel.assert_called_once_with(
            wakeword_models=[wake_word.MODEL_NAME],
            inference_framework="onnx",
        )


def test_on_wake_detected_prints_score(capsys):
    wake_word.on_wake_detected(0.87)
    out = capsys.readouterr().out
    assert "WAKE WORD DETECTED" in out
    assert "0.87" in out