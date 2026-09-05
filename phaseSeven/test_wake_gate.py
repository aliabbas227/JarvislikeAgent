"""
test_wake_gate.py -- Phase 7 tests for wake_gate.WakeWordGate.

No real microphone or openwakeword model needed: sounddevice.InputStream
is replaced with a fake that captures the callback so tests can invoke
it directly with synthetic audio, and the wake-word model is a small
FakeModel double -- same pattern as Phase 3's injectable FakeModel.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

import wake_gate as wake_gate_module
from wake_gate import WakeWordGate


class FakeModel:
    """Returns a fixed score for model_name regardless of input audio,
    controllable per-test via .score. Mirrors the real openwakeword
    Model's interface, including reset() -- WakeWordGate.start() calls
    it on every (re)start to clear the real Model's internal
    prediction/preprocessor buffers (see start()'s docstring)."""

    def __init__(self, model_name: str, score: float = 0.0):
        self.model_name = model_name
        self.score = score
        self.predict_calls = 0
        self.reset_calls = 0

    def predict(self, audio):
        self.predict_calls += 1
        return {self.model_name: self.score}

    def reset(self):
        self.reset_calls += 1


class FakeStream:
    """Stand-in for sd.InputStream that captures the callback instead of
    opening a real device, so tests can fire it manually."""

    instances = []

    def __init__(self, samplerate, blocksize, channels, dtype, callback):
        self.callback = callback
        self.started = False
        self.stopped = False
        self.closed = False
        FakeStream.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def patch_stream(monkeypatch):
    FakeStream.instances = []
    monkeypatch.setattr(wake_gate_module.sd, "InputStream", FakeStream)
    yield


@pytest.fixture(autouse=True)
def patch_sleep(monkeypatch):
    """Replaces time.sleep with a spy across the whole module -- used
    by both wait_for_wake()'s polling loop and start()'s
    restart_settle_seconds delay. Autouse so no existing test suddenly
    takes real wall-clock time; tests targeting the settle delay itself
    assert against this spy instead of measuring elapsed time."""
    spy = MagicMock()
    monkeypatch.setattr(wake_gate_module.time, "sleep", spy)
    return spy


def _fire_chunk(gate: WakeWordGate, samples: int = 1280):
    chunk = np.zeros((samples, 1), dtype=np.int16)
    gate._callback(chunk, samples, None, None)


def test_start_opens_and_starts_a_stream():
    gate = WakeWordGate(FakeModel("hey_jarvis"), "hey_jarvis", threshold=0.5)
    gate.start()
    assert len(FakeStream.instances) == 1
    assert FakeStream.instances[0].started is True


def test_start_resets_the_model_every_time():
    """The real fix for the spurious-relisten bug: openWakeWord's Model
    is a single object reused for the whole process (see start()'s
    docstring) -- its internal prediction/preprocessor buffers must be
    cleared on every restart, not just the InputStream recreated,
    or a stale high-score buffer from a previous detection can
    resurface the moment scoring resumes."""
    model = FakeModel("hey_jarvis")
    gate = WakeWordGate(model, "hey_jarvis", threshold=0.5)
    gate.start()
    assert model.reset_calls == 1
    gate.stop()
    gate.start()
    assert model.reset_calls == 2


def test_stop_stops_and_closes_the_stream():
    gate = WakeWordGate(FakeModel("hey_jarvis"), "hey_jarvis", threshold=0.5)
    gate.start()
    gate.stop()
    stream = FakeStream.instances[0]
    assert stream.stopped is True
    assert stream.closed is True
    assert gate._stream is None


def test_stop_before_start_does_not_raise():
    gate = WakeWordGate(FakeModel("hey_jarvis"), "hey_jarvis", threshold=0.5)
    gate.stop()  # should be a no-op, not an error


def test_stop_is_idempotent():
    gate = WakeWordGate(FakeModel("hey_jarvis"), "hey_jarvis", threshold=0.5)
    gate.start()
    gate.stop()
    gate.stop()  # second call should not raise


def test_low_score_does_not_set_wake_event():
    model = FakeModel("hey_jarvis", score=0.1)
    gate = WakeWordGate(model, "hey_jarvis", threshold=0.5, startup_skip_frames=0)
    gate.start()
    _fire_chunk(gate)
    assert not gate._wake_event.is_set()


def test_high_score_sets_wake_event():
    model = FakeModel("hey_jarvis", score=0.9)
    gate = WakeWordGate(model, "hey_jarvis", threshold=0.5, startup_skip_frames=0)
    gate.start()
    _fire_chunk(gate)
    assert gate._wake_event.is_set()


def test_score_exactly_at_threshold_does_not_trigger():
    """Matches the '> threshold' (not >=) semantics carried over from
    Phase 3's wake_word.py."""
    model = FakeModel("hey_jarvis", score=0.5)
    gate = WakeWordGate(model, "hey_jarvis", threshold=0.5, startup_skip_frames=0)
    gate.start()
    _fire_chunk(gate)
    assert not gate._wake_event.is_set()


def test_missing_model_name_in_predictions_defaults_to_zero_score():
    class EmptyModel:
        def predict(self, audio):
            return {}

        def reset(self):
            pass

    gate = WakeWordGate(EmptyModel(), "hey_jarvis", threshold=0.5, startup_skip_frames=0)
    gate.start()
    _fire_chunk(gate)
    assert not gate._wake_event.is_set()


def test_wait_for_wake_returns_once_event_is_set_and_clears_it():
    model = FakeModel("hey_jarvis", score=0.9)
    gate = WakeWordGate(model, "hey_jarvis", threshold=0.5, startup_skip_frames=0)
    gate.start()
    _fire_chunk(gate)

    gate.wait_for_wake(poll_interval=0.01)  # should return promptly
    assert not gate._wake_event.is_set()  # cleared for the next round


def test_cooldown_prevents_immediate_retrigger():
    model = FakeModel("hey_jarvis", score=0.9)
    gate = WakeWordGate(
        model, "hey_jarvis", threshold=0.5, cooldown_seconds=100.0, startup_skip_frames=0
    )
    gate.start()
    _fire_chunk(gate)
    gate.wait_for_wake(poll_interval=0.01)

    _fire_chunk(gate)  # within the cooldown window
    assert not gate._wake_event.is_set()


def test_restart_after_stop_opens_a_fresh_stream():
    gate = WakeWordGate(FakeModel("hey_jarvis"), "hey_jarvis", threshold=0.5)
    gate.start()
    gate.stop()
    gate.start()
    assert len(FakeStream.instances) == 2
    assert FakeStream.instances[1].started is True


# ---------------------------------------------------------------------------
# startup_skip_frames -- warm-up window after (re)starting the stream
# ---------------------------------------------------------------------------
# Found via manual testing: reopening the InputStream after a turn ends
# could register a spurious wake almost immediately. See start()'s
# docstring for why this is a fixed frame count rather than reusing
# cooldown_seconds.

def test_startup_skip_frames_discards_initial_callbacks():
    model = FakeModel("hey_jarvis", score=0.9)
    gate = WakeWordGate(model, "hey_jarvis", threshold=0.5, startup_skip_frames=3)
    gate.start()

    for _ in range(3):  # exactly the skip window -- none of these should count
        _fire_chunk(gate)
    assert not gate._wake_event.is_set()
    assert model.predict_calls == 0  # discarded before ever reaching the model

    _fire_chunk(gate)  # the first frame past the warm-up window
    assert gate._wake_event.is_set()
    assert model.predict_calls == 1


def test_startup_skip_counter_resets_on_restart():
    model = FakeModel("hey_jarvis", score=0.9)
    gate = WakeWordGate(model, "hey_jarvis", threshold=0.5, startup_skip_frames=2)
    gate.start()
    _fire_chunk(gate)
    _fire_chunk(gate)
    _fire_chunk(gate)  # past the warm-up window -- this one should register
    assert gate._wake_event.is_set()

    gate.stop()
    gate.start()  # fresh stream -- warm-up window applies again
    _fire_chunk(gate)
    assert not gate._wake_event.is_set()


def test_startup_skip_frames_defaults_to_nonzero():
    """The default (not explicitly disabled) should skip at least the
    first callback -- guards against the warm-up window silently being
    a no-op by default."""
    model = FakeModel("hey_jarvis", score=0.9)
    gate = WakeWordGate(model, "hey_jarvis", threshold=0.5)  # no startup_skip_frames override
    gate.start()
    _fire_chunk(gate)
    assert not gate._wake_event.is_set()


# ---------------------------------------------------------------------------
# restart_settle_seconds -- real pause before reopening the device
# ---------------------------------------------------------------------------
# Confirmed via manual testing with the diagnostic score/frame logging
# added alongside startup_skip_frames: a spurious wake fired with a
# genuinely high confidence score (0.99) at the very first scored
# frame, well under a second after the stream reopened -- too high and
# too immediate to be generic ambient noise, pointing at stale audio
# bleeding in from whichever stream had just closed on the same
# device. startup_skip_frames alone wasn't enough to prevent it.

def test_restart_settle_seconds_skipped_on_first_start(patch_sleep):
    gate = WakeWordGate(FakeModel("hey_jarvis"), "hey_jarvis", threshold=0.5)
    gate.start()
    patch_sleep.assert_not_called()  # nothing could have "just closed" the device yet


def test_restart_settle_seconds_applied_before_reopening_after_stop(patch_sleep):
    gate = WakeWordGate(
        FakeModel("hey_jarvis"), "hey_jarvis", threshold=0.5, restart_settle_seconds=0.3
    )
    gate.start()
    patch_sleep.reset_mock()
    gate.stop()
    gate.start()
    patch_sleep.assert_called_once_with(0.3)


def test_restart_settle_seconds_zero_disables_the_pause(patch_sleep):
    gate = WakeWordGate(
        FakeModel("hey_jarvis"), "hey_jarvis", threshold=0.5, restart_settle_seconds=0.0
    )
    gate.start()
    gate.stop()
    gate.start()
    patch_sleep.assert_not_called()


def test_restart_settle_seconds_defaults_to_nonzero(patch_sleep):
    """Guards against the settle delay silently being a no-op by
    default, same intent as test_startup_skip_frames_defaults_to_nonzero."""
    gate = WakeWordGate(FakeModel("hey_jarvis"), "hey_jarvis", threshold=0.5)  # no override
    gate.start()
    gate.stop()
    gate.start()
    patch_sleep.assert_called_once()
    assert patch_sleep.call_args.args[0] > 0
