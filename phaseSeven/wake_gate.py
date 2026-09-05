"""
wake_gate.py -- Phase 7: pause/resume control around a wake-word model.

Phase 3's wake_word.listen_loop() blocks forever once started and has
no way to release the microphone -- fine for a standalone wake-word
demo, but the orchestrator needs to hand the mic off to the turn
recorder (voice_io.record_until_silence) after a wake, then take it
back afterward. Only one InputStream should safely own the mic at a
time: the wake-word listener OR the turn recorder, never both.

WakeWordGate wraps the same InputStream + callback shape Phase 3 used,
but as start()/stop()/wait_for_wake() so it can be paused instead of
running forever.

The wake-word model itself (openWakeWord's Model, or a test double) is
injected rather than constructed here -- same reasoning as Phase 3's
FakeModel pattern: this module should be fully testable without the
real openwakeword package or a real microphone installed. Model
construction (build_model()) still lives in Phase 3's wake_word.py and
is reused as-is by orchestrator.py.
"""

import logging
import threading
import time

import numpy as np
import sounddevice as sd

logger = logging.getLogger("wake_gate")

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_SAMPLES = 1280  # 80ms at 16kHz, openWakeWord's expected frame size


class WakeWordGate:
    """Start/stop-able wake-word listener. Sets an internal event when
    the wake word fires; the caller blocks on wait_for_wake() to
    consume it."""

    def __init__(
        self,
        model,
        model_name: str,
        threshold: float,
        cooldown_seconds: float = 2.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        chunk_samples: int = DEFAULT_CHUNK_SAMPLES,
        startup_skip_frames: int = 4,
        restart_settle_seconds: float = 0.3,
    ):
        self._model = model
        self._model_name = model_name
        self._threshold = threshold
        self._cooldown_seconds = cooldown_seconds
        self._sample_rate = sample_rate
        self._chunk_samples = chunk_samples
        # See start()'s docstring for why this exists. ~4 frames at the
        # default 80ms/frame chunk size is ~320ms -- enough to skip a
        # freshly opened stream's first-callback transients, short
        # enough that no one waiting to say the wake word would notice.
        self._startup_skip_frames = startup_skip_frames
        # See start()'s docstring. A real gap before reopening the
        # device, not just discarding frames after -- addresses stale
        # audio bleeding into the new stream's first callback(s) from
        # whichever InputStream (the turn recorder's, or this gate's
        # own from a prior turn) just closed on the same device.
        self._restart_settle_seconds = restart_settle_seconds

        self._stream = None
        self._wake_event = threading.Event()
        self._last_trigger_time = 0.0
        self._frames_since_start = 0
        self._stream_started_at = 0.0
        self._has_started_before = False

    def _callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"Audio status: {status}")

        self._frames_since_start += 1
        if self._frames_since_start <= self._startup_skip_frames:
            return  # still in the post-(re)start warm-up window -- see start()

        audio = indata[:, 0].astype(np.int16)
        predictions = self._model.predict(audio)
        score = predictions.get(self._model_name, 0)

        # Diagnostic logging: any score worth a second look, whether or
        # not it actually crosses the trigger bar. Deliberately added
        # instead of guessing again at a fix -- a startup-transient
        # theory (more startup_skip_frames needed) and a plain
        # ambient-noise/threshold theory (WAKE_THRESHOLD too low) lead
        # to different fixes, and both look identical from the outside
        # ("re-triggered right after going idle") since this gate is
        # only ever listening during IDLE in the first place. This
        # line is what tells them apart: a trigger at a small frame
        # count / low elapsed-since-start points at the former: one
        # that keeps happening well into an otherwise-quiet IDLE
        # period points at the latter.
        if score > self._threshold * 0.7:
            elapsed = time.monotonic() - self._stream_started_at
            logger.info(
                f"Wake model score={score:.3f} (threshold={self._threshold}) "
                f"at frame #{self._frames_since_start} since start, "
                f"{elapsed:.2f}s since stream (re)opened"
            )

        now = time.monotonic()
        if score > self._threshold and (now - self._last_trigger_time) > self._cooldown_seconds:
            self._last_trigger_time = now
            self._wake_event.set()

    def start(self) -> None:
        """Opens the mic and begins scoring audio against the wake
        word. Safe to call again after stop() -- a fresh InputStream is
        opened each time, same reasoning as voice_io.py's
        fresh-TTS-engine-per-call pattern: recreating cleanly is more
        reliable than trying to reset a stream object.

        Root cause, confirmed via manual testing with diagnostic score
        logging across three attempts: reopening the InputStream after
        a turn ends could register a spurious wake almost immediately,
        every time, at the exact same frame number and elapsed time,
        with a near-certain confidence score (0.98-0.99). That
        consistency was the tell -- real ambient noise wouldn't line up
        that precisely every time. Two earlier fixes targeted the audio
        STREAM/driver layer (startup_skip_frames, restart_settle_seconds
        below) and neither helped, because the actual stale state lived
        somewhere else entirely: openWakeWord's Model object (`self._model`)
        is constructed ONCE in orchestrator.py and reused for the whole
        process -- restarting the InputStream does nothing to it.
        Model.predict() maintains its own internal `prediction_buffer` (a
        rolling window of recent raw scores, smoothed for stability) and
        preprocessor buffer, and Model.reset() exists specifically to
        clear both. After a genuine "hey jarvis" detection, that buffer
        is full of high scores; nothing was ever clearing it before this
        fix, so the model kept "remembering" the old detection the next
        time scoring resumed -- explaining both the exact repeat pattern
        (always the same few frames in, since startup_skip_frames delays
        when predict() is even called again, during which the stale
        buffer sits untouched) and why it only ever happened after a
        turn that included a real detection.

        startup_skip_frames and restart_settle_seconds (below) are left
        in place as cheap, harmless secondary layers in case a real
        stream-level transient also exists, but model.reset() is the
        fix that actually matters here -- verify in logs that a
        spurious trigger doesn't recur before assuming either of the
        other two knobs still need tuning."""
        if self._has_started_before and self._restart_settle_seconds > 0:
            time.sleep(self._restart_settle_seconds)

        self._model.reset()
        self._wake_event.clear()
        self._frames_since_start = 0
        self._stream_started_at = time.monotonic()
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            blocksize=self._chunk_samples,
            channels=1,
            dtype="int16",
            callback=self._callback,
        )
        self._stream.start()
        self._has_started_before = True

    def stop(self) -> None:
        """Releases the mic. Safe to call even if already stopped or
        never started -- the orchestrator's shutdown path calls this
        unconditionally."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    def wait_for_wake(self, poll_interval: float = 0.1) -> None:
        """Blocks until the wake word fires, then clears the event so
        the next call waits for a fresh detection. Polls in a loop
        rather than a bare Event.wait() so a KeyboardInterrupt during
        IDLE breaks out promptly instead of potentially hanging on some
        platforms."""
        while not self._wake_event.is_set():
            time.sleep(poll_interval)
        self._wake_event.clear()
