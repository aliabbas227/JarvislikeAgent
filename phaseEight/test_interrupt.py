"""
test_interrupt.py -- Phase 8 milestone 3 tests for barge-in.

Fully mocked: no microphone, no TTS, no openWakeWord package. The audio
layer is faked at the sounddevice seam and the model at the
_get_model() seam, so this suite proves the chunking, the scoring
decision, and the start/stop lifecycle -- not that a real mic hears a
real wake word, which only a real voice test can show.

The lifecycle tests carry real weight here. Two of them pin lessons
this project paid for on real hardware: the listener must build its OWN
model (a shared openWakeWord Model carries prediction state between
calls and re-fires), and it must release the microphone before the
orchestrator opens its own stream for the follow-up window.
"""

import threading
import time

import numpy as np
import pytest

import interrupt


# ---------------------------------------------------------------------------
# split_into_speech_chunks
# ---------------------------------------------------------------------------

def test_splits_on_sentence_boundaries():
    chunks = interrupt.split_into_speech_chunks("First one. Second one! Third one?")
    assert chunks == ["First one.", "Second one!", "Third one?"]


def test_a_single_sentence_stays_whole():
    assert interrupt.split_into_speech_chunks("Just the one sentence.") == ["Just the one sentence."]


def test_empty_text_produces_no_chunks():
    assert interrupt.split_into_speech_chunks("   ") == []


def test_a_long_run_on_sentence_gets_split_further():
    """A 40-word sentence takes about fifteen seconds to say, during
    which the wake word would do nothing. Barge-in that ignores you for
    fifteen seconds is not barge-in."""
    long_sentence = ", ".join([f"clause number {i} of this very long sentence" for i in range(12)]) + "."
    chunks = interrupt.split_into_speech_chunks(long_sentence)
    assert len(chunks) > 1
    assert all(len(c) <= interrupt.MAX_CHUNK_CHARS + 60 for c in chunks)


def test_splitting_preserves_all_the_words():
    """Whatever the chunking does, it must not silently drop text --
    that would be a reply that quietly says less than the model wrote."""
    text = "The first sentence here. A second, with a comma, follows it. And a third."
    joined = " ".join(interrupt.split_into_speech_chunks(text))
    assert joined.split() == text.split()


# ---------------------------------------------------------------------------
# Fakes for the audio and model layers
# ---------------------------------------------------------------------------

class _FakeModel:
    """openWakeWord stand-in returning a scripted sequence of scores."""

    def __init__(self, scores, name="hey_jarvis"):
        self._scores = list(scores)
        self._name = name
        self.resets = 0
        self.predictions = 0

    def reset(self):
        self.resets += 1

    def predict(self, audio):
        self.predictions += 1
        score = self._scores.pop(0) if self._scores else 0.0
        return {self._name: score}


def _silent_stream(closed=None, opened=None):
    """A fake InputStream that always returns silence."""

    class _Fake:
        def __init__(self, **kwargs):
            if opened is not None:
                opened.append(kwargs)

        def start(self):
            pass

        def read(self, frames):
            time.sleep(0.005)
            return np.zeros((interrupt.CHUNK_SAMPLES, 1), dtype="int16"), False

        def stop(self):
            pass

        def close(self):
            if closed is not None:
                closed.append("closed")

    return _Fake


def _run_listener(monkeypatch, scores, currently_speaking="", timeout=2.0):
    """Runs a listener against a scripted score sequence."""
    model = _FakeModel(scores)
    monkeypatch.setattr(interrupt.sd, "InputStream", _silent_stream())
    monkeypatch.setattr(interrupt.InterruptListener, "_get_model", lambda self: model)
    monkeypatch.setattr(interrupt.InterruptListener, "_model_name", "hey_jarvis", raising=False)

    listener = interrupt.InterruptListener()
    listener._model_name = "hey_jarvis"
    listener.now_speaking(currently_speaking)
    listener.start()
    deadline = time.time() + timeout
    while time.time() < deadline and not listener.triggered:
        time.sleep(0.01)
    listener.stop()
    return listener, model


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def test_a_confident_score_triggers(monkeypatch):
    listener, _ = _run_listener(monkeypatch, [0.0, 0.0, 0.999])
    assert listener.triggered


def test_scores_below_the_threshold_do_not_trigger(monkeypatch):
    """openWakeWord scored Jarvis's own speech at 0.000 in real testing,
    so ordinary playback must never cross the bar."""
    listener, _ = _run_listener(monkeypatch, [0.0, 0.01, 0.2, 0.4] * 5, timeout=0.7)
    assert not listener.triggered


def test_the_threshold_is_honoured(monkeypatch):
    monkeypatch.setattr(interrupt, "BARGE_IN_THRESHOLD", 0.95)
    listener, _ = _run_listener(monkeypatch, [0.6] * 20, timeout=0.7)
    assert not listener.triggered


def test_best_score_is_recorded_for_diagnosis(monkeypatch):
    """Separates "heard you and wasn't sure" from "heard nothing at
    all" -- indistinguishable the first time barge-in failed."""
    listener, _ = _run_listener(monkeypatch, [0.1, 0.42, 0.2] * 5, timeout=0.7)
    assert listener.best_score == pytest.approx(0.42)


def test_jarvis_saying_its_own_name_does_not_interrupt_itself(monkeypatch):
    """The one case where echo could plausibly trigger a wake-word
    detector: a reply that literally contains the wake word."""
    listener, _ = _run_listener(
        monkeypatch, [0.999] * 10,
        currently_speaking="I'm Jarvis, your assistant.",
        timeout=0.7,
    )
    assert not listener.triggered


def test_the_guard_is_scoped_to_the_current_sentence(monkeypatch):
    """Held for a whole reply instead, one mention of the name would
    disable barge-in for the entire answer."""
    listener, _ = _run_listener(
        monkeypatch, [0.999] * 5,
        currently_speaking="Humidity is high at ninety seven percent.",
    )
    assert listener.triggered


# ---------------------------------------------------------------------------
# Model handling -- lessons from Phase 7's milestone 2
# ---------------------------------------------------------------------------

def test_the_model_is_reset_before_listening(monkeypatch):
    """Phase 7 spent three real-hardware passes finding that an
    openWakeWord Model keeps its prediction buffer between calls and
    re-fires on a stale detection. reset() is that fix."""
    _, model = _run_listener(monkeypatch, [0.999])
    assert model.resets >= 1


def test_the_model_is_reset_on_every_restart(monkeypatch):
    """Once at construction is not enough -- the stale state builds up
    after each detection, so it has to be cleared per session."""
    model = _FakeModel([0.0] * 50)
    monkeypatch.setattr(interrupt.sd, "InputStream", _silent_stream())
    monkeypatch.setattr(interrupt.InterruptListener, "_get_model", lambda self: model)

    listener = interrupt.InterruptListener()
    listener._model_name = "hey_jarvis"
    for _ in range(3):
        listener.start()
        time.sleep(0.05)
        listener.stop()
    assert model.resets == 3


def test_the_listener_builds_its_own_model_not_the_gates(monkeypatch):
    """Sharing the orchestrator's Model would put that same stale
    prediction state on both sides of a single turn."""
    built = []

    def fake_build():
        built.append("built")
        return _FakeModel([0.0])

    import types
    fake_wake_word = types.ModuleType("wake_word")
    fake_wake_word.MODEL_NAME = "hey_jarvis"
    fake_wake_word.build_model = fake_build
    monkeypatch.setitem(__import__("sys").modules, "wake_word", fake_wake_word)

    listener = interrupt.InterruptListener()
    assert listener._get_model() is not None
    assert built == ["built"]


def test_the_model_is_built_once_and_reused(monkeypatch):
    """Building it per reply would add a model load to every answer."""
    built = []

    def fake_build():
        built.append("built")
        return _FakeModel([0.0])

    import types
    fake_wake_word = types.ModuleType("wake_word")
    fake_wake_word.MODEL_NAME = "hey_jarvis"
    fake_wake_word.build_model = fake_build
    monkeypatch.setitem(__import__("sys").modules, "wake_word", fake_wake_word)

    listener = interrupt.InterruptListener()
    listener._get_model()
    listener._get_model()
    assert built == ["built"]


# ---------------------------------------------------------------------------
# Lifecycle and failure modes
# ---------------------------------------------------------------------------

def test_a_microphone_that_cannot_open_is_not_fatal(monkeypatch):
    """Barge-in is an enhancement. A dead mic means "no barge-in this
    reply", never a crashed voice loop -- same fail-soft reasoning
    Phase 7 used for the memory store."""
    def no_mic(*args, **kwargs):
        raise OSError("no such device")

    monkeypatch.setattr(interrupt.sd, "InputStream", no_mic)
    monkeypatch.setattr(interrupt.InterruptListener, "_get_model", lambda self: _FakeModel([0.999]))

    listener = interrupt.InterruptListener()
    listener.start()
    time.sleep(0.1)
    listener.stop()
    assert not listener.triggered


def test_a_model_that_cannot_build_is_not_fatal(monkeypatch):
    def no_model(self):
        raise RuntimeError("model weights missing")

    monkeypatch.setattr(interrupt.sd, "InputStream", _silent_stream())
    monkeypatch.setattr(interrupt.InterruptListener, "_get_model", no_model)

    listener = interrupt.InterruptListener()
    listener.start()
    time.sleep(0.1)
    listener.stop()
    assert not listener.triggered


def test_a_scoring_failure_is_not_fatal(monkeypatch):
    class _Exploding:
        def reset(self):
            pass

        def predict(self, audio):
            raise ValueError("bad frame")

    monkeypatch.setattr(interrupt.sd, "InputStream", _silent_stream())
    monkeypatch.setattr(interrupt.InterruptListener, "_get_model", lambda self: _Exploding())

    listener = interrupt.InterruptListener()
    listener._model_name = "hey_jarvis"
    listener.start()
    time.sleep(0.1)
    listener.stop()
    assert not listener.triggered


def test_stop_joins_the_thread(monkeypatch):
    """Two InputStreams on one device is exactly the class of problem
    Phase 7's milestone 2 spent three real-hardware passes on."""
    monkeypatch.setattr(interrupt.sd, "InputStream", _silent_stream())
    monkeypatch.setattr(interrupt.InterruptListener, "_get_model", lambda self: _FakeModel([0.0] * 100))

    listener = interrupt.InterruptListener()
    listener._model_name = "hey_jarvis"
    listener.start()
    assert listener._thread.is_alive()
    listener.stop()
    assert listener._thread is None


def test_the_stream_is_closed_on_stop(monkeypatch):
    closed = []
    monkeypatch.setattr(interrupt.sd, "InputStream", _silent_stream(closed=closed))
    monkeypatch.setattr(interrupt.InterruptListener, "_get_model", lambda self: _FakeModel([0.0] * 100))

    listener = interrupt.InterruptListener()
    listener._model_name = "hey_jarvis"
    listener.start()
    time.sleep(0.05)
    listener.stop()
    assert closed == ["closed"]


def test_the_stream_uses_openwakewords_frame_size(monkeypatch):
    """predict() expects exactly 1280 samples at 16kHz; a different
    blocksize silently produces garbage scores."""
    opened = []
    monkeypatch.setattr(interrupt.sd, "InputStream", _silent_stream(opened=opened))
    monkeypatch.setattr(interrupt.InterruptListener, "_get_model", lambda self: _FakeModel([0.0] * 20))

    listener = interrupt.InterruptListener()
    listener._model_name = "hey_jarvis"
    listener.start()
    time.sleep(0.05)
    listener.stop()
    assert opened[0]["blocksize"] == 1280
    assert opened[0]["samplerate"] == 16000


# ---------------------------------------------------------------------------
# speak_interruptibly
# ---------------------------------------------------------------------------

@pytest.fixture
def spoken(monkeypatch):
    """Captures what was actually spoken, and never makes a sound.

    Forces the OFFLINE path, so these tests exercise chunking and the
    interrupt flow without reaching edge-tts over the network. The
    neural path gets its own tests below.
    """
    said = []
    monkeypatch.setattr(interrupt.voice, "is_available", lambda: False)
    monkeypatch.setattr(interrupt.voice_io, "speak_text", lambda text: said.append(text))
    return said


@pytest.fixture
def no_listener(monkeypatch):
    """Replaces the listener with one that never triggers, so speaking
    can be tested without a microphone."""
    events = []

    class _Quiet:
        triggered = False
        best_score = 0.5  # "heard something", suppresses the diagnostic

        def start(self):
            events.append("start")

        def stop(self):
            events.append("stop")

        def now_speaking(self, text):
            events.append(("speaking", text))

    monkeypatch.setattr(interrupt, "InterruptListener", _Quiet)
    return events


LONG_REPLY = (
    "This is the first sentence of a reply. This is the second sentence. "
    "Here is a third one to make sure there is plenty to interrupt."
)


def test_a_long_reply_is_spoken_in_chunks(spoken, no_listener):
    interrupt.speak_interruptibly(LONG_REPLY)
    assert len(spoken) == 3


def test_a_short_reply_skips_the_whole_apparatus(spoken, no_listener):
    """A four-word confirmation is over before anyone could react. This
    is also what keeps "Goodbye." and the confirmation heads-up out of
    the interruptible path without this module knowing its call sites."""
    interrupt.speak_interruptibly("Got it.")
    assert spoken == ["Got it."]
    assert no_listener == []  # no mic stream, no model


def test_a_single_long_sentence_is_still_interruptible(spoken, no_listener):
    """Changed in milestone 4. It used to skip the listener, because
    with pyttsx3 there was nothing to interrupt BETWEEN when there was
    only one chunk. Playback can now be stopped mid-word, so a single
    long sentence is exactly as interruptible as any other reply."""
    interrupt.speak_interruptibly("x" * 100)
    assert "start" in no_listener


def test_empty_text_says_nothing(spoken, no_listener):
    interrupt.speak_interruptibly("   ")
    assert spoken == []


def test_each_chunk_is_announced_before_it_is_spoken(spoken, no_listener):
    """now_speaking has to lead the audio, or the wake-word guard is
    checking against the previous sentence."""
    interrupt.speak_interruptibly(LONG_REPLY)
    announced = [e[1] for e in no_listener if isinstance(e, tuple) and e[0] == "speaking"]
    assert announced == spoken


def test_the_listener_is_always_stopped_even_if_speaking_raises(monkeypatch, no_listener):
    """The mic must be released before the orchestrator opens its own
    stream for the follow-up window."""
    def boom(text):
        raise RuntimeError("audio device died")

    monkeypatch.setattr(interrupt.voice, "is_available", lambda: False)
    monkeypatch.setattr(interrupt.voice_io, "speak_text", boom)
    with pytest.raises(RuntimeError):
        interrupt.speak_interruptibly(LONG_REPLY)
    assert "stop" in no_listener


def test_the_offline_fallback_still_stops_at_a_chunk_boundary(monkeypatch):
    """When the neural voice is unavailable, playback is pyttsx3 again,
    which cannot be stopped mid-sentence -- so interruption degrades to
    milestone 3's original behavior rather than failing outright."""
    said = []
    listener_ref = []

    real_init = interrupt.InterruptListener.__init__

    def capture_init(self, *a, **kw):
        real_init(self, *a, **kw)
        listener_ref.append(self)

    def speak_then_interrupt(text):
        said.append(text)
        listener_ref[0].on_trigger()  # wake word heard during chunk 1

    monkeypatch.setattr(interrupt.InterruptListener, "__init__", capture_init)
    monkeypatch.setattr(interrupt.InterruptListener, "start", lambda self: None)
    monkeypatch.setattr(interrupt.InterruptListener, "stop", lambda self: None)
    monkeypatch.setattr(interrupt.InterruptListener, "triggered", property(lambda self: True))
    monkeypatch.setattr(interrupt.InterruptListener, "best_score", property(lambda self: 0.9))
    monkeypatch.setattr(interrupt.voice, "is_available", lambda: False)
    monkeypatch.setattr(interrupt.voice_io, "speak_text", speak_then_interrupt)

    interrupt.speak_interruptibly(LONG_REPLY)
    assert len(said) == 1, "should have stopped after the chunk it was interrupted during"


def test_an_interrupt_before_any_audio_speaks_nothing(monkeypatch):
    said = []
    listener_ref = []
    real_init = interrupt.InterruptListener.__init__

    def capture_init(self, *a, **kw):
        real_init(self, *a, **kw)
        listener_ref.append(self)

    monkeypatch.setattr(interrupt.InterruptListener, "__init__", capture_init)
    # Fire the trigger the moment the listener starts, before speaking.
    monkeypatch.setattr(interrupt.InterruptListener, "start",
                        lambda self: self.on_trigger())
    monkeypatch.setattr(interrupt.InterruptListener, "stop", lambda self: None)
    monkeypatch.setattr(interrupt.InterruptListener, "triggered", property(lambda self: True))
    monkeypatch.setattr(interrupt.InterruptListener, "best_score", property(lambda self: 0.9))
    monkeypatch.setattr(interrupt.voice, "is_available", lambda: False)
    monkeypatch.setattr(interrupt.voice_io, "speak_text", lambda text: said.append(text))

    interrupt.speak_interruptibly(LONG_REPLY)
    assert said == []


def test_disabling_barge_in_restores_plain_speech(spoken, no_listener, monkeypatch):
    """The first thing to turn off if speech ever misbehaves -- it must
    put back exactly Phase 7's behavior: one call, no listener."""
    monkeypatch.setattr(interrupt, "BARGE_IN_ENABLED", False)
    interrupt.speak_interruptibly(LONG_REPLY)
    assert spoken == [LONG_REPLY]
    assert no_listener == []


# ---------------------------------------------------------------------------
# The neural voice path (milestone 4)
# ---------------------------------------------------------------------------

def test_the_neural_voice_is_used_when_available(monkeypatch, no_listener):
    """The whole reply goes to voice.speak() in one call -- it handles
    its own sentence pipelining internally."""
    calls = []
    monkeypatch.setattr(interrupt.voice, "is_available", lambda: True)
    monkeypatch.setattr(
        interrupt.voice, "speak",
        lambda text, chunks=None, should_stop=None, on_chunk=None: calls.append((text, chunks)),
    )
    interrupt.speak_interruptibly(LONG_REPLY)
    assert len(calls) == 1
    assert calls[0][0] == LONG_REPLY
    assert len(calls[0][1]) == 3  # pre-split for it


def test_the_listener_can_stop_playback_instantly(monkeypatch):
    """The milestone 4 upgrade: on_trigger cuts audio mid-word, rather
    than waiting for the current sentence to end. sounddevice playback
    can be stopped cross-thread; pyttsx3 could not."""
    stopped = []

    def fake_speak(text, chunks=None, should_stop=None, on_chunk=None):
        # Simulate the listener firing while audio is playing.
        listener_ref[0].on_trigger()
        stopped.append(should_stop.is_set())

    listener_ref = [None]
    real_init = interrupt.InterruptListener.__init__

    def capture_init(self, *a, **kw):
        real_init(self, *a, **kw)
        listener_ref[0] = self

    monkeypatch.setattr(interrupt.InterruptListener, "__init__", capture_init)
    monkeypatch.setattr(interrupt.InterruptListener, "start", lambda self: None)
    monkeypatch.setattr(interrupt.InterruptListener, "stop", lambda self: None)
    monkeypatch.setattr(interrupt.InterruptListener, "triggered", property(lambda self: False))
    monkeypatch.setattr(interrupt.InterruptListener, "best_score", property(lambda self: 0.9))
    monkeypatch.setattr(interrupt.voice, "is_available", lambda: True)
    monkeypatch.setattr(interrupt.voice, "speak", fake_speak)

    interrupt.speak_interruptibly(LONG_REPLY)
    assert stopped == [True], "on_trigger did not set the playback stop flag"


# ---------------------------------------------------------------------------
# for_speech -- "sir" always lands at the end of its sentence
# ---------------------------------------------------------------------------
# Both rules exist because this is read aloud, not read on a page.
# edge-tts renders a comma as an audible pause, so ", sir" becomes
# "... [pause] sir" and a mid-sentence ", sir," gets two pauses around
# it. Compared out loud, the end-of-sentence form was clearly cleaner.
#
# Enforced in code because prompting lost: instructed explicitly, and
# tested live, DeepSeek still produced ", sir" in 4 of 5 replies.

@pytest.mark.parametrize("written,spoken_as", [
    ("It is just past eight, sir.", "It is just past eight sir."),
    ("Good evening, sir!", "Good evening sir!"),
    ("And you, sir?", "And you sir?"),
    ("You are most welcome, sir", "You are most welcome sir."),
])
def test_a_sentence_final_sir_loses_its_comma(written, spoken_as):
    assert interrupt.voice.for_speech(written) == spoken_as


@pytest.mark.parametrize("written,spoken_as", [
    ("Currently open, sir, eight windows.", "Currently open, eight windows sir."),
    ("I checked the forecast, sir, and it looks like rain.",
     "I checked the forecast, and it looks like rain sir."),
])
def test_a_mid_sentence_sir_moves_to_the_end(written, spoken_as):
    """Two pauses around a mid-sentence address is the worst-sounding
    case of all, and the reason the whole rule exists."""
    assert interrupt.voice.for_speech(written) == spoken_as


@pytest.mark.parametrize("written,spoken_as", [
    ("Barely doing anything, sir - that's the short version.",
     "Barely doing anything - that's the short version sir."),
    ("One moment, sir; the disk is spinning up.",
     "One moment; the disk is spinning up sir."),
])
def test_a_mid_sentence_sir_before_a_dash_or_semicolon_also_moves(written, spoken_as):
    """Regression test from live output. An earlier comma-only pattern
    missed ", sir -" entirely, and the model produced exactly that in 2
    of 5 real replies -- it reaches for a dash at least as readily as a
    second comma."""
    assert interrupt.voice.for_speech(written) == spoken_as


def test_a_leading_sir_moves_to_the_end_and_recapitalises():
    """Removing "Sir, " would otherwise leave the sentence starting
    lowercase."""
    assert interrupt.voice.for_speech("Sir, the build has finished.") ==         "The build has finished sir."


def test_sir_is_moved_not_deleted():
    """Deleting a mid-sentence address would silently cost the reply its
    only use of it and erode the character over a session."""
    out = interrupt.voice.for_speech("I checked the forecast, sir, and it looks fine.")
    assert out.lower().count("sir") == 1


def test_each_sentence_is_handled_independently():
    written = "Good evening, sir. The machine is in fine spirits, sir, and all is well."
    assert interrupt.voice.for_speech(written) ==         "Good evening sir. The machine is in fine spirits, and all is well sir."


def test_sentences_do_not_run_together():
    """A rewritten sentence must keep whatever separated it from the
    next, or speech comes out as "...sir.The machine..."."""
    out = interrupt.voice.for_speech("Good evening, sir. The build finished.")
    assert ". The" in out


def test_an_already_correct_sir_is_left_alone():
    assert interrupt.voice.for_speech("Good evening sir.") == "Good evening sir."


def test_a_sentence_that_is_only_an_address_is_left_alone():
    """Nothing to move it to the end of."""
    assert interrupt.voice.for_speech("Sir.") == "Sir."


def test_text_without_sir_is_untouched():
    line = "The machine is fine, the disk is fine, everything is fine."
    assert interrupt.voice.for_speech(line) == line


def test_sir_inside_another_word_is_not_touched():
    """Whole-word matching only -- otherwise "sire" or "sirloin" would
    be mangled."""
    line = "The sirloin is in the oven and the sire is asleep."
    assert interrupt.voice.for_speech(line) == line


def test_for_speech_handles_empty_input():
    assert interrupt.voice.for_speech("") == ""
    assert interrupt.voice.for_speech(None) == ""


# ---------------------------------------------------------------------------
# strip_markdown -- the speech engine reads markers, it doesn't interpret them
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("written,spoken_as", [
    ("That would be *hard*.", "That would be hard."),
    ("The **memory** is at 97 percent.", "The memory is at 97 percent."),
    ("It is ***very*** important.", "It is very important."),
    ("Your file is at `C:/Users/Example/notes.txt`.", "Your file is at C:/Users/Example/notes.txt."),
])
def test_emphasis_and_code_markers_are_removed(written, spoken_as):
    """edge-tts reads the characters rather than interpreting them, so
    "*hard*" is spoken with the asterisks pronounced."""
    assert interrupt.voice.strip_markdown(written) == spoken_as


def test_headings_and_bullets_are_removed():
    written = "# Status\n- Claude is open\n- Spotify is open"
    assert interrupt.voice.strip_markdown(written) == "Status\nClaude is open\nSpotify is open"


def test_the_words_between_markers_are_never_lost():
    """Losing emphasis is a small cost. Losing content would not be."""
    assert "important" in interrupt.voice.strip_markdown("It is **important**.")


def test_arithmetic_asterisks_are_left_alone():
    """A bare "*" with spaces around it is multiplication, not emphasis.
    The pattern requires a non-space character adjacent to the marker."""
    line = "Multiply 3 * 4 and 5 * 6."
    assert interrupt.voice.strip_markdown(line) == line


def test_plain_text_is_untouched():
    line = "No markdown here at all."
    assert interrupt.voice.strip_markdown(line) == line


def test_markdown_is_stripped_by_for_speech_too():
    """for_speech runs both passes -- markdown first, so the sir pass
    sees the text as it will actually be spoken."""
    assert interrupt.voice.for_speech("That would be *hard*, sir.") == "That would be hard sir."
