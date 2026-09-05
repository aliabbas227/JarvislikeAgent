"""
calibrate_barge_in.py -- Phase 8 diagnostic: can barge-in actually hear
you say the wake word while Jarvis is talking?

Companion to phaseSeven/calibrate_mic.py. This exists because
barge-in's first real-hardware failure was invisible from the outside:
saying the interrupt word did nothing, and there was no way to tell
whether it had been misheard, rejected, or -- as it turned out --
whether the microphone had picked up anything at all. The default input
device had quietly moved to a different, powered-off headset.

Originally this measured loudness thresholds, because detection used
Whisper and needed one. Detection is now openWakeWord, which has no
loudness threshold to tune, so the useful questions changed:

  1. which microphone is actually being used, and is anything reaching it
  2. does Jarvis's own voice score high enough to interrupt itself
  3. does YOUR voice saying the wake word score high enough to interrupt

Question 3 is the one that matters, and it is answered with the real
model at the real threshold, not inferred from volume.

Usage:
    python calibrate_barge_in.py
"""

import re
import statistics
import sys
import threading
import time

import numpy as np
import sounddevice as sd

import interrupt
import voice_io

LISTEN_SECONDS = 5.0

JARVIS_LINE = (
    "This is Jarvis speaking at a normal volume so the microphone can "
    "measure how much of my own voice it picks up while I am talking."
)


def _hardware_name(device: str) -> str:
    """The part of a Windows device name identifying the physical
    hardware, role prefix stripped.

    "Microphone (Wireless Headset)" and "Headphones (Wireless Headset)" are one
    headset, not two devices. Comparing full strings -- or first words,
    which are always "Microphone" and "Headphones" -- says they differ
    every time, which is how this tool once told someone their matched
    headset was a split-device setup.
    """
    inside = re.search(r"\(([^)]*)\)", device)
    name = inside.group(1) if inside else device
    return re.sub(r"^\d+-\s*", "", name).strip().lower()


def _same_hardware(input_device: str, output_device: str) -> bool:
    a, b = _hardware_name(input_device), _hardware_name(output_device)
    return bool(a) and bool(b) and (a in b or b in a)


def score_and_level(model, model_name, seconds: float) -> tuple:
    """Runs the real wake-word model over the real mic for `seconds`,
    returning (best score, peak RMS). Peak RMS is reported alongside
    because a score of zero means something completely different when
    the mic is dead than when it is merely not hearing the phrase."""
    model.reset()
    best_score, peak = 0.0, 0.0
    stream = sd.InputStream(
        samplerate=interrupt.SAMPLE_RATE,
        blocksize=interrupt.CHUNK_SAMPLES,
        channels=1,
        dtype="int16",
    )
    stream.start()
    try:
        deadline = time.time() + seconds
        while time.time() < deadline:
            frame, _ = stream.read(interrupt.CHUNK_SAMPLES)
            peak = max(peak, voice_io._rms(frame))
            score = model.predict(frame[:, 0].astype(np.int16)).get(model_name, 0.0)
            best_score = max(best_score, score)
    finally:
        stream.stop()
        stream.close()
    return best_score, peak


def main() -> None:
    print("Barge-in calibration (openWakeWord detector)\n")

    try:
        input_device = sd.query_devices(kind="input")["name"]
        output_device = sd.query_devices(kind="output")["name"]
    except Exception as e:
        print(f"Could not query audio devices: {e}")
        return

    print(f"  input  device: {input_device}")
    print(f"  output device: {output_device}")
    if not _same_hardware(input_device, output_device):
        print(
            "\n  NOTE: input and output look like different devices. Fine in\n"
            "  itself, and it actually makes self-interruption less likely --\n"
            "  but check the input above really is the mic you speak into.\n"
            "  On this machine the default input silently moved to a\n"
            "  powered-off headset once, and barge-in simply stopped working."
        )
    print()

    from wake_word import MODEL_NAME

    print(f"  wake phrase: {MODEL_NAME.replace('_', ' ')!r}")
    print(f"  threshold:   {interrupt.BARGE_IN_THRESHOLD}\n")

    try:
        model = interrupt.InterruptListener()._get_model()
    except Exception as e:
        print(f"Could not build the wake-word model: {e}")
        print("Try: python -c \"import openwakeword.utils as u; u.download_models()\"")
        return

    print("1. Measuring Jarvis's own voice (say nothing)...")
    result = {}
    done = threading.Event()

    def collect():
        result["echo"] = score_and_level(model, MODEL_NAME, 6.0)
        done.set()

    threading.Thread(target=collect, daemon=True).start()
    voice_io.speak_text(JARVIS_LINE)
    done.wait(timeout=9)
    echo_score, echo_peak = result.get("echo", (0.0, 0.0))
    print(f"   Jarvis's own voice: best score {echo_score:.3f}   peak RMS {echo_peak:.0f}")

    print(f"\n2. Now SAY '{MODEL_NAME.replace('_', ' ')}' a couple of times "
          f"({LISTEN_SECONDS:.0f}s)...")
    time.sleep(0.5)
    you_score, you_peak = score_and_level(model, MODEL_NAME, LISTEN_SECONDS)
    print(f"   your voice:         best score {you_score:.3f}   peak RMS {you_peak:.0f}")

    print("\n--- Verdict ---")
    if you_peak < 50:
        print("  PROBLEM: the microphone is picking up almost nothing at all.")
        print("  Barge-in cannot work, and neither can the wake word or ordinary")
        print("  recording. Check that the input device above is the mic you")
        print("  actually speak into, that it is switched on, and that it is not")
        print("  muted in Windows sound settings.")
    elif you_score < interrupt.BARGE_IN_THRESHOLD:
        print(f"  PROBLEM: the mic hears you (peak {you_peak:.0f}) but the wake word")
        print(f"  only scored {you_score:.3f}, under the {interrupt.BARGE_IN_THRESHOLD} threshold.")
        print("  Try saying it more clearly and a little slower, closer to the mic.")
        print(f"  If it is consistently just under, lower BARGE_IN_THRESHOLD in")
        print("  phaseEight/.env -- but check the echo score below first.")
    elif echo_score >= interrupt.BARGE_IN_THRESHOLD:
        print(f"  PROBLEM: Jarvis's own voice scored {echo_score:.3f}, over the threshold.")
        print("  It would interrupt itself. Raise BARGE_IN_THRESHOLD in")
        print("  phaseEight/.env above that score.")
    else:
        print("  Good: you clear the threshold and Jarvis does not.")
        print(f"    you    {you_score:.3f}  >=  {interrupt.BARGE_IN_THRESHOLD}")
        print(f"    Jarvis {echo_score:.3f}  <   {interrupt.BARGE_IN_THRESHOLD}")
        print(f"\n  Margin between you and Jarvis: {you_score - echo_score:.3f} "
              f"(bigger is better; 0.9+ is typical)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
