"""
calibrate_mic.py -- Phase 7 diagnostic: find a good SILENCE_THRESHOLD.

voice_io.py's silence detector (Phase 2) is a simple RMS-amplitude
threshold on each 100ms audio chunk. If SILENCE_THRESHOLD sits too
close to (or below) the room's actual background noise floor, a
recording either never registers real speech as distinct from
background, or never sees a clean trailing silence -- and just runs to
MAX_RECORD_SECONDS every time. That's what's happening if you're
seeing "Max recording duration hit" / "No speech detected before
timeout" a lot.

This prints live RMS levels for ~15 seconds so you can read off:
  - the resting level with nobody talking (background noise floor)
  - the level while you're actually speaking normally

Pick SILENCE_THRESHOLD roughly halfway between those two numbers --
comfortably above the background floor, comfortably below your
speaking level.

Usage:
    python calibrate_mic.py
    (stay quiet for the first few seconds, then talk normally for the rest)
"""

import time

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.1  # matches voice_io.py's CHUNK_DURATION
DURATION_SECONDS = 15


def _rms(chunk: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(chunk.astype(np.float64)))))


def main() -> None:
    chunk_frames = int(CHUNK_DURATION * SAMPLE_RATE)
    levels = []

    print(f"Calibrating for {DURATION_SECONDS}s. Stay quiet for the first ~5s,")
    print("then talk normally for the rest. Watch the numbers as you go.\n")

    try:
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16")
        stream.start()
    except Exception as e:
        print(f"Could not open microphone: {e}")
        return

    start = time.time()
    try:
        while time.time() - start < DURATION_SECONDS:
            chunk, _ = stream.read(chunk_frames)
            level = _rms(chunk)
            levels.append(level)
            bar = "#" * min(int(level / 50), 80)
            print(f"RMS: {level:7.1f}  {bar}")
    finally:
        stream.stop()
        stream.close()

    if not levels:
        print("No audio captured -- check your microphone.")
        return

    levels = np.array(levels)
    quiet_window = levels[: max(1, len(levels) // 3)]  # rough proxy for the "stay quiet" portion

    print("\n--- Summary ---")
    print(f"Approx. background floor (first ~1/3 of the run): {quiet_window.mean():.0f} avg, {quiet_window.max():.0f} peak")
    print(f"Overall peak level seen (likely your speech):     {levels.max():.0f}")
    suggested = int(quiet_window.max() + (levels.max() - quiet_window.max()) * 0.4)
    print(f"\nSuggested SILENCE_THRESHOLD to try: ~{suggested}")
    print("(comfortably above the background floor, comfortably below your speaking peak)")
    print("Set this in phaseSeven/.env as SILENCE_THRESHOLD=<value> and re-run.")


if __name__ == "__main__":
    main()