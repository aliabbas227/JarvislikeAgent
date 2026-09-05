import time
import numpy as np
import sounddevice as sd
from openwakeword.model import Model
from dotenv import load_dotenv
import os

load_dotenv()

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1280  # 80ms at 16kHz, openWakeWord's expected frame size
THRESHOLD = float(os.getenv("WAKE_THRESHOLD", "0.5"))
MODEL_NAME = os.getenv("WAKE_MODEL", "hey_jarvis")

def build_model():
    return Model(
        wakeword_models=[MODEL_NAME],
        inference_framework="onnx",
    )

def listen_loop(model, on_detect, cooldown_seconds=2.0):
    last_trigger_time = 0

    def callback(indata, frames, time_info, status):
        nonlocal last_trigger_time
        if status:
            print(f"Audio status: {status}")
        audio = indata[:, 0].astype(np.int16)
        predictions = model.predict(audio)
        score = predictions.get(MODEL_NAME, 0)

        now = time.monotonic()
        if score > THRESHOLD and (now - last_trigger_time) > cooldown_seconds:
            last_trigger_time = now
            on_detect(score)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SAMPLES,
        channels=1,
        dtype="int16",
        callback=callback,
    ):
        print(f"Listening for '{MODEL_NAME}'... (Ctrl+C to stop)")
        while True:
            sd.sleep(1000)

def on_wake_detected(score):
    print(f"WAKE WORD DETECTED (score={score:.2f})")
    # Phase 3 stops here — no LLM call, no voice_chat.py import.

if __name__ == "__main__":
    model = build_model()
    try:
        listen_loop(model, on_wake_detected)
    except KeyboardInterrupt:
        print("\nStopped listening.")