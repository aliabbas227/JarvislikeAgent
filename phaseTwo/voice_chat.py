"""
voice_chat.py — Phase 2: the voice version of Phase 1's jarvis.py.
 
Loop: record until silence -> transcribe -> send to LLM (reusing Phase 1's
call_llm/get_system_prompt/trim_history) -> speak the reply -> repeat.
 
This does NOT duplicate the LLM backend logic — it imports Phase 1's
jarvis.py directly. That's a deliberate exception to "build each phase
standalone": voice_io.py (the actual new thing this phase teaches) is fully
independent and independently tested. The LLM call is proven, tested
infrastructure from Phase 1, not something this phase needs to re-learn.
 
Adjust PHASE_ONE_PATH below if your folder layout differs from
PROJECTS/JARVIS/phaseOne, phaseTwo/.
"""
 
import logging
import os
import sys
from pathlib import Path
 
from dotenv import load_dotenv
 
load_dotenv()
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("voice_chat")
 
from voice_io import (
    AudioRecordingError,
    SpeechError,
    TranscriptionError,
    record_until_silence,
    speak_text,
    transcribe_audio,
)
 
PHASE_ONE_PATH = Path(__file__).resolve().parent.parent / "phaseOne"
sys.path.insert(0, str(PHASE_ONE_PATH))
 
# Exact-match voice commands checked BEFORE anything is sent to the LLM.
# Deliberately exact/short phrases, not substring matches — "quit" as a
# substring would wrongly trigger on something like "did I quit too early?"
EXIT_PHRASES = {"quit", "exit", "stop", "goodbye", "shut down", "shutdown"}
 
try:
    from jarvis import call_llm, get_system_prompt, trim_history
except ImportError as e:
    logger.error(
        f"Could not import Phase 1's jarvis.py from {PHASE_ONE_PATH}. "
        "Update PHASE_ONE_PATH at the top of voice_chat.py to point at your "
        "phaseOne folder."
    )
    raise
 
 
def speak_safe(text: str) -> None:
    """Speaks text; falls back to printing if TTS itself is broken, so a
    dead audio driver never kills the whole loop."""
    try:
        speak_text(text)
    except SpeechError as e:
        logger.warning(f"TTS failed, printing instead: {e}")
        print(f"Jarvis (text fallback): {text}")
 
 
def run_voice_loop() -> None:
    history = [{"role": "system", "content": get_system_prompt()}]
    logger.info("Voice chat ready. Speak after 'Listening...' appears. Say 'quit' or press Ctrl+C to exit.")
 
    while True:
        try:
            wav_path = record_until_silence()
        except AudioRecordingError as e:
            logger.warning(f"Recording issue: {e}. Try again.")
            continue
 
        try:
            user_text = transcribe_audio(wav_path)
        except TranscriptionError as e:
            logger.warning(f"Couldn't transcribe that: {e}. Try again.")
            continue
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass
 
        logger.info(f"You said: {user_text}")
 
        normalized = user_text.strip().lower().rstrip(".!?")
        if normalized in EXIT_PHRASES:
            logger.info("Exit command recognized. Shutting down.")
            speak_safe("Goodbye.")
            break
 
        history.append({"role": "user", "content": user_text})
        history = trim_history(history)
 
        try:
            reply = call_llm(history)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            speak_safe("Sorry, I had trouble reaching my brain just now.")
            continue
 
        logger.info(f"Jarvis: {reply}")
        history.append({"role": "assistant", "content": reply})
        speak_safe(reply)
 
 
if __name__ == "__main__":
    try:
        run_voice_loop()
    except KeyboardInterrupt:
        logger.info("Voice chat stopped.")
 