"""
audio_device.py -- pin the audio devices once, for every phase at once.

WHY THIS EXISTS
---------------
Every phase opens its microphone the same way::

    sd.InputStream(samplerate=..., channels=1, dtype="int16")

with no `device=` argument, in three separate live places:

    phaseTwo/voice_io.py     the turn recorder
    phaseSeven/wake_gate.py  the wake word
    phaseEight/interrupt.py  barge-in

plus `sd.play()` in phaseEight/voice.py for output. All four therefore
follow whatever Windows currently calls the default -- and that moves.
Measured across three runs minutes apart in one session: a controller
headset (mic peak 143), the same headset (peak 20), then a powered-off
wireless headset (**peak 1**), while output drifted onto a monitor over
HDMI. It presents as "Jarvis stopped hearing me" with no error at all,
which is the worst possible shape for a bug.

It also fails outright, not just quietly. Measured on this machine:

    sd.default.device         -> [-1, 1]
    MME/DirectSound/WASAPI    -> default_input_device = -1
    sd.InputStream()          -> PortAudioError: Error querying device -1

so with no default input device registered, Jarvis cannot hear anything
at all and dies at the wake gate before a single turn runs.

HOW IT FIXES ALL THREE PHASES WITHOUT EDITING THEM
--------------------------------------------------
`sd.default.device` is module-level state on the one `sounddevice`
module object shared by every import in the process. Setting it here,
once, before the orchestrator starts means every later
`sd.InputStream()` that passes no `device=` picks it up automatically --
the wake gate, the recorder and the barge-in listener included.

That matters for more than tidiness. Pinning only Phase 8 would leave
the wake word and the turn recorder on a *different* device from
barge-in, which is its own bug; and editing phases 2, 3 and 7 to thread
a device argument through would break the standalone-phase discipline
those finished phases are built on. One assignment, no edits, all four
call sites.

MATCHING BY NAME, NOT INDEX
---------------------------
Device indices shuffle whenever anything connects, disconnects or wakes
up -- index 9 today is a different device tomorrow. Configuration is
therefore a case-insensitive name fragment, resolved to an index at
startup. This is the same identifier-drift reasoning already applied to
close_window (handles re-verified by title, because Windows reuses
them) and kill_process (PIDs pinned by creation time), and ambiguity is
handled the same way those do: an unclear match is REFUSED with the
candidates listed, never guessed.
"""

import logging
import os

import sounddevice as sd

logger = logging.getLogger(__name__)

# The whole pipeline is 16 kHz mono int16 -- openWakeWord's models and
# Whisper both want 16 kHz, so there is no resampling anywhere. Probing
# at the rate actually used means a device that passes here is a device
# the real streams can open.
PROBE_SAMPLERATE = 16000
PROBE_CHANNELS = 1
PROBE_DTYPE = "int16"

# Peak amplitude at or below which a live probe is treated as "this
# device is not actually hearing anything". Chosen from the measured
# values behind this module: a working mic idled at 20 and a
# powered-off headset produced 1. Deliberately a WARNING and never a
# hard failure -- a genuinely silent room is not a broken microphone,
# and refusing to start over it would be worse than the bug.
SILENT_PEAK_THRESHOLD = 2
PROBE_SECONDS = 0.3

# Host APIs in the order we would rather use them.
#
# WDM-KS is last for a measured reason. phaseTwo/voice_io.py records
# with a BLOCKING stream.read(); phaseSeven/wake_gate.py uses a
# CALLBACK. WDM-KS supports the callback and not the blocking read --
# opening one and reading gives::
#
#     PaErrorCode -9999: 'Blocking API not supported yet'
#
# so a WDM-KS microphone produces the nastiest possible split: the wake
# word works, and then every actual turn fails to record. Worse,
# sd.check_input_settings() PASSES for such a device, which is why
# _usable() below does a real open and a real read instead of trusting
# it.
HOST_API_PREFERENCE = (
    "Windows WASAPI",
    "MME",
    "Windows DirectSound",
    "Windows WDM-KS",
)


def _devices_of_kind(kind: str) -> list:
    """
    [(index, name, hostapi_name)] for every device with channels of
    `kind` ("input" or "output"), ordered by HOST_API_PREFERENCE and
    then by index -- so callers that take the first workable candidate
    get the most compatible one, not merely the lowest-numbered.
    """
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    out = []
    for index, device in enumerate(sd.query_devices()):
        if device[key] > 0:
            hostapi = sd.query_hostapis(device["hostapi"])["name"]
            out.append((index, device["name"], hostapi))

    def rank(entry):
        _index, _name, hostapi = entry
        try:
            return HOST_API_PREFERENCE.index(hostapi)
        except ValueError:
            return len(HOST_API_PREFERENCE)

    return sorted(out, key=lambda e: (rank(e), e[0]))


def describe_devices() -> str:
    """Human-readable inventory, for logs and for the error message a
    person actually has to act on."""
    lines = []
    for kind in ("input", "output"):
        lines.append(f"  available {kind} devices:")
        found = _devices_of_kind(kind)
        if not found:
            lines.append("    (none)")
        for index, name, hostapi in found:
            lines.append(f"    [{index:2}] {name}  ({hostapi})")
    return "\n".join(lines)


def _usable(index, kind: str) -> bool:
    """
    True if a stream of `kind` can actually be opened AND USED on
    `index` the way this pipeline uses it.

    Deliberately not just sd.check_input_settings(). That helper passes
    for WDM-KS devices whose blocking read then fails at runtime with
    PaErrorCode -9999 (see HOST_API_PREFERENCE), which would hand back
    a microphone that wakes but cannot record. So the settings check is
    only a cheap pre-filter here, and the real test is opening the
    stream and performing the same blocking read voice_io.py performs.

    Costs a few tens of milliseconds per candidate, once, at startup,
    and only until the first device passes.
    """
    if index is None or index < 0:
        return False
    try:
        if kind == "input":
            sd.check_input_settings(
                device=index, samplerate=PROBE_SAMPLERATE,
                channels=PROBE_CHANNELS, dtype=PROBE_DTYPE,
            )
        else:
            sd.check_output_settings(device=index, channels=PROBE_CHANNELS)
    except Exception:
        return False

    stream = None
    try:
        if kind == "input":
            stream = sd.InputStream(
                device=index, samplerate=PROBE_SAMPLERATE,
                channels=PROBE_CHANNELS, dtype=PROBE_DTYPE,
            )
            stream.start()
            stream.read(int(PROBE_SAMPLERATE * 0.02))
        else:
            stream = sd.OutputStream(
                device=index, samplerate=PROBE_SAMPLERATE,
                channels=PROBE_CHANNELS, dtype=PROBE_DTYPE,
            )
            stream.start()
        return True
    except Exception as e:
        logger.debug(f"{kind} device [{index}] rejected: {e}")
        return False
    finally:
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass


def resolve_device(spec: str, kind: str) -> tuple:
    """
    Resolves a configured device spec to (index, name).

    `spec` is either a device index ("9") or a case-insensitive name
    fragment ("realtek"). Returns (None, reason) if it cannot be
    resolved -- including when it is AMBIGUOUS, which is refused rather
    than guessed: silently picking one of two microphones is exactly
    how you end up debugging "Jarvis stopped hearing me" again.
    """
    spec = (spec or "").strip()
    if not spec:
        return None, "no device configured"

    candidates = _devices_of_kind(kind)

    if spec.lstrip("-").isdigit():
        index = int(spec)
        match = [c for c in candidates if c[0] == index]
        if not match:
            return None, f"index {index} is not an available {kind} device"
        return index, match[0][1]

    hits = [c for c in candidates if spec.lower() in c[1].lower()]
    if not hits:
        return None, f"no {kind} device name contains {spec!r}"
    if len(hits) > 1:
        listed = "; ".join(f"[{i}] {n}" for i, n, _ in hits[:6])
        return None, f"{len(hits)} {kind} devices match {spec!r}: {listed}. Be more specific."
    return hits[0][0], hits[0][1]


def _choose(kind: str, env_var: str) -> tuple:
    """
    Picks the device to pin for `kind`, in descending order of how much
    the choice was actually intended:

      1. what the person configured, if it resolves
      2. the current system default, if it is usable
      3. the first usable device, loudly, as a last resort

    Returns (index, name, note). index is None only when the machine
    genuinely has no usable device of this kind.
    """
    spec = os.getenv(env_var, "")
    if spec.strip():
        index, detail = resolve_device(spec, kind)
        if index is not None and _usable(index, kind):
            return index, detail, f"pinned from {env_var}"
        if index is not None:
            logger.error(
                f"{env_var}={spec!r} resolved to [{index}] {detail}, but it cannot be "
                f"opened at {PROBE_SAMPLERATE} Hz. Falling back.\n{describe_devices()}"
            )
        else:
            logger.error(
                f"{env_var}={spec!r} could not be used: {detail}\n{describe_devices()}"
            )

    current = sd.default.device[0 if kind == "input" else 1]
    if _usable(current, kind):
        name = sd.query_devices(current)["name"]
        return current, name, "system default"

    for index, name, _hostapi in _devices_of_kind(kind):
        if _usable(index, kind):
            logger.warning(
                f"No usable default {kind} device (sd.default.device says {current}). "
                f"Falling back to [{index}] {name}. "
                f"Set {env_var} in phaseEight/.env to pin this deliberately.\n"
                f"{describe_devices()}"
            )
            return index, name, "fallback -- nothing was configured or default"

    hint = ""
    if kind == "input":
        # The shape this actually takes on Windows: the microphone is
        # still visible at the kernel-streaming level, so it looks
        # present in the list below, while MME/DirectSound/WASAPI all
        # report default_input_device = -1. That combination means the
        # recording device is disabled or not permitted, not missing.
        hint = (
            "\n  Every input device here is Windows WDM-KS, which cannot do the "
            "blocking read the recorder needs.\n"
            "  That usually means no recording device is ENABLED in Windows: check "
            "Settings > System > Sound >\n"
            "  More sound settings > Recording, enable and set a default microphone, "
            "and check\n"
            "  Settings > Privacy & security > Microphone."
        )
    logger.error(
        f"No usable {kind} device at all. Jarvis will not be able to "
        f"{'hear' if kind == 'input' else 'speak'}.{hint}\n{describe_devices()}"
    )
    return None, None, "unavailable"


def probe_input_level(seconds: float = PROBE_SECONDS) -> int:
    """
    Opens the pinned microphone briefly and returns the peak amplitude.

    This is what distinguishes the three failures that were previously
    indistinguishable from outside: a device that will not open at all
    (raises), a device that opens but hears nothing -- the powered-off
    headset case, which measured a peak of 1 -- and a working mic.
    Returns -1 if the probe could not run.
    """
    try:
        import numpy as np

        stream = sd.InputStream(
            samplerate=PROBE_SAMPLERATE, channels=PROBE_CHANNELS, dtype=PROBE_DTYPE
        )
        stream.start()
        frames, _overflowed = stream.read(int(PROBE_SAMPLERATE * seconds))
        stream.stop()
        stream.close()
        return int(np.abs(frames).max())
    except Exception as e:
        logger.warning(f"Could not probe the microphone level: {e}")
        return -1


def pin_devices(probe: bool = True) -> dict:
    """
    Resolves and pins sd.default.device for the whole process.

    Call once, before anything opens a stream -- which in practice
    means before run_orchestrator(), since the wake gate opens the
    microphone immediately.

    Returns a report dict (used by the tests and worth logging):
    {"input": (index, name, note), "output": (...), "peak": int|None}.
    Never raises: a machine with no microphone should still start and
    say so, rather than dying with a PortAudioError from three frames
    deep inside the wake gate.
    """
    in_index, in_name, in_note = _choose("input", "AUDIO_INPUT_DEVICE")
    out_index, out_name, out_note = _choose("output", "AUDIO_OUTPUT_DEVICE")

    sd.default.device = (
        in_index if in_index is not None else sd.default.device[0],
        out_index if out_index is not None else sd.default.device[1],
    )

    if in_index is not None:
        logger.info(f"Microphone: [{in_index}] {in_name} ({in_note})")
    if out_index is not None:
        logger.info(f"Speakers:   [{out_index}] {out_name} ({out_note})")

    peak = None
    if probe and in_index is not None:
        peak = probe_input_level()
        if peak < 0:
            pass  # already warned inside probe_input_level
        elif peak <= SILENT_PEAK_THRESHOLD:
            logger.warning(
                f"The microphone opened but heard almost nothing (peak {peak}). "
                "If that is not just a quiet room, the device may be muted or "
                "powered off -- a powered-off headset measured a peak of 1 here. "
                "Jarvis will not wake until this reads a real level."
            )
        else:
            logger.info(f"Microphone level check: peak {peak}. Sounds alive.")

    return {
        "input": (in_index, in_name, in_note),
        "output": (out_index, out_name, out_note),
        "peak": peak,
    }


if __name__ == "__main__":
    # Diagnostic entry point. Answers the question the original bug
    # made unanswerable from outside: which device is Jarvis actually
    # going to open, and can it hear anything through it?
    #
    # Companion to phaseSeven/calibrate_mic.py and
    # phaseEight/calibrate_barge_in.py -- those tune thresholds on a
    # microphone that works; this one tells you whether you have one.
    import sys
    from pathlib import Path

    from dotenv import load_dotenv

    # Absolute path, per this project's own standing rule -- a bare
    # load_dotenv() resolves relative to the caller and has silently
    # loaded the wrong file three times in this codebase already.
    load_dotenv(Path(__file__).resolve().parent / ".env")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print(describe_devices())
    print()
    report = pin_devices()
    print()
    if report["input"][0] is None:
        print("No usable microphone. Jarvis cannot hear in this state.")
        sys.exit(1)
    print(f"Jarvis will listen on:  [{report['input'][0]}] {report['input'][1]}")
    print(f"Jarvis will speak on:   [{report['output'][0]}] {report['output'][1]}")
    if report["peak"] is not None and report["peak"] > SILENT_PEAK_THRESHOLD:
        print(f"Microphone is live (peak {report['peak']}).")
    print()
    print("Pin these deliberately by setting AUDIO_INPUT_DEVICE / AUDIO_OUTPUT_DEVICE")
    print("in phaseEight/.env to a name fragment, e.g. AUDIO_INPUT_DEVICE=Onboard")
