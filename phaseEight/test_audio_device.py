"""
Tests for audio_device.py -- device pinning.

No real microphone: `sounddevice` is replaced wholesale with a fake
whose device table each test writes itself. That matters here more than
usual, because the bug this module exists for is *machine state* -- a
default that moves, a device that enumerates but cannot be opened -- and
those states cannot be reproduced on demand against real hardware.

Read the module docstring in audio_device.py first; the reasoning behind
the name-not-index matching and the WDM-KS deprioritisation lives there.
"""

import sys
import types

import pytest


# ---------------------------------------------------------------------------
# A fake sounddevice
# ---------------------------------------------------------------------------

class FakeDefault:
    def __init__(self, device):
        self.device = device


class FakeSounddevice(types.SimpleNamespace):
    """Reimplements only the handful of sounddevice calls audio_device
    makes. `unopenable` names the indices that pass check_*_settings and
    then fail on a real open -- the WDM-KS 'Blocking API not supported
    yet' case, which is the whole reason _usable() opens a stream at all
    rather than trusting the settings check."""

    def __init__(self, devices, hostapis, default, unopenable=(), peak=500):
        self._devices = devices
        self._hostapis = hostapis
        self.default = FakeDefault(list(default))
        self._unopenable = set(unopenable)
        self._peak = peak

    def query_devices(self, index=None):
        if index is None:
            return self._devices
        return self._devices[index]

    def query_hostapis(self, index=None):
        if index is None:
            return self._hostapis
        return self._hostapis[index]

    def check_input_settings(self, device=None, **kwargs):
        if self._devices[device]["max_input_channels"] < 1:
            raise RuntimeError("not an input device")

    def check_output_settings(self, device=None, **kwargs):
        if self._devices[device]["max_output_channels"] < 1:
            raise RuntimeError("not an output device")

    def _stream(self, device):
        peak = self._peak
        unopenable = device in self._unopenable

        class _S:
            def start(self_inner):
                pass

            def read(self_inner, frames):
                if unopenable:
                    raise RuntimeError(
                        "Error opening InputStream: Unanticipated host error "
                        "[PaErrorCode -9999]: 'Blocking API not supported yet'"
                    )
                import numpy as np

                return np.full((frames, 1), peak, dtype="int16"), False

            def stop(self_inner):
                pass

            def close(self_inner):
                pass

        return _S()

    def InputStream(self, device=None, **kwargs):
        target = self.default.device[0] if device is None else device
        if target is None or target < 0:
            raise RuntimeError(f"Error querying device {target}")
        return self._stream(target)

    def OutputStream(self, device=None, **kwargs):
        return self._stream(-999)  # never in _unopenable


def _device(name, hostapi, ins=0, outs=0):
    return {"name": name, "hostapi": hostapi,
            "max_input_channels": ins, "max_output_channels": outs}


HOSTAPIS = [
    {"name": "MME"},
    {"name": "Windows DirectSound"},
    {"name": "Windows WASAPI"},
    {"name": "Windows WDM-KS"},
]


def make_sd(**kwargs):
    """A healthy-looking machine: one MME mic, one WASAPI mic, one
    WDM-KS mic, plus outputs."""
    devices = [
        _device("Speakers (Onboard Audio)", 0, outs=2),                 # 0
        _device("Microphone (Onboard Audio)", 0, ins=2),  # 1  MME
        _device("Headset Microphone (Wireless Headset)", 2, ins=2),         # 2  WASAPI
        _device("Microphone (Virtual Audio)", 3, ins=8),        # 3  WDM-KS
        _device("Display Audio (NVIDIA High Definition)", 0, outs=2),   # 4
    ]
    kwargs.setdefault("default", [1, 0])
    return FakeSounddevice(devices, HOSTAPIS, **kwargs)


@pytest.fixture
def ad(monkeypatch):
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    import audio_device as module
    return module


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("AUDIO_INPUT_DEVICE", raising=False)
    monkeypatch.delenv("AUDIO_OUTPUT_DEVICE", raising=False)


# ---------------------------------------------------------------------------
# resolve_device -- name matching, and refusing to guess
# ---------------------------------------------------------------------------

def test_a_name_fragment_resolves_to_an_index(ad, monkeypatch):
    monkeypatch.setattr(ad, "sd", make_sd())
    index, name = ad.resolve_device("onboard", "input")
    assert index == 1
    assert "Onboard Audio" in name


def test_matching_is_case_insensitive(ad, monkeypatch):
    monkeypatch.setattr(ad, "sd", make_sd())
    assert ad.resolve_device("WIRELESS", "input")[0] == 2


def test_a_bare_index_still_works(ad, monkeypatch):
    monkeypatch.setattr(ad, "sd", make_sd())
    assert ad.resolve_device("2", "input")[0] == 2


def test_an_index_that_is_not_an_input_device_is_rejected(ad, monkeypatch):
    monkeypatch.setattr(ad, "sd", make_sd())
    index, reason = ad.resolve_device("0", "input")  # a speaker
    assert index is None
    assert "not an available input device" in reason


def test_an_ambiguous_fragment_is_refused_not_guessed(ad, monkeypatch):
    # Same discipline as close_window's ambiguous window match: silently
    # picking one of two microphones is how you end up debugging
    # "Jarvis stopped hearing me" all over again.
    monkeypatch.setattr(ad, "sd", make_sd())
    index, reason = ad.resolve_device("microphone", "input")
    assert index is None
    assert "Be more specific" in reason


def test_an_ambiguous_fragment_lists_the_candidates(ad, monkeypatch):
    monkeypatch.setattr(ad, "sd", make_sd())
    _index, reason = ad.resolve_device("microphone", "input")
    assert "Onboard" in reason and "Virtual" in reason


def test_an_unknown_fragment_says_so(ad, monkeypatch):
    monkeypatch.setattr(ad, "sd", make_sd())
    index, reason = ad.resolve_device("bluetooth banana", "input")
    assert index is None
    assert "no input device name contains" in reason


def test_an_empty_spec_is_not_an_error(ad, monkeypatch):
    monkeypatch.setattr(ad, "sd", make_sd())
    assert ad.resolve_device("", "input") == (None, "no device configured")


# ---------------------------------------------------------------------------
# Host API preference -- the WDM-KS trap
# ---------------------------------------------------------------------------

def test_wdm_ks_is_ranked_last(ad):
    # It supports the wake gate's callback but NOT voice_io.py's
    # blocking read, so a WDM-KS mic wakes and then fails every turn.
    assert ad.HOST_API_PREFERENCE[-1] == "Windows WDM-KS"


def test_candidates_come_back_in_preference_order(ad, monkeypatch):
    monkeypatch.setattr(ad, "sd", make_sd())
    order = [hostapi for _i, _n, hostapi in ad._devices_of_kind("input")]
    assert order.index("Windows WDM-KS") == len(order) - 1


def test_a_device_that_passes_settings_but_fails_to_read_is_unusable(ad, monkeypatch):
    # The exact measured failure: check_input_settings() passes, the
    # blocking read raises PaErrorCode -9999.
    monkeypatch.setattr(ad, "sd", make_sd(unopenable={3}))
    assert ad._usable(3, "input") is False
    assert ad._usable(1, "input") is True


# ---------------------------------------------------------------------------
# _choose -- configured, then default, then fallback
# ---------------------------------------------------------------------------

def test_a_configured_device_wins_over_the_system_default(ad, monkeypatch, clean_env):
    monkeypatch.setattr(ad, "sd", make_sd(default=[1, 0]))
    monkeypatch.setenv("AUDIO_INPUT_DEVICE", "wireless")
    index, _name, note = ad._choose("input", "AUDIO_INPUT_DEVICE")
    assert index == 2
    assert "pinned" in note


def test_the_system_default_is_used_when_nothing_is_configured(ad, monkeypatch, clean_env):
    monkeypatch.setattr(ad, "sd", make_sd(default=[2, 0]))
    index, _name, note = ad._choose("input", "AUDIO_INPUT_DEVICE")
    assert index == 2
    assert note == "system default"


def test_a_broken_default_falls_back_to_a_usable_device(ad, monkeypatch, clean_env):
    # The live failure: sd.default.device == [-1, 1], every host API
    # reporting default_input_device = -1.
    monkeypatch.setattr(ad, "sd", make_sd(default=[-1, 0]))
    index, _name, note = ad._choose("input", "AUDIO_INPUT_DEVICE")
    assert index == 2, "should pick the WASAPI mic, not the WDM-KS one"
    assert "fallback" in note


def test_the_fallback_skips_devices_that_cannot_actually_be_read(ad, monkeypatch, clean_env):
    # Device 2 (WASAPI) is preferred but unreadable, so this must fall
    # past it to the MME one rather than returning the best-ranked
    # device that happens not to work.
    monkeypatch.setattr(ad, "sd", make_sd(default=[-1, 0], unopenable={2}))
    index, _name, _note = ad._choose("input", "AUDIO_INPUT_DEVICE")
    assert index == 1


def test_an_unresolvable_configuration_falls_back_rather_than_dying(ad, monkeypatch, clean_env):
    monkeypatch.setattr(ad, "sd", make_sd(default=[2, 0]))
    monkeypatch.setenv("AUDIO_INPUT_DEVICE", "nonexistent device")
    index, _name, note = ad._choose("input", "AUDIO_INPUT_DEVICE")
    assert index == 2
    assert note == "system default"


def test_no_usable_input_at_all_reports_unavailable(ad, monkeypatch, clean_env):
    sd = make_sd(default=[-1, 0], unopenable={1, 2, 3})
    monkeypatch.setattr(ad, "sd", sd)
    index, name, note = ad._choose("input", "AUDIO_INPUT_DEVICE")
    assert (index, name, note) == (None, None, "unavailable")


def test_the_no_microphone_error_is_actionable(ad, monkeypatch, clean_env, caplog):
    # The original bug was invisible; the replacement must not be.
    monkeypatch.setattr(ad, "sd", make_sd(default=[-1, 0], unopenable={1, 2, 3}))
    with caplog.at_level("ERROR"):
        ad._choose("input", "AUDIO_INPUT_DEVICE")
    assert "Sound" in caplog.text, "should point at Windows sound settings"


# ---------------------------------------------------------------------------
# pin_devices -- the one assignment that fixes all four call sites
# ---------------------------------------------------------------------------

def test_pinning_sets_the_process_wide_default(ad, monkeypatch, clean_env):
    # This single assignment is what reaches the wake gate, the turn
    # recorder, barge-in and playback -- none of which pass device=.
    sd = make_sd(default=[-1, 0])
    monkeypatch.setattr(ad, "sd", sd)
    ad.pin_devices(probe=False)
    assert sd.default.device[0] == 2


def test_pinning_reports_what_it_chose(ad, monkeypatch, clean_env):
    monkeypatch.setattr(ad, "sd", make_sd(default=[2, 0]))
    report = ad.pin_devices(probe=False)
    assert report["input"][0] == 2
    assert "Wireless Headset" in report["input"][1]


def test_pinning_never_raises_without_a_microphone(ad, monkeypatch, clean_env):
    # A machine with no mic must still start and say so, rather than
    # dying with a PortAudioError three frames inside the wake gate.
    monkeypatch.setattr(ad, "sd", make_sd(default=[-1, 0], unopenable={1, 2, 3}))
    report = ad.pin_devices(probe=False)
    assert report["input"][0] is None


def test_a_missing_microphone_leaves_the_output_default_alone(ad, monkeypatch, clean_env):
    sd = make_sd(default=[-1, 4], unopenable={1, 2, 3})
    monkeypatch.setattr(ad, "sd", sd)
    ad.pin_devices(probe=False)
    assert sd.default.device[1] == 4


# ---------------------------------------------------------------------------
# The live level probe -- telling "silent" from "broken"
# ---------------------------------------------------------------------------

def test_a_live_microphone_probes_above_the_silence_floor(ad, monkeypatch, clean_env):
    monkeypatch.setattr(ad, "sd", make_sd(default=[1, 0], peak=500))
    assert ad.probe_input_level(seconds=0.01) == 500


def test_a_powered_off_headset_is_warned_about(ad, monkeypatch, clean_env, caplog):
    # Measured: a working mic idled at 20, a powered-off headset gave 1.
    monkeypatch.setattr(ad, "sd", make_sd(default=[1, 0], peak=1))
    with caplog.at_level("WARNING"):
        report = ad.pin_devices(probe=True)
    assert report["peak"] == 1
    assert "heard almost nothing" in caplog.text


def test_a_silent_probe_is_a_warning_and_never_a_failure(ad, monkeypatch, clean_env):
    # A quiet room is not a broken microphone.
    monkeypatch.setattr(ad, "sd", make_sd(default=[1, 0], peak=0))
    report = ad.pin_devices(probe=True)
    assert report["input"][0] == 1, "must still pin the device"


def test_a_probe_that_cannot_run_does_not_crash_startup(ad, monkeypatch, clean_env):
    monkeypatch.setattr(ad, "sd", make_sd(default=[1, 0], unopenable={1}))
    assert ad.probe_input_level(seconds=0.01) == -1


# ---------------------------------------------------------------------------
# describe_devices -- the diagnostic that made this findable
# ---------------------------------------------------------------------------

def test_the_inventory_lists_both_kinds(ad, monkeypatch):
    monkeypatch.setattr(ad, "sd", make_sd())
    text = ad.describe_devices()
    assert "available input devices" in text
    assert "available output devices" in text


def test_the_inventory_names_the_host_api(ad, monkeypatch):
    # Which host API a device is on is the whole diagnosis when
    # everything is WDM-KS.
    monkeypatch.setattr(ad, "sd", make_sd())
    assert "Windows WDM-KS" in ad.describe_devices()
