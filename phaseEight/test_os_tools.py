"""
test_os_tools.py -- Phase 8 milestone 2 tests for the OS/window tools.

The window and power layers are faked at the ctypes/subprocess seam
rather than driven for real: a test suite that genuinely closed windows
or slept the machine would be unrunnable. The parts worth testing here
are the decisions -- what gets gated, what gets refused outright, how
ambiguity is handled, and whether the stale-handle and PID-reuse
guards actually fire -- not whether Windows' own PostMessage works.

Nothing in this file kills a process, closes a window, or touches
power state: every gated tool is exercised by proposing and then
either cancelling or confirming against a fake executor.
"""

import pytest

import gate
import os_tools


@pytest.fixture(autouse=True)
def clean_gate():
    gate._pending_actions.clear()
    yield
    gate._pending_actions.clear()


@pytest.fixture
def windows(monkeypatch):
    """Fakes the window layer with a controllable list of (hwnd, title)."""
    listing = [(101, "Notepad"), (202, "Firefox - Home"), (303, "Firefox - Docs")]
    monkeypatch.setattr(os_tools, "IS_WINDOWS", True)
    monkeypatch.setattr(os_tools, "_enum_visible_windows", lambda: list(listing))
    monkeypatch.setattr(os_tools, "_window_rect", lambda h: {"x": 0, "y": 0, "width": 800, "height": 600})
    monkeypatch.setattr(os_tools, "_window_title", lambda h: dict(listing).get(h, ""))
    return listing


# ---------------------------------------------------------------------------
# Window listing and resolution
# ---------------------------------------------------------------------------

def test_list_windows_returns_titles_and_geometry(windows):
    result = os_tools.list_windows()
    assert result["count"] == 3
    assert result["windows"][0]["title"] == "Notepad"
    assert result["windows"][0]["width"] == 800


def test_list_windows_filters_by_title(windows):
    assert os_tools.list_windows(title_contains="firefox")["count"] == 2


def test_window_tools_refuse_cleanly_off_windows(monkeypatch):
    monkeypatch.setattr(os_tools, "IS_WINDOWS", False)
    assert "error" in os_tools.list_windows()
    assert "error" in os_tools.lock_screen()
    assert "error" in os_tools.power_action("sleep")


def test_ambiguous_title_is_refused_rather_than_guessed(windows):
    """Picking the first of two matching Firefox windows would be right
    often enough to be trusted and wrong often enough to close the
    wrong one -- and close_window is what usually follows."""
    result = os_tools.close_window("firefox")
    assert "error" in result
    assert "2 windows match" in result["error"]


def test_ambiguity_error_lists_the_candidates(windows):
    """So the model can ask which one, instead of just failing."""
    error = os_tools.close_window("firefox")["error"]
    assert "Firefox - Home" in error and "Firefox - Docs" in error


def test_no_match_is_reported_clearly(windows):
    assert "No visible window" in os_tools.close_window("Photoshop")["error"]


def test_window_matching_is_case_insensitive(windows):
    assert os_tools.list_windows(title_contains="NOTEPAD")["count"] == 1


# ---------------------------------------------------------------------------
# close_window -- gated
# ---------------------------------------------------------------------------

def test_close_window_only_proposes(windows, monkeypatch):
    posted = []
    monkeypatch.setattr(os_tools.ctypes, "windll", _FakeWindll(posted))
    result = os_tools.close_window("Notepad")
    assert result["status"] == "confirmation_required"
    assert posted == []


def test_close_window_names_the_window_in_the_message(windows):
    """The message is the entire basis for the human's decision at the
    terminal, so it has to name the concrete target."""
    assert "Notepad" in os_tools.close_window("Notepad")["message"]


def test_confirming_close_window_sends_the_close(windows, monkeypatch):
    posted = []
    monkeypatch.setattr(os_tools.ctypes, "windll", _FakeWindll(posted))
    token = os_tools.close_window("Notepad")["token"]
    assert gate.confirm_pending_action(token)["status"] == "closed"
    assert posted == [(101, 0x0010)]


def test_cancelling_close_window_sends_nothing(windows, monkeypatch):
    posted = []
    monkeypatch.setattr(os_tools.ctypes, "windll", _FakeWindll(posted))
    token = os_tools.close_window("Notepad")["token"]
    gate.cancel_pending_action(token)
    assert posted == []


def test_close_window_aborts_if_the_handle_changed_meaning(windows, monkeypatch):
    """Between proposal and confirmation a window can close and Windows
    can hand its handle to a different one. Confirming a stale handle
    would close whatever inherited it."""
    posted = []
    monkeypatch.setattr(os_tools.ctypes, "windll", _FakeWindll(posted))
    token = os_tools.close_window("Notepad")["token"]

    monkeypatch.setattr(os_tools, "_window_title", lambda h: "Some Other App")
    outcome = gate.confirm_pending_action(token)
    assert "error" in outcome
    assert posted == []


class _FakeWindll:
    """Minimal stand-in for ctypes.windll, recording PostMessageW calls."""

    def __init__(self, posted):
        self.user32 = self._User32(posted)

    class _User32:
        def __init__(self, posted):
            self._posted = posted

        def PostMessageW(self, hwnd, msg, wparam, lparam):
            self._posted.append((hwnd, msg))
            return True

        def LockWorkStation(self):
            return True


# ---------------------------------------------------------------------------
# focus_window
# ---------------------------------------------------------------------------

def test_focus_window_reports_refusal_honestly(windows, monkeypatch):
    """Windows' foreground lock lets SetForegroundWindow report success
    while silently declining. A tool that claims success when focus did
    not move is worse than one that admits it, because the model's next
    step is usually to send that window a hotkey."""
    class _Refusing:
        class user32:
            @staticmethod
            def ShowWindow(hwnd, cmd):
                return True

            @staticmethod
            def SetForegroundWindow(hwnd):
                return True

            @staticmethod
            def GetForegroundWindow():
                return 999  # something else entirely

    monkeypatch.setattr(os_tools.ctypes, "windll", _Refusing)
    monkeypatch.setattr(os_tools, "_window_title", lambda h: "Notepad" if h == 101 else "Other")
    result = os_tools.focus_window("Notepad")
    assert result["status"] == "focus_refused"
    assert result["focused_instead"] == "Other"


# ---------------------------------------------------------------------------
# Processes
# ---------------------------------------------------------------------------

class _FakeProcess:
    def __init__(self, pid, name, created=1000.0, memory=100 * 1024 * 1024):
        self.pid = pid
        self._name = name
        self._created = created
        self.info = {"pid": pid, "name": name, "memory_info": type("M", (), {"rss": memory})()}
        self.terminated = False

    def name(self):
        return self._name

    def create_time(self):
        return self._created

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0


@pytest.fixture
def processes(monkeypatch):
    """Fakes psutil with a controllable process table."""
    table = {
        1: _FakeProcess(1, "chrome.exe", memory=500 * 1024 * 1024),
        2: _FakeProcess(2, "notepad.exe", memory=20 * 1024 * 1024),
        3: _FakeProcess(3, "lsass.exe"),
        4: _FakeProcess(4, "chrome.exe", memory=300 * 1024 * 1024),
    }

    class _FakePsutil:
        NoSuchProcess = type("NoSuchProcess", (Exception,), {})
        AccessDenied = type("AccessDenied", (Exception,), {})
        TimeoutExpired = type("TimeoutExpired", (Exception,), {})

        @staticmethod
        def process_iter(attrs=None):
            return list(table.values())

        @staticmethod
        def Process(pid):
            if pid not in table:
                raise _FakePsutil.NoSuchProcess()
            return table[pid]

    monkeypatch.setattr(os_tools, "_get_psutil", lambda: _FakePsutil)
    return table


def test_list_processes_sorts_by_memory(processes):
    """The question behind this is almost always 'what is eating my
    machine', not 'what is running in PID order'."""
    result = os_tools.list_processes()
    assert [p["name"] for p in result["processes"]][:2] == ["chrome.exe", "chrome.exe"]
    assert result["processes"][0]["memory_mb"] == 500.0


def test_list_processes_filters_by_name(processes):
    assert os_tools.list_processes(name_contains="notepad")["count"] == 1


def test_list_processes_respects_the_limit(processes):
    assert len(os_tools.list_processes(limit=2)["processes"]) == 2


def test_kill_process_only_proposes(processes):
    result = os_tools.kill_process(pid=2)
    assert result["status"] == "confirmation_required"
    assert processes[2].terminated is False


def test_kill_process_names_the_target_in_the_message(processes):
    message = os_tools.kill_process(pid=2)["message"]
    assert "notepad.exe" in message and "2" in message


def test_confirming_kill_actually_terminates(processes):
    token = os_tools.kill_process(pid=2)["token"]
    assert gate.confirm_pending_action(token)["status"] == "killed"
    assert processes[2].terminated is True


def test_cancelling_kill_terminates_nothing(processes):
    token = os_tools.kill_process(pid=2)["token"]
    gate.cancel_pending_action(token)
    assert processes[2].terminated is False


def test_protected_processes_are_refused_outright_not_gated(processes):
    """Killing lsass.exe logs out or bluescreens the machine. A
    confirmation prompt is not a good place to be discovering that, so
    it never becomes a proposal at all."""
    result = os_tools.kill_process(pid=3)
    assert "error" in result
    assert "protected" in result["error"]
    assert gate.pending_count() == 0


def test_killing_by_ambiguous_name_is_refused(processes):
    """'Close Chrome' meaning two processes at once is exactly what
    should not ride in on a single confirmation."""
    result = os_tools.kill_process(name="chrome")
    assert "2 processes match" in result["error"]


def test_killing_by_unique_name_works(processes):
    assert os_tools.kill_process(name="notepad")["status"] == "confirmation_required"


def test_kill_requires_a_pid_or_a_name(processes):
    assert "error" in os_tools.kill_process()


def test_kill_aborts_when_the_pid_was_reused(processes):
    """Between proposal and confirmation the target can exit and the OS
    can reassign its PID. Confirming a stale PID would kill an
    unrelated, newer process."""
    token = os_tools.kill_process(pid=2)["token"]
    processes[2]._created = 9999.0  # same pid, different process instance
    outcome = gate.confirm_pending_action(token)
    assert "error" in outcome
    assert "reused" in outcome["error"]
    assert processes[2].terminated is False


def test_killing_an_already_exited_process_is_not_an_error(processes):
    token = os_tools.kill_process(pid=2)["token"]
    del processes[2]
    assert gate.confirm_pending_action(token)["status"] == "already_exited"


# ---------------------------------------------------------------------------
# Media / clipboard / power
# ---------------------------------------------------------------------------

def test_media_control_rejects_anything_off_the_allowlist(monkeypatch):
    """The reason this tool is ungated where press_hotkey is gated is
    precisely that the key set is closed. If arbitrary keys got through,
    that justification would be void."""
    pressed = []
    monkeypatch.setitem(__import__("sys").modules, "pyautogui", _FakePyautogui(pressed))
    assert "error" in os_tools.media_control("alt+f4")
    assert "error" in os_tools.media_control("win+l")
    assert pressed == []


def test_media_control_sends_allowlisted_keys(monkeypatch):
    pressed = []
    monkeypatch.setitem(__import__("sys").modules, "pyautogui", _FakePyautogui(pressed))
    assert os_tools.media_control("volume_up")["status"] == "sent"
    assert pressed == ["volumeup"]


def test_media_control_forces_the_pyautogui_failsafe_on(monkeypatch):
    """Phase 6 forced FAILSAFE on unconditionally and gave nothing a way
    to turn it off. Same rule here."""
    fake = _FakePyautogui([])
    fake.FAILSAFE = False
    monkeypatch.setitem(__import__("sys").modules, "pyautogui", fake)
    os_tools.media_control("mute")
    assert fake.FAILSAFE is True


class _FakePyautogui:
    FAILSAFE = False

    def __init__(self, pressed):
        self._pressed = pressed

    def press(self, key):
        self._pressed.append(key)


def test_power_action_only_proposes(monkeypatch):
    launched = []
    monkeypatch.setattr(os_tools, "IS_WINDOWS", True)
    monkeypatch.setattr(os_tools.subprocess, "Popen", lambda *a, **k: launched.append(a))
    assert os_tools.power_action("shutdown")["status"] == "confirmation_required"
    assert launched == []


def test_power_action_rejects_unknown_actions(monkeypatch):
    monkeypatch.setattr(os_tools, "IS_WINDOWS", True)
    assert "error" in os_tools.power_action("explode")


def test_power_action_warns_that_jarvis_itself_stops(monkeypatch):
    monkeypatch.setattr(os_tools, "IS_WINDOWS", True)
    assert "Jarvis itself will stop" in os_tools.power_action("shutdown")["message"]
    assert "Jarvis itself will stop" not in os_tools.power_action("sleep")["message"]


def test_power_action_never_uses_a_shell(monkeypatch):
    """Argument lists, not shell strings -- nothing here should be
    turnable into command injection by a creative argument."""
    calls = []
    monkeypatch.setattr(os_tools, "IS_WINDOWS", True)
    monkeypatch.setattr(os_tools.subprocess, "Popen", lambda *a, **k: calls.append((a, k)))
    token = os_tools.power_action("sleep")["token"]
    gate.confirm_pending_action(token)
    args, kwargs = calls[0]
    assert isinstance(args[0], list)
    assert kwargs["shell"] is False


def test_clipboard_roundtrip(monkeypatch):
    store = {"value": ""}

    class _FakeClip:
        @staticmethod
        def copy(text):
            store["value"] = text

        @staticmethod
        def paste():
            return store["value"]

    monkeypatch.setattr(os_tools, "_get_pyperclip", lambda: _FakeClip)
    os_tools.set_clipboard("hello there")
    assert os_tools.get_clipboard()["content"] == "hello there"


def test_get_clipboard_truncates(monkeypatch):
    class _FakeClip:
        @staticmethod
        def paste():
            return "x" * 100

    monkeypatch.setattr(os_tools, "_get_pyperclip", lambda: _FakeClip)
    result = os_tools.get_clipboard(max_chars=10)
    assert len(result["content"]) == 10
    assert result["truncated"] is True
    assert result["char_count"] == 100


def test_take_screenshot_refuses_to_overwrite(tmp_path):
    """Same rule as Phase 6's write_file: creating is safe, replacing
    is not."""
    existing = tmp_path / "shot.png"
    existing.write_text("not really a png")
    assert "already exists" in os_tools.take_screenshot(str(existing))["error"]
    assert existing.read_text() == "not really a png"


# ---------------------------------------------------------------------------
# Registry contract
# ---------------------------------------------------------------------------

def test_schemas_and_functions_cover_the_same_names():
    schema_names = {s["function"]["name"] for s in os_tools.TOOL_SCHEMAS}
    assert schema_names == set(os_tools.TOOL_FUNCTIONS)


def test_the_gate_is_not_reachable_as_a_tool():
    """Mirrors Phase 6's own guardrail test. The model must never have a
    tool-calling path to confirm or cancel."""
    names = set(os_tools.TOOL_FUNCTIONS)
    assert not names & {"confirm_pending_action", "cancel_pending_action", "propose"}
