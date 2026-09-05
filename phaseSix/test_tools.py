"""
test_tools.py — Phase 6 tests.

Filesystem operations (list_directory, read_file, delete_file) use
pytest's tmp_path fixture — a real temp directory, not a mock, since
these are plain stdlib os calls with no hardware/model dependency to
fake out (only fake what's actually expensive or unavailable, same
reasoning as every prior phase's mocking pattern). Application
launching IS mocked — os.startfile/subprocess.Popen would actually
open windows on the machine running the tests.
"""

import sys
import time
import types

import pytest

import tools
@pytest.fixture(autouse=True)
def _clear_pending():
    """Each test gets a clean pending-actions registry."""
    tools._pending_actions.clear()
    yield
    tools._pending_actions.clear()


# ---------------------------------------------------------------------------
# list_directory
# ---------------------------------------------------------------------------

def test_list_directory_lists_files_and_dirs(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "subdir").mkdir()

    result = tools.list_directory(str(tmp_path))

    names = {e["name"] for e in result["entries"]}
    assert names == {"a.txt", "subdir"}
    types = {e["name"]: e["type"] for e in result["entries"]}
    assert types["a.txt"] == "file"
    assert types["subdir"] == "directory"


def test_list_directory_dirs_sort_before_files(tmp_path):
    (tmp_path / "z_file.txt").write_text("x")
    (tmp_path / "a_dir").mkdir()

    result = tools.list_directory(str(tmp_path))
    assert result["entries"][0]["name"] == "a_dir"


def test_list_directory_nonexistent_path_returns_error():
    result = tools.list_directory("/definitely/not/a/real/path/xyz")
    assert "error" in result


def test_list_directory_file_path_returns_error(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    result = tools.list_directory(str(f))
    assert "error" in result


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

def test_read_file_returns_content(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("hello world")

    result = tools.read_file(str(f))
    assert result["content"] == "hello world"
    assert result["truncated"] is False


def test_read_file_truncates_long_content(tmp_path):
    f = tmp_path / "big.txt"
    f.write_text("x" * 100)

    result = tools.read_file(str(f), max_chars=10)
    assert len(result["content"]) == 10
    assert result["truncated"] is True


def test_read_file_refuses_binary(tmp_path):
    f = tmp_path / "bin.dat"
    f.write_bytes(b"\x00\x01\x02\x03")

    result = tools.read_file(str(f))
    assert "error" in result


def test_read_file_missing_file_returns_error(tmp_path):
    result = tools.read_file(str(tmp_path / "nope.txt"))
    assert "error" in result


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------

def test_write_file_creates_new_file(tmp_path):
    f = tmp_path / "new.txt"

    result = tools.write_file(str(f), content="hello there")

    assert result["status"] == "created"
    assert f.read_text() == "hello there"


def test_write_file_creates_missing_parent_dirs(tmp_path):
    f = tmp_path / "nested" / "deep" / "new.txt"

    result = tools.write_file(str(f), content="x")

    assert result["status"] == "created"
    assert f.read_text() == "x"


def test_write_file_default_content_is_empty(tmp_path):
    f = tmp_path / "empty.txt"

    tools.write_file(str(f))

    assert f.read_text() == ""


def test_write_file_refuses_existing_without_overwrite(tmp_path):
    f = tmp_path / "existing.txt"
    f.write_text("original")

    result = tools.write_file(str(f), content="new content")

    assert "error" in result
    assert f.read_text() == "original"  # untouched


def test_write_file_overwrite_true_proposes_but_does_not_overwrite(tmp_path):
    f = tmp_path / "existing.txt"
    f.write_text("original")

    result = tools.write_file(str(f), content="new content", overwrite=True)

    assert result["status"] == "confirmation_required"
    assert f.read_text() == "original"  # still untouched


def test_confirm_pending_action_executes_overwrite(tmp_path):
    f = tmp_path / "existing.txt"
    f.write_text("original")

    proposal = tools.write_file(str(f), content="new content", overwrite=True)
    result = tools.confirm_pending_action(proposal["token"])

    assert result["status"] == "overwritten"
    assert f.read_text() == "new content"


def test_cancel_pending_overwrite_leaves_file_untouched(tmp_path):
    f = tmp_path / "existing.txt"
    f.write_text("original")

    proposal = tools.write_file(str(f), content="new content", overwrite=True)
    tools.cancel_pending_action(proposal["token"])

    assert f.read_text() == "original"


# ---------------------------------------------------------------------------
# read_screen — mocked, since there's no real display or Tesseract binary
# in a sandbox/CI environment (same reasoning Phase 5 used for faking
# chromadb before memory.py's lazy import ever runs)
# ---------------------------------------------------------------------------

def _install_fake_screen_deps(monkeypatch, ocr_text="mocked screen text", capture_exc=None, ocr_exc=None, grab_signature="modern"):
    """Installs fake PIL and pytesseract modules into sys.modules so
    read_screen() can be exercised without Pillow, pytesseract, a real
    screen, or the external Tesseract binary actually being present.
    Returns a dict the caller can inspect for what was passed through.

    grab_signature="legacy" simulates a pre-9.3 Pillow whose grab()
    doesn't accept all_screens at all, to exercise the fallback path."""
    captured = {}

    def fake_grab(bbox=None, all_screens=False):
        if grab_signature == "legacy" and all_screens:
            raise TypeError("grab() got an unexpected keyword argument 'all_screens'")
        captured["bbox"] = bbox
        captured["all_screens"] = all_screens
        if capture_exc:
            raise capture_exc
        return object()  # stand-in for a PIL Image

    def fake_grab_legacy_signature(bbox=None):
        # A true pre-9.3 grab() doesn't even accept the kwarg -- calling
        # it with all_screens=... raises TypeError before the function
        # body ever runs, which is what read_screen's except TypeError
        # branch is designed to catch.
        captured["bbox"] = bbox
        if capture_exc:
            raise capture_exc
        return object()

    fake_pil = types.ModuleType("PIL")
    fake_pil.ImageGrab = types.SimpleNamespace(
        grab=fake_grab_legacy_signature if grab_signature == "legacy" else fake_grab
    )
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)

    def fake_image_to_string(image):
        if ocr_exc:
            raise ocr_exc
        return ocr_text

    fake_pytesseract = types.ModuleType("pytesseract")
    fake_pytesseract.pytesseract = types.SimpleNamespace(tesseract_cmd=None)
    fake_pytesseract.image_to_string = fake_image_to_string
    monkeypatch.setitem(sys.modules, "pytesseract", fake_pytesseract)

    captured["pytesseract_module"] = fake_pytesseract
    return captured


def test_read_screen_returns_ocr_text(monkeypatch):
    _install_fake_screen_deps(monkeypatch, ocr_text="hello from the screen")

    result = tools.read_screen()

    assert result["text"] == "hello from the screen"
    assert result["region"] == "full screen"


def test_read_screen_strips_whitespace(monkeypatch):
    _install_fake_screen_deps(monkeypatch, ocr_text="  padded text  \n")

    result = tools.read_screen()

    assert result["text"] == "padded text"


def test_read_screen_empty_result_reports_no_text(monkeypatch):
    _install_fake_screen_deps(monkeypatch, ocr_text="   ")

    result = tools.read_screen()

    assert result["text"] == "(no readable text detected)"


def test_read_screen_with_region_passes_correct_bbox(monkeypatch):
    captured = _install_fake_screen_deps(monkeypatch)

    result = tools.read_screen(region=[10, 20, 100, 50])

    assert captured["bbox"] == (10, 20, 110, 70)  # left, top, left+width, top+height
    assert result["region"] == [10, 20, 100, 50]


def test_read_screen_invalid_region_length_returns_error(monkeypatch):
    _install_fake_screen_deps(monkeypatch)

    result = tools.read_screen(region=[1, 2, 3])

    assert "error" in result


def test_read_screen_capture_failure_returns_error(monkeypatch):
    _install_fake_screen_deps(monkeypatch, capture_exc=RuntimeError("no display"))

    result = tools.read_screen()

    assert "error" in result


def test_read_screen_ocr_failure_mentions_tesseract(monkeypatch):
    _install_fake_screen_deps(monkeypatch, ocr_exc=RuntimeError("tesseract is not installed"))

    result = tools.read_screen()

    assert "error" in result
    assert "tesseract" in result["error"].lower()


def test_read_screen_missing_pillow_returns_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "PIL", None)  # simulates Pillow not installed

    result = tools.read_screen()

    assert "error" in result
    assert "pillow" in result["error"].lower()


def test_read_screen_honors_tesseract_cmd_env_var(monkeypatch):
    captured = _install_fake_screen_deps(monkeypatch)
    monkeypatch.setenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")

    tools.read_screen()

    assert captured["pytesseract_module"].pytesseract.tesseract_cmd == r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def test_read_screen_is_llm_reachable():
    assert "read_screen" in tools.TOOL_FUNCTIONS
    schema_names = {s["function"]["name"] for s in tools.TOOL_SCHEMAS}
    assert "read_screen" in schema_names


def test_read_screen_defaults_to_primary_monitor_only(monkeypatch):
    captured = _install_fake_screen_deps(monkeypatch)

    result = tools.read_screen()

    assert captured["all_screens"] is False
    assert result["all_screens"] is False


def test_read_screen_all_screens_true_captures_virtual_desktop(monkeypatch):
    captured = _install_fake_screen_deps(monkeypatch)

    result = tools.read_screen(all_screens=True)

    assert captured["all_screens"] is True
    assert result["all_screens"] is True


def test_read_screen_all_screens_with_region_passes_both(monkeypatch):
    captured = _install_fake_screen_deps(monkeypatch)

    tools.read_screen(region=[-1920, 0, 1920, 1080], all_screens=True)

    assert captured["bbox"] == (-1920, 0, 0, 1080)
    assert captured["all_screens"] is True


def test_read_screen_legacy_pillow_without_all_screens_falls_back(monkeypatch):
    """Old Pillow that doesn't accept the all_screens kwarg at all should
    still work fine for a plain single-monitor read."""
    _install_fake_screen_deps(monkeypatch, grab_signature="legacy")

    result = tools.read_screen()

    assert "error" not in result


def test_read_screen_legacy_pillow_with_all_screens_returns_clear_error(monkeypatch):
    _install_fake_screen_deps(monkeypatch, grab_signature="legacy")

    result = tools.read_screen(all_screens=True)

    assert "error" in result
    assert "pillow" in result["error"].lower()
    assert "9.3" in result["error"]


# ---------------------------------------------------------------------------
# open_application
# ---------------------------------------------------------------------------

def test_open_application_uses_startfile_on_windows(monkeypatch):
    calls = []
    monkeypatch.setattr(tools.platform, "system", lambda: "Windows")
    monkeypatch.setattr(tools.os, "startfile", lambda target: calls.append(target), raising=False)

    result = tools.open_application("notepad")

    assert result["status"] == "launched"
    assert calls == ["notepad"]


def test_open_application_spotify_uses_uri_launcher(monkeypatch):
    calls = []
    monkeypatch.setattr(tools.platform, "system", lambda: "Windows")
    monkeypatch.setattr(tools.os, "startfile", lambda target: calls.append(target), raising=False)

    tools.open_application("Spotify", query="lofi beats")

    assert calls == ["spotify:search:lofi beats"]


def test_open_application_spotify_no_query_uses_bare_uri(monkeypatch):
    calls = []
    monkeypatch.setattr(tools.platform, "system", lambda: "Windows")
    monkeypatch.setattr(tools.os, "startfile", lambda target: calls.append(target), raising=False)

    tools.open_application("spotify")

    assert calls == ["spotify:"]


def test_open_application_macos_uses_open_command(monkeypatch):
    calls = []
    monkeypatch.setattr(tools.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(tools.subprocess, "Popen", lambda args: calls.append(args))

    tools.open_application("TextEdit")

    assert calls == [["open", "TextEdit"]]


def test_open_application_linux_uses_xdg_open(monkeypatch):
    calls = []
    monkeypatch.setattr(tools.platform, "system", lambda: "Linux")
    monkeypatch.setattr(tools.subprocess, "Popen", lambda args: calls.append(args))

    tools.open_application("gedit")

    assert calls == [["xdg-open", "gedit"]]


def test_open_application_failure_returns_error(monkeypatch):
    monkeypatch.setattr(tools.platform, "system", lambda: "Windows")

    def _raise(target):
        raise OSError("no such app")

    monkeypatch.setattr(tools.os, "startfile", _raise, raising=False)

    result = tools.open_application("nonexistent_app")
    assert "error" in result


# ---------------------------------------------------------------------------
# delete_file — two-phase confirmation
# ---------------------------------------------------------------------------

def test_delete_file_proposes_but_does_not_delete(tmp_path):
    f = tmp_path / "target.txt"
    f.write_text("keep me for now")

    result = tools.delete_file(str(f))

    assert result["status"] == "confirmation_required"
    assert "token" in result
    assert f.exists()  # nothing deleted yet


def test_delete_file_missing_file_returns_error(tmp_path):
    result = tools.delete_file(str(tmp_path / "nope.txt"))
    assert "error" in result


def test_confirm_pending_action_deletes_file(tmp_path):
    f = tmp_path / "target.txt"
    f.write_text("bye")

    proposal = tools.delete_file(str(f))
    result = tools.confirm_pending_action(proposal["token"])

    assert result["status"] == "deleted"
    assert not f.exists()


def test_confirm_pending_action_invalid_token_returns_error():
    result = tools.confirm_pending_action("not-a-real-token")
    assert "error" in result
    assert "invalid" in result["error"].lower()


def test_confirm_pending_action_token_single_use(tmp_path):
    f = tmp_path / "target.txt"
    f.write_text("bye")
    proposal = tools.delete_file(str(f))

    tools.confirm_pending_action(proposal["token"])
    second_attempt = tools.confirm_pending_action(proposal["token"])

    assert "error" in second_attempt


def test_cancel_pending_action_does_not_delete(tmp_path):
    f = tmp_path / "target.txt"
    f.write_text("keep me")
    proposal = tools.delete_file(str(f))

    result = tools.cancel_pending_action(proposal["token"])

    assert result["status"] == "cancelled"
    assert f.exists()
    # token is no longer usable after cancellation
    assert "error" in tools.confirm_pending_action(proposal["token"])


def test_expired_token_cannot_be_confirmed(tmp_path):
    f = tmp_path / "target.txt"
    f.write_text("bye")
    proposal = tools.delete_file(str(f))

    # simulate the confirmation having been sitting around too long
    tools._pending_actions[proposal["token"]]["created_at"] = (
        time.time() - (tools.PENDING_TTL_SECONDS + 1)
    )

    result = tools.confirm_pending_action(proposal["token"])

    assert "error" in result
    assert "expired" in result["error"].lower()
    assert f.exists()


def test_expired_token_message_is_distinct_from_invalid_token_message():
    """A human who waited too long deserves a different message than one
    who mistyped or reused a token — that distinction is the whole point
    of this fix."""
    f_result = tools.confirm_pending_action("never-existed")
    assert "expired" not in f_result["error"].lower()
    assert "invalid" in f_result["error"].lower() or "already used" in f_result["error"].lower()


# ---------------------------------------------------------------------------
# execute_tool dispatch — the LLM-facing surface
# ---------------------------------------------------------------------------

def test_execute_tool_unknown_name_raises_keyerror():
    with pytest.raises(KeyError):
        tools.execute_tool("not_a_real_tool", {})


def test_confirm_and_cancel_are_not_llm_reachable():
    """The guardrail only holds if the model has no path to these —
    verify they're absent from both the dispatch table and the schema
    list the model is given."""
    assert "confirm_pending_action" not in tools.TOOL_FUNCTIONS
    assert "cancel_pending_action" not in tools.TOOL_FUNCTIONS
    schema_names = {s["function"]["name"] for s in tools.TOOL_SCHEMAS}
    assert "confirm_pending_action" not in schema_names
    assert "cancel_pending_action" not in schema_names


def test_write_file_is_llm_reachable():
    assert "write_file" in tools.TOOL_FUNCTIONS
    schema_names = {s["function"]["name"] for s in tools.TOOL_SCHEMAS}
    assert "write_file" in schema_names


def test_execute_tool_dispatches_to_list_directory(tmp_path):
    (tmp_path / "f.txt").write_text("x")
    result = tools.execute_tool("list_directory", {"path": str(tmp_path)})
    assert result["entries"][0]["name"] == "f.txt"


def test_execute_tool_delete_file_only_proposes(tmp_path):
    f = tmp_path / "target.txt"
    f.write_text("still here")

    result = tools.execute_tool("delete_file", {"path": str(f)})

    assert result["status"] == "confirmation_required"
    assert f.exists()


# ---------------------------------------------------------------------------
# Mouse / keyboard automation — mocked, since there's no real display or
# input device to drive in a sandbox/CI environment
# ---------------------------------------------------------------------------

def _install_fake_pyautogui(monkeypatch, move_exc=None, write_exc=None, click_exc=None, hotkey_exc=None):
    """Installs a fake pyautogui module into sys.modules. Returns the
    fake module so tests can inspect exactly what was called."""
    calls = {}

    def fake_moveTo(x, y):
        calls["moveTo"] = (x, y)
        if move_exc:
            raise move_exc

    def fake_write(text, interval=0.0):
        calls["write"] = (text, interval)
        if write_exc:
            raise write_exc

    def fake_click(x=None, y=None, clicks=1, button="left"):
        calls["click"] = {"x": x, "y": y, "clicks": clicks, "button": button}
        if click_exc:
            raise click_exc

    def fake_hotkey(*keys):
        calls["hotkey"] = keys
        if hotkey_exc:
            raise hotkey_exc

    fake_module = types.ModuleType("pyautogui")
    fake_module.FAILSAFE = False  # will be forced True by _get_pyautogui — verified below
    fake_module.moveTo = fake_moveTo
    fake_module.write = fake_write
    fake_module.click = fake_click
    fake_module.hotkey = fake_hotkey
    fake_module.calls = calls

    monkeypatch.setitem(sys.modules, "pyautogui", fake_module)
    return fake_module


def test_get_pyautogui_forces_failsafe_on(monkeypatch):
    fake = _install_fake_pyautogui(monkeypatch)
    assert fake.FAILSAFE is False  # sanity check on the fake's initial state

    tools._get_pyautogui()

    assert fake.FAILSAFE is True


# ---------------------------------------------------------------------------
# _get_active_window_title / get_active_window — the fix for the "hotkey
# hit the wrong window" finding
# ---------------------------------------------------------------------------

def test_active_window_title_non_windows_returns_placeholder(monkeypatch):
    monkeypatch.setattr(tools.platform, "system", lambda: "Linux")

    title = tools._get_active_window_title()

    assert "windows-only" in title.lower()


def _install_fake_user32(monkeypatch, window_title="Visual Studio Code"):
    """Fakes ctypes.windll.user32's three calls _get_active_window_title
    uses, without needing a real Windows environment."""
    class FakeUser32:
        @staticmethod
        def GetForegroundWindow():
            return 12345

        @staticmethod
        def GetWindowTextLengthW(hwnd):
            return len(window_title)

        @staticmethod
        def GetWindowTextW(hwnd, buf, size):
            buf.value = window_title

    fake_windll = types.SimpleNamespace(user32=FakeUser32())
    monkeypatch.setattr(tools.platform, "system", lambda: "Windows")
    monkeypatch.setattr(tools.ctypes, "windll", fake_windll, raising=False)
    return fake_windll


def test_active_window_title_returns_real_title_on_windows(monkeypatch):
    _install_fake_user32(monkeypatch, window_title="Visual Studio Code")

    title = tools._get_active_window_title()

    assert title == "Visual Studio Code"


def test_active_window_title_failure_returns_placeholder_not_exception(monkeypatch):
    monkeypatch.setattr(tools.platform, "system", lambda: "Windows")

    class BrokenUser32:
        def GetForegroundWindow(self):
            raise OSError("no window manager")

    monkeypatch.setattr(
        tools.ctypes, "windll", types.SimpleNamespace(user32=BrokenUser32()), raising=False
    )

    title = tools._get_active_window_title()

    assert "could not determine" in title.lower()


def test_get_active_window_is_llm_reachable_ungated():
    assert "get_active_window" in tools.TOOL_FUNCTIONS
    schema_names = {s["function"]["name"] for s in tools.TOOL_SCHEMAS}
    assert "get_active_window" in schema_names
    # not a destructive/confirmation-gated action
    assert "get_active_window" not in (
        "confirm_pending_action", "cancel_pending_action"
    )


def test_click_confirmation_message_names_active_window(monkeypatch):
    _install_fake_pyautogui(monkeypatch)
    _install_fake_user32(monkeypatch, window_title="Notepad")

    result = tools.click(50, 60)

    assert result["active_window"] == "Notepad"
    assert "Notepad" in result["message"]


def test_press_hotkey_confirmation_message_names_active_window(monkeypatch):
    """The core regression test for the real finding: a hotkey meant for
    a just-launched app landing on whatever's actually focused instead
    (e.g. closing VS Code instead of the Notepad it was meant for). The
    confirmation message must surface that mismatch before it happens."""
    _install_fake_pyautogui(monkeypatch)
    _install_fake_user32(monkeypatch, window_title="Visual Studio Code")

    result = tools.press_hotkey(["alt", "f4"])

    assert result["active_window"] == "Visual Studio Code"
    assert "Visual Studio Code" in result["message"]
    assert "alt+f4" in result["message"]


# ---------------------------------------------------------------------------
# _abort_if_focus_shifted — the second-round fix. Naming the window in the
# confirmation message (above) wasn't enough on its own: confirming at the
# terminal shifts focus back before the action fires, even when the named
# window was correct at proposal time. This re-checks immediately before
# execution and refuses if focus moved.
# ---------------------------------------------------------------------------

def test_confirm_click_proceeds_when_focus_unchanged(monkeypatch):
    fake = _install_fake_pyautogui(monkeypatch)
    _install_fake_user32(monkeypatch, window_title="Notepad")

    proposal = tools.click(50, 60)
    result = tools.confirm_pending_action(proposal["token"])

    assert result["status"] == "clicked"
    assert "click" in fake.calls


def test_confirm_click_aborts_when_focus_shifted(monkeypatch):
    """The direct regression test for the real incident: propose while
    Notepad has focus, but by the time confirmation actually executes,
    the terminal (here standing in for VS Code) has focus instead --
    the click must NOT fire."""
    fake = _install_fake_pyautogui(monkeypatch)
    _install_fake_user32(monkeypatch, window_title="Notepad")
    proposal = tools.click(50, 60)

    # simulate focus having shifted back to the terminal by the time the
    # human finishes typing the confirmation password
    _install_fake_user32(monkeypatch, window_title="Visual Studio Code")

    result = tools.confirm_pending_action(proposal["token"])

    assert "error" in result
    assert "notepad" in result["error"].lower()
    assert "visual studio code" in result["error"].lower()
    assert "click" not in fake.calls  # the click must never have fired


def test_confirm_hotkey_aborts_when_focus_shifted(monkeypatch):
    fake = _install_fake_pyautogui(monkeypatch)
    _install_fake_user32(monkeypatch, window_title="Notepad")
    proposal = tools.press_hotkey(["alt", "f4"])

    _install_fake_user32(monkeypatch, window_title="Visual Studio Code")

    result = tools.confirm_pending_action(proposal["token"])

    assert "error" in result
    assert "hotkey" not in fake.calls  # the keys must never have been sent


def test_confirm_hotkey_proceeds_when_focus_unchanged(monkeypatch):
    fake = _install_fake_pyautogui(monkeypatch)
    _install_fake_user32(monkeypatch, window_title="Notepad")

    proposal = tools.press_hotkey(["alt", "f4"])
    result = tools.confirm_pending_action(proposal["token"])

    assert result["status"] == "pressed"
    assert fake.calls["hotkey"] == ("alt", "f4")


def test_confirm_click_skips_focus_check_on_non_windows(monkeypatch):
    """On non-Windows, _get_active_window_title() always returns the same
    placeholder, so expected_window == current_window trivially and the
    action should still proceed rather than deadlock against itself."""
    fake = _install_fake_pyautogui(monkeypatch)
    monkeypatch.setattr(tools.platform, "system", lambda: "Linux")

    proposal = tools.click(50, 60)
    result = tools.confirm_pending_action(proposal["token"])

    assert result["status"] == "clicked"
    assert "click" in fake.calls


def test_move_mouse_moves_immediately_no_confirmation(monkeypatch):
    fake = _install_fake_pyautogui(monkeypatch)

    result = tools.move_mouse(100, 200)

    assert result["status"] == "moved"
    assert fake.calls["moveTo"] == (100, 200)


def test_move_mouse_missing_pyautogui_returns_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyautogui", None)
    result = tools.move_mouse(1, 1)
    assert "error" in result


def test_type_text_types_immediately_no_confirmation(monkeypatch):
    fake = _install_fake_pyautogui(monkeypatch)

    result = tools.type_text("hello world")

    assert result["status"] == "typed"
    assert fake.calls["write"] == ("hello world", 0.0)


def test_type_text_empty_string_returns_error(monkeypatch):
    _install_fake_pyautogui(monkeypatch)
    result = tools.type_text("")
    assert "error" in result


def test_click_only_proposes_does_not_click(monkeypatch):
    fake = _install_fake_pyautogui(monkeypatch)

    result = tools.click(50, 60)

    assert result["status"] == "confirmation_required"
    assert "click" not in fake.calls  # nothing actually happened yet


def test_click_invalid_button_returns_error(monkeypatch):
    _install_fake_pyautogui(monkeypatch)
    result = tools.click(1, 1, button="laser")
    assert "error" in result


def test_click_zero_clicks_returns_error(monkeypatch):
    _install_fake_pyautogui(monkeypatch)
    result = tools.click(1, 1, clicks=0)
    assert "error" in result


def test_confirm_pending_action_executes_click(monkeypatch):
    fake = _install_fake_pyautogui(monkeypatch)
    proposal = tools.click(50, 60, button="right", clicks=2)

    result = tools.confirm_pending_action(proposal["token"])

    assert result["status"] == "clicked"
    assert fake.calls["click"] == {"x": 50, "y": 60, "clicks": 2, "button": "right"}


def test_cancel_pending_click_never_clicks(monkeypatch):
    fake = _install_fake_pyautogui(monkeypatch)
    proposal = tools.click(50, 60)

    tools.cancel_pending_action(proposal["token"])

    assert "click" not in fake.calls


def test_press_hotkey_only_proposes_does_not_press(monkeypatch):
    fake = _install_fake_pyautogui(monkeypatch)

    result = tools.press_hotkey(["ctrl", "s"])

    assert result["status"] == "confirmation_required"
    assert "hotkey" not in fake.calls


def test_press_hotkey_empty_list_returns_error(monkeypatch):
    _install_fake_pyautogui(monkeypatch)
    result = tools.press_hotkey([])
    assert "error" in result


def test_confirm_pending_action_executes_hotkey(monkeypatch):
    fake = _install_fake_pyautogui(monkeypatch)
    proposal = tools.press_hotkey(["alt", "f4"])

    result = tools.confirm_pending_action(proposal["token"])

    assert result["status"] == "pressed"
    assert fake.calls["hotkey"] == ("alt", "f4")


def test_confirm_pending_click_failsafe_exception_returns_error(monkeypatch):
    """pyautogui.FailSafeException (mouse yanked to a corner mid-action)
    must surface as a clean error, not crash the confirmation flow."""
    _install_fake_pyautogui(monkeypatch, click_exc=RuntimeError("fail-safe triggered"))
    proposal = tools.click(50, 60)

    result = tools.confirm_pending_action(proposal["token"])

    assert "error" in result


def test_move_mouse_and_type_text_are_llm_reachable_without_confirmation():
    """move_mouse/type_text should be directly callable — no confirmation
    layer, per the explicit low-risk decision."""
    assert "move_mouse" in tools.TOOL_FUNCTIONS
    assert "type_text" in tools.TOOL_FUNCTIONS
    schema_names = {s["function"]["name"] for s in tools.TOOL_SCHEMAS}
    assert "move_mouse" in schema_names
    assert "type_text" in schema_names


def test_click_and_press_hotkey_are_llm_reachable():
    assert "click" in tools.TOOL_FUNCTIONS
    assert "press_hotkey" in tools.TOOL_FUNCTIONS
    schema_names = {s["function"]["name"] for s in tools.TOOL_SCHEMAS}
    assert "click" in schema_names
    assert "press_hotkey" in schema_names


# ---------------------------------------------------------------------------
# execute_tool dispatch — the LLM-facing surface (continued)
# ---------------------------------------------------------------------------

def test_execute_tool_click_only_proposes(monkeypatch):
    fake = _install_fake_pyautogui(monkeypatch)

    result = tools.execute_tool("click", {"x": 1, "y": 1})

    assert result["status"] == "confirmation_required"
    assert "click" not in fake.calls