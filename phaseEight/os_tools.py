"""
os_tools.py -- Phase 8, milestone 2: deeper OS and window control.

Phase 6 gave Jarvis hands: launch an app, read the screen, move the
mouse, type, click, press a hotkey. What it could not do is *find* or
*manage* anything -- it had no way to know which windows were open,
bring one to the front, close one, see what was running, stop a runaway
process, reach the clipboard, or control playback. In practice that
meant the model had to drive everything through screen OCR and blind
hotkeys, which is exactly the fragile path Phase 6's own system prompt
warns it away from.

Everything here is local: no API keys, no network. Windows-first, same
as the rest of this project, using ctypes against user32 directly
rather than adding pywin32 -- matching the choice Phase 6 already made
for get_active_window(). On other platforms the window and power tools
return a clear error rather than pretending.

Guardrail split, stated explicitly rather than assumed (Phase 5's
handoff recommended deciding this up front, and Phase 6 followed the
same practice):

  UNGATED -- reversible, or destroys nothing:
      list_windows, focus_window, move_window, list_processes,
      get_clipboard, set_clipboard, media_control, take_screenshot,
      get_system_status, lock_screen

  GATED -- can destroy unsaved work or interrupt everything:
      close_window, kill_process, power_action

Three of those calls deserve their reasoning written down:

  lock_screen is ungated even though it is disruptive, because it is
  the one power action that destroys nothing (you log back in and
  everything is exactly where you left it) and because gating it is
  self-defeating: the entire point is saying "lock my machine" while
  walking away from it, and a gate would require staying to type a
  password in order to lock.

  media_control is ungated despite Phase 6 gating hotkeys, because it
  is not a hotkey tool. It sends one of six fixed media keys from a
  closed allowlist, none of which can close a window, quit an app, or
  trigger anything system-level. Phase 6 gated press_hotkey because it
  accepts arbitrary combinations; that reasoning does not transfer to
  a fixed safe set.

  set_clipboard is ungated, matching Phase 6's judgment that typing is
  lower-risk than clicking. It does discard whatever was on the
  clipboard before, which is a small real loss -- noted, and judged not
  worth a password prompt.
"""

import ctypes
import os
import platform
import subprocess
import time
from ctypes import wintypes

import gate

IS_WINDOWS = platform.system() == "Windows"

NOT_WINDOWS = {"error": "This tool is Windows-only, and this machine is not running Windows."}

# The six media keys pyautogui exposes, as a closed allowlist. See the
# module docstring for why a fixed set is ungated where press_hotkey's
# arbitrary combinations are not.
MEDIA_KEYS = {
    "play_pause": "playpause",
    "next_track": "nexttrack",
    "previous_track": "prevtrack",
    "volume_up": "volumeup",
    "volume_down": "volumedown",
    "mute": "volumemute",
}

# Windows will not let a process be killed by name alone here -- see
# kill_process for why this is a deliberate restriction and not an
# oversight.
PROTECTED_PROCESS_NAMES = {
    "system", "system idle process", "registry", "csrss.exe", "wininit.exe",
    "winlogon.exe", "services.exe", "lsass.exe", "smss.exe", "svchost.exe",
    "dwm.exe", "explorer.exe",
}


# ---------------------------------------------------------------------------
# Window enumeration (ctypes/user32, no extra dependency)
# ---------------------------------------------------------------------------

class _RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


def _window_title(hwnd) -> str:
    """Title of one window handle, or "" if it has none. Never raises."""
    try:
        user32 = ctypes.windll.user32
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value or ""
    except Exception:
        return ""


def _is_cloaked(hwnd) -> bool:
    """True for a window Windows reports as visible but is not actually
    showing.

    Suspended UWP apps (Settings, Calculator, Mail) leave an
    ApplicationFrameWindow behind that IsWindowVisible() happily calls
    visible, while DWM has it cloaked and nothing is on screen. Before
    this check, list_windows reported those as real -- and the model
    dutifully told the user it could see "two Settings windows" that
    were not open in any sense a person would recognise.

    It also made the window tools disagree with each other: the
    accessibility tree does not contain cloaked windows, so asking for
    the elements of a window list_windows had just advertised returned
    "no window found".

    Never raises -- on any failure the window is treated as real, which
    is the same behavior as before this check existed.
    """
    try:
        DWMWA_CLOAKED = 14
        cloaked = ctypes.c_int(0)
        result = ctypes.windll.dwmapi.DwmGetWindowAttribute(
            wintypes.HWND(hwnd),
            ctypes.c_int(DWMWA_CLOAKED),
            ctypes.byref(cloaked),
            ctypes.sizeof(cloaked),
        )
        return result == 0 and cloaked.value != 0
    except Exception:
        return False


def _enum_visible_windows() -> list:
    """Every visible top-level window that has a title, as
    [(hwnd, title), ...].

    Untitled, invisible and DWM-cloaked windows are skipped because they
    are not things a person can refer to out loud (see _is_cloaked for
    why "visible" is not enough on its own). A voice assistant asked to
    "close the browser" needs the list a human would recognize, not the
    hundreds of invisible message-only and tool windows Windows keeps
    around -- handing the model all of them would mostly be a way for
    it to pick the wrong one.
    """
    results = []
    user32 = ctypes.windll.user32
    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def callback(hwnd, _lparam):
        if user32.IsWindowVisible(hwnd) and not _is_cloaked(hwnd):
            title = _window_title(hwnd)
            if title:
                results.append((hwnd, title))
        return True

    user32.EnumWindows(enum_proc(callback), 0)
    return results


def _window_rect(hwnd) -> dict:
    try:
        rect = _RECT()
        if not ctypes.windll.user32.GetWindowRect(hwnd, ctypes.pointer(rect)):
            return {}
        return {
            "x": rect.left,
            "y": rect.top,
            "width": rect.right - rect.left,
            "height": rect.bottom - rect.top,
        }
    except Exception:
        return {}


def _find_windows(title_contains: str) -> list:
    needle = title_contains.strip().lower()
    return [(h, t) for h, t in _enum_visible_windows() if needle in t.lower()]


def _resolve_one_window(title_contains: str):
    """Resolves a spoken description to exactly one window, or returns
    an error dict explaining why it couldn't.

    Ambiguity is reported rather than guessed at. Picking the first of
    four matching browser windows would be right often enough to be
    trusted and wrong often enough to close the wrong one, and the tool
    that follows this resolution is frequently close_window.
    """
    matches = _find_windows(title_contains)
    if not matches:
        return None, {"error": f"No visible window found with '{title_contains}' in its title."}
    if len(matches) > 1:
        return None, {
            "error": (
                f"{len(matches)} windows match '{title_contains}': "
                + "; ".join(t for _, t in matches[:6])
                + ". Be more specific."
            )
        }
    return matches[0], None


# ---------------------------------------------------------------------------
# Window tools
# ---------------------------------------------------------------------------

def list_windows(title_contains: str = None) -> dict:
    """Read-only. Lists visible windows, optionally filtered by title."""
    if not IS_WINDOWS:
        return dict(NOT_WINDOWS)
    try:
        windows = _find_windows(title_contains) if title_contains else _enum_visible_windows()
    except Exception as e:
        return {"error": f"Could not enumerate windows: {e}"}
    return {
        "count": len(windows),
        "windows": [{"title": t, **_window_rect(h)} for h, t in windows],
    }


def focus_window(title_contains: str) -> dict:
    """
    Brings a window to the foreground. Ungated -- switching windows
    destroys nothing.

    Windows' foreground lock means SetForegroundWindow can report
    success while silently declining to switch (it flashes the taskbar
    icon instead), so the result is verified by re-reading the
    foreground window rather than trusting the return value -- the same
    thing Phase 6's _try_refocus_window() does, and for the same
    reason. A tool that says "focused" when it didn't is worse than one
    that admits it failed, because the model's next step is usually to
    send that window a hotkey.
    """
    if not IS_WINDOWS:
        return dict(NOT_WINDOWS)

    match, error = _resolve_one_window(title_contains)
    if error:
        return error
    hwnd, title = match

    try:
        user32 = ctypes.windll.user32
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE -- un-minimize first
        user32.SetForegroundWindow(hwnd)
        time.sleep(0.15)  # let the window manager actually switch
        focused = _window_title(user32.GetForegroundWindow())
    except Exception as e:
        return {"error": f"Could not focus '{title}': {e}"}

    if focused != title:
        return {
            "status": "focus_refused",
            "window": title,
            "focused_instead": focused,
            "note": (
                "Windows declined the focus change (foreground lock). The window "
                "was restored if minimized, but does not have keyboard focus -- "
                "do not send it a hotkey or click."
            ),
        }
    return {"status": "focused", "window": title}


def move_window(title_contains: str, x: int = None, y: int = None,
                width: int = None, height: int = None) -> dict:
    """Moves and/or resizes a window. Ungated -- fully reversible, and
    nothing is lost. Omitted values keep the window's current ones."""
    if not IS_WINDOWS:
        return dict(NOT_WINDOWS)

    match, error = _resolve_one_window(title_contains)
    if error:
        return error
    hwnd, title = match

    current = _window_rect(hwnd)
    if not current:
        return {"error": f"Could not read the current position of '{title}'."}

    target = {
        "x": current["x"] if x is None else x,
        "y": current["y"] if y is None else y,
        "width": current["width"] if width is None else width,
        "height": current["height"] if height is None else height,
    }
    if target["width"] <= 0 or target["height"] <= 0:
        return {"error": "width and height must be positive."}

    try:
        ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        moved = ctypes.windll.user32.MoveWindow(
            hwnd, target["x"], target["y"], target["width"], target["height"], True
        )
    except Exception as e:
        return {"error": f"Could not move '{title}': {e}"}

    if not moved:
        return {"error": f"Windows refused to move '{title}'."}
    return {"status": "moved", "window": title, **target}


def close_window(title_contains: str) -> dict:
    """
    GATED. Proposes closing a window; closes nothing on this call.

    Closing a window can discard unsaved work, which puts it in exactly
    the same bucket as Phase 6's delete_file. It goes through the same
    two-phase gate: this returns a token, and only a human confirming
    out-of-band actually sends the close.

    The window handle is re-verified at confirmation time. Between
    proposal and confirmation a window can close on its own and Windows
    can hand its handle to a completely different window -- confirming
    a stale handle would close whatever inherited it. The title is
    re-read and must still match before anything is sent, which is the
    same class of check as Phase 6's _abort_if_focus_shifted().
    """
    if not IS_WINDOWS:
        return dict(NOT_WINDOWS)

    match, error = _resolve_one_window(title_contains)
    if error:
        return error
    hwnd, title = match

    def _execute():
        if _window_title(hwnd) != title:
            return {
                "error": (
                    f"'{title}' is no longer the window it was when this was proposed "
                    "(it closed, or its title changed). Nothing was closed."
                )
            }
        WM_CLOSE = 0x0010
        ctypes.windll.user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
        return {"status": "closed", "window": title}

    return gate.propose(
        action="close_window",
        message=f"About to close the window '{title}'. Any unsaved work in it may be lost.",
        execute=_execute,
        window=title,
    )


# ---------------------------------------------------------------------------
# Process tools
# ---------------------------------------------------------------------------

def _get_psutil():
    """Lazy import, same pattern as Phase 6's _get_tesseract()/
    _get_pyautogui() -- keeps importing this module cheap and lets the
    window/clipboard tools be used and tested without psutil."""
    import psutil

    return psutil


def list_processes(name_contains: str = None, limit: int = 60) -> dict:
    """
    Read-only. Running processes, biggest memory first.

    Sorted by memory rather than listed in PID order because the
    question behind this is almost always "what is eating my machine",
    and because an unfiltered PID-ordered dump of 300 processes is
    mostly noise the model then has to summarize. `limit` caps it for
    the same reason.
    """
    try:
        psutil = _get_psutil()
    except ImportError as e:
        return {"error": f"psutil is not installed: {e}. Run: pip install psutil"}

    needle = name_contains.strip().lower() if name_contains else None
    processes = []
    for proc in psutil.process_iter(["pid", "name", "memory_info", "cpu_percent"]):
        try:
            info = proc.info
            name = info.get("name") or ""
            if needle and needle not in name.lower():
                continue
            memory = info.get("memory_info")
            processes.append({
                "pid": info["pid"],
                "name": name,
                "memory_mb": round(memory.rss / (1024 * 1024), 1) if memory else None,
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # A process that exited mid-scan, or one this user can't
            # inspect, shouldn't blow up the whole listing -- same
            # per-entry tolerance Phase 6's list_directory uses.
            continue

    processes.sort(key=lambda p: p["memory_mb"] or 0, reverse=True)
    return {
        "count": len(processes),
        "filter": name_contains,
        "processes": processes[:limit],
    }


def kill_process(pid: int = None, name: str = None) -> dict:
    """
    GATED. Proposes terminating a process; kills nothing on this call.

    Killing a process discards whatever it had unsaved, so it goes
    through the same two-phase gate as delete_file.

    Requires a PID, or a name that matches exactly one process. A name
    matching several is refused rather than resolved to "all of them"
    -- "close Chrome" meaning eleven processes at once is precisely the
    kind of thing that should not ride in on a single confirmation.

    Two guards beyond the gate itself:

      - A protected-process allowlist (see PROTECTED_PROCESS_NAMES).
        Killing lsass.exe or winlogon.exe bluescreens or logs out the
        machine immediately. These are refused outright rather than
        gated, because there is no legitimate voice-assistant reason to
        reach for them and a confirmation prompt is not a good place to
        be discovering that.
      - PID reuse. Between proposal and confirmation the target can
        exit and the OS can reassign its PID to something else --
        confirming a stale PID would kill an unrelated, newer process.
        The process's creation timestamp is captured at proposal and
        re-checked at confirmation, which is the standard way to pin a
        PID to the exact process instance it referred to.
    """
    try:
        psutil = _get_psutil()
    except ImportError as e:
        return {"error": f"psutil is not installed: {e}. Run: pip install psutil"}

    if pid is None and not name:
        return {"error": "Give either a pid or a name."}

    try:
        if pid is not None:
            target = psutil.Process(pid)
        else:
            needle = name.strip().lower()
            matches = [
                p for p in psutil.process_iter(["pid", "name"])
                if needle in (p.info.get("name") or "").lower()
            ]
            if not matches:
                return {"error": f"No running process matching '{name}'."}
            if len(matches) > 1:
                return {
                    "error": (
                        f"{len(matches)} processes match '{name}': "
                        + ", ".join(f"{p.info['name']} (pid {p.info['pid']})" for p in matches[:6])
                        + ". Give a specific pid."
                    )
                }
            target = psutil.Process(matches[0].info["pid"])

        target_name = target.name()
        target_pid = target.pid
        created_at = target.create_time()
    except psutil.NoSuchProcess:
        return {"error": f"No such process: {pid}"}
    except psutil.AccessDenied:
        return {"error": f"Access denied reading process {pid or name}."}

    if target_name.lower() in PROTECTED_PROCESS_NAMES:
        return {
            "error": (
                f"'{target_name}' is a protected system process. Killing it would "
                "crash or log out this machine, so it is refused outright rather "
                "than offered for confirmation."
            )
        }

    def _execute():
        try:
            proc = psutil.Process(target_pid)
            if proc.create_time() != created_at:
                return {
                    "error": (
                        f"PID {target_pid} is no longer the '{target_name}' process that "
                        "was proposed -- it exited and the PID was reused. Nothing was killed."
                    )
                }
            proc.terminate()
            proc.wait(timeout=5)
            return {"status": "killed", "name": target_name, "pid": target_pid}
        except psutil.NoSuchProcess:
            return {"status": "already_exited", "name": target_name, "pid": target_pid}
        except psutil.TimeoutExpired:
            return {
                "status": "still_running",
                "name": target_name,
                "pid": target_pid,
                "note": "Asked it to close but it did not exit within 5 seconds.",
            }
        except psutil.AccessDenied:
            return {"error": f"Access denied killing {target_name} (pid {target_pid})."}

    return gate.propose(
        action="kill_process",
        message=(
            f"About to terminate '{target_name}' (pid {target_pid}). "
            "Any unsaved work in it will be lost."
        ),
        execute=_execute,
        name=target_name,
        pid=target_pid,
    )


# ---------------------------------------------------------------------------
# Clipboard
# ---------------------------------------------------------------------------

def _get_pyperclip():
    import pyperclip

    return pyperclip


def get_clipboard(max_chars: int = 20000) -> dict:
    """Read-only. Truncated like Phase 6's read_file, and for the same
    reason: no single tool call should be able to dump an unbounded
    blob into the model's context."""
    try:
        text = _get_pyperclip().paste()
    except ImportError as e:
        return {"error": f"pyperclip is not installed: {e}. Run: pip install pyperclip"}
    except Exception as e:
        return {"error": f"Could not read the clipboard: {e}"}

    text = text or ""
    truncated = len(text) > max_chars
    return {
        "content": text[:max_chars],
        "truncated": truncated,
        "char_count": len(text),
    }


def set_clipboard(text: str) -> dict:
    """Ungated -- see the module docstring. Replaces whatever was on the
    clipboard, which is a small real loss and the reason this is worth
    a note rather than nothing."""
    try:
        _get_pyperclip().copy(text)
    except ImportError as e:
        return {"error": f"pyperclip is not installed: {e}. Run: pip install pyperclip"}
    except Exception as e:
        return {"error": f"Could not write to the clipboard: {e}"}
    return {"status": "copied", "char_count": len(text)}


# ---------------------------------------------------------------------------
# Media / volume
# ---------------------------------------------------------------------------

def media_control(action: str) -> dict:
    """Ungated. Sends one of six fixed media keys -- see the module
    docstring for why this is not the same thing as Phase 6's gated
    press_hotkey."""
    key = MEDIA_KEYS.get(action.strip().lower())
    if key is None:
        return {"error": f"Unknown action '{action}'. Valid: {', '.join(sorted(MEDIA_KEYS))}."}
    try:
        import pyautogui

        pyautogui.FAILSAFE = True  # forced on, same as Phase 6 -- never disableable
        pyautogui.press(key)
    except ImportError as e:
        return {"error": f"pyautogui is not installed: {e}. Run: pip install pyautogui"}
    except Exception as e:
        return {"error": f"Could not send media key: {e}"}
    return {"status": "sent", "action": action, "key": key}


# ---------------------------------------------------------------------------
# Screenshot / system status / power
# ---------------------------------------------------------------------------

def take_screenshot(path: str, all_screens: bool = False) -> dict:
    """
    Saves a screenshot to a file. Distinct from Phase 6's read_screen,
    which OCRs to text and deliberately never writes the image
    anywhere.

    That distinction is the whole reason this needs saying: read_screen
    keeps the capture in memory for one call precisely because a
    screenshot can contain anything on screen -- a password field, a
    private conversation. This tool writes that to disk, where it
    persists. It is still ungated, because creating a new file destroys
    nothing, but it refuses to overwrite an existing one, exactly as
    Phase 6's write_file does.
    """
    abs_path = os.path.abspath(path)
    if os.path.exists(abs_path):
        return {"error": f"'{abs_path}' already exists. Choose a different filename."}

    try:
        from PIL import ImageGrab
    except ImportError as e:
        return {"error": f"Pillow is not installed: {e}. Run: pip install Pillow"}

    try:
        parent = os.path.dirname(abs_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        image = ImageGrab.grab(all_screens=all_screens)
        image.save(abs_path)
    except Exception as e:
        return {"error": f"Could not save screenshot: {e}"}

    return {
        "status": "saved",
        "path": abs_path,
        "size": f"{image.width}x{image.height}",
        "note": "This image may contain anything that was on screen, and it is now on disk.",
    }


def get_system_status() -> dict:
    """Read-only. CPU, memory, disk, battery, uptime."""
    try:
        psutil = _get_psutil()
    except ImportError as e:
        return {"error": f"psutil is not installed: {e}. Run: pip install psutil"}

    try:
        memory = psutil.virtual_memory()
        status = {
            "cpu_percent": psutil.cpu_percent(interval=0.3),
            "memory_used_pct": memory.percent,
            "memory_used_gb": round(memory.used / (1024 ** 3), 1),
            "memory_total_gb": round(memory.total / (1024 ** 3), 1),
            "uptime_hours": round((time.time() - psutil.boot_time()) / 3600, 1),
        }
        battery = psutil.sensors_battery()
        if battery is not None:
            status["battery_percent"] = round(battery.percent)
            status["battery_plugged_in"] = battery.power_plugged
        return status
    except Exception as e:
        return {"error": f"Could not read system status: {e}"}


def lock_screen() -> dict:
    """Ungated -- see the module docstring for why this one power action
    is not gated when the others are."""
    if not IS_WINDOWS:
        return dict(NOT_WINDOWS)
    try:
        if not ctypes.windll.user32.LockWorkStation():
            return {"error": "Windows refused to lock the workstation."}
    except Exception as e:
        return {"error": f"Could not lock: {e}"}
    return {"status": "locked"}


def power_action(action: str) -> dict:
    """
    GATED. Proposes sleeping, shutting down, or restarting.

    The most disruptive thing in this entire tool surface: it ends every
    running program at once, unsaved work included, and on shutdown or
    restart it also ends the Jarvis process delivering the answer. Gated
    for the obvious reason, and worth noting that the gate is the only
    thing standing between a misheard sentence and the machine turning
    itself off.

    Commands are invoked as argument lists, never through a shell, so
    nothing here can be turned into command injection by a creative
    argument. `action` is checked against a closed set first regardless.
    """
    if not IS_WINDOWS:
        return dict(NOT_WINDOWS)

    commands = {
        "sleep": ["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"],
        "shutdown": ["shutdown", "/s", "/t", "0"],
        "restart": ["shutdown", "/r", "/t", "0"],
    }
    key = action.strip().lower()
    if key not in commands:
        return {"error": f"Unknown action '{action}'. Valid: sleep, shutdown, restart. (To lock, use lock_screen.)"}

    def _execute():
        try:
            subprocess.Popen(commands[key], shell=False)
        except OSError as e:
            return {"error": f"Could not {key}: {e}"}
        return {"status": f"{key} started"}

    return gate.propose(
        action=f"power_{key}",
        message=(
            f"About to {key.upper()} this machine. Every running program will be "
            "closed and any unsaved work will be lost."
            + (" Jarvis itself will stop." if key != "sleep" else "")
        ),
        execute=_execute,
        power_action=key,
    )


TOOL_FUNCTIONS = {
    "list_windows": list_windows,
    "focus_window": focus_window,
    "move_window": move_window,
    "close_window": close_window,
    "list_processes": list_processes,
    "kill_process": kill_process,
    "get_clipboard": get_clipboard,
    "set_clipboard": set_clipboard,
    "media_control": media_control,
    "take_screenshot": take_screenshot,
    "get_system_status": get_system_status,
    "lock_screen": lock_screen,
    "power_action": power_action,
    # gate.confirm_pending_action / gate.cancel_pending_action are
    # deliberately absent, exactly as Phase 6 keeps its own out. The
    # model must never have a tool-calling path to either.
}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_windows",
            "description": (
                "List the visible windows currently open, with their positions and sizes. "
                "Use this to find out what is actually open before trying to focus, move, "
                "or close something -- it is far more reliable than guessing from a screen "
                "reading."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title_contains": {
                        "type": "string",
                        "description": "Optional case-insensitive filter on the window title.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "focus_window",
            "description": (
                "Bring a window to the foreground and give it keyboard focus, by a substring "
                "of its title. Must match exactly one window. Use this before sending a "
                "hotkey or click to a specific app. Check the result: Windows sometimes "
                "refuses a focus change, and the tool reports that honestly rather than "
                "claiming success."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title_contains": {
                        "type": "string",
                        "description": "Case-insensitive substring of the target window's title.",
                    }
                },
                "required": ["title_contains"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_window",
            "description": (
                "Move and/or resize a window. Omitted values leave that dimension unchanged. "
                "Call list_windows first to see current positions and the screen layout."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title_contains": {"type": "string", "description": "Substring of the window title."},
                    "x": {"type": "integer", "description": "New left edge in screen pixels."},
                    "y": {"type": "integer", "description": "New top edge in screen pixels."},
                    "width": {"type": "integer", "description": "New width in pixels."},
                    "height": {"type": "integer", "description": "New height in pixels."},
                },
                "required": ["title_contains"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "close_window",
            "description": (
                "Close a window. Requires separate human confirmation you cannot see or "
                "influence -- treat a 'closed' or 'cancelled' outcome as normal, not an "
                "error. Must match exactly one window."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title_contains": {"type": "string", "description": "Substring of the window title."}
                },
                "required": ["title_contains"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_processes",
            "description": (
                "List running processes, largest memory use first. Use this for questions "
                "about what is running or what is slowing the machine down."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name_contains": {"type": "string", "description": "Optional case-insensitive name filter."},
                    "limit": {"type": "integer", "description": "Max processes to return (default 60)."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kill_process",
            "description": (
                "Terminate a running process, by pid or by a name matching exactly one "
                "process. Requires separate human confirmation. Protected system processes "
                "are refused. Prefer close_window for ordinary applications -- it lets them "
                "shut down cleanly and prompt about unsaved work."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pid": {"type": "integer", "description": "Process id. Preferred -- unambiguous."},
                    "name": {"type": "string", "description": "Process name, if the pid isn't known."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_clipboard",
            "description": "Read the current text contents of the system clipboard.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_chars": {"type": "integer", "description": "Truncation limit (default 20000)."}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_clipboard",
            "description": (
                "Put text on the system clipboard, replacing what was there. Useful for "
                "handing the user something long that would be tedious to hear read aloud."
            ),
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Text to copy."}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "media_control",
            "description": (
                "Control media playback and system volume: play_pause, next_track, "
                "previous_track, volume_up, volume_down, mute. Volume steps are small -- "
                "call it repeatedly for a bigger change."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": sorted(MEDIA_KEYS),
                        "description": "Which media key to send.",
                    }
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "take_screenshot",
            "description": (
                "Save a screenshot to an image file. This writes to disk and persists -- to "
                "simply READ what is on screen, use read_screen instead, which never saves "
                "anything. Refuses to overwrite an existing file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Where to save the .png file."},
                    "all_screens": {
                        "type": "boolean",
                        "description": "Capture all monitors instead of just the primary one.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_status",
            "description": (
                "Current CPU load, memory use, uptime, and battery level. Use this for "
                "'how is my PC doing' style questions."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lock_screen",
            "description": (
                "Lock the workstation immediately. Does not require confirmation -- it "
                "destroys nothing and everything is exactly as left after logging back in."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "power_action",
            "description": (
                "Sleep, shut down, or restart the machine. Requires separate human "
                "confirmation. Shutdown and restart also stop Jarvis itself. To merely "
                "lock the screen, use lock_screen instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["sleep", "shutdown", "restart"],
                        "description": "Which power action to take.",
                    }
                },
                "required": ["action"],
            },
        },
    },
]
