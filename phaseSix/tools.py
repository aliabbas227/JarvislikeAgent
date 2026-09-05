"""
tools.py — Phase 6: OS-control tools.

Adds real OS interaction (launch apps, browse the filesystem, read files,
delete files) behind the same tool-schema + execute_tool() dispatch
pattern Phase 4 established in phaseFour/tools.py. Reused deliberately,
not automatically — see PHASE6_HANDOFF.md for why.

Guardrail, built in from day one per the roadmap: delete_file() is a
two-phase action. Calling it through execute_tool() (the path the LLM
can reach) only ever *proposes* a deletion and returns a confirmation
token — it never deletes anything itself. The actual deletion only
happens via confirm_pending_action(), which is deliberately NOT in
TOOL_FUNCTIONS/TOOL_SCHEMAS and is only ever called from os_agent.py's
CLI loop, after a real human types "yes" at a real terminal prompt. This
means a prompt-injected instruction (e.g. from a poisoned Phase-5-style
memory, if a future phase ever wires the two together) cannot delete a
file on its own — it can, at most, get as far as a *proposal* that a
human has to explicitly approve out-of-band from the model. See Phase 5
handoff bug #4 for the concrete threat this defends against.
"""

import ctypes
import os
import platform
import subprocess
import time
import uuid


# ---------------------------------------------------------------------------
# Pending destructive-action registry (in-memory, per-process)
# ---------------------------------------------------------------------------

PENDING_TTL_SECONDS = 300  # confirmation tokens expire after 5 minutes

# token -> {"action": "delete"|"overwrite", "path": str, "content": str|None, "created_at": float}
_pending_actions = {}


def _prune_expired():
    now = time.time()
    expired = [t for t, a in _pending_actions.items() if now - a["created_at"] > PENDING_TTL_SECONDS]
    for t in expired:
        del _pending_actions[t]


# ---------------------------------------------------------------------------
# Read-only tools
# ---------------------------------------------------------------------------

def list_directory(path: str = ".") -> dict:
    """Lists files and subdirectories at `path`. Read-only — no guardrail
    needed, since listing can't destroy anything."""
    try:
        if not os.path.isdir(path):
            return {"error": f"Not a directory: {path}"}
        entries = []
        with os.scandir(path) as it:
            for entry in it:
                try:
                    entries.append({
                        "name": entry.name,
                        "type": "directory" if entry.is_dir() else "file",
                        "size_bytes": entry.stat().st_size if entry.is_file() else None,
                    })
                except OSError:
                    # A single unreadable entry (permissions, broken
                    # symlink, etc.) shouldn't blow up the whole listing.
                    entries.append({"name": entry.name, "type": "unknown", "size_bytes": None})
        entries.sort(key=lambda e: (e["type"] != "directory", e["name"].lower()))
        return {"path": os.path.abspath(path), "entries": entries}
    except OSError as e:
        return {"error": f"Could not list directory: {e}"}


def read_file(path: str, max_chars: int = 20000) -> dict:
    """Reads a text file, truncated to max_chars. Read-only. Refuses
    obviously-binary files (best-effort, via a null-byte sniff) rather
    than dumping garbage into the model's context."""
    try:
        if not os.path.isfile(path):
            return {"error": f"Not a file: {path}"}
        with open(path, "rb") as f:
            raw = f.read(max_chars + 1)
        if b"\x00" in raw:
            return {"error": f"'{path}' looks like a binary file — refusing to read as text."}
        text = raw.decode("utf-8", errors="replace")
        truncated = len(raw) > max_chars
        if truncated:
            text = text[:max_chars]
        return {"path": os.path.abspath(path), "content": text, "truncated": truncated}
    except OSError as e:
        return {"error": f"Could not read file: {e}"}


# ---------------------------------------------------------------------------
# Screen reading (OCR)
# ---------------------------------------------------------------------------

def _get_tesseract():
    """Lazily imports pytesseract (same pattern as Phase 2's lazy
    whisper/pyttsx3 imports and Phase 5's lazy chromadb import) — keeps
    `import tools` cheap and lets the rest of this module be used/tested
    without pytesseract or the Tesseract binary installed. Honors
    TESSERACT_CMD if the user set it, since (like Phase 2's ffmpeg PATH
    gotcha) the Tesseract *binary* is a separate install from the
    `pytesseract` pip package and Windows won't always find it on PATH
    automatically."""
    import pytesseract

    cmd = os.getenv("TESSERACT_CMD")
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    return pytesseract


def read_screen(region: list = None, all_screens: bool = False) -> dict:
    """Captures a screenshot and OCRs it to text. Read-only in the sense
    that nothing on the system is changed -- but unlike read_file, this
    can pick up *anything* currently visible on screen (a password
    field, a private chat, someone else's document), so the captured
    image is never written to disk and is discarded the moment OCR is
    done, kept in memory only for the duration of this one call.

    `region`, if given, is [left, top, width, height] in screen pixels
    -- pass one to OCR a specific window/area instead of the whole
    screen, which both narrows what gets captured and speeds up OCR on
    a large multi-monitor setup.

    `all_screens`, if True, captures the full virtual desktop spanning
    every connected monitor instead of just the primary one -- Windows
    is treated as the reference platform here (per the project's
    hardware), and this is a Windows-only Pillow feature requiring
    Pillow 9.3+. Without it, a second monitor is invisible to this
    tool entirely: PIL's default grab() only sees the primary display.
    Coordinates on a secondary monitor positioned to the left of or
    above the primary one are negative in the combined virtual-desktop
    coordinate space that all_screens=True uses -- pass all_screens=True
    whenever using `region` to target anything outside the primary
    monitor's own bounds, not just when you want the whole combined
    image."""
    try:
        from PIL import ImageGrab
    except ImportError as e:
        return {"error": f"Pillow is not installed: {e}. Run: pip install Pillow"}

    bbox = None
    if region is not None:
        if len(region) != 4:
            return {"error": "region must be [left, top, width, height]."}
        left, top, width, height = region
        bbox = (left, top, left + width, top + height)

    try:
        image = ImageGrab.grab(bbox=bbox, all_screens=all_screens)
    except TypeError:
        # Older Pillow (<9.3) doesn't have the all_screens parameter at
        # all -- fail with a clear, actionable message rather than a
        # confusing TypeError, but only if the caller actually asked
        # for it; a plain single-monitor grab still works fine on old
        # Pillow versions.
        if all_screens:
            return {
                "error": (
                    "This Pillow version doesn't support all_screens=True "
                    "(needs Pillow 9.3+). Run: pip install --upgrade Pillow"
                )
            }
        try:
            image = ImageGrab.grab(bbox=bbox)
        except Exception as e:
            return {"error": f"Could not capture screen: {e}"}
    except Exception as e:
        return {"error": f"Could not capture screen: {e}"}

    try:
        pytesseract = _get_tesseract()
    except ImportError as e:
        return {"error": f"pytesseract is not installed: {e}. Run: pip install pytesseract"}

    try:
        text = pytesseract.image_to_string(image)
    except Exception as e:
        return {
            "error": (
                f"OCR failed: {e}. Make sure the Tesseract OCR engine itself is "
                "installed (not just the pytesseract pip package) and on PATH, "
                "or set TESSERACT_CMD in .env to its full install path."
            )
        }

    text = text.strip()
    return {
        "text": text if text else "(no readable text detected)",
        "region": list(region) if region is not None else "full screen",
        "all_screens": all_screens,
        "char_count": len(text),
    }


# ---------------------------------------------------------------------------
# Application launching
# ---------------------------------------------------------------------------

# Apps that take a search/deep-link query rather than a plain launch.
# Carried over from Phase 4's open_spotify (built, mock-tested, then
# deliberately deferred to this phase per the roadmap — OS control,
# not tool-calling).
_URI_LAUNCHERS = {
    "spotify": lambda query: f"spotify:search:{query}" if query else "spotify:",
}


def open_application(app_name: str, query: str = "") -> dict:
    """Launches an application by name. Windows-first (os.startfile),
    since that's the current dev machine — falls back to `open` on
    macOS and `xdg-open` on Linux for portability, untested on those.

    Not sandboxed beyond what the OS itself does — this can launch
    anything launchable, which is why it's scoped to *opening* apps
    only, never passing arbitrary shell commands through."""
    app_key = app_name.strip().lower()
    system = platform.system()

    try:
        if app_key in _URI_LAUNCHERS:
            target = _URI_LAUNCHERS[app_key](query)
        else:
            target = app_name

        if system == "Windows":
            os.startfile(target)  # noqa: os.startfile is Windows-only
        elif system == "Darwin":
            subprocess.Popen(["open", target])
        else:
            subprocess.Popen(["xdg-open", target])

        return {"status": "launched", "app": app_name, "query": query or None}
    except OSError as e:
        return {"error": f"Could not launch '{app_name}': {e}"}


# ---------------------------------------------------------------------------
# write_file — safe when creating, confirmation-gated when overwriting
# ---------------------------------------------------------------------------

def write_file(path: str, content: str = "", overwrite: bool = False) -> dict:
    """Creates a new file directly (no guardrail needed — nothing existing
    is destroyed). If `path` already exists, this refuses unless
    `overwrite=True` is passed, and even then only *proposes* the
    overwrite — actually replacing existing content is exactly as
    destructive as delete_file, so it goes through the same two-phase
    confirmation gate rather than happening on this call."""
    abs_path = os.path.abspath(path)
    exists = os.path.isfile(abs_path)

    if exists and not overwrite:
        return {
            "error": (
                f"'{abs_path}' already exists. Pass overwrite=true to replace it "
                "(this will require human confirmation)."
            )
        }

    if not exists:
        try:
            parent = os.path.dirname(abs_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(content)
            return {"status": "created", "path": abs_path}
        except OSError as e:
            return {"error": f"Could not create file: {e}"}

    # exists and overwrite=True -> gated, same pattern as delete_file
    _prune_expired()
    token = uuid.uuid4().hex[:8]
    _pending_actions[token] = {
        "action": "overwrite",
        "path": abs_path,
        "content": content,
        "created_at": time.time(),
    }
    return {
        "status": "confirmation_required",
        "token": token,
        "path": abs_path,
        "message": f"About to overwrite the existing contents of '{abs_path}'. Requires human confirmation.",
    }


# ---------------------------------------------------------------------------
# Mouse / keyboard automation
# ---------------------------------------------------------------------------
#
# Guardrail decision, made explicitly rather than assumed (per Phase 5
# handoff's recommendation to decide this kind of thing up front): the
# user specified clicks require confirmation, typing/reading text is
# lower risk and doesn't. Keyboard *hotkeys* (Ctrl+W, Alt+F4, Win+L,
# Ctrl+Shift+Esc, etc.) are placed in the gated bucket alongside
# clicks, not the ungated "typing" bucket -- a shortcut combination can
# close an unsaved document, quit an application, or trigger a
# system-level action just as unilaterally as a click can. This is a
# judgment call beyond what was explicitly specified, not a literal
# reading of "typing is lower risk," and is documented here for that
# reason.
#
# pyautogui.FAILSAFE is forced on unconditionally (moving the mouse to
# a screen corner aborts whatever pyautogui is doing) -- there is no
# parameter anywhere in this module to turn it off. That's a real,
# free emergency-stop mechanism and nothing here should be able to
# disable it, including the model.

def _get_pyautogui():
    """Lazy import, same reasoning as _get_tesseract() -- keeps
    `import tools` cheap and this module testable without pyautogui or
    a real display/window manager installed."""
    import pyautogui

    pyautogui.FAILSAFE = True
    return pyautogui


# Delay between each key-down/key-up in _send_hotkey(). Confirmed by
# manual testing: pyautogui.hotkey()'s default near-instantaneous
# timing works fine for app-level shortcuts (Ctrl+A, Delete both fired
# correctly), but silently fails for Alt+F4 specifically -- the
# keystroke reports success with no error, but Windows never registers
# it as the system-level "close window" command. System shortcuts like
# Alt+F4 need Windows to see the modifier key genuinely held as the
# second key fires, which needs more time between events than
# hotkey()'s default provides. Configurable in case 50ms isn't enough
# on a slower/more loaded machine.
HOTKEY_KEY_DELAY_SECONDS = float(os.getenv("HOTKEY_KEY_DELAY_SECONDS", "0.05"))


def _send_hotkey(pyautogui_module, keys: list) -> None:
    """Sends a key combination with explicit delays between each
    individual key-down and key-up, instead of pyautogui.hotkey()'s
    default near-simultaneous press-and-release. See
    HOTKEY_KEY_DELAY_SECONDS above for why this exists -- this is not
    a hypothetical fix, it's the specific change that made Alt+F4 start
    working after Ctrl+A/Delete were already confirmed to work with
    default timing."""
    for key in keys:
        pyautogui_module.keyDown(key)
        time.sleep(HOTKEY_KEY_DELAY_SECONDS)
    for key in reversed(keys):
        pyautogui_module.keyUp(key)
        time.sleep(HOTKEY_KEY_DELAY_SECONDS)


def move_mouse(x: int, y: int) -> dict:
    """Moves the mouse cursor to (x, y). No guardrail -- moving the
    cursor alone can't cause anything to happen (no click involved), so
    this executes immediately rather than proposing first.

    Coordinates are in the same virtual-desktop space as read_screen's
    `region` -- Windows reports mouse position across the full
    multi-monitor desktop by default (unlike ImageGrab's screenshot
    capture, which needed all_screens=True to see past the primary
    monitor). No equivalent flag is needed here: a negative x or y
    targeting a monitor positioned left of/above the primary one just
    works. See click()'s docstring for a safety caveat that does NOT
    extend across monitors the same way."""
    try:
        pyautogui = _get_pyautogui()
    except ImportError as e:
        return {"error": f"pyautogui is not installed: {e}. Run: pip install pyautogui"}

    try:
        pyautogui.moveTo(x, y)
        return {"status": "moved", "x": x, "y": y}
    except Exception as e:
        return {"error": f"Could not move mouse: {e}"}


def type_text(text: str, interval: float = 0.0) -> dict:
    """Types literal text via the keyboard at whatever currently has
    focus. No guardrail, per the explicit decision above -- but it's
    worth being deliberate about window focus before calling this,
    since it types wherever the cursor already is, not somewhere it
    verifies first."""
    if not text:
        return {"error": "text must be non-empty."}

    try:
        pyautogui = _get_pyautogui()
    except ImportError as e:
        return {"error": f"pyautogui is not installed: {e}. Run: pip install pyautogui"}

    try:
        pyautogui.write(text, interval=interval)
        return {"status": "typed", "char_count": len(text)}
    except Exception as e:
        return {"error": f"Could not type text: {e}"}


def _get_active_window_title() -> str:
    """Returns the title of whatever window currently has OS keyboard
    focus. Windows-only, using ctypes' user32 directly (no extra
    dependency beyond stdlib). This exists specifically so a click/
    hotkey confirmation can show the human *what's actually about to
    receive it* -- not just the coordinates or keys -- since "whatever
    currently has focus" is not always the window a human (or the
    model) assumes it is. Windows' foreground-lock behavior can leave
    a background-launched app (e.g. Notepad via open_application, if
    the process running this is itself in a terminal embedded in
    another application, like VS Code's integrated terminal) without
    real OS focus even though it's visibly on screen -- this surfaces
    that mismatch before an action fires, rather than after something
    unintended gets closed.

    Never raises -- returns a placeholder string on any failure, since
    this is a safety nicety for the confirmation message, not
    something that should ever block the action itself."""
    if platform.system() != "Windows":
        return "(unknown — active window detection is Windows-only)"
    try:
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value or "(no window title)"
    except Exception:
        return "(could not determine active window)"


def get_active_window() -> dict:
    """Read-only, ungated -- lets the model check what currently has
    focus *before* deciding to send a click or hotkey, as a first line
    of defense that the confirmation-message check (in click() and
    press_hotkey() below) backs up."""
    return {"title": _get_active_window_title()}


def _get_active_window_handle():
    """Returns the window handle (HWND) of whatever currently has OS
    focus, or None on non-Windows/any failure. Companion to
    _get_active_window_title() -- most callers only need the title;
    a handle is only useful for the refocus attempt in
    _try_refocus_window() below, so this is kept separate rather than
    always fetched."""
    if platform.system() != "Windows":
        return None
    try:
        return ctypes.windll.user32.GetForegroundWindow()
    except Exception:
        return None


def _try_refocus_window(hwnd, expected_title: str) -> bool:
    """Best-effort attempt to bring a specific window back to the
    foreground right before a click/hotkey fires, so a confirmed action
    can still reach the window that was actually proposed even though
    confirming at the terminal shifted focus away from it in between.

    Windows restricts which processes may successfully call
    SetForegroundWindow (its "foreground lock"), so this is NOT
    guaranteed to work from a background process. It has a real chance
    of working here specifically because this process just finished
    handling genuine keyboard input (the human typing the confirmation
    password) -- one of the conditions Windows honors -- but that's not
    a guarantee, and no attempt is made to work around the restriction
    if it fails.

    Returns True only if the refocus is *verified* to have actually
    happened -- by re-reading the foreground window afterward -- not
    just because SetForegroundWindow reported success, since Windows
    can return success while silently declining to actually switch
    focus (it flashes the taskbar icon instead)."""
    if hwnd is None:
        return False
    try:
        ctypes.windll.user32.SetForegroundWindow(hwnd)
        time.sleep(0.15)  # give the window manager a moment to actually switch
        return _get_active_window_title() == expected_title
    except Exception:
        return False


def click(x: int, y: int, target_window_contains: str, button: str = "left", clicks: int = 1) -> dict:
    """LLM-reachable entry point. Does NOT click anything. Proposes a
    click at (x, y) and registers a pending confirmation token, exactly
    like delete_file -- a click can do anything a human clicking there
    could do (approve a dialog, launch a program, submit a form), so it
    goes through the same two-phase gate.

    target_window_contains is REQUIRED: a substring (case-insensitive)
    that must appear in the currently-focused window's title. This is
    not optional politeness -- it is the actual safety anchor, added
    after a real incident. Earlier versions derived "the expected
    window" implicitly from whatever had focus the instant this
    function was called -- but in a multi-turn conversation, that
    instant is not necessarily when the human/model actually decided
    what to target; it's just whenever the tool call happened to fire.
    Confirmed case: the model announced an action in one turn without
    calling the tool (a known DeepSeek reliability gap), the human
    replied "go ahead" to prompt it, and typing that reply handed OS
    focus back to the terminal *before* the model's next turn actually
    called press_hotkey(). The old implicit baseline captured "the
    terminal has focus" as if it were the intended target -- there was
    no drift to detect, because the bad baseline was recorded *after*
    the drift already happened -- and Alt+F4 fired directly at the
    terminal's own window (closing it). Requiring an explicit declared
    target, checked against reality both now and again right before
    firing, closes that gap: a stale or simply wrong assumption about
    what's focused gets caught immediately instead of quietly becoming
    the new trusted baseline.

    SAFETY NOTE, multi-monitor specifically: (x, y) work across both
    monitors (see move_mouse's docstring), but pyautogui's FAILSAFE
    panic-corner check does NOT -- it's anchored to the PRIMARY
    monitor's four corners only (based on pyautogui.size()). If a
    click sequence is misbehaving on a SECONDARY monitor, dragging the
    mouse to that monitor's own corner will not trigger the abort --
    you have to drag all the way to the primary monitor's corner
    instead. Worth knowing before treating the failsafe as a reliable
    stop while working on a non-primary display."""
    if not target_window_contains or not target_window_contains.strip():
        return {
            "error": (
                "target_window_contains is required — state which window you "
                "believe should currently be focused (e.g. 'Notepad', 'Task "
                "Manager'). Call get_active_window first if you're not sure."
            )
        }
    if button not in ("left", "right", "middle"):
        return {"error": f"Unsupported button: {button!r}. Use 'left', 'right', or 'middle'."}
    if clicks < 1:
        return {"error": "clicks must be >= 1."}

    active_window = _get_active_window_title()
    active_hwnd = _get_active_window_handle()

    if platform.system() == "Windows" and target_window_contains.lower() not in active_window.lower():
        return {
            "error": (
                f"You said you're targeting a window containing '{target_window_contains}', "
                f"but the currently focused window is '{active_window}'. Not proposing "
                "this click — bring the intended window to focus first, or re-check "
                "with get_active_window."
            )
        }

    # Fail fast if the coordinates are already, obviously outside the
    # currently-focused window's bounds -- no point making a human type
    # a password for a click that's already known to be wrong. See
    # _abort_if_click_out_of_bounds()'s docstring for why keyboard focus
    # alone doesn't guarantee a click lands where confirmed; this is the
    # same check, just run early rather than only at execution time.
    rect = _get_window_rect(active_hwnd)
    if rect is not None:
        left, top, right, bottom = rect
        if not (left <= x <= right and top <= y <= bottom):
            return {
                "error": (
                    f"({x}, {y}) is outside the currently-focused window's bounds "
                    f"(({left}, {top}) to ({right}, {bottom})). Not proposing this "
                    "click -- it would land on a different window than the one "
                    "you'd be confirming."
                )
            }

    _prune_expired()
    token = uuid.uuid4().hex[:8]
    _pending_actions[token] = {
        "action": "click",
        "x": x,
        "y": y,
        "button": button,
        "clicks": clicks,
        "expected_window": active_window,
        "expected_hwnd": active_hwnd,
        "target_window_contains": target_window_contains,
        "created_at": time.time(),
    }
    return {
        "status": "confirmation_required",
        "token": token,
        "active_window": active_window,
        "message": (
            f"About to {clicks}x {button}-click at ({x}, {y}), targeting a window "
            f"containing '{target_window_contains}' (currently focused: "
            f"'{active_window}'). Requires human confirmation."
        ),
    }


def press_hotkey(keys: list, target_window_contains: str) -> dict:
    """LLM-reachable entry point. Does NOT press anything. Proposes a
    keyboard shortcut combination (e.g. ["ctrl", "s"]) and registers a
    pending confirmation token -- gated the same as click(), see the
    module-level note above for why.

    target_window_contains is REQUIRED, same reasoning and same real
    incident as click()'s docstring -- see there for the full story.
    Short version: the window that has focus when this function is
    *called* is not reliably the window the caller actually means to
    act on, especially across multiple conversation turns. An explicit
    declared target, checked against reality now and again right
    before firing, is what actually closes that gap -- not just naming
    whatever's currently focused in the confirmation message."""
    if not keys:
        return {"error": "keys must be a non-empty list, e.g. ['ctrl', 's']."}
    if not target_window_contains or not target_window_contains.strip():
        return {
            "error": (
                "target_window_contains is required — state which window you "
                "believe should currently be focused (e.g. 'Notepad', 'Task "
                "Manager'). Call get_active_window first if you're not sure."
            )
        }

    active_window = _get_active_window_title()
    active_hwnd = _get_active_window_handle()

    if platform.system() == "Windows" and target_window_contains.lower() not in active_window.lower():
        return {
            "error": (
                f"You said you're targeting a window containing '{target_window_contains}', "
                f"but the currently focused window is '{active_window}'. Not proposing "
                "this hotkey — bring the intended window to focus first, or re-check "
                "with get_active_window."
            )
        }

    _prune_expired()
    token = uuid.uuid4().hex[:8]
    _pending_actions[token] = {
        "action": "hotkey",
        "keys": list(keys),
        "expected_window": active_window,
        "expected_hwnd": active_hwnd,
        "target_window_contains": target_window_contains,
        "created_at": time.time(),
    }
    combo = "+".join(keys)
    return {
        "status": "confirmation_required",
        "token": token,
        "active_window": active_window,
        "message": (
            f"About to press hotkey: {combo}, targeting a window containing "
            f"'{target_window_contains}' (currently focused: '{active_window}'). "
            "Requires human confirmation."
        ),
    }


# ---------------------------------------------------------------------------
# Destructive tools: delete_file / write_file(overwrite=True) / click() /
# press_hotkey() — two-phase, confirmation-gated
# ---------------------------------------------------------------------------

def delete_file(path: str) -> dict:
    """LLM-reachable entry point. Does NOT delete anything. Validates
    the path and registers a pending confirmation token — the model
    reports this back to the user as "needs confirmation." The actual
    deletion only happens through confirm_pending_action(), which the
    model has no way to call itself (see module docstring)."""
    _prune_expired()

    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        return {"error": f"Not a file (or doesn't exist): {path}"}

    token = uuid.uuid4().hex[:8]
    _pending_actions[token] = {"action": "delete", "path": abs_path, "created_at": time.time()}

    return {
        "status": "confirmation_required",
        "token": token,
        "path": abs_path,
        "message": f"About to permanently delete '{abs_path}'. Requires human confirmation.",
    }


class FocusShiftedError(Exception):
    """Raised when the window that had focus at proposal time no longer
    matches what has focus right before a click/hotkey actually fires.
    See _abort_if_focus_shifted()'s docstring for why this matters."""


class ClickOutOfBoundsError(Exception):
    """Raised when a proposed click's (x, y) falls outside the bounds of
    the window that's confirmed to be focused. See
    _abort_if_click_out_of_bounds()'s docstring for why this check is
    necessary even after the focus check above has already passed."""


def _abort_if_focus_shifted(action: dict) -> None:
    """Re-checks OS focus immediately before a click/hotkey actually
    executes. Validates against target_window_contains (the caller's
    EXPLICITLY DECLARED intent) when present, not just exact equality
    against whatever was incidentally focused at proposal time -- see
    click()'s docstring for the real incident that makes this
    distinction matter: an implicit "whatever had focus a moment ago"
    baseline can itself already be wrong, especially across
    conversation turns. Falls back to exact-match against
    expected_window for older/other action types that don't carry a
    declared target.

    If the currently-focused window doesn't satisfy the target, attempts
    to win focus back via _try_refocus_window() -- then INDEPENDENTLY
    re-checks against the same target afterward, rather than trusting
    _try_refocus_window's own internal verification, since that verifies
    against the exact original title, and what actually matters here is
    whether the declared intent is satisfied."""
    expected = action.get("expected_window")
    if expected is None:
        return  # no baseline was recorded (e.g. non-Windows) -- nothing to compare
    if platform.system() != "Windows":
        return  # nothing genuinely verifiable off Windows -- don't block on it

    target_contains = (action.get("target_window_contains") or "").lower()

    def _satisfies_target(title: str) -> bool:
        if target_contains:
            return target_contains in title.lower()
        return title == expected

    current = _get_active_window_title()
    if _satisfies_target(current):
        return

    _try_refocus_window(action.get("expected_hwnd"), expected)
    current = _get_active_window_title()
    if _satisfies_target(current):
        return

    target_desc = f"a window containing '{action['target_window_contains']}'" if target_contains else f"'{expected}'"
    raise FocusShiftedError(
        f"Refusing to proceed: expected {target_desc} to be focused, but "
        f"'{current}' has focus now, and an attempt to refocus did not "
        "succeed. Nothing was sent. Bring the intended window to the front "
        "yourself and ask again."
    )


class _RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


def _get_window_rect(hwnd):
    """Returns (left, top, right, bottom) screen coordinates of hwnd's
    window, or None on non-Windows/any failure. Companion check to the
    focus check above -- confirmed by manual testing to matter: a
    proposed click can be confirmed against "the right window has
    focus" and still land somewhere else entirely, because a mouse
    click goes to whatever window physically occupies those screen
    coordinates, NOT to whatever currently has keyboard focus. Those
    are independent facts about the OS, and only checking one of them
    (as _abort_if_focus_shifted alone did) leaves exactly this gap."""
    if hwnd is None or platform.system() != "Windows":
        return None
    try:
        rect = _RECT()
        rect_ptr = ctypes.pointer(rect)
        if not ctypes.windll.user32.GetWindowRect(hwnd, rect_ptr):
            return None
        return (rect.left, rect.top, rect.right, rect.bottom)
    except Exception:
        return None


def _abort_if_click_out_of_bounds(action: dict) -> None:
    """Verifies a proposed click's (x, y) actually falls within the
    bounds of whatever window is currently focused (which
    _abort_if_focus_shifted has already confirmed matches what was
    proposed). Confirming the right window has focus is not the same
    as confirming a click at arbitrary coordinates will land inside
    it -- clicks target screen position, not focus. If the window rect
    can't be determined (e.g. non-Windows, or the OS call fails), this
    does NOT block the click -- an unverifiable check should never be
    treated as a failed one, only a genuinely out-of-bounds coordinate
    should stop the action."""
    hwnd = _get_active_window_handle()
    rect = _get_window_rect(hwnd)
    if rect is None:
        return  # can't verify -- don't block on something we can't check

    left, top, right, bottom = rect
    x, y = action["x"], action["y"]
    if not (left <= x <= right and top <= y <= bottom):
        raise ClickOutOfBoundsError(
            f"Refusing to proceed: the target ({x}, {y}) is outside the focused "
            f"window's bounds (({left}, {top}) to ({right}, {bottom})). This click "
            "would land on whatever window actually occupies that screen position, "
            "not the one that was confirmed. Nothing was sent."
        )


def confirm_pending_action(token: str) -> dict:
    """Executes a previously-registered destructive action (delete,
    overwrite, click, or hotkey). NEVER call this from inside the LLM
    tool-calling loop — it must only be reached after a real human
    confirms at the terminal (see os_agent.py). Not part of
    TOOL_FUNCTIONS/TOOL_SCHEMAS on purpose.

    Deliberately does NOT call _prune_expired() first — pruning would
    silently erase the difference between "this token never existed /
    was already used" and "this token existed but you took too long,"
    and those need distinct messages. A human who waited past the TTL
    and got a generic "invalid token" has no way to tell that from a
    typo or a stale token from an old file; a human told specifically
    that it expired knows exactly what to do (ask again)."""
    action = _pending_actions.get(token)
    if action is None:
        return {"error": "That confirmation token is invalid or was already used. Nothing was changed."}

    age_seconds = time.time() - action["created_at"]
    if age_seconds > PENDING_TTL_SECONDS:
        del _pending_actions[token]
        return {
            "error": (
                f"That confirmation expired after sitting unanswered for "
                f"{PENDING_TTL_SECONDS} seconds. Nothing was changed. "
                "Ask again to get a fresh confirmation."
            )
        }

    del _pending_actions[token]
    try:
        if action["action"] == "delete":
            os.remove(action["path"])
            return {"status": "deleted", "path": action["path"]}
        elif action["action"] == "overwrite":
            with open(action["path"], "w", encoding="utf-8") as f:
                f.write(action["content"])
            return {"status": "overwritten", "path": action["path"]}
        elif action["action"] == "click":
            _abort_if_focus_shifted(action)
            _abort_if_click_out_of_bounds(action)
            pyautogui = _get_pyautogui()
            pyautogui.click(x=action["x"], y=action["y"], clicks=action["clicks"], button=action["button"])
            return {
                "status": "clicked",
                "x": action["x"],
                "y": action["y"],
                "button": action["button"],
                "clicks": action["clicks"],
            }
        elif action["action"] == "hotkey":
            _abort_if_focus_shifted(action)
            pyautogui = _get_pyautogui()
            _send_hotkey(pyautogui, action["keys"])
            return {"status": "pressed", "keys": action["keys"]}
        else:
            return {"error": f"Unknown pending action type: {action['action']!r}"}
    except ImportError as e:
        return {"error": f"pyautogui is not installed: {e}. Run: pip install pyautogui"}
    except OSError as e:
        return {"error": f"Action failed: {e}"}
    except Exception as e:
        # pyautogui actions (click/hotkey) can raise things other than
        # OSError -- e.g. pyautogui.FailSafeException if the mouse was
        # moved to a screen corner to abort. Surface it as an error
        # rather than letting it crash the whole confirmation flow.
        return {"error": f"Action failed: {e}"}


def cancel_pending_action(token: str) -> dict:
    """Discards a pending confirmation without executing it (human said no)."""
    _pending_actions.pop(token, None)
    return {"status": "cancelled"}


# ---------------------------------------------------------------------------
# Tool schemas + dispatch (LLM-facing surface)
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS = {
    "list_directory": list_directory,
    "read_file": read_file,
    "read_screen": read_screen,
    "write_file": write_file,
    "open_application": open_application,
    "delete_file": delete_file,
    "move_mouse": move_mouse,
    "type_text": type_text,
    "click": click,
    "press_hotkey": press_hotkey,
    "get_active_window": get_active_window,
    # confirm_pending_action / cancel_pending_action deliberately absent —
    # the model must never be able to call these directly.
}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and subdirectories at a given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default: current directory)."}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the text contents of a file, truncated to a max length.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file."},
                    "max_chars": {"type": "integer", "description": "Max characters to return (default 5000)."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_screen",
            "description": (
                "Take a screenshot and OCR it to text. Captures the primary monitor by "
                "default (or a specific region), or the full multi-monitor virtual "
                "desktop if all_screens is true. Nothing is saved to disk."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Optional [left, top, width, height] in screen pixels to capture a specific area instead of the whole screen.",
                    },
                    "all_screens": {
                        "type": "boolean",
                        "description": "Set true to capture across all connected monitors instead of just the primary one.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create a new file with the given text content. If the file already "
                "exists, this fails unless overwrite=true is passed, and even then only "
                "proposes the overwrite — it requires separate human confirmation before "
                "any existing content is actually replaced."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to create."},
                    "content": {"type": "string", "description": "Text content to write (default: empty file)."},
                    "overwrite": {"type": "boolean", "description": "Set true to replace an existing file (requires confirmation)."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_application",
            "description": "Launch an application by name (e.g. 'notepad', 'spotify').",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {"type": "string", "description": "Application name."},
                    "query": {"type": "string", "description": "Optional search query, for apps that support deep-link search (e.g. Spotify)."},
                },
                "required": ["app_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": (
                "Propose deleting a file. This does NOT delete anything — it registers a "
                "pending action that requires explicit human confirmation outside the "
                "model's control before anything is actually removed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to delete."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_mouse",
            "description": "Move the mouse cursor to a specific screen position, without clicking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "X coordinate in screen pixels."},
                    "y": {"type": "integer", "description": "Y coordinate in screen pixels."},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": "Type literal text via the keyboard at whatever currently has focus.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to type."},
                    "interval": {"type": "number", "description": "Optional seconds to pause between keystrokes (default 0)."},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": (
                "Propose a mouse click at a screen position. This does NOT click anything — "
                "it registers a pending action that requires explicit human confirmation "
                "outside the model's control before the click actually happens."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "X coordinate in screen pixels."},
                    "y": {"type": "integer", "description": "Y coordinate in screen pixels."},
                    "target_window_contains": {
                        "type": "string",
                        "description": (
                            "REQUIRED. A substring you believe should currently appear in the "
                            "focused window's title (e.g. 'Notepad'). Checked against reality "
                            "before proposing and again right before the click fires — call "
                            "get_active_window first if you're not certain what's focused."
                        ),
                    },
                    "button": {"type": "string", "description": "'left', 'right', or 'middle' (default 'left')."},
                    "clicks": {"type": "integer", "description": "Number of clicks, e.g. 2 for a double-click (default 1)."},
                },
                "required": ["x", "y", "target_window_contains"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "press_hotkey",
            "description": (
                "Propose a keyboard shortcut combination, e.g. ['ctrl', 's']. This does NOT "
                "press anything — it registers a pending action that requires explicit human "
                "confirmation outside the model's control before the keys are actually pressed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Keys to press together, e.g. ['ctrl', 'alt', 'delete'] or ['alt', 'f4'].",
                    },
                    "target_window_contains": {
                        "type": "string",
                        "description": (
                            "REQUIRED. A substring you believe should currently appear in the "
                            "focused window's title (e.g. 'Task Manager'). Checked against reality "
                            "before proposing and again right before the keys fire — call "
                            "get_active_window first if you're not certain what's focused. This "
                            "matters especially after any back-and-forth with the human, since "
                            "typing a reply can change what has focus."
                        ),
                    },
                },
                "required": ["keys", "target_window_contains"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_active_window",
            "description": (
                "Check which window currently has keyboard focus. Read-only, no "
                "confirmation needed. Useful before proposing a click or hotkey, to "
                "confirm the intended window is actually focused first."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def execute_tool(name: str, arguments: dict) -> dict:
    if name not in TOOL_FUNCTIONS:
        raise KeyError(f"Unknown tool: {name}")
    return TOOL_FUNCTIONS[name](**arguments)