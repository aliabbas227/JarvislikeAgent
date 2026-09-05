"""
screen.py -- Phase 8, milestone 5: seeing the screen, and being able to
act on what is seen.

Phase 6 could take a screenshot and OCR it to a wall of text with no
positions attached. Its own system prompt admits what that costs:
"read_screen is OCR text extraction, NOT vision -- it cannot locate
icons, buttons, or other non-text UI elements, and does not return
coordinates for anything it reads. Do not try to find a button by
guessing screen regions." Which left move_mouse and click as tools the
model could technically call and never sensibly aim, because nothing
ever told it where anything was.

This module fixes the aiming problem and the seeing problem separately,
because they are separate problems and conflating them is why
screen-control agents tend not to work.

## Grounding: where things are

`list_ui_elements` and `click_element` read Windows UI Automation --
the same accessibility tree screen readers use. Every ordinary Windows
application publishes its buttons, menus, text fields and list items
there, each with a name, a control type, and an exact rectangle.

This is deliberately NOT a vision problem. A vision model asked "where
is the Save button" estimates coordinates from a downscaled image and
is routinely tens of pixels out, which on a large display is the
difference between a button and the thing beside it. UI Automation
answers the same question exactly, in a millisecond, with no model
involved. Clicking by NAME through the accessibility tree is what makes
mouse control real rather than a gimmick.

`read_screen_text` covers what UIA cannot: text drawn as pixels, in
apps that publish nothing. It is OCR again, but with bounding boxes
this time, so a word on screen becomes something clickable.

## Vision: what things are

`look_at_screen` sends the actual screenshot to a local vision model
through Ollama and asks it a question. This is the "see what I see"
half -- icons with no accessible name, images, games, video, a chart,
anything whose meaning is pictorial rather than textual.

It runs locally on purpose. A screenshot is the most sensitive thing
this assistant can capture, since it contains whatever happened to be
on screen; sending it to a hosted API would mean that leaving the
machine on every look. Phase 6 made the same call for its own
screenshot handling, keeping captures in memory and never writing them
to disk.

## The division of labour

    "what is on my screen?"        -> look_at_screen  (understanding)
    "where is the Save button?"    -> list_ui_elements (exact rectangle)
    "click Save"                   -> click_element    (name, not pixels)
    "what does that error say?"    -> read_screen_text (OCR + boxes)

Vision understands; grounding acts. Asking the vision model for
coordinates and clicking them is the failure mode this layout exists to
avoid.
"""

import base64
import io
import logging
import os
import platform
import time

import gate

logger = logging.getLogger("screen")

IS_WINDOWS = platform.system() == "Windows"

NOT_WINDOWS = {"error": "This tool is Windows-only, and this machine is not running Windows."}

# Control types worth showing the model. The accessibility tree contains
# a great deal of structural furniture (panes, groups, custom wrappers)
# that cannot be interacted with and would only crowd out the real
# targets.
INTERACTIVE_CONTROL_TYPES = {
    "ButtonControl", "MenuItemControl", "EditControl", "CheckBoxControl",
    "RadioButtonControl", "ComboBoxControl", "ListItemControl", "TabItemControl",
    "HyperlinkControl", "TreeItemControl", "SliderControl", "SplitButtonControl",
    "MenuControl", "ToolBarControl", "DocumentControl",
}

# How deep to walk the tree, and how long to spend doing it. Both matter:
# a deep application (a browser with a complex page) can expose thousands
# of nodes, and this runs mid-turn with someone waiting.
UI_TREE_MAX_DEPTH = int(os.getenv("UI_TREE_MAX_DEPTH", "12"))
UI_TREE_TIME_BUDGET_SECONDS = float(os.getenv("UI_TREE_TIME_BUDGET_SECONDS", "6"))

VISION_MODEL = os.getenv("VISION_MODEL", "qwen2.5vl:7b")


def _ollama_base_url() -> str:
    """Ollama's base URL, with any endpoint path stripped off.

    OLLAMA_URL is shared with Phase 1, which set it to a full endpoint
    ("http://localhost:11434/api/chat") rather than a host. Appending
    "/api/generate" to that produced ".../api/chat/api/generate" and a
    404. Reading only the scheme and host means either form works, and
    means this keeps working if Phase 1 ever changes which endpoint it
    points at.
    """
    from urllib.parse import urlparse

    raw = os.getenv("OLLAMA_VISION_URL") or os.getenv("OLLAMA_URL") or "http://localhost:11434"
    parsed = urlparse(raw if "//" in raw else f"http://{raw}")
    if not parsed.netloc:
        return "http://localhost:11434"
    return f"{parsed.scheme}://{parsed.netloc}"


OLLAMA_URL = _ollama_base_url()
VISION_TIMEOUT_SECONDS = float(os.getenv("VISION_TIMEOUT_SECONDS", "120"))

# How long Ollama keeps the vision model resident after a look.
#
# Measured on this machine (RTX 3060, qwen2.5vl:7b): a cold look costs
# ~41s because the model has to load into VRAM, a warm one 2.6-4.5s.
# Ollama's own default unloads after 5 minutes idle, so an assistant
# used intermittently pays the cold cost almost every time.
#
# The default is left at Ollama's, deliberately, rather than pinning the
# model in memory: it occupies roughly 6GB of a 12GB card, and this
# machine also plays games. Raise it (e.g. "30m") if you use screen
# vision often and would rather spend the VRAM than the wait.
VISION_KEEP_ALIVE = os.getenv("VISION_KEEP_ALIVE", "5m")

# Screenshots are downscaled before going to the vision model. A 4K
# capture is far more detail than a 7B model uses, and costs real time
# to encode and process.
VISION_MAX_WIDTH = int(os.getenv("VISION_MAX_WIDTH", "1280"))


def _get_uia():
    """Lazy import, same pattern as Phase 6's _get_tesseract()/
    _get_pyautogui() -- keeps importing this module cheap and lets the
    OCR and vision halves work without uiautomation installed.

    comtypes (which uiautomation sits on) logs its type-library cache
    activity at INFO, which lands in the middle of a voice session as
    two lines about writeable cache directories every time the
    accessibility tree is touched. Quietened to WARNING here rather than
    globally, so a real comtypes problem still surfaces.
    """
    for noisy in ("comtypes", "comtypes.client", "comtypes._comobject"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    import uiautomation

    return uiautomation


# ---------------------------------------------------------------------------
# Grounding: the accessibility tree
# ---------------------------------------------------------------------------

def _target_window(auto, window_title_contains: str = None):
    """The window to inspect: one matching the given title, or whatever
    currently has focus."""
    if not window_title_contains:
        return auto.GetForegroundControl(), None

    needle = window_title_contains.strip().lower()
    matches = [
        w for w in auto.GetRootControl().GetChildren()
        if needle in (w.Name or "").lower()
    ]
    if not matches:
        return None, {"error": f"No window found with '{window_title_contains}' in its title."}
    if len(matches) > 1:
        return None, {
            "error": (
                f"{len(matches)} windows match '{window_title_contains}': "
                + "; ".join((w.Name or "?")[:40] for w in matches[:6])
                + ". Be more specific."
            )
        }
    return matches[0], None


def _walk(control, depth: int, deadline: float, found: list, max_elements: int) -> None:
    """Breadth-limited walk of the accessibility tree.

    Bounded by depth, element count AND wall clock, because all three
    can run away independently: a browser page is deep, a long list is
    wide, and a busy application can simply be slow to answer.
    """
    if depth > UI_TREE_MAX_DEPTH or len(found) >= max_elements:
        return
    if time.monotonic() > deadline:
        return

    try:
        children = control.GetChildren()
    except Exception:
        return  # a control that vanished mid-walk is not an error

    for child in children:
        if len(found) >= max_elements or time.monotonic() > deadline:
            return
        try:
            name = (child.Name or "").strip()
            control_type = child.ControlTypeName
            rect = child.BoundingRectangle

            usable = (
                name
                and control_type in INTERACTIVE_CONTROL_TYPES
                and rect.width() > 0
                and rect.height() > 0
            )
            if usable:
                found.append({
                    "name": name[:120],
                    "type": control_type.replace("Control", ""),
                    "center_x": rect.left + rect.width() // 2,
                    "center_y": rect.top + rect.height() // 2,
                    "width": rect.width(),
                    "height": rect.height(),
                })
        except Exception:
            continue  # one unreadable node must not fail the whole walk

        _walk(child, depth + 1, deadline, found, max_elements)


def list_ui_elements(window_title_contains: str = None, name_contains: str = None,
                     max_elements: int = 80) -> dict:
    """
    Read-only. The clickable things in a window, with exact centres.

    This is the tool that makes mouse control usable: it answers "where
    is X" precisely, rather than asking a model to estimate pixels from
    a picture.
    """
    if not IS_WINDOWS:
        return dict(NOT_WINDOWS)
    try:
        auto = _get_uia()
    except ImportError as e:
        return {"error": f"uiautomation is not installed: {e}. Run: pip install uiautomation"}

    window, error = _target_window(auto, window_title_contains)
    if error:
        return error
    if window is None:
        return {"error": "Could not determine which window to inspect."}

    found = []
    started = time.monotonic()
    try:
        _walk(window, 0, started + UI_TREE_TIME_BUDGET_SECONDS, found, max_elements)
    except Exception as e:
        return {"error": f"Could not read the window's accessibility tree: {e}"}

    if name_contains:
        needle = name_contains.strip().lower()
        found = [e for e in found if needle in e["name"].lower()]

    result = {
        "window": (window.Name or "(untitled)")[:120],
        "count": len(found),
        "elements": found,
    }
    if time.monotonic() - started > UI_TREE_TIME_BUDGET_SECONDS:
        result["note"] = (
            "Stopped early -- this window's element tree is large. "
            "Filter with name_contains for a narrower look."
        )
    elif len(found) >= max_elements:
        result["note"] = f"Stopped at {max_elements} elements; there may be more."
    if not found:
        result["note"] = (
            "No named, interactive elements found. This application may draw its own "
            "interface without publishing accessibility information -- try "
            "read_screen_text for visible text, or look_at_screen to see it."
        )
    return result


def click_element(name_contains: str, window_title_contains: str = None,
                  button: str = "left", clicks: int = 1) -> dict:
    """
    GATED. Proposes clicking a named element; clicks nothing on this
    call.

    Goes through the same two-phase confirmation as Phase 6's
    coordinate `click`, and for exactly the same reason: a click can do
    whatever the thing under it does. Knowing the target by name makes
    the click ACCURATE, not safe -- the proposal message can now say
    "the Save button" instead of "(840, 512)", which makes the human's
    decision better informed, but it is still their decision.

    The element is re-resolved at confirmation time. A window can
    re-layout between proposing and confirming, and clicking a
    remembered rectangle that now belongs to something else is exactly
    the failure this gate exists to prevent -- the same class of check
    as the stale-handle guard on close_window and the PID-reuse guard on
    kill_process.
    """
    if not IS_WINDOWS:
        return dict(NOT_WINDOWS)

    listing = list_ui_elements(window_title_contains, name_contains=name_contains)
    if "error" in listing:
        return listing

    matches = listing["elements"]
    if not matches:
        return {
            "error": (
                f"No clickable element named like '{name_contains}' in "
                f"'{listing['window']}'. Call list_ui_elements to see what is there."
            )
        }
    if len(matches) > 1:
        exact = [e for e in matches if e["name"].lower() == name_contains.strip().lower()]
        if len(exact) != 1:
            return {
                "error": (
                    f"{len(matches)} elements match '{name_contains}': "
                    + "; ".join(f"{e['name']} ({e['type']})" for e in matches[:6])
                    + ". Be more specific."
                )
            }
        matches = exact

    target = matches[0]
    window_name = listing["window"]

    def _execute():
        # Re-resolve rather than trusting the remembered rectangle.
        fresh = list_ui_elements(window_title_contains, name_contains=target["name"])
        candidates = [e for e in fresh.get("elements", []) if e["name"] == target["name"]]
        if not candidates:
            return {
                "error": (
                    f"'{target['name']}' is no longer on screen. Nothing was clicked."
                )
            }
        spot = candidates[0]
        try:
            import pyautogui

            pyautogui.FAILSAFE = True  # forced on, same as Phase 6 -- never disableable
            pyautogui.click(x=spot["center_x"], y=spot["center_y"],
                            clicks=clicks, button=button)
        except ImportError as e:
            return {"error": f"pyautogui is not installed: {e}."}
        except Exception as e:
            return {"error": f"Could not click: {e}"}
        return {
            "status": "clicked",
            "element": target["name"],
            "type": target["type"],
            "x": spot["center_x"],
            "y": spot["center_y"],
        }

    return gate.propose(
        action="click_element",
        message=(
            f"About to {button}-click the {target['type']} '{target['name']}' "
            f"in '{window_name}'. A click can do whatever that control does."
        ),
        execute=_execute,
        element=target["name"],
    )


# ---------------------------------------------------------------------------
# OCR with positions
# ---------------------------------------------------------------------------

def read_screen_text(region: list = None, all_screens: bool = False,
                     min_confidence: int = 55, max_items: int = 120) -> dict:
    """
    Read-only. Visible text WITH the position of each word, so text on
    screen becomes something that can be pointed at.

    Phase 6's read_screen returns the same text as an undifferentiated
    block with no coordinates. This is the same OCR engine asked a
    better question. Use it when a window publishes no accessibility
    information -- otherwise list_ui_elements is exact where this is a
    best guess.

    As with Phase 6's version, the captured image is never written to
    disk and is discarded as soon as OCR finishes.
    """
    try:
        from PIL import ImageGrab
    except ImportError as e:
        return {"error": f"Pillow is not installed: {e}. Run: pip install Pillow"}

    bbox = None
    offset_x = offset_y = 0
    if region is not None:
        if len(region) != 4:
            return {"error": "region must be [left, top, width, height]."}
        left, top, width, height = region
        bbox = (left, top, left + width, top + height)
        offset_x, offset_y = left, top

    try:
        image = ImageGrab.grab(bbox=bbox, all_screens=all_screens)
    except Exception as e:
        return {"error": f"Could not capture screen: {e}"}

    try:
        import pytesseract
        from pytesseract import Output

        cmd = os.getenv("TESSERACT_CMD")
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
        data = pytesseract.image_to_data(image, output_type=Output.DICT)
    except ImportError as e:
        return {"error": f"pytesseract is not installed: {e}. Run: pip install pytesseract"}
    except Exception as e:
        return {
            "error": (
                f"OCR failed: {e}. Make sure the Tesseract engine itself is installed "
                "(not just the pip package), or set TESSERACT_CMD in .env."
            )
        }

    items = []
    for i, word in enumerate(data.get("text", [])):
        word = (word or "").strip()
        if not word:
            continue
        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            continue
        if confidence < min_confidence:
            continue
        items.append({
            "text": word,
            "center_x": offset_x + data["left"][i] + data["width"][i] // 2,
            "center_y": offset_y + data["top"][i] + data["height"][i] // 2,
            "confidence": round(confidence),
        })
        if len(items) >= max_items:
            break

    return {
        "count": len(items),
        "items": items,
        "full_text": " ".join(item["text"] for item in items),
        "region": list(region) if region is not None else "full screen",
    }


# ---------------------------------------------------------------------------
# Vision: a local model looking at the actual pixels
# ---------------------------------------------------------------------------

def _capture_for_vision(region: list = None, all_screens: bool = False):
    """Grabs the screen and returns it base64-encoded, downscaled.

    Never touches disk. A screenshot is the most sensitive thing this
    assistant can capture -- whatever happened to be on screen -- so it
    exists in memory for the duration of one call and no longer, which
    is the same rule Phase 6 set for its own screen reading.
    """
    from PIL import ImageGrab

    bbox = None
    if region is not None:
        if len(region) != 4:
            raise ValueError("region must be [left, top, width, height].")
        left, top, width, height = region
        bbox = (left, top, left + width, top + height)

    image = ImageGrab.grab(bbox=bbox, all_screens=all_screens)
    if image.width > VISION_MAX_WIDTH:
        ratio = VISION_MAX_WIDTH / image.width
        image = image.resize((VISION_MAX_WIDTH, int(image.height * ratio)))

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii"), image.size


def look_at_screen(question: str = None, region: list = None,
                   all_screens: bool = False) -> dict:
    """
    Read-only. Actually looks at the screen with a vision model and
    answers a question about it.

    This is the half OCR and the accessibility tree cannot do: icons
    with no name, images, video, games, charts, layout, anything whose
    meaning is pictorial. Ask it what something looks like, what an
    unlabelled button probably does, or what is happening on screen.

    It will NOT give reliable coordinates, and should not be asked for
    them. Vision models estimate positions from a downscaled image and
    are routinely tens of pixels out. Use list_ui_elements to find out
    where something is; use this to find out what it is.

    Runs locally through Ollama so the screenshot never leaves the
    machine.
    """
    import requests

    try:
        encoded, size = _capture_for_vision(region, all_screens)
    except ImportError as e:
        return {"error": f"Pillow is not installed: {e}."}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Could not capture screen: {e}"}

    prompt = question or "Describe what is on this screen, concisely."
    started = time.monotonic()
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": VISION_MODEL,
                "prompt": prompt,
                "images": [encoded],
                "stream": False,
                "keep_alive": VISION_KEEP_ALIVE,
            },
            timeout=VISION_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.ConnectionError:
        return {
            "error": (
                f"Could not reach Ollama at {OLLAMA_URL}. Is it running? "
                "The vision model runs locally, so Ollama must be up."
            )
        }
    except requests.exceptions.Timeout:
        return {"error": f"The vision model did not answer within {VISION_TIMEOUT_SECONDS:.0f}s."}
    except requests.exceptions.RequestException as e:
        return {"error": f"Vision request failed: {e}"}
    except ValueError as e:
        return {"error": f"Vision model returned unreadable data: {e}"}

    elapsed = time.monotonic() - started

    answer = (data.get("response") or "").strip()
    if not answer:
        return {"error": "The vision model returned nothing."}

    if elapsed > 15:
        # Otherwise a 40-second pause looks like a hang rather than a
        # model being paged into VRAM.
        logger.info(
            f"Vision took {elapsed:.0f}s -- {VISION_MODEL} was cold. "
            f"Subsequent looks take a few seconds while it stays loaded "
            f"(VISION_KEEP_ALIVE={VISION_KEEP_ALIVE})."
        )

    return {
        "answer": answer,
        "model": VISION_MODEL,
        "looked_at": list(region) if region is not None else "full screen",
        "image_size": f"{size[0]}x{size[1]}",
    }


TOOL_FUNCTIONS = {
    "list_ui_elements": list_ui_elements,
    "click_element": click_element,
    "read_screen_text": read_screen_text,
    "look_at_screen": look_at_screen,
}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_ui_elements",
            "description": (
                "List the clickable elements of a window (buttons, menu items, text "
                "fields, list items) with their exact on-screen centres, read from "
                "Windows' accessibility tree. THIS is how you find out where something "
                "is -- it is exact, where a screenshot is a guess. Use it before any "
                "clicking. Defaults to the focused window."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "window_title_contains": {
                        "type": "string",
                        "description": "Which window to inspect. Defaults to the focused one.",
                    },
                    "name_contains": {
                        "type": "string",
                        "description": "Only return elements whose name contains this.",
                    },
                    "max_elements": {"type": "integer", "description": "Cap (default 80)."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click_element",
            "description": (
                "Click a named element found via the accessibility tree -- far more "
                "reliable than clicking coordinates, because it targets the control "
                "itself. Strongly prefer this over the coordinate-based click tool. "
                "Requires separate human confirmation, and the element is re-checked "
                "at confirmation time in case the window moved."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name_contains": {
                        "type": "string",
                        "description": "Name of the element, e.g. 'Save' or 'Close'.",
                    },
                    "window_title_contains": {
                        "type": "string",
                        "description": "Which window. Defaults to the focused one.",
                    },
                    "button": {"type": "string", "enum": ["left", "right", "middle"]},
                    "clicks": {"type": "integer", "description": "1 for a click, 2 to double-click."},
                },
                "required": ["name_contains"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_screen_text",
            "description": (
                "Read visible text WITH the position of each word, so text on screen "
                "can be pointed at. Use when an application publishes no accessibility "
                "information and list_ui_elements comes back empty. For plain reading "
                "with no positions, read_screen is cheaper."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Optional [left, top, width, height] to narrow the capture.",
                    },
                    "all_screens": {"type": "boolean", "description": "Include all monitors."},
                    "min_confidence": {"type": "integer", "description": "OCR confidence floor (default 55)."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "look_at_screen",
            "description": (
                "Actually LOOK at the screen with a vision model and answer a question "
                "about it. Use for anything pictorial: icons with no label, images, "
                "video, games, charts, layout, or 'what am I looking at'. It does NOT "
                "give reliable coordinates -- never click based on positions it "
                "describes; use list_ui_elements for that."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "What to ask about the screen. Defaults to describing it.",
                    },
                    "region": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Optional [left, top, width, height] to look at part of it.",
                    },
                    "all_screens": {"type": "boolean", "description": "Include all monitors."},
                },
            },
        },
    },
]
