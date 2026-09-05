"""
test_screen.py -- Phase 8 milestone 5 tests for screen understanding.

Fully mocked: no accessibility tree, no OCR engine, no vision model, no
screenshots. The UIA layer is faked at the _get_uia() seam, OCR at
pytesseract, and vision at the requests seam.

The tests worth reading are the ones about the DIVISION OF LABOUR --
that click_element resolves through the accessibility tree rather than
from anything a vision model said, and that it re-resolves at
confirmation time. Those are the properties that make mouse control
land on the thing it aimed at.
"""

import sys
import types

import pytest

import gate
import screen


@pytest.fixture(autouse=True)
def clean_gate():
    gate._pending_actions.clear()
    yield
    gate._pending_actions.clear()


# ---------------------------------------------------------------------------
# A fake accessibility tree
# ---------------------------------------------------------------------------

class _Rect:
    def __init__(self, left, top, right, bottom):
        self.left, self.top, self.right, self.bottom = left, top, right, bottom

    def width(self):
        return self.right - self.left

    def height(self):
        return self.bottom - self.top


class _Control:
    def __init__(self, name, control_type, rect, children=None):
        self.Name = name
        self.ControlTypeName = control_type
        self.BoundingRectangle = rect
        self._children = children or []

    def GetChildren(self):
        return self._children


def _window(*children, name="Notepad"):
    return _Control(name, "WindowControl", _Rect(0, 0, 800, 600), list(children))


def _fake_uia(window, monkeypatch, extra_windows=()):
    root = _Control("Desktop", "PaneControl", _Rect(0, 0, 800, 600),
                    [window, *extra_windows])
    module = types.SimpleNamespace(
        GetForegroundControl=lambda: window,
        GetRootControl=lambda: root,
    )
    monkeypatch.setattr(screen, "_get_uia", lambda: module)
    monkeypatch.setattr(screen, "IS_WINDOWS", True)
    return module


SAVE = _Control("Save", "ButtonControl", _Rect(100, 200, 200, 240))
CANCEL = _Control("Cancel", "ButtonControl", _Rect(220, 200, 320, 240))
PANE = _Control("some pane", "PaneControl", _Rect(0, 0, 800, 100))
UNNAMED = _Control("", "ButtonControl", _Rect(400, 200, 500, 240))
ZERO_SIZE = _Control("Hidden", "ButtonControl", _Rect(0, 0, 0, 0))


# ---------------------------------------------------------------------------
# list_ui_elements
# ---------------------------------------------------------------------------

def test_it_reports_elements_with_exact_centres(monkeypatch):
    """The whole point: an exact centre, not an estimate from a picture."""
    _fake_uia(_window(SAVE, CANCEL), monkeypatch)
    result = screen.list_ui_elements()
    names = {e["name"]: e for e in result["elements"]}
    assert names["Save"]["center_x"] == 150
    assert names["Save"]["center_y"] == 220


def test_it_skips_structural_furniture(monkeypatch):
    """Panes and groups cannot be interacted with and would crowd out
    the real targets."""
    _fake_uia(_window(SAVE, PANE), monkeypatch)
    assert [e["name"] for e in screen.list_ui_elements()["elements"]] == ["Save"]


def test_it_skips_unnamed_and_zero_size_controls(monkeypatch):
    """An unnamed control cannot be asked for by name, and a zero-size
    one is not on screen."""
    _fake_uia(_window(SAVE, UNNAMED, ZERO_SIZE), monkeypatch)
    assert [e["name"] for e in screen.list_ui_elements()["elements"]] == ["Save"]


def test_it_finds_elements_nested_deep_in_the_tree(monkeypatch):
    """Real applications nest controls several layers down inside panes."""
    nested = _Control("panel", "PaneControl", _Rect(0, 0, 800, 600),
                      [_Control("inner", "PaneControl", _Rect(0, 0, 800, 600), [SAVE])])
    _fake_uia(_window(nested), monkeypatch)
    assert [e["name"] for e in screen.list_ui_elements()["elements"]] == ["Save"]


def test_a_name_filter_narrows_the_list(monkeypatch):
    _fake_uia(_window(SAVE, CANCEL), monkeypatch)
    result = screen.list_ui_elements(name_contains="can")
    assert [e["name"] for e in result["elements"]] == ["Cancel"]


def test_an_empty_result_explains_what_to_try_next(monkeypatch):
    """Apps that draw themselves (games, some Electron apps) publish
    nothing. Saying so beats returning an empty list."""
    _fake_uia(_window(PANE), monkeypatch)
    result = screen.list_ui_elements()
    assert result["count"] == 0
    assert "read_screen_text" in result["note"]


def test_the_element_cap_is_respected(monkeypatch):
    many = [_Control(f"Button {i}", "ButtonControl", _Rect(i, 0, i + 10, 10)) for i in range(50)]
    _fake_uia(_window(*many), monkeypatch)
    result = screen.list_ui_elements(max_elements=5)
    assert result["count"] == 5
    assert "5 elements" in result["note"]


def test_an_ambiguous_window_name_is_refused(monkeypatch):
    """Same rule as everywhere else in this phase: ambiguity is reported,
    never guessed at."""
    target = _window(SAVE, name="Notes - Editor")
    other = _window(CANCEL, name="Notes - Backup")
    _fake_uia(target, monkeypatch, extra_windows=(other,))
    result = screen.list_ui_elements(window_title_contains="Notes")
    assert "2 windows match" in result["error"]


def test_a_missing_window_is_reported(monkeypatch):
    _fake_uia(_window(SAVE), monkeypatch)
    assert "No window found" in screen.list_ui_elements(window_title_contains="Photoshop")["error"]


def test_it_refuses_cleanly_off_windows(monkeypatch):
    monkeypatch.setattr(screen, "IS_WINDOWS", False)
    assert "error" in screen.list_ui_elements()
    assert "error" in screen.click_element("Save")


# ---------------------------------------------------------------------------
# click_element -- the gated one
# ---------------------------------------------------------------------------

def test_clicking_only_proposes(monkeypatch):
    _fake_uia(_window(SAVE), monkeypatch)
    clicked = []
    monkeypatch.setitem(sys.modules, "pyautogui", types.SimpleNamespace(
        FAILSAFE=False, click=lambda **kw: clicked.append(kw)))

    result = screen.click_element("Save")
    assert result["status"] == "confirmation_required"
    assert clicked == []


def test_the_proposal_names_the_element_not_coordinates(monkeypatch):
    """A human approving a click should be told "the Save button", not
    "(150, 220)". Knowing the target by name makes the decision better
    informed -- it does not make the click safe, which is why it is
    still gated."""
    _fake_uia(_window(SAVE), monkeypatch)
    message = screen.click_element("Save")["message"]
    assert "Save" in message
    assert "150" not in message


def test_confirming_clicks_the_resolved_centre(monkeypatch):
    _fake_uia(_window(SAVE), monkeypatch)
    clicked = []
    monkeypatch.setitem(sys.modules, "pyautogui", types.SimpleNamespace(
        FAILSAFE=False, click=lambda **kw: clicked.append(kw)))

    token = screen.click_element("Save")["token"]
    outcome = gate.confirm_pending_action(token)
    assert outcome["status"] == "clicked"
    assert clicked[0]["x"] == 150 and clicked[0]["y"] == 220


def test_cancelling_clicks_nothing(monkeypatch):
    _fake_uia(_window(SAVE), monkeypatch)
    clicked = []
    monkeypatch.setitem(sys.modules, "pyautogui", types.SimpleNamespace(
        FAILSAFE=False, click=lambda **kw: clicked.append(kw)))

    gate.cancel_pending_action(screen.click_element("Save")["token"])
    assert clicked == []


def test_the_element_is_re_resolved_at_confirmation_time(monkeypatch):
    """A window can re-layout between proposing and confirming, and
    clicking a remembered rectangle that now belongs to something else
    is exactly what this gate exists to prevent -- the same class of
    check as close_window's stale handle and kill_process's PID reuse."""
    window = _window(SAVE)
    _fake_uia(window, monkeypatch)
    clicked = []
    monkeypatch.setitem(sys.modules, "pyautogui", types.SimpleNamespace(
        FAILSAFE=False, click=lambda **kw: clicked.append(kw)))

    token = screen.click_element("Save")["token"]
    # The button moves before the human confirms.
    window._children = [_Control("Save", "ButtonControl", _Rect(500, 400, 600, 440))]

    gate.confirm_pending_action(token)
    assert clicked[0]["x"] == 550, "clicked the stale position instead of re-resolving"


def test_confirming_aborts_if_the_element_vanished(monkeypatch):
    window = _window(SAVE)
    _fake_uia(window, monkeypatch)
    clicked = []
    monkeypatch.setitem(sys.modules, "pyautogui", types.SimpleNamespace(
        FAILSAFE=False, click=lambda **kw: clicked.append(kw)))

    token = screen.click_element("Save")["token"]
    window._children = []  # dialog closed while the human was deciding

    outcome = gate.confirm_pending_action(token)
    assert "error" in outcome
    assert clicked == []


def test_an_ambiguous_element_is_refused(monkeypatch):
    _fake_uia(_window(
        _Control("Save", "ButtonControl", _Rect(0, 0, 10, 10)),
        _Control("Save As", "ButtonControl", _Rect(20, 0, 30, 10)),
    ), monkeypatch)
    assert "2 elements match" in screen.click_element("Sav")["error"]


def test_an_exact_name_wins_over_a_partial_match(monkeypatch):
    """"Save" should click Save, not refuse because "Save As" exists."""
    _fake_uia(_window(
        _Control("Save", "ButtonControl", _Rect(100, 200, 200, 240)),
        _Control("Save As", "ButtonControl", _Rect(220, 200, 320, 240)),
    ), monkeypatch)
    assert screen.click_element("Save")["status"] == "confirmation_required"


def test_a_missing_element_points_at_list_ui_elements(monkeypatch):
    _fake_uia(_window(SAVE), monkeypatch)
    assert "list_ui_elements" in screen.click_element("Print")["error"]


def test_the_failsafe_is_forced_on(monkeypatch):
    """Phase 6 forced pyautogui's failsafe on unconditionally and gave
    nothing a way to disable it. Same rule here."""
    _fake_uia(_window(SAVE), monkeypatch)
    fake = types.SimpleNamespace(FAILSAFE=False, click=lambda **kw: None)
    monkeypatch.setitem(sys.modules, "pyautogui", fake)

    gate.confirm_pending_action(screen.click_element("Save")["token"])
    assert fake.FAILSAFE is True


# ---------------------------------------------------------------------------
# read_screen_text -- OCR with positions
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_ocr(monkeypatch):
    data = {
        "text": ["Save", "", "Cancel", "noise"],
        "conf": ["96", "-1", "88", "12"],
        "left": [100, 0, 220, 400],
        "top": [200, 0, 200, 300],
        "width": [100, 0, 100, 50],
        "height": [40, 0, 40, 20],
    }
    fake_pytesseract = types.SimpleNamespace(
        pytesseract=types.SimpleNamespace(tesseract_cmd=""),
        image_to_data=lambda image, output_type=None: data,
        Output=types.SimpleNamespace(DICT="dict"),
    )
    monkeypatch.setitem(sys.modules, "pytesseract", fake_pytesseract)
    # Pillow is genuinely installed, so patch the real grab() rather
    # than stubbing the package -- `from PIL import ImageGrab` does not
    # work against a bare namespace object.
    from PIL import ImageGrab

    monkeypatch.setattr(ImageGrab, "grab", lambda **kw: object())
    return data


def test_ocr_returns_words_with_positions(fake_ocr):
    result = screen.read_screen_text()
    words = {i["text"]: i for i in result["items"]}
    assert words["Save"]["center_x"] == 150
    assert words["Save"]["center_y"] == 220


def test_ocr_drops_low_confidence_noise(fake_ocr):
    """OCR emits confident garbage from icons and gradients. Anything
    the engine itself doubts is not worth offering as a click target."""
    assert "noise" not in {i["text"] for i in screen.read_screen_text()["items"]}


def test_ocr_skips_empty_strings(fake_ocr):
    assert all(i["text"] for i in screen.read_screen_text()["items"])


def test_ocr_offsets_positions_by_the_region(fake_ocr):
    """A region capture reports coordinates relative to itself, but a
    click needs screen coordinates."""
    result = screen.read_screen_text(region=[1000, 500, 400, 300])
    words = {i["text"]: i for i in result["items"]}
    assert words["Save"]["center_x"] == 1000 + 150
    assert words["Save"]["center_y"] == 500 + 220


def test_ocr_rejects_a_malformed_region(fake_ocr):
    assert "error" in screen.read_screen_text(region=[1, 2, 3])


# ---------------------------------------------------------------------------
# look_at_screen -- the vision half
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_vision(monkeypatch):
    captured = {}

    class _Response:
        status_code = 200

        @staticmethod
        def raise_for_status():
            pass

        @staticmethod
        def json():
            return {"response": "A code editor is open."}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        return _Response()

    monkeypatch.setattr(screen, "_capture_for_vision", lambda region, all_screens: ("BASE64", (1280, 720)))
    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(
        post=fake_post,
        exceptions=types.SimpleNamespace(
            ConnectionError=type("CE", (Exception,), {}),
            Timeout=type("TO", (Exception,), {}),
            RequestException=type("RE", (Exception,), {}),
        ),
    ))
    return captured


def test_vision_answers_a_question(fake_vision):
    result = screen.look_at_screen("what is open?")
    assert result["answer"] == "A code editor is open."
    assert fake_vision["payload"]["prompt"] == "what is open?"


def test_vision_sends_the_image(fake_vision):
    screen.look_at_screen("what is open?")
    assert fake_vision["payload"]["images"] == ["BASE64"]


def test_vision_calls_the_local_endpoint(fake_vision):
    """The screenshot is the most sensitive thing this assistant can
    capture, so it stays on the machine."""
    screen.look_at_screen("what is open?")
    assert "localhost" in fake_vision["url"] or "127.0.0.1" in fake_vision["url"]
    assert fake_vision["url"].endswith("/api/generate")


def test_vision_has_a_default_question(fake_vision):
    screen.look_at_screen()
    assert "escribe" in fake_vision["payload"]["prompt"]


def test_the_ollama_base_url_tolerates_a_full_endpoint(monkeypatch):
    """OLLAMA_URL is shared with Phase 1, which set it to
    ".../api/chat". Appending "/api/generate" to that produced a 404 --
    found on the first real call."""
    monkeypatch.setenv("OLLAMA_URL", "http://localhost:11434/api/chat")
    monkeypatch.delenv("OLLAMA_VISION_URL", raising=False)
    assert screen._ollama_base_url() == "http://localhost:11434"


def test_the_ollama_base_url_accepts_a_plain_host(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://192.168.1.5:11434")
    monkeypatch.delenv("OLLAMA_VISION_URL", raising=False)
    assert screen._ollama_base_url() == "http://192.168.1.5:11434"


# ---------------------------------------------------------------------------
# Registry contract
# ---------------------------------------------------------------------------

def test_schemas_and_functions_cover_the_same_names():
    assert {s["function"]["name"] for s in screen.TOOL_SCHEMAS} == set(screen.TOOL_FUNCTIONS)


def test_the_gate_is_not_reachable_as_a_tool():
    assert not set(screen.TOOL_FUNCTIONS) & {"confirm_pending_action", "cancel_pending_action"}


def test_the_vision_tool_warns_against_clicking_what_it_describes():
    """The specific failure this milestone is designed around: asking
    the vision model where something is, then clicking that."""
    description = next(
        s["function"]["description"] for s in screen.TOOL_SCHEMAS
        if s["function"]["name"] == "look_at_screen"
    )
    assert "coordinates" in description.lower()
    assert "list_ui_elements" in description
