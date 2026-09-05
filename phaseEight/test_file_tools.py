"""
test_file_tools.py -- Phase 8 milestone 2 tests for the file/system tools.

Real filesystem work against pytest's tmp_path rather than mocked
os/shutil: these tools are almost entirely filesystem semantics, and
mocking the filesystem would test the mock. Nothing here touches
anything outside tmp_path.

The conditional-gate tests are the important ones. The rule inherited
from Phase 6's write_file -- creating is safe, replacing is not -- has
to hold in both directions: a safe operation must not demand a
password, and a destructive one must not slip through without one.
"""

import os
import zipfile

import pytest

import file_tools
import gate


@pytest.fixture(autouse=True)
def clean_gate():
    gate._pending_actions.clear()
    yield
    gate._pending_actions.clear()


# ---------------------------------------------------------------------------
# find_files
# ---------------------------------------------------------------------------

@pytest.fixture
def tree(tmp_path):
    (tmp_path / "notes.txt").write_text("the quick brown fox")
    (tmp_path / "report.pdf").write_bytes(b"%PDF-1.4 fake")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "deep_notes.txt").write_text("nothing relevant here")
    skipped = tmp_path / "node_modules"
    skipped.mkdir()
    (skipped / "hidden_notes.txt").write_text("the quick brown fox")
    return tmp_path


def test_find_files_matches_a_glob(tree):
    result = file_tools.find_files("*.txt", root=str(tree))
    assert {os.path.basename(m["path"]) for m in result["matches"]} == {"notes.txt", "deep_notes.txt"}


def test_find_files_searches_subdirectories(tree):
    result = file_tools.find_files("deep_*.txt", root=str(tree))
    assert result["count"] == 1


def test_find_files_skips_noise_directories(tree):
    """A search that spends its budget inside node_modules and reports
    'not found' has given a wrong answer, not a slow one."""
    result = file_tools.find_files("hidden_notes.txt", root=str(tree))
    assert result["count"] == 0


def test_find_files_filters_by_content(tree):
    result = file_tools.find_files("*.txt", root=str(tree), contains="quick brown")
    assert result["count"] == 1
    assert os.path.basename(result["matches"][0]["path"]) == "notes.txt"


def test_find_files_content_filter_ignores_binary(tree):
    (tree / "blob.txt").write_bytes(b"quick brown\x00\x01binary")
    result = file_tools.find_files("*.txt", root=str(tree), contains="quick brown")
    assert all("blob" not in m["path"] for m in result["matches"])


def test_find_files_caps_results_and_says_so(tree):
    for i in range(10):
        (tree / f"bulk{i}.txt").write_text("x")
    result = file_tools.find_files("bulk*.txt", root=str(tree), max_results=3)
    assert result["count"] == 3
    assert "3 matches" in result["note"]


def test_find_files_is_case_insensitive(tree):
    (tree / "SHOUTING.TXT").write_text("x")
    assert file_tools.find_files("shouting.txt", root=str(tree))["count"] == 1


def test_find_files_on_a_missing_root_is_an_error_not_a_crash(tmp_path):
    assert "error" in file_tools.find_files("*.txt", root=str(tmp_path / "nope"))


def test_find_files_finding_nothing_is_not_an_error(tree):
    result = file_tools.find_files("*.nonexistent", root=str(tree))
    assert result["count"] == 0
    assert "error" not in result


# ---------------------------------------------------------------------------
# create_directory
# ---------------------------------------------------------------------------

def test_create_directory_creates_nested_paths(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    assert file_tools.create_directory(str(target))["status"] == "created"
    assert target.is_dir()


def test_create_directory_on_an_existing_one_is_not_an_error(tmp_path):
    """'Make sure this folder exists' is what the request always means."""
    file_tools.create_directory(str(tmp_path / "x"))
    assert file_tools.create_directory(str(tmp_path / "x"))["status"] == "exists"


# ---------------------------------------------------------------------------
# move_path / copy_path -- the conditional gate
# ---------------------------------------------------------------------------

def test_move_to_a_free_destination_happens_immediately(tmp_path):
    """Ungated on purpose: a move to a free path is reversible."""
    src = tmp_path / "a.txt"
    src.write_text("hello")
    result = file_tools.move_path(str(src), str(tmp_path / "b.txt"))
    assert result["status"] == "moved"
    assert (tmp_path / "b.txt").read_text() == "hello"
    assert not src.exists()


def test_move_onto_an_existing_file_is_gated(tmp_path):
    src, dst = tmp_path / "a.txt", tmp_path / "b.txt"
    src.write_text("new")
    dst.write_text("original")
    result = file_tools.move_path(str(src), str(dst))
    assert result["status"] == "confirmation_required"
    assert dst.read_text() == "original"
    assert src.exists()


def test_confirming_a_gated_move_actually_moves(tmp_path):
    src, dst = tmp_path / "a.txt", tmp_path / "b.txt"
    src.write_text("new")
    dst.write_text("original")
    token = file_tools.move_path(str(src), str(dst))["token"]
    assert gate.confirm_pending_action(token)["status"] == "moved"
    assert dst.read_text() == "new"


def test_cancelling_a_gated_move_changes_nothing(tmp_path):
    src, dst = tmp_path / "a.txt", tmp_path / "b.txt"
    src.write_text("new")
    dst.write_text("original")
    token = file_tools.move_path(str(src), str(dst))["token"]
    gate.cancel_pending_action(token)
    assert dst.read_text() == "original"
    assert src.read_text() == "new"


def test_moving_into_a_directory_lands_inside_it(tmp_path):
    """'Move this into Documents' means inside, not replacing it."""
    src = tmp_path / "a.txt"
    src.write_text("hello")
    folder = tmp_path / "folder"
    folder.mkdir()
    result = file_tools.move_path(str(src), str(folder))
    assert result["status"] == "moved"
    assert (folder / "a.txt").read_text() == "hello"


def test_moving_into_a_directory_gates_when_it_would_replace(tmp_path):
    src = tmp_path / "a.txt"
    src.write_text("new")
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "a.txt").write_text("original")
    assert file_tools.move_path(str(src), str(folder))["status"] == "confirmation_required"
    assert (folder / "a.txt").read_text() == "original"


def test_move_from_a_missing_source_is_an_error(tmp_path):
    assert "error" in file_tools.move_path(str(tmp_path / "nope.txt"), str(tmp_path / "b.txt"))


def test_move_onto_itself_is_refused(tmp_path):
    src = tmp_path / "a.txt"
    src.write_text("hello")
    assert "error" in file_tools.move_path(str(src), str(src))


def test_copy_to_a_free_destination_happens_immediately(tmp_path):
    src = tmp_path / "a.txt"
    src.write_text("hello")
    assert file_tools.copy_path(str(src), str(tmp_path / "b.txt"))["status"] == "copied"
    assert src.exists()  # copy leaves the original


def test_copy_onto_an_existing_file_is_gated(tmp_path):
    src, dst = tmp_path / "a.txt", tmp_path / "b.txt"
    src.write_text("new")
    dst.write_text("original")
    assert file_tools.copy_path(str(src), str(dst))["status"] == "confirmation_required"
    assert dst.read_text() == "original"


def test_copy_handles_directories(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "inner.txt").write_text("hi")
    file_tools.copy_path(str(source), str(tmp_path / "dst"))
    assert (tmp_path / "dst" / "inner.txt").read_text() == "hi"


# ---------------------------------------------------------------------------
# extract_archive
# ---------------------------------------------------------------------------

def _make_zip(path, entries):
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return path


def test_extract_into_a_clean_destination_is_ungated(tmp_path):
    archive = _make_zip(tmp_path / "a.zip", {"one.txt": "1", "two.txt": "2"})
    result = file_tools.extract_archive(str(archive), str(tmp_path / "out"))
    assert result["status"] == "extracted"
    assert (tmp_path / "out" / "one.txt").read_text() == "1"


def test_extract_that_would_overwrite_is_gated(tmp_path):
    archive = _make_zip(tmp_path / "a.zip", {"one.txt": "new"})
    out = tmp_path / "out"
    out.mkdir()
    (out / "one.txt").write_text("original")
    result = file_tools.extract_archive(str(archive), str(out))
    assert result["status"] == "confirmation_required"
    assert (out / "one.txt").read_text() == "original"


def test_confirming_a_gated_extract_overwrites(tmp_path):
    archive = _make_zip(tmp_path / "a.zip", {"one.txt": "new"})
    out = tmp_path / "out"
    out.mkdir()
    (out / "one.txt").write_text("original")
    token = file_tools.extract_archive(str(archive), str(out))["token"]
    gate.confirm_pending_action(token)
    assert (out / "one.txt").read_text() == "new"


def test_extract_refuses_zip_slip_traversal(tmp_path):
    """A crafted archive that writes outside the destination, over any
    file the process can reach. Refused before anything is written --
    and refused outright rather than gated, since there is no
    legitimate reason to approve it."""
    archive = _make_zip(tmp_path / "evil.zip", {"../escaped.txt": "pwned"})
    result = file_tools.extract_archive(str(archive), str(tmp_path / "out"))
    assert "error" in result
    assert "not safe" in result["error"]
    assert not (tmp_path / "escaped.txt").exists()


def test_extract_refuses_absolute_paths(tmp_path):
    archive = _make_zip(tmp_path / "evil.zip", {"/etc/passwd": "pwned"})
    assert "error" in file_tools.extract_archive(str(archive), str(tmp_path / "out"))


def test_extract_rejects_a_non_zip(tmp_path):
    plain = tmp_path / "notazip.zip"
    plain.write_text("definitely not a zip")
    assert "error" in file_tools.extract_archive(str(plain))


# ---------------------------------------------------------------------------
# get_disk_usage / read_document
# ---------------------------------------------------------------------------

def test_disk_usage_returns_real_numbers(tmp_path):
    result = file_tools.get_disk_usage(str(tmp_path))
    assert result["total_gb"] > 0
    assert result["free_gb"] <= result["total_gb"]


def test_read_document_rejects_unsupported_extensions(tmp_path):
    plain = tmp_path / "notes.txt"
    plain.write_text("hello")
    result = file_tools.read_document(str(plain))
    assert "read_file" in result["error"]


def test_read_document_on_a_missing_file_is_an_error(tmp_path):
    assert "error" in file_tools.read_document(str(tmp_path / "nope.pdf"))


def test_read_document_reads_a_real_docx(tmp_path):
    docx = pytest.importorskip("docx")
    path = tmp_path / "doc.docx"
    document = docx.Document()
    document.add_paragraph("First paragraph.")
    document.add_paragraph("Second paragraph.")
    document.save(str(path))

    result = file_tools.read_document(str(path))
    assert "First paragraph." in result["content"]
    assert result["paragraphs"] == 2


def test_read_document_truncates_and_says_so(tmp_path):
    docx = pytest.importorskip("docx")
    path = tmp_path / "big.docx"
    document = docx.Document()
    for _ in range(200):
        document.add_paragraph("x" * 100)
    document.save(str(path))

    result = file_tools.read_document(str(path), max_chars=500)
    assert len(result["content"]) == 500
    assert result["truncated"] is True


def test_read_document_reports_an_image_only_pdf_clearly(tmp_path):
    """A scanned PDF has no text layer. Saying so points at read_screen
    instead of silently returning nothing."""
    pypdf = pytest.importorskip("pypdf")
    path = tmp_path / "scan.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(path, "wb") as f:
        writer.write(f)

    result = file_tools.read_document(str(path))
    assert "read_screen" in result["error"]


# ---------------------------------------------------------------------------
# Registry contract
# ---------------------------------------------------------------------------

def test_schemas_and_functions_cover_the_same_names():
    schema_names = {s["function"]["name"] for s in file_tools.TOOL_SCHEMAS}
    assert schema_names == set(file_tools.TOOL_FUNCTIONS)
