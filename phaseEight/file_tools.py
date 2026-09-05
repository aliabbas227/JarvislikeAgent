"""
file_tools.py -- Phase 8, milestone 2: file and system work.

Phase 6 could list a directory, read a text file, create a file, and
(gated) delete or overwrite one. What it could not do is *find*
anything. Every "where did I save that" question was unanswerable
unless the model already knew the exact path, which meant the most
ordinary file request a person makes out loud was the one thing the
file tools couldn't handle.

This module adds finding, organising, and reading formats that aren't
plain text.

Guardrail split, same reasoning style as os_tools.py:

  UNGATED -- creates or reads only:
      find_files, create_directory, get_disk_usage, read_document

  CONDITIONALLY GATED -- ungated when nothing is destroyed, gated when
  something existing would be replaced:
      move_path, copy_path, extract_archive

That conditional split is not a new invention. It is exactly the rule
Phase 6 already established for write_file: creating a new file needs
no guardrail because nothing existing is lost, while replacing the
contents of one that already exists is as destructive as deleting it
and goes through the full two-phase gate. Moving a file to a free
destination is reversible (move it back); moving it onto an existing
file is not.
"""

import fnmatch
import os
import shutil
import time
import zipfile

import gate

# Directories skipped when walking. Not an optimisation so much as a
# correctness measure: a search that spends its whole time budget
# inside node_modules or a venv returns nothing useful and reports it
# as "not found", which is a wrong answer rather than a slow one.
SKIP_DIRECTORIES = {
    "node_modules", "__pycache__", ".git", ".svn", "venv", ".venv",
    "$RECYCLE.BIN", "System Volume Information", ".mypy_cache",
    ".pytest_cache", "site-packages", "AppData",
}

FIND_TIME_BUDGET_SECONDS = float(os.getenv("FIND_TIME_BUDGET_SECONDS", "60"))
DOCUMENT_MAX_CHARS = int(os.getenv("DOCUMENT_MAX_CHARS", "30000"))


# ---------------------------------------------------------------------------
# Finding
# ---------------------------------------------------------------------------

def find_files(pattern: str, root: str = None, contains: str = None,
               max_results: int = 100) -> dict:
    """
    Read-only. Walks `root` looking for files whose NAME matches
    `pattern` (a glob such as "*.pdf" or "invoice*"), optionally also
    filtering to those whose TEXT contains `contains`.

    Bounded by both a result cap and a wall-clock budget
    (FIND_TIME_BUDGET_SECONDS), and it reports which limit it hit.
    Both bounds exist for the same reason: this is called mid-voice-turn
    with a person waiting, and an unbounded filesystem walk over a
    900GB drive is indistinguishable from a hang. Saying "here are 25
    matches, there may be more" is a useful answer; silence for four
    minutes is not.

    `contains` only reads files small enough to be plausibly text and
    skips anything with a null byte in it, borrowing Phase 6's
    read_file binary sniff rather than trying to decode a 2GB video as
    UTF-8.
    """
    search_root = os.path.abspath(root or os.path.expanduser("~"))
    if not os.path.isdir(search_root):
        return {"error": f"Not a directory: {search_root}"}

    needle = contains.lower() if contains else None
    started = time.time()
    matches = []
    hit_limit = None

    for dirpath, dirnames, filenames in os.walk(search_root):
        if time.time() - started > FIND_TIME_BUDGET_SECONDS:
            hit_limit = "time"
            break
        # Pruning in place is what actually stops os.walk descending.
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRECTORIES and not d.startswith(".")]

        for filename in filenames:
            if not fnmatch.fnmatch(filename.lower(), pattern.lower()):
                continue
            full_path = os.path.join(dirpath, filename)
            try:
                stat = os.stat(full_path)
                if needle and not _file_text_contains(full_path, needle, stat.st_size):
                    continue
                matches.append({
                    "path": full_path,
                    "size_bytes": stat.st_size,
                    "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
                })
            except OSError:
                continue

            if len(matches) >= max_results:
                hit_limit = "results"
                break
        if hit_limit:
            break

    matches.sort(key=lambda m: m["modified"], reverse=True)
    result = {
        "searched": search_root,
        "pattern": pattern,
        "count": len(matches),
        "matches": matches,
    }
    if hit_limit == "time":
        result["note"] = (
            f"Stopped after {FIND_TIME_BUDGET_SECONDS:.0f}s. There may be more matches "
            "-- search a narrower root directory."
        )
    elif hit_limit == "results":
        result["note"] = f"Stopped at {max_results} matches. There may be more."
    return result


def _file_text_contains(path: str, needle: str, size_bytes: int) -> bool:
    """Best-effort content match. Never raises -- an unreadable file is
    simply not a match, since a permissions error on one file in a walk
    should not fail the whole search."""
    if size_bytes > 5_000_000:  # don't try to grep a 5MB+ file mid-turn
        return False
    try:
        with open(path, "rb") as f:
            raw = f.read()
        if b"\x00" in raw:  # binary, same sniff Phase 6's read_file uses
            return False
        return needle in raw.decode("utf-8", errors="replace").lower()
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Organising
# ---------------------------------------------------------------------------

def create_directory(path: str) -> dict:
    """Ungated -- creates only, destroys nothing. Succeeds quietly if
    the directory already exists, since "make sure this folder exists"
    is what the request always means."""
    abs_path = os.path.abspath(path)
    try:
        already = os.path.isdir(abs_path)
        os.makedirs(abs_path, exist_ok=True)
    except OSError as e:
        return {"error": f"Could not create directory: {e}"}
    return {"status": "exists" if already else "created", "path": abs_path}


def _propose_or_do(verb: str, past: str, src: str, dst: str, operation) -> dict:
    """
    Shared logic for move_path and copy_path: run it directly when the
    destination is free, gate it when something would be replaced.

    Directly implements Phase 6's write_file rule -- creating is safe,
    replacing is not -- so the two operations stay consistent with each
    other and with the phase that set the precedent.

    `past` is passed in rather than derived as verb + "d" because that
    produced "copyd". This string is not internal: it is the `status`
    the model reads back and the word the orchestrator prints in its
    [NOTE] line at the terminal, where a human is deciding whether what
    happened is what they asked for.
    """
    abs_src = os.path.abspath(src)
    abs_dst = os.path.abspath(dst)

    if not os.path.exists(abs_src):
        return {"error": f"Source does not exist: {abs_src}"}

    # Moving into a directory means landing *inside* it, which is what a
    # person means by "move this into Documents" -- and it changes what
    # counts as the thing being replaced.
    if os.path.isdir(abs_dst):
        abs_dst = os.path.join(abs_dst, os.path.basename(abs_src))

    if abs_src == abs_dst:
        return {"error": "Source and destination are the same path."}

    if not os.path.exists(abs_dst):
        try:
            parent = os.path.dirname(abs_dst)
            if parent:
                os.makedirs(parent, exist_ok=True)
            operation(abs_src, abs_dst)
        except (OSError, shutil.Error) as e:
            return {"error": f"Could not {verb}: {e}"}
        return {"status": past, "source": abs_src, "path": abs_dst}

    def _execute():
        try:
            if os.path.isdir(abs_dst):
                shutil.rmtree(abs_dst)
            operation(abs_src, abs_dst)
        except (OSError, shutil.Error) as e:
            return {"error": f"Could not {verb}: {e}"}
        return {"status": past, "source": abs_src, "path": abs_dst}

    return gate.propose(
        action=f"{verb}_overwrite",
        message=(
            f"About to {verb} '{abs_src}' onto '{abs_dst}', which already exists. "
            "Its current contents will be replaced and cannot be recovered."
        ),
        execute=_execute,
        path=abs_dst,
    )


def move_path(source: str, destination: str) -> dict:
    """Moves or renames a file or folder. Ungated when the destination
    is free (a move is reversible -- move it back), gated when it would
    replace something existing."""
    return _propose_or_do("move", "moved", source, destination, shutil.move)


def copy_path(source: str, destination: str) -> dict:
    """Copies a file or folder. Same conditional gate as move_path."""
    def _copy(src, dst):
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    return _propose_or_do("copy", "copied", source, destination, _copy)


def extract_archive(path: str, destination: str = None) -> dict:
    """
    Extracts a .zip archive. Ungated when nothing would be replaced,
    gated when extraction would overwrite existing files -- the same
    rule as move_path/copy_path, just applied across many files at
    once, which is exactly why it is worth checking rather than
    assuming.

    Refuses entries with absolute paths or `..` components. That is the
    Zip Slip vulnerability: a crafted archive can otherwise write
    outside the destination directory entirely, over any file the
    process can reach. Python's own extractall() has guarded against
    this since 3.6.something, but the check is explicit here anyway --
    this is a path where the input is an untrusted file and the tool
    calling it is driven by a language model.
    """
    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        return {"error": f"Not a file: {abs_path}"}
    if not zipfile.is_zipfile(abs_path):
        return {"error": f"'{abs_path}' is not a .zip archive. Only zip files are supported."}

    dest = os.path.abspath(destination or os.path.splitext(abs_path)[0])

    try:
        with zipfile.ZipFile(abs_path) as archive:
            names = archive.namelist()
    except (OSError, zipfile.BadZipFile) as e:
        return {"error": f"Could not read archive: {e}"}

    unsafe = [n for n in names if os.path.isabs(n) or ".." in n.replace("\\", "/").split("/")]
    if unsafe:
        return {
            "error": (
                f"Refusing to extract: {len(unsafe)} entries would write outside "
                f"'{dest}' (e.g. {unsafe[0]!r}). This archive is not safe."
            )
        }

    conflicts = [n for n in names if os.path.exists(os.path.join(dest, n)) and not n.endswith("/")]

    def _extract():
        try:
            with zipfile.ZipFile(abs_path) as archive:
                archive.extractall(dest)
        except (OSError, zipfile.BadZipFile) as e:
            return {"error": f"Extraction failed: {e}"}
        return {"status": "extracted", "path": dest, "file_count": len(names)}

    if not conflicts:
        return _extract()

    return gate.propose(
        action="extract_overwrite",
        message=(
            f"About to extract '{abs_path}' into '{dest}', replacing "
            f"{len(conflicts)} existing file(s) (e.g. {conflicts[0]}). "
            "Their current contents cannot be recovered."
        ),
        execute=_extract,
        path=dest,
    )


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def get_disk_usage(path: str = None) -> dict:
    """Read-only. Free and used space on the drive holding `path`."""
    target = os.path.abspath(path or os.path.expanduser("~"))
    try:
        usage = shutil.disk_usage(target)
    except OSError as e:
        return {"error": f"Could not read disk usage: {e}"}
    return {
        "path": target,
        "total_gb": round(usage.total / (1024 ** 3), 1),
        "used_gb": round(usage.used / (1024 ** 3), 1),
        "free_gb": round(usage.free / (1024 ** 3), 1),
        "percent_used": round(usage.used / usage.total * 100, 1) if usage.total else None,
    }


def read_document(path: str, max_chars: int = DOCUMENT_MAX_CHARS) -> dict:
    """
    Read-only. Extracts text from a PDF or .docx, which Phase 6's
    read_file cannot do -- it refuses both, correctly, on its
    binary-file sniff.

    Truncated for the same reason every other reader in this project
    is: one tool call should not be able to push an entire book into
    the model's context. Reports the page or paragraph count so the
    model can say how much it did not see, rather than silently
    answering from a fraction of the document as though it were the
    whole thing.
    """
    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        return {"error": f"Not a file: {abs_path}"}

    extension = os.path.splitext(abs_path)[1].lower()
    if extension == ".pdf":
        return _read_pdf(abs_path, max_chars)
    if extension == ".docx":
        return _read_docx(abs_path, max_chars)
    return {
        "error": (
            f"read_document handles .pdf and .docx. For '{extension or 'no extension'}' "
            "use read_file instead."
        )
    }


def _read_pdf(path: str, max_chars: int) -> dict:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        return {"error": f"pypdf is not installed: {e}. Run: pip install pypdf"}

    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            return {"error": f"'{path}' is password-protected."}
        pages = []
        total = 0
        for page in reader.pages:
            pages.append(page.extract_text() or "")
            total += len(pages[-1])
            if total > max_chars:
                break
        text = "\n".join(pages).strip()
        page_count = len(reader.pages)
    except Exception as e:
        return {"error": f"Could not read PDF: {e}"}

    if not text:
        return {
            "error": (
                f"'{path}' has no extractable text -- it is probably a scanned image. "
                "Open it and use read_screen instead."
            )
        }

    return {
        "path": path,
        "content": text[:max_chars],
        "truncated": len(text) > max_chars,
        "pages": page_count,
        "pages_read": len(pages),
    }


def _read_docx(path: str, max_chars: int) -> dict:
    try:
        import docx
    except ImportError as e:
        return {"error": f"python-docx is not installed: {e}. Run: pip install python-docx"}

    try:
        document = docx.Document(path)
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    except Exception as e:
        return {"error": f"Could not read document: {e}"}

    text = "\n".join(paragraphs).strip()
    if not text:
        return {"error": f"'{path}' contains no readable text."}

    return {
        "path": path,
        "content": text[:max_chars],
        "truncated": len(text) > max_chars,
        "paragraphs": len(paragraphs),
    }


TOOL_FUNCTIONS = {
    "find_files": find_files,
    "create_directory": create_directory,
    "move_path": move_path,
    "copy_path": copy_path,
    "extract_archive": extract_archive,
    "get_disk_usage": get_disk_usage,
    "read_document": read_document,
}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "find_files",
            "description": (
                "Search the filesystem for files by name pattern, optionally also filtering "
                "by text content. Use this for any 'where is', 'find my', or 'what did I save' "
                "question instead of guessing at a path. Searches the user's home folder by "
                "default. Bounded by time and result count, and it tells you when it stopped early."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Filename glob, e.g. '*.pdf', 'invoice*', '*report*.docx'.",
                    },
                    "root": {
                        "type": "string",
                        "description": "Directory to search under. Defaults to the user's home folder.",
                    },
                    "contains": {
                        "type": "string",
                        "description": "Optional text that must appear inside the file.",
                    },
                    "max_results": {"type": "integer", "description": "Cap on matches (default 100)."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_directory",
            "description": "Create a folder, including any missing parent folders.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Folder to create."}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_path",
            "description": (
                "Move or rename a file or folder. If the destination is a folder, the source "
                "lands inside it. Happens immediately when the destination is free; requires "
                "human confirmation if it would replace something that already exists."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "File or folder to move."},
                    "destination": {"type": "string", "description": "New path, or a folder to move it into."},
                },
                "required": ["source", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "copy_path",
            "description": (
                "Copy a file or folder. Same confirmation rule as move_path: immediate when "
                "the destination is free, confirmed when it would replace something."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "File or folder to copy."},
                    "destination": {"type": "string", "description": "Where to copy it to."},
                },
                "required": ["source", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_archive",
            "description": (
                "Extract a .zip archive. Defaults to a folder beside the archive. Requires "
                "human confirmation only if extracting would overwrite existing files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The .zip file."},
                    "destination": {"type": "string", "description": "Optional folder to extract into."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_disk_usage",
            "description": "How much space is free and used on the drive holding a given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Any path on the drive. Defaults to home."}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": (
                "Extract text from a PDF or Word (.docx) file. Use this instead of read_file "
                "for those formats -- read_file refuses them as binary. For scanned PDFs with "
                "no text layer, this says so, and read_screen is the fallback."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the .pdf or .docx file."},
                    "max_chars": {"type": "integer", "description": "Truncation limit (default 8000)."},
                },
                "required": ["path"],
            },
        },
    },
]
