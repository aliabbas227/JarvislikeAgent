"""
memory.py — Phase 5: persistent, semantic memory store.

Answers exactly two questions:
    1. How do I store a piece of text so it survives across process runs?
    2. How do I retrieve the pieces most relevant to a new query?

Backed by ChromaDB (a local, on-disk vector database) using its bundled
default embedding function (sentence-transformers "all-MiniLM-L6-v2" —
runs locally on CPU or GPU, no API key, downloads once and caches).
This matches the project's local-first, upgrade-when-needed philosophy
— swap embedding backends later (e.g. a hosted embeddings API) only if
local quality turns out to be the bottleneck, same pattern as
jarvis.py's LLM_BACKEND seam.

Deliberately does NOT decide *what* is worth remembering — that's a
prompt/product design problem left to the caller (memory_chat.py). This
module is just the storage/retrieval primitive.
"""

import os
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("memory")

# Deliberately NOT a relative path like "./chroma_store" by default.
# Manual testing found that running memory_chat.py from a different
# working directory (e.g. `python phaseFive\memory_chat.py` from
# PROJECTS\JARVIS instead of from inside phaseFive\) silently resolved
# to a different, empty store instead of erroring — "0 memories
# stored" with no warning, easy to miss. Anchoring the default to this
# file's own location makes the store's location independent of cwd.
# CHROMA_DB_PATH in .env still overrides this if you want it elsewhere.
_DEFAULT_CHROMA_DB_PATH = str(Path(__file__).resolve().parent / "chroma_store")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", _DEFAULT_CHROMA_DB_PATH)
COLLECTION_NAME = os.getenv("MEMORY_COLLECTION", "jarvis_memories")
DEFAULT_N_RESULTS = int(os.getenv("MEMORY_RESULTS", 5))

# Distance below which a new memory is treated as a near-duplicate of
# an existing one and updates it in place instead of appending a copy.
# Deliberately tight — this is for catching "the same fact, restated or
# typo-fixed" (e.g. "remember my name is Sam" said twice), not for
# catching "a related fact" (e.g. "my name is Sam" vs. "I go by Sam at
# work" should both be kept — they're different information, even
# though they'd score close together). 0 (or any non-positive value)
# disables dedup entirely.
DEDUP_DISTANCE_THRESHOLD = float(os.getenv("DEDUP_DISTANCE_THRESHOLD", 0.05))


class MemoryError(Exception):
    """Raised when the memory store can't be opened, written to, or read."""


class MemoryStore:
    """
    Thin wrapper around a persistent Chroma collection.

    chromadb is imported lazily inside __init__ (not at module load
    time) for the same reason voice_io.py lazily imports whisper and
    pyttsx3 in Phase 2: keeps `import memory` cheap, and lets code that
    only needs the *interface* (like tests, via a faked-out chromadb
    module) run without the real package installed.
    """

    def __init__(self, persist_directory: str = None, collection_name: str = None):
        try:
            import chromadb
        except ImportError as e:
            raise MemoryError(
                "chromadb is not installed. Run: pip install chromadb"
            ) from e

        self.persist_directory = persist_directory or CHROMA_DB_PATH
        self.collection_name = collection_name or COLLECTION_NAME

        try:
            self._client = chromadb.PersistentClient(path=self.persist_directory)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name
            )
        except Exception as e:
            raise MemoryError(f"Could not open memory store: {e}") from e

    def add_memory(self, text: str, metadata: dict = None, dedup_threshold: float = None) -> str:
        """
        Stores a single fact/memory. Returns the memory's id.

        Before storing, checks the single nearest existing memory (if
        any); if it's within dedup_threshold distance, this updates
        that memory in place (new text, refreshed metadata) and returns
        its existing id, rather than appending a near-duplicate. Pass
        dedup_threshold=0 to force a plain append (e.g. for tests, or
        if you deliberately want duplicates).

        dedup_threshold defaults to the module-level
        DEDUP_DISTANCE_THRESHOLD when not given explicitly.
        """
        text = (text or "").strip()
        if not text:
            raise MemoryError("Refusing to store an empty memory.")

        threshold = DEDUP_DISTANCE_THRESHOLD if dedup_threshold is None else dedup_threshold

        if threshold and threshold > 0:
            existing_id = self._find_near_duplicate(text, threshold)
            if existing_id:
                return self._update_memory(existing_id, text, metadata)

        return self._insert_memory(text, metadata)

    def _find_near_duplicate(self, text: str, threshold: float) -> str:
        """Returns the id of the nearest existing memory if it's within
        threshold distance of text, else None. Returns None (rather
        than raising) on any lookup failure — a failed dedup check
        should never block a legitimate store, it should just fall
        back to a normal append."""
        try:
            if self._collection.count() == 0:
                return None
            results = self._collection.query(query_texts=[text], n_results=1)
        except Exception as e:
            logger.warning(f"Dedup check failed, storing as a new memory instead: {e}")
            return None

        ids = results.get("ids", [[]])[0]
        dists = results.get("distances", [[]])[0]
        if ids and dists and dists[0] <= threshold:
            return ids[0]
        return None

    def _update_memory(self, memory_id: str, text: str, metadata: dict = None) -> str:
        """Overwrites an existing memory's text in place (used when
        add_memory detects a near-duplicate). Preserves the original
        created_at and adds/refreshes updated_at, rather than treating
        this as a brand-new memory."""
        try:
            existing = self._collection.get(ids=[memory_id])
            existing_meta = (existing.get("metadatas") or [{}])[0] or {}
        except Exception as e:
            raise MemoryError(f"Could not read existing memory {memory_id}: {e}") from e

        meta = dict(existing_meta)
        meta["updated_at"] = datetime.now(timezone.utc).isoformat()
        if metadata:
            meta.update(metadata)

        try:
            self._collection.update(ids=[memory_id], documents=[text], metadatas=[meta])
        except Exception as e:
            raise MemoryError(f"Could not update memory {memory_id}: {e}") from e

        logger.info(f"Near-duplicate detected; updated existing memory {memory_id} instead of storing a copy: {text!r}")
        return memory_id

    def _insert_memory(self, text: str, metadata: dict = None) -> str:
        memory_id = str(uuid.uuid4())
        meta = {"created_at": datetime.now(timezone.utc).isoformat()}
        if metadata:
            meta.update(metadata)

        try:
            self._collection.add(ids=[memory_id], documents=[text], metadatas=[meta])
        except Exception as e:
            raise MemoryError(f"Could not store memory: {e}") from e

        logger.info(f"Stored memory {memory_id}: {text!r}")
        return memory_id

    def search_memories(self, query: str, n_results: int = None) -> list:
        """
        Returns the n_results memories most semantically similar to
        query, each as {"id", "text", "metadata", "distance"}, sorted
        nearest first. Returns an empty list if the store has nothing
        yet, or if query is empty — never raises just because there's
        no data to search.
        """
        query = (query or "").strip()
        if not query:
            return []

        n_results = n_results or DEFAULT_N_RESULTS

        try:
            count = self._collection.count()
        except Exception as e:
            raise MemoryError(f"Could not read memory store: {e}") from e

        if count == 0:
            return []

        n_results = min(n_results, count)

        try:
            results = self._collection.query(query_texts=[query], n_results=n_results)
        except Exception as e:
            raise MemoryError(f"Memory search failed: {e}") from e

        out = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for i in range(len(ids)):
            out.append(
                {"id": ids[i], "text": docs[i], "metadata": metas[i], "distance": dists[i]}
            )
        return out

    def get_all_memories(self) -> list:
        """Returns every stored memory, unranked (no query). Useful for
        debugging/inspection — not the retrieval path used for RAG."""
        try:
            results = self._collection.get()
        except Exception as e:
            raise MemoryError(f"Could not list memories: {e}") from e

        out = []
        ids = results.get("ids", [])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        for i in range(len(ids)):
            out.append({"id": ids[i], "text": docs[i], "metadata": metas[i]})
        return out

    def delete_memory(self, memory_id: str) -> None:
        try:
            self._collection.delete(ids=[memory_id])
        except Exception as e:
            raise MemoryError(f"Could not delete memory {memory_id}: {e}") from e

    def clear_all(self) -> None:
        """Deletes every memory in this collection. Destructive — callers
        (the CLI) should confirm with the user before calling this."""
        try:
            all_ids = self._collection.get().get("ids", [])
            if all_ids:
                self._collection.delete(ids=all_ids)
        except Exception as e:
            raise MemoryError(f"Could not clear memory store: {e}") from e

    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception as e:
            raise MemoryError(f"Could not count memories: {e}") from e