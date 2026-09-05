"""
test_memory.py — Phase 5: mocked pytest tests for memory.py.

No real chromadb collection, no embedding model download, no disk I/O —
chromadb is faked out via sys.modules before memory.py's lazy
`import chromadb` ever runs. Same "no hardware/heavy-dependency needed
to run the suite" philosophy as every prior phase's tests (FakeModel in
Phase 3, mocked sounddevice/whisper/pyttsx3 in Phase 2).
"""

import importlib
import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_chromadb(monkeypatch):
    """Installs a fake `chromadb` module in sys.modules so `import
    chromadb` inside MemoryStore.__init__ succeeds without the real
    package installed. Returns the fake collection so tests can
    configure return values and assert on calls."""
    fake_collection = MagicMock()
    fake_collection.count.return_value = 0

    fake_client = MagicMock()
    fake_client.get_or_create_collection.return_value = fake_collection

    fake_module = types.ModuleType("chromadb")
    fake_module.PersistentClient = MagicMock(return_value=fake_client)

    monkeypatch.setitem(sys.modules, "chromadb", fake_module)
    return fake_collection


@pytest.fixture
def memory_module(fake_chromadb):
    import memory as m
    importlib.reload(m)  # ensure a fresh module object per test
    return m


@pytest.fixture
def store(memory_module, tmp_path):
    return memory_module.MemoryStore(persist_directory=str(tmp_path))


class TestMemoryStoreInit:
    def test_raises_memory_error_if_chromadb_not_installed(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "chromadb", None)  # forces ImportError
        import memory as m
        importlib.reload(m)

        with pytest.raises(m.MemoryError):
            m.MemoryStore(persist_directory=str(tmp_path))


class TestAddMemory:
    def test_returns_id_and_calls_collection_add(self, store, fake_chromadb):
        memory_id = store.add_memory("my name is Sam")

        assert isinstance(memory_id, str) and memory_id

        fake_chromadb.add.assert_called_once()
        _, kwargs = fake_chromadb.add.call_args
        assert kwargs["documents"] == ["my name is Sam"]
        assert kwargs["ids"] == [memory_id]
        assert "created_at" in kwargs["metadatas"][0]

    def test_merges_custom_metadata(self, store, fake_chromadb):
        store.add_memory("fact", metadata={"source": "manual"})

        _, kwargs = fake_chromadb.add.call_args
        assert kwargs["metadatas"][0]["source"] == "manual"
        assert "created_at" in kwargs["metadatas"][0]  # still auto-added

    def test_empty_text_raises(self, store, memory_module):
        with pytest.raises(memory_module.MemoryError):
            store.add_memory("   ")

    def test_collection_error_raises_memory_error(self, store, fake_chromadb, memory_module):
        fake_chromadb.add.side_effect = RuntimeError("disk full")
        with pytest.raises(memory_module.MemoryError):
            store.add_memory("some fact")


class TestAddMemoryDedup:
    def test_updates_existing_memory_when_near_duplicate(self, store, fake_chromadb):
        fake_chromadb.count.return_value = 1
        fake_chromadb.query.return_value = {
            "ids": [["existing-id"]],
            "distances": [[0.01]],  # well under the default 0.05 threshold
        }
        fake_chromadb.get.return_value = {
            "ids": ["existing-id"],
            "documents": ["my name is Al"],
            "metadatas": [{"created_at": "2026-01-01T00:00:00+00:00"}],
        }

        result_id = store.add_memory("my name is Sam")

        assert result_id == "existing-id"
        fake_chromadb.add.assert_not_called()
        fake_chromadb.update.assert_called_once()
        _, kwargs = fake_chromadb.update.call_args
        assert kwargs["ids"] == ["existing-id"]
        assert kwargs["documents"] == ["my name is Sam"]
        assert kwargs["metadatas"][0]["created_at"] == "2026-01-01T00:00:00+00:00"
        assert "updated_at" in kwargs["metadatas"][0]

    def test_stores_new_when_nearest_match_is_not_close_enough(self, store, fake_chromadb):
        fake_chromadb.count.return_value = 1
        fake_chromadb.query.return_value = {
            "ids": [["existing-id"]],
            "distances": [[0.9]],  # well over the default 0.05 threshold
        }

        result_id = store.add_memory("I'm building a project called Jarvis")

        fake_chromadb.update.assert_not_called()
        fake_chromadb.add.assert_called_once()
        _, kwargs = fake_chromadb.add.call_args
        assert kwargs["ids"] == [result_id]

    def test_skips_dedup_check_when_store_empty(self, store, fake_chromadb):
        fake_chromadb.count.return_value = 0

        store.add_memory("first ever memory")

        fake_chromadb.query.assert_not_called()
        fake_chromadb.add.assert_called_once()

    def test_dedup_threshold_zero_disables_dedup(self, store, fake_chromadb):
        fake_chromadb.count.return_value = 1
        fake_chromadb.query.return_value = {
            "ids": [["existing-id"]],
            "distances": [[0.0]],  # exact match — would dedup by default
        }

        store.add_memory("my name is Sam", dedup_threshold=0)

        fake_chromadb.query.assert_not_called()  # dedup check skipped entirely
        fake_chromadb.add.assert_called_once()
        fake_chromadb.update.assert_not_called()

    def test_dedup_check_failure_falls_back_to_normal_insert(self, store, fake_chromadb):
        fake_chromadb.count.return_value = 1
        fake_chromadb.query.side_effect = RuntimeError("index unavailable")

        # Should not raise — a broken dedup check shouldn't block storing.
        result_id = store.add_memory("some fact")

        fake_chromadb.add.assert_called_once()
        _, kwargs = fake_chromadb.add.call_args
        assert kwargs["ids"] == [result_id]

    def test_update_lookup_failure_raises_memory_error(self, store, fake_chromadb, memory_module):
        fake_chromadb.count.return_value = 1
        fake_chromadb.query.return_value = {
            "ids": [["existing-id"]],
            "distances": [[0.01]],
        }
        fake_chromadb.get.side_effect = RuntimeError("cannot read")

        with pytest.raises(memory_module.MemoryError):
            store.add_memory("my name is Sam")


class TestSearchMemories:
    def test_returns_empty_list_when_store_empty(self, store, fake_chromadb):
        fake_chromadb.count.return_value = 0
        assert store.search_memories("anything") == []
        fake_chromadb.query.assert_not_called()

    def test_returns_empty_list_for_empty_query(self, store, fake_chromadb):
        assert store.search_memories("   ") == []
        fake_chromadb.query.assert_not_called()

    def test_formats_results_sorted_nearest_first(self, store, fake_chromadb):
        fake_chromadb.count.return_value = 2
        fake_chromadb.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["fact one", "fact two"]],
            "metadatas": [[{"created_at": "t1"}, {"created_at": "t2"}]],
            "distances": [[0.1, 0.5]],
        }

        results = store.search_memories("query text", n_results=2)

        assert results == [
            {"id": "id1", "text": "fact one", "metadata": {"created_at": "t1"}, "distance": 0.1},
            {"id": "id2", "text": "fact two", "metadata": {"created_at": "t2"}, "distance": 0.5},
        ]

    def test_n_results_capped_at_store_count(self, store, fake_chromadb):
        fake_chromadb.count.return_value = 1
        fake_chromadb.query.return_value = {
            "ids": [["id1"]],
            "documents": [["only fact"]],
            "metadatas": [[{}]],
            "distances": [[0.2]],
        }

        store.search_memories("query", n_results=5)

        _, kwargs = fake_chromadb.query.call_args
        assert kwargs["n_results"] == 1


class TestGetAllMemories:
    def test_returns_all_stored_memories(self, store, fake_chromadb):
        fake_chromadb.get.return_value = {
            "ids": ["id1", "id2"],
            "documents": ["fact one", "fact two"],
            "metadatas": [{}, {}],
        }

        results = store.get_all_memories()

        assert len(results) == 2
        assert results[0]["text"] == "fact one"
        assert results[1]["id"] == "id2"


class TestDeleteAndClear:
    def test_delete_memory_calls_collection_delete(self, store, fake_chromadb):
        store.delete_memory("id1")
        fake_chromadb.delete.assert_called_once_with(ids=["id1"])

    def test_clear_all_deletes_every_id(self, store, fake_chromadb):
        fake_chromadb.get.return_value = {"ids": ["id1", "id2"]}
        store.clear_all()
        fake_chromadb.delete.assert_called_once_with(ids=["id1", "id2"])

    def test_clear_all_no_op_when_already_empty(self, store, fake_chromadb):
        fake_chromadb.get.return_value = {"ids": []}
        store.clear_all()
        fake_chromadb.delete.assert_not_called()


class TestCount:
    def test_count_returns_collection_count(self, store, fake_chromadb):
        fake_chromadb.count.return_value = 7
        assert store.count() == 7