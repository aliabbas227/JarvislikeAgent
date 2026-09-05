"""
test_memory_chat.py — Phase 5: tests for memory_chat.py's pure-logic
helpers (no LLM call, no memory store, no phaseOne import needed —
that's exactly why those imports are lazy inside run_chat_loop(), see
the comment at the top of memory_chat.py).
"""

from memory_chat import (
    build_augmented_message,
    find_memory_by_prefix,
    format_memory_line,
    is_remember_command,
    strip_remember_prefix,
)


class FakeStore:
    """Minimal stand-in for MemoryStore — find_memory_by_prefix only
    calls get_all_memories(), so that's all this needs to provide.
    Avoids pulling in the chromadb-mocking fixtures from test_memory.py
    just to test id-prefix matching logic."""

    def __init__(self, memories):
        self._memories = memories

    def get_all_memories(self):
        return self._memories


class TestIsRememberCommand:
    def test_matches_remember_space(self):
        assert is_remember_command("remember my name is Sam")

    def test_matches_remember_colon(self):
        assert is_remember_command("remember: my name is Sam")

    def test_matches_remember_that(self):
        assert is_remember_command("remember that I like coffee")

    def test_case_insensitive(self):
        assert is_remember_command("REMEMBER my birthday is in May")

    def test_does_not_match_unrelated_text(self):
        assert not is_remember_command("what do you remember about me?")

    def test_does_not_match_empty_string(self):
        assert not is_remember_command("")


class TestStripRememberPrefix:
    def test_strips_remember_space(self):
        assert strip_remember_prefix("remember my name is Sam") == "my name is Sam"

    def test_strips_remember_colon(self):
        assert strip_remember_prefix("remember: my name is Sam") == "my name is Sam"

    def test_strips_remember_that(self):
        assert strip_remember_prefix("remember that I like coffee") == "I like coffee"

    def test_returns_stripped_text_unchanged_if_no_prefix(self):
        assert strip_remember_prefix("  just a normal message  ") == "just a normal message"


class TestBuildAugmentedMessage:
    def test_returns_plain_input_when_nothing_retrieved(self):
        assert build_augmented_message("hello", []) == "hello"

    def test_folds_memories_into_preamble(self):
        retrieved = [{"text": "my name is Sam"}, {"text": "I'm building Jarvis"}]
        result = build_augmented_message("what's my name?", retrieved)

        assert "my name is Sam" in result
        assert "I'm building Jarvis" in result
        assert "what's my name?" in result
        # Original message must still be recoverable/readable at the end,
        # since that's the part the model should treat as the actual ask.
        assert result.endswith("what's my name?")


class TestFormatMemoryLine:
    def test_shows_short_id_and_text(self):
        mem = {"id": "abcdef12-3456-7890-abcd-ef1234567890", "text": "my name is Sam"}
        assert format_memory_line(mem) == "[abcdef12] my name is Sam"


class TestFindMemoryByPrefix:
    def test_finds_single_match(self):
        store = FakeStore([
            {"id": "abcdef12-0000", "text": "fact one"},
            {"id": "112233ab-0000", "text": "fact two"},
        ])
        matches = find_memory_by_prefix(store, "abcdef")
        assert len(matches) == 1
        assert matches[0]["text"] == "fact one"

    def test_case_insensitive(self):
        store = FakeStore([{"id": "ABCDEF12-0000", "text": "fact one"}])
        matches = find_memory_by_prefix(store, "abcdef")
        assert len(matches) == 1

    def test_returns_multiple_on_ambiguous_prefix(self):
        store = FakeStore([
            {"id": "abc11111-0000", "text": "fact one"},
            {"id": "abc22222-0000", "text": "fact two"},
        ])
        matches = find_memory_by_prefix(store, "abc")
        assert len(matches) == 2

    def test_returns_empty_list_for_no_match(self):
        store = FakeStore([{"id": "abcdef12-0000", "text": "fact one"}])
        assert find_memory_by_prefix(store, "zzz") == []

    def test_returns_empty_list_for_empty_prefix(self):
        store = FakeStore([{"id": "abcdef12-0000", "text": "fact one"}])
        assert find_memory_by_prefix(store, "   ") == []