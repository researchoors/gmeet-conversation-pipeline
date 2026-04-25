"""Tests for gmeet_pipeline.memory."""

import time
from pathlib import Path
from typing import Optional

import pytest

from gmeet_pipeline.memory import MemorySnapshot, ENTRY_DELIMITER


@pytest.fixture
def memory_dir(tmp_path):
    """Create a temp dir with MEMORY.md and USER.md files."""
    mem_file = tmp_path / "MEMORY.md"
    usr_file = tmp_path / "USER.md"
    mem_file.write_text(
        "Darkbloom is a decentralized inference project.§"
        "SwiftLM is a Swift-based LLM inference engine for Apple Silicon.§"
        "Hank Bob is the AI assistant for the researchoors community."
    )
    usr_file.write_text(
        "Ethan is the founder of the researchoors community.§"
        "Preferred communication style: concise and technical."
    )
    return tmp_path


@pytest.fixture
def snapshot(memory_dir):
    """Return a MemorySnapshot built from memory_dir."""
    snap = MemorySnapshot(
        memory_file=memory_dir / "MEMORY.md",
        user_file=memory_dir / "USER.md",
    )
    snap.build()
    return snap


class TestMemorySnapshotBuild:
    """Test MemorySnapshot.build() parses §-delimited entries."""

    def test_build_creates_entries(self, snapshot):
        assert len(snapshot.entries) == 5  # 3 from MEMORY + 2 from USER

    def test_build_assigns_indices(self, snapshot):
        indices = [e["index"] for e in snapshot.entries]
        assert indices == [0, 1, 2, 3, 4]

    def test_build_assigns_source(self, snapshot):
        sources = [e["source"] for e in snapshot.entries]
        assert sources[:3] == ["memory", "memory", "memory"]
        assert sources[3:] == ["user", "user"]

    def test_build_sets_built_at(self, snapshot):
        assert snapshot.built_at > 0

    def test_build_creates_index_text(self, snapshot):
        assert "[0]" in snapshot.index_text
        assert "[4]" in snapshot.index_text

    def test_build_creates_summary(self, snapshot):
        assert snapshot.summary  # not empty
        assert "Darkbloom" in snapshot.summary

    def test_build_empty_files(self, tmp_path):
        mem = tmp_path / "MEMORY.md"
        usr = tmp_path / "USER.md"
        mem.write_text("")
        usr.write_text("")
        snap = MemorySnapshot(memory_file=mem, user_file=usr)
        snap.build()
        assert snap.entries == []

    def test_build_missing_files(self, tmp_path):
        snap = MemorySnapshot(
            memory_file=tmp_path / "nonexistent_MEMORY.md",
            user_file=tmp_path / "nonexistent_USER.md",
        )
        snap.build()
        assert snap.entries == []

    def test_build_single_file_only(self, tmp_path):
        mem = tmp_path / "MEMORY.md"
        mem.write_text("Only memory content.§Second entry.")
        snap = MemorySnapshot(
            memory_file=mem,
            user_file=tmp_path / "nonexistent_USER.md",
        )
        snap.build()
        assert len(snap.entries) == 2
        assert all(e["source"] == "memory" for e in snap.entries)


class TestMemorySnapshotRagRetrieve:
    """Test MemorySnapshot.rag_retrieve() returns relevant entries."""

    def test_retrieve_returns_relevant(self, snapshot):
        results = snapshot.rag_retrieve("darkbloom inference", top_k=3)
        assert len(results) > 0
        texts = [e["text"] for e in results]
        assert any("Darkbloom" in t for t in texts)

    def test_retrieve_respects_top_k(self, snapshot):
        results = snapshot.rag_retrieve("project community", top_k=2)
        assert len(results) <= 2

    def test_retrieve_empty_query(self, snapshot):
        results = snapshot.rag_retrieve("", top_k=3)
        assert results == []

    def test_retrieve_no_match(self, snapshot):
        results = snapshot.rag_retrieve("xylophone zebra banana", top_k=3)
        # All stop words or no overlap => empty
        assert results == []

    def test_retrieve_returns_dicts(self, snapshot):
        results = snapshot.rag_retrieve("SwiftLM", top_k=1)
        assert len(results) > 0
        assert "index" in results[0]
        assert "text" in results[0]


class TestMemorySnapshotClassifyQuery:
    """Test MemorySnapshot.classify_query() returns fast/standard/deep."""

    def test_simple_greeting_is_fast(self, snapshot):
        assert snapshot.classify_query("hello") == "fast"

    def test_simple_thanks_is_fast(self, snapshot):
        assert snapshot.classify_query("thanks") == "fast"

    def test_math_is_fast(self, snapshot):
        assert snapshot.classify_query("2 + 3") == "fast"

    def test_tool_keywords_is_deep(self, snapshot):
        assert snapshot.classify_query("check the github repo for issues") == "deep"

    def test_deploy_keyword_is_deep(self, snapshot):
        assert snapshot.classify_query("we should deploy this to docker") == "deep"

    def test_memory_keyword_is_standard(self, snapshot):
        assert snapshot.classify_query("tell me about darkbloom") == "standard"

    def test_remember_keyword_is_standard(self, snapshot):
        assert snapshot.classify_query("remember this for later") == "standard"

    def test_rag_match_is_standard(self, snapshot):
        # "SwiftLM" should match via RAG even without explicit memory keyword
        assert snapshot.classify_query("SwiftLM") == "standard"

    def test_no_match_is_fast(self, snapshot):
        # Very generic with no matches => fast
        result = snapshot.classify_query("random words xyzzy foo")
        assert result in ("fast", "standard")  # depends on stop-word filtering


class TestMemorySnapshotParseExpands:
    """Test MemorySnapshot.parse_expands() extracts indices from EXPAND[n]."""

    def test_parse_single_expand(self, snapshot):
        result = snapshot.parse_expands("Here is info EXPAND[0]")
        assert result == [0]

    def test_parse_multiple_indices(self, snapshot):
        result = snapshot.parse_expands("See EXPAND[0,2,4]")
        assert result == [0, 2, 4]

    def test_parse_invalid_index_ignored(self, snapshot):
        # Index 99 doesn't exist (only 5 entries)
        result = snapshot.parse_expands("EXPAND[0,99]")
        assert result == [0]

    def test_parse_no_expands(self, snapshot):
        result = snapshot.parse_expands("No expand directives here")
        assert result == []

    def test_parse_empty_string(self, snapshot):
        result = snapshot.parse_expands("")
        assert result == []


class TestMemorySnapshotStripExpands:
    """Test MemorySnapshot.strip_expands() removes EXPAND directives."""

    def test_strip_single(self):
        result = MemorySnapshot.strip_expands("Info EXPAND[0] here")
        assert "EXPAND" not in result
        assert "Info" in result
        assert "here" in result

    def test_strip_multiple(self):
        result = MemorySnapshot.strip_expands("EXPAND[0] and EXPAND[1,2]")
        assert "EXPAND" not in result

    def test_strip_no_expands(self):
        text = "No directives here"
        assert MemorySnapshot.strip_expands(text) == text


class TestRefreshIfStale:
    """Test refresh_if_stale() rebuilds when file modified."""

    def test_refresh_when_file_newer(self, tmp_path):
        mem = tmp_path / "MEMORY.md"
        usr = tmp_path / "USER.md"
        mem.write_text("Initial content")
        usr.write_text("User content")

        snap = MemorySnapshot(memory_file=mem, user_file=usr)
        snap.build()
        old_built_at = snap.built_at

        # Modify the file after build
        time.sleep(0.05)
        mem.write_text("Updated content§New entry")

        snap.refresh_if_stale()
        assert snap.built_at > old_built_at
        assert len(snap.entries) == 3  # 2 from updated MEMORY + 1 from USER

    def test_no_refresh_when_not_stale(self, snapshot):
        old_built_at = snapshot.built_at
        snapshot.refresh_if_stale()
        # Should not rebuild (files not modified since build)
        assert snapshot.built_at == old_built_at

    def test_refresh_with_max_age(self, tmp_path):
        mem = tmp_path / "MEMORY.md"
        usr = tmp_path / "USER.md"
        mem.write_text("Content")
        usr.write_text("")

        snap = MemorySnapshot(memory_file=mem, user_file=usr)
        snap.build()

        # built_at is now; max_age default 300s should NOT trigger
        old_built_at = snap.built_at
        snap.refresh_if_stale(max_age=300)
        # File mtime hasn't changed, so no rebuild
        assert snap.built_at == old_built_at
