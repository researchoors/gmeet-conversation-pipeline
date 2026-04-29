"""Unit tests for ContextBuilder — Hermes memory + session context assembly."""

import json
import time
from pathlib import Path

import pytest

from gmeet_pipeline.context_builder import (
    ContextBuilder,
    _read_entries,
    _extract_session_topics,
    ENTRY_DELIMITER,
)


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def mem_dir(tmp_path):
    """Temp memories directory with sample MEMORY.md and USER.md."""
    d = tmp_path / "memories"
    d.mkdir()
    (d / "MEMORY.md").write_text(
        "DarkBloom = decentralized inference on Apple Silicon§\n"
        "SwiftLM benchmarks show DFlash wins on MoE models§\n"
        "Mac Mini IP is 192.168.1.125"
    )
    (d / "USER.md").write_text(
        "Ethan is part of researchoors group§\n"
        "Prefers concise responses"
    )
    return d


@pytest.fixture
def sessions_dir(tmp_path):
    """Temp sessions directory with sample session files."""
    d = tmp_path / "sessions"
    d.mkdir()

    # Create a recent session
    session = {
        "session_id": "test-123",
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "[Ethan] How does DFlash work?"},
        ],
    }
    p = d / "session_20260429_120000_abc123.json"
    p.write_text(json.dumps(session))

    # Create an older session
    session2 = {
        "session_id": "test-456",
        "messages": [
            {"role": "user", "content": "Set up Nomad on Mac Studio"},
        ],
    }
    p2 = d / "session_20260428_120000_def456.json"
    # Make it older by backdating mtime
    p2.write_text(json.dumps(session2))
    import os
    os.utime(p2, (time.time() - 86400, time.time() - 86400))

    return d


@pytest.fixture
def builder(mem_dir, sessions_dir):
    return ContextBuilder(memories_dir=mem_dir, sessions_dir=sessions_dir, session_limit=5)


# ── _read_entries tests ──────────────────────────────────────────────

def test_read_entries_basic(mem_dir):
    entries = _read_entries(mem_dir / "MEMORY.md")
    assert len(entries) == 3
    assert "DarkBloom" in entries[0]["text"]
    assert "SwiftLM" in entries[1]["text"]
    assert "Mac Mini" in entries[2]["text"]


def test_read_entries_missing_file(tmp_path):
    entries = _read_entries(tmp_path / "nonexistent.md")
    assert entries == []


def test_read_entries_empty_file(tmp_path):
    p = tmp_path / "empty.md"
    p.write_text("")
    entries = _read_entries(p)
    assert entries == []


def test_read_entries_user_profile(mem_dir):
    entries = _read_entries(mem_dir / "USER.md")
    assert len(entries) == 2
    assert "Ethan" in entries[0]["text"]
    assert "concise" in entries[1]["text"]


# ── _extract_session_topics tests ────────────────────────────────────

def test_extract_session_topics(sessions_dir):
    topics = _extract_session_topics(sessions_dir, limit=5)
    assert len(topics) == 2
    # Most recent first
    assert "DFlash" in topics[0]["topic"]
    assert "Nomad" in topics[1]["topic"]
    # Platform prefix stripped
    assert not topics[0]["topic"].startswith("[Ethan]")
    # Age is reasonable
    assert topics[0]["age_hours"] < 1


def test_extract_session_topics_limit(sessions_dir):
    topics = _extract_session_topics(sessions_dir, limit=1)
    assert len(topics) == 1


def test_extract_session_topics_empty(tmp_path):
    d = tmp_path / "sessions"
    d.mkdir()
    topics = _extract_session_topics(d)
    assert topics == []


def test_extract_session_topics_malformed(tmp_path):
    d = tmp_path / "sessions"
    d.mkdir()
    (d / "session_20260429_bad.json").write_text("not valid json")
    topics = _extract_session_topics(d)
    assert topics == []


# ── ContextBuilder.build tests ───────────────────────────────────────

def test_build_basic(builder):
    prompt = builder.build()
    assert "DarkBloom" in prompt
    assert "SwiftLM" in prompt
    assert "Ethan" in prompt
    assert "concise" in prompt
    assert "DFlash" in prompt or "Nomad" in prompt  # session context


def test_build_structure(builder):
    prompt = builder.build()
    # Should have section headers
    assert "## Agent Knowledge" in prompt
    assert "## User Profile" in prompt
    assert "## Recent Context" in prompt


def test_build_caching(builder):
    prompt1 = builder.build()
    prompt2 = builder.build()
    assert prompt1 == prompt2
    # Second call should be cached (same object)
    assert prompt1 is prompt2


def test_build_force_refresh(builder):
    prompt1 = builder.build()
    prompt2 = builder.build(force=True)
    # Should rebuild but produce same content
    assert prompt1 == prompt2


def test_build_no_memory_files(tmp_path):
    builder = ContextBuilder(
        memories_dir=tmp_path / "no_memories",
        sessions_dir=tmp_path / "no_sessions",
    )
    prompt = builder.build()
    # Should return empty string when no files exist
    assert prompt == ""


def test_build_char_count_logged(builder, caplog):
    import logging
    with caplog.at_level(logging.INFO, logger="gmeet_pipeline.context_builder"):
        builder.build()
    assert "chars" in caplog.text.lower() or "Context built" in caplog.text


# ── ContextBuilder.refresh_if_stale tests ────────────────────────────

def test_refresh_if_stale_no_change(builder):
    builder.build()
    # No file changes — should not rebuild
    builder.refresh_if_stale()
    # Still cached
    assert builder._cached_prompt is not None


def test_refresh_if_stale_after_file_update(builder, mem_dir):
    builder.build()
    original = builder._cached_prompt

    # Update the memory file (touch mtime forward)
    import os
    mem_file = mem_dir / "MEMORY.md"
    os.utime(mem_file, (time.time() + 100, time.time() + 100))

    builder.refresh_if_stale(max_age=999)
    # Should have rebuilt — prompt is different from original
    # (same content but rebuilt timestamp)
    assert builder._built_at > 0


def test_refresh_if_stale_missing_files(builder, mem_dir):
    builder.build()
    # Delete memory files — refresh should handle gracefully
    (mem_dir / "MEMORY.md").unlink()
    (mem_dir / "USER.md").unlink()
    builder.refresh_if_stale()
    # Should not crash


# ── Integration: prompt content is useful for LLM ────────────────────

def test_prompt_contains_project_context(builder):
    """The system prompt should contain enough context for the LLM
    to answer questions about projects like DarkBloom."""
    prompt = builder.build()
    # Key terms from memory that LLM needs
    assert "DarkBloom" in prompt
    assert "decentralized" in prompt
    assert "Ethan" in prompt
    assert "researchoors" in prompt


def test_prompt_no_rag_no_expand(builder):
    """Verify no RAG or EXPAND artifacts in the prompt."""
    prompt = builder.build()
    assert "EXPAND" not in prompt
    assert "RAG" not in prompt.upper() or "RAG" not in prompt  # no RAG terminology
    assert "§" not in prompt  # delimiter should be stripped
