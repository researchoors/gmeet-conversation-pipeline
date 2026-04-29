"""Tests for gmeet_pipeline.state."""

import asyncio
from typing import Optional

import pytest

from gmeet_pipeline.state import BotSession, BotRegistry


class TestBotSession:
    """Test BotSession creation with defaults."""

    def test_default_status(self):
        s = BotSession(bot_id="b1", meeting_url="https://meet.google.com/test")
        assert s.status == "joining"

    def test_default_speaking(self):
        s = BotSession(bot_id="b1", meeting_url="https://meet.google.com/test")
        assert s.speaking is False

    def test_default_transcript(self):
        s = BotSession(bot_id="b1", meeting_url="https://meet.google.com/test")
        assert s.transcript == []

    def test_default_conversation(self):
        s = BotSession(bot_id="b1", meeting_url="https://meet.google.com/test")
        assert s.conversation == []

    def test_default_last_processed_ts(self):
        s = BotSession(bot_id="b1", meeting_url="https://meet.google.com/test")
        assert s.last_processed_ts == ""

    def test_default_expanded_entries(self):
        s = BotSession(bot_id="b1", meeting_url="https://meet.google.com/test")
        assert s.expanded_entries == set()

    def test_default_created_at(self):
        s = BotSession(bot_id="b1", meeting_url="https://meet.google.com/test")
        assert s.created_at  # not empty

    def test_respond_lock_is_asyncio_lock(self):
        s = BotSession(bot_id="b1", meeting_url="https://meet.google.com/test")
        assert isinstance(s.respond_lock, asyncio.Lock)

    def test_custom_values(self):
        s = BotSession(
            bot_id="b2",
            meeting_url="https://meet.google.com/abc",
            status="active",
            speaking=True,
        )
        assert s.status == "active"
        assert s.speaking is True


@pytest.mark.asyncio
class TestBotRegistryCreate:
    """Test BotRegistry.create()."""

    async def test_create_adds_session(self):
        registry = BotRegistry()
        session = await registry.create("bot-1", "https://meet.google.com/test")
        assert isinstance(session, BotSession)
        assert session.bot_id == "bot-1"
        assert session.meeting_url == "https://meet.google.com/test"
        assert "bot-1" in registry

    async def test_create_with_kwargs(self):
        registry = BotRegistry()
        session = await registry.create(
            "bot-2", "https://meet.google.com/test", status="active"
        )
        assert session.status == "active"


@pytest.mark.asyncio
class TestBotRegistryGet:
    """Test BotRegistry.get()."""

    async def test_get_returns_session(self):
        registry = BotRegistry()
        await registry.create("bot-x", "https://meet.google.com/test")
        session = registry.get("bot-x")
        assert session is not None
        assert session.bot_id == "bot-x"

    async def test_get_returns_none_for_missing(self):
        registry = BotRegistry()
        result = registry.get("nonexistent")
        assert result is None


@pytest.mark.asyncio
class TestBotRegistryRemove:
    """Test BotRegistry.remove()."""

    async def test_remove_returns_session(self):
        registry = BotRegistry()
        await registry.create("bot-r", "https://meet.google.com/test")
        removed = await registry.remove("bot-r")
        assert removed is not None
        assert removed.bot_id == "bot-r"
        assert "bot-r" not in registry

    async def test_remove_returns_none_for_missing(self):
        registry = BotRegistry()
        result = await registry.remove("nonexistent")
        assert result is None


class TestBotRegistryContains:
    """Test BotRegistry.__contains__."""

    def test_contains_existing(self, mock_registry):
        assert "test-bot-001" in mock_registry

    def test_contains_missing(self, mock_registry):
        assert "nonexistent" not in mock_registry


@pytest.mark.asyncio
class TestBotRegistryListBots:
    """Test BotRegistry.list_bots() returns serializable dict."""

    async def test_list_bots_returns_dict(self, mock_registry):
        result = await mock_registry.list_bots()
        assert isinstance(result, dict)
        assert "test-bot-001" in result

    async def test_list_bots_has_expected_keys(self, mock_registry):
        result = await mock_registry.list_bots()
        bot_data = result["test-bot-001"]
        expected_keys = {
            "bot_id", "meeting_url", "status", "speaking",
            "pipeline_state", "last_llm_ms", "last_tts_ms", "last_total_ms",
            "queue_depth", "participants",
            "last_processed_ts", "transcript_count",
            "conversation_count", "created_at",
        }
        assert set(bot_data.keys()) == expected_keys

    async def test_list_bots_no_lock_objects(self, mock_registry):
        """list_bots should not include any Lock objects (not serializable)."""
        import json
        result = await mock_registry.list_bots()
        # Should be JSON-serializable
        serialized = json.dumps(result)
        assert serialized  # no exception raised

    async def test_list_bots_empty_registry(self):
        registry = BotRegistry()
        result = await registry.list_bots()
        assert result == {}

    async def test_list_bots_transcript_count(self, mock_registry):
        session = mock_registry.get("test-bot-001")
        session.transcript.append({"speaker": "Alice", "text": "hello"})
        result = await mock_registry.list_bots()
        assert result["test-bot-001"]["transcript_count"] == 1
