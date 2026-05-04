"""Tests for gmeet_pipeline.webhook."""

import asyncio
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gmeet_pipeline.webhook import RecallWebhookHandler
from gmeet_pipeline.state import BotRegistry, BotSession
from gmeet_pipeline.llm.base import BaseLLM
from gmeet_pipeline.tts.base import BaseTTS
from gmeet_pipeline.ws_manager import ConnectionManager


# Concrete stubs for abstract base classes
class StubLLM(BaseLLM):
    async def generate(self, conversation, message, bot_state=None):
        return "Test response"


class StubTTS(BaseTTS):
    async def generate(self, text, bot_id):
        return "test_audio.wav"


@pytest.fixture
def registry():
    reg = BotRegistry()
    session = BotSession(
        bot_id="bot-001",
        meeting_url="https://meet.google.com/test",
        status="in_meeting",
    )
    reg._bots["bot-001"] = session
    return reg


@pytest.fixture
def handler(registry):
    llm = StubLLM()
    tts = StubTTS()
    ws_mgr = ConnectionManager()
    return RecallWebhookHandler(
        registry=registry,
        llm=llm,
        tts=tts,
        ws_manager=ws_mgr,
        audio_queue={},
    )


@pytest.mark.asyncio
class TestHandleStatusChange:
    """Test _handle_status_change updates session status."""

    async def test_updates_session_status(self, handler, registry):
        data = {
            "status": {"code": "in_meeting"},
        }
        await handler._handle_status_change("bot-001", data)
        session = registry.get("bot-001")
        assert session.status == "in_meeting"

    async def test_updates_to_ended(self, handler, registry):
        data = {
            "status": {"code": "ended"},
        }
        await handler._handle_status_change("bot-001", data)
        session = registry.get("bot-001")
        assert session.status == "ended"

    async def test_unknown_bot_does_not_crash(self, handler):
        data = {"status": {"code": "in_meeting"}}
        # Should not raise
        await handler._handle_status_change("unknown-bot", data)

    async def test_string_status(self, handler, registry):
        data = {"status": "ready"}
        await handler._handle_status_change("bot-001", data)
        session = registry.get("bot-001")
        assert session.status == "ready"


@pytest.mark.asyncio
class TestHandleTranscript:
    """Test _handle_transcript fires process_and_respond."""

    async def test_transcript_adds_to_session(self, handler, registry):
        data = {
            "data": {
                "participant": {"name": "Alice"},
                "words": [{"text": "Hello"}, {"text": "there"}],
                "started_at": "2025-01-01T00:00:00Z",
            }
        }
        with patch.object(handler, "_ensure_worker"):
            await handler._handle_transcript("bot-001", data)

        session = registry.get("bot-001")
        assert len(session.transcript) == 1
        assert session.transcript[0]["speaker"] == "Alice"
        assert session.transcript[0]["text"] == "Hello there"
        assert session.response_queue.qsize() == 1

    async def test_transcript_skips_hank_bob_as_speaker(self, handler, registry):
        data = {
            "data": {
                "participant": {"name": "Hank Bob"},
                "words": [{"text": "I"}, {"text": "speak"}],
                "started_at": "2025-01-01T00:00:00Z",
            }
        }
        with patch.object(handler, "_ensure_worker"):
            await handler._handle_transcript("bot-001", data)

        session = registry.get("bot-001")
        assert len(session.transcript) == 0  # skipped
        assert session.response_queue.qsize() == 0

    async def test_transcript_skips_hank_in_name(self, handler, registry):
        data = {
            "data": {
                "participant": {"name": "Hank"},
                "words": [{"text": "Hi"}],
                "started_at": "2025-01-01T00:00:00Z",
            }
        }
        with patch.object(handler, "_ensure_worker"):
            await handler._handle_transcript("bot-001", data)

        session = registry.get("bot-001")
        assert len(session.transcript) == 0

    async def test_transcript_skips_unknown_bot(self, handler):
        data = {
            "data": {
                "participant": {"name": "Alice"},
                "words": [{"text": "Hello"}],
                "started_at": "2025-01-01T00:00:00Z",
            }
        }
        # Should not raise, should return early
        await handler._handle_transcript("unknown-bot", data)

    async def test_transcript_skips_empty_text(self, handler, registry):
        data = {
            "data": {
                "participant": {"name": "Alice"},
                "words": [],
                "started_at": "2025-01-01T00:00:00Z",
            }
        }
        with patch.object(handler, "_ensure_worker"):
            await handler._handle_transcript("bot-001", data)

        session = registry.get("bot-001")
        assert len(session.transcript) == 0

    async def test_silence_command_switches_mode_without_queueing(self, handler, registry):
        data = {
            "data": {
                "participant": {"name": "Alice"},
                "words": [{"text": "Hank"}, {"text": "Bob"}, {"text": "quiet"}],
                "started_at": "2025-01-01T00:00:00Z",
            }
        }
        with patch.object(handler, "_ensure_worker") as ensure_worker:
            await handler._handle_transcript("bot-001", data)

        session = registry.get("bot-001")
        assert session.response_mode == "silent_transcribe"
        assert session.response_queue.qsize() == 0
        assert session.mode_events[-1]["mode"] == "silent_transcribe"
        ensure_worker.assert_not_called()

    async def test_silent_mode_records_but_does_not_queue(self, handler, registry):
        session = registry.get("bot-001")
        session.response_mode = "silent_transcribe"
        data = {
            "data": {
                "participant": {"name": "Alice"},
                "words": [{"text": "keep"}, {"text": "transcribing"}],
                "started_at": "2025-01-01T00:00:01Z",
            }
        }
        with patch.object(handler, "_ensure_worker") as ensure_worker:
            await handler._handle_transcript("bot-001", data)

        assert len(session.transcript) == 1
        assert session.response_queue.qsize() == 0
        ensure_worker.assert_not_called()

    async def test_wake_command_reactivates_and_queues(self, handler, registry):
        session = registry.get("bot-001")
        session.response_mode = "silent_transcribe"
        data = {
            "data": {
                "participant": {"name": "Alice"},
                "words": [{"text": "Hey"}, {"text": "Hank"}, {"text": "Bob"}],
                "started_at": "2025-01-01T00:00:02Z",
            }
        }
        with patch.object(handler, "_ensure_worker") as ensure_worker:
            await handler._handle_transcript("bot-001", data)

        assert session.response_mode == "active"
        assert session.response_queue.qsize() == 1
        ensure_worker.assert_called_once()

    async def test_transcript_extracts_action_candidate(self, handler, registry):
        data = {
            "data": {
                "participant": {"name": "Alice"},
                "words": [{"text": "After"}, {"text": "this"}, {"text": "call"}, {"text": "start"}, {"text": "a"}, {"text": "Hermes"}, {"text": "session"}],
                "started_at": "2025-01-01T00:00:03Z",
            }
        }
        with patch.object(handler, "_ensure_worker"):
            await handler._handle_transcript("bot-001", data)

        session = registry.get("bot-001")
        assert len(session.action_candidates) == 1
        assert session.action_candidates[0].target == "hermes"


@pytest.mark.asyncio
class TestHandleRouting:
    """Test RecallWebhookHandler.handle() routes events correctly."""

    async def test_routes_status_change(self, handler):
        body = {
            "event": "bot.status_change",
            "data": {
                "bot": {"id": "bot-001"},
                "status": {"code": "in_meeting"},
            },
        }
        with patch.object(handler, "_handle_status_change", new_callable=AsyncMock) as mock:
            result = await handler.handle(body)
            mock.assert_awaited_once()

    async def test_routes_transcript(self, handler):
        body = {
            "event": "transcript.data",
            "data": {
                "bot": {"id": "bot-001"},
                "data": {
                    "participant": {"name": "Alice"},
                    "words": [{"text": "Hi"}],
                    "started_at": "2025-01-01T00:00:00Z",
                },
            },
        }
        with patch.object(handler, "_handle_transcript", new_callable=AsyncMock) as mock:
            result = await handler.handle(body)
            mock.assert_awaited_once()

    async def test_routes_partial_transcript(self, handler):
        body = {
            "event": "transcript.partial_data",
            "data": {
                "bot": {"id": "bot-001"},
                "data": {
                    "participant": {"name": "Alice"},
                    "words": [{"text": "Hi"}],
                },
            },
        }
        result = await handler.handle(body)
        assert result["ok"] is True

    async def test_unhandled_event(self, handler):
        body = {
            "event": "unknown.event",
            "data": {"bot": {"id": "bot-001"}},
        }
        result = await handler.handle(body)
        assert result["ok"] is True

    async def test_returns_ok(self, handler):
        body = {"event": "something", "data": {}}
        result = await handler.handle(body)
        assert result == {"ok": True}
