"""Integration tests for the full webhook pipeline flow.

Tests transcript -> wake word -> LLM -> TTS -> audio queue with mocked backends.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gmeet_pipeline.webhook import RecallWebhookHandler
from gmeet_pipeline.state import BotRegistry, BotSession
from gmeet_pipeline.llm.base import BaseLLM
from gmeet_pipeline.tts.base import BaseTTS
from gmeet_pipeline.ws_manager import ConnectionManager


class MockLLM(BaseLLM):
    """LLM that returns predictable responses."""

    def __init__(self, response="Test response"):
        self.response = response
        self.call_count = 0
        self.last_conversation = None
        self.last_message = None
        self.last_bot_state = None

    async def generate(self, conversation, message, bot_state=None):
        self.call_count += 1
        self.last_conversation = conversation
        self.last_message = message
        self.last_bot_state = bot_state
        return self.response


class MockTTS(BaseTTS):
    """TTS that returns a predictable filename."""

    def __init__(self, audio_dir="/tmp/test_tts"):
        self.call_count = 0
        self.last_text = None
        self._audio_dir = audio_dir

    async def generate(self, text, bot_id):
        self.call_count += 1
        self.last_text = text
        return f"tts_{bot_id[:8]}_{self.call_count}.wav"


@pytest.fixture
def registry_with_session():
    """Registry with a live bot session."""
    reg = BotRegistry()
    session = BotSession(
        bot_id="bot-integ-001",
        meeting_url="https://meet.google.com/test-integ",
        status="in_call_recording",
    )
    session.participants["Ethan"] = {
        "join_ts": "2025-01-01T00:00:00Z",
        "is_speaking": False,
    }
    reg._bots["bot-integ-001"] = session
    return reg


@pytest.fixture
def mock_llm():
    return MockLLM(response="Hey Ethan, good to hear from you!")


@pytest.fixture
def mock_tts():
    return MockTTS()


@pytest.fixture
def audio_queue():
    return {}


@pytest.fixture
def handler(registry_with_session, mock_llm, mock_tts, audio_queue):
    ws_mgr = ConnectionManager()
    h = RecallWebhookHandler(
        registry=registry_with_session,
        llm=mock_llm,
        tts=mock_tts,
        ws_manager=ws_mgr,
        audio_queue=audio_queue,
    )
    return h


def _transcript_payload(speaker, words, bot_id="bot-integ-001", ts="2025-01-01T00:00:00Z"):
    """Build a realistic Recall transcript.data payload."""
    return {
        "event": "transcript.data",
        "data": {
            "bot": {"id": bot_id},
            "data": {
                "participant": {"name": speaker},
                "words": [{"text": w} for w in words],
                "started_at": ts,
            },
        },
    }


@pytest.mark.asyncio
class TestFullPipelineFlow:
    """Test the complete transcript -> LLM -> TTS -> audio queue flow."""

    async def test_transcript_triggers_llm_then_tts(
        self, handler, registry_with_session, mock_llm, mock_tts, audio_queue
    ):
        """A normal transcript should go through LLM -> TTS -> audio queue."""
        body = _transcript_payload("Ethan", ["Hello", "Hank", "Bob"])

        with patch.object(handler, "_ensure_worker"):
            await handler._handle_transcript("bot-integ-001", body["data"])

        session = registry_with_session.get("bot-integ-001")
        assert session.response_queue.qsize() == 1

        # Simulate what the queue worker does
        msg = await session.response_queue.get()
        messages = [msg]

        # Run _process_batch
        await handler._process_batch("bot-integ-001", messages)

        # Verify LLM was called
        assert mock_llm.call_count == 1
        assert "Ethan" in mock_llm.last_message

        # Verify TTS was called
        assert mock_tts.call_count == 1
        assert mock_tts.last_text == "Hey Ethan, good to hear from you!"

        # Verify audio was queued
        assert "bot-integ-001" in audio_queue
        assert len(audio_queue["bot-integ-001"]) == 1
        assert audio_queue["bot-integ-001"][0]["text"] == "Hey Ethan, good to hear from you!"

    async def test_silence_command_skips_llm_and_tts(
        self, handler, registry_with_session, mock_llm, mock_tts, audio_queue
    ):
        """A silence command should not trigger LLM or TTS."""
        body = _transcript_payload("Ethan", ["Hank", "Bob", "shut", "up"])

        with patch.object(handler, "_ensure_worker"):
            await handler._handle_transcript("bot-integ-001", body["data"])

        assert mock_llm.call_count == 0
        assert mock_tts.call_count == 0
        session = registry_with_session.get("bot-integ-001")
        assert session.response_mode == "silent_transcribe"

    async def test_silent_mode_ignores_normal_utterance(
        self, handler, registry_with_session, mock_llm, mock_tts
    ):
        """In silent mode, normal utterances are recorded but not processed."""
        session = registry_with_session.get("bot-integ-001")
        session.response_mode = "silent_transcribe"

        body = _transcript_payload(
            "Ethan", ["What", "do", "you", "think?"], ts="2025-01-01T00:00:01Z"
        )

        with patch.object(handler, "_ensure_worker"):
            await handler._handle_transcript("bot-integ-001", body["data"])

        # Transcript recorded
        assert len(session.transcript) == 1
        # But LLM/TTS not called
        assert mock_llm.call_count == 0
        assert mock_tts.call_count == 0

    async def test_wake_command_reactivates(
        self, handler, registry_with_session, mock_llm, mock_tts
    ):
        """Wake command reactivates and queues the utterance."""
        session = registry_with_session.get("bot-integ-001")
        session.response_mode = "silent_transcribe"

        body = _transcript_payload(
            "Ethan", ["Hank", "Bob", "wake", "up"], ts="2025-01-01T00:00:02Z"
        )

        with patch.object(handler, "_ensure_worker"):
            await handler._handle_transcript("bot-integ-001", body["data"])

        assert session.response_mode == "active"
        assert session.response_queue.qsize() == 1


@pytest.mark.asyncio
class TestQueueWorker:
    """Test the queue worker lifecycle."""

    async def test_ensure_worker_starts_task(self, handler, registry_with_session):
        """_ensure_worker should create an asyncio task for the bot."""
        handler._ensure_worker("bot-integ-001")
        assert "bot-integ-001" in handler._workers
        worker = handler._workers["bot-integ-001"]
        assert not worker.done()
        # Clean up
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass

    async def test_queue_worker_exits_on_ended_session(
        self, handler, registry_with_session
    ):
        """Queue worker should exit when session status is 'ended'."""
        session = registry_with_session.get("bot-integ-001")
        session.status = "ended"

        handler._ensure_worker("bot-integ-001")
        worker = handler._workers["bot-integ-001"]

        # Give it a moment to check status and exit
        await asyncio.sleep(0.1)
        assert worker.done()

    async def test_queue_worker_processes_batch(
        self, handler, registry_with_session, mock_llm, mock_tts, audio_queue
    ):
        """Queue worker should pick up queued messages and process them."""
        session = registry_with_session.get("bot-integ-001")

        # Put a message in the queue
        await session.response_queue.put(
            {
                "speaker": "Ethan",
                "text": "Hello Hank Bob",
                "ts": "2025-01-01T00:00:00Z",
            }
        )

        # Start the worker
        handler._ensure_worker("bot-integ-001")
        worker = handler._workers["bot-integ-001"]

        # Wait for processing
        await asyncio.sleep(0.5)

        # LLM should have been called
        assert mock_llm.call_count >= 1

        # Clean up
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
class TestProcessBatch:
    """Test _process_batch and _process_batch_inner directly."""

    async def test_process_batch_single_message(
        self, handler, registry_with_session, mock_llm, mock_tts, audio_queue
    ):
        """Single message batch should call LLM then TTS."""
        messages = [
            {"speaker": "Ethan", "text": "Hey Hank", "ts": "2025-01-01T00:00:00Z"}
        ]
        await handler._process_batch("bot-integ-001", messages)

        assert mock_llm.call_count == 1
        assert mock_tts.call_count == 1
        assert len(audio_queue.get("bot-integ-001", [])) == 1

    async def test_process_batch_multi_speaker(
        self, handler, registry_with_session, mock_llm, mock_tts, audio_queue
    ):
        """Multi-speaker batch should combine messages for LLM."""
        messages = [
            {"speaker": "Alice", "text": "Hey", "ts": "2025-01-01T00:00:00Z"},
            {"speaker": "Bob", "text": "What's up", "ts": "2025-01-01T00:00:01Z"},
        ]
        await handler._process_batch("bot-integ-001", messages)

        assert mock_llm.call_count == 1
        # LLM message should contain both speakers
        assert "Alice" in mock_llm.last_message
        assert "Bob" in mock_llm.last_message

    async def test_process_batch_llm_returns_none(
        self, handler, registry_with_session, mock_tts, audio_queue
    ):
        """If LLM returns None (agent stays silent), TTS should not be called."""
        silent_llm = MockLLM(response=None)
        handler.llm = silent_llm

        messages = [
            {"speaker": "Ethan", "text": "Random comment", "ts": "2025-01-01T00:00:00Z"}
        ]
        await handler._process_batch("bot-integ-001", messages)

        assert mock_tts.call_count == 0
        assert "bot-integ-001" not in audio_queue or len(
            audio_queue.get("bot-integ-001", [])
        ) == 0

    async def test_process_batch_llm_returns_silent_token(
        self, handler, registry_with_session, mock_tts, audio_queue
    ):
        """If LLM returns None (from SILENT token in FlashLLM), TTS should be skipped."""
        # SILENT_TOKENS filtering happens in FlashLLM, returning None.
        # When LLM returns None, the webhook skips TTS.
        silent_llm = MockLLM(response=None)
        handler.llm = silent_llm

        messages = [
            {"speaker": "Ethan", "text": "Whatever", "ts": "2025-01-01T00:00:00Z"}
        ]
        await handler._process_batch("bot-integ-001", messages)

        assert mock_tts.call_count == 0

    async def test_process_batch_pipeline_state_tracking(
        self, handler, registry_with_session, mock_llm, mock_tts
    ):
        """Pipeline state should transition through llm -> tts -> speaking."""
        session = registry_with_session.get("bot-integ-001")

        messages = [
            {"speaker": "Ethan", "text": "Hello", "ts": "2025-01-01T00:00:00Z"}
        ]
        await handler._process_batch("bot-integ-001", messages)

        # After processing, pipeline should be "speaking" (audio queued)
        assert session.pipeline_state == "speaking"
        # last_llm_ms and last_tts_ms are set (may be 0 for instant mocks, use >=0)
        assert session.last_llm_ms >= 0
        assert session.last_tts_ms >= 0

    async def test_process_batch_exception_resets_state(
        self, handler, registry_with_session
    ):
        """If TTS raises, pipeline state should reset to idle."""
        failing_tts = MockTTS()
        failing_tts.generate = AsyncMock(side_effect=RuntimeError("TTS engine crash"))
        handler.tts = failing_tts

        session = registry_with_session.get("bot-integ-001")
        messages = [
            {"speaker": "Ethan", "text": "Hello", "ts": "2025-01-01T00:00:00Z"}
        ]

        # Should not raise, but state should be idle
        await handler._process_batch("bot-integ-001", messages)
        assert session.pipeline_state == "idle"


@pytest.mark.asyncio
class TestAudioQueueIntegration:
    """Test that TTS output is correctly pushed to the shared audio queue."""

    async def test_audio_queue_receives_filename_and_text(
        self, handler, registry_with_session, mock_llm, mock_tts, audio_queue
    ):
        """Audio queue entries should have filename and text."""
        messages = [
            {"speaker": "Ethan", "text": "Hi Hank", "ts": "2025-01-01T00:00:00Z"}
        ]
        await handler._process_batch("bot-integ-001", messages)

        entry = audio_queue["bot-integ-001"][0]
        assert "filename" in entry
        assert "text" in entry
        assert entry["filename"].endswith(".wav")
        assert entry["text"] == "Hey Ethan, good to hear from you!"

    async def test_multiple_utterances_queue_sequentially(
        self, handler, registry_with_session, mock_llm, mock_tts, audio_queue
    ):
        """Multiple processed batches should append to the audio queue."""
        for i in range(3):
            messages = [
                {
                    "speaker": "Ethan",
                    "text": f"Message {i}",
                    "ts": f"2025-01-01T00:00:0{i}Z",
                }
            ]
            await handler._process_batch("bot-integ-001", messages)

        assert len(audio_queue["bot-integ-001"]) == 3


@pytest.mark.asyncio
class TestHandleCallEnded:
    """Test call ended / terminal event handling."""

    async def test_call_ended_finalizes_session(self, handler, registry_with_session):
        """bot.call_ended should set session status to ended."""
        body = {
            "event": "bot.call_ended",
            "data": {"bot": {"id": "bot-integ-001"}},
        }
        result = await handler.handle(body)
        assert result["ok"] is True
        session = registry_with_session.get("bot-integ-001")
        assert session.status == "ended"

    async def test_handle_verification_ping(self, handler):
        """Empty event with None bot_id should not crash (Recall verification ping)."""
        body = {"event": "", "data": {"bot_id": None}}
        result = await handler.handle(body)
        assert result["ok"] is True

    async def test_status_change_to_in_call_recording(
        self, handler, registry_with_session
    ):
        """Recall's actual status code should update session correctly."""
        data = {"status": {"code": "in_call_recording"}}
        await handler._handle_status_change("bot-integ-001", data)
        session = registry_with_session.get("bot-integ-001")
        assert session.status == "in_call_recording"

    async def test_status_change_to_in_call_not_recording(
        self, handler, registry_with_session
    ):
        data = {"status": {"code": "in_call_not_recording"}}
        await handler._handle_status_change("bot-integ-001", data)
        session = registry_with_session.get("bot-integ-001")
        assert session.status == "in_call_not_recording"
