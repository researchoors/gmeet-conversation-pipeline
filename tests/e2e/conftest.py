"""E2E test configuration and shared fixtures."""

import asyncio
import time
import uuid
from typing import Optional

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from gmeet_pipeline.config import GmeetSettings
from gmeet_pipeline.llm.base import BaseLLM
from gmeet_pipeline.tts.base import BaseTTS
from gmeet_pipeline.transports.base import BaseTransport
from gmeet_pipeline.state import BotRegistry
from gmeet_pipeline.ws_manager import ConnectionManager
from gmeet_pipeline.server import GmeetServer


# ---------------------------------------------------------------------------
# Mock backends
# ---------------------------------------------------------------------------

class MockLLM(BaseLLM):
    """Deterministic LLM that echoes a fixed response after a configurable delay."""

    def __init__(self, response: str = "Hey there! Good to hear from you.", delay_ms: int = 50):
        self._response = response
        self._delay = delay_ms / 1000.0
        self.call_count = 0
        self.last_conversation = None
        self.last_message = None

    async def generate(self, conversation: list, message: str, bot_state=None) -> Optional[str]:
        self.call_count += 1
        self.last_conversation = conversation
        self.last_message = message
        await asyncio.sleep(self._delay)
        return self._response


class SilentLLM(BaseLLM):
    """LLM that always returns None (agent stays silent)."""

    def __init__(self):
        self.call_count = 0

    async def generate(self, conversation: list, message: str, bot_state=None) -> Optional[str]:
        self.call_count += 1
        return None


class MockTTS(BaseTTS):
    """TTS that writes a tiny WAV file after a configurable delay."""

    def __init__(self, audio_dir: str, delay_ms: int = 10):
        self._audio_dir = audio_dir
        self._delay = delay_ms / 1000.0
        self.call_count = 0
        self.last_text = None

    async def generate(self, text: str, bot_id: str) -> Optional[str]:
        import struct
        from pathlib import Path

        self.call_count += 1
        self.last_text = text
        await asyncio.sleep(self._delay)

        # Ensure directory exists
        Path(self._audio_dir).mkdir(parents=True, exist_ok=True)

        # Write a minimal valid WAV: 22050 Hz, 16-bit mono, 0.1s silence
        sample_rate = 22050
        num_samples = sample_rate // 10
        filename = f"e2e_tts_{uuid.uuid4().hex[:8]}.wav"
        filepath = Path(self._audio_dir) / filename

        data = b"\x00\x00" * num_samples
        fmt_chunk = b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        data_chunk = b"data" + struct.pack("<I", len(data)) + data
        riff = b"RIFF" + struct.pack("<I", 4 + len(fmt_chunk) + len(data_chunk)) + b"WAVE"
        filepath.write_bytes(riff + fmt_chunk + data_chunk)

        return filename


class FailingTTS(BaseTTS):
    """TTS that always returns None (simulates TTS failure)."""

    def __init__(self):
        self.call_count = 0

    async def generate(self, text: str, bot_id: str) -> Optional[str]:
        self.call_count += 1
        return None


class MockTransport(BaseTransport):
    """Transport that pretends to create/leave bots via Recall."""

    def __init__(self):
        self.bots_created = []
        self.bots_left = []

    async def join(self, meeting_url: str, bot_name: str = "Hank Bob", **kwargs) -> dict:
        bot_id = str(uuid.uuid4())
        self.bots_created.append({"bot_id": bot_id, "meeting_url": meeting_url, "bot_name": bot_name})
        return {"id": bot_id, "status": "joining"}

    async def leave(self, bot_id: str) -> dict:
        self.bots_left.append(bot_id)
        return {"status": "leaving"}

    async def get_status(self, bot_id: str) -> Optional[str]:
        return "in_meeting"


# ---------------------------------------------------------------------------
# Webhook payload builders (mimic real Recall.ai payloads)
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def make_transcript_data(bot_id: str, speaker: str, text: str, started_at: str = "") -> dict:
    """Build a realistic transcript.data webhook payload."""
    words = [{"text": w, "start": 0.0, "end": 0.0} for w in text.split()]
    return {
        "event": "transcript.data",
        "data": {
            "bot": {"id": bot_id},
            "data": {
                "participant": {"name": speaker, "id": speaker.lower().replace(" ", "-")},
                "words": words,
                "started_at": started_at or _now_iso(),
            },
        },
    }


def make_partial_transcript(bot_id: str, speaker: str, text: str) -> dict:
    """Build a transcript.partial_data webhook payload."""
    words = [{"text": w, "start": 0.0, "end": 0.0} for w in text.split()]
    return {
        "event": "transcript.partial_data",
        "data": {
            "bot": {"id": bot_id},
            "data": {
                "participant": {"name": speaker, "id": speaker.lower().replace(" ", "-")},
                "words": words,
            },
        },
    }


def make_status_change(bot_id: str, status_code: str = "in_meeting") -> dict:
    return {
        "event": "bot.status_change",
        "data": {
            "bot": {"id": bot_id},
            "status": {"code": status_code},
        },
    }


def make_call_ended(bot_id: str) -> dict:
    return {
        "event": "bot.call_ended",
        "data": {
            "bot": {"id": bot_id},
            "bot_id": bot_id,
        },
    }


def make_participant_join(bot_id: str, name: str) -> dict:
    return {
        "event": "participant_events.join",
        "data": {
            "bot": {"id": bot_id},
            "data": {
                "participant": {"name": name, "id": name.lower().replace(" ", "-")},
            },
        },
    }


def make_participant_leave(bot_id: str, name: str) -> dict:
    return {
        "event": "participant_events.leave",
        "data": {
            "bot": {"id": bot_id},
            "data": {
                "participant": {"name": name, "id": name.lower().replace(" ", "-")},
            },
        },
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_settings(tmp_path):
    """Settings pointing to a temp audio dir."""
    return GmeetSettings(
        recall_api_key="test-key",
        openrouter_key="test-or-key",
        tts_backend="local",
        llm_routing="simple",
        hermes_home=str(tmp_path / "hermes"),
    )


@pytest.fixture
def mock_llm():
    return MockLLM()


@pytest.fixture
def mock_tts(mock_settings):
    return MockTTS(audio_dir=str(mock_settings.audio_dir))


@pytest.fixture
def mock_transport():
    return MockTransport()


@pytest_asyncio.fixture
async def e2e_client(mock_settings, mock_llm, mock_tts, mock_transport):
    """Async HTTP client wired to the real GmeetServer with mock backends."""
    registry = BotRegistry()
    ws_manager = ConnectionManager()
    server = GmeetServer(
        settings=mock_settings,
        transport=mock_transport,
        llm=mock_llm,
        tts=mock_tts,
        ws_manager=ws_manager,
        registry=registry,
    )

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Attach server internals for assertions
        client._server = server
        client._registry = registry
        client._mock_llm = mock_llm
        client._mock_tts = mock_tts
        client._mock_transport = mock_transport
        yield client


# ---------------------------------------------------------------------------
# Auto-mark async tests
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(config, items):
    """Auto-mark async tests with @pytest.mark.asyncio."""
    for item in items:
        if asyncio.iscoroutinefunction(item.obj):
            item.add_marker(pytest.mark.asyncio)
