"""Shared test fixtures for gmeet_pipeline test suite."""

import asyncio
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pytest

from gmeet_pipeline.config import GmeetSettings
from gmeet_pipeline.state import BotRegistry, BotSession
from gmeet_pipeline.ws_manager import ConnectionManager


@pytest.fixture
def mock_settings(tmp_path):
    """Return a GmeetSettings instance with safe test values."""
    return GmeetSettings(
        recall_api_key="test-recall-key",
        recall_base="https://us-west-2.recall.ai/api/v1",
        openrouter_key="test-or-key",
        llm_model="test/model",
        fast_model="test/fast",
        standard_model="test/standard",
        deep_model="test/deep",
        elevenlabs_key="test-11-key",
        elevenlabs_voice="test-voice-id",
        elevenlabs_model="eleven_multilingual_v2",
        service_url="https://test.example.com",
        port=9999,
        tts_backend="elevenlabs",
        llm_routing="simple",
        hermes_home=str(tmp_path / ".hermes"),
        api_key="test-api-key",
        webhook_secret="test-wh-secret",
    )


@pytest.fixture
def mock_registry():
    """Return a BotRegistry with a pre-populated session."""
    registry = BotRegistry()
    # We add a session manually (synchronously) for test convenience
    session = BotSession(
        bot_id="test-bot-001",
        meeting_url="https://meet.google.com/test",
        status="in_meeting",
    )
    registry._bots["test-bot-001"] = session
    return registry


@pytest.fixture
def mock_ws_manager():
    """Return a ConnectionManager with no connections."""
    return ConnectionManager()


@pytest.fixture
def mock_llm():
    """Return a SimpleOpenRouterLLM with a fake API key."""
    from gmeet_pipeline.llm.openrouter import SimpleOpenRouterLLM
    return SimpleOpenRouterLLM(
        api_key="fake-test-key",
        model="test/model",
        service_url="https://test.example.com",
    )
