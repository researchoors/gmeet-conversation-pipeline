"""E2E testing framework for the gmeet-conversation-pipeline.

Spins up the real FastAPI server with mocked external services (Recall API,
OpenRouter LLM) and sends realistic webhook payloads through the full pipeline.

Usage:
    pytest tests/e2e/ -v
"""

from .conftest import (
    MockLLM,
    SilentLLM,
    MockTTS,
    FailingTTS,
    MockTransport,
    make_transcript_data,
    make_partial_transcript,
    make_status_change,
    make_call_ended,
    make_participant_join,
    make_participant_leave,
)

__all__ = [
    "MockLLM", "SilentLLM", "MockTTS", "FailingTTS", "MockTransport",
    "make_transcript_data", "make_partial_transcript", "make_status_change",
    "make_call_ended", "make_participant_join", "make_participant_leave",
]
