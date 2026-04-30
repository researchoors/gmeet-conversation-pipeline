"""Unit tests for FlashLLM — single-model Gemini Flash with front-loaded context."""

import json
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gmeet_pipeline.context_builder import ContextBuilder
from gmeet_pipeline.llm.flash import FlashLLM, SILENT_TOKENS


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def mock_context_dir(tmp_path):
    """Temp memories directory with sample content."""
    d = tmp_path / "memories"
    d.mkdir()
    (d / "MEMORY.md").write_text("DarkBloom = decentralized inference on Apple Silicon§\nSwiftLM benchmarks show DFlash wins on MoE")
    (d / "USER.md").write_text("Ethan is part of researchoors group§\nPrefers concise responses")
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    return d, sessions


@pytest.fixture
def flash_llm(mock_context_dir):
    mem_dir, sess_dir = mock_context_dir
    builder = ContextBuilder(memories_dir=mem_dir, sessions_dir=sess_dir)
    return FlashLLM(
        api_key="test-key",
        model="google/gemini-2.5-flash",
        service_url="https://test.example.com",
        context_builder=builder,
    )


# ── System prompt construction tests ─────────────────────────────────

def test_system_prompt_contains_persona(flash_llm):
    """ContextBuilder output is just memory context — persona is added by FlashLLM.generate()."""
    prompt = flash_llm.context_builder.build()
    # ContextBuilder only produces memory + user + session context
    assert "DarkBloom" in prompt
    assert "Ethan" in prompt
    # The persona gets prepended in generate()


def test_system_prompt_contains_memory(flash_llm):
    """Context should include Hermes memory entries."""
    prompt = flash_llm.context_builder.build()
    assert "DarkBloom" in prompt
    assert "Ethan" in prompt


def test_system_prompt_no_rag_artifacts(flash_llm):
    """No RAG, EXPAND, or classification artifacts."""
    prompt = flash_llm.context_builder.build()
    assert "EXPAND[" not in prompt
    assert "classify" not in prompt.lower()


# ── generate() tests with mocked HTTP ────────────────────────────────

@pytest.mark.asyncio
async def test_generate_returns_response(flash_llm):
    """Successful LLM call returns the response text."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Hey! DFlash is great for MoE models."}}],
        "usage": {"prompt_tokens": 500, "completion_tokens": 20},
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("gmeet_pipeline.llm.flash.httpx.AsyncClient", return_value=mock_client):
        result = await flash_llm.generate(
            conversation=[],
            message="Ethan said: How does DFlash work?",
        )

    assert result == "Hey! DFlash is great for MoE models."


@pytest.mark.asyncio
async def test_generate_silent_response(flash_llm):
    """LLM returning a SILENT token should produce None."""
    for token in SILENT_TOKENS:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": token}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 1},
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("gmeet_pipeline.llm.flash.httpx.AsyncClient", return_value=mock_client):
            result = await flash_llm.generate(
                conversation=[], message="Someone said: hi"
            )
        assert result is None, f"SILENT token '{token}' should produce None"


@pytest.mark.asyncio
async def test_generate_api_error(flash_llm):
    """API error should return None gracefully."""
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.text = "Rate limited"

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("gmeet_pipeline.llm.flash.httpx.AsyncClient", return_value=mock_client):
        result = await flash_llm.generate(
            conversation=[], message="test"
        )
    assert result is None


@pytest.mark.asyncio
async def test_generate_network_error(flash_llm):
    """Network error should return None gracefully."""
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=Exception("Connection refused"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("gmeet_pipeline.llm.flash.httpx.AsyncClient", return_value=mock_client):
        result = await flash_llm.generate(
            conversation=[], message="test"
        )
    assert result is None


@pytest.mark.asyncio
async def test_generate_passes_context_in_system_prompt(flash_llm):
    """The system prompt sent to the API should contain memory context."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Response"}}],
        "usage": {"prompt_tokens": 500, "completion_tokens": 5},
    }

    captured_messages = None

    async def capture_post(url, **kwargs):
        nonlocal captured_messages
        captured_messages = kwargs.get("json", {}).get("messages", [])
        return mock_response

    mock_client = AsyncMock()
    mock_client.post = capture_post
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("gmeet_pipeline.llm.flash.httpx.AsyncClient", return_value=mock_client):
        await flash_llm.generate(
            conversation=[{"role": "user", "content": "Ethan: hello"}],
            message="Ethan said: hello",
        )

    assert captured_messages is not None
    system_msg = captured_messages[0]
    assert system_msg["role"] == "system"
    # System prompt should contain memory context
    assert "DarkBloom" in system_msg["content"]
    assert "Ethan" in system_msg["content"]
    # Should also contain the persona
    assert "Hank Bob" in system_msg["content"]


@pytest.mark.asyncio
async def test_generate_conversation_truncation(flash_llm):
    """Conversation should be truncated to last 20 turns."""
    # Build a conversation with 30 turns
    conversation = []
    for i in range(30):
        conversation.append({"role": "user", "content": f"msg {i}"})
        conversation.append({"role": "assistant", "content": f"resp {i}"})

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 5},
    }

    captured_messages = None

    async def capture_post(url, **kwargs):
        nonlocal captured_messages
        captured_messages = kwargs.get("json", {}).get("messages", [])
        return mock_response

    mock_client = AsyncMock()
    mock_client.post = capture_post
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("gmeet_pipeline.llm.flash.httpx.AsyncClient", return_value=mock_client):
        await flash_llm.generate(conversation=conversation, message="new msg")

    # system + last 20 + new message = 22
    assert len(captured_messages) == 22
    # conversation[-20:] from a 60-item list = items 40-59, first is "msg 20"
    assert "msg 20" in captured_messages[1]["content"]


@pytest.mark.asyncio
async def test_generate_uses_flash_model(flash_llm):
    """Should use the configured model (default: gemini-2.5-flash)."""
    captured_model = None

    async def capture_post(url, **kwargs):
        nonlocal captured_model
        captured_model = kwargs.get("json", {}).get("model")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 5},
        }
        return mock_response

    mock_client = AsyncMock()
    mock_client.post = capture_post
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("gmeet_pipeline.llm.flash.httpx.AsyncClient", return_value=mock_client):
        await flash_llm.generate(conversation=[], message="test")

    assert captured_model == "google/gemini-2.5-flash"


# ── No routing / no RAG verification ──────────────────────────────────

@pytest.mark.asyncio
async def test_no_rag_retrieval_called(flash_llm):
    """FlashLLM should never call RAG retrieval — context is pre-baked."""
    # If it works without any rag_retrieve calls, that's the test
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {},
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("gmeet_pipeline.llm.flash.httpx.AsyncClient", return_value=mock_client):
        result = await flash_llm.generate(conversation=[], message="test")
    # Just verify it returns without needing any RAG mock
    assert result == "ok"


def test_flash_llm_no_classify_method(flash_llm):
    """FlashLLM should not have classify_query or rag_retrieve methods."""
    # These are VoiceGatewayLLM concerns
    assert not hasattr(flash_llm, "classify_query")
    assert not hasattr(flash_llm, "rag_retrieve")
