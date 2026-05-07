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


# ── FlashLLM local mode (base_url + no_think) ────────────────────────


@pytest.fixture
def local_flash_llm(mock_context_dir):
    """FlashLLM configured for local OpenAI-compatible server."""
    mem_dir, sess_dir = mock_context_dir
    builder = ContextBuilder(memories_dir=mem_dir, sessions_dir=sess_dir)
    return FlashLLM(
        api_key="local-key",
        model="mlx-community/gemma-3-4b-it-4bit",
        service_url="https://test.example.com",
        context_builder=builder,
        base_url="http://localhost:8080/v1",
        no_think=True,
    )


@pytest.fixture
def local_flash_llm_no_think_false(mock_context_dir):
    """FlashLLM configured for local server without no_think."""
    mem_dir, sess_dir = mock_context_dir
    builder = ContextBuilder(memories_dir=mem_dir, sessions_dir=sess_dir)
    return FlashLLM(
        api_key="local-key",
        model="mlx-community/gemma-3-4b-it-4bit",
        service_url="https://test.example.com",
        context_builder=builder,
        base_url="http://localhost:8080/v1",
        no_think=False,
    )


class TestFlashLLMLocalMode:
    """Test FlashLLM with custom base_url (local OpenAI-compatible server)."""

    def test_custom_base_url_constructor(self, local_flash_llm):
        """FlashLLM with custom base_url stores it correctly."""
        assert local_flash_llm.base_url == "http://localhost:8080/v1"

    def test_no_think_constructor(self, local_flash_llm):
        """FlashLLM with no_think=True stores it correctly."""
        assert local_flash_llm.no_think is True

    def test_no_think_false_constructor(self, local_flash_llm_no_think_false):
        """FlashLLM with no_think=False stores it correctly."""
        assert local_flash_llm_no_think_false.no_think is False

    def test_default_base_url_is_openrouter(self, flash_llm):
        """FlashLLM without base_url defaults to OpenRouter."""
        assert flash_llm.base_url == "https://openrouter.ai/api/v1"

    @pytest.mark.asyncio
    async def test_local_mode_hits_base_url(self, local_flash_llm):
        """Local mode should hit base_url/chat/completions, not OpenRouter."""
        captured_url = None

        async def capture_post(url, **kwargs):
            nonlocal captured_url
            captured_url = url
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {},
            }
            return mock_response

        mock_client = AsyncMock()
        mock_client.post = capture_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("gmeet_pipeline.llm.flash.httpx.AsyncClient", return_value=mock_client):
            await local_flash_llm.generate(conversation=[], message="test")

        assert captured_url is not None
        assert "localhost:8080" in captured_url
        assert "openrouter.ai" not in captured_url

    @pytest.mark.asyncio
    async def test_local_mode_omits_referer_header(self, local_flash_llm):
        """Local mode should NOT include HTTP-Referer header."""
        captured_headers = None

        async def capture_post(url, **kwargs):
            nonlocal captured_headers
            captured_headers = kwargs.get("headers", {})
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {},
            }
            return mock_response

        mock_client = AsyncMock()
        mock_client.post = capture_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("gmeet_pipeline.llm.flash.httpx.AsyncClient", return_value=mock_client):
            await local_flash_llm.generate(conversation=[], message="test")

        assert "HTTP-Referer" not in captured_headers

    @pytest.mark.asyncio
    async def test_openrouter_mode_includes_referer_header(self, flash_llm):
        """OpenRouter mode should include HTTP-Referer header."""
        captured_headers = None

        async def capture_post(url, **kwargs):
            nonlocal captured_headers
            captured_headers = kwargs.get("headers", {})
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {},
            }
            return mock_response

        mock_client = AsyncMock()
        mock_client.post = capture_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("gmeet_pipeline.llm.flash.httpx.AsyncClient", return_value=mock_client):
            await flash_llm.generate(conversation=[], message="test")

        assert "HTTP-Referer" in captured_headers

    @pytest.mark.asyncio
    async def test_no_think_true_prepends_prefix(self, local_flash_llm):
        """no_think=True should prepend /no_think\\n to user message."""
        captured_messages = None

        async def capture_post(url, **kwargs):
            nonlocal captured_messages
            captured_messages = kwargs.get("json", {}).get("messages", [])
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {},
            }
            return mock_response

        mock_client = AsyncMock()
        mock_client.post = capture_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("gmeet_pipeline.llm.flash.httpx.AsyncClient", return_value=mock_client):
            await local_flash_llm.generate(conversation=[], message="test question")

        # Last message is the user message
        user_msg = captured_messages[-1]
        assert user_msg["role"] == "user"
        assert user_msg["content"].startswith("/no_think\n")
        assert "test question" in user_msg["content"]

    @pytest.mark.asyncio
    async def test_no_think_false_no_prefix(self, local_flash_llm_no_think_false):
        """no_think=False should NOT prepend /no_think to user message."""
        captured_messages = None

        async def capture_post(url, **kwargs):
            nonlocal captured_messages
            captured_messages = kwargs.get("json", {}).get("messages", [])
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {},
            }
            return mock_response

        mock_client = AsyncMock()
        mock_client.post = capture_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("gmeet_pipeline.llm.flash.httpx.AsyncClient", return_value=mock_client):
            await local_flash_llm_no_think_false.generate(
                conversation=[], message="test question"
            )

        user_msg = captured_messages[-1]
        assert user_msg["role"] == "user"
        assert not user_msg["content"].startswith("/no_think\n")

    @pytest.mark.asyncio
    async def test_local_mode_url_format(self, local_flash_llm):
        """Local mode URL should be base_url/chat/completions."""
        captured_url = None

        async def capture_post(url, **kwargs):
            nonlocal captured_url
            captured_url = url
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {},
            }
            return mock_response

        mock_client = AsyncMock()
        mock_client.post = capture_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("gmeet_pipeline.llm.flash.httpx.AsyncClient", return_value=mock_client):
            await local_flash_llm.generate(conversation=[], message="test")

        assert captured_url == "http://localhost:8080/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_local_mode_trailing_slash_handling(self, mock_context_dir):
        """Trailing slash on base_url should be stripped."""
        mem_dir, sess_dir = mock_context_dir
        builder = ContextBuilder(memories_dir=mem_dir, sessions_dir=sess_dir)
        llm = FlashLLM(
            api_key="local-key",
            model="test-model",
            base_url="http://localhost:8080/v1/",
            context_builder=builder,
        )

        captured_url = None

        async def capture_post(url, **kwargs):
            nonlocal captured_url
            captured_url = url
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {},
            }
            return mock_response

        mock_client = AsyncMock()
        mock_client.post = capture_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("gmeet_pipeline.llm.flash.httpx.AsyncClient", return_value=mock_client):
            await llm.generate(conversation=[], message="test")

        # Should not have double slashes
        assert "//chat" not in captured_url
        assert captured_url == "http://localhost:8080/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_local_mode_includes_auth_header(self, local_flash_llm):
        """Local mode should include Authorization header with api_key."""
        captured_headers = None

        async def capture_post(url, **kwargs):
            nonlocal captured_headers
            captured_headers = kwargs.get("headers", {})
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {},
            }
            return mock_response

        mock_client = AsyncMock()
        mock_client.post = capture_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("gmeet_pipeline.llm.flash.httpx.AsyncClient", return_value=mock_client):
            await local_flash_llm.generate(conversation=[], message="test")

        assert "Authorization" in captured_headers
        assert captured_headers["Authorization"] == "Bearer local-key"

    @pytest.mark.asyncio
    async def test_env_var_base_url_override(self, mock_context_dir, monkeypatch):
        """GMEET_LLM_BASE_URL env var should be used as base_url fallback."""
        monkeypatch.setenv("GMEET_LLM_BASE_URL", "http://custom:9999/v1")
        mem_dir, sess_dir = mock_context_dir
        builder = ContextBuilder(memories_dir=mem_dir, sessions_dir=sess_dir)
        llm = FlashLLM(
            api_key="test",
            model="test-model",
            context_builder=builder,
            # base_url not set explicitly — should fall back to env var
        )
        assert llm.base_url == "http://custom:9999/v1"

    @pytest.mark.asyncio
    async def test_env_var_no_think_override(self, mock_context_dir, monkeypatch):
        """GMEET_LLM_NO_THINK env var should set no_think."""
        monkeypatch.setenv("GMEET_LLM_NO_THINK", "true")
        mem_dir, sess_dir = mock_context_dir
        builder = ContextBuilder(memories_dir=mem_dir, sessions_dir=sess_dir)
        llm = FlashLLM(
            api_key="test",
            model="test-model",
            context_builder=builder,
        )
        assert llm.no_think is True
