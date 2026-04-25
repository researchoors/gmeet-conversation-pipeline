"""Tests for gmeet_pipeline.llm."""

import json
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from gmeet_pipeline.llm.base import BaseLLM
from gmeet_pipeline.llm.openrouter import SimpleOpenRouterLLM, VoiceGatewayLLM, _parse_expands, _strip_expands
from gmeet_pipeline.memory import MemorySnapshot


class TestBaseLLMIsABC:
    """Test BaseLLM is an abstract base class."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseLLM()


@pytest.mark.asyncio
class TestSimpleOpenRouterLLMGenerate:
    """Test SimpleOpenRouterLLM.generate()."""

    async def test_generate_returns_response_text(self):
        llm = SimpleOpenRouterLLM(api_key="test-key", model="test/model")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello from Hank!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        with patch("gmeet_pipeline.llm.openrouter.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await llm.generate([], "Hi there")

        assert result == "Hello from Hank!"

    async def test_generate_returns_none_for_silent_tokens(self):
        llm = SimpleOpenRouterLLM(api_key="test-key", model="test/model")

        for token in ["SILENT", "NO_RESPONSE", "PASS", "SKIP"]:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": token}}],
            }

            with patch("gmeet_pipeline.llm.openrouter.httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client_cls.return_value = mock_client

                result = await llm.generate([], "irrelevant")
                assert result is None, f"Expected None for token '{token}'"

    async def test_generate_returns_none_on_api_error(self):
        llm = SimpleOpenRouterLLM(api_key="test-key", model="test/model")

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("gmeet_pipeline.llm.openrouter.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await llm.generate([], "Hi")
            assert result is None

    async def test_generate_returns_none_on_exception(self):
        llm = SimpleOpenRouterLLM(api_key="test-key", model="test/model")

        with patch("gmeet_pipeline.llm.openrouter.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=Exception("Network error"))
            mock_client_cls.return_value = mock_client

            result = await llm.generate([], "Hi")
            assert result is None


@pytest.mark.asyncio
class TestVoiceGatewayLLMGenerate:
    """Test VoiceGatewayLLM.generate()."""

    def _make_memory_snapshot(self, tmp_path):
        mem = tmp_path / "MEMORY.md"
        usr = tmp_path / "USER.md"
        mem.write_text("Darkbloom is a decentralized inference project.§SwiftLM is an engine.")
        usr.write_text("Ethan is the founder.")
        snap = MemorySnapshot(memory_file=mem, user_file=usr)
        snap.build()
        return snap

    async def test_generate_routes_to_correct_model(self, tmp_path):
        """Test that 'fast' query routes to fast_model."""
        snap = self._make_memory_snapshot(tmp_path)
        llm = VoiceGatewayLLM(
            api_key="test-key",
            fast_model="fast-model",
            standard_model="standard-model",
            deep_model="deep-model",
            memory_snapshot=snap,
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Fast reply"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }

        # "hi" should classify as "fast" → uses fast_model
        with patch("gmeet_pipeline.llm.openrouter.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await llm.generate([], "hi")

        assert result == "Fast reply"
        call_args = mock_client.post.call_args
        request_body = call_args[1]["json"]
        assert request_body["model"] == "fast-model"

    async def test_generate_parses_expand_directives(self, tmp_path):
        snap = self._make_memory_snapshot(tmp_path)
        llm = VoiceGatewayLLM(
            api_key="test-key",
            fast_model="fast-model",
            standard_model="standard-model",
            deep_model="deep-model",
            memory_snapshot=snap,
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "See details EXPAND[0] for more."}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }

        bot_state = {"expanded_entries": set()}
        with patch("gmeet_pipeline.llm.openrouter.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await llm.generate([], "tell me about darkbloom", bot_state=bot_state)

        # EXPAND[0] should be stripped from output
        assert "EXPAND" not in result
        # Index 0 should be added to bot_state
        assert 0 in bot_state["expanded_entries"]


class TestVoiceGatewayLLMBuildSystemPrompt:
    """Test VoiceGatewayLLM.build_system_prompt()."""

    def _make_memory_snapshot(self, tmp_path):
        mem = tmp_path / "MEMORY.md"
        usr = tmp_path / "USER.md"
        mem.write_text("Darkbloom project info.§SwiftLM engine details.")
        usr.write_text("Ethan founder info.")
        snap = MemorySnapshot(memory_file=mem, user_file=usr)
        snap.build()
        return snap

    def test_includes_memory_context(self, tmp_path):
        snap = self._make_memory_snapshot(tmp_path)
        llm = VoiceGatewayLLM(
            api_key="test-key",
            fast_model="fast",
            standard_model="standard",
            deep_model="deep",
            memory_snapshot=snap,
        )
        prompt = llm.build_system_prompt("standard")
        assert "User context:" in prompt
        assert snap.summary in prompt

    def test_includes_rag_entries(self, tmp_path):
        snap = self._make_memory_snapshot(tmp_path)
        llm = VoiceGatewayLLM(
            api_key="test-key",
            fast_model="fast",
            standard_model="standard",
            deep_model="deep",
            memory_snapshot=snap,
        )
        rag_entries = [{"index": 0, "source": "memory", "text": "Darkbloom project info."}]
        prompt = llm.build_system_prompt("standard", rag_entries=rag_entries)
        assert "Relevant memory:" in prompt
        assert "Darkbloom" in prompt

    def test_no_memory_returns_base_prompt(self):
        llm = VoiceGatewayLLM(
            api_key="test-key",
            fast_model="fast",
            standard_model="standard",
            deep_model="deep",
            memory_snapshot=None,
        )
        prompt = llm.build_system_prompt("fast")
        assert "Hank Bob" in prompt

    def test_fast_path_includes_summary(self, tmp_path):
        snap = self._make_memory_snapshot(tmp_path)
        llm = VoiceGatewayLLM(
            api_key="test-key",
            fast_model="fast",
            standard_model="standard",
            deep_model="deep",
            memory_snapshot=snap,
        )
        prompt = llm.build_system_prompt("fast")
        # Fast path includes summary but not entry index
        assert "User context:" in prompt


class TestModuleHelpers:
    """Test module-level EXPAND helper functions."""

    def test_parse_expands_with_memory(self, tmp_path):
        mem = tmp_path / "MEMORY.md"
        usr = tmp_path / "USER.md"
        mem.write_text("Entry 0§Entry 1§Entry 2")
        usr.write_text("")
        snap = MemorySnapshot(memory_file=mem, user_file=usr)
        snap.build()

        result = _parse_expands("EXPAND[0,2]", snap)
        assert result == [0, 2]

    def test_parse_expands_no_memory(self):
        result = _parse_expands("EXPAND[0]", None)
        assert result == []

    def test_strip_expands(self):
        result = _strip_expands("See EXPAND[0] for details")
        assert "EXPAND" not in result
        assert "See" in result
        assert "for details" in result
