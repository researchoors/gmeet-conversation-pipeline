"""Tests for gmeet_pipeline.tts."""

import asyncio
import json
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gmeet_pipeline.tts.base import BaseTTS
from gmeet_pipeline.tts.elevenlabs import ElevenLabsTTS
from gmeet_pipeline.tts.local import LocalTTS
from gmeet_pipeline.ws_manager import ConnectionManager


class TestBaseTTSIsABC:
    """Test BaseTTS is an abstract base class."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseTTS()


@pytest.mark.asyncio
class TestElevenLabsTTSGenerate:
    """Test ElevenLabsTTS.generate() — streaming then fallback."""

    def _make_tts(self, tmp_path, ws_manager=None):
        if ws_manager is None:
            ws_manager = ConnectionManager()
        audio_dir = tmp_path / "audio"
        return ElevenLabsTTS(
            api_key="test-11-key",
            voice_id="test-voice",
            model_id="eleven_multilingual_v2",
            audio_dir=audio_dir,
            ws_manager=ws_manager,
        )

    async def test_generate_tries_streaming_first(self, tmp_path):
        """generate() should attempt streaming path first."""
        tts = self._make_tts(tmp_path)

        with patch.object(tts, "_generate_streaming", new_callable=AsyncMock, return_value="streamed:bot1") as mock_stream:
            with patch.object(tts, "_generate_fallback", new_callable=AsyncMock, return_value="fallback.mp3") as mock_fallback:
                result = await tts.generate("Hello", "bot1")

        assert result == "streamed:bot1"
        mock_stream.assert_awaited_once_with("Hello", "bot1")
        mock_fallback.assert_not_awaited()

    async def test_generate_falls_back_on_streaming_failure(self, tmp_path):
        """generate() should fall back to REST when streaming fails."""
        tts = self._make_tts(tmp_path)

        with patch.object(tts, "_generate_streaming", new_callable=AsyncMock, side_effect=Exception("WS failed")):
            with patch.object(tts, "_generate_fallback", new_callable=AsyncMock, return_value="fallback.mp3"):
                result = await tts.generate("Hello", "bot1")

        assert result == "fallback.mp3"

    async def test_fallback_makes_http_request(self, tmp_path):
        """Test _generate_fallback calls the REST endpoint."""
        ws_manager = MagicMock()
        ws_manager.broadcast_json = AsyncMock()
        tts = self._make_tts(tmp_path, ws_manager)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"\xff\xfb\x90\x00" * 100  # fake mp3 data

        with patch("gmeet_pipeline.tts.elevenlabs.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await tts._generate_fallback("Hello", "bot1")

        assert result is not None
        assert result.endswith(".mp3")
        # File should be written
        audio_dir = tmp_path / "audio"
        assert (audio_dir / result).exists()

    async def test_fallback_returns_none_on_error(self, tmp_path):
        """Test _generate_fallback returns None on API error."""
        ws_manager = MagicMock()
        ws_manager.broadcast_json = AsyncMock()
        tts = self._make_tts(tmp_path, ws_manager)

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch("gmeet_pipeline.tts.elevenlabs.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await tts._generate_fallback("Hello", "bot1")

        assert result is None


class TestLocalTTSEnsureInit:
    """Test LocalTTS._ensure_init() lazy loading."""

    def test_ensure_init_not_initialized_by_default(self, tmp_path):
        tts = LocalTTS(
            rvc_model_path="",
            rvc_exp_dir="",
            rvc_repo_dir="",
            audio_dir=str(tmp_path / "audio"),
        )
        assert tts._initialized is False
        assert tts._kokoro_pipeline is None

    def test_ensure_init_loads_kokoro(self, tmp_path):
        """Test that _ensure_init attempts to import kokoro."""
        tts = LocalTTS(
            rvc_model_path="",
            rvc_exp_dir="",
            rvc_repo_dir="",
            audio_dir=str(tmp_path / "audio"),
        )

        mock_pipeline = MagicMock()
        with patch.dict("sys.modules", {"kokoro": MagicMock(KPipeline=MagicMock(return_value=mock_pipeline))}):
            tts._ensure_init()

        assert tts._initialized is True

    def test_ensure_init_handles_missing_kokoro(self, tmp_path):
        """Test that _ensure_init handles missing kokoro gracefully."""
        tts = LocalTTS(
            rvc_model_path="",
            rvc_exp_dir="",
            rvc_repo_dir="",
            audio_dir=str(tmp_path / "audio"),
        )

        with patch.dict("sys.modules", {}):
            # kokoro import will fail
            tts._ensure_init()

        assert tts._initialized is True
        assert tts._kokoro_pipeline is None  # gracefully failed

    def test_ensure_init_only_runs_once(self, tmp_path):
        """Test that _ensure_init is idempotent."""
        tts = LocalTTS(
            rvc_model_path="",
            rvc_exp_dir="",
            rvc_repo_dir="",
            audio_dir=str(tmp_path / "audio"),
        )
        tts._initialized = True
        # Should not re-run
        tts._ensure_init()
        # Still True (no crash, no double init)
        assert tts._initialized is True


@pytest.mark.asyncio
class TestLocalTTSGenerate:
    """Test LocalTTS.generate() behavior when pipeline is unavailable."""

    async def test_generate_returns_none_when_no_pipeline(self, tmp_path):
        """When kokoro pipeline is None, generate should return None."""
        tts = LocalTTS(
            rvc_model_path="",
            rvc_exp_dir="",
            rvc_repo_dir="",
            audio_dir=str(tmp_path / "audio"),
        )
        tts._initialized = True
        tts._kokoro_pipeline = None

        result = await tts.generate("Hello", "bot1")
        assert result is None
