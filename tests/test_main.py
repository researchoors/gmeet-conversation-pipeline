"""Tests for gmeet_pipeline.main — create_app routing and main() CLI."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gmeet_pipeline.config import GmeetSettings
from gmeet_pipeline.main import create_app


# ── Helpers ───────────────────────────────────────────────────────────


def _make_settings(**overrides):
    """Create a GmeetSettings for testing with safe defaults."""
    defaults = dict(
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
        hermes_home="/tmp/gmeet-test-home",
        api_key="test-api-key",
        webhook_secret="test-wh-secret",
    )
    defaults.update(overrides)
    return GmeetSettings(**defaults)


# ── create_app routing ────────────────────────────────────────────────


class TestCreateAppLocalRouting:
    """Test create_app with llm_routing='local'."""

    def test_local_routing_creates_flash_llm(self, tmp_path):
        """llm_routing='local' should create a FlashLLM with base_url and no_think."""
        settings = _make_settings(
            llm_routing="local",
            llm_base_url="http://localhost:8080/v1",
            llm_api_key="local-key",
            llm_no_think=True,
            tts_backend="elevenlabs",
            hermes_home=str(tmp_path),
        )
        server = create_app(settings)
        from gmeet_pipeline.llm.flash import FlashLLM
        assert isinstance(server.llm, FlashLLM)
        assert server.llm.base_url == "http://localhost:8080/v1"
        assert server.llm.no_think is True
        assert server.llm.api_key == "local-key"

    def test_local_routing_does_not_require_openrouter_key(self, tmp_path):
        """llm_routing='local' should work without openrouter_key."""
        settings = _make_settings(
            llm_routing="local",
            openrouter_key="",
            llm_base_url="http://localhost:8080/v1",
            llm_api_key="unused",
            tts_backend="elevenlabs",
            hermes_home=str(tmp_path),
        )
        server = create_app(settings)
        from gmeet_pipeline.llm.flash import FlashLLM
        assert isinstance(server.llm, FlashLLM)

    def test_local_routing_uses_llm_model(self, tmp_path):
        """llm_routing='local' should use settings.llm_model."""
        settings = _make_settings(
            llm_routing="local",
            llm_model="mlx-community/gemma-3-4b-it-4bit",
            llm_base_url="http://localhost:8080/v1",
            tts_backend="elevenlabs",
            hermes_home=str(tmp_path),
        )
        server = create_app(settings)
        assert server.llm.model == "mlx-community/gemma-3-4b-it-4bit"


class TestCreateAppFlashRouting:
    """Test create_app with llm_routing='flash'."""

    def test_flash_routing_creates_flash_llm(self, tmp_path):
        """llm_routing='flash' should create a FlashLLM with OpenRouter."""
        settings = _make_settings(
            llm_routing="flash",
            tts_backend="elevenlabs",
            hermes_home=str(tmp_path),
        )
        server = create_app(settings)
        from gmeet_pipeline.llm.flash import FlashLLM
        assert isinstance(server.llm, FlashLLM)

    def test_flash_routing_uses_openrouter_key(self, tmp_path):
        """llm_routing='flash' should pass openrouter_key as api_key."""
        settings = _make_settings(
            llm_routing="flash",
            openrouter_key="test-or-key",
            tts_backend="elevenlabs",
            hermes_home=str(tmp_path),
        )
        server = create_app(settings)
        assert server.llm.api_key == "test-or-key"

    def test_flash_routing_no_base_url(self, tmp_path):
        """llm_routing='flash' should use default (OpenRouter) base_url."""
        settings = _make_settings(
            llm_routing="flash",
            tts_backend="elevenlabs",
            hermes_home=str(tmp_path),
        )
        server = create_app(settings)
        assert server.llm.base_url == "https://openrouter.ai/api/v1"


class TestCreateAppVoiceGatewayRouting:
    """Test create_app with llm_routing='voice_gateway'."""

    def test_voice_gateway_routing_creates_llm(self, tmp_path):
        settings = _make_settings(
            llm_routing="voice_gateway",
            tts_backend="elevenlabs",
            hermes_home=str(tmp_path),
        )
        server = create_app(settings)
        from gmeet_pipeline.llm.openrouter import VoiceGatewayLLM
        assert isinstance(server.llm, VoiceGatewayLLM)


class TestCreateAppSimpleRouting:
    """Test create_app with llm_routing='simple'."""

    def test_simple_routing_creates_llm(self, tmp_path):
        settings = _make_settings(
            llm_routing="simple",
            tts_backend="elevenlabs",
            hermes_home=str(tmp_path),
        )
        server = create_app(settings)
        from gmeet_pipeline.llm.openrouter import SimpleOpenRouterLLM
        assert isinstance(server.llm, SimpleOpenRouterLLM)


class TestCreateAppServerSetup:
    """Test that create_app sets up the server correctly."""

    def test_server_has_audio_queue(self, tmp_path):
        settings = _make_settings(
            llm_routing="simple",
            tts_backend="elevenlabs",
            hermes_home=str(tmp_path),
        )
        server = create_app(settings)
        assert isinstance(server.audio_queue, dict)

    def test_server_has_registry(self, tmp_path):
        settings = _make_settings(
            llm_routing="simple",
            tts_backend="elevenlabs",
            hermes_home=str(tmp_path),
        )
        server = create_app(settings)
        assert server.registry is not None

    def test_server_has_webhook_handler(self, tmp_path):
        settings = _make_settings(
            llm_routing="simple",
            tts_backend="elevenlabs",
            hermes_home=str(tmp_path),
        )
        server = create_app(settings)
        assert server.webhook_handler is not None

    def test_no_recall_key_sets_transport_none(self, tmp_path):
        settings = _make_settings(
            recall_api_key="",
            llm_routing="simple",
            tts_backend="elevenlabs",
            hermes_home=str(tmp_path),
        )
        server = create_app(settings)
        assert server.transport is None

    def test_with_recall_key_creates_transport(self, tmp_path):
        settings = _make_settings(
            llm_routing="simple",
            tts_backend="elevenlabs",
            hermes_home=str(tmp_path),
        )
        server = create_app(settings)
        assert server.transport is not None

    def test_webhook_handler_ws_manager_set(self, tmp_path):
        settings = _make_settings(
            llm_routing="simple",
            tts_backend="elevenlabs",
            hermes_home=str(tmp_path),
        )
        server = create_app(settings)
        assert server.webhook_handler.ws_manager is not None

    def test_creates_audio_dir(self, tmp_path):
        audio_dir = tmp_path / "audio_cache" / "meeting_tts"
        settings = _make_settings(
            llm_routing="simple",
            tts_backend="elevenlabs",
            hermes_home=str(tmp_path),
        )
        server = create_app(settings)
        assert Path(settings.audio_dir).exists()


class TestMainCLI:
    """Test the main() CLI entry point."""

    def test_main_exits_without_recall_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GMEET_HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("GMEET_RECALL_API_KEY", raising=False)
        with pytest.raises(SystemExit):
            from gmeet_pipeline.main import main
            main()

    def test_main_local_routing_skips_openrouter_check(self, tmp_path, monkeypatch):
        """main() with llm_routing='local' should not require openrouter_key."""
        monkeypatch.setenv("GMEET_RECALL_API_KEY", "test")
        monkeypatch.setenv("GMEET_LLM_ROUTING", "local")
        monkeypatch.setenv("GMEET_OPENROUTER_KEY", "")
        monkeypatch.setenv("GMEET_HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("GMEET_TTS_BACKEND", "elevenlabs")
        monkeypatch.setenv("GMEET_ELEVENLABS_KEY", "test")

        # Should NOT raise SystemExit — uvicorn.run will block, so mock it
        with patch("gmeet_pipeline.main.uvicorn.run"):
            from gmeet_pipeline.main import main
            main()  # Should not raise SystemExit

    def test_main_non_local_routing_requires_openrouter_key(self, tmp_path, monkeypatch):
        """main() with llm_routing!='local' should require openrouter_key."""
        monkeypatch.setenv("GMEET_RECALL_API_KEY", "test")
        monkeypatch.setenv("GMEET_LLM_ROUTING", "flash")
        monkeypatch.setenv("GMEET_OPENROUTER_KEY", "")
        monkeypatch.setenv("GMEET_HERMES_HOME", str(tmp_path))

        with pytest.raises(SystemExit):
            from gmeet_pipeline.main import main
            main()
