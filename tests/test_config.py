"""Tests for gmeet_pipeline.config."""

import json
import os
from pathlib import Path
from typing import Optional
from unittest.mock import patch, MagicMock

import pytest

from gmeet_pipeline.config import GmeetSettings, _try_dotenv, _openrouter_key_from_auth_json


class TestGmeetSettingsDefaults:
    """Test that all GmeetSettings fields have expected defaults."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        """Clear all GMEET_ env vars so defaults are tested in isolation."""
        for key in list(os.environ):
            if key.startswith("GMEET_"):
                monkeypatch.delenv(key, raising=False)

    def test_recall_defaults(self):
        s = GmeetSettings()
        assert s.recall_api_key == ""
        assert s.recall_base == "https://us-west-2.recall.ai/api/v1"

    def test_llm_defaults(self):
        s = GmeetSettings()
        assert s.openrouter_key == ""
        assert s.llm_model == "anthropic/claude-sonnet-4"
        assert s.fast_model == "google/gemini-2.5-flash"
        assert s.standard_model == "openai/gpt-4.1-mini"
        assert s.deep_model == "anthropic/claude-sonnet-4"

    def test_elevenlabs_defaults(self):
        s = GmeetSettings()
        assert s.elevenlabs_key == ""
        assert s.elevenlabs_voice == ""
        assert s.elevenlabs_model == "eleven_multilingual_v2"

    def test_rvc_defaults(self):
        s = GmeetSettings()
        assert s.rvc_model_path == ""
        assert s.rvc_exp_dir == ""
        assert s.rvc_repo_dir == ""
        assert s.rvc_f0_method == "rmvpe"
        assert s.rvc_f0_up_key == 0
        assert s.rvc_index_rate == 0.0

    def test_kokoro_default(self):
        s = GmeetSettings()
        assert s.kokoro_voice == "af_heart"

    def test_service_defaults(self):
        s = GmeetSettings()
        assert s.service_url == ""
        assert s.port == 9120

    def test_backend_selector_defaults(self):
        s = GmeetSettings()
        assert s.tts_backend == "local"
        assert s.llm_routing == "local"

    def test_hermes_home_default(self):
        s = GmeetSettings()
        assert s.hermes_home == str(Path.home() / ".hermes")


class TestComputedFields:
    """Test that computed path fields derive from hermes_home."""

    def test_audio_dir(self):
        s = GmeetSettings(hermes_home="/tmp/testhome")
        assert s.audio_dir == Path("/tmp/testhome/audio_cache/meeting_tts")

    def test_memory_file(self):
        s = GmeetSettings(hermes_home="/tmp/testhome")
        assert s.memory_file == Path("/tmp/testhome/memories/MEMORY.md")

    def test_user_file(self):
        s = GmeetSettings(hermes_home="/tmp/testhome")
        assert s.user_file == Path("/tmp/testhome/memories/USER.md")

    def test_computed_fields_change_with_hermes_home(self):
        s1 = GmeetSettings(hermes_home="/a")
        s2 = GmeetSettings(hermes_home="/b")
        assert s1.audio_dir != s2.audio_dir
        assert s1.memory_file != s2.memory_file
        assert s1.user_file != s2.user_file


class TestEnvVarOverride:
    """Test that environment variables override defaults."""

    def test_port_override(self, monkeypatch):
        monkeypatch.setenv("GMEET_PORT", "8888")
        s = GmeetSettings()
        assert s.port == 8888

    def test_recall_api_key_override(self, monkeypatch):
        monkeypatch.setenv("GMEET_RECALL_API_KEY", "my-secret-key")
        s = GmeetSettings()
        assert s.recall_api_key == "my-secret-key"

    def test_tts_backend_override(self, monkeypatch):
        monkeypatch.setenv("GMEET_TTS_BACKEND", "local")
        s = GmeetSettings()
        assert s.tts_backend == "local"

    def test_hermes_home_override(self, monkeypatch):
        monkeypatch.setenv("GMEET_HERMES_HOME", "/custom/path")
        s = GmeetSettings()
        assert s.hermes_home == "/custom/path"


class TestLoad:
    """Test the GmeetSettings.load() class method."""

    def test_load_returns_settings(self, monkeypatch):
        """load() should return a GmeetSettings instance even without .env or auth.json."""
        # Ensure no .env interference
        monkeypatch.setenv("GMEET_HERMES_HOME", "/tmp/nonexistent_test_home")
        settings = GmeetSettings.load()
        assert isinstance(settings, GmeetSettings)

    def test_load_calls_try_dotenv(self, monkeypatch):
        """load() should call _try_dotenv()."""
        monkeypatch.setenv("GMEET_HERMES_HOME", "/tmp/nonexistent_test_home")
        with patch("gmeet_pipeline.config._try_dotenv") as mock_dotenv:
            GmeetSettings.load()
            mock_dotenv.assert_called_once()

    def test_load_falls_back_to_auth_json(self, tmp_path, monkeypatch):
        """When openrouter_key is empty, load() should try auth.json."""
        auth_path = tmp_path / "auth.json"
        auth_path.write_text(json.dumps({
            "credential_pool": {
                "openrouter": [
                    {"access_token": "test-active-key", "status": "active"}
                ]
            }
        }))
        monkeypatch.setenv("GMEET_HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("GMEET_OPENROUTER_KEY", raising=False)
        with patch("gmeet_pipeline.config._try_dotenv"):
            settings = GmeetSettings.load()
        assert settings.openrouter_key == "test-active-key"

    def test_load_skips_exhausted_auth_json(self, tmp_path, monkeypatch):
        """load() should skip exhausted keys in auth.json."""
        auth_path = tmp_path / "auth.json"
        auth_path.write_text(json.dumps({
            "credential_pool": {
                "openrouter": [
                    {"access_token": "exhausted-key", "status": "exhausted"},
                    {"access_token": "active-key-2", "status": "active"},
                ]
            }
        }))
        monkeypatch.setenv("GMEET_HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("GMEET_OPENROUTER_KEY", raising=False)
        with patch("gmeet_pipeline.config._try_dotenv"):
            settings = GmeetSettings.load()
        assert settings.openrouter_key == "active-key-2"

    def test_load_env_key_takes_priority(self, tmp_path, monkeypatch):
        """Environment variable should take priority over auth.json."""
        auth_path = tmp_path / "auth.json"
        auth_path.write_text(json.dumps({
            "credential_pool": {
                "openrouter": [
                    {"access_token": "auth-json-key", "status": "active"}
                ]
            }
        }))
        monkeypatch.setenv("GMEET_HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("GMEET_OPENROUTER_KEY", "env-key")
        settings = GmeetSettings.load()
        assert settings.openrouter_key == "env-key"


class TestAuthJsonHelpers:
    """Test the private _openrouter_key_from_auth_json helper."""

    def test_no_auth_json(self, tmp_path):
        result = _openrouter_key_from_auth_json(str(tmp_path))
        assert result == ""

    def test_valid_auth_json(self, tmp_path):
        (tmp_path / "auth.json").write_text(json.dumps({
            "credential_pool": {
                "openrouter": [
                    {"access_token": "key-1", "status": "active"}
                ]
            }
        }))
        result = _openrouter_key_from_auth_json(str(tmp_path))
        assert result == "key-1"

    def test_empty_token_skipped(self, tmp_path):
        (tmp_path / "auth.json").write_text(json.dumps({
            "credential_pool": {
                "openrouter": [
                    {"access_token": "", "status": "active"},
                    {"access_token": "key-2", "status": "active"},
                ]
            }
        }))
        result = _openrouter_key_from_auth_json(str(tmp_path))
        assert result == "key-2"

    def test_invalid_json_returns_empty(self, tmp_path):
        (tmp_path / "auth.json").write_text("not json")
        result = _openrouter_key_from_auth_json(str(tmp_path))
        assert result == ""
