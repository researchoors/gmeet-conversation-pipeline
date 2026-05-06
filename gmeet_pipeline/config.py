"""
gmeet_pipeline.config — Single Pydantic-settings source of truth for all
environment-driven configuration previously scattered across meeting_agent.py
and meeting_agent_rvc.py as os.environ.get() globals.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GmeetSettings(BaseSettings):
    """Consolidated configuration for the Google Meet agent pipeline.

    Every field reads from an environment variable with the ``GMEET_`` prefix.
    For example, ``recall_api_key`` → ``GMEET_RECALL_API_KEY``.
    """

    model_config = SettingsConfigDict(env_prefix="GMEET_")

    # ── Recall.ai ──────────────────────────────────────────────────────
    recall_api_key: str = ""
    recall_base: str = "https://us-west-2.recall.ai/api/v1"

    # ── OpenRouter / LLM ───────────────────────────────────────────────
    openrouter_key: str = ""
    llm_model: str = "anthropic/claude-sonnet-4"
    fast_model: str = "google/gemini-2.5-flash"
    standard_model: str = "openai/gpt-4.1-mini"
    deep_model: str = "anthropic/claude-sonnet-4"

    # ── ElevenLabs TTS ─────────────────────────────────────────────────
    elevenlabs_key: str = ""
    elevenlabs_voice: str = ""
    elevenlabs_model: str = "eleven_multilingual_v2"

    # ── RVC voice cloning ──────────────────────────────────────────────
    rvc_model_path: str = ""
    rvc_exp_dir: str = ""
    rvc_repo_dir: str = ""
    rvc_f0_method: str = "rmvpe"
    rvc_f0_up_key: int = 0
    rvc_index_rate: float = 0.75
    rvc_filter_radius: int = 3
    rvc_rms_mix_rate: float = 0.25
    rvc_protect: float = 0.33

    # ── Kokoro TTS ─────────────────────────────────────────────────────
    kokoro_voice: str = "af_heart"

    # ── Service / network ──────────────────────────────────────────────
    service_url: str = ""
    port: int = 9120

    # ── Backend selectors ──────────────────────────────────────────────
    tts_backend: Literal["elevenlabs", "local"] = "local"
    llm_routing: Literal["simple", "voice_gateway", "flash", "local"] = "flash"

    # ── Auth ──────────────────────────────────────────────────────────
    api_key: str = ""  # Bearer token for admin endpoints (GMEET_API_KEY)
    webhook_secret: str = ""  # Secret embedded in webhook URL path (GMEET_WEBHOOK_SECRET)

    # ── Post-call Hermes action routing ───────────────────────────────
    post_call_hermes_enabled: bool = False
    post_call_hermes_cmd: str = "hermes"
    post_call_model: str = "google/gemini-2.5-flash"
    post_call_provider: str = "openrouter"
    post_call_toolsets: str = "terminal,file,skills,session_search"
    post_call_max_parallel_sessions: int = 3
    post_call_dry_run: bool = False

    # ── Paths ──────────────────────────────────────────────────────────
    hermes_home: str = str(Path.home() / ".hermes")

    # ── Computed paths (derived from hermes_home) ──────────────────────

    @computed_field  # type: ignore[prop-decorator]
    @property
    def audio_dir(self) -> Path:
        return Path(self.hermes_home) / "audio_cache" / "meeting_tts"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def memory_file(self) -> Path:
        return Path(self.hermes_home) / "memories" / "MEMORY.md"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def user_file(self) -> Path:
        return Path(self.hermes_home) / "memories" / "USER.md"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def meeting_artifacts_dir(self) -> Path:
        return Path(self.hermes_home) / "gmeet-artifacts"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def action_inbox_dir(self) -> Path:
        return Path(self.hermes_home) / "data-inbox" / "gmeet"

    # ── Factory ────────────────────────────────────────────────────────

    @classmethod
    def load(cls) -> GmeetSettings:
        """Build settings with .env and auth.json credential-pool fallbacks.

        Resolution order for each value:
        1. Explicit environment variable (``GMEET_*`` prefix)
        2. ``~/.hermes/.env`` file (loaded via python-dotenv)
        3. ``~/.hermes/auth.json`` credential pool (openrouter key only)
        4. Field default
        """
        # 1. Try python-dotenv to populate env from ~/.hermes/.env
        _try_dotenv()

        # 2. Build the settings (reads env vars via pydantic-settings)
        settings = cls()

        # 3. If openrouter_key is still empty, try auth.json credential pool
        if not settings.openrouter_key:
            key_from_pool = _openrouter_key_from_auth_json(settings.hermes_home)
            if key_from_pool:
                settings.openrouter_key = key_from_pool

        return settings


# ── Private helpers ────────────────────────────────────────────────────


def _try_dotenv() -> None:
    """Best-effort load of ``~/.hermes/.env`` via python-dotenv."""
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]

        env_path = Path.home() / ".hermes" / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except Exception:
        pass


def _openrouter_key_from_auth_json(hermes_home: str) -> str:
    """Return the first non-exhausted OpenRouter key from auth.json, or ''."""
    auth_path = Path(hermes_home) / "auth.json"
    if not auth_path.exists():
        return ""
    try:
        data = json.loads(auth_path.read_text())
        for entry in data.get("credential_pool", {}).get("openrouter", []):
            token = entry.get("access_token", "")
            if token and entry.get("status") != "exhausted":
                return token
    except Exception:
        pass
    return ""
