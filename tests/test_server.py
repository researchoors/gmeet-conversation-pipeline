"""Tests for gmeet_pipeline.server."""

import asyncio
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from gmeet_pipeline.config import GmeetSettings
from gmeet_pipeline.state import BotRegistry, BotSession
from gmeet_pipeline.transports.base import BaseTransport
from gmeet_pipeline.llm.base import BaseLLM
from gmeet_pipeline.tts.base import BaseTTS
from gmeet_pipeline.ws_manager import ConnectionManager
from gmeet_pipeline.server import GmeetServer


# Concrete stubs for abstract base classes
class StubTransport(BaseTransport):
    async def join(self, meeting_url, bot_name="Hank Bob", **kwargs):
        return {"id": "stub-bot-001"}

    async def leave(self, bot_id):
        return {"status": "leaving"}

    async def get_status(self, bot_id):
        return "in_meeting"


class StubLLM(BaseLLM):
    async def generate(self, conversation, message, bot_state=None):
        return "Stub response"


class StubTTS(BaseTTS):
    async def generate(self, text, bot_id):
        return "stub_audio.wav"


@pytest.fixture
def test_settings(tmp_path):
    return GmeetSettings(
        recall_api_key="test-key",
        recall_base="https://us-west-2.recall.ai/api/v1",
        openrouter_key="test-or-key",
        service_url="https://test.example.com",
        port=9999,
        tts_backend="elevenlabs",
        llm_routing="simple",
        hermes_home=str(tmp_path / ".hermes"),
        api_key="test-api-key",
        webhook_secret="test-wh-secret",
    )


@pytest.fixture
def server(test_settings, tmp_path):
    registry = BotRegistry()
    ws_manager = ConnectionManager()
    transport = StubTransport()
    llm = StubLLM()
    tts = StubTTS()

    # Ensure audio dir exists
    Path(test_settings.audio_dir).mkdir(parents=True, exist_ok=True)

    return GmeetServer(
        settings=test_settings,
        transport=transport,
        llm=llm,
        tts=tts,
        ws_manager=ws_manager,
        registry=registry,
        memory_snapshot=None,
    )


@pytest.fixture
def client(server):
    return TestClient(server.app)


class TestCreateApp:
    """Test GmeetServer has all expected components."""

    def test_server_has_app(self, server):
        assert server.app is not None

    def test_server_has_settings(self, server):
        assert server.settings is not None

    def test_server_has_registry(self, server):
        assert server.registry is not None

    def test_server_has_transport(self, server):
        assert server.transport is not None

    def test_server_has_llm(self, server):
        assert server.llm is not None

    def test_server_has_tts(self, server):
        assert server.tts is not None

    def test_server_has_ws_manager(self, server):
        assert server.ws_manager is not None

    def test_server_has_webhook_handler(self, server):
        assert server.webhook_handler is not None


class TestServerRoutes:
    """Test GmeetServer has all expected routes."""

    def test_has_health_route(self, server):
        routes = [r.path for r in server.app.routes]
        assert "/health" in routes

    def test_has_bots_route(self, server):
        routes = [r.path for r in server.app.routes]
        assert "/api/bots" in routes

    def test_has_webhook_route(self, server):
        routes = [r.path for r in server.app.routes]
        assert "/webhook/recall" in routes

    def test_has_agent_page_route(self, server):
        routes = [r.path for r in server.app.routes]
        assert "/" in routes

    def test_has_join_route(self, server):
        routes = [r.path for r in server.app.routes]
        assert "/api/bot/join" in routes

    def test_has_leave_route(self, server):
        routes = [r.path for r in server.app.routes]
        assert "/api/bot/{bot_id}/leave" in routes


class TestHealthEndpoint:
    """Test /health endpoint returns status."""

    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_health_includes_tts_backend(self, client):
        response = client.get("/health")
        data = response.json()
        assert "tts_backend" in data

    def test_health_includes_llm_routing(self, client):
        response = client.get("/health")
        data = response.json()
        assert "llm_routing" in data


class TestBotsEndpoint:
    """Test /api/bots endpoint returns bot list."""

    def test_bots_empty(self, client):
        response = client.get("/api/bots", headers={"Authorization": "Bearer test-api-key"})
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert len(data) == 0

    def test_bots_with_session(self, server, client):
        # Add a session directly
        session = BotSession(bot_id="bot-1", meeting_url="https://meet.google.com/test")
        server.registry._bots["bot-1"] = session

        response = client.get("/api/bots", headers={"Authorization": "Bearer test-api-key"})
        assert response.status_code == 200
        data = response.json()
        assert "bot-1" in data

    def test_bots_requires_auth(self, client):
        response = client.get("/api/bots")
        assert response.status_code == 401

    def test_bots_wrong_key(self, client):
        response = client.get("/api/bots", headers={"Authorization": "Bearer wrong-key"})
        assert response.status_code == 401


class TestWebhookEndpoint:
    """Test /webhook/recall delegates to handler."""

    def test_webhook_delegates(self, server, client):
        with patch.object(server.webhook_handler, "handle", new_callable=AsyncMock, return_value={"ok": True}) as mock:
            response = client.post(
                "/webhook/recall/test-wh-secret",
                json={"event": "test", "data": {}},
            )
            assert response.status_code == 200

    def test_webhook_returns_handler_result(self, server, client):
        with patch.object(server.webhook_handler, "handle", new_callable=AsyncMock, return_value={"ok": True}):
            response = client.post(
                "/webhook/recall/test-wh-secret",
                json={"event": "test", "data": {}},
            )
            data = response.json()
            assert data["ok"] is True

    def test_webhook_wrong_secret(self, client):
        response = client.post(
            "/webhook/recall/wrong-secret",
            json={"event": "test", "data": {}},
        )
        assert response.status_code == 401


class TestAgentPageRoute:
    """Test / serves agent HTML."""

    def test_agent_page_returns_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Hank Bob" in response.text


class TestSessionStateEndpoint:
    """Test /api/session-state returns live response mode metadata."""

    def test_session_state_includes_mode_event_count(self, server, client):
        session = BotSession(bot_id="bot-1", meeting_url="https://meet.google.com/test")
        session.response_mode = "silent_transcribe"
        session.mode_events.append({"mode": "silent_transcribe"})
        server.registry._bots["bot-1"] = session

        response = client.get("/api/session-state")
        assert response.status_code == 200
        bot = response.json()["bots"][0]
        assert bot["response_mode"] == "silent_transcribe"
        assert bot["mode_event_count"] == 1


class TestJoinEndpoint:
    """Test /api/bot/join creates a bot session."""

    def test_join_requires_meeting_url(self, client):
        response = client.post("/api/bot/join", json={}, headers={"Authorization": "Bearer test-api-key"})
        assert response.status_code == 400

    def test_join_creates_session(self, server, client):
        with patch.object(server.transport, "join", new_callable=AsyncMock, return_value={"id": "new-bot"}):
            response = client.post(
                "/api/bot/join",
                json={"meeting_url": "https://meet.google.com/new"},
                headers={"Authorization": "Bearer test-api-key"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["bot_id"] == "new-bot"

    def test_join_requires_auth(self, client):
        response = client.post("/api/bot/join", json={"meeting_url": "https://meet.google.com/new"})
        assert response.status_code == 401


class TestRecallStatusMonitor:
    """Test server-side Recall lifecycle polling for post-call finalization."""

    def test_join_starts_status_monitor_task(self, server, client):
        with patch.object(server, "_start_status_monitor") as monitor:
            response = client.post(
                "/api/bot/join",
                headers={"Authorization": "Bearer test-api-key"},
                json={"meeting_url": "https://meet.google.com/test"},
            )

        assert response.status_code == 200
        monitor.assert_called_once_with("stub-bot-001")

    @pytest.mark.asyncio
    async def test_monitor_finalizes_when_recall_reports_done(self, server):
        session = await server.registry.create("bot-done", "https://meet.google.com/test")
        statuses = iter(["in_call_recording", "done"])

        async def fake_status(bot_id):
            return next(statuses)

        server.transport.get_status = fake_status
        with patch.object(server.webhook_handler, "_finalize_call") as finalize:
            await server._monitor_recall_status("bot-done", poll_interval=0, max_polls=3)

        finalize.assert_called_once_with(session, "recall_done")
        assert session.status == "done"

    @pytest.mark.asyncio
    async def test_monitor_ignores_non_terminal_statuses(self, server):
        await server.registry.create("bot-active", "https://meet.google.com/test")
        server.transport.get_status = AsyncMock(return_value="in_call_recording")

        with patch.object(server.webhook_handler, "_finalize_call") as finalize:
            await server._monitor_recall_status("bot-active", poll_interval=0, max_polls=2)

        finalize.assert_not_called()
