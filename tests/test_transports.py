"""Tests for gmeet_pipeline.transports."""

import json
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from gmeet_pipeline.transports.base import BaseTransport
from gmeet_pipeline.transports.recall import RecallTransport


class TestBaseTransportIsABC:
    """Test BaseTransport is an abstract base class."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseTransport()

    def test_has_abstract_methods(self):
        # Verify abstract methods exist
        assert "join" in [m for m in dir(BaseTransport)]
        assert "leave" in [m for m in dir(BaseTransport)]
        assert "get_status" in [m for m in dir(BaseTransport)]


@pytest.mark.asyncio
class TestRecallTransportJoin:
    """Test RecallTransport.join() makes correct API call."""

    async def test_join_makes_post_request(self):
        transport = RecallTransport(
            api_key="test-key",
            base_url="https://us-west-2.recall.ai/api/v1",
            service_url="https://test.example.com",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "bot-123", "status": {"code": "joining"}}

        with patch("gmeet_pipeline.transports.recall.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await transport.join("https://meet.google.com/test", "Hank Bob")

        assert result["id"] == "bot-123"
        mock_client.post.assert_awaited_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://us-west-2.recall.ai/api/v1/bot/"
        # Verify Authorization header
        headers = call_args[1]["headers"]
        assert headers["Authorization"] == "Token test-key"

    async def test_join_raises_on_non_2xx(self):
        transport = RecallTransport(
            api_key="test-key",
            base_url="https://us-west-2.recall.ai/api/v1",
            service_url="https://test.example.com",
        )

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch("gmeet_pipeline.transports.recall.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            with pytest.raises(Exception, match="Recall API error"):
                await transport.join("https://meet.google.com/test")


@pytest.mark.asyncio
class TestRecallTransportLeave:
    """Test RecallTransport.leave() makes correct API call."""

    async def test_leave_makes_post_request(self):
        transport = RecallTransport(
            api_key="test-key",
            base_url="https://us-west-2.recall.ai/api/v1",
            service_url="https://test.example.com",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("gmeet_pipeline.transports.recall.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await transport.leave("bot-123")

        assert result["status"] == "leaving"
        call_args = mock_client.post.call_args
        assert "bot-123/leave/" in call_args[0][0]


@pytest.mark.asyncio
class TestRecallTransportGetStatus:
    """Test RecallTransport.get_status() extracts status code."""

    async def test_get_status_returns_code(self):
        transport = RecallTransport(
            api_key="test-key",
            base_url="https://us-west-2.recall.ai/api/v1",
            service_url="https://test.example.com",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": {"code": "in_meeting"}}

        with patch("gmeet_pipeline.transports.recall.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await transport.get_status("bot-123")

        assert result == "in_meeting"

    async def test_get_status_returns_none_on_error(self):
        transport = RecallTransport(
            api_key="test-key",
            base_url="https://us-west-2.recall.ai/api/v1",
            service_url="https://test.example.com",
        )

        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("gmeet_pipeline.transports.recall.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await transport.get_status("bot-123")

        assert result is None
