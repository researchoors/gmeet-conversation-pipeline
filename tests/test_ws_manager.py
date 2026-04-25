"""Tests for gmeet_pipeline.ws_manager."""

import asyncio
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gmeet_pipeline.ws_manager import ConnectionManager


class TestConnectionManagerInit:
    """Test ConnectionManager initialization."""

    def test_empty_connections(self):
        mgr = ConnectionManager()
        assert mgr.connections == []
        assert mgr.count == 0


class TestConnectionManagerConnect:
    """Test connect/disconnect."""

    @pytest.mark.asyncio
    async def test_connect_adds_connection(self):
        mgr = ConnectionManager()
        mock_ws = MagicMock()
        mock_ws.accept = AsyncMock()
        await mgr.connect(mock_ws)
        assert len(mgr.connections) == 1
        assert mock_ws in mgr.connections

    @pytest.mark.asyncio
    async def test_connect_calls_accept(self):
        mgr = ConnectionManager()
        mock_ws = MagicMock()
        mock_ws.accept = AsyncMock()
        await mgr.connect(mock_ws)
        mock_ws.accept.assert_awaited_once()

    def test_disconnect_removes_connection(self):
        mgr = ConnectionManager()
        mock_ws = MagicMock()
        mgr.connections.append(mock_ws)
        mgr.disconnect(mock_ws)
        assert mock_ws not in mgr.connections

    def test_disconnect_ignores_missing(self):
        mgr = ConnectionManager()
        mock_ws = MagicMock()
        # Should not raise
        mgr.disconnect(mock_ws)


class TestBroadcastJson:
    """Test broadcast_json sends to all connections."""

    @pytest.mark.asyncio
    async def test_broadcast_json_sends_to_all(self):
        mgr = ConnectionManager()
        ws1 = MagicMock()
        ws1.send_json = AsyncMock()
        ws2 = MagicMock()
        ws2.send_json = AsyncMock()
        mgr.connections = [ws1, ws2]

        data = {"type": "start", "text": "hello"}
        await mgr.broadcast_json(data)
        ws1.send_json.assert_awaited_once_with(data)
        ws2.send_json.assert_awaited_once_with(data)


class TestBroadcastBinary:
    """Test broadcast_binary sends to all connections."""

    @pytest.mark.asyncio
    async def test_broadcast_binary_sends_to_all(self):
        mgr = ConnectionManager()
        ws1 = MagicMock()
        ws1.send_bytes = AsyncMock()
        ws2 = MagicMock()
        ws2.send_bytes = AsyncMock()
        mgr.connections = [ws1, ws2]

        payload = b"\x00\x01\x02"
        await mgr.broadcast_binary(payload)
        ws1.send_bytes.assert_awaited_once_with(payload)
        ws2.send_bytes.assert_awaited_once_with(payload)


class TestDeadConnections:
    """Test dead connections are cleaned up on broadcast."""

    @pytest.mark.asyncio
    async def test_dead_connection_removed_on_broadcast_json(self):
        mgr = ConnectionManager()
        alive_ws = MagicMock()
        alive_ws.send_json = AsyncMock()
        dead_ws = MagicMock()
        dead_ws.send_json = AsyncMock(side_effect=Exception("Connection closed"))
        mgr.connections = [alive_ws, dead_ws]

        await mgr.broadcast_json({"type": "test"})
        assert dead_ws not in mgr.connections
        assert alive_ws in mgr.connections

    @pytest.mark.asyncio
    async def test_dead_connection_removed_on_broadcast_binary(self):
        mgr = ConnectionManager()
        alive_ws = MagicMock()
        alive_ws.send_bytes = AsyncMock()
        dead_ws = MagicMock()
        dead_ws.send_bytes = AsyncMock(side_effect=Exception("Connection closed"))
        mgr.connections = [alive_ws, dead_ws]

        await mgr.broadcast_binary(b"\x00")
        assert dead_ws not in mgr.connections
        assert alive_ws in mgr.connections


class TestCountProperty:
    """Test count property."""

    def test_count_empty(self):
        mgr = ConnectionManager()
        assert mgr.count == 0

    @pytest.mark.asyncio
    async def test_count_after_connect(self):
        mgr = ConnectionManager()
        mock_ws = MagicMock()
        mock_ws.accept = AsyncMock()
        await mgr.connect(mock_ws)
        assert mgr.count == 1

    @pytest.mark.asyncio
    async def test_count_after_disconnect(self):
        mgr = ConnectionManager()
        mock_ws = MagicMock()
        mock_ws.accept = AsyncMock()
        await mgr.connect(mock_ws)
        mgr.disconnect(mock_ws)
        assert mgr.count == 0
