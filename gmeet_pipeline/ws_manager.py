"""WebSocket connection manager for streaming PCM audio to agent pages."""

import logging
from typing import Optional

from fastapi import WebSocket


logger = logging.getLogger("gmeet_pipeline.ws_manager")


class ConnectionManager:
    """Manages WebSocket connections to agent pages for streaming PCM audio."""

    def __init__(self):
        self.connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)
        logger.info(f"WS client connected. Total: {len(self.connections)}")

    def disconnect(self, ws: WebSocket):
        if ws in self.connections:
            self.connections.remove(ws)
        logger.info(f"WS client disconnected. Total: {len(self.connections)}")

    async def broadcast_binary(self, data: bytes):
        """Send raw PCM chunk to all connected agent pages."""
        dead = []
        for ws in self.connections:
            try:
                await ws.send_bytes(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    async def broadcast_json(self, data: dict):
        """Send JSON control message to all connected agent pages."""
        dead = []
        for ws in self.connections:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    @property
    def count(self) -> int:
        return len(self.connections)
