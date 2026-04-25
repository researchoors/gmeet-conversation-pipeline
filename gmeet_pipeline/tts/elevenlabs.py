"""ElevenLabs TTS backend — streaming WebSocket with HTTP REST fallback.

Ported from meeting_agent.py (lines 175-307).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from pathlib import Path
from typing import Optional

import httpx
import websockets

from .base import BaseTTS

logger = logging.getLogger("gmeet_pipeline.tts.elevenlabs")


class ElevenLabsTTS(BaseTTS):
    """Generate speech via the ElevenLabs WebSocket streaming API.

    On success the PCM audio is broadcast through the supplied *ws_manager*
    (ConnectionManager) using the same protocol as the original monolith:

    1. ``start`` JSON  — includes ``bot_id``, ``text``, ``sampleRate``
    2. Binary PCM payload (signed 16-bit LE, 22050 Hz)
    3. ``end`` JSON

    If the WebSocket stream fails the class automatically falls back to the
    synchronous REST endpoint, saves an MP3, and broadcasts a
    ``fallback_audio`` JSON message instead.
    """

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        model_id: str,
        audio_dir: Path,
        ws_manager,  # ConnectionManager — imported here to avoid circular dep
    ) -> None:
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.audio_dir = audio_dir
        self.ws_manager = ws_manager

        # Ensure the audio directory exists
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def generate(self, text: str, bot_id: str) -> Optional[str]:
        """Return an identifier for the generated audio, or ``None``.

        The actual audio is streamed / broadcast via *ws_manager* as a
        side-effect (matching the original monolith behaviour).
        """
        try:
            return await self._generate_streaming(text, bot_id)
        except Exception as exc:
            logger.error(f"TTS streaming failed: {exc}, falling back to REST")
            return await self._generate_fallback(text, bot_id)

    # ------------------------------------------------------------------
    # Streaming path (WebSocket)
    # ------------------------------------------------------------------

    async def _generate_streaming(self, text: str, bot_id: str) -> Optional[str]:
        """Fetch TTS via ElevenLabs WebSocket, buffer all PCM chunks, then
        broadcast as one complete blob.

        This avoids the choppiness caused by chunked PCM playback through
        Recall's tab audio capture.  The tradeoff is ~1-2 s extra latency
        (waiting for full audio), but the output is clean.
        """
        uri = (
            f"wss://api.elevenlabs.io/v1/text-to-speech/"
            f"{self.voice_id}/stream-input"
        )

        async with websockets.connect(uri) as ws:
            # Beginning of stream — config + auth
            bos_message = {
                "text": " ",
                "voice_settings": {
                    "stability": 0.7,
                    "similarity_boost": 0.75,
                },
                "xi_api_key": self.api_key,
                "model_id": self.model_id,
                "output_format": "pcm_22050",
            }
            await ws.send(json.dumps(bos_message))

            # Send the actual text with flush
            text_message = {
                "text": text,
                "flush": True,
            }
            await ws.send(json.dumps(text_message))

            pcm_buffer = bytearray()
            chunk_count = 0

            while True:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=15)
                except asyncio.TimeoutError:
                    logger.warning("ElevenLabs stream timeout — closing")
                    break

                if isinstance(message, bytes):
                    pcm_buffer.extend(message)
                    chunk_count += 1
                elif isinstance(message, str):
                    data = json.loads(message)
                    if data.get("isFinal"):
                        break
                    elif data.get("error"):
                        logger.error(
                            f"ElevenLabs streaming error: {data['error']}"
                        )
                        break
                    elif "audio" in data:
                        try:
                            pcm_bytes = base64.b64decode(data["audio"])
                            pcm_buffer.extend(pcm_bytes)
                            chunk_count += 1
                        except Exception as exc:
                            logger.error(f"Failed to decode audio chunk: {exc}")

        if not pcm_buffer:
            logger.warning(
                "No PCM data received from ElevenLabs, falling back"
            )
            return await self._generate_fallback(text, bot_id)

        logger.info(
            f"TTS buffered: {chunk_count} chunks, "
            f"{len(pcm_buffer)} bytes PCM for bot {bot_id}"
        )

        # Broadcast the complete PCM as one blob — agent page plays it as a
        # single AudioBuffer
        await self.ws_manager.broadcast_json(
            {
                "type": "start",
                "bot_id": bot_id,
                "text": text,
                "sampleRate": 22050,
            }
        )
        await self.ws_manager.broadcast_binary(bytes(pcm_buffer))
        await self.ws_manager.broadcast_json(
            {
                "type": "end",
                "bot_id": bot_id,
            }
        )

        return f"streamed:{bot_id}"

    # ------------------------------------------------------------------
    # Fallback path (REST / synchronous)
    # ------------------------------------------------------------------

    async def _generate_fallback(self, text: str, bot_id: str) -> Optional[str]:
        """Non-streaming fallback — generates full MP3 and broadcasts URL."""
        filename = f"tts_{uuid.uuid4().hex[:8]}.mp3"
        filepath = self.audio_dir / filename

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/"
                    f"{self.voice_id}",
                    headers={
                        "xi-api-key": self.api_key,
                        "Content-Type": "application/json",
                        "Accept": "audio/mpeg",
                    },
                    json={
                        "text": text,
                        "model_id": self.model_id,
                        "voice_settings": {
                            "stability": 0.7,
                            "similarity_boost": 0.75,
                        },
                    },
                )

                if resp.status_code != 200:
                    logger.error(
                        f"TTS fallback error: {resp.status_code} "
                        f"{resp.text[:200]}"
                    )
                    return None

                filepath.write_bytes(resp.content)

            # Broadcast fallback audio URL via ws_manager
            await self.ws_manager.broadcast_json(
                {
                    "type": "start",
                    "bot_id": bot_id,
                    "text": text,
                }
            )
            await self.ws_manager.broadcast_json(
                {
                    "type": "fallback_audio",
                    "bot_id": bot_id,
                    # The caller (server.py) is responsible for constructing
                    # the full public URL; we emit the filename so the
                    # consuming layer can build /audio/{filename}
                    "filename": filename,
                }
            )
            await self.ws_manager.broadcast_json(
                {
                    "type": "end",
                    "bot_id": bot_id,
                }
            )

            logger.info(f"TTS fallback: {filename}")
            return filename

        except Exception as exc:
            logger.error(f"TTS fallback failed: {exc}")
            return None
