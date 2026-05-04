"""FastAPI server — routes only, all logic delegated to injected components."""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .config import GmeetSettings
from .state import BotRegistry
from .transports.base import BaseTransport
from .llm.base import BaseLLM
from .tts.base import BaseTTS
from .ws_manager import ConnectionManager
from .agent_page import get_agent_html
from .webhook import RecallWebhookHandler


logger = logging.getLogger("gmeet_pipeline.server")

# Paths that the bot's Chrome tab needs — no auth required
_OPEN_PATHS = {"/", "/health"}
_OPEN_PREFIXES = ("/audio/", "/api/audio-queue", "/api/transcript", "/api/debug", "/api/session-state", "/ws/")


class AuthMiddleware(BaseHTTPMiddleware):
    """API key auth for admin endpoints; webhook secret for Recall; open for bot-facing."""

    def __init__(self, app, api_key: str, webhook_secret: str):
        super().__init__(app)
        self.api_key = api_key
        self.webhook_secret = webhook_secret

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Open paths — no auth
        if path in _OPEN_PATHS or any(path.startswith(p) for p in _OPEN_PREFIXES):
            return await call_next(request)

        # Webhook — validate secret in URL path
        if path.startswith("/webhook/recall"):
            if self.webhook_secret:
                # Expected path: /webhook/recall/{secret}
                parts = path.rstrip("/").split("/")
                if len(parts) >= 4 and parts[3] == self.webhook_secret:
                    # Rewrite to the handler path without the secret
                    request.scope["path"] = "/webhook/recall"
                    request.scope["raw_path"] = b"/webhook/recall"
                    return await call_next(request)
                return JSONResponse({"error": "invalid webhook secret"}, status_code=401)
            # No secret configured — allow (backward compat)
            return await call_next(request)

        # Admin endpoints — require Bearer token
        if self.api_key:
            auth = request.headers.get("Authorization", "")
            if auth == f"Bearer {self.api_key}":
                return await call_next(request)
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        # No API key configured — allow all (backward compat)
        return await call_next(request)


class GmeetServer:
    """Wires together all components and exposes FastAPI routes."""

    def __init__(
        self,
        settings: GmeetSettings,
        transport: BaseTransport,
        llm: BaseLLM,
        tts: BaseTTS,
        ws_manager: ConnectionManager,
        registry: BotRegistry,
        memory_snapshot=None,
    ):
        self.settings = settings
        self.transport = transport
        self.llm = llm
        self.tts = tts
        self.ws_manager = ws_manager
        self.registry = registry
        self.memory_snapshot = memory_snapshot
        self.audio_queue: dict = {}

        self.webhook_handler = RecallWebhookHandler(
            registry=registry,
            llm=llm,
            tts=tts,
            ws_manager=ws_manager if hasattr(tts, 'ws_manager') else None,
            audio_queue=self.audio_queue,
            artifact_dir=str(self.settings.meeting_artifacts_dir),
            post_call_enabled=self.settings.post_call_hermes_enabled,
            post_call_hermes_cmd=self.settings.post_call_hermes_cmd,
            post_call_model=self.settings.post_call_model,
            post_call_provider=self.settings.post_call_provider,
            post_call_toolsets=self.settings.post_call_toolsets,
            post_call_inbox_dir=str(self.settings.action_inbox_dir),
            post_call_max_parallel_sessions=self.settings.post_call_max_parallel_sessions,
            post_call_dry_run=self.settings.post_call_dry_run,
        )

        self.app = FastAPI(title="Hank Bob Meeting Agent")
        self.app.add_middleware(
            AuthMiddleware,
            api_key=self.settings.api_key,
            webhook_secret=self.settings.webhook_secret,
        )
        self._register_routes()

    def _register_routes(self):
        app = self.app

        @app.get("/", response_class=HTMLResponse)
        async def agent_page():
            return get_agent_html(
                tts_backend=self.settings.tts_backend,
                memory_enabled=self.memory_snapshot is not None,
            )

        @app.websocket("/ws/audio")
        async def ws_audio(websocket: WebSocket):
            """WebSocket endpoint for streaming PCM audio (ElevenLabs mode)."""
            await self.ws_manager.connect(websocket)
            try:
                while True:
                    data = await websocket.receive_text()
                    if data == "ping":
                        await websocket.send_json({"type": "pong"})
            except WebSocketDisconnect:
                self.ws_manager.disconnect(websocket)

        @app.get("/api/transcript")
        async def get_transcript(since: int = 0):
            entries = []
            for bot_id in self.registry._bots:
                session = self.registry.get(bot_id)
                if session:
                    for i, entry in enumerate(session.transcript):
                        if i >= since:
                            entries.append({**entry, "idx": i})
            return {"entries": entries}

        @app.get("/audio/{filename}")
        async def serve_audio(filename: str):
            filepath = Path(self.settings.audio_dir) / filename
            if not filepath.exists():
                raise HTTPException(404, "Audio not found")
            media_type = "audio/mpeg" if filename.endswith(".mp3") else "audio/wav"
            return FileResponse(
                str(filepath),
                media_type=media_type,
                headers={"Cache-Control": "no-cache"},
            )

        @app.post("/api/bot/join")
        async def join_meeting(request: Request):
            body = await request.json()
            meeting_url = body.get("meeting_url")
            if not meeting_url:
                raise HTTPException(400, "meeting_url required")

            bot_name = body.get("bot_name", "Hank Bob")

            try:
                data = await self.transport.join(meeting_url, bot_name)
            except Exception as e:
                raise HTTPException(500, str(e))

            bot_id = data.get("id")

            session = await self.registry.create(bot_id, meeting_url)
            logger.info(f"Bot {bot_id} created, joining {meeting_url}")
            return {"bot_id": bot_id, "status": "joining", "data": data}

        @app.post("/api/bot/{bot_id}/leave")
        async def leave_meeting(bot_id: str):
            try:
                await self.transport.leave(bot_id)
            except Exception as e:
                logger.error(f"Leave error: {e}")

            session = self.registry.get(bot_id)
            if session:
                session.status = "leaving"
            return {"status": "leaving"}

        @app.post("/webhook/recall")
        async def recall_webhook(request: Request):
            body = await request.json()
            return await self.webhook_handler.handle(body)

        @app.get("/api/bots")
        async def list_bots():
            return await self.registry.list_bots()

        @app.get("/api/session-state")
        async def session_state():
            """Rich session state for agent page debug overlay."""
            bots = []
            for bot_id in self.registry._bots:
                session = self.registry.get(bot_id)
                if not session:
                    continue
                bots.append({
                    "bot_id": bot_id,
                    "status": session.status,
                    "pipeline_state": session.pipeline_state,
                    "speaking": session.speaking,
                    "queue_depth": session.response_queue.qsize(),
                    "participants": list(session.participants.keys()),
                    "response_mode": session.response_mode,
                    "action_candidate_count": len(session.action_candidates),
                    "last_llm_ms": session.last_llm_ms,
                    "last_tts_ms": session.last_tts_ms,
                    "last_total_ms": session.last_total_ms,
                    "transcript_count": len(session.transcript),
                    "last_transcript": session.transcript[-1] if session.transcript else None,
                })
            return {"bots": bots, "llm_routing": self.settings.llm_routing, "tts_backend": self.settings.tts_backend}

        @app.get("/api/audio-queue")
        async def get_audio_queue():
            all_items = []
            for bot_id, items in self.audio_queue.items():
                all_items.extend(items)
            return {"items": all_items, "server_time": datetime.now(timezone.utc).isoformat()}

        @app.post("/api/debug")
        async def client_debug(request: Request):
            body = await request.json()
            msg = body.get("msg", "")
            logger.info(f"[CLIENT] {msg}")
            return {"ok": True}

        @app.get("/health")
        async def health():
            health_data = {
                "status": "ok",
                "active_bots": len(self.registry._bots),
                "ws_clients": self.ws_manager.count,
                "tts_backend": self.settings.tts_backend,
                "llm_routing": self.settings.llm_routing,
            }
            if self.memory_snapshot:
                health_data["memory_entries"] = len(self.memory_snapshot.entries)
                health_data["memory_snapshot_age"] = round(
                    __import__("time").time() - self.memory_snapshot.built_at, 1
                )
            return health_data

        @app.get("/api/memory")
        async def get_memory_snapshot():
            if not self.memory_snapshot:
                return {"error": "Memory not enabled"}
            self.memory_snapshot.refresh_if_stale()
            return {
                "summary": self.memory_snapshot.summary,
                "entry_count": len(self.memory_snapshot.entries),
                "index": self.memory_snapshot.index_text,
                "entries": [
                    {"index": e["index"], "source": e["source"], "text": e["text"][:200]}
                    for e in self.memory_snapshot.entries
                ],
            }

        @app.post("/api/bench/setup")
        async def bench_setup():
            bot_id = f"bench-{uuid.uuid4().hex[:8]}"
            await self.registry.create(bot_id, "bench://synthetic")
            return {"bot_id": bot_id, "status": "ready for bench"}

        @app.post("/api/bench/tts")
        async def benchmark_tts_only(request: Request):
            body = await request.json()
            text = body.get("text", "I tell ya what, that propane grill is the only way to cook a proper burger.")
            bot_id = body.get("bot_id")

            session = self.registry.get(bot_id)
            if not session:
                raise HTTPException(400, f"bot_id required and must be active. Active: {list(self.registry._bots.keys())}")

            import time as _time
            t1 = _time.monotonic()
            result = await self.tts.generate(text, bot_id)
            t2 = _time.monotonic()

            timings = {
                "tts_ms": round((t2 - t1) * 1000),
                "total_ms": round((t2 - t1) * 1000),
                "input_text": text,
                "input_chars": len(text),
                "audio_file": result,
                "tts_backend": self.settings.tts_backend,
            }

            if result:
                fp = Path(self.settings.audio_dir) / result
                if fp.exists():
                    try:
                        import soundfile as sf
                        data, sr = sf.read(str(fp))
                        timings["audio_duration_s"] = round(len(data) / sr, 2)
                        timings["real_time_factor"] = round(
                            timings["tts_ms"] / (len(data) / sr * 1000), 3
                        )
                    except ImportError:
                        pass

            return {"timings": timings}

        @app.post("/api/bench")
        async def benchmark_pipeline(request: Request):
            body = await request.json()
            text = body.get("text", "Hey Hank, what do you think about MLX?")
            bot_id = body.get("bot_id")

            session = self.registry.get(bot_id)
            if not session:
                raise HTTPException(400, f"bot_id required and must be active. Active: {list(self.registry._bots.keys())}")

            lock = session.respond_lock
            if lock.locked():
                return {"error": "Bot is already processing a response", "status": "busy"}

            import time as _time

            async with lock:
                t0 = _time.monotonic()

                if not session.conversation:
                    session.conversation = [
                        {"role": "user", "content": "Ethan: Hey everyone, Hank Bob is here with us today."},
                        {"role": "assistant", "content": "Hey! Good to be here. What are we working on?"},
                    ]

                context_msg = f"Ethan said: {text}"
                t2 = _time.monotonic()
                bot_state = {"expanded_entries": session.expanded_entries}
                response_text = await self.llm.generate(session.conversation, context_msg, bot_state=bot_state)
                t3 = _time.monotonic()

                timings = {
                    "llm_ms": round((t3 - t2) * 1000),
                }

                if not response_text:
                    return {"error": "LLM chose to stay silent", "timings": timings}

                t4 = _time.monotonic()
                result = await self.tts.generate(response_text, bot_id)
                t5 = _time.monotonic()

                timings["tts_ms"] = round((t5 - t4) * 1000)
                timings["total_server_ms"] = round((t5 - t0) * 1000)
                timings["response_text"] = response_text
                timings["audio_file"] = result
                timings["tts_backend"] = self.settings.tts_backend

            return {"timings": timings}
