"""Bot session state management — replaces raw active_bots dict."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class BotSession:
    """Represents a single bot instance's runtime state."""

    bot_id: str
    meeting_url: str
    status: str = "joining"
    transcript: list = field(default_factory=list)  # [{speaker, text, timestamp}]
    conversation: list = field(default_factory=list)  # [{role, content}] for LLM
    speaking: bool = False
    respond_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    response_queue: asyncio.Queue = field(default_factory=asyncio.Queue)  # queued speaker messages
    last_processed_ts: str = ""
    expanded_entries: set = field(default_factory=set)  # voice gateway EXPAND tracking
    participants: dict = field(default_factory=dict)  # {name: {join_ts, is_speaking}}
    response_mode: str = "active"  # active | silent_transcribe
    mode_events: list = field(default_factory=list)  # [{mode, reason, speaker, text, timestamp}]
    action_candidates: list = field(default_factory=list)  # generic post-call action candidates
    pipeline_state: str = "idle"  # idle | queuing | llm | tts | speaking
    last_llm_ms: int = 0
    last_tts_ms: int = 0
    last_total_ms: int = 0
    post_call_finalized: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class BotRegistry:
    """Thread-safe registry of all active BotSession instances."""

    def __init__(self) -> None:
        self._bots: dict[str, BotSession] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    async def create(self, bot_id: str, meeting_url: str, **kwargs) -> BotSession:
        """Create and register a new BotSession.

        Any extra keyword arguments are forwarded to the BotSession constructor
        (e.g. status="active", last_processed_ts="...").
        """
        session = BotSession(bot_id=bot_id, meeting_url=meeting_url, **kwargs)
        async with self._lock:
            self._bots[bot_id] = session
        return session

    def get(self, bot_id: str) -> Optional[BotSession]:
        """Return a BotSession by id, or None if not found."""
        return self._bots.get(bot_id)

    async def remove(self, bot_id: str) -> Optional[BotSession]:
        """Remove and return a BotSession by id, or None if not found."""
        async with self._lock:
            return self._bots.pop(bot_id, None)

    async def list_bots(self) -> dict:
        """Return a serialisable snapshot of all active bots."""
        async with self._lock:
            return {
                bid: {
                    "bot_id": s.bot_id,
                    "meeting_url": s.meeting_url,
                    "status": s.status,
                    "speaking": s.speaking,
                    "pipeline_state": s.pipeline_state,
                    "last_llm_ms": s.last_llm_ms,
                    "last_tts_ms": s.last_tts_ms,
                    "last_total_ms": s.last_total_ms,
                    "queue_depth": s.response_queue.qsize(),
                    "participants": list(s.participants.keys()),
                    "response_mode": s.response_mode,
                    "mode_event_count": len(s.mode_events),
                    "action_candidate_count": len(s.action_candidates),
                    "last_processed_ts": s.last_processed_ts,
                    "transcript_count": len(s.transcript),
                    "conversation_count": len(s.conversation),
                    "created_at": s.created_at,
                }
                for bid, s in self._bots.items()
            }

    def __contains__(self, bot_id: str) -> bool:
        """Allow `bot_id in registry` syntax."""
        return bot_id in self._bots
