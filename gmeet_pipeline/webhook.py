"""Recall.ai webhook handler — processes transcript and participant events."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from .state import BotRegistry, BotSession
from .llm.base import BaseLLM
from .tts.base import BaseTTS
from .ws_manager import ConnectionManager


logger = logging.getLogger("gmeet_pipeline.webhook")


class RecallWebhookHandler:
    """Handles Recall.ai webhook events and orchestrates the response pipeline."""

    def __init__(
        self,
        registry: BotRegistry,
        llm: BaseLLM,
        tts: BaseTTS,
        ws_manager: Optional[ConnectionManager] = None,
        audio_queue: Optional[dict] = None,
    ):
        self.registry = registry
        self.llm = llm
        self.tts = tts
        self.ws_manager = ws_manager
        self.audio_queue = audio_queue if audio_queue is not None else {}

    async def handle(self, body: dict) -> dict:
        """Process a Recall.ai webhook payload."""
        event = body.get("event", "")
        data = body.get("data", {})

        bot_id = data.get("bot", {}).get("id") or data.get("bot_id")
        logger.info(f"Recall webhook: event={event} bot_id={bot_id}")
        logger.debug(f"Full payload keys: {list(body.keys())} data_keys: {list(data.keys())}")

        if event in ("bot.status_change", "status_change"):
            await self._handle_status_change(bot_id, data)
        elif event in ("bot.call_ended", "call_ended"):
            logger.info(f"Bot {bot_id} call ended")
            session = self.registry.get(bot_id)
            if session:
                session.status = "ended"
        elif event == "transcript.data":
            await self._handle_transcript(bot_id, data)
        elif event == "transcript.partial_data":
            self._handle_partial_transcript(bot_id, data)
        elif event == "participant_events.join":
            self._handle_participant_join(data)
        elif event == "participant_events.leave":
            self._handle_participant_leave(data)
        else:
            logger.info(f"Unhandled event: {event}")

        return {"ok": True}

    async def _handle_status_change(self, bot_id: str, data: dict):
        status = data.get("status", {})
        new_status = status.get("code", "") if isinstance(status, dict) else str(status)
        logger.info(f"Bot {bot_id} status: {new_status}")

        session = self.registry.get(bot_id)
        if session:
            session.status = new_status
            if new_status == "in_meeting":
                logger.info(f"Bot {bot_id} joined meeting!")
            elif new_status == "ended":
                logger.info(f"Bot {bot_id} left meeting")

    async def _handle_transcript(self, bot_id: str, data: dict):
        transcript_data = data.get("data", {})
        participant = transcript_data.get("participant", {})
        speaker = participant.get("name") or f"Participant-{participant.get('id', '?')}"
        words = transcript_data.get("words", [])
        text = " ".join(w.get("text", "") for w in words).strip()

        if not text:
            return

        session = self.registry.get(bot_id)
        if not session:
            logger.warning(f"Transcript for unknown bot {bot_id}")
            return

        # Skip if Hank Bob is the speaker
        if "hank" in speaker.lower() or "bob" in speaker.lower():
            return

        ts = transcript_data.get("started_at") or datetime.now(timezone.utc).isoformat()

        # Add to transcript and conversation
        entry = {"speaker": speaker, "text": text, "timestamp": ts}
        session.transcript.append(entry)
        logger.info(f"[{speaker}]: {text}")

        session.conversation.append({"role": "user", "content": f"{speaker}: {text}"})

        # Fire and forget the response pipeline
        asyncio.create_task(self._process_and_respond(bot_id, speaker, text, ts))

    def _handle_partial_transcript(self, bot_id: str, data: dict):
        transcript_data = data.get("data", {})
        participant = transcript_data.get("participant", {})
        speaker = participant.get("name") or f"Participant-{participant.get('id', '?')}"
        words = transcript_data.get("words", [])
        text = " ".join(w.get("text", "") for w in words).strip()
        if text:
            logger.debug(f"[{speaker}] (partial): {text}")

    def _handle_participant_join(self, data: dict):
        participant = data.get("data", {}).get("participant", {})
        name = participant.get("name") or f"Participant-{participant.get('id', '?')}"
        logger.info(f"Participant joined: {name}")

    def _handle_participant_leave(self, data: dict):
        participant = data.get("data", {}).get("participant", {})
        name = participant.get("name") or f"Participant-{participant.get('id', '?')}"
        logger.info(f"Participant left: {name}")

    async def _process_and_respond(self, bot_id: str, speaker: str, text: str, ts: str = ""):
        """Full pipeline: transcript → LLM → TTS → audio delivery."""
        try:
            await self._process_and_respond_inner(bot_id, speaker, text, ts)
        except Exception as e:
            logger.error(f"_process_and_respond EXCEPTION: {type(e).__name__}: {e}", exc_info=True)

    async def _process_and_respond_inner(self, bot_id: str, speaker: str, text: str, ts: str = ""):
        """Full pipeline: transcript → LLM → TTS → audio delivery."""
        session = self.registry.get(bot_id)
        if not session:
            return

        lock = session.respond_lock
        if not lock:
            return

        # Deduplicate by timestamp
        if ts and ts == session.last_processed_ts:
            return
        if ts:
            session.last_processed_ts = ts

        if lock.locked():
            logger.info("Skipping — bot already processing a response")
            return

        async with lock:
            session = self.registry.get(bot_id)
            if not session:
                return

            t0 = asyncio.get_event_loop().time()

            # Brief debounce
            await asyncio.sleep(0.5)

            session = self.registry.get(bot_id)
            if not session:
                return

            # LLM
            context_msg = f"{speaker} said: {text}"
            t1 = asyncio.get_event_loop().time()
            bot_state = {
                "expanded_entries": session.expanded_entries,
            }
            response_text = await self.llm.generate(session.conversation, context_msg, bot_state=bot_state)
            t2 = asyncio.get_event_loop().time()
            llm_ms = (t2 - t1) * 1000

            if not response_text:
                logger.info(f"Hank chose to stay silent after {speaker}'s message")
                return

            logger.info(f"Hank responds: {response_text}")

            session.conversation.append({"role": "assistant", "content": response_text})
            session.transcript.append({
                "speaker": "Hank Bob",
                "text": response_text,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            # TTS
            session.speaking = True
            t3 = asyncio.get_event_loop().time()
            logger.info(f"TTS generate starting for: {response_text[:60]}...")
            try:
                result = await asyncio.wait_for(
                    self.tts.generate(response_text, bot_id),
                    timeout=30.0,
                )
                t4 = asyncio.get_event_loop().time()
                tts_ms = (t4 - t3) * 1000

                if result:
                    total_ms = (t4 - t0) * 1000
                    logger.info(
                        f"BENCH | llm={llm_ms:.0f}ms tts={tts_ms:.0f}ms "
                        f"total_server={total_ms:.0f}ms | audio={result}"
                    )
                    # Add to audio queue so the agent page can play it
                    self.audio_queue.setdefault(bot_id, []).append({
                        "filename": result,
                        "text": response_text,
                    })
                    logger.info(f"Audio queue now: {list(self.audio_queue.keys())} = {[(k, len(v)) for k, v in self.audio_queue.items()]}")
                else:
                    logger.error("TTS generation failed — no audio produced")
            finally:
                session.speaking = False
