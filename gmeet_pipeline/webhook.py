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
        self._workers: dict[str, asyncio.Task] = {}  # bot_id -> queue worker task

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
                # Cancel the queue worker
                worker = self._workers.pop(bot_id, None)
                if worker and not worker.done():
                    worker.cancel()
        elif event == "transcript.data":
            await self._handle_transcript(bot_id, data)
        elif event == "transcript.partial_data":
            self._handle_partial_transcript(bot_id, data)
        elif event == "participant_events.join":
            await self._handle_participant_join(bot_id, data)
        elif event == "participant_events.leave":
            await self._handle_participant_leave(bot_id, data)
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
                # Start the queue worker when bot enters meeting
                self._ensure_worker(bot_id)
            elif new_status == "ended":
                logger.info(f"Bot {bot_id} left meeting")
                worker = self._workers.pop(bot_id, None)
                if worker and not worker.done():
                    worker.cancel()

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

        # Deduplicate by timestamp
        if ts and ts == session.last_processed_ts:
            return
        if ts:
            session.last_processed_ts = ts

        # Add to transcript and conversation
        entry = {"speaker": speaker, "text": text, "timestamp": ts}
        session.transcript.append(entry)
        logger.info(f"[{speaker}]: {text}")

        session.conversation.append({"role": "user", "content": f"{speaker}: {text}"})

        # Queue the message instead of fire-and-forget
        await session.response_queue.put({"speaker": speaker, "text": text, "ts": ts})
        queue_depth = session.response_queue.qsize()
        logger.info(f"Queued {speaker}'s message (queue depth: {queue_depth})")
        session.pipeline_state = "queuing" if queue_depth > 1 else "idle"

        # Ensure the queue worker is running
        self._ensure_worker(bot_id)

    def _handle_partial_transcript(self, bot_id: str, data: dict):
        transcript_data = data.get("data", {})
        participant = transcript_data.get("participant", {})
        speaker = participant.get("name") or f"Participant-{participant.get('id', '?')}"
        words = transcript_data.get("words", [])
        text = " ".join(w.get("text", "") for w in words).strip()
        if text:
            logger.debug(f"[{speaker}] (partial): {text}")
            # Mark speaker as currently speaking
            session = self.registry.get(bot_id)
            if session and speaker in session.participants:
                session.participants[speaker]["is_speaking"] = True

    async def _handle_participant_join(self, bot_id: str, data: dict):
        participant = data.get("data", {}).get("participant", {})
        name = participant.get("name") or f"Participant-{participant.get('id', '?')}"
        logger.info(f"Participant joined: {name}")
        session = self.registry.get(bot_id)
        if session:
            session.participants[name] = {
                "join_ts": datetime.now(timezone.utc).isoformat(),
                "is_speaking": False,
            }

    async def _handle_participant_leave(self, bot_id: str, data: dict):
        participant = data.get("data", {}).get("participant", {})
        name = participant.get("name") or f"Participant-{participant.get('id', '?')}"
        logger.info(f"Participant left: {name}")
        session = self.registry.get(bot_id)
        if session and name in session.participants:
            del session.participants[name]

    def _ensure_worker(self, bot_id: str):
        """Ensure the queue worker task is running for this bot."""
        worker = self._workers.get(bot_id)
        if worker is None or worker.done():
            self._workers[bot_id] = asyncio.create_task(
                self._queue_worker(bot_id), name=f"queue-worker-{bot_id[:8]}"
            )

    async def _queue_worker(self, bot_id: str):
        """Processes messages from the response queue one at a time."""
        session = self.registry.get(bot_id)
        if not session:
            return

        logger.info(f"Queue worker started for bot {bot_id[:8]}")
        try:
            while True:
                session = self.registry.get(bot_id)
                if not session or session.status in ("ended", "leaving"):
                    break

                # Wait for the next message
                try:
                    msg = await asyncio.wait_for(
                        session.response_queue.get(), timeout=30.0
                    )
                except asyncio.TimeoutError:
                    # No messages for 30s, check if bot still active
                    continue
                except asyncio.CancelledError:
                    break

                # Drain the queue — batch all pending messages into one response
                messages = [msg]
                while not session.response_queue.empty():
                    try:
                        extra = session.response_queue.get_nowait()
                        messages.append(extra)
                    except asyncio.QueueEmpty:
                        break

                if len(messages) > 1:
                    logger.info(f"Batching {len(messages)} queued messages")

                # Process the batch
                await self._process_batch(bot_id, messages)

        except asyncio.CancelledError:
            logger.info(f"Queue worker cancelled for bot {bot_id[:8]}")
        except Exception as e:
            logger.error(f"Queue worker error: {type(e).__name__}: {e}", exc_info=True)

    async def _process_batch(self, bot_id: str, messages: list[dict]):
        """Process a batch of speaker messages through LLM → TTS → audio."""
        session = self.registry.get(bot_id)
        if not session:
            return

        try:
            await self._process_batch_inner(bot_id, messages)
        except Exception as e:
            logger.error(f"_process_batch EXCEPTION: {type(e).__name__}: {e}", exc_info=True)
            if session:
                session.pipeline_state = "idle"

    async def _process_batch_inner(self, bot_id: str, messages: list[dict]):
        """Process a batch of speaker messages through LLM → TTS → audio."""
        session = self.registry.get(bot_id)
        if not session:
            return

        t0 = asyncio.get_event_loop().time()

        # Build context from the batch
        if len(messages) == 1:
            speaker = messages[0]["speaker"]
            text = messages[0]["text"]
            context_msg = f"{speaker} said: {text}"
        else:
            # Multi-speaker: combine all messages
            parts = [f"{m['speaker']} said: {m['text']}" for m in messages]
            context_msg = " | ".join(parts)
            speaker = messages[-1]["speaker"]  # last speaker for logging

        # LLM
        session.pipeline_state = "llm"
        t1 = asyncio.get_event_loop().time()
        bot_state = {
            "expanded_entries": session.expanded_entries,
            "participants": list(session.participants.keys()),
        }
        response_text = await self.llm.generate(session.conversation, context_msg, bot_state=bot_state)
        t2 = asyncio.get_event_loop().time()
        llm_ms = (t2 - t1) * 1000
        session.last_llm_ms = int(llm_ms)

        if not response_text:
            logger.info(f"Hank chose to stay silent after {speaker}'s message")
            session.pipeline_state = "idle"
            return

        logger.info(f"Hank responds: {response_text}")

        session.conversation.append({"role": "assistant", "content": response_text})
        session.transcript.append({
            "speaker": "Hank Bob",
            "text": response_text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # TTS
        session.pipeline_state = "tts"
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
            session.last_tts_ms = int(tts_ms)

            if result:
                total_ms = (t4 - t0) * 1000
                session.last_total_ms = int(total_ms)
                session.pipeline_state = "speaking"
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
                session.pipeline_state = "idle"
        finally:
            session.speaking = False
            # Pipeline returns to idle after audio is queued
            # (actual speaking state is managed by agent page playback)
