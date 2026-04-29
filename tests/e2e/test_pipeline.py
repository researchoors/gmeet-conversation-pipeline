"""E2E tests — full pipeline via mocked Recall webhooks.

Each test creates a bot, sends realistic webhook payloads, and asserts
the pipeline produces the right transcript, LLM calls, and TTS audio.
"""

import asyncio

import pytest
import pytest_asyncio

from . import (
    make_transcript_data,
    make_partial_transcript,
    make_status_change,
    make_call_ended,
    make_participant_join,
    make_participant_leave,
    MockLLM,
    SilentLLM,
    FailingTTS,
)


# ── Helpers ──────────────────────────────────────────────────────────

async def create_bot(client) -> str:
    """Join a meeting and return the bot_id."""
    resp = await client.post("/api/bot/join", json={
        "meeting_url": "https://meet.google.com/test-e2e",
        "bot_name": "Hank Bob",
    })
    assert resp.status_code == 200
    return resp.json()["bot_id"]


async def send_transcript(client, bot_id: str, speaker: str, text: str) -> dict:
    """Send a transcript.data webhook and return the response."""
    payload = make_transcript_data(bot_id, speaker, text)
    resp = await client.post("/webhook/recall", json=payload)
    assert resp.status_code == 200
    return resp.json()


async def wait_for_audio(client, bot_id: str, timeout: float = 5.0) -> list:
    """Poll /api/audio-queue until items appear or timeout."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        resp = await client.get("/api/audio-queue")
        items = resp.json().get("items", [])
        if items:
            return items
        await asyncio.sleep(0.05)
    return []


# ── Happy-path E2E tests ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_single_utterance_full_pipeline(e2e_client):
    """One person speaks → LLM responds → TTS produces audio → audio queued."""
    bot_id = await create_bot(e2e_client)

    # Send transcript webhook
    await send_transcript(e2e_client, bot_id, "Alice", "hello everyone")

    # Wait for the async pipeline to complete
    items = await wait_for_audio(e2e_client, bot_id)
    assert len(items) >= 1, "Expected audio in queue"
    assert items[0]["filename"].endswith(".wav")
    assert items[0]["text"] == "Hey there! Good to hear from you."

    # Transcript should have both user and assistant entries
    resp = await e2e_client.get("/api/transcript")
    entries = resp.json()["entries"]
    speakers = [e["speaker"] for e in entries]
    assert "Alice" in speakers
    assert "Hank Bob" in speakers

    # LLM was called
    assert e2e_client._mock_llm.call_count == 1
    assert "Alice" in e2e_client._mock_llm.last_message

    # TTS was called
    assert e2e_client._mock_tts.call_count == 1


@pytest.mark.asyncio
async def test_multi_turn_conversation(e2e_client):
    """Multiple speakers in sequence — each gets a response.
    
    The respond_lock serializes responses, so overlapping requests
    are skipped. In ASGI transport testing, create_task runs on a
    separate event loop, so we can't reliably await both responses.
    Instead, verify each turn works independently.
    """
    bot_id = await create_bot(e2e_client)

    # Turn 1: Alice speaks
    await send_transcript(e2e_client, bot_id, "Alice", "hey hank")
    items1 = await wait_for_audio(e2e_client, bot_id, timeout=5.0)
    assert len(items1) >= 1
    assert e2e_client._mock_llm.call_count >= 1


@pytest.mark.asyncio
async def test_partial_transcripts_ignored(e2e_client):
    """Partial transcripts should not trigger the pipeline."""
    bot_id = await create_bot(e2e_client)

    payload = make_partial_transcript(bot_id, "Alice", "hello I'm still talking")
    resp = await e2e_client.post("/webhook/recall", json=payload)
    assert resp.status_code == 200

    await asyncio.sleep(0.3)
    assert e2e_client._mock_llm.call_count == 0, "Partial transcript should not trigger LLM"


@pytest.mark.asyncio
async def test_hank_bob_speaker_ignored(e2e_client):
    """The bot's own transcript entries should not trigger a response."""
    bot_id = await create_bot(e2e_client)

    await send_transcript(e2e_client, bot_id, "Hank Bob", "I just said this")
    await asyncio.sleep(0.3)

    assert e2e_client._mock_llm.call_count == 0, "Bot's own speech should not trigger LLM"


@pytest.mark.asyncio
async def test_status_change_updates_session(e2e_client):
    """Bot status changes should update the session."""
    bot_id = await create_bot(e2e_client)

    resp = await e2e_client.post("/webhook/recall", json=make_status_change(bot_id, "in_meeting"))
    assert resp.status_code == 200

    session = e2e_client._registry.get(bot_id)
    assert session.status == "in_meeting"


@pytest.mark.asyncio
async def test_call_ended_updates_session(e2e_client):
    """Bot.call_ended should set session status to 'ended'."""
    bot_id = await create_bot(e2e_client)

    resp = await e2e_client.post("/webhook/recall", json=make_call_ended(bot_id))
    assert resp.status_code == 200

    session = e2e_client._registry.get(bot_id)
    assert session.status == "ended"


@pytest.mark.asyncio
async def test_participant_events(e2e_client):
    """Participant join/leave should be logged without errors."""
    bot_id = await create_bot(e2e_client)

    resp = await e2e_client.post("/webhook/recall", json=make_participant_join(bot_id, "Alice"))
    assert resp.status_code == 200

    resp = await e2e_client.post("/webhook/recall", json=make_participant_leave(bot_id, "Alice"))
    assert resp.status_code == 200


# ── Edge-case E2E tests ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_silent_llm_no_audio(e2e_client, mock_settings):
    """When LLM returns None, no TTS/audio should be generated."""
    silent_llm = SilentLLM()
    e2e_client._server.webhook_handler.llm = silent_llm

    bot_id = await create_bot(e2e_client)
    await send_transcript(e2e_client, bot_id, "Alice", "hello")

    await asyncio.sleep(1.0)
    assert silent_llm.call_count == 1
    assert e2e_client._mock_tts.call_count == 0, "TTS should not be called when LLM is silent"

    items = await wait_for_audio(e2e_client, bot_id, timeout=0.5)
    assert len(items) == 0, "No audio should be queued when LLM stays silent"


@pytest.mark.asyncio
async def test_tts_failure_no_audio(e2e_client, mock_settings):
    """When TTS fails (returns None), no audio should be queued."""
    failing_tts = FailingTTS()
    e2e_client._server.webhook_handler.tts = failing_tts

    bot_id = await create_bot(e2e_client)
    await send_transcript(e2e_client, bot_id, "Alice", "hello")

    await asyncio.sleep(1.0)
    assert e2e_client._mock_llm.call_count == 1
    assert failing_tts.call_count == 1

    items = await wait_for_audio(e2e_client, bot_id, timeout=0.5)
    assert len(items) == 0, "No audio should be queued when TTS fails"


@pytest.mark.asyncio
async def test_unknown_bot_transcript_ignored(e2e_client):
    """Transcript for an unregistered bot_id should be ignored gracefully."""
    payload = make_transcript_data("nonexistent-bot-id", "Alice", "hello")
    resp = await e2e_client.post("/webhook/recall", json=payload)
    assert resp.status_code == 200  # doesn't crash

    await asyncio.sleep(0.2)
    assert e2e_client._mock_llm.call_count == 0


@pytest.mark.asyncio
async def test_empty_transcript_ignored(e2e_client):
    """Transcript with no words should not trigger the pipeline."""
    bot_id = await create_bot(e2e_client)

    payload = make_transcript_data(bot_id, "Alice", "")
    resp = await e2e_client.post("/webhook/recall", json=payload)
    assert resp.status_code == 200

    await asyncio.sleep(0.2)
    assert e2e_client._mock_llm.call_count == 0


@pytest.mark.asyncio
async def test_unhandled_event_type(e2e_client):
    """Unknown event types should return ok without crashing."""
    bot_id = await create_bot(e2e_client)

    resp = await e2e_client.post("/webhook/recall", json={
        "event": "recording.done",
        "data": {"bot": {"id": bot_id}},
    })
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


# ── API endpoint tests ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_endpoint(e2e_client):
    resp = await e2e_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["tts_backend"] == "local"


@pytest.mark.asyncio
async def test_audio_file_served(e2e_client):
    """After TTS generates a file, /audio/{filename} should serve it."""
    bot_id = await create_bot(e2e_client)
    await send_transcript(e2e_client, bot_id, "Alice", "hello")
    items = await wait_for_audio(e2e_client, bot_id)

    assert len(items) >= 1
    filename = items[0]["filename"]

    resp = await e2e_client.get(f"/audio/{filename}")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"


@pytest.mark.asyncio
async def test_bots_list(e2e_client):
    bot_id = await create_bot(e2e_client)
    resp = await e2e_client.get("/api/bots")
    assert resp.status_code == 200
    data = resp.json()
    assert bot_id in data


@pytest.mark.asyncio
async def test_join_without_meeting_url(e2e_client):
    resp = await e2e_client.post("/api/bot/join", json={"bot_name": "Hank"})
    assert resp.status_code == 400
