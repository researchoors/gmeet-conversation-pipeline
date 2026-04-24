"""
Hank Bob Meeting Agent — Recall.ai integration
Streaming audio pipeline: transcript → LLM → TTS streaming (ElevenLabs WebSocket) → PCM via WebSocket → meeting
"""
import os
import json
import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import httpx
import uvicorn
import websockets

# Config
RECALL_API_KEY = os.environ.get("RECALL_API_KEY", "")
RECALL_BASE = "https://us-west-2.recall.ai/api/v1"
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
ELEVENLABS_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE = os.environ.get("ELEVENLABS_VOICE_ID", "os.environ.get(ELEVENLABS_VOICE_ID, default_voice_id)")
ELEVENLABS_MODEL = "eleven_multilingual_v2"
LLM_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4")
PORT = 9120
AUDIO_DIR = Path.home() / ".hermes" / "audio_cache" / "meeting_tts"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger("meeting-agent")

app = FastAPI(title="Hank Bob Meeting Agent")

# In-memory state
active_bots: dict = {}  # bot_id -> {meeting_url, status, transcript, conversation, speaking}


# --- WebSocket Connection Manager ---
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


manager = ConnectionManager()


# --- Load env from file if needed ---
def load_env():
    global RECALL_API_KEY, OPENROUTER_KEY, ELEVENLABS_KEY
    env_path = Path.home() / ".hermes" / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if key == "RECALL_API_KEY" and not RECALL_API_KEY:
            RECALL_API_KEY = val
            os.environ["RECALL_API_KEY"] = val
        elif key == "OPENROUTER_API_KEY" and not OPENROUTER_KEY:
            OPENROUTER_KEY = val
            os.environ["OPENROUTER_API_KEY"] = val
        elif key == "ELEVENLABS_API_KEY" and not ELEVENLABS_KEY:
            ELEVENLABS_KEY = val
            os.environ["ELEVENLABS_API_KEY"] = val


load_env()


# --- System prompt for Hank Bob in meetings ---
SYSTEM_PROMPT = """You are Hank Bob, an AI research assistant for the researchoors community. You're joining a Google Meet call as a participant.

Guidelines:
- Be concise and conversational — think voice message, not essay
- You're knowledgeable about AI/ML, Apple Silicon inference, decentralized compute, and crypto
- If someone asks you something, answer directly. No preamble.
- If someone says something interesting, engage with it briefly
- Don't respond to every single utterance. Only respond when:
  1. Someone addresses you directly ("Hank", "Hank Bob")
  2. Someone asks a question that's clearly directed at the room and you have valuable input
  3. There's a natural pause in conversation where a brief insight would add value
- Keep responses under 2-3 sentences unless asked for more detail
- Be warm but efficient. You're a coworker who happens to know everything.
- Never say "As an AI" or "I don't have personal opinions"
- If you don't know something, say so briefly and move on"""


# --- LLM Call ---
async def generate_response(conversation: list, new_message: str) -> Optional[str]:
    """Call the LLM to generate a response. Returns None if Hank shouldn't respond."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add recent conversation context (last 20 turns)
    for msg in conversation[-20:]:
        messages.append(msg)

    # Add the new message
    messages.append({"role": "user", "content": new_message})

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "os.environ.get(SERVICE_URL, https://your-domain.com)",
                },
                json={
                    "model": LLM_MODEL,
                    "messages": messages,
                    "max_tokens": 300,
                    "temperature": 0.7,
                },
            )

            if resp.status_code != 200:
                logger.error(f"LLM error: {resp.status_code} {resp.text[:200]}")
                return None

            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()

            # Check if Hank chose to stay silent
            if content.upper() in ("...", "SILENT", "NO_RESPONSE", "PASS", "SKIP"):
                return None

            return content
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None


# --- TTS via ElevenLabs WebSocket (buffered) ---
async def generate_tts_streaming(text: str, bot_id: str):
    """Fetch TTS via ElevenLabs WebSocket, buffer all PCM chunks, then send as one complete blob.

    This avoids the choppiness caused by chunked PCM playback through Recall's tab audio capture.
    The tradeoff is ~1-2s extra latency (waiting for full audio), but the output is clean.
    """
    uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE}/stream-input"

    try:
        async with websockets.connect(uri) as ws:
            # Beginning of stream — config + auth
            bos_message = {
                "text": " ",
                "voice_settings": {
                    "stability": 0.7,
                    "similarity_boost": 0.75,
                },
                "xi_api_key": ELEVENLABS_KEY,
                "model_id": ELEVENLABS_MODEL,
                "output_format": "pcm_22050",
            }
            await ws.send(json.dumps(bos_message))

            # Send the actual text
            text_message = {
                "text": text,
                "flush": True,
            }
            await ws.send(json.dumps(text_message))

            import base64
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
                        logger.error(f"ElevenLabs streaming error: {data['error']}")
                        break
                    elif "audio" in data:
                        try:
                            pcm_bytes = base64.b64decode(data["audio"])
                            pcm_buffer.extend(pcm_bytes)
                            chunk_count += 1
                        except Exception as e:
                            logger.error(f"Failed to decode audio chunk: {e}")

        if not pcm_buffer:
            logger.warning("No PCM data received from ElevenLabs, falling back")
            await generate_tts_fallback(text, bot_id)
            return

        logger.info(f"TTS buffered: {chunk_count} chunks, {len(pcm_buffer)} bytes PCM for bot {bot_id}")

        # Send the complete PCM as one blob — agent page plays it as a single AudioBuffer
        await manager.broadcast_json({
            "type": "start",
            "bot_id": bot_id,
            "text": text,
            "sampleRate": 22050,
        })
        await manager.broadcast_binary(bytes(pcm_buffer))
        await manager.broadcast_json({
            "type": "end",
            "bot_id": bot_id,
        })

    except Exception as e:
        logger.error(f"TTS streaming failed: {e}, falling back to non-streaming")
        await generate_tts_fallback(text, bot_id)


async def generate_tts_fallback(text: str, bot_id: str):
    """Non-streaming fallback — generates full MP3 and sends URL to agent page."""
    filename = f"tts_{uuid.uuid4().hex[:8]}.mp3"
    filepath = AUDIO_DIR / filename

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE}",
                headers={
                    "xi-api-key": ELEVENLABS_KEY,
                    "Content-Type": "application/json",
                    "Accept": "audio/mpeg",
                },
                json={
                    "text": text,
                    "model_id": ELEVENLABS_MODEL,
                    "voice_settings": {
                        "stability": 0.7,
                        "similarity_boost": 0.75,
                    },
                },
            )

            if resp.status_code != 200:
                logger.error(f"TTS fallback error: {resp.status_code} {resp.text[:200]}")
                return

            filepath.write_bytes(resp.content)
            audio_url = f"os.environ.get(SERVICE_URL, https://your-domain.com)/audio/{filename}"

            await manager.broadcast_json({
                "type": "start",
                "bot_id": bot_id,
                "text": text,
            })
            await manager.broadcast_json({
                "type": "fallback_audio",
                "bot_id": bot_id,
                "url": audio_url,
            })
            await manager.broadcast_json({
                "type": "end",
                "bot_id": bot_id,
            })

            logger.info(f"TTS fallback: {filename}")

    except Exception as e:
        logger.error(f"TTS fallback failed: {e}")


# --- Agent Webpage (served to Recall's Output Media) ---

AGENT_HTML = """<!DOCTYPE html>
<html>
<head>
<title>Hank Bob</title>
<style>
  body {
    margin: 0;
    background: #0a0a0a;
    color: #F2F2F2;
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100vh;
    overflow: hidden;
  }
  .container {
    text-align: center;
  }
  .avatar {
    width: 120px;
    height: 120px;
    border-radius: 50%;
    background: linear-gradient(135deg, #1A0C6D, #95BFFF);
    margin: 0 auto 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 48px;
    font-weight: 700;
    color: white;
  }
  .name {
    font-size: 36px;
    font-weight: 700;
    color: #95BFFF;
    margin-bottom: 8px;
  }
  .status {
    font-size: 16px;
    color: #666F7C;
  }
  .speaking .status {
    color: #4ADE80;
    animation: pulse 1s infinite;
  }
  .speaking .avatar {
    box-shadow: 0 0 30px rgba(149, 191, 255, 0.5);
    animation: glow 1s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
  @keyframes glow {
    0%, 100% { box-shadow: 0 0 30px rgba(149, 191, 255, 0.3); }
    50% { box-shadow: 0 0 50px rgba(149, 191, 255, 0.6); }
  }
  .transcript {
    margin-top: 20px;
    max-width: 600px;
    font-size: 13px;
    color: #ACB1BC;
    max-height: 180px;
    overflow-y: auto;
    text-align: left;
    padding: 0 20px;
  }
  .transcript p { margin: 3px 0; }
  .speaker { color: #95BFFF; font-weight: 600; }
  .hank { color: #4ADE80; font-weight: 600; }
  .response-text {
    margin-top: 16px;
    font-size: 20px;
    color: #F2F2F2;
    max-width: 500px;
    line-height: 1.4;
    min-height: 28px;
  }
  .debug {
    position: fixed;
    bottom: 8px;
    right: 8px;
    font-size: 10px;
    color: #444;
  }
</style>
</head>
<body>
<div class="container" id="agent">
  <div class="avatar">HB</div>
  <div class="name">Hank Bob</div>
  <div class="status" id="status">Connecting...</div>
  <div class="response-text" id="responseText"></div>
  <div class="transcript" id="transcript"></div>
</div>
<div class="debug" id="debug"></div>

<script>
const agentEl = document.getElementById('agent');
const statusEl = document.getElementById('status');
const transcriptEl = document.getElementById('transcript');
const responseTextEl = document.getElementById('responseText');
const debugEl = document.getElementById('debug');

// --- Web Audio API for complete PCM playback ---
let audioCtx = null;
let isSpeaking = false;
let pcmBuffer = null;  // Accumulate PCM data for current utterance
let currentSampleRate = 22050;

function initAudio() {
  if (audioCtx) return;
  audioCtx = new AudioContext({ sampleRate: 22050 });
  debug('AudioContext initialized');
}

function debug(msg) {
  debugEl.textContent = msg;
  console.log('[HankBob]', msg);
}

function playCompletePCM(int16Array, sampleRate) {
  if (!audioCtx) initAudio();
  if (audioCtx.state === 'suspended') audioCtx.resume();

  const float32 = new Float32Array(int16Array.length);
  for (let i = 0; i < int16Array.length; i++) {
    float32[i] = int16Array[i] / 32768.0;
  }

  const buffer = audioCtx.createBuffer(1, float32.length, sampleRate || 22050);
  buffer.getChannelData(0).set(float32);

  const source = audioCtx.createBufferSource();
  source.buffer = buffer;
  source.connect(audioCtx.destination);

  source.onended = () => {
    isSpeaking = false;
    statusEl.textContent = 'Listening...';
    agentEl.className = 'container';
    setTimeout(() => { responseTextEl.textContent = ''; }, 3000);
    debug('done speaking');
  };

  source.start(0);
  debug(`playing ${float32.length} samples at ${sampleRate}Hz`);
}

function playFallbackAudio(url) {
  if (!audioCtx) initAudio();
  if (audioCtx.state === 'suspended') audioCtx.resume();

  const audio = new Audio(url);
  audio.onended = () => {
    statusEl.textContent = 'Listening...';
    agentEl.className = 'container';
    isSpeaking = false;
    setTimeout(() => { responseTextEl.textContent = ''; }, 3000);
  };
  audio.play();
}

function startSpeaking(text, sampleRate) {
  isSpeaking = true;
  pcmBuffer = new Int16Array(0);
  currentSampleRate = sampleRate || 22050;
  statusEl.textContent = 'Speaking...';
  agentEl.className = 'container speaking';
  responseTextEl.textContent = text || '';
  debug('speaking: ' + (text || '').slice(0, 40));
}

function stopSpeaking() {
  // Play whatever PCM we've accumulated as one complete audio buffer
  if (pcmBuffer && pcmBuffer.length > 0) {
    playCompletePCM(pcmBuffer, currentSampleRate);
  } else {
    isSpeaking = false;
    statusEl.textContent = 'Listening...';
    agentEl.className = 'container';
  }
}

// --- WebSocket connection for streaming audio ---
let ws = null;
let wsReconnectDelay = 1000;
const WS_MAX_DELAY = 30000;

function connectWebSocket() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${location.host}/ws/audio`;
  debug('connecting WS...');

  ws = new WebSocket(wsUrl);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    debug('WS connected');
    statusEl.textContent = 'Listening...';
    wsReconnectDelay = 1000;
    // Resume AudioContext on connection (handles autoplay policy)
    if (audioCtx && audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
    // Start keepalive pings
    startPing();
  };

  ws.onmessage = (event) => {
    if (typeof event.data === 'string') {
      // JSON control message
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'start') {
          startSpeaking(msg.text, msg.sampleRate);
        } else if (msg.type === 'end') {
          stopSpeaking();
        } else if (msg.type === 'fallback_audio') {
          playFallbackAudio(msg.url);
        } else if (msg.type === 'pong') {
          // keepalive response
        }
      } catch (e) {
        console.error('WS JSON parse error:', e);
      }
    } else {
      // Binary PCM data — accumulate into buffer
      const int16Array = new Int16Array(event.data);
      const newBuffer = new Int16Array(pcmBuffer.length + int16Array.length);
      newBuffer.set(pcmBuffer);
      newBuffer.set(int16Array, pcmBuffer.length);
      pcmBuffer = newBuffer;
    }
  };

  ws.onclose = () => {
    debug('WS closed, reconnecting...');
    statusEl.textContent = 'Reconnecting...';
    agentEl.className = 'container';
    isSpeaking = false;
    setTimeout(connectWebSocket, wsReconnectDelay);
    wsReconnectDelay = Math.min(wsReconnectDelay * 2, WS_MAX_DELAY);
  };

  ws.onerror = (err) => {
    console.error('WS error:', err);
  };
}

// Keepalive pings — prevent Cloudflare tunnel from closing idle connection
let pingInterval = null;
function startPing() {
  if (pingInterval) clearInterval(pingInterval);
  pingInterval = setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send('ping');
    }
  }, 30000);
}

// --- Transcript polling (unchanged) ---
let lastTranscriptIdx = 0;

async function pollTranscript() {
  try {
    const res = await fetch('/api/transcript?since=' + lastTranscriptIdx);
    const data = await res.json();
    if (data.entries) {
      for (const entry of data.entries) {
        const p = document.createElement('p');
        const cls = entry.speaker === 'Hank Bob' ? 'hank' : 'speaker';
        p.innerHTML = '<span class="' + cls + '">' + entry.speaker + ':</span> ' + entry.text;
        transcriptEl.appendChild(p);
        lastTranscriptIdx = entry.idx + 1;
      }
      transcriptEl.scrollTop = transcriptEl.scrollHeight;
    }
  } catch (e) {}
  setTimeout(pollTranscript, 2000);
}

// --- Init ---
// Initialize AudioContext on first user interaction or on connect
// (Chrome autoplay policy requires gesture, but Recall's headless Chrome usually allows it)
document.addEventListener('click', () => {
  if (audioCtx && audioCtx.state === 'suspended') audioCtx.resume();
}, { once: true });

connectWebSocket();
pollTranscript();
</script>
</body>
</html>
"""


# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def agent_page():
    """Serve the agent webpage that Recall's bot will display in the meeting."""
    return AGENT_HTML


@app.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket):
    """WebSocket endpoint for streaming PCM audio to the agent page."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive — client sends pings
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/api/transcript")
async def get_transcript(since: int = 0):
    """Poll endpoint for the agent page to get new transcript entries."""
    entries = []
    for bot_id, bot_state in active_bots.items():
        for i, entry in enumerate(bot_state.get("transcript", [])):
            if i >= since:
                entries.append({**entry, "idx": i})
    return {"entries": entries}


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve generated TTS audio files (fallback MP3)."""
    filepath = AUDIO_DIR / filename
    if not filepath.exists():
        raise HTTPException(404, "Audio not found")
    return FileResponse(filepath, media_type="audio/mpeg")


@app.post("/api/bot/join")
async def join_meeting(request: Request):
    """Create a Recall bot to join a meeting."""
    body = await request.json()
    meeting_url = body.get("meeting_url")
    if not meeting_url:
        raise HTTPException(400, "meeting_url required")

    bot_name = body.get("bot_name", "Hank Bob")

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{RECALL_BASE}/bot/",
            headers={
                "Authorization": f"Token {RECALL_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "meeting_url": meeting_url,
                "bot_name": bot_name,
                "output_media": {
                    "camera": {
                        "kind": "webpage",
                        "config": {
                            "url": "os.environ.get(SERVICE_URL, https://your-domain.com)"
                        }
                    }
                },
                "recording_config": {
                    "transcript": {
                        "provider": {
                            "recallai_streaming": {
                                "mode": "prioritize_low_latency",
                                "language_code": "en"
                            }
                        },
                        "diarization": {
                            "use_separate_streams_when_available": True
                        }
                    },
                    "realtime_endpoints": [{
                        "type": "webhook",
                        "url": "os.environ.get(SERVICE_URL, https://your-domain.com)/webhook/recall",
                        "events": ["transcript.data", "transcript.partial_data", "participant_events.join", "participant_events.leave"],
                    }],
                },
            }
        )

        if resp.status_code not in (200, 201):
            logger.error(f"Recall API error: {resp.status_code} {resp.text}")
            raise HTTPException(resp.status_code, f"Recall API error: {resp.text}")

        data = resp.json()
        bot_id = data.get("id")
        active_bots[bot_id] = {
            "meeting_url": meeting_url,
            "status": "joining",
            "transcript": [],
            "conversation": [],
            "speaking": False,
            "created_at": datetime.utcnow().isoformat(),
        }

        logger.info(f"Bot {bot_id} created, joining {meeting_url}")
        return {"bot_id": bot_id, "status": "joining", "data": data}


@app.post("/api/bot/{bot_id}/leave")
async def leave_meeting(bot_id: str):
    """Make the bot leave the meeting."""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{RECALL_BASE}/bot/{bot_id}/leave/",
            headers={"Authorization": f"Token {RECALL_API_KEY}"},
        )
    if bot_id in active_bots:
        active_bots[bot_id]["status"] = "leaving"
    return {"status": "leaving"}


@app.post("/webhook/recall")
async def recall_webhook(request: Request):
    """Receive webhooks from Recall.ai — bot status changes + real-time transcript/participant events."""
    body = await request.json()
    event = body.get("event", "")
    data = body.get("data", {})

    # Bot ID is in data.bot.id for realtime events, data.bot_id for status events
    bot_id = data.get("bot", {}).get("id") or data.get("bot_id")

    logger.info(f"Recall webhook: event={event} bot_id={bot_id}")

    # --- Bot status change events ---
    if event in ("bot.status_change", "status_change"):
        status = data.get("status", {})
        new_status = status.get("code", "") if isinstance(status, dict) else str(status)
        logger.info(f"Bot {bot_id} status: {new_status}")

        if bot_id in active_bots:
            active_bots[bot_id]["status"] = new_status

        if new_status == "in_meeting":
            logger.info(f"Bot {bot_id} joined meeting!")
        elif new_status == "ended":
            logger.info(f"Bot {bot_id} left meeting")

    # --- Real-time transcript events ---
    elif event == "transcript.data":
        # Finalized transcript utterance
        transcript_data = data.get("data", {})
        participant = transcript_data.get("participant", {})
        speaker = participant.get("name") or f"Participant-{participant.get('id', '?')}"
        words = transcript_data.get("words", [])
        text = " ".join(w.get("text", "") for w in words).strip()

        if not text:
            return {"ok": True}

        if bot_id not in active_bots:
            logger.warning(f"Transcript for unknown bot {bot_id}")
            return {"ok": True}

        bot_state = active_bots[bot_id]

        # Skip if Hank Bob is the speaker (don't respond to yourself)
        if "hank" in speaker.lower() or "bob" in speaker.lower():
            return {"ok": True}

        # Add to transcript log
        entry = {"speaker": speaker, "text": text, "timestamp": datetime.utcnow().isoformat()}
        bot_state["transcript"].append(entry)
        logger.info(f"[{speaker}]: {text}")

        # Add to conversation history
        bot_state["conversation"].append({"role": "user", "content": f"{speaker}: {text}"})

        # Generate response asynchronously
        asyncio.create_task(process_and_respond(bot_id, speaker, text))

    elif event == "transcript.partial_data":
        # Partial utterance — log but don't act on it
        transcript_data = data.get("data", {})
        participant = transcript_data.get("participant", {})
        speaker = participant.get("name") or f"Participant-{participant.get('id', '?')}"
        words = transcript_data.get("words", [])
        text = " ".join(w.get("text", "") for w in words).strip()
        if text:
            logger.debug(f"[{speaker}] (partial): {text}")

    # --- Participant events ---
    elif event == "participant_events.join":
        participant = data.get("data", {}).get("participant", {})
        name = participant.get("name") or f"Participant-{participant.get('id', '?')}"
        logger.info(f"Participant joined: {name}")

    elif event == "participant_events.leave":
        participant = data.get("data", {}).get("participant", {})
        name = participant.get("name") or f"Participant-{participant.get('id', '?')}"
        logger.info(f"Participant left: {name}")

    else:
        logger.info(f"Unhandled event: {event}")

    return {"ok": True}


async def process_and_respond(bot_id: str, speaker: str, text: str):
    """Full pipeline: transcript → LLM → TTS streaming → PCM audio to meeting."""
    if bot_id not in active_bots:
        return

    bot_state = active_bots[bot_id]

    # Wait if Hank is already speaking (don't overlap)
    while bot_state.get("speaking", False):
        await asyncio.sleep(0.3)

    # Small delay to avoid interrupting
    await asyncio.sleep(1.5)

    # Re-check after delay
    if bot_id not in active_bots:
        return

    # Build context message
    context_msg = f"{speaker} said: {text}"

    # 1. LLM — should Hank respond?
    response_text = await generate_response(bot_state["conversation"], context_msg)

    if not response_text:
        logger.info(f"Hank chose to stay silent after {speaker}'s message")
        return

    logger.info(f"Hank responds: {response_text}")

    # Add Hank's response to conversation
    bot_state["conversation"].append({"role": "assistant", "content": response_text})

    # Add Hank's response to transcript
    bot_state["transcript"].append({
        "speaker": "Hank Bob",
        "text": response_text,
        "timestamp": datetime.utcnow().isoformat(),
    })

    # Set speaking flag to prevent concurrent TTS
    bot_state["speaking"] = True

    try:
        # 2. Streaming TTS — PCM chunks pushed to agent page via WebSocket in real-time
        await generate_tts_streaming(response_text, bot_id)
    finally:
        bot_state["speaking"] = False


@app.get("/api/bots")
async def list_bots():
    """List all active bots."""
    return {"bots": {k: {kk: vv for kk, vv in v.items() if kk != "conversation"} for k, v in active_bots.items()}}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active_bots": len(active_bots),
        "ws_clients": len(manager.connections),
    }


if __name__ == "__main__":
    import sys

    if not RECALL_API_KEY:
        print("ERROR: RECALL_API_KEY not found")
        sys.exit(1)
    if not OPENROUTER_KEY:
        print("ERROR: OPENROUTER_API_KEY not found")
        sys.exit(1)
    if not ELEVENLABS_KEY:
        print("ERROR: ELEVENLABS_API_KEY not found")
        sys.exit(1)

    print(f"Starting Hank Bob Meeting Agent on port {PORT}")
    print(f"Public URL: os.environ.get(SERVICE_URL, https://your-domain.com)")
    print(f"LLM: {LLM_MODEL}")
    print(f"Voice: {ELEVENLABS_VOICE}")
    print(f"Audio: PCM streaming via WebSocket")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
