"""
Hank Bob Meeting Agent — Recall.ai integration
Kokoro+RVC local TTS pipeline + Voice Gateway (memory snapshots + RAG + routing)

Architecture:
- Recall webhook → transcript → Voice Gateway (snapshot/RAG/routing) → LLM → Kokoro TTS → RVC → WAV → AudioBufferSourceNode
- Zero API calls for TTS — fully local on Apple Silicon
- asyncio.Lock per bot prevents concurrent/duplicate responses
- Referencable memory snapshots: LLM sees entry index, can EXPAND[n] for details
"""

import os
import sys
import json
import asyncio
import logging
import uuid
import time
import re
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import httpx
import uvicorn
import soundfile as sf

# ============================================================
# Config
# ============================================================
RECALL_API_KEY = os.environ.get("RECALL_API_KEY", "")
RECALL_BASE = "https://us-west-2.recall.ai/api/v1"
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-4.1-mini")  # Changed default

# Voice Gateway models
FAST_MODEL = os.environ.get("FAST_MODEL", "google/gemini-2.5-flash")
STANDARD_MODEL = os.environ.get("STANDARD_MODEL", "openai/gpt-4.1-mini")
DEEP_MODEL = os.environ.get("DEEP_MODEL", "anthropic/claude-sonnet-4")

# RVC config
RVC_MODEL_PATH = os.environ.get("RVC_MODEL_PATH",
    "/Users/inference2/.hermes/voice_references/rvc_dataset_48k/HankHill_e100.pth")
RVC_EXP_DIR = "/Users/inference2/.hermes/voice_references/rvc_dataset_48k"
RVC_REPO_DIR = "/Users/inference2/RVC"
RVC_F0_METHOD = os.environ.get("RVC_F0_METHOD", "rmvpe")
RVC_F0_UP_KEY = int(os.environ.get("RVC_F0_UP_KEY", "0"))
RVC_INDEX_RATE = float(os.environ.get("RVC_INDEX_RATE", "0.0"))

# Kokoro config
KOKORO_VOICE = os.environ.get("KOKORO_VOICE", "af_heart")

# Memory paths
HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
MEMORY_FILE = HERMES_HOME / "memories" / "MEMORY.md"
USER_FILE = HERMES_HOME / "memories" / "USER.md"

PORT = 9120
AUDIO_DIR = Path.home() / ".hermes" / "audio_cache" / "meeting_tts"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger("meeting-agent")

app = FastAPI(title="Hank Bob Meeting Agent")

# In-memory state
active_bots: dict = {}
audio_queue: dict = {}

# ============================================================
# Load env
# ============================================================
def load_env():
    global RECALL_API_KEY, OPENROUTER_KEY
    env_path = HERMES_HOME / ".env"
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

load_env()

# Also try auth.json credential pool for a working OpenRouter key
def _get_openrouter_key():
    auth_path = HERMES_HOME / "auth.json"
    if auth_path.exists():
        try:
            d = json.loads(auth_path.read_text())
            for k in d.get("credential_pool", {}).get("openrouter", []):
                token = k.get("access_token", "")
                if token and k.get("status") != "exhausted":
                    return token
        except Exception:
            pass
    return OPENROUTER_KEY

OPENROUTER_KEY = _get_openrouter_key()


# ============================================================
# Voice Gateway: Memory Snapshot + RAG + Routing
# ============================================================
ENTRY_DELIMITER = "§"

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "then", "once", "and", "but", "or",
    "nor", "not", "so", "yet", "both", "either", "neither", "each",
    "every", "all", "any", "few", "more", "most", "other", "some",
    "such", "no", "only", "own", "same", "than", "too", "very",
    "just", "because", "if", "when", "where", "how", "what", "which",
    "who", "whom", "this", "that", "these", "those", "i", "me", "my",
    "we", "our", "you", "your", "he", "him", "his", "she", "her",
    "it", "its", "they", "them", "their", "about", "up", "also",
    "know", "tell", "get", "got", "like", "think", "want", "need",
}

TOOL_KEYWORDS = {"github", "repo", "issue", "pr ", "pull request", "commit", "deploy",
                 "install", "server", "docker", "nomad", "latest issues"}

MEMORY_KEYWORDS = {"remember", "recall", "know about", "what is", "who is",
                   "tell me about", "explain", "describe", "about the",
                   "darkbloom", "d-inference", "swiftlm", "dflash", "researchoors",
                   "layr", "benchmark", "project", "setup", "configured",
                   "hermes", "ethan", "hank"}

SIMPLE_PATTERNS = [
    re.compile(r'^(hi|hey|hello|yo|sup|what\'s up|thanks|ok|yes|no|sure|cool|nice|got it)', re.I),
    re.compile(r'^\d+\s*[\+\-\*\/]\s*\d+'),
]

EXPAND_RE = re.compile(r'EXPAND\[(\d+(?:\s*,\s*\d+)*)\]')


class MemorySnapshot:
    """Parses MEMORY.md + USER.md into §-delimited entries with index + compressed summary."""

    def __init__(self):
        self.entries = []  # list of dicts: {index, source, text}
        self.summary = ""  # ultra-compressed overview (~80 tokens)
        self.index_text = ""  # one-line-per-entry index
        self.built_at = 0.0

    def refresh_if_stale(self, max_age=300):
        mem_mtime = MEMORY_FILE.stat().st_mtime if MEMORY_FILE.exists() else 0
        usr_mtime = USER_FILE.stat().st_mtime if USER_FILE.exists() else 0
        if max(mem_mtime, usr_mtime) > self.built_at:
            self.build()

    def build(self):
        entries = []
        for source, path in [("memory", MEMORY_FILE), ("user", USER_FILE)]:
            if not path.exists():
                continue
            content = path.read_text().strip()
            if not content:
                continue
            for chunk in content.split(ENTRY_DELIMITER):
                chunk = chunk.strip()
                if chunk:
                    entries.append({"index": len(entries), "source": source, "text": chunk})

        self.entries = entries
        self.built_at = time.time()

        # Build index text
        lines = []
        for e in entries:
            preview = e["text"][:80].replace("\n", " ").strip()
            if len(e["text"]) > 80:
                preview += "..."
            lines.append(f"[{e['index']}] {preview}")
        self.index_text = "\n".join(lines)

        # Build compressed summary
        parts = []
        for e in entries[:5]:
            first = e["text"].split(".")[0].strip()
            if first:
                parts.append(first)
        self.summary = " | ".join(parts)

        logger.info(f"Memory snapshot built: {len(entries)} entries")

    def rag_retrieve(self, query, top_k=3):
        query_terms = {w.lower() for w in re.findall(r'\w+', query) if w.lower() not in STOP_WORDS}
        if not query_terms:
            return []
        scored = []
        for e in self.entries:
            entry_terms = {w.lower() for w in re.findall(r'\w+', e["text"]) if w.lower() not in STOP_WORDS}
            overlap = len(query_terms & entry_terms)
            if overlap > 0:
                scored.append((overlap, e))
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:top_k]]

    @staticmethod
    def classify_query(query):
        q_lower = query.lower().strip()
        for pat in SIMPLE_PATTERNS:
            if pat.match(q_lower):
                return "fast"
        query_words = set(re.findall(r'\w+', q_lower))
        if query_words & TOOL_KEYWORDS:
            return "deep"
        for kw in MEMORY_KEYWORDS:
            if kw in q_lower:
                return "standard"
        if memory.rag_retrieve(query, top_k=1):
            return "standard"
        return "fast"

    @staticmethod
    def parse_expands(text):
        indices = set()
        for match in EXPAND_RE.finditer(text):
            for num_str in match.group(1).split(","):
                try:
                    idx = int(num_str.strip())
                    if 0 <= idx < len(memory.entries):
                        indices.add(idx)
                except ValueError:
                    continue
        return sorted(indices)

    @staticmethod
    def strip_expands(text):
        return EXPAND_RE.sub("", text).strip()


# Global snapshot
memory = MemorySnapshot()
memory.build()


# ============================================================
# Kokoro + RVC TTS Engine (lazy-loaded, runs in-process)
# ============================================================
class LocalTTSEngine:
    """Local TTS pipeline: Kokoro (text→speech) + RVC (voice conversion)."""

    def __init__(self):
        self._kokoro = None
        self._vc = None
        self._config = None
        self._ready = False

    def _ensure_init(self):
        if self._ready:
            return

        logger.info("Initializing local TTS engine (Kokoro + RVC)...")
        t0 = time.time()

        # Kokoro
        from kokoro import KPipeline
        self._kokoro = KPipeline(lang_code='a')

        # RVC
        sys.path.insert(0, RVC_REPO_DIR)
        os.environ.update({
            "weight_root": RVC_EXP_DIR,
            "index_root": f"{RVC_EXP_DIR}/Index",
            "outside_index_root": f"{RVC_EXP_DIR}/Index",
            "rmvpe_root": f"{RVC_REPO_DIR}/assets/rmvpe",
        })

        from infer.modules.vc.modules import VC
        from configs.config import Config

        self._config = Config()
        self._vc = VC(self._config)

        # Load RVC model
        model_name = Path(RVC_MODEL_PATH).name
        self._vc.get_vc(model_name, 0.5, 0.33)

        self._ready = True
        elapsed = time.time() - t0
        logger.info(f"Local TTS engine ready in {elapsed:.1f}s")

    def generate(self, text: str) -> tuple:
        """Generate audio from text. Returns (sample_rate, np_float32_array)."""
        self._ensure_init()

        # Step 1: Kokoro TTS
        generator = self._kokoro(text, voice=KOKORO_VOICE)
        all_audio = []
        for gs, ps, audio in generator:
            all_audio.append(audio)
        kokoro_audio = np.concatenate(all_audio)
        kokoro_sr = 24000
        logger.info(f"Kokoro: {len(kokoro_audio)/kokoro_sr:.1f}s audio from {len(text)} chars")

        # Save temp WAV for RVC (it reads from file)
        tmp_wav = AUDIO_DIR / f"_kokoro_tmp_{uuid.uuid4().hex[:6]}.wav"
        sf.write(str(tmp_wav), kokoro_audio, kokoro_sr)

        try:
            # Step 2: RVC voice conversion
            result = self._vc.vc_single(
                sid=0,
                input_audio_path=str(tmp_wav),
                f0_up_key=RVC_F0_UP_KEY,
                f0_file="",
                f0_method=RVC_F0_METHOD,
                file_index="",
                file_index2="",
                index_rate=RVC_INDEX_RATE,
                filter_radius=3,
                resample_sr=0,
                rms_mix_rate=0.25,
                protect=0.33,
            )

            info = result[0]
            sr, audio_int16 = result[1]
            audio_float = audio_int16.flatten().astype(np.float32) / 32768.0
            logger.info(f"RVC: {info} → {len(audio_float)/sr:.1f}s at {sr}Hz")

            return sr, audio_float
        finally:
            try:
                tmp_wav.unlink()
            except:
                pass

# Singleton
tts_engine = LocalTTSEngine()


# ============================================================
# System prompt (enhanced with memory snapshot)
# ============================================================
BASE_SYSTEM = """You are Hank Bob, an AI research assistant for the researchoors community. You're joining a Google Meet call as a participant.

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


def build_system_prompt(path, rag_entries=None, expanded_indices=None):
    """Build system prompt with memory context based on routing path."""
    parts = [BASE_SYSTEM]

    # Always include snapshot summary
    parts.append(f"\nUser context: {memory.summary}")

    if path in ("standard", "deep"):
        # Include entry index so LLM can EXPAND specific entries
        parts.append(f"\nMemory entries available (use EXPAND[n] to request details):\n{memory.index_text}")

    # Add RAG-enriched entries
    if rag_entries:
        parts.append("\nRelevant memory:")
        for e in rag_entries:
            parts.append(f"  {e['text']}")

    # Add expanded entries from follow-up
    if expanded_indices is not None:
        expanded = [memory.entries[i] for i in expanded_indices if i < len(memory.entries)]
        if expanded:
            parts.append("\nExpanded entries:")
            for e in expanded:
                parts.append(f"  {e['text']}")

    return "\n".join(parts)


# ============================================================
# LLM Call (with voice gateway routing)
# ============================================================
async def generate_response(conversation: list, new_message: str, bot_state: dict = None) -> Optional[str]:
    """Call the LLM with memory-aware routing. Returns None if Hank shouldn't respond."""
    # Refresh snapshot if stale
    memory.refresh_if_stale()

    # Classify query
    path = MemorySnapshot.classify_query(new_message)

    # Build RAG context
    rag_entries = None
    if path in ("standard", "deep"):
        rag_entries = memory.rag_retrieve(new_message, top_k=3)

    # Get expanded entries from bot state (multi-turn)
    expanded = None
    if bot_state and "expanded_entries" in bot_state:
        expanded = sorted(bot_state["expanded_entries"])

    # Build system prompt
    system_prompt = build_system_prompt(path, rag_entries=rag_entries, expanded_indices=expanded)

    # Select model
    if path == "fast":
        model = FAST_MODEL
    elif path == "standard":
        model = STANDARD_MODEL
    else:
        model = DEEP_MODEL

    messages = [{"role": "system", "content": system_prompt}]
    for msg in conversation[-20:]:
        messages.append(msg)
    messages.append({"role": "user", "content": new_message})

    try:
        t0 = asyncio.get_event_loop().time()
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://meet.model-optimizors.com",
                },
                json={
                    "model": model,
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

            t1 = asyncio.get_event_loop().time()
            usage = data.get("usage", {})
            prompt_tok = usage.get("prompt_tokens", 0)
            comp_tok = usage.get("completion_tokens", 0)

            logger.info(
                f"LLM path={path} model={model.split('/')[-1]} "
                f"latency={((t1-t0)*1000):.0f}ms "
                f"prompt={prompt_tok} comp={comp_tok} "
                f"rag={len(rag_entries) if rag_entries else 0}"
            )

            if content.upper() in ("...", "SILENT", "NO_RESPONSE", "PASS", "SKIP"):
                return None

            # Parse EXPAND directives
            expand_indices = MemorySnapshot.parse_expands(content)
            if expand_indices and bot_state is not None:
                if "expanded_entries" not in bot_state:
                    bot_state["expanded_entries"] = set()
                bot_state["expanded_entries"].update(expand_indices)
                logger.info(f"EXPAND requested: entries {expand_indices}")
            content = MemorySnapshot.strip_expands(content)

            return content
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None


# ============================================================
# TTS: Local Kokoro+RVC pipeline
# ============================================================
async def generate_tts_audio(text: str, bot_id: str) -> Optional[str]:
    """Generate TTS via local Kokoro+RVC. Returns the filename of the saved WAV."""
    try:
        # Run in thread pool to not block the event loop
        loop = asyncio.get_event_loop()
        sr, audio_float = await loop.run_in_executor(None, tts_engine.generate, text)

        # Save WAV
        audio_id = str(uuid.uuid4())[:8]
        filename = f"{bot_id[:8]}_{audio_id}.wav"
        filepath = AUDIO_DIR / filename
        sf.write(str(filepath), audio_float, sr)

        duration = len(audio_float) / sr
        logger.info(f"TTS WAV ready: {duration:.1f}s, {filepath.stat().st_size} bytes")

        # Add to audio queue
        if bot_id not in audio_queue:
            audio_queue[bot_id] = []
        audio_queue[bot_id].append({
            "id": audio_id,
            "text": text,
            "filename": filename,
        })

        return filename

    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# Agent Webpage
# ============================================================

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
    transition: all 0.3s;
  }
  .avatar {
    width: 80px; height: 80px;
    background: #1a1a2e;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 12px;
    font-size: 28px; font-weight: 700;
    color: #95BFFF;
    border: 3px solid #2a2a4e;
  }
  .speaking .avatar {
    border-color: #95BFFF;
    box-shadow: 0 0 30px rgba(149, 191, 255, 0.5);
    animation: glow 1s infinite;
  }
  .name { font-size: 18px; font-weight: 600; color: #E0E0E0; }
  .status { font-size: 13px; color: #888; margin-top: 6px; }
  .response-text {
    margin-top: 16px; font-size: 20px; color: #F2F2F2;
    max-width: 500px; line-height: 1.4; min-height: 28px;
  }
  .tts-badge {
    position: fixed; top: 12px; right: 12px;
    background: #1a3a1a; color: #4CAF50; padding: 4px 10px;
    border-radius: 8px; font-size: 11px; font-weight: 600;
  }
  .memory-badge {
    position: fixed; top: 12px; left: 12px;
    background: #1a1a3a; color: #95BFFF; padding: 4px 10px;
    border-radius: 8px; font-size: 11px; font-weight: 600;
  }
  @keyframes glow {
    0%, 100% { box-shadow: 0 0 30px rgba(149, 191, 255, 0.3); }
    50% { box-shadow: 0 0 50px rgba(149, 191, 255, 0.6); }
  }
</style>
</head>
<body>
<div class="tts-badge">🔊 Local Kokoro+RVC</div>
<div class="memory-badge">🧠 Memory Snapshot</div>
<div class="container" id="agent">
  <div class="avatar">HB</div>
  <div class="name">Hank Bob</div>
  <div class="status" id="status">Connecting...</div>
  <div class="response-text" id="responseText"></div>
</div>

<script>
const agentEl = document.getElementById('agent');
const statusEl = document.getElementById('status');
const responseTextEl = document.getElementById('responseText');

let isSpeaking = false;
let currentSource = null;
let pollInterval = null;
let lastAudioCount = 0;

let audioCtx = null;

async function initAudio() {
  if (audioCtx) {
    if (audioCtx.state === 'suspended') {
      await audioCtx.resume();
    }
    return;
  }
  audioCtx = new AudioContext();
  if (audioCtx.state === 'suspended') {
    const tryResume = async () => {
      try { await audioCtx.resume(); } catch(e) {}
      if (audioCtx.state === 'suspended') setTimeout(tryResume, 100);
    };
    tryResume();
  }
  debug('AudioContext ready, state=' + audioCtx.state + ' sampleRate=' + audioCtx.sampleRate);
}

function debug(msg) {
  console.log('[HankBob]', msg);
  fetch('/api/debug', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({msg})});
}

async function pollAudio() {
  try {
    const resp = await fetch('/api/audio-queue');
    if (!resp.ok) return;
    const data = await resp.json();
    const items = data.items || [];
    if (items.length > lastAudioCount) {
      const latest = items[items.length - 1];
      if (latest && latest.filename) {
        await playAudio(latest.filename, latest.text);
      }
      lastAudioCount = items.length;
    }
  } catch (e) {
    console.error('Poll error:', e);
  }
}

async function playAudio(filename, text) {
  if (isSpeaking) return;
  isSpeaking = true;
  statusEl.textContent = 'Speaking...';
  agentEl.className = 'container speaking';
  responseTextEl.textContent = text || '';

  try {
    await initAudio();
    if (audioCtx.state === 'suspended') await audioCtx.resume();

    const t_fetch_start = performance.now();
    const audioResp = await fetch('/audio/' + filename);
    const arrayBuffer = await audioResp.arrayBuffer();
    const t_fetch_end = performance.now();

    const t_decode_start = performance.now();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    const t_decode_end = performance.now();

    const source = audioCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioCtx.destination);
    currentSource = source;

    source.onended = () => {
      isSpeaking = false;
      currentSource = null;
      statusEl.textContent = 'Listening...';
      agentEl.className = 'container';
      setTimeout(() => { responseTextEl.textContent = ''; }, 3000);
    };

    const t_play_start = performance.now();
    source.start(0);

    const fetch_ms = (t_fetch_end - t_fetch_start).toFixed(0);
    const decode_ms = (t_decode_end - t_decode_start).toFixed(0);
    const play_overhead_ms = (t_play_start - t_decode_end).toFixed(0);
    const client_total_ms = (t_play_start - t_fetch_start).toFixed(0);
    debug(
      `CLIENT_BENCH | fetch=${fetch_ms}ms decode=${decode_ms}ms play_overhead=${play_overhead_ms}ms ` +
      `client_total=${client_total_ms}ms | ${audioBuffer.duration.toFixed(1)}s ${audioBuffer.sampleRate}Hz`
    );

  } catch (e) {
    debug('audio_error: ' + String(e));
    isSpeaking = false;
    currentSource = null;
    statusEl.textContent = 'Listening...';
    agentEl.className = 'container';
  }
}

statusEl.textContent = 'Listening...';
pollInterval = setInterval(pollAudio, 500);
</script>
</body>
</html>
"""


# ============================================================
# API Endpoints
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def agent_page():
    return AGENT_HTML

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active_bots": len(active_bots),
        "audio_queue_size": sum(len(v) for v in audio_queue.values()),
        "tts_engine": "local_kokoro_rvc",
        "rvc_model": Path(RVC_MODEL_PATH).name,
        "memory_entries": len(memory.entries),
        "memory_snapshot_age": round(time.time() - memory.built_at, 1),
    }

@app.post("/api/debug")
async def client_debug(request: Request):
    body = await request.json()
    msg = body.get("msg", "")
    logger.info(f"[CLIENT] {msg}")
    return {"ok": True}

@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    filepath = AUDIO_DIR / filename
    if not filepath.exists():
        raise HTTPException(404, "Audio file not found")
    media_type = "audio/mpeg" if filename.endswith(".mp3") else "audio/wav"
    return FileResponse(
        str(filepath),
        media_type=media_type,
        headers={"Cache-Control": "no-cache"},
    )

@app.get("/api/audio-queue")
async def get_audio_queue():
    all_items = []
    for bot_id, items in audio_queue.items():
        all_items.extend(items)
    return {"items": all_items, "server_time": datetime.now(timezone.utc).isoformat()}

@app.post("/api/bot/join")
async def join_meeting(request: Request):
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
                "variant": {"google_meet": "web_4_core"},
                "output_media": {
                    "camera": {
                        "kind": "webpage",
                        "config": {
                            "url": "https://meet.model-optimizors.com"
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
                        "url": "https://meet.model-optimizors.com/webhook/recall",
                        "events": ["transcript.data", "transcript.partial_data",
                                   "participant_events.join", "participant_events.leave"],
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
            "respond_lock": asyncio.Lock(),
            "last_processed_ts": "",
            "expanded_entries": set(),  # Voice gateway: track EXPAND state per bot
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(f"Bot {bot_id} created, joining {meeting_url}")
        return {"bot_id": bot_id, "status": "joining", "data": data}

@app.post("/api/bot/{bot_id}/leave")
async def leave_meeting(bot_id: str):
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
    body = await request.json()
    event = body.get("event", "")
    data = body.get("data", {})

    bot_id = data.get("bot", {}).get("id") or data.get("bot_id")
    logger.info(f"Recall webhook: event={event} bot_id={bot_id}")

    if event in ("bot.status_change", "status_change"):
        status = data.get("status", {})
        new_status = status.get("code", "") if isinstance(status, dict) else str(status)
        logger.info(f"Bot {bot_id} status: {new_status}")
        if bot_id in active_bots:
            active_bots[bot_id]["status"] = new_status

    elif event == "transcript.data":
        t0_webhook = asyncio.get_event_loop().time()
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

        if "hank" in speaker.lower() or "bob" in speaker.lower():
            return {"ok": True}

        ts = transcript_data.get("started_at") or datetime.now(timezone.utc).isoformat()
        entry = {"speaker": speaker, "text": text, "timestamp": ts}
        bot_state["transcript"].append(entry)
        logger.info(f"[{speaker}]: {text}")

        bot_state["conversation"].append({"role": "user", "content": f"{speaker}: {text}"})

        asyncio.create_task(process_and_respond(bot_id, speaker, text, ts, t0_webhook))

    elif event == "transcript.partial_data":
        pass

    elif event == "participant_events.join":
        participant = data.get("data", {}).get("participant", {})
        name = participant.get("name") or f"Participant-{participant.get('id', '?')}"
        logger.info(f"Participant joined: {name}")

    elif event == "participant_events.leave":
        participant = data.get("data", {}).get("participant", {})
        name = participant.get("name") or f"Participant-{participant.get('id', '?')}"
        logger.info(f"Participant left: {name}")

    return {"ok": True}


async def process_and_respond(bot_id: str, speaker: str, text: str, ts: str = "", t0_webhook: float = 0):
    if bot_id not in active_bots:
        return

    bot_state = active_bots[bot_id]
    lock = bot_state.get("respond_lock")
    if not lock:
        return

    if ts and ts == bot_state.get("last_processed_ts"):
        return
    if ts:
        bot_state["last_processed_ts"] = ts

    if lock.locked():
        logger.info("Skipping — bot already processing a response")
        return

    async with lock:
        if bot_id not in active_bots:
            return

        t1_pre_debounce = asyncio.get_event_loop().time()
        await asyncio.sleep(0.5)
        t2_post_debounce = asyncio.get_event_loop().time()

        if bot_id not in active_bots:
            return

        # LLM (now with voice gateway routing)
        context_msg = f"{speaker} said: {text}"
        t3_llm_start = asyncio.get_event_loop().time()
        response_text = await generate_response(bot_state["conversation"], context_msg, bot_state=bot_state)
        t4_llm_end = asyncio.get_event_loop().time()
        llm_ms = (t4_llm_end - t3_llm_start) * 1000

        if not response_text:
            logger.info(f"Hank chose to stay silent after {speaker}'s message")
            return

        logger.info(f"Hank responds: {response_text}")

        bot_state["conversation"].append({"role": "assistant", "content": response_text})
        bot_state["transcript"].append({
            "speaker": "Hank Bob",
            "text": response_text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # TTS (Kokoro + RVC)
        bot_state["speaking"] = True
        t5_tts_start = asyncio.get_event_loop().time()
        try:
            filename = await generate_tts_audio(response_text, bot_id)
            t6_tts_end = asyncio.get_event_loop().time()
            tts_ms = (t6_tts_end - t5_tts_start) * 1000

            if filename:
                total_ms = (t6_tts_end - t0_webhook) * 1000 if t0_webhook else 0
                debounce_ms = (t2_post_debounce - t1_pre_debounce) * 1000
                logger.info(
                    f"BENCH | debounce={debounce_ms:.0f}ms llm={llm_ms:.0f}ms tts={tts_ms:.0f}ms "
                    f"total_server={total_ms:.0f}ms | audio={filename}"
                )
            else:
                logger.error("TTS generation failed — no audio produced")
        finally:
            bot_state["speaking"] = False

@app.post("/api/bench/setup")
async def bench_setup():
    bot_id = f"bench-{uuid.uuid4().hex[:8]}"
    active_bots[bot_id] = {
        "meeting_url": "bench://synthetic",
        "status": "bench",
        "transcript": [],
        "conversation": [],
        "speaking": False,
        "respond_lock": asyncio.Lock(),
        "last_processed_ts": "",
        "expanded_entries": set(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return {"bot_id": bot_id, "status": "ready for bench"}

@app.get("/api/bots")
async def list_bots():
    return {"bots": {k: {kk: vv for kk, vv in v.items() if kk != "conversation"} for k, v in active_bots.items()}}

@app.post("/api/bench/tts")
async def benchmark_tts_only(request: Request):
    """Benchmark just the TTS pipeline (Kokoro+RVC) without LLM. Provide text directly."""
    body = await request.json()
    text = body.get("text", "I tell ya what, that propane grill is the only way to cook a proper burger.")
    bot_id = body.get("bot_id")

    if not bot_id or bot_id not in active_bots:
        raise HTTPException(400, f"bot_id required and must be active. Active: {list(active_bots.keys())}")

    timings = {}

    t0 = asyncio.get_event_loop().time()

    # TTS only
    t1 = asyncio.get_event_loop().time()
    filename = await generate_tts_audio(text, bot_id)
    t2 = asyncio.get_event_loop().time()

    timings["tts_ms"] = round((t2 - t1) * 1000)
    timings["total_ms"] = round((t2 - t0) * 1000)
    timings["input_text"] = text
    timings["input_chars"] = len(text)
    timings["audio_file"] = filename
    timings["tts_engine"] = "local_kokoro_rvc"

    if filename:
        fp = AUDIO_DIR / filename
        if fp.exists():
            data, sr = sf.read(str(fp))
            timings["audio_duration_s"] = round(len(data) / sr, 2)
            timings["real_time_factor"] = round(timings["tts_ms"] / (len(data) / sr * 1000), 3)
            timings["audio_size_kb"] = round(fp.stat().st_size / 1024, 1)

    return {"timings": timings}

@app.post("/api/bench")
async def benchmark_pipeline(request: Request):
    """Run a synthetic benchmark: fake transcript → LLM (voice gateway) → Kokoro+RVC TTS → audio ready."""
    body = await request.json()
    text = body.get("text", "Hey Hank, what do you think about MLX?")
    bot_id = body.get("bot_id")

    if not bot_id or bot_id not in active_bots:
        raise HTTPException(400, f"bot_id required and must be active. Active: {list(active_bots.keys())}")

    bot_state = active_bots[bot_id]
    lock = bot_state.get("respond_lock")
    if not lock:
        raise HTTPException(400, "No lock on bot")

    if lock.locked():
        return {"error": "Bot is already processing a response", "status": "busy"}

    timings = {}

    async with lock:
        t0 = asyncio.get_event_loop().time()

        # LLM (voice gateway routed)
        t2 = asyncio.get_event_loop().time()
        if not bot_state["conversation"]:
            bot_state["conversation"] = [
                {"role": "user", "content": "Ethan: Hey everyone, Hank Bob is here with us today."},
                {"role": "assistant", "content": "Hey! Good to be here. What are we working on?"},
            ]
        context_msg = f"Ethan said: {text}"
        response_text = await generate_response(bot_state["conversation"], context_msg, bot_state=bot_state)
        t3 = asyncio.get_event_loop().time()
        timings["llm_ms"] = round((t3 - t2) * 1000)

        path = MemorySnapshot.classify_query(text)
        timings["llm_path"] = path

        if not response_text:
            return {"error": "LLM chose to stay silent", "timings": timings}

        # TTS (Kokoro + RVC)
        t4 = asyncio.get_event_loop().time()
        filename = await generate_tts_audio(response_text, bot_id)
        t5 = asyncio.get_event_loop().time()
        timings["tts_ms"] = round((t5 - t4) * 1000)

        timings["total_server_ms"] = round((t5 - t0) * 1000)
        timings["response_text"] = response_text
        timings["audio_file"] = filename
        timings["tts_engine"] = "local_kokoro_rvc"

        # Get actual audio duration
        if filename:
            fp = AUDIO_DIR / filename
            if fp.exists():
                data, sr = sf.read(str(fp))
                timings["audio_duration_s"] = round(len(data) / sr, 2)

    return {"timings": timings}

# Memory snapshot debug endpoint
@app.get("/api/memory")
async def get_memory_snapshot():
    memory.refresh_if_stale()
    return {
        "summary": memory.summary,
        "entry_count": len(memory.entries),
        "index": memory.index_text,
        "entries": [
            {"index": e["index"], "source": e["source"], "text": e["text"][:200]}
            for e in memory.entries
        ],
    }


if __name__ == "__main__":
    if not RECALL_API_KEY:
        print("ERROR: RECALL_API_KEY not found")
        sys.exit(1)
    if not OPENROUTER_KEY:
        print("ERROR: OPENROUTER_API_KEY not found")
        sys.exit(1)

    print(f"Starting Hank Bob Meeting Agent on port {PORT}")
    print(f"Public URL: https://meet.model-optimizors.com")
    print(f"LLM routing: fast={FAST_MODEL} standard={STANDARD_MODEL} deep={DEEP_MODEL}")
    print(f"TTS: Local Kokoro+RVC (Hank Hill)")
    print(f"Memory: {len(memory.entries)} entries loaded")
    print(f"RVC model: {RVC_MODEL_PATH}")
    print(f"Audio: WAV → decodeAudioData → AudioBufferSourceNode")

    uvicorn.run(app, host="0.0.0.0", port=PORT)
