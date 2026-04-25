"""
Hank Bob Meeting Agent — Recall.ai integration
Kokoro+RVC local TTS pipeline (replaces ElevenLabs REST)

Architecture:
- Recall webhook → transcript → LLM → Kokoro TTS → RVC voice conversion → WAV → AudioBufferSourceNode
- Zero API calls for TTS — fully local on Apple Silicon
- asyncio.Lock per bot prevents concurrent/duplicate responses
"""

import os
import sys
import json
import asyncio
import logging
import uuid
import time
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
LLM_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4")

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

PORT = 9120
AUDIO_DIR = Path.home() / ".hermes" / "audio_cache" / "meeting_tts"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# VRM avatar directory
AVATAR_DIR = Path.home() / ".hermes" / "avatars"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger("meeting-agent")

app = FastAPI(title="Hank Bob Meeting Agent")

# In-memory state
active_bots: dict = {}

# Audio queue: bot_id -> list of {id, text, filename}
audio_queue: dict = {}

# ============================================================
# Load env
# ============================================================
def load_env():
    global RECALL_API_KEY, OPENROUTER_KEY
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

load_env()

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
            # Cleanup temp file
            try:
                tmp_wav.unlink()
            except:
                pass

# Singleton
tts_engine = LocalTTSEngine()


# ============================================================
# System prompt
# ============================================================
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


# ============================================================
# LLM Call
# ============================================================
async def generate_response(conversation: list, new_message: str) -> Optional[str]:
    """Call the LLM to generate a response. Returns None if Hank shouldn't respond."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in conversation[-20:]:
        messages.append(msg)
    messages.append({"role": "user", "content": new_message})

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://meet.model-optimizors.com",
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
            if content.upper() in ("...", "SILENT", "NO_RESPONSE", "PASS", "SKIP"):
                return None
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
AGENT_HTML = """<!DOCTYPE html>
<html>
<head>
<title>Hank Bob</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #0d0d0d;
    overflow: hidden;
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
  }
  #canvas {
    width: 100vw;
    height: 100vh;
    display: block;
  }
  .badge {
    position: fixed; top: 12px; right: 12px;
    background: #1a3a1a; color: #4CAF50; padding: 4px 10px;
    border-radius: 8px; font-size: 11px; font-weight: 600;
    color: #F2F2F2; z-index: 10;
  }
  .status-bar {
    position: fixed; bottom: 40px; left: 0; right: 0;
    text-align: center;
    font-size: 14px;
    color: #888;
    height: 20px;
    z-index: 10;
  }
  .response-text {
    position: fixed; bottom: 10px; left: 0; right: 0;
    text-align: center;
    font-size: 18px;
    color: #E0E0E0;
    max-width: 500px;
    margin: 0 auto;
    line-height: 1.4;
    min-height: 28px;
    z-index: 10;
  }
  .loading {
    position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
    color: #95BFFF; font-size: 18px; z-index: 20;
  }
</style>
</head>
<body>
<div class="badge">🔊 Kokoro+RVC · 🧠 Snapshot</div>
<div class="loading" id="loading">Loading avatar...</div>
<div class="status-bar" id="status">Listening...</div>
<div class="response-text" id="responseText"></div>
<canvas id="canvas"></canvas>

<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.169.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.169.0/examples/jsm/",
    "@pixiv/three-vrm": "https://cdn.jsdelivr.net/npm/@pixiv/three-vrm@3.1.0/lib/three-vrm.module.js"
  }
}
</script>

<script type="module">
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from 'https://cdn.jsdelivr.net/npm/@pixiv/three-vrm@3.1.0/lib/three-vrm.module.js';

const statusEl = document.getElementById('status');
const responseTextEl = document.getElementById('responseText');
const loadingEl = document.getElementById('loading');

let isSpeaking = false;
let currentSource = null;
let pollInterval = null;
let lastAudioCount = 0;
let audioCtx = null;
let analyser = null;
let animFrame = null;
let vrm = null;
let clock = new THREE.Clock();

// ============================================================
// Three.js Scene
// ============================================================
const canvas = document.getElementById('canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d0d0d);

// Camera
const camera = new THREE.PerspectiveCamera(30, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 1.3, 2.0);
camera.lookAt(0, 1.0, 0);

// Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambientLight);

const keyLight = new THREE.DirectionalLight(0xffffff, 1.2);
keyLight.position.set(1, 2, 1);
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0x95BFFF, 0.4);
fillLight.position.set(-1, 1, 1);
scene.add(fillLight);

const rimLight = new THREE.DirectionalLight(0x95BFFF, 0.6);
rimLight.position.set(0, 1.5, -1);
scene.add(rimLight);

// ============================================================
// Load VRM
// ============================================================
const loader = new GLTFLoader();
loader.register((parser) => new VRMLoaderPlugin(parser));

loader.load(
  '/avatar.vrm',
  (gltf) => {
    vrm = gltf.userData.vrm;
    VRMUtils.removeUnnecessaryVertices(gltf.scene);
    VRMUtils.removeUnnecessaryJoints(gltf.scene);
    
    // Rotate to face camera (VRM faces +Z by default)
    vrm.scene.rotation.y = Math.PI;
    scene.add(vrm.scene);

    // Expose VRM to window for debugging
    window.__vrm = vrm;
    
    // Build viseme map for lip sync
    buildVisemeMap(vrm);
    
    loadingEl.style.display = 'none';
    
    // Log available expressions
    if (vrm.expressionManager) {
      const exprs = vrm.expressionManager.expressions;
      const names = exprs ? Object.keys(exprs) : [];
      console.log('[HankBob] Expression names:', names.join(', '));
      
      // Try to get actual expression names from the VRM data
      const expressionMap = {};
      names.forEach(name => {
        try {
          const expr = exprs[name];
          if (expr && expr.name) expressionMap[name] = expr.name;
        } catch(e) {}
      });
      console.log('[HankBob] Expression map:', JSON.stringify(expressionMap));
      
      // Check for visemes (VRM1 uses VRMC_vrm_expression with presetName)
      const visemes = ['aa', 'ih', 'ou', 'ee', 'oh', 'happy', 'angry', 'sad', 'surprised', 'neutral'];
      const found = [];
      names.forEach(name => {
        try {
          const expr = exprs[name];
          const preset = expr?.presetName || expr?.name || '';
          if (visemes.includes(preset.toLowerCase())) found.push(preset);
        } catch(e) {}
      });
      console.log('[HankBob] Visemes found:', found.join(', ') || 'NONE — will use amplitude fallback');
    }
  },
  (progress) => {
    console.log('[HankBob] Loading:', (progress.loaded / progress.total * 100).toFixed(0) + '%');
  },
  (error) => {
    console.error('[HankBob] VRM load error:', error);
    loadingEl.textContent = 'Avatar load failed — using fallback';
    setTimeout(() => { loadingEl.style.display = 'none'; }, 3000);
  }
);

// ============================================================
// Viseme mapping
// ============================================================
// This VRM has expressions indexed 0-17 with names VRMExpression_aa, VRMExpression_ee, etc.
// We need to find the viseme indices at runtime
let VISEME_MAP = {}; // will be populated after VRM load
let currentVisemeWeights = {};

function buildVisemeMap(vrm) {
  const exprs = vrm.expressionManager.expressions;
  const visemeNames = ['aa', 'ih', 'ou', 'ee', 'oh'];
  const nameToIdx = {};
  Object.keys(exprs).forEach(idx => {
    try {
      const name = exprs[idx].name || '';
      // Extract the expression type from "VRMExpression_aa"
      const match = name.match(/VRMExpression_(\w+)/);
      if (match) nameToIdx[match[1]] = parseInt(idx);
    } catch(e) {}
  });
  visemeNames.forEach(v => {
    if (nameToIdx[v] !== undefined) {
      VISEME_MAP[v] = nameToIdx[v];
    }
  });
  console.log('[HankBob] Viseme map:', JSON.stringify(VISEME_MAP));
}

// Frequency band mapping to visemes
// Low freq → "oh" (round mouth), mid → "aa" (open), high → "ee" (wide)
function audioToVisemeWeights(dataArray) {
  const len = dataArray.length;
  
  // Split into frequency bands
  let low = 0, mid = 0, high = 0;
  const lowEnd = Math.floor(len * 0.15);
  const midEnd = Math.floor(len * 0.4);
  
  for (let i = 1; i < lowEnd; i++) low += dataArray[i];
  for (let i = lowEnd; i < midEnd; i++) mid += dataArray[i];
  for (let i = midEnd; i < Math.min(len, midEnd + 20); i++) high += dataArray[i];
  
  low /= (lowEnd - 1);
  mid /= (midEnd - lowEnd);
  high /= 20;
  
  // Overall amplitude
  const amplitude = (low + mid + high) / 3;
  const norm = Math.min(1, amplitude / 100);
  
  if (norm < 0.08) {
    // Silent — close mouth
    return { aa: 0, ih: 0, ou: 0, ee: 0, oh: 0 };
  }
  
  // Blend based on frequency distribution
  const total = low + mid + high + 1;
  const lowR = low / total;
  const midR = mid / total;
  const highR = high / total;
  
  return {
    aa: norm * midR * 2.5,      // open mouth — mid frequencies
    ih: norm * 0.3,             // slight — always a bit
    ou: norm * lowR * 2.5,      // round mouth — low frequencies
    ee: norm * highR * 2.5,     // wide mouth — high frequencies
    oh: norm * lowR * 1.5,      // round — low freq
  };
}

// Set expression by viseme name using index
function setExpression(visemeName, weight) {
  const idx = VISEME_MAP[visemeName];
  if (idx === undefined) return;
  try {
    vrm.expressionManager.setValue(idx, Math.min(1, Math.max(0, weight)));
  } catch(e) {}
}

// ============================================================
// Animation loop
// ============================================================
function animate() {
  requestAnimationFrame(animate);
  
  const delta = clock.getDelta();
  
  if (vrm) {
    // Update VRM
    vrm.update(delta);
    
    // Lip sync: apply viseme weights from audio analyser
    if (isSpeaking && analyser) {
      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      analyser.getByteFrequencyData(dataArray);
      
      const weights = audioToVisemeWeights(dataArray);
      
      // Smooth transition
      const smooth = 0.3;
      Object.keys(weights).forEach(v => {
        if (!currentVisemeWeights[v]) currentVisemeWeights[v] = 0;
        currentVisemeWeights[v] = currentVisemeWeights[v] * (1 - smooth) + (weights[v] || 0) * smooth;
        setExpression(v, currentVisemeWeights[v]);
      });
    } else {
      // Close mouth when not speaking
      const decay = 0.15;
      Object.keys(VISEME_MAP).forEach(v => {
        if (!currentVisemeWeights[v]) currentVisemeWeights[v] = 0;
        currentVisemeWeights[v] *= (1 - decay);
        setExpression(v, currentVisemeWeights[v]);
      });
    }
    
    // Idle animation: subtle breathing + slight sway
    const t = clock.elapsedTime;
    if (vrm.humanoid) {
      try {
        const chest = vrm.humanoid.getNormalizedBoneNode('chest');
        if (chest) {
          chest.rotation.x = Math.sin(t * 1.2) * 0.015;
        }
        const head = vrm.humanoid.getNormalizedBoneNode('head');
        if (head) {
          head.rotation.y = Math.sin(t * 0.5) * 0.03;
          head.rotation.x = Math.sin(t * 0.7) * 0.01;
        }
        const spine = vrm.humanoid.getNormalizedBoneNode('spine');
        if (spine) {
          spine.rotation.x = Math.sin(t * 1.2) * 0.01;
        }
      } catch(e) {}
    }
  }
  
  renderer.render(scene, camera);
}
animate();

// Resize
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ============================================================
// Audio + Polling (same as before but with lip sync)
// ============================================================
async function initAudio() {
  if (audioCtx) {
    if (audioCtx.state === 'suspended') await audioCtx.resume();
    return;
  }
  audioCtx = new AudioContext();
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 256;
  analyser.smoothingTimeConstant = 0.7;
  analyser.connect(audioCtx.destination);

  if (audioCtx.state === 'suspended') {
    const tryResume = async () => {
      try { await audioCtx.resume(); } catch(e) {}
      if (audioCtx.state === 'suspended') setTimeout(tryResume, 100);
    };
    tryResume();
  }
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
  responseTextEl.textContent = text || '';

  try {
    await initAudio();
    if (audioCtx.state === 'suspended') await audioCtx.resume();

    const audioResp = await fetch('/audio/' + filename);
    const arrayBuffer = await audioResp.arrayBuffer();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

    const source = audioCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(analyser);
    currentSource = source;

    source.onended = () => {
      isSpeaking = false;
      currentSource = null;
      statusEl.textContent = 'Listening...';
      setTimeout(() => { responseTextEl.textContent = ''; }, 3000);
    };

    source.start(0);

  } catch (e) {
    debug('audio_error: ' + String(e));
    isSpeaking = false;
    currentSource = null;
    statusEl.textContent = 'Listening...';
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

# VRM avatar endpoint
@app.get("/avatar.vrm")
async def serve_avatar():
    """Serve the VRM avatar file for the 3D character."""
    vrm_path = AVATAR_DIR / "avatar.vrm"
    if not vrm_path.exists():
        # Try constraint.vrm as fallback
        vrm_path = AVATAR_DIR / "constraint.vrm"
    if not vrm_path.exists():
        raise HTTPException(404, "Avatar file not found")
    return FileResponse(
        str(vrm_path),
        media_type="model/vrm",
        headers={"Cache-Control": "public, max-age=3600"},
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

        # LLM
        context_msg = f"{speaker} said: {text}"
        t3_llm_start = asyncio.get_event_loop().time()
        response_text = await generate_response(bot_state["conversation"], context_msg)
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
    """Run a synthetic benchmark: fake transcript → LLM → Kokoro+RVC TTS → audio ready."""
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

        # LLM
        t2 = asyncio.get_event_loop().time()
        # Seed conversation so Hank knows he's in a call and should respond
        if not bot_state["conversation"]:
            bot_state["conversation"] = [
                {"role": "user", "content": "Ethan: Hey everyone, Hank Bob is here with us today."},
                {"role": "assistant", "content": "Hey! Good to be here. What are we working on?"},
            ]
        context_msg = f"Ethan said: {text}"
        response_text = await generate_response(bot_state["conversation"], context_msg)
        t3 = asyncio.get_event_loop().time()
        timings["llm_ms"] = round((t3 - t2) * 1000)

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


if __name__ == "__main__":
    if not RECALL_API_KEY:
        print("ERROR: RECALL_API_KEY not found")
        sys.exit(1)
    if not OPENROUTER_KEY:
        print("ERROR: OPENROUTER_API_KEY not found")
        sys.exit(1)

    print(f"Starting Hank Bob Meeting Agent on port {PORT}")
    print(f"Public URL: https://meet.model-optimizors.com")
    print(f"LLM: {LLM_MODEL}")
    print(f"TTS: Local Kokoro+RVC (Hank Hill)")
    print(f"RVC model: {RVC_MODEL_PATH}")
    print(f"Audio: WAV → decodeAudioData → AudioBufferSourceNode")

    uvicorn.run(app, host="0.0.0.0", port=PORT)
