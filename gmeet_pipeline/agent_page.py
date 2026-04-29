"""Agent page HTML templates served to Recall's Output Media."""

# Shared base prompt for both agent pages
AGENT_DISPLAY_NAME = "Hank Bob"
AGENT_INITIALS = "HB"


def get_agent_html(tts_backend: str = "elevenlabs", memory_enabled: bool = False) -> str:
    """Generate agent page HTML based on configuration.
    
    Args:
        tts_backend: "elevenlabs" for WebSocket streaming, "local" for WAV polling
        memory_enabled: Show memory badge in UI
    """
    if tts_backend == "local":
        return _LOCAL_AGENT_HTML
    return _ELEVENLABS_AGENT_HTML


# --- ElevenLabs streaming agent page (WebSocket PCM) ---

_ELEVENLABS_AGENT_HTML = """<!DOCTYPE html>
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

let audioCtx = null;
let isSpeaking = false;
let pcmBuffer = null;
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
  audio.onerror = (e) => {
    debug('fallback_audio_error: ' + (e?.message || 'Event(type=' + e.type + ')') + ' src=' + url);
    isSpeaking = false;
    statusEl.textContent = 'Listening...';
    agentEl.className = 'container';
  };
  audio.onended = () => {
    statusEl.textContent = 'Listening...';
    agentEl.className = 'container';
    isSpeaking = false;
    setTimeout(() => { responseTextEl.textContent = ''; }, 3000);
  };
  audio.play().catch(e => {
    debug('fallback_play_error: ' + e?.message);
    isSpeaking = false;
    statusEl.textContent = 'Listening...';
    agentEl.className = 'container';
  });
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
  if (pcmBuffer && pcmBuffer.length > 0) {
    playCompletePCM(pcmBuffer, currentSampleRate);
  } else {
    isSpeaking = false;
    statusEl.textContent = 'Listening...';
    agentEl.className = 'container';
  }
}

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
    if (audioCtx && audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
    startPing();
  };

  ws.onmessage = (event) => {
    if (typeof event.data === 'string') {
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

let pingInterval = null;
function startPing() {
  if (pingInterval) clearInterval(pingInterval);
  pingInterval = setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send('ping');
    }
  }, 30000);
}

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

document.addEventListener('click', () => {
  if (audioCtx && audioCtx.state === 'suspended') audioCtx.resume();
}, { once: true });

connectWebSocket();
pollTranscript();
</script>
</body>
</html>
"""


# --- Local TTS agent page (WAV polling) ---

_LOCAL_AGENT_HTML = """<!DOCTYPE html>
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
    if (!audioResp.ok) {
      throw new Error('fetch failed: ' + audioResp.status + ' ' + audioResp.statusText);
    }
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
    const errDetail = e?.message || e?.name || (e instanceof Event ? `Event(type=${e.type})` : String(e));
    debug('audio_error: ' + errDetail + ' | fetch_status=' + (audioResp?.status || 'n/a'));
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
