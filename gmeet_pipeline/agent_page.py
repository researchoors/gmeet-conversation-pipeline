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
    font-family: -apple-system, BlinkMacSystemFont, 'SF Mono', monospace;
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100vh;
    overflow: hidden;
  }
  .main {
    display: flex;
    width: 100%;
    height: 100vh;
  }
  .center {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
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
    transition: all 0.3s;
  }
  .speaking .avatar {
    border-color: #95BFFF;
    box-shadow: 0 0 30px rgba(149, 191, 255, 0.5);
    animation: glow 1s infinite;
  }
  .name { font-size: 18px; font-weight: 600; color: #E0E0E0; }
  .status { font-size: 13px; color: #888; margin-top: 6px; }
  .status-listening { color: #4ADE80; }
  .status-llm { color: #FBBF24; }
  .status-tts { color: #F97316; }
  .status-speaking { color: #4ADE80; animation: pulse 1s infinite; }
  .status-queuing { color: #EF4444; }
  .response-text {
    margin-top: 16px; font-size: 18px; color: #F2F2F2;
    max-width: 500px; line-height: 1.4; min-height: 28px;
    text-align: center;
  }
  /* Debug overlay — right panel */
  .debug-panel {
    width: 260px;
    background: #0d0d0d;
    border-left: 1px solid #1a1a2e;
    padding: 12px;
    font-size: 11px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .debug-section {
    border-bottom: 1px solid #1a1a1a;
    padding-bottom: 8px;
  }
  .debug-section:last-child { border-bottom: none; }
  .debug-label {
    color: #555;
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 3px;
  }
  .debug-value { color: #ccc; font-size: 12px; }
  .debug-value.highlight { color: #95BFFF; }
  .debug-value.warn { color: #FBBF24; }
  .debug-value.error { color: #EF4444; }
  .debug-value.good { color: #4ADE80; }
  .participant-list {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }
  .participant-chip {
    background: #1a1a2e;
    color: #95BFFF;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
  }
  .pipeline-bar {
    display: flex;
    gap: 2px;
    margin-top: 4px;
  }
  .pipeline-step {
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 9px;
    background: #1a1a1a;
    color: #555;
  }
  .pipeline-step.active {
    background: #1a1a3a;
    color: #95BFFF;
  }
  .pipeline-step.active-step {
    background: #1a3a1a;
    color: #4ADE80;
    font-weight: 600;
  }
  .badge {
    position: fixed;
    top: 8px;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 10px;
    font-weight: 600;
  }
  .tts-badge { left: 8px; background: #1a3a1a; color: #4CAF50; }
  .memory-badge { left: 110px; background: #1a1a3a; color: #95BFFF; }
  .llm-badge { left: 230px; background: #1a1a3a; color: #FBBF24; }
  @keyframes glow {
    0%, 100% { box-shadow: 0 0 30px rgba(149, 191, 255, 0.3); }
    50% { box-shadow: 0 0 50px rgba(149, 191, 255, 0.6); }
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
</style>
</head>
<body>
<div class="badge tts-badge">🔊 Kokoro</div>
<div class="badge memory-badge">🧠 Flash</div>
<div class="badge llm-badge" id="llmBadge">⚡ Gemini Flash</div>
<div class="main">
  <div class="center">
    <div class="container" id="agent">
      <div class="avatar">HB</div>
      <div class="name">Hank Bob</div>
      <div class="status" id="status">Connecting...</div>
      <div class="response-text" id="responseText"></div>
    </div>
  </div>
  <div class="debug-panel" id="debugPanel">
    <div class="debug-section">
      <div class="debug-label">Pipeline</div>
      <div class="pipeline-bar" id="pipelineBar">
        <span class="pipeline-step" id="step-idle">idle</span>
        <span class="pipeline-step" id="step-llm">LLM</span>
        <span class="pipeline-step" id="step-tts">TTS</span>
        <span class="pipeline-step" id="step-speaking">speak</span>
      </div>
    </div>
    <div class="debug-section">
      <div class="debug-label">Queue</div>
      <div class="debug-value" id="queueDepth">0 pending</div>
    </div>
    <div class="debug-section">
      <div class="debug-label">Last Latency</div>
      <div class="debug-value" id="latencyInfo">—</div>
    </div>
    <div class="debug-section">
      <div class="debug-label">Participants</div>
      <div class="participant-list" id="participantList">
        <span style="color:#555">none yet</span>
      </div>
    </div>
    <div class="debug-section">
      <div class="debug-label">Last Transcript</div>
      <div class="debug-value" id="lastTranscript" style="font-size:10px;max-height:60px;overflow:hidden">—</div>
    </div>
  </div>
</div>

<script>
const agentEl = document.getElementById('agent');
const statusEl = document.getElementById('status');
const responseTextEl = document.getElementById('responseText');
const queueDepthEl = document.getElementById('queueDepth');
const latencyInfoEl = document.getElementById('latencyInfo');
const participantListEl = document.getElementById('participantList');
const lastTranscriptEl = document.getElementById('lastTranscript');

let isSpeaking = false;
let currentSource = null;
let lastAudioCount = 0;
let audioCtx = null;

async function initAudio() {
  if (audioCtx) {
    if (audioCtx.state === 'suspended') await audioCtx.resume();
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
}

function debug(msg) {
  console.log('[HankBob]', msg);
  fetch('/api/debug', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({msg})}).catch(()=>{});
}

function updatePipelineUI(state) {
  const labels = {
    idle: 'Listening...',
    queuing: 'Queued...',
    llm: 'Thinking...',
    tts: 'Generating speech...',
    speaking: 'Speaking...',
  };
  statusEl.textContent = labels[state] || state;
  statusEl.className = 'status status-' + state;
  agentEl.className = (state === 'speaking' || state === 'llm' || state === 'tts') ? 'container speaking' : 'container';

  const allSteps = ['idle', 'llm', 'tts', 'speaking'];
  const activeIdx = allSteps.indexOf(state);
  allSteps.forEach((step, i) => {
    const el = document.getElementById('step-' + step);
    if (!el) return;
    el.className = 'pipeline-step';
    if (i < activeIdx) el.classList.add('active');
    if (i === activeIdx) el.classList.add('active-step');
  });
}

function formatLatency(ms) {
  if (!ms || ms === 0) return '—';
  if (ms < 1000) return ms + 'ms';
  return (ms / 1000).toFixed(1) + 's';
}

async function pollAudio() {
  try {
    const resp = await fetch('/api/audio-queue');
    if (!resp.ok) return;
    const data = await resp.json();
    const items = data.items || [];
    if (items.length > lastAudioCount) {
      const latest = items[items.length - 1];
      if (latest && latest.filename) await playAudio(latest.filename, latest.text);
      lastAudioCount = items.length;
    }
  } catch (e) {}
}

async function pollSessionState() {
  try {
    const resp = await fetch('/api/session-state');
    if (!resp.ok) return;
    const data = await resp.json();
    const bot = (data.bots || [])[0];
    if (!bot) return;

    updatePipelineUI(bot.pipeline_state || 'idle');

    const qd = bot.queue_depth || 0;
    queueDepthEl.textContent = qd + ' pending';
    queueDepthEl.className = 'debug-value' + (qd > 2 ? ' error' : qd > 0 ? ' warn' : '');

    if (bot.last_total_ms > 0) {
      latencyInfoEl.innerHTML =
        '<span class="debug-value highlight">Total ' + formatLatency(bot.last_total_ms) + '</span>' +
        '<br>LLM ' + formatLatency(bot.last_llm_ms) + ' · TTS ' + formatLatency(bot.last_tts_ms);
    }

    const parts = bot.participants || [];
    if (parts.length > 0) {
      participantListEl.innerHTML = parts.map(p =>
        '<span class="participant-chip">' + p + '</span>'
      ).join('');
    }

    const last = bot.last_transcript;
    if (last) {
      lastTranscriptEl.textContent = (last.speaker || '?') + ': ' + (last.text || '').slice(0, 80);
    }
  } catch (e) {}
}

async function playAudio(filename, text) {
  if (isSpeaking) return;
  isSpeaking = true;
  updatePipelineUI('speaking');
  responseTextEl.textContent = text || '';

  try {
    await initAudio();
    if (audioCtx.state === 'suspended') await audioCtx.resume();

    const audioResp = await fetch('/audio/' + filename);
    if (!audioResp.ok) throw new Error('fetch failed: ' + audioResp.status);
    const arrayBuffer = await audioResp.arrayBuffer();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

    const source = audioCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioCtx.destination);
    currentSource = source;

    source.onended = () => {
      isSpeaking = false;
      currentSource = null;
      updatePipelineUI('idle');
      setTimeout(() => { responseTextEl.textContent = ''; }, 3000);
    };

    source.start(0);
    debug('playing ' + audioBuffer.duration.toFixed(1) + 's audio');
  } catch (e) {
    debug('audio_error: ' + (e?.message || e));
    isSpeaking = false;
    currentSource = null;
    updatePipelineUI('idle');
  }
}

updatePipelineUI('idle');
setInterval(pollAudio, 500);
setInterval(pollSessionState, 1000);
</script>
</body>
</html>
"""
