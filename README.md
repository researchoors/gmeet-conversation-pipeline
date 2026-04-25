# gmeet-conversation-pipeline

Google Meet conversation agent powered by [Recall.ai](https://recall.ai). Joins calls as a bot, receives real-time transcripts, generates contextual responses via LLM, and speaks them back through TTS — all in a streaming pipeline.

## System Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────────────────────────────────────┐
│ Google Meet  │────▶│  Recall.ai   │────▶│              gmeet_pipeline server               │
│ Participants │     │   bot API    │     │                                                  │
└─────────────┘     └──────┬───────┘     │  ┌─────────────┐  webhook /status_change          │
                           │             │  │  Transport   │  /transcript.data               │
                    webhook events        │  │  (RecallAI)  │  /participant_events            │
                    (transcript,          │  └──────┬──────┘                                  │
                     status, etc.)        │         │                                         │
                           │             │         ▼                                         │
                           └────────────▶│  ┌──────────────┐                                 │
                                         │  │   Webhook     │  fires _process_and_respond()   │
                                         │  │  Handler      │  with per-bot asyncio.Lock      │
                                         │  └──────┬───────┘                                 │
                                         │         │                                         │
                                         │         ▼                                         │
                                         │  ┌──────────────┐                                 │
                                         │  │     LLM       │  simple → single model          │
                                         │  │  (OpenRouter) │  voice_gateway → classify query │
                                         │  │               │    fast / standard / deep         │
                                         │  └──────┬───────┘    + RAG + EXPAND[n] memory       │
                                         │         │                                         │
                                         │         ▼                                         │
                                         │  ┌──────────────┐                                 │
                                         │  │     TTS       │  elevenlabs → PCM WS streaming  │
                                         │  │               │  local → Kokoro + RVC → WAV     │
                                         │  └──────┬───────┘                                 │
                                         │         │                                         │
                                         │         ▼                                         │
                                         │  ┌──────────────┐                                 │
                                         │  │  WS Manager   │  broadcast PCM/JSON → agent page │
                                         │  └──────────────┘                                 │
                                         └──────────────────────────────────────────────────┘
```

### Data flow

1. **Bot joins** — `POST /api/bot/join` → Recall Transport creates a bot → BotSession registered in BotRegistry
2. **Transcript arrives** — Recall webhook `POST /webhook/recall` → WebhookHandler parses speaker + text → appends to session conversation
3. **Response pipeline** — `_process_and_respond()` acquires per-bot lock, debounces 500ms, then:
   - **LLM** generates a response (or stays silent via `SILENT` tokens)
   - **TTS** converts response to audio
   - **WS Manager** streams PCM (ElevenLabs) or serves WAV file (local)
4. **Agent page** — Browser at `/` receives audio via WebSocket `/ws/audio` and plays it through the meeting

### Pluggable backends

| Layer | Base class | Implementations | Selected by |
|-------|-----------|----------------|-------------|
| Transport | `BaseTransport` | `RecallTransport` | `RECALL_API_KEY` |
| LLM | `BaseLLM` | `SimpleOpenRouterLLM`, `VoiceGatewayLLM` | `LLM_ROUTING` |
| TTS | `BaseTTS` | `ElevenLabsTTS`, `LocalTTS` | `TTS_BACKEND` |

**SimpleOpenRouterLLM** — single model, no memory. Quick setup.

**VoiceGatewayLLM** — memory-aware multi-model routing:
- `fast` → Gemini Flash (greetings, short replies)
- `standard` → GPT-4.1-mini + RAG from MemorySnapshot (project questions)
- `deep` → Claude Sonnet + full RAG + EXPAND (tool use, complex queries)
- Parses `EXPAND[n]` directives for multi-turn memory drill-down

**ElevenLabsTTS** — WebSocket PCM streaming (primary) with REST MP3 fallback. Streams 22kHz signed 16-bit LE PCM to agent pages.

**LocalTTS** — Kokoro KPipeline → optional RVC voice conversion → WAV file. Runs inference in a thread pool to avoid blocking the event loop. Heavy ML deps (Kokoro, RVC, torch, numpy) are lazy-loaded on first use.

### Key modules

| Module | Responsibility |
|--------|---------------|
| `main.py` | Composition root — loads config, wires all components via DI, runs uvicorn |
| `config.py` | Pydantic-settings single source of truth (`GMEET_*` env vars, `.env`, `auth.json` fallback) |
| `server.py` | FastAPI routes — delegates all logic to injected components |
| `webhook.py` | Recall webhook handler — orchestrates the transcript → LLM → TTS pipeline |
| `state.py` | `BotRegistry` + `BotSession` — per-bot conversation, transcript, locks |
| `ws_manager.py` | WebSocket connection manager — broadcasts PCM/JSON to agent pages |
| `memory.py` | `MemorySnapshot` — parses `MEMORY.md` + `USER.md`, RAG retrieval, query classification |
| `agent_page.py` | Serves the browser-based agent control page |

## Quick start

1. Copy `.env.example` to `.env` and fill in your credentials
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python -m gmeet_pipeline.main`

### With Docker

```bash
docker build -t gmeet-pipeline .
docker run --env-file .env -p 9120:9120 gmeet-pipeline
```

## Environment Variables

All variables use the `GMEET_` prefix (pydantic-settings). The most common ones:

| Variable | Description | Default |
|----------|-------------|---------|
| `GMEET_RECALL_API_KEY` | Recall.ai API key | — |
| `GMEET_OPENROUTER_API_KEY` | OpenRouter API key for LLM | — |
| `GMEET_SERVICE_URL` | Your service's public URL (for webhooks) | — |
| `GMEET_TTS_BACKEND` | `elevenlabs` or `local` | `elevenlabs` |
| `GMEET_LLM_ROUTING` | `simple` or `voice_gateway` | `simple` |
| `GMEET_ELEVENLABS_API_KEY` | ElevenLabs API key (if using ElevenLabs TTS) | — |
| `GMEET_ELEVENLABS_VOICE_ID` | ElevenLabs voice ID | — |
| `GMEET_PORT` | Server port | `9120` |
| `GMEET_HERMES_HOME` | Base dir for memory/audio paths | `~/.hermes` |

Full list in `gmeet_pipeline/config.py`.

## Requirements

- Python 3.11+
- Recall.ai account
- OpenRouter account
- ElevenLabs account (if `TTS_BACKEND=elevenlabs`)
- Kokoro + RVC (if `TTS_BACKEND=local` on Apple Silicon)
