# AGENTS.md

Guide for AI agents working on this codebase.

## Project Overview

Google Meet conversation agent. A bot joins calls via Recall.ai, receives real-time transcripts through webhooks, generates responses via LLM (OpenRouter), and speaks them back through TTS — all in a streaming pipeline.

## Architecture

```
Google Meet → Recall.ai bot → webhook events → PipelineServer
                                                  ↓
                                           WebhookHandler
                                           ├→ LLM (OpenRouter)
                                           └→ TTS (Kokoro or ElevenLabs)
                                                  ↓
                                           audio_queue → agent page (in bot's Chrome) → meeting audio
```

**Key data flow**: Recall sends `transcript.data` webhooks → `WebhookHandler._process_and_respond()` fires as `asyncio.create_task` → LLM generates response → TTS produces WAV → appended to `audio_queue` → agent page polls `/api/audio-queue` and plays audio via Web Audio API.

## Module Map

| Module | What it does |
|--------|-------------|
| `main.py` | Composition root — loads config, wires components via DI, starts uvicorn |
| `config.py` | Pydantic-settings (`Gmeet_*` env vars). Single source of truth for all config |
| `server.py` | FastAPI routes. Thin layer — delegates all logic to injected components |
| `webhook.py` | Recall webhook handler — orchestrates transcript → LLM → TTS pipeline |
| `state.py` | `BotRegistry` + `BotSession` — per-bot conversation state, locks, transcript |
| `agent_page.py` | HTML templates served as the bot's camera output page |
| `tts/local.py` | Kokoro KPipeline → optional RVC → WAV. Lazy-loads heavy ML deps |
| `tts/elevenlabs.py` | WebSocket PCM streaming with REST MP3 fallback |
| `llm/openrouter.py` | SimpleLLM (single model) and VoiceGatewayLLM (memory-aware multi-model routing) |
| `transports/recall.py` | Recall.ai API client — create/leave bots, configure webhooks |
| `memory.py` | Memory snapshot loader for voice gateway RAG |
| `ws_manager.py` | WebSocket connection manager for ElevenLabs PCM streaming |

## Critical Patterns & Pitfalls

### Mutable default arguments
Python falsy-trap: `self.audio_queue = audio_queue or {}` is WRONG because `{}` is falsy — it creates a new dict instead of using the shared reference. Always use `if x is not None else {}` for mutable defaults passed via DI.

### asyncio.Lock in dataclasses
`BotSession.respond_lock` is an `asyncio.Lock` created via `default_factory`. It must only be used within the running event loop. Don't create `BotSession` outside an async context.

### TTS lazy initialization
`LocalTTS._ensure_init()` loads Kokoro (~5s first time) and RVC. It silently catches `ImportError` and sets `_kokoro_pipeline = None`. If Kokoro isn't installed, TTS returns `None` with only a log message — no exception propagates. Check logs if audio is missing.

### Thread pool for TTS
`LocalTTS.generate()` runs `_generate_sync` via `loop.run_in_executor(None, ...)` to avoid blocking the event loop. First call loads the model (~5s), subsequent calls ~300ms.

### espeak-ng on macOS
LocalTTS requires `espeak-ng` at runtime. On macOS it loads from `/opt/homebrew/lib/libespeak-ng.dylib` and sets `ESPEAK_DATA_PATH`. This is hardcoded in `_ensure_init()` — if the path changes, TTS will silently fail.

### Recall bot output_media
The bot's camera feed loads the agent page via `output_media.camera.config.url` pointing to `SERVICE_URL`. This is how audio gets played in the meeting — the agent page polls `/api/audio-queue` and uses `AudioContext.decodeAudioData()` to play WAVs.

### No audio_queue clearing
The `/api/audio-queue` endpoint returns all items without removing them. The agent page tracks `lastAudioCount` client-side to avoid replaying. Don't add clearing logic without updating the agent page JS.

## Running

```bash
# Local dev
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install kokoro espeakng-loader spacy  # for local TTS
python -m gmeet_pipeline.main

# Tests
python -m pytest tests/ -v

# Docker
docker build -t gmeet-pipeline .
```

Server runs on port 9120. Needs a public URL (Cloudflare tunnel, etc.) for Recall webhooks.

## Configuration

All config via `GMEET_*` env vars or `.env` file. See `.env.example`.

Key selectors:
- `GMEET_TTS_BACKEND=local|elevenlabs` (default: `local`)
- `GMEET_LLM_ROUTING=simple|voice_gateway` (default: `simple`)

The pipeline reads from `~/.hermes/.env` via `_try_dotenv()` if no local `.env` exists.

## Testing

- 171 tests, run with `python -m pytest tests/ -v`
- CI: Python 3.11, ubuntu-latest (`.github/workflows/ci.yml`)
- `conftest.py` provides `mock_settings` and `mock_registry` fixtures
- Tests use `unittest.mock.MagicMock` for LLM/TTS/transport — no real API calls
- When changing defaults in `config.py`, update corresponding assertions in `tests/test_config.py`

## PR Conventions

- Main branch is protected — all changes via PR
- Branch naming: `fix/`, `feat/`, `refactor/`, `docs/`
- Keep commits focused; one logical change per commit
- No secrets or API keys in commits (validated in CI)
