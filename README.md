# gmeet-conversation-pipeline

Google Meet conversation pipeline powered by [Recall.ai](https://recall.ai). Joins Meet calls, captures real-time transcripts, generates AI summaries, and produces spoken recaps via TTS.

## Architecture

```
Google Meet → Recall.ai bot → Transcript webhook → LLM → TTS → Audio playback
```

### Modular design

The pipeline is organized as a Python package with pluggable backends:

| Layer | Abstraction | Implementations |
|-------|-------------|-----------------|
| Transport | `BaseTransport` | `RecallTransport` (webhook + WebSocket) |
| LLM | `BaseLLM` | `OpenRouterLLM` (simple / voice-gateway routing) |
| TTS | `BaseTTS` | `ElevenLabsTTS` (PCM streaming), `LocalTTS` (Kokoro + RVC on Apple Silicon) |

All configuration is environment-driven via `gmeet_pipeline.config.GmeetSettings` (pydantic-settings).

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

| Variable | Description |
|----------|-------------|
| `RECALL_API_KEY` | Recall.ai API key |
| `OPENROUTER_API_KEY` | OpenRouter API key for LLM |
| `SERVICE_URL` | Your service's public URL (for webhooks) |
| `TTS_BACKEND` | `elevenlabs` or `local` (default: `elevenlabs`) |
| `ELEVENLABS_API_KEY` | ElevenLabs API key (required if `TTS_BACKEND=elevenlabs`) |
| `ELEVENLABS_VOICE_ID` | ElevenLabs voice ID |

## Requirements

- Python 3.11+
- Recall.ai account
- OpenRouter account
- ElevenLabs account (if using ElevenLabs TTS)
- Kokoro + RVC (if using local TTS on Apple Silicon)
