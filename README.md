# gmeet-conversation-pipeline

Google Meet conversation pipeline powered by [Recall.ai](https://recall.ai). Joins Meet calls, captures real-time transcripts, generates AI summaries, and produces spoken recaps via ElevenLabs TTS.

## Architecture

```
Google Meet → Recall.ai bot → Transcript webhook → LLM summary → ElevenLabs TTS → Audio playback
```

## Features

- **Auto-join**: Connects to Google Meet via Recall.ai with configurable bot settings
- **Real-time transcription**: Receives transcript webhooks from Recall.ai as the call progresses
- **AI summarization**: Uses OpenRouter (Claude Sonnet) to generate structured meeting summaries
- **Spoken recaps**: Converts summaries to natural speech via ElevenLabs TTS
- **Audio serving**: Serves generated audio for playback through a web interface

## Setup

1. Copy `.env.example` to `.env` and fill in your credentials
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python meeting_agent.py`

## Environment Variables

| Variable | Description |
|----------|-------------|
| `RECALL_API_KEY` | Recall.ai API key |
| `ELEVENLABS_API_KEY` | ElevenLabs API key |
| `ELEVENLABS_VOICE_ID` | ElevenLabs voice ID for TTS |
| `OPENROUTER_API_KEY` | OpenRouter API key for LLM |
| `SERVICE_URL` | Your service's public URL (for webhooks) |

## Requirements

- Python 3.9+
- Recall.ai account
- ElevenLabs account
- OpenRouter account
