# Voice Gateway

Lightweight LLM router with referencable memory snapshots for sub-second voice agent responses.

## Architecture

Three-layer memory access:
1. **Snapshot** (~80 tok, always injected) — compressed summary of all memory entries with index refs `[0]-[N]`
2. **EXPAND protocol** — LLM emits `EXPAND[3,7]` when it needs detail on specific entries
3. **RAG enrichment** (~200 tok) — keyword overlap scoring on §-delimited entries, top-k appended

Three model paths:
| Path | Model | Prompt | TTFT | Total | Use case |
|---|---|---|---|---|---|
| ⚡ Fast | Gemini 2.5 Flash | snapshot only | ~0.8s | ~1.5s | greetings, simple Q |
| 🔄 Standard | GPT-4.1 Mini | snapshot + RAG | ~0.8s | ~2.5s | memory recall |
| 🧠 Deep | Full Hermes Agent | 8.5K+ tools | ~6s | 15-50s | GitHub, complex (async) |

## Quick Start

```bash
# Install deps
pip install fastapi uvicorn httpx

# Set env vars
export OPENROUTER_API_KEY=sk-or-...

# Run
python gateway.py
# → http://localhost:8643
```

## Endpoints

| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible chat (streaming + non-streaming) |
| `GET /v1/models` | List available models |
| `GET /health` | Health check + entry count |
| `GET /v1/snapshot` | Debug — view current snapshot state |

## Request Format

Standard OpenAI Chat Completions format:

```json
{
  "model": "voice-gateway",
  "messages": [{"role": "user", "content": "What do you know about DarkBloom?"}],
  "stream": true
}
```

### Custom Headers

| Header | Description |
|---|---|
| `X-Conversation-Id` | Session ID for multi-turn (EXPAND state persists) |
| `X-Path` | Force routing: `fast`, `standard`, or `deep` |

## Response Meta

Every response includes a `meta` object:

```json
{
  "path": "standard",
  "latency_ms": 2290,
  "rag_entries": 2,
  "expanded": [3, 7],
  "snapshot_entries": 17
}
```

## Memory Format

Reads from `~/.hermes/memories/MEMORY.md` and `USER.md`. Entries are `§`-delimited:

```
DarkBloom = decentralized inference (consumers↔Apple Silicon providers)...
§
GitHub auth: hankbob ghp_ PAT has full repo scope...
§
SwiftLM ~/benchmarks/SwiftLM (feat/add-dflash)...
```

Each entry gets an index `[0]`, `[1]`, etc. The LLM sees the index in its system prompt and can request full entry text via `EXPAND[n]`.

## Configuration

| Env Var | Default | Description |
|---|---|---|
| `VOICE_GATEWAY_PORT` | `8643` | Server port |
| `HERMES_HOME` | `~/.hermes` | Hermes config directory |

Model assignments are configured in `gateway.py` constants:
- `FAST_MODEL` — simple queries (default: `google/gemini-2.5-flash`)
- `STANDARD_MODEL` — memory recall (default: `openai/gpt-4.1-mini`)
- `DEEP_MODEL` — routes to full Hermes endpoint at `localhost:8642`

## Benchmarks

M3 Ultra, April 2026:

| Query | Path | TTFT | Total |
|---|---|---|---|
| Hey what's up? | fast | 1.3s | 1.5s |
| What is 2+2? | fast | 0.8s | 1.0s |
| DarkBloom? | standard | 1.3s | 2.5s |
| SwiftLM benchmarks? | standard | 1.0s | 2.2s |
| Who is Ethan? | standard | 0.8s | 1.6s |

## License

MIT
