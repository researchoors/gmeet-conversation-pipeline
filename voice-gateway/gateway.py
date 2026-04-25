"""
Voice Gateway — Lightweight LLM router with referencable memory snapshots.

Architecture:
  Snapshot (~80 tok, always injected) → entry index refs [0]-[N]
  RAG enrichment (top-k § entries by keyword overlap)
  EXPAND protocol (LLM emits EXPAND[n,m] → gateway injects full entries)
  Model routing: fast (Flash) / standard (4.1-mini) / deep (full Hermes)
"""

import json
import re
import time
import logging
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("voice-gateway")

# ─── Config ───────────────────────────────────────────────────────────────────

HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
MEMORY_DIR = HERMES_HOME / "memories"
MEMORY_FILE = MEMORY_DIR / "MEMORY.md"
USER_FILE = MEMORY_DIR / "USER.md"

# OpenRouter auth — try credential pool first, then .env
def _get_openrouter_key() -> str:
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
    # Fallback to .env
    env_path = HERMES_HOME / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("No OpenRouter API key found")

OPENROUTER_KEY = _get_openrouter_key()
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HERMES_ENDPOINT = "http://localhost:8642/v1/chat/completions"

# Model assignments
FAST_MODEL = "google/gemini-2.5-flash"
STANDARD_MODEL = "openai/gpt-4.1-mini"
DEEP_MODEL = "hermes-agent"  # routes to full Hermes endpoint

# ─── Snapshot Builder ─────────────────────────────────────────────────────────

ENTRY_DELIMITER = "§"

@dataclass
class MemoryEntry:
    index: int
    source: str  # "memory" or "user"
    text: str
    summary: str = ""  # auto-generated compressed version

@dataclass
class Snapshot:
    """Pre-computed snapshot: compressed summary + entry index."""
    summary: str = ""  # ~80 token compressed overview
    entries: list = field(default_factory=list)
    index_text: str = ""  # one-line-per-entry index for system prompt
    raw_built_at: float = 0.0

    def refresh_if_stale(self, max_age_seconds: int = 300):
        """Rebuild snapshot if memory files changed."""
        mem_mtime = MEMORY_FILE.stat().st_mtime if MEMORY_FILE.exists() else 0
        usr_mtime = USER_FILE.stat().st_mtime if USER_FILE.exists() else 0
        latest = max(mem_mtime, usr_mtime)
        if latest > self.raw_built_at:
            self._build()

    def _build(self):
        """Parse MEMORY.md + USER.md into entries + compressed snapshot."""
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
                    idx = len(entries)
                    entries.append(MemoryEntry(
                        index=idx,
                        source=source,
                        text=chunk,
                        summary=self._compress_entry(chunk, idx),
                    ))

        self.entries = entries
        self.raw_built_at = time.time()

        # Build index text — one line per entry, shows what's at each ref
        index_lines = []
        for e in entries:
            # First ~60 chars as preview
            preview = e.text[:80].replace("\n", " ").strip()
            if len(e.text) > 80:
                preview += "..."
            index_lines.append(f"[{e.index}] {preview}")
        self.index_text = "\n".join(index_lines)

        # Build compressed summary — dense overview of user identity
        user_entries = [e for e in entries if e.source == "user"]
        mem_entries = [e for e in entries if e.source == "memory"]

        # Extract key facts for ultra-compressed summary
        summary_parts = []
        for e in user_entries[:3]:  # top 3 user entries
            first_sentence = e.text.split(".")[0].strip()
            summary_parts.append(first_sentence)
        if mem_entries:
            # Pick first sentence of first 2 memory entries
            for e in mem_entries[:2]:
                first_sentence = e.text.split(".")[0].strip()
                summary_parts.append(first_sentence)

        self.summary = " | ".join(summary_parts)

    @staticmethod
    def _compress_entry(text: str, idx: int) -> str:
        """Create a short label for an entry."""
        first = text.split(".")[0].strip()
        if len(first) > 60:
            first = first[:57] + "..."
        return f"[{idx}] {first}"


# Global snapshot — auto-refreshes
snapshot = Snapshot()
snapshot._build()


# ─── RAG Layer ────────────────────────────────────────────────────────────────

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

def rag_retrieve(query: str, top_k: int = 3) -> list[MemoryEntry]:
    """Keyword-overlap scoring against § entries. Returns top-k matches."""
    query_terms = {w.lower() for w in re.findall(r'\w+', query) if w.lower() not in STOP_WORDS}
    if not query_terms:
        return []

    scored = []
    for entry in snapshot.entries:
        entry_terms = {w.lower() for w in re.findall(r'\w+', entry.text) if w.lower() not in STOP_WORDS}
        overlap = len(query_terms & entry_terms)
        if overlap > 0:
            scored.append((overlap, entry))

    scored.sort(key=lambda x: -x[0])
    return [entry for _, entry in scored[:top_k]]


# ─── EXPAND Protocol ──────────────────────────────────────────────────────────

EXPAND_RE = re.compile(r'EXPAND\[(\d+(?:\s*,\s*\d+)*)\]')

def parse_expands(text: str) -> list[int]:
    """Find all EXPAND[n,m,...] directives in LLM output. Returns entry indices."""
    indices = set()
    for match in EXPAND_RE.finditer(text):
        for num_str in match.group(1).split(","):
            try:
                idx = int(num_str.strip())
                if 0 <= idx < len(snapshot.entries):
                    indices.add(idx)
            except ValueError:
                continue
    return sorted(indices)

def strip_expands(text: str) -> str:
    """Remove EXPAND[] directives from response text."""
    return EXPAND_RE.sub("", text).strip()


# ─── Model Router ─────────────────────────────────────────────────────────────

# Simple heuristic classification
TOOL_KEYWORDS = {"github", "repo", "issue", "pr ", "pull request", "commit", "deploy",
                 "install", "server", "docker", "nomad", "latest issues"}
MEMORY_KEYWORDS = {"remember", "recall", "know about", "what is", "who is",
                   "tell me about", "explain", "describe", "about the",
                   "darkbloom", "d-inference", "swiftlm", "dflash", "researchoors",
                   "layr", "benchmark", "project", "setup", "configured"}
SIMPLE_PATTERNS = [
    re.compile(r'^(hi|hey|hello|yo|sup|what\'s up|thanks|ok|yes|no|sure|cool|nice|got it)', re.I),
    re.compile(r'^\d+\s*[\+\-\*\/]\s*\d+'),  # math
    re.compile(r'^(what is|what\'s|define|spell|translate|convert)\s+\w', re.I),
]

def classify_query(query: str) -> str:
    """Returns: 'fast', 'standard', or 'deep'."""
    q_lower = query.lower().strip()

    # Check for simple patterns
    for pat in SIMPLE_PATTERNS:
        if pat.match(q_lower):
            return "fast"

    # Check for tool-needing keywords
    query_words = set(re.findall(r'\w+', q_lower))
    if query_words & TOOL_KEYWORDS:
        return "deep"

    # Check for memory-relevant keywords
    for kw in MEMORY_KEYWORDS:
        if kw in q_lower:
            return "standard"

    # Check if RAG has hits — if yes, standard path
    if rag_retrieve(query, top_k=1):
        return "standard"

    # Default to fast
    return "fast"


# ─── System Prompt Builder ────────────────────────────────────────────────────

BASE_SYSTEM = "You are a helpful voice assistant for Ethan. Be concise — speak in short, natural sentences. You have access to memory entries referenced by index."

def build_system_prompt(
    path: str,
    rag_entries: Optional[list] = None,
    expanded_indices: Optional[list] = None,
) -> str:
    """Build system prompt based on routing path and context."""
    parts = [BASE_SYSTEM]

    # Always include snapshot summary
    parts.append(f"\nUser context: {snapshot.summary}")

    if path in ("standard", "deep"):
        # Include entry index
        parts.append(f"\nMemory entries available (use EXPAND[n] to request details):\n{snapshot.index_text}")

    if path == "fast":
        # Minimal — just the summary, no index
        pass

    # Add RAG-enriched entries
    if rag_entries:
        parts.append("\nRelevant memory:")
        for e in rag_entries:
            parts.append(f"  {e.text}")

    # Add expanded entries from follow-up
    if expanded_indices is not None:
        expanded = [snapshot.entries[i] for i in expanded_indices if i < len(snapshot.entries)]
        if expanded:
            parts.append("\nExpanded entries:")
            for e in expanded:
                parts.append(f"  {e.text}")

    return "\n".join(parts)


# ─── LLM Callers ──────────────────────────────────────────────────────────────

async def call_openrouter_stream(
    messages: list[dict],
    model: str,
    max_tokens: int = 512,
):
    """Stream from OpenRouter. Yields (event_type, data) tuples.
    event_type: "token" | "usage" | "done"
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        async with client.stream(
            "POST",
            OPENROUTER_URL,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENROUTER_KEY}",
            },
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    yield ("done", None)
                    return
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Usage
                if chunk.get("usage") and chunk["usage"].get("prompt_tokens"):
                    yield ("usage", chunk["usage"])

                # Content delta
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if "content" in delta and delta["content"]:
                    yield ("token", delta["content"])


async def call_hermes_stream(
    messages: list[dict],
):
    """Stream from full Hermes endpoint. Yields (event_type, data)."""
    payload = {
        "model": "hermes-agent",
        "messages": messages,
        "stream": True,
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        async with client.stream(
            "POST",
            HERMES_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    yield ("done", None)
                    return
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if "content" in delta and delta["content"]:
                    yield ("token", delta["content"])


# ─── Conversation State ───────────────────────────────────────────────────────

# Simple in-memory session store (conversation_id → state)
sessions: dict[str, dict] = {}

def get_session(conv_id: str) -> dict:
    if conv_id not in sessions:
        sessions[conv_id] = {
            "history": [],          # list of {role, content} messages
            "expanded": set(),      # entry indices already expanded
            "last_path": None,
        }
    return sessions[conv_id]


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(title="Voice Gateway", version="0.1.0")


@app.get("/health")
async def health():
    snapshot.refresh_if_stale()
    return {
        "status": "ok",
        "entries": len(snapshot.entries),
        "snapshot_age": round(time.time() - snapshot.raw_built_at, 1),
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint with memory-aware routing."""
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    conv_id = request.headers.get("X-Conversation-Id", "default")
    force_path = request.headers.get("X-Path")  # optional: "fast"|"standard"|"deep"
    max_tokens = body.get("max_tokens", 512)

    # Refresh snapshot if stale
    snapshot.refresh_if_stale()

    # Get user query (last user message)
    user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_msg = m.get("content", "")
            break

    # Classify routing
    path = force_path or classify_query(user_msg)

    # Get session state
    session = get_session(conv_id)

    # Build RAG context
    rag_entries = None
    if path in ("standard", "deep"):
        rag_entries = rag_retrieve(user_msg, top_k=3)

    # Build system prompt
    system_prompt = build_system_prompt(
        path=path,
        rag_entries=rag_entries,
        expanded_indices=sorted(session["expanded"]) if session["expanded"] else None,
    )

    # Assemble messages for LLM
    llm_messages = [{"role": "system", "content": system_prompt}] + messages

    # Track timing
    t_start = time.time()
    first_token_time = None
    full_text = ""
    prompt_tokens = 0
    completion_tokens = 0

    if path == "deep":
        # Route to full Hermes endpoint
        streamer = call_hermes_stream(llm_messages)
    else:
        model = FAST_MODEL if path == "fast" else STANDARD_MODEL
        streamer = call_openrouter_stream(llm_messages, model=model, max_tokens=max_tokens)

    if not stream:
        # Non-streaming: collect full response
        async for event_type, data in streamer:
            if event_type == "token":
                full_text += data
            elif event_type == "usage":
                prompt_tokens = data.get("prompt_tokens", 0)
                completion_tokens = data.get("completion_tokens", 0)
            elif event_type == "done":
                break

        elapsed = time.time() - t_start

        # Parse EXPAND directives
        expand_indices = parse_expands(full_text)
        if expand_indices:
            session["expanded"].update(expand_indices)
            full_text = strip_expands(full_text)

        # Store in session history
        session["history"].append({"role": "user", "content": user_msg})
        session["history"].append({"role": "assistant", "content": full_text})
        session["last_path"] = path

        return JSONResponse({
            "id": f"vg-{int(t_start)}",
            "object": "chat.completion",
            "created": int(t_start),
            "model": path,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": full_text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens or len(full_text.split()),
                "total_tokens": prompt_tokens + (completion_tokens or len(full_text.split())),
            },
            "meta": {
                "path": path,
                "latency_ms": round(elapsed * 1000),
                "rag_entries": len(rag_entries) if rag_entries else 0,
                "expanded": expand_indices if expand_indices else None,
                "snapshot_entries": len(snapshot.entries),
            },
        })

    else:
        # Streaming response — SSE format
        async def generate():
            nonlocal first_token_time, full_text, prompt_tokens, completion_tokens
            async for event_type, data in streamer:
                if event_type == "token":
                    if first_token_time is None:
                        first_token_time = time.time() - t_start
                    full_text += data
                    chunk = {
                        "id": f"vg-{int(t_start)}",
                        "object": "chat.completion.chunk",
                        "created": int(t_start),
                        "model": path,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": data},
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                elif event_type == "usage":
                    prompt_tokens = data.get("prompt_tokens", 0)
                    completion_tokens = data.get("completion_tokens", 0)

                elif event_type == "done":
                    # Parse EXPAND directives
                    expand_indices = parse_expands(full_text)
                    if expand_indices:
                        # Store expanded indices for next turn
                        session["expanded"].update(expand_indices)

                    # Store in session
                    session["history"].append({"role": "user", "content": user_msg})
                    session["history"].append({"role": "assistant", "content": full_text})
                    session["last_path"] = path

                    # Final chunk
                    final = {
                        "id": f"vg-{int(t_start)}",
                        "object": "chat.completion.chunk",
                        "created": int(t_start),
                        "model": path,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens or len(full_text.split()),
                            "total_tokens": prompt_tokens + (completion_tokens or len(full_text.split())),
                        },
                        "meta": {
                            "path": path,
                            "ttft_ms": round(first_token_time * 1000) if first_token_time else None,
                            "rag_entries": len(rag_entries) if rag_entries else 0,
                            "expanded": expand_indices if expand_indices else None,
                        },
                    }
                    yield f"data: {json.dumps(final)}\n\n"
                    yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "voice-fast", "object": "model", "owned_by": "voice-gateway"},
            {"id": "voice-standard", "object": "model", "owned_by": "voice-gateway"},
            {"id": "voice-deep", "object": "model", "owned_by": "voice-gateway"},
        ],
    }


@app.get("/v1/snapshot")
async def get_snapshot():
    """Debug endpoint — view current snapshot state."""
    snapshot.refresh_if_stale()
    return {
        "summary": snapshot.summary,
        "entry_count": len(snapshot.entries),
        "index": snapshot.index_text,
        "entries": [
            {"index": e.index, "source": e.source, "text": e.text[:200]}
            for e in snapshot.entries
        ],
        "sessions": len(sessions),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("VOICE_GATEWAY_PORT", 8643))
    logger.info(f"Starting Voice Gateway on :{port}")
    logger.info(f"Memory entries loaded: {len(snapshot.entries)}")
    logger.info(f"Fast model: {FAST_MODEL}")
    logger.info(f"Standard model: {STANDARD_MODEL}")
    logger.info(f"Deep endpoint: {HERMES_ENDPOINT}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
