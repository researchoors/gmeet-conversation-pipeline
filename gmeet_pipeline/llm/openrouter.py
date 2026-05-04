"""
gmeet_pipeline.llm.openrouter — OpenRouter-backed LLM implementations.

Two concrete strategies:

* **SimpleOpenRouterLLM** — Straight single-model call, ported from
  ``meeting_agent.py`` lines 129–171.

* **VoiceGatewayLLM** — Memory-aware routing with RAG retrieval,
  query classification, and multi-model selection, ported from
  ``meeting_agent_rvc.py`` lines 402–486.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

import httpx

from .base import BaseLLM
from ..memory import MemorySnapshot

logger = logging.getLogger("gmeet_pipeline.llm.openrouter")

# ── Tokens that signal the agent should stay silent ──────────────────
SILENT_TOKENS = frozenset({"...", "SILENT", "NO_RESPONSE", "PASS", "SKIP"})

# ── Regex for EXPAND directives in LLM output ────────────────────────
EXPAND_RE = re.compile(r"EXPAND\[(\d+(?:\s*,\s*\d+)*)\]")

# ── Shared system prompt (Hank Bob persona) ─────────────────────────
_HANK_BOB_PROMPT = """\
You are Hank Bob, an AI research assistant for the researchoors community. You're joining a Google Meet call as a participant.

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
- If you don't know something, say so briefly and move on

Action-awareness policy:
- Listen for explicit tasks, decisions, and follow-up requests during the call.
- Briefly acknowledge that you captured actionable work, but do not claim you completed side effects unless a tool actually ran.
- Post-call Hermes automation will synthesize the transcript and route arbitrary authorized actions; Linear is only one possible target."""


# =====================================================================
# SimpleOpenRouterLLM
# =====================================================================

class SimpleOpenRouterLLM(BaseLLM):
    """Single-model OpenRouter call — the baseline Hank Bob agent.

    Ported from ``meeting_agent.py`` ``generate_response()``.
    """

    SYSTEM_PROMPT: str = _HANK_BOB_PROMPT

    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-sonnet-4",
        service_url: str = "https://your-domain.com",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.service_url = service_url

    async def generate(
        self,
        conversation: list,
        message: str,
        bot_state: dict | None = None,
    ) -> Optional[str]:
        """Call the LLM to generate a response.

        Returns ``None`` if Hank shouldn't respond or on API failure.
        """
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        # Recent conversation context (last 20 turns)
        for msg in conversation[-20:]:
            messages.append(msg)

        # The new user message
        messages.append({"role": "user", "content": message})

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": self.service_url,
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": 300,
                        "temperature": 0.7,
                    },
                )

                if resp.status_code != 200:
                    logger.error(
                        "LLM error: %s %s", resp.status_code, resp.text[:200]
                    )
                    return None

                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()

                # Check if Hank chose to stay silent
                if content.upper() in SILENT_TOKENS:
                    return None

                return content

        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return None


# =====================================================================
# VoiceGatewayLLM
# =====================================================================

class VoiceGatewayLLM(BaseLLM):
    """Memory-aware, multi-model OpenRouter agent with RAG retrieval.

    Ported from ``meeting_agent_rvc.py`` ``generate_response()`` /
    ``build_system_prompt()``.

    Flow per ``generate()`` call:
    1. Refresh memory snapshot if stale.
    2. Classify the query → ``fast`` / ``standard`` / ``deep``.
    3. RAG-retrieve context for ``standard`` and ``deep`` paths.
    4. Build a system prompt enriched with memory + RAG + expanded entries.
    5. Select model based on path.
    6. Call OpenRouter chat/completions.
    7. Parse ``EXPAND[n]`` directives, update *bot_state*, and strip them.
    """

    BASE_SYSTEM: str = _HANK_BOB_PROMPT

    def __init__(
        self,
        api_key: str,
        fast_model: str = "google/gemini-2.5-flash",
        standard_model: str = "openai/gpt-4.1-mini",
        deep_model: str = "anthropic/claude-sonnet-4",
        service_url: str = "https://meet.model-optimizors.com",
        memory_snapshot: MemorySnapshot | None = None,
    ) -> None:
        self.api_key = api_key
        self.fast_model = fast_model
        self.standard_model = standard_model
        self.deep_model = deep_model
        self.service_url = service_url
        self.memory = memory_snapshot

    # ── System prompt builder ────────────────────────────────────────

    def build_system_prompt(
        self,
        path: str,
        rag_entries: list[dict] | None = None,
        expanded_indices: list[int] | None = None,
    ) -> str:
        """Build system prompt with memory context based on routing *path*."""
        parts: list[str] = [self.BASE_SYSTEM]

        mem = self.memory
        if mem is None:
            return "\n".join(parts)

        # Always include snapshot summary
        parts.append(f"\nUser context: {mem.summary}")

        if path in ("standard", "deep"):
            # Include entry index so LLM can EXPAND specific entries
            parts.append(
                f"\nMemory entries available (use EXPAND[n] to request details):\n"
                f"{mem.index_text}"
            )

        # Add RAG-enriched entries
        if rag_entries:
            parts.append("\nRelevant memory:")
            for e in rag_entries:
                parts.append(f"  {e['text']}")

        # Add expanded entries from follow-up
        if expanded_indices is not None:
            expanded = [
                mem.entries[i] for i in expanded_indices if i < len(mem.entries)
            ]
            if expanded:
                parts.append("\nExpanded entries:")
                for e in expanded:
                    parts.append(f"  {e['text']}")

        return "\n".join(parts)

    # ── Core generate ────────────────────────────────────────────────

    async def generate(
        self,
        conversation: list,
        message: str,
        bot_state: dict | None = None,
    ) -> Optional[str]:
        """Call the LLM with memory-aware routing.

        Returns ``None`` if the agent should stay silent or on API failure.
        """
        mem = self.memory

        # 1. Refresh snapshot if stale
        if mem is not None:
            mem.refresh_if_stale()

        # 2. Classify query
        path = "fast"
        if mem is not None:
            path = mem.classify_query(message)

        # 3. Build RAG context
        rag_entries: list[dict] | None = None
        if mem is not None and path in ("standard", "deep"):
            rag_entries = mem.rag_retrieve(message, top_k=3)

        # 4. Get expanded entries from bot state (multi-turn)
        expanded: list[int] | None = None
        if bot_state and "expanded_entries" in bot_state:
            expanded = sorted(bot_state["expanded_entries"])

        # 5. Build system prompt
        system_prompt = self.build_system_prompt(
            path, rag_entries=rag_entries, expanded_indices=expanded
        )

        # 6. Select model
        if path == "fast":
            model = self.fast_model
        elif path == "standard":
            model = self.standard_model
        else:
            model = self.deep_model

        messages = [{"role": "system", "content": system_prompt}]
        for msg in conversation[-20:]:
            messages.append(msg)
        messages.append({"role": "user", "content": message})

        try:
            t0 = asyncio.get_event_loop().time()

            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": self.service_url,
                    },
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": 300,
                        "temperature": 0.7,
                    },
                )

                if resp.status_code != 200:
                    logger.error(
                        "LLM error: %s %s", resp.status_code, resp.text[:200]
                    )
                    return None

                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()

                # Log latency + usage
                t1 = asyncio.get_event_loop().time()
                usage = data.get("usage", {})
                prompt_tok = usage.get("prompt_tokens", 0)
                comp_tok = usage.get("completion_tokens", 0)
                logger.info(
                    "LLM path=%s model=%s latency=%.0fms prompt=%s comp=%s rag=%s",
                    path,
                    model.split("/")[-1],
                    (t1 - t0) * 1000,
                    prompt_tok,
                    comp_tok,
                    len(rag_entries) if rag_entries else 0,
                )

                # Check if agent chose to stay silent
                if content.upper() in SILENT_TOKENS:
                    return None

                # 7. Parse EXPAND directives and update bot state
                expand_indices = _parse_expands(content, mem)
                if expand_indices and bot_state is not None:
                    if "expanded_entries" not in bot_state:
                        bot_state["expanded_entries"] = set()
                    bot_state["expanded_entries"].update(expand_indices)
                    logger.info("EXPAND requested: entries %s", expand_indices)

                content = _strip_expands(content)
                return content

        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return None


# ── Module-level helpers (EXPAND parsing) ────────────────────────────


def _parse_expands(text: str, mem: MemorySnapshot | None) -> list[int]:
    """Parse ``EXPAND[n1,n2,…]`` directives, returning valid entry indices."""
    if mem is None:
        return []
    indices: set[int] = set()
    for match in EXPAND_RE.finditer(text):
        for num_str in match.group(1).split(","):
            try:
                idx = int(num_str.strip())
                if 0 <= idx < len(mem.entries):
                    indices.add(idx)
            except ValueError:
                continue
    return sorted(indices)


def _strip_expands(text: str) -> str:
    """Remove ``EXPAND[…]`` directives from *text*."""
    return EXPAND_RE.sub("", text).strip()
