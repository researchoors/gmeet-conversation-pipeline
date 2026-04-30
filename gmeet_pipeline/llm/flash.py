"""
FlashLLM — single-model Gemini Flash agent with front-loaded context.

No routing tiers, no RAG retrieval, no EXPAND directives.
Context is baked into the system prompt once at call start via ContextBuilder.
During the call, conversation turns just append naturally.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

from .base import BaseLLM
from ..context_builder import ContextBuilder

logger = logging.getLogger("gmeet_pipeline.llm.flash")

# Tokens that signal the agent should stay silent
SILENT_TOKENS = frozenset({"SILENT", "NO_RESPONSE", "PASS", "SKIP"})

# Hank Bob persona — the constant part
_HANK_BOB_PERSONA = """\
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
- Use the context below to inform your responses — it contains real knowledge about the people and projects in this conversation

Multi-speaker rules:
- Multiple people may be in the call. Messages are prefixed with the speaker's name (e.g. "Ethan: ...")
- Address people by name when replying, especially if multiple people are talking
- If two people ask questions back-to-back, you can address both in one response
- Don't repeat someone's name if they're the only other person in the call"""


class FlashLLM(BaseLLM):
    """Single-model Gemini Flash with front-loaded Hermes context."""

    def __init__(
        self,
        api_key: str,
        model: str = "google/gemini-2.5-flash",
        service_url: str = "https://meet.model-optimizors.com",
        context_builder: ContextBuilder | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.service_url = service_url
        self.context_builder = context_builder or ContextBuilder()

    async def generate(
        self,
        conversation: list,
        message: str,
        bot_state: dict | None = None,
    ) -> Optional[str]:
        """Call the LLM with front-loaded context.

        Returns None if the agent should stay silent or on API failure.
        """
        # Build system prompt: persona + full context
        context = self.context_builder.build()
        if context:
            system_prompt = f"{_HANK_BOB_PERSONA}\n\n{context}"
        else:
            system_prompt = _HANK_BOB_PERSONA

        messages = [{"role": "system", "content": system_prompt}]
        # Recent conversation context (last 20 turns)
        for msg in conversation[-20:]:
            messages.append(msg)
        # The new user message
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
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": 300,
                        "temperature": 0.7,
                        "extra": {
                            "google": {
                                "thinking_config": {
                                    "thinking_budget": 0,
                                }
                            }
                        },
                    },
                )

                t1 = asyncio.get_event_loop().time()

                if resp.status_code != 200:
                    logger.error(
                        "LLM error: %s %s", resp.status_code, resp.text[:200]
                    )
                    return None

                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()

                # Log latency + usage
                usage = data.get("usage", {})
                logger.info(
                    "FlashLLM model=%s latency=%.0fms prompt=%s comp=%s",
                    self.model.split("/")[-1],
                    (t1 - t0) * 1000,
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                )

                # Check if agent chose to stay silent
                if content.upper() in SILENT_TOKENS:
                    return None

                return content

        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return None
