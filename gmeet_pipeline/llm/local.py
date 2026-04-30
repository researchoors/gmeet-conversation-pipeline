"""
LocalLLM — in-process MLX inference on Apple Silicon.

Loads weights from HuggingFace on first call, keeps model in memory.
Supports streaming: yields sentence chunks as they're generated,
skipping the Qwen3 thinking block. This enables pipelined LLM→TTS
where the first sentence hits TTS while the LLM is still generating.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import AsyncGenerator, Optional

from .base import BaseLLM, SILENT_TOKENS
from ..context_builder import ContextBuilder

logger = logging.getLogger("gmeet_pipeline.llm.local")

# Sentence-end pattern: period, exclamation, question mark followed by space or end
_SENTENCE_END = re.compile(r'([.!?])\s')

# Hank Bob persona
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


def _chunk_sentences(text: str, buffer: str = "") -> tuple[list[str], str]:
    """Split text into complete sentences, returning chunks + leftover buffer.

    A sentence boundary is [.!?] followed by whitespace or end-of-string.
    Returns (list_of_complete_sentences, remaining_buffer).
    """
    combined = buffer + text
    chunks = []
    last_end = 0

    for m in _SENTENCE_END.finditer(combined):
        end = m.end()  # includes the space after punctuation
        sentence = combined[last_end:end].strip()
        if sentence:
            chunks.append(sentence)
        last_end = end

    remaining = combined[last_end:]
    return chunks, remaining


class LocalLLM(BaseLLM):
    """In-process MLX LLM with front-loaded Hermes context."""

    def __init__(
        self,
        model_id: str = "mlx-community/Qwen3.5-35B-A3B-4bit",
        max_tokens: int = 200,
        temperature: float = 0.7,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.context_builder = context_builder or ContextBuilder()

        # Lazy-loaded MLX model + tokenizer
        self._model = None
        self._tokenizer = None
        self._sampler = None
        self._load_lock = asyncio.Lock()

    def _ensure_loaded(self):
        """Load model + tokenizer from HuggingFace (first call only)."""
        if self._model is not None:
            return

        import mlx_lm
        from mlx_lm.sample_utils import make_sampler

        t0 = time.monotonic()
        logger.info("LocalLLM loading %s from HuggingFace...", self.model_id)
        self._model, self._tokenizer = mlx_lm.load(self.model_id)
        self._sampler = make_sampler(temp=self.temperature)
        t1 = time.monotonic()
        logger.info(
            "LocalLLM loaded %s in %.1fs", self.model_id.split("/")[-1], t1 - t0
        )

    def _generate_sync(self, prompt: str) -> str:
        """Synchronous generation — call from thread pool."""
        import mlx_lm

        self._ensure_loaded()

        t0 = time.monotonic()
        output = mlx_lm.generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            sampler=self._sampler,
            verbose=False,
        )
        t1 = time.monotonic()

        # Trim: mlx_lm.generate returns the full string including prompt
        response = output[len(prompt):].strip() if output.startswith(prompt) else output.strip()

        # Remove Qwen3 thinking block
        response = self._strip_thinking(response)

        # Remove trailing <|im_end|> if present
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0].strip()

        latency_ms = (t1 - t0) * 1000
        logger.info(
            "LocalLLM model=%s latency=%.0fms response=%s",
            self.model_id.split("/")[-1],
            latency_ms,
            response[:80],
        )
        return response

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove Qwen3 <think>...</think> blocks from text."""
        return re.sub(r'<think\b[^>]*>.*?</think\s*>', '', text, flags=re.DOTALL).strip()

    async def generate(
        self,
        conversation: list,
        message: str,
        bot_state: dict | None = None,
    ) -> Optional[str]:
        """One-shot generation (non-streaming)."""
        chunks = []
        async for chunk in self.generate_stream(conversation, message, bot_state):
            chunks.append(chunk)
        full = " ".join(chunks)
        if not full or full.upper() in SILENT_TOKENS:
            return None
        return full

    async def generate_stream(
        self,
        conversation: list,
        message: str,
        bot_state: dict | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream response as sentence chunks, skipping the thinking block.

        Yields complete sentences as they're produced by the LLM.
        The thinking block (Qwen3 <think>...</think>) is silently consumed.
        Once thinking is done, tokens are buffered into sentences and
        yielded immediately on sentence boundaries.
        """
        # Ensure model is loaded
        async with self._load_lock:
            if self._model is None:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._ensure_loaded)

        # Build system prompt: persona + full context
        context = self.context_builder.build()
        if context:
            system_prompt = f"{_HANK_BOB_PERSONA}\n\n{context}"
        else:
            system_prompt = _HANK_BOB_PERSONA

        # Build messages for chat template
        messages = [{"role": "system", "content": system_prompt}]
        for msg in conversation[-20:]:
            messages.append(msg)
        messages.append({"role": "user", "content": message})

        # Apply chat template
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # We need to bridge synchronous stream_generate to async.
        # Use a queue: the sync thread pushes tokens, the async side consumes.
        token_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _stream_sync():
            """Run stream_generate in a thread, pushing tokens to the queue."""
            import mlx_lm
            self._ensure_loaded()
            try:
                for resp in mlx_lm.stream_generate(
                    self._model,
                    self._tokenizer,
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    sampler=self._sampler,
                ):
                    token_queue.put_nowait(resp.text)
            except Exception as exc:
                logger.error("stream_generate error: %s", exc)
            finally:
                token_queue.put_nowait(None)  # sentinel: generation done

        loop.run_in_executor(None, _stream_sync)

        # Consume tokens from the queue
        in_think = False
        think_depth = 0
        response_buffer = ""
        full_response = ""
        t0 = time.monotonic()
        first_chunk_time = None

        while True:
            token = await token_queue.get()
            if token is None:
                break

            # Track thinking block state machine
            if "<think" in token and not in_think:
                in_think = True
                # Token might have <think plus some content
                after = token.split(">", 1)
                if len(after) > 1:
                    think_depth = 1
                continue

            if in_think:
                think_depth += token.count("<think")
                think_depth -= token.count("</think>")
                if think_depth <= 0:
                    in_think = False
                    # Anything after </think> in this token is response text
                    after = token.split("</think>", 1)
                    if len(after) > 1 and after[1].strip():
                        response_buffer += after[1]
                continue

            # Normal response token
            response_buffer += token

            # Try to chunk into sentences
            chunks, response_buffer = _chunk_sentences(response_buffer)
            for chunk in chunks:
                if first_chunk_time is None:
                    first_chunk_time = time.monotonic()
                    logger.info(
                        "LocalLLM first-chunk latency=%.0fms",
                        (first_chunk_time - t0) * 1000,
                    )
                full_response += " " + chunk
                yield chunk

        # Flush remaining buffer (last partial sentence)
        if response_buffer.strip():
            chunk = response_buffer.strip()
            full_response += " " + chunk
            yield chunk

        # Remove trailing <|im_end|> from last chunk check
        # (already handled by sentence chunker not matching it)

        total_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "LocalLLM stream complete: model=%s total=%.0fms response=%s",
            self.model_id.split("/")[-1],
            total_ms,
            full_response.strip()[:80],
        )

        # If the entire response is a silent token, we've already yielded it.
        # The caller (webhook handler) should check for silent tokens.
