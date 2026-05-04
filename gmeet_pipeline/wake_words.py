"""Wake and silence command classification for Hank Bob.

This module intentionally starts with transcript-text matching. Recall already
provides low-latency STT text, so text matching is the lowest-risk first slice;
audio-level keyword spotting can be added later behind the same interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional


@dataclass(frozen=True)
class WakeCommand:
    """A classified wake/silence command."""

    kind: Optional[str]
    confidence: float = 0.0
    matched_phrase: str = ""


_BOT_ALIASES = (
    "hank bob",
    "hey hank bob",
    "hankbot",
    "hank bot",
    "hank bop",
    "hang bob",
    "hank barb",
    "hank bab",
)
_SILENCE_PHRASES = (
    "shut up",
    "quiet",
    "be quiet",
    "not now",
    "stop talking",
    "mute",
    "silence",
)
_WAKE_PHRASES = (
    "wake up",
    "come back",
    "listen",
    "you there",
)


def normalize_text(text: str) -> str:
    """Normalize transcript text for conservative phrase matching."""

    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _contains_bot_name(text: str) -> tuple[bool, float, str]:
    for alias in _BOT_ALIASES:
        if alias in text:
            return True, 1.0, alias

    tokens = text.split()
    best_score = 0.0
    best_span = ""
    for size in (1, 2, 3):
        for idx in range(0, max(0, len(tokens) - size + 1)):
            span = " ".join(tokens[idx : idx + size])
            for alias in _BOT_ALIASES:
                score = SequenceMatcher(None, span, alias).ratio()
                if score > best_score:
                    best_score = score
                    best_span = span
    return best_score >= 0.78, best_score, best_span


def classify_wake_command(text: str) -> WakeCommand:
    """Classify text as a Hank Bob wake/silence command, if any."""

    normalized = normalize_text(text)
    if not normalized:
        return WakeCommand(kind=None)

    has_name, name_score, matched = _contains_bot_name(normalized)
    if not has_name:
        return WakeCommand(kind=None)

    if any(phrase in normalized for phrase in _SILENCE_PHRASES):
        return WakeCommand(kind="silence", confidence=min(1.0, name_score), matched_phrase=matched)

    if normalized in _BOT_ALIASES or (normalized.startswith("hey ") and len(normalized.split()) <= 4):
        return WakeCommand(kind="wake", confidence=min(1.0, name_score), matched_phrase=matched)

    if any(phrase in normalized for phrase in _WAKE_PHRASES):
        return WakeCommand(kind="wake", confidence=min(1.0, name_score), matched_phrase=matched)

    # Mentioning the bot name alone is a wake/addressing command, but avoid
    # treating arbitrary long utterances with a name mention as wake commands.
    if len(normalized.split()) <= 4:
        return WakeCommand(kind="wake", confidence=min(1.0, name_score), matched_phrase=matched)

    return WakeCommand(kind=None)
