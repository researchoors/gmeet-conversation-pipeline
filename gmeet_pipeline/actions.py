"""Generic action candidate extraction for meeting transcripts."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Literal

from .wake_words import normalize_text

Target = Literal["linear", "github", "terminal", "file", "browser", "hermes", "unknown"]
Intent = Literal[
    "create_issue",
    "update_issue",
    "comment_issue",
    "spawn_session",
    "run_command",
    "edit_file",
    "research",
    "send_message",
    "unknown",
]
Status = Literal["candidate", "confirmed", "needs_human_review", "executed", "skipped"]


@dataclass
class ActionCandidate:
    """A possible post-call action with provenance."""

    id: str
    speaker: str
    quote: str
    timestamp: str
    target: Target
    intent: Intent
    confidence: float
    refs: list[str] = field(default_factory=list)
    status: Status = "candidate"

    def to_dict(self) -> dict:
        return asdict(self)


def _candidate_id(speaker: str, text: str, target: str, intent: str, refs: list[str]) -> str:
    raw = "|".join([speaker, normalize_text(text), target, intent, ",".join(refs)])
    return "act-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _make_candidate(
    *, speaker: str, text: str, timestamp: str, target: Target, intent: Intent,
    confidence: float, refs: list[str] | None = None,
) -> ActionCandidate:
    refs = refs or []
    return ActionCandidate(
        id=_candidate_id(speaker, text, target, intent, refs),
        speaker=speaker,
        quote=text,
        timestamp=timestamp,
        target=target,
        intent=intent,
        confidence=confidence,
        refs=refs,
    )


def extract_action_candidates(speaker: str, text: str, timestamp: str) -> list[ActionCandidate]:
    """Extract generic action candidates from a finalized transcript utterance.

    This first slice is deliberately rule-based: it captures explicit action
    phrasing without adding another LLM call to the live meeting path.
    """

    normalized = normalize_text(text)
    refs = re.findall(r"\b[A-Z]{2,10}-\d+\b", text.upper())
    candidates: list[ActionCandidate] = []

    if refs and re.search(r"\b(update|comment|add|move|close|mark)\b", normalized):
        intent: Intent = "comment_issue" if "comment" in normalized or "add" in normalized else "update_issue"
        candidates.append(_make_candidate(
            speaker=speaker, text=text, timestamp=timestamp,
            target="linear", intent=intent, confidence=0.92, refs=refs,
        ))
        return candidates

    if "linear" in normalized and re.search(r"\b(make|create|open|file)\b.*\b(linear|ticket|issue)\b", normalized):
        candidates.append(_make_candidate(
            speaker=speaker, text=text, timestamp=timestamp,
            target="linear", intent="create_issue", confidence=0.88,
        ))
        return candidates

    if re.search(r"\b(make|create|open|file)\b.*\b(ticket|issue)\b", normalized) or "turn this into tickets" in normalized:
        candidates.append(_make_candidate(
            speaker=speaker, text=text, timestamp=timestamp,
            target="unknown", intent="create_issue", confidence=0.74,
        ))
        return candidates

    if "hermes session" in normalized or "start a hermes" in normalized or "spawn a hermes" in normalized:
        candidates.append(_make_candidate(
            speaker=speaker, text=text, timestamp=timestamp,
            target="hermes", intent="spawn_session", confidence=0.9,
        ))
        return candidates

    if re.search(r"\b(research|look up|find out|investigate)\b", normalized):
        candidates.append(_make_candidate(
            speaker=speaker, text=text, timestamp=timestamp,
            target="hermes", intent="research", confidence=0.72,
        ))
        return candidates

    if re.search(r"\b(run|check|deploy|restart|fix|edit|write|send)\b", normalized):
        candidates.append(_make_candidate(
            speaker=speaker, text=text, timestamp=timestamp,
            target="hermes", intent="unknown", confidence=0.6,
        ))

    return candidates


def merge_action_candidate(existing: list[ActionCandidate], candidate: ActionCandidate) -> list[ActionCandidate]:
    """Return candidates with near duplicates collapsed."""

    key = (normalize_text(candidate.quote), candidate.target, candidate.intent, tuple(candidate.refs))
    for item in existing:
        item_key = (normalize_text(item.quote), item.target, item.intent, tuple(item.refs))
        if item_key == key:
            return existing
    return [*existing, candidate]
