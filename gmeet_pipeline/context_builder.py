"""
ContextBuilder — assembles a rich system prompt from Hermes memory + recent sessions.

At call start, reads:
  1. ~/.hermes/memories/MEMORY.md  (agent's knowledge/notes, §-delimited)
  2. ~/.hermes/memories/USER.md    (user profile/preferences, §-delimited)
  3. Recent session files from ~/.hermes/sessions/ (last N, extracts first user message as topic)

Produces a single system prompt string that gets baked into the LLM call.
No RAG, no classification, no EXPAND — just front-load everything.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("gmeet_pipeline.context_builder")

ENTRY_DELIMITER = "§"

# Default paths — match Hermes agent's memory store
DEFAULT_HERMES_HOME = Path.home() / ".hermes"
DEFAULT_MEMORIES_DIR = DEFAULT_HERMES_HOME / "memories"
DEFAULT_SESSIONS_DIR = DEFAULT_HERMES_HOME / "sessions"


def _read_entries(path: Path) -> list[dict]:
    """Read a §-delimited memory file into entries."""
    if not path.exists():
        return []
    content = path.read_text().strip()
    if not content:
        return []
    entries = []
    for i, chunk in enumerate(content.split(ENTRY_DELIMITER)):
        chunk = chunk.strip()
        if chunk:
            entries.append({"index": i, "text": chunk})
    return entries


def _extract_session_topics(sessions_dir: Path, limit: int = 5) -> list[dict]:
    """Extract topic previews from the most recent Hermes session files."""
    if not sessions_dir.exists():
        return []

    sessions = sorted(
        sessions_dir.glob("session_2*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    topics = []
    for session_path in sessions:
        if len(topics) >= limit:
            break
        try:
            with open(session_path) as f:
                data = json.load(f)
            messages = data.get("messages", [])
            # Find first user message as topic
            for msg in messages[:10]:
                if msg.get("role") == "user":
                    content = msg.get("content", "").strip()
                    if content:
                        # Strip platform prefix like [Ethan]
                        clean = re.sub(r"^\[.*?\]\s*", "", content)
                        topics.append({
                            "topic": clean[:200],
                            "file": session_path.name,
                            "age_hours": round(
                                (time.time() - session_path.stat().st_mtime) / 3600, 1
                            ),
                        })
                        break
        except (json.JSONDecodeError, OSError, KeyError):
            continue

    return topics


class ContextBuilder:
    """Builds a rich system prompt from Hermes memory store + session context."""

    def __init__(
        self,
        memories_dir: Path = DEFAULT_MEMORIES_DIR,
        sessions_dir: Path = DEFAULT_SESSIONS_DIR,
        session_limit: int = 5,
    ):
        self.memories_dir = memories_dir
        self.sessions_dir = sessions_dir
        self.session_limit = session_limit
        self._built_at: float = 0.0
        self._cached_prompt: Optional[str] = None

    def build(self, force: bool = False) -> str:
        """Build the system prompt. Cached for max_age seconds."""
        if not force and self._cached_prompt and (time.time() - self._built_at) < 60:
            return self._cached_prompt

        parts: list[str] = []

        # 1. Memory entries (full text, not compressed)
        memory_entries = _read_entries(self.memories_dir / "MEMORY.md")
        if memory_entries:
            parts.append("## Agent Knowledge")
            for e in memory_entries:
                parts.append(e["text"])

        # 2. User profile entries
        user_entries = _read_entries(self.memories_dir / "USER.md")
        if user_entries:
            parts.append("\n## User Profile")
            for e in user_entries:
                parts.append(e["text"])

        # 3. Recent session context
        session_topics = _extract_session_topics(self.sessions_dir, self.session_limit)
        if session_topics:
            parts.append("\n## Recent Context")
            for s in session_topics:
                age = s["age_hours"]
                if age < 1:
                    age_str = f"{int(age * 60)}m ago"
                else:
                    age_str = f"{age:.0f}h ago"
                parts.append(f"- ({age_str}) {s['topic']}")

        self._cached_prompt = "\n\n".join(parts)
        self._built_at = time.time()

        total_chars = len(self._cached_prompt)
        logger.info(
            f"Context built: {len(memory_entries)} memory entries, "
            f"{len(user_entries)} user entries, "
            f"{len(session_topics)} sessions, "
            f"{total_chars} chars"
        )
        return self._cached_prompt

    def refresh_if_stale(self, max_age: int = 300) -> None:
        """Rebuild if memory files have been updated since last build."""
        for path in [self.memories_dir / "MEMORY.md", self.memories_dir / "USER.md"]:
            if path.exists() and path.stat().st_mtime > self._built_at:
                self.build(force=True)
                return
