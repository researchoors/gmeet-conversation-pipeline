"""
MemorySnapshot — standalone memory snapshot + RAG + query classification.

Parses MEMORY.md + USER.md into §-delimited entries with index + compressed summary.
Extracted from meeting_agent_rvc.py for reuse across the gmeet_pipeline.
"""

import re
import time
import logging
from pathlib import Path

logger = logging.getLogger("gmeet_pipeline.memory")

# ============================================================
# Constants
# ============================================================
ENTRY_DELIMITER = "§"

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

TOOL_KEYWORDS = {"github", "repo", "issue", "pr ", "pull request", "commit", "deploy",
                 "install", "server", "docker", "nomad", "latest issues"}

MEMORY_KEYWORDS = {"remember", "recall", "know about", "what is", "who is",
                   "tell me about", "explain", "describe", "about the",
                   "darkbloom", "d-inference", "swiftlm", "dflash", "researchoors",
                   "layr", "benchmark", "project", "setup", "configured",
                   "hermes", "ethan", "hank"}

SIMPLE_PATTERNS = [
    re.compile(r'^(hi|hey|hello|yo|sup|what\'s up|thanks|ok|yes|no|sure|cool|nice|got it)', re.I),
    re.compile(r'^\d+\s*[\+\-\*\/]\s*\d+'),
]

EXPAND_RE = re.compile(r'EXPAND\[(\d+(?:\s*,\s*\d+)*)\]')


class MemorySnapshot:
    """Parses MEMORY.md + USER.md into §-delimited entries with index + compressed summary."""

    def __init__(self, memory_file: Path, user_file: Path):
        self.memory_file = memory_file
        self.user_file = user_file
        self.entries: list[dict] = []      # list of dicts: {index, source, text}
        self.summary: str = ""             # ultra-compressed overview (~80 tokens)
        self.index_text: str = ""          # one-line-per-entry index
        self.built_at: float = 0.0

    def refresh_if_stale(self, max_age: int = 300) -> None:
        mem_mtime = self.memory_file.stat().st_mtime if self.memory_file.exists() else 0
        usr_mtime = self.user_file.stat().st_mtime if self.user_file.exists() else 0
        if max(mem_mtime, usr_mtime) > self.built_at:
            self.build()

    def build(self) -> None:
        entries: list[dict] = []
        for source, path in [("memory", self.memory_file), ("user", self.user_file)]:
            if not path.exists():
                continue
            content = path.read_text().strip()
            if not content:
                continue
            for chunk in content.split(ENTRY_DELIMITER):
                chunk = chunk.strip()
                if chunk:
                    entries.append({"index": len(entries), "source": source, "text": chunk})

        self.entries = entries
        self.built_at = time.time()

        # Build index text
        lines: list[str] = []
        for e in entries:
            preview = e["text"][:80].replace("\n", " ").strip()
            if len(e["text"]) > 80:
                preview += "..."
            lines.append(f"[{e['index']}] {preview}")
        self.index_text = "\n".join(lines)

        # Build compressed summary
        parts: list[str] = []
        for e in entries[:5]:
            first = e["text"].split(".")[0].strip()
            if first:
                parts.append(first)
        self.summary = " | ".join(parts)

        logger.info(f"Memory snapshot built: {len(entries)} entries")

    def rag_retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        query_terms = {w.lower() for w in re.findall(r'\w+', query) if w.lower() not in STOP_WORDS}
        if not query_terms:
            return []
        scored: list[tuple[int, dict]] = []
        for e in self.entries:
            entry_terms = {w.lower() for w in re.findall(r'\w+', e["text"]) if w.lower() not in STOP_WORDS}
            overlap = len(query_terms & entry_terms)
            if overlap > 0:
                scored.append((overlap, e))
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:top_k]]

    def classify_query(self, query: str) -> str:
        """Classify query complexity → 'fast', 'standard', or 'deep'."""
        q_lower = query.lower().strip()
        for pat in SIMPLE_PATTERNS:
            if pat.match(q_lower):
                return "fast"
        query_words = set(re.findall(r'\w+', q_lower))
        if query_words & TOOL_KEYWORDS:
            return "deep"
        for kw in MEMORY_KEYWORDS:
            if kw in q_lower:
                return "standard"
        if self.rag_retrieve(query, top_k=1):
            return "standard"
        return "fast"

    def parse_expands(self, text: str) -> list[int]:
        """Parse EXPAND[n1,n2,...] directives from LLM output, returning valid entry indices."""
        indices: set[int] = set()
        for match in EXPAND_RE.finditer(text):
            for num_str in match.group(1).split(","):
                try:
                    idx = int(num_str.strip())
                    if 0 <= idx < len(self.entries):
                        indices.add(idx)
                except ValueError:
                    continue
        return sorted(indices)

    @staticmethod
    def strip_expands(text: str) -> str:
        """Remove EXPAND[...] directives from text."""
        return EXPAND_RE.sub("", text).strip()
