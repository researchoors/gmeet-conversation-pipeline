"""Meeting artifact persistence for post-call automation."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .state import BotSession


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, set):
        return sorted(value)
    return value


def _safe_filename_part(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", text).strip("-") or "meeting"


def meeting_artifact_data(session: BotSession, end_reason: str = "call_ended") -> dict:
    """Return a JSON-serializable meeting artifact for a bot session."""

    return {
        "schema_version": 1,
        "bot_id": session.bot_id,
        "meeting_url": session.meeting_url,
        "status": session.status,
        "end_reason": end_reason,
        "created_at": session.created_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "participants": _jsonable(session.participants),
        "transcript": _jsonable(session.transcript),
        "conversation": _jsonable(session.conversation),
        "mode_events": _jsonable(session.mode_events),
        "action_candidates": _jsonable(session.action_candidates),
        "timing": {
            "last_llm_ms": session.last_llm_ms,
            "last_tts_ms": session.last_tts_ms,
            "last_total_ms": session.last_total_ms,
        },
    }


def write_meeting_artifact(session: BotSession, root: str | Path, end_reason: str = "call_ended") -> Path:
    """Persist a meeting artifact and return its path."""

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    started = _safe_filename_part(session.created_at.replace(":", ""))
    bot = _safe_filename_part(session.bot_id)
    path = root_path / f"{started}-{bot}.json"
    path.write_text(json.dumps(meeting_artifact_data(session, end_reason=end_reason), indent=2, sort_keys=True))
    return path
