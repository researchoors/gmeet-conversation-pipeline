"""Generic post-call action routing for meeting artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_LOW_CONFIDENCE_THRESHOLD = 0.7


def _summarize_transcript(transcript: list[dict]) -> str:
    if not transcript:
        return "No transcript entries captured."
    snippets = []
    for entry in transcript[:5]:
        speaker = entry.get("speaker", "Unknown")
        text = entry.get("text", "")
        snippets.append(f"{speaker}: {text}")
    return " | ".join(snippets)


def _required_toolsets(target: str, intent: str) -> str:
    if target in {"linear", "github"}:
        return "terminal,file,skills,session_search"
    if intent in {"research"}:
        return "terminal,file,skills,session_search"
    if intent in {"send_message"}:
        return "messaging,skills,session_search"
    return "terminal,file,skills,session_search"


def _action_prompt(candidate: dict) -> str:
    return (
        "Execute this post-call meeting action if authorized by the Hermes runtime. "
        "If authorization/tooling is unavailable, produce an audit item instead.\n\n"
        f"Speaker: {candidate.get('speaker')}\n"
        f"Quote: {candidate.get('quote')}\n"
        f"Target: {candidate.get('target')}\n"
        f"Intent: {candidate.get('intent')}\n"
        f"Refs: {candidate.get('refs', [])}\n"
    )


def build_action_route_plan(artifact: dict[str, Any], default_model: str) -> dict[str, Any]:
    """Build a bounded generic route plan from a meeting artifact.

    The router is intentionally deterministic and cheap. It partitions explicit
    high-confidence candidates into independently spawnable fan-out groups and
    sends ambiguous items to the inbox for human review.
    """

    transcript = artifact.get("transcript", [])
    candidates = artifact.get("action_candidates", [])
    plan: dict[str, Any] = {
        "summary": _summarize_transcript(transcript),
        "decisions": [],
        "actions": [],
        "fanout_groups": [],
        "inbox_items": [],
    }

    for candidate in candidates:
        confidence = float(candidate.get("confidence", 0.0))
        if confidence < _LOW_CONFIDENCE_THRESHOLD or candidate.get("intent") == "unknown":
            item = dict(candidate)
            item["status"] = "needs_human_review"
            item["reason"] = "low_confidence_or_unknown_intent"
            plan["inbox_items"].append(item)
            continue

        action = dict(candidate)
        action["status"] = "ready"
        action["risk"] = "low"
        action["model"] = default_model
        action["required_toolsets"] = _required_toolsets(
            str(candidate.get("target", "unknown")),
            str(candidate.get("intent", "unknown")),
        )
        action["prompt"] = _action_prompt(candidate)
        plan["actions"].append(action)
        plan["fanout_groups"].append({
            "id": f"fanout-{candidate.get('id', len(plan['fanout_groups']))}",
            "action_ids": [candidate.get("id")],
            "model": default_model,
            "toolsets": action["required_toolsets"],
            "prompt": action["prompt"],
        })

    return plan


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-") or "inbox-item"


def write_inbox_items(plan: dict[str, Any], inbox_path: str | Path) -> list[Path]:
    """Write review-required items to the Hermes data inbox."""

    root = Path(inbox_path)
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for idx, item in enumerate(plan.get("inbox_items", []), start=1):
        item_id = _safe_name(str(item.get("id") or f"item-{idx}"))
        path = root / f"{item_id}.json"
        suffix = 1
        while path.exists():
            suffix += 1
            path = root / f"{item_id}-{suffix}.json"
        path.write_text(json.dumps(item, indent=2, sort_keys=True))
        paths.append(path)
    return paths
