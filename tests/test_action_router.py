"""Tests for post-call generic action routing and inbox fallback."""

import json

from gmeet_pipeline.action_router import build_action_route_plan, write_inbox_items


def test_builds_generic_route_plan_with_fanout_groups():
    artifact = {
        "transcript": [
            {"speaker": "Ethan", "text": "After this call start a Hermes session to research deployment", "timestamp": "t1"},
            {"speaker": "Ethan", "text": "Make a Linear ticket for wake words", "timestamp": "t2"},
        ],
        "action_candidates": [
            {"id": "a1", "target": "hermes", "intent": "research", "quote": "research deployment", "confidence": 0.9, "speaker": "Ethan", "timestamp": "t1", "refs": []},
            {"id": "a2", "target": "linear", "intent": "create_issue", "quote": "ticket for wake words", "confidence": 0.9, "speaker": "Ethan", "timestamp": "t2", "refs": []},
        ],
    }

    plan = build_action_route_plan(artifact, default_model="google/gemini-2.5-flash")

    assert "summary" in plan
    assert len(plan["actions"]) == 2
    assert len(plan["fanout_groups"]) == 2
    assert {a["target"] for a in plan["actions"]} == {"hermes", "linear"}
    assert all(group["model"] == "google/gemini-2.5-flash" for group in plan["fanout_groups"])


def test_low_confidence_actions_go_to_inbox(tmp_path):
    artifact = {
        "action_candidates": [
            {"id": "a1", "target": "hermes", "intent": "unknown", "quote": "maybe do that thing", "confidence": 0.4, "speaker": "Ethan", "timestamp": "t1", "refs": []},
        ]
    }

    plan = build_action_route_plan(artifact, default_model="google/gemini-2.5-flash")
    paths = write_inbox_items(plan, tmp_path)

    assert len(plan["actions"]) == 0
    assert len(plan["inbox_items"]) == 1
    assert len(paths) == 1
    data = json.loads(paths[0].read_text())
    assert data["status"] == "needs_human_review"
    assert data["quote"] == "maybe do that thing"
