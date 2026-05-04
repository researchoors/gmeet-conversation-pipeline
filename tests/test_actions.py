"""Tests for generic meeting action extraction."""

from gmeet_pipeline.actions import extract_action_candidates, merge_action_candidate


def test_extracts_generic_hermes_action_candidate():
    candidates = extract_action_candidates(
        speaker="Ethan",
        text="After this call start a Hermes session to research the deployment failure",
        timestamp="2026-05-04T00:00:00Z",
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.target == "hermes"
    assert candidate.intent == "spawn_session"
    assert candidate.status == "candidate"
    assert candidate.confidence >= 0.8
    assert candidate.quote.startswith("After this call")


def test_extracts_linear_as_one_possible_target():
    candidates = extract_action_candidates(
        speaker="Ethan",
        text="Make a Linear ticket for wake words",
        timestamp="2026-05-04T00:00:00Z",
    )

    assert len(candidates) == 1
    assert candidates[0].target == "linear"
    assert candidates[0].intent == "create_issue"


def test_extracts_explicit_issue_ref_update():
    candidates = extract_action_candidates(
        speaker="Ethan",
        text="Update DAR-12 with this context",
        timestamp="2026-05-04T00:00:00Z",
    )

    assert len(candidates) == 1
    assert candidates[0].target == "linear"
    assert candidates[0].intent == "update_issue"
    assert candidates[0].refs == ["DAR-12"]


def test_dedupes_near_duplicate_candidate():
    existing = extract_action_candidates(
        speaker="Ethan",
        text="Make a Linear ticket for wake words",
        timestamp="2026-05-04T00:00:00Z",
    )[0]
    duplicate = extract_action_candidates(
        speaker="Ethan",
        text="make a linear ticket for wake words",
        timestamp="2026-05-04T00:00:01Z",
    )[0]

    merged = merge_action_candidate([existing], duplicate)

    assert len(merged) == 1
    assert merged[0].quote == existing.quote
