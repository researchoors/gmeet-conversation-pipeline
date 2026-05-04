"""Tests for meeting artifacts and post-call Hermes orchestration."""

import json

from gmeet_pipeline.artifacts import write_meeting_artifact
from gmeet_pipeline.post_call import build_post_call_prompt, spawn_post_call_sessions
from gmeet_pipeline.state import BotSession
from gmeet_pipeline.actions import ActionCandidate


def test_writes_meeting_artifact_with_transcript_and_actions(tmp_path):
    session = BotSession(bot_id="bot-123", meeting_url="https://meet.google.com/abc-defg-hij")
    session.transcript.append({"speaker": "Ethan", "text": "Make a ticket", "timestamp": "t1"})
    session.conversation.append({"role": "assistant", "content": "I'll capture that for after the call."})
    session.action_candidates.append(
        ActionCandidate(
            id="act-1",
            speaker="Ethan",
            quote="Make a ticket",
            timestamp="t1",
            target="linear",
            intent="create_issue",
            confidence=0.9,
        )
    )

    path = write_meeting_artifact(session, tmp_path, end_reason="call_ended")
    data = json.loads(path.read_text())

    assert data["bot_id"] == "bot-123"
    assert data["meeting_url"] == "https://meet.google.com/abc-defg-hij"
    assert data["end_reason"] == "call_ended"
    assert data["transcript"][0]["text"] == "Make a ticket"
    assert data["conversation"][0]["content"] == "I'll capture that for after the call."
    assert data["action_candidates"][0]["target"] == "linear"


def test_post_call_prompt_is_generic_and_low_cost(tmp_path):
    artifact = tmp_path / "artifact.json"
    artifact.write_text('{"transcript": []}')

    prompt = build_post_call_prompt(artifact, inbox_path=tmp_path / "inbox")

    assert str(artifact) in prompt
    assert "arbitrary follow-up actions" in prompt
    assert "Linear is only one possible target" in prompt
    assert "Hermes data inbox" in prompt
    assert "lower-cost" in prompt


def test_spawn_post_call_sessions_uses_lower_cost_model_and_dry_run(tmp_path):
    artifact = tmp_path / "artifact.json"
    artifact.write_text('{"transcript": []}')

    result = spawn_post_call_sessions(
        artifact_path=artifact,
        hermes_cmd="hermes",
        model="google/gemini-2.5-flash",
        provider="openrouter",
        toolsets="terminal,file,skills,session_search",
        inbox_path=tmp_path / "inbox",
        dry_run=True,
    )

    assert result["spawned"] is False
    command = result["command"]
    assert "--model" in command
    assert "google/gemini-2.5-flash" in command
    assert "--provider" in command
    assert "openrouter" in command
