"""Optional real-Hermes smoke test for post-call follow-up actions.

This test is intentionally skipped by default because it launches the real Hermes
CLI and requires a configured model/provider in the environment. Enable it in CI
or locally with:

    GMEET_RUN_REAL_HERMES_E2E=1 python -m pytest tests/test_real_hermes_post_call.py -v
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


def test_real_hermes_agent_can_process_post_call_artifact(tmp_path: Path):
    """Launch a real Hermes one-shot agent against a synthetic call artifact.

    The task is intentionally local and low-risk: read the artifact and write a
    small JSON receipt. This verifies the post-call handoff contract with the
    actual Hermes runtime without requiring Linear/GitHub credentials or making
    external side effects beyond the model call itself.
    """

    if os.environ.get("GMEET_RUN_REAL_HERMES_E2E") != "1":
        pytest.skip("set GMEET_RUN_REAL_HERMES_E2E=1 to run real Hermes E2E")

    hermes_cmd = os.environ.get("GMEET_REAL_HERMES_CMD", "hermes")
    hermes_path = shutil.which(hermes_cmd)
    if not hermes_path:
        pytest.skip(f"Hermes CLI not found: {hermes_cmd}")

    artifact_path = tmp_path / "meeting-artifact.json"
    receipt_path = tmp_path / "post-call-receipt.json"
    artifact = {
        "schema_version": 1,
        "bot_id": "bot-real-hermes-test",
        "meeting_url": "https://meet.google.com/test-test-test",
        "transcript": [
            {
                "speaker": "Ethan",
                "text": "After this call, write a local receipt that says hello.",
                "timestamp": "2026-05-05T00:00:00Z",
            }
        ],
        "action_candidates": [
            {
                "id": "act-local-receipt",
                "speaker": "Ethan",
                "quote": "write a local receipt that says hello",
                "timestamp": "2026-05-05T00:00:00Z",
                "target": "file",
                "intent": "edit_file",
                "confidence": 0.99,
                "refs": [],
                "status": "candidate",
            }
        ],
    }
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    prompt = f"""
You are running a CI smoke test for the GMeet post-call Hermes handoff.

Read the meeting artifact JSON at:
{artifact_path}

Then write exactly this JSON object to:
{receipt_path}

{{
  "ok": true,
  "source": "gmeet-post-call-real-hermes",
  "artifact_bot_id": "bot-real-hermes-test",
  "message": "hello"
}}

Do not perform any network, Linear, GitHub, messaging, or shell side effects.
After writing the file, respond with the receipt path only.
""".strip()

    command = [
        hermes_path,
        "chat",
        "-q",
        prompt,
        "--source",
        "gmeet-post-call-real-e2e",
        "--toolsets",
        "file",
    ]
    model = os.environ.get("GMEET_REAL_HERMES_MODEL")
    provider = os.environ.get("GMEET_REAL_HERMES_PROVIDER")
    if model:
        command.extend(["--model", model])
    if provider:
        command.extend(["--provider", provider])

    result = subprocess.run(command, text=True, capture_output=True, timeout=240)

    assert result.returncode == 0, result.stderr[-2000:] + "\n" + result.stdout[-2000:]
    assert receipt_path.exists(), result.stdout[-2000:] + "\n" + result.stderr[-2000:]

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt == {
        "ok": True,
        "source": "gmeet-post-call-real-hermes",
        "artifact_bot_id": "bot-real-hermes-test",
        "message": "hello",
    }
