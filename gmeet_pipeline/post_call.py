"""Post-call Hermes orchestration for generic meeting actions."""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
from pathlib import Path

from .action_router import build_action_route_plan, write_inbox_items

logger = logging.getLogger("gmeet_pipeline.post_call")


def build_post_call_prompt(artifact_path: str | Path, inbox_path: str | Path) -> str:
    """Build a self-contained prompt for a lower-cost post-call Hermes run."""

    artifact = Path(artifact_path)
    inbox = Path(inbox_path)
    paths_json = json.dumps({"artifact_path": str(artifact), "inbox_path": str(inbox)}, indent=2)
    return f"""You are a lower-cost post-call Hermes action router.

Read this JSON for file locations:
```json
{paths_json}
```

Your job is to synthesize the transcript into arbitrary follow-up actions. Linear is only one possible target; other valid targets include GitHub, repo/code work, research, docs, terminal checks, messages, reminders, and follow-up planning.

Use the tools available to this Hermes runtime and obey its authorization/approval policy. If a tool is unavailable, an action is ambiguous, or authorization blocks it, do not pretend the action succeeded. Instead write a structured review item to the Hermes data inbox from the JSON above.

Process:
1. Summarize the meeting: decisions, open questions, and important context.
2. Convert explicit requests into a structured action plan.
3. Execute only high-confidence, low-risk actions that the runtime is authorized to perform.
4. Fan out independent work into separate Hermes sessions only when useful and bounded.
5. Use lower-cost / lower-thinking model behavior by default; escalate only for high-risk or complex work.
6. For Linear/GitHub issue trackers or docs, search existing artifacts before creating duplicates. Update/comment explicit refs like DAR-12 instead of creating duplicates.
7. Produce this exact output schema as JSON:
{{
  "summary": "...",
  "actions_taken": [],
  "spawned_sessions": [],
  "artifacts_or_urls": [],
  "approvals_requested": [],
  "inbox_items": [],
  "skipped_items": []
}}
"""


def build_post_call_command(
    *,
    artifact_path: str | Path,
    hermes_cmd: str,
    model: str,
    provider: str = "",
    toolsets: str = "terminal,file,skills,session_search",
    inbox_path: str | Path,
    max_parallel_sessions: int = 3,
) -> list[str]:
    """Build the Hermes CLI command for post-call processing."""

    prompt = build_post_call_prompt(artifact_path, inbox_path)
    prompt += f"\nMaximum parallel fan-out sessions: {max_parallel_sessions}.\n"
    command = [hermes_cmd, "chat", "-q", prompt, "--source", "gmeet-post-call"]
    if toolsets:
        command.extend(["--toolsets", toolsets])
    if model:
        command.extend(["--model", model])
    if provider:
        command.extend(["--provider", provider])
    return command


def spawn_post_call_sessions(
    *,
    artifact_path: str | Path,
    hermes_cmd: str,
    model: str,
    provider: str = "",
    toolsets: str = "terminal,file,skills,session_search",
    inbox_path: str | Path,
    max_parallel_sessions: int = 3,
    dry_run: bool = False,
) -> dict:
    """Spawn the post-call Hermes run or return a dry-run command."""

    artifact_path = Path(artifact_path)
    try:
        artifact = json.loads(artifact_path.read_text())
        route_plan = build_action_route_plan(artifact, default_model=model)
        write_inbox_items(route_plan, inbox_path)
        route_plan_path = artifact_path.with_suffix(".route-plan.json")
        route_plan_path.write_text(json.dumps(route_plan, indent=2, sort_keys=True))
    except Exception as exc:
        # The spawned Hermes session can still inspect the raw artifact. Do not
        # block post-call processing on router/inbox best-effort failures.
        logger.warning("Post-call route planning failed: %s", exc, exc_info=True)
        route_plan_path = None

    command = build_post_call_command(
        artifact_path=artifact_path,
        hermes_cmd=hermes_cmd,
        model=model,
        provider=provider,
        toolsets=toolsets,
        inbox_path=inbox_path,
        max_parallel_sessions=max_parallel_sessions,
    )
    printable = " ".join(shlex.quote(part) for part in command)
    if dry_run:
        return {"spawned": False, "command": printable, "pid": None, "route_plan": str(route_plan_path) if route_plan_path else None}

    proc = subprocess.Popen(  # noqa: S603 - command is list-form and privileged config-driven.
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
    )
    return {"spawned": True, "command": printable, "pid": proc.pid, "route_plan": str(route_plan_path) if route_plan_path else None}
