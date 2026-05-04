# GMeet Post-Call Hermes Actions + Wake Mode Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Turn Hank Bob from a conversational Google Meet participant into an action-aware meeting agent that can mute/wake by voice and launch a post-call Hermes session to execute any appropriate follow-up actions from the transcript, not just Linear work.

**Architecture:** Keep the live call path low-risk: transcript events update session state, mode state, and an action ledger. On call end, persist a meeting artifact and spawn a separate lower-thinking Hermes one-shot session with the artifact path and a self-contained action prompt. The spawned Hermes runtime decides authorization, available tools, and whether to execute, ask for approval, or emit an audit-only plan. Do not hard-code Linear as the only target.

**Tech Stack:** FastAPI gmeet-conversation-pipeline, Recall.ai webhooks, existing `BotSession` registry, Hermes CLI (`hermes chat -q`), Hermes runtime authorization/tool policy, optional Linear/GitHub/file/terminal tools depending on the spawned agent profile.

---

## Phase 1 — Voice Response Mode State Machine

### Task 1: Add per-session response mode fields

**Objective:** Store whether Hank Bob should respond or only transcribe.

**Files:**
- Modify: `gmeet_pipeline/state.py`
- Test: `tests/test_state.py` or existing state test file

**Implementation shape:**
- Add `response_mode: Literal["active", "silent_transcribe"] = "active"` to `BotSession`.
- Add `mode_events: list[dict] = field(default_factory=list)` to track transitions.

**Acceptance:** New sessions default to `active` and mode events serialize in any session-state/debug endpoint.

### Task 2: Add wake/silence phrase matcher

**Objective:** Detect text commands like "Hank Bob quiet" and "Hey Hank Bob" from Recall STT output.

**Files:**
- Create: `gmeet_pipeline/wake_words.py`
- Test: `tests/test_wake_words.py`

**Implementation shape:**
- Normalize text: lowercase, strip punctuation, collapse whitespace.
- Include aliases for expected STT variants: `hank bob`, `hankbot`, `hank bop`, `hang bob`, `hank barb`, etc.
- Use a conservative fuzzy match with `difflib.SequenceMatcher` for the bot-name span.
- Classify:
  - silence: bot name + (`shut up`, `quiet`, `not now`, `stop talking`, `mute`, `be quiet`)
  - wake: bot name alone or bot name + (`wake up`, `come back`, `listen`, `you there`)
- Return structured result: `{kind: "silence"|"wake"|None, confidence, matched_phrase}`.

**Acceptance:** Fixtures for the phrases above pass without calling LLM/TTS.

### Task 3: Gate LLM/TTS queueing by response mode

**Objective:** Let silence commands stop responses while continuing transcript capture.

**Files:**
- Modify: `gmeet_pipeline/webhook.py`
- Test: webhook transcript fixture tests

**Implementation shape:**
- On finalized transcript, always append transcript/conversation.
- Run wake/silence classifier before queueing response.
- If silence command: set `session.response_mode = "silent_transcribe"`, append mode event, do not queue TTS.
- If wake command while silent: set `active`, append mode event, optionally acknowledge once.
- If `silent_transcribe`: record utterance but skip `session.response_queue.put(...)`.

**Acceptance:** While silent, `transcript_count` increases and `last_llm_ms` does not change.

---

## Phase 2 — General Action Ledger During Conversation

### Task 4: Add action candidate model

**Objective:** Preserve possible post-call actions with provenance without assuming the target system in advance.

**Files:**
- Modify: `gmeet_pipeline/state.py`
- Create: `gmeet_pipeline/actions.py`
- Test: `tests/test_actions.py`

**Implementation shape:**
Each candidate should include:
- `id`
- `speaker`
- `quote`
- `timestamp`
- `target`: `linear|github|terminal|file|browser|hermes|unknown`
- `intent`: `create_issue|update_issue|comment_issue|spawn_session|run_command|edit_file|research|send_message|unknown`
- `confidence`: float
- `refs`: e.g. `DAR-12`
- `status`: `candidate|confirmed|needs_human_review|executed|skipped`

**Acceptance:** Candidate objects are JSON-serializable and dedupe by normalized quote + target + refs.

### Task 5: Extract action candidates from transcript events

**Objective:** Recognize explicit meeting utterances as generic action candidates.

**Files:**
- Modify: `gmeet_pipeline/actions.py`
- Modify: `gmeet_pipeline/webhook.py`
- Test: `tests/test_actions.py`

**Implementation shape:**
- Start with rule-based extractor, not an LLM call.
- Patterns should capture broad action classes:
  - `make/create/open (a )?(linear|ticket|issue)` → issue candidate
  - `update/comment/add.*(DAR-\d+)` → issue update/comment candidate
  - `after this call.*hermes session` → Hermes spawn candidate
  - `run/check/look up/research/send/edit/fix/deploy` → generic Hermes action candidate
  - `turn this into tickets` → issue create-many candidate
- Add candidates on finalized transcript.
- Keep the extractor target-agnostic where possible; the post-call Hermes runtime resolves tool availability and authorization.

**Acceptance:** User's call request produces candidates for post-call Hermes actions, including but not limited to Linear issue work.

---

## Phase 3 — Transcript Artifact + Post-Call Hermes Spawn

### Task 6: Persist meeting artifact on call end

**Objective:** Write the complete meeting context to disk when Recall reports terminal call state.

**Files:**
- Create: `gmeet_pipeline/artifacts.py`
- Modify: `gmeet_pipeline/webhook.py`
- Test: `tests/test_artifacts.py`

**Artifact path:** `~/.hermes/gmeet-artifacts/{started_at}-{bot_id}.json`

**Artifact contents:**
- meeting URL
- bot ID
- participants
- transcript entries
- Hank Bob responses
- mode events
- action candidates
- timing metrics
- end reason/status changes

**Acceptance:** A fixture `bot.call_ended` writes an artifact containing both human utterances and bot responses.

### Task 7: Build self-contained lower-thinking Hermes post-call prompt

**Objective:** Generate a prompt that a fresh, cheaper/lower-thinking Hermes session can execute without current chat context.

**Files:**
- Create: `gmeet_pipeline/post_call.py`
- Test: `tests/test_post_call.py`

**Prompt requirements:**
- Read the artifact path.
- Summarize decisions/open questions/action items.
- Execute appropriate follow-up actions using whatever tools the spawned Hermes runtime is authorized to use.
- Let Hermes runtime authorization/approval policy decide whether side effects are allowed.
- For Linear/GitHub issue actions, search existing issues before creating duplicates and update/comment explicit refs like `DAR-12`.
- Execute only high-confidence explicit asks.
- Put ambiguous items under `needs_human_review`.
- Final response must list actions taken, URLs/artifacts touched, approvals requested/denied, and skipped items.

**Acceptance:** Prompt includes artifact path, generic action policy, authorization delegation to Hermes runtime, confidence policy, lower-thinking model requirement, model budget, fan-out policy, inbox fallback, and exact expected output schema.

### Task 8: Add generic action router and fan-out plan

**Objective:** Convert the transcript artifact into a bounded action routing plan before spawning execution sessions.

**Files:**
- Create: `gmeet_pipeline/action_router.py`
- Test: `tests/test_action_router.py`

**Implementation shape:**
- Input: meeting artifact JSON.
- Output: structured plan with `summary`, `decisions`, `actions[]`, `inbox_items[]`, and `fanout_groups[]`.
- Each action includes: target, operation, prompt, required toolsets, confidence, risk, suggested model, and whether it can run autonomously.
- Group independent actions into separate session runs.
- Route ambiguous/low-confidence items to the data inbox.

**Model policy:**
- Default summarizer/router model should be lower-cost/lower-thinking.
- Escalate only when `risk=high` or the user explicitly requests deeper reasoning.

**Acceptance:** A mixed transcript fixture creates one coding-session action, one Linear action, and one inbox review item without making Linear the default.

---

### Task 9: Spawn Hermes one-shot after call end

**Objective:** Kick off a separate Hermes session once the call is over.

**Files:**
- Modify: `gmeet_pipeline/post_call.py`
- Modify: `gmeet_pipeline/config.py`
- Modify: `gmeet_pipeline/webhook.py`
- Test: `tests/test_post_call.py`

**Config:**
- `GMEET_POST_CALL_HERMES_ENABLED=true|false`
- `GMEET_POST_CALL_HERMES_CMD=hermes`
- `GMEET_POST_CALL_TOOLSETS=terminal,file,skills,session_search
GMEET_POST_CALL_MODEL=<cheap-fast-model>
GMEET_POST_CALL_PROVIDER=<provider>`

**Command shape:**
```bash
hermes chat -q "$PROMPT" \
  --source gmeet-post-call \
  --toolsets terminal,file,skills,session_search \
  --model "$GMEET_POST_CALL_MODEL" \
  --provider "$GMEET_POST_CALL_PROVIDER"
```

The spawned session should use a lower-thinking/cheaper model by default. If a required tool is unavailable or authorization blocks the action, the prompt should tell it to emit an audit item or approval request instead of pretending.

**Acceptance:** Call-end fixture verifies subprocess invocation with the generated prompt. Integration test can disable actual spawn.

---

## Phase 4 — Live Conversation Context

### Task 10: Teach the live LLM to listen for actionable goals

**Objective:** Make Hank Bob acknowledge action-oriented requests appropriately without doing side effects mid-call.

**Files:**
- Modify: `gmeet_pipeline/llm/local.py` and/or prompt builder used by flash routing
- Modify: `gmeet_pipeline/context_builder.py`
- Test: prompt snapshot test if available

**Prompt language:**
- This is a meeting agent.
- Listen for explicit tasks, decisions, and ticket/update requests.
- Acknowledge captured actions briefly.
- Do not claim an action has been completed during the call unless a tool actually ran.
- Post-call automation will synthesize and execute confirmed actions.

**Acceptance:** Prompt contains this policy, and test response does not claim fake Linear completion.

---

## Phase 5 — Linear Issue Publishing

### Task 10: Publish/update implementation tracking issues

**Objective:** Track this plan as independent vertical-slice issues in the available tracker. Linear is preferred when configured, but the implementation should not depend on Linear being the only target.

**Blocked by:** None for inbox publishing. Linear publishing is blocked by Linear credential/tool availability in the active Hermes environment.

**Issue set:**
1. Post-call lower-thinking Hermes session for generic transcript-derived actions — P2
2. Wake/silence voice state machine — P2
3. Meeting action-intent extraction ledger — P3
4. Transcript artifact persistence — P3
5. Live prompt policy for action-aware meeting behavior — P4

**Acceptance:** Each work item links back to this plan and has acceptance criteria. If Linear is unavailable, the inbox receives equivalent work-item files.
