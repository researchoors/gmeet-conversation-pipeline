# gmeet-conversation-pipeline Refactor Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Refactor two 900-1100 line monoliths into a modular package with abstracted transport, TTS, and LLM layers.

**Architecture:** Split `meeting_agent.py` and `meeting_agent_rvc.py` into `gmeet_pipeline/` package. Abstract transport (Recall.ai, future Chrome/Twilio), TTS (ElevenLabs, local Kokoro+RVC), and LLM (OpenRouter with/without voice gateway routing). Pydantic settings replace scattered globals. BotSession dataclass replaces raw dicts.

**Tech Stack:** Python 3.11, FastAPI, pydantic-settings, httpx, httpx

---

### Task 1: Create branch and package structure

**Objective:** Set up feature branch and empty package skeleton

**Files:**
- Create: `gmeet_pipeline/__init__.py`
- Create: `gmeet_pipeline/config.py` (empty)
- Create: `gmeet_pipeline/state.py` (empty)
- Create: `gmeet_pipeline/transports/__init__.py`
- Create: `gmeet_pipeline/transports/base.py` (empty)
- Create: `gmeet_pipeline/transports/recall.py` (empty)
- Create: `gmeet_pipeline/llm/__init__.py`
- Create: `gmeet_pipeline/llm/base.py` (empty)
- Create: `gmeet_pipeline/llm/openrouter.py` (empty)
- Create: `gmeet_pipeline/tts/__init__.py`
- Create: `gmeet_pipeline/tts/base.py` (empty)
- Create: `gmeet_pipeline/tts/elevenlabs.py` (empty)
- Create: `gmeet_pipeline/tts/local.py` (empty)
- Create: `gmeet_pipeline/memory.py` (empty)
- Create: `gmeet_pipeline/webhook.py` (empty)
- Create: `gmeet_pipeline/ws_manager.py` (empty)
- Create: `gmeet_pipeline/agent_page.py` (empty)
- Create: `gmeet_pipeline/server.py` (empty)
- Create: `gmeet_pipeline/main.py` (empty)

**Step 1:** Create feature branch

```bash
cd /tmp/gmeet-refactor
git checkout -b refactor/modular-package
```

**Step 2:** Create package structure

```bash
mkdir -p gmeet_pipeline/transports gmeet_pipeline/llm gmeet_pipeline/tts
touch gmeet_pipeline/__init__.py
touch gmeet_pipeline/transports/__init__.py
touch gmeet_pipeline/llm/__init__.py
touch gmeet_pipeline/tts/__init__.py
```

**Step 3:** Commit

```bash
git add -A && git commit -m "chore: scaffold gmeet_pipeline package structure"
```

---

### Task 2: config.py — Pydantic settings

**Objective:** Replace all scattered `os.environ.get()` globals with a single Pydantic BaseSettings class

**Files:**
- Create: `gmeet_pipeline/config.py`

**Step 1:** Write config with all fields from both monoliths

Key fields:
- recall_api_key, recall_base
- openrouter_key, llm_model
- fast_model, standard_model, deep_model (voice gateway)
- elevenlabs_key, elevenlabs_voice, elevenlabs_model
- rvc_model_path, rvc_exp_dir, rvc_repo_dir, rvc_f0_method, rvc_f0_up_key, rvc_index_rate
- kokoro_voice
- service_url, port
- tts_backend: Literal["elevenlabs", "local"] = "elevenlabs"
- llm_routing: Literal["simple", "voice_gateway"] = "simple"
- memory_file, user_file paths
- hermes_home, audio_dir

**Step 2:** Commit

```bash
git add gmeet_pipeline/config.py && git commit -m "feat: add pydantic settings config"
```

---

### Task 3: state.py — BotSession and BotRegistry

**Objective:** Replace raw `active_bots` dict with typed dataclass + registry

**Files:**
- Create: `gmeet_pipeline/state.py`

**Step 1:** Define BotSession dataclass with fields: bot_id, meeting_url, status, transcript, conversation, speaking, respond_lock, last_processed_ts, expanded_entries, created_at

**Step 2:** Define BotRegistry class with: create(), get(), remove(), list() methods. Thread-safe with asyncio.Lock.

**Step 3:** Commit

```bash
git add gmeet_pipeline/state.py && git commit -m "feat: add BotSession dataclass and BotRegistry"
```

---

### Task 4: transports/base.py — Abstract transport

**Objective:** Define the transport interface for joining/leaving meetings

**Files:**
- Create: `gmeet_pipeline/transports/base.py`

**Step 1:** Define ABC with: async join(meeting_url, bot_name, config) -> bot_id, async leave(bot_id), async get_status(bot_id) -> str

**Step 2:** Commit

---

### Task 5: transports/recall.py — Recall.ai transport

**Objective:** Extract Recall.ai bot creation/leaving from both monoliths

**Files:**
- Create: `gmeet_pipeline/transports/recall.py`

**Step 1:** Implement RecallTransport(BaseTransport) with join(), leave(), get_status()

**Step 2:** Commit

---

### Task 6: llm/base.py + llm/openrouter.py — LLM abstraction

**Objective:** Abstract LLM calls. Simple path = direct OpenRouter. Voice gateway path = routing + RAG + EXPAND.

**Files:**
- Create: `gmeet_pipeline/llm/base.py`
- Create: `gmeet_pipeline/llm/openrouter.py`

**Step 1:** Define BaseLLM ABC with: async generate(conversation, message, bot_state) -> Optional[str]

**Step 2:** Implement SimpleOpenRouterLLM (from meeting_agent.py logic)

**Step 3:** Implement VoiceGatewayLLM (from meeting_agent_rvc.py logic with routing/RAG/EXPAND)

**Step 4:** Commit

---

### Task 7: tts/base.py, tts/elevenlabs.py, tts/local.py — TTS abstraction

**Objective:** Abstract TTS. ElevenLabs = streaming WebSocket + fallback. Local = Kokoro + RVC.

**Files:**
- Create: `gmeet_pipeline/tts/base.py`
- Create: `gmeet_pipeline/tts/elevenlabs.py`
- Create: `gmeet_pipeline/tts/local.py`

**Step 1:** Define BaseTTS ABC with: async generate(text, bot_id) -> Optional[str]

**Step 2:** Implement ElevenLabsTTS (streaming + fallback from meeting_agent.py)

**Step 3:** Implement LocalTTS (Kokoro+RVC from meeting_agent_rvc.py, with MemorySnapshot dependency)

**Step 4:** Commit

---

### Task 8: memory.py — MemorySnapshot extraction

**Objective:** Extract MemorySnapshot class into standalone module

**Files:**
- Create: `gmeet_pipeline/memory.py`

**Step 1:** Port MemorySnapshot class from meeting_agent_rvc.py verbatim

**Step 2:** Commit

---

### Task 9: ws_manager.py + agent_page.py — UI extraction

**Objective:** Extract WebSocket manager and agent HTML into their own modules

**Files:**
- Create: `gmeet_pipeline/ws_manager.py`
- Create: `gmeet_pipeline/agent_page.py`

**Step 1:** Port ConnectionManager class

**Step 2:** Port both AGENT_HTML templates (simple + RVC) — decide which to keep or make configurable

**Step 3:** Commit

---

### Task 10: webhook.py — Recall webhook handler

**Objective:** Extract webhook processing into its own module

**Files:**
- Create: `gmeet_pipeline/webhook.py`

**Step 1:** Port recall_webhook logic as a class or function that takes config/registry/llm/tts deps

**Step 2:** Commit

---

### Task 11: server.py + main.py — FastAPI app and entry point

**Objective:** Wire everything together. server.py = routes only. main.py = startup + uvicorn.

**Files:**
- Create: `gmeet_pipeline/server.py`
- Create: `gmeet_pipeline/main.py`

**Step 1:** Create FastAPI app in server.py with all routes, injecting deps from config

**Step 2:** Create main.py that loads config, instantiates components, and runs uvicorn

**Step 3:** Commit

---

### Task 12: Update Dockerfile, requirements.txt, .env.example

**Objective:** Update infra files for new package structure

**Files:**
- Modify: `Dockerfile`
- Modify: `requirements.txt`
- Modify: `.env.example`

**Step 1:** Add pydantic-settings to requirements.txt

**Step 2:** Update Dockerfile to COPY gmeet_pipeline/ and use `python -m gmeet_pipeline.main`

**Step 3:** Update .env.example with all new config fields

**Step 4:** Commit

---

### Task 13: Keep legacy entry points as thin wrappers

**Objective:** meeting_agent.py and meeting_agent_rvc.py become thin wrappers that import from gmeet_pipeline

**Files:**
- Modify: `meeting_agent.py`
- Modify: `meeting_agent_rvc.py`

**Step 1:** Replace meeting_agent.py body with: config override for tts=elevenlabs, llm=simple, then call main()

**Step 2:** Replace meeting_agent_rvc.py body with: config override for tts=local, llm=voice_gateway, then call main()

**Step 3:** Commit

---

### Task 14: Final integration test

**Objective:** Verify the refactored package works end-to-end

**Step 1:** Run `python -m gmeet_pipeline.main --help` or verify import works

**Step 2:** Run `python meeting_agent.py` — verify it starts and loads config

**Step 3:** Run `python meeting_agent_rvc.py` — verify it starts

**Step 4:** Push branch, create PR
