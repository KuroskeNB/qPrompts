# Architecture

## Overview

The pipeline has two entry points — a CLI (`runner.py`) and a web UI (`web.py`) — that share the same engine layer. No business logic lives in the entry points; they only handle I/O and presentation.

```
┌──────────────────────────────────────────────────────────────┐
│                        Entry Points                          │
│                                                              │
│   runner.py (CLI)              web.py (FastAPI + SSE)        │
└──────────────────┬──────────────────────┬────────────────────┘
                   │                      │
                   ▼                      ▼
┌──────────────────────────────────────────────────────────────┐
│                        Engine Layer                          │
│                                                              │
│  config_loader.py  → loads & validates project_config.yaml   │
│  level1_engine.py  → regex / heuristic rule executor         │
│  level2_engine.py  → LLM-as-a-judge rubric executor          │
│  level3_engine.py  → multi-turn conversation simulator       │
│  ollama_client.py  → stdlib-only Ollama REST client          │
│  gemini_client.py  → stdlib-only Gemini REST client          │
└──────────────────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│                      Project Layer                           │
│                                                              │
│  projects/<name>/project_config.yaml  (rules + rubrics)     │
│  projects/<name>/prompts/*.xml        (system prompts)       │
│  projects/<name>/test_cases/*.json    (test cases)           │
│  projects/<name>/reports/             (run history)          │
└──────────────────────────────────────────────────────────────┘
```

---

## LLM Provider Abstraction

Both `OllamaClient` and `GeminiClient` expose the same duck-typed interface:

```python
client.chat(messages, system=None, temperature=0.0) -> str
client.is_available() -> bool
```

The factory `_make_client(config)` in `level2_engine.py` and `level3_engine.py` dispatches to the correct client based on `config["meta"]["provider"]`:

```python
if provider == "gemini":
    return GeminiClient(api_key=..., model=...)
else:
    return OllamaClient(base_url=..., model=...)
```

This means the engine layer is provider-agnostic — swapping providers requires zero engine changes.

---

## Gemini Client (`gemini_client.py`)

Stdlib-only (`urllib`, `json`, `threading`). No third-party dependencies.

- Converts OpenAI-style `messages` to Gemini format (`"assistant"` → `"model"`)
- Injects `system` as `systemInstruction` in the payload
- **Rate limiting**: sleeps `60 / rpm_limit` seconds after every successful call (default 12 RPM, safely under free-tier 15 RPM). Sleep is interruptible via `cancel_event`.
- **Retry on 429/503**: up to 4 retries with 10s sleep between attempts. Sleep is interruptible.
- `cancel_event` parameter (optional `threading.Event`): if set during any sleep, raises `GeminiError("cancelled")` immediately
- **`last_api_latency_ms`**: instance attribute set after each successful HTTP response, before the RPM sleep. Captures pure Gemini API response time (network + model inference only). Read by the L3 engine to populate per-turn latency badges.

```python
client = GeminiClient(api_key="...", model="gemini-2.0-flash-lite")
reply = client.chat(messages, system="...", cancel_event=my_event)
api_ms = client.last_api_latency_ms  # excludes RPM throttle sleep
```

### LangSmith Integration

`GeminiClient` optionally posts a trace to LangSmith after every successful call (fire-and-forget background thread, `daemon=False`). Configure via constructor parameters:

| Parameter | Description |
|-----------|-------------|
| `langsmith_api_key` | LangSmith personal or service API key |
| `langsmith_project` | Display project name (fallback if `session_id` not set) |
| `langsmith_endpoint` | `https://api.smith.langchain.com` (US) or EU variant |
| `langsmith_tenant_id` | Tenant UUID — sent as `X-Tenant-ID` header |
| `langsmith_session_id` | Project UUID — routes runs to the correct project (overrides `project_name`) |

The run payload includes: inputs (messages + system), outputs (reply text), start/end timestamps, `dotted_order` (`YYYYMMDDTHHMMSSffffffZ{uuid}` format required by LangSmith), token usage from `usageMetadata`, and latency. Trace failures are logged as warnings and never raise.

---

## Config Loading (`config_loader.py`)

Config loading is a two-step process:

### Step 1 — `load_config(project_path)`
- Reads `project_config.yaml`
- Validates against a jsonschema (Draft 7) enforcing required fields, types, and enum values
- Checks for duplicate IDs in `level1.rules` and `level2.rubrics`
- Returns the validated config dict

### Step 2 — `load_prompt(config, project_path, filename)`
- Reads the requested `.xml` file from `prompts/`
- Injects content into `config["meta"]["system_prompt"]`, overriding any inline value
- Falls back to `default.xml` if the requested file is not found (with a warning)

This separation means `project_config.yaml` owns all structural settings (rules, rubrics, thresholds) and prompt files own the natural language instructions. Prompt iteration never requires a config change.

---

## Level 1 Engine

**No LLM involvement.** All checks are deterministic and instant.

```
for each rule in config.level1.rules:
    dispatch to handler by rule.type
    return RuleResult(passed, detail, excerpt)
```

Rule handlers:

| Handler | Logic |
|---------|-------|
| `regex_forbidden` | `re.search(pattern, response)` → FAIL if match found |
| `regex_required` | `re.search(pattern, response)` → FAIL if no match |
| `max_sentences` | Split by `[.!?]+`, strip, filter empty strings, count |
| `max_words` | `len(response.split())` |
| `starts_with_phrase` | Check first N words for any listed phrase (case-insensitive by default) |

`max_sentences` logic:
1. `re.split(r'[.!?]+', response)` — split on sentence-ending punctuation
2. `.strip()` each fragment
3. Filter empty strings
4. `len()` of result = sentence count

---

## Level 2 Engine

**One LLM call per rubric per test case.**

```
for each rubric:
    build prompt = JUDGE_SYSTEM + rubric instruction + user_message + agent_response
    call client.chat(temperature=0.0)
    parse JSON verdict → (passed, reason)
```

The judge system prompt instructs the model to output `{"pass": true/false, "reason": "..."}`. Parsing is fault-tolerant with four fallback strategies:

1. Strict `json.loads` on the full reply
2. Regex extract first `{...}` block, then `json.loads`
3. Regex field extraction for `"pass"` and `"reason"` keys
4. Fall back to `FAIL` with a parse-error note (never crashes)

Temperature is always `0.0` for the judge role to keep verdicts deterministic.

`cancel_event` is threaded from `web.py → Level2Engine.run() → _evaluate_rubric() → client.chat()`, so Cancel aborts mid-sleep on every retry.

---

## Level 3 Engine

**Three distinct LLM roles, all served by the configured provider.**

```
┌──────────────────────────────────────────────────┐
│  User Simulator   │ temp=0.7  │ persona + goal    │
│  Agent Under Test │ temp=0.3  │ project prompt    │
│  Goal Verifier    │ temp=0.0  │ fixed evaluator   │
└──────────────────────────────────────────────────┘
```

Turn loop:

```
generate or use opening_line → user_msg

for turn in 1..N:
    agent_reply = agent_client.chat(history, system=project_prompt, temp=0.3)
    latency_ms  = agent_client.last_api_latency_ms   ← pure API time, no RPM sleep
    check turn_assertions on agent_reply
    if turn < N:
        user_msg = user_sim.chat(sim_history, system=persona, temp=0.7)

transcript = format all turns
goal_achieved, reason = goal_verifier.chat(transcript, goal_state, temp=0.0)
```

### Per-Turn Latency

`TurnResult.latency_ms` stores the pure Gemini API response time for each agent turn — measured from the moment the HTTP request is sent to the moment the response is received, **excluding** the mandatory RPM throttle sleep. This is exposed in the UI as an `⏱ X.Xs` badge between the USER and AGENT bubbles, and included in Copy Results output.

> The RPM sleep (5s at 12 RPM) and retry sleeps (10s per 429) are excluded from the badge intentionally — they are infrastructure overhead, not model latency.

### Turn Assertions

Currently one type: `ends_with_question`.

Checks that the agent's response ends with `?` and contains at most `max_questions` question marks. Includes an **exception list** for terminal phrases — if the response contains any of `hold`, `transfer`, `connect`, `goodbye`, `conclude`, or `moment`, the assertion auto-passes (closing question not appropriate for call-end flows).

### Pass Condition

An L3 case **passes** if `goal_achieved = true` and no error occurred. Per-turn assertion failures (`✗`) are displayed as informational but do not block the PASS verdict — goal achievement is the primary signal.

### Cancel Support

`cancel_event` threads through `run() → _simulate_user() → _verify_goal() → client.chat()`, making all Gemini sleeps (rate-limit and retry) interruptible.

---

## Web UI (`web.py`)

Built with FastAPI. All HTML/CSS/JS is embedded as a single string — no build step, no node_modules, no separate template files. One file to run.

### SSE Streaming

The `/run` endpoint is a `StreamingResponse` with `media_type="text/event-stream"`. Events are sent as each case completes:

```
data: {"type": "meta", "provider": "gemini", "model": "...", "prompt": "..."}\n\n
data: {"type": "l1_case", "case_id": "...", "passed": true, "rules": [...]}\n\n
data: {"type": "l2_thinking", "case_id": "..."}\n\n
data: {"type": "l2_case", "case_id": "...", "rubrics": [...]}\n\n
data: {"type": "l3_thinking", "case_id": "...", "turns": 4}\n\n
data: {"type": "l3_case", "case_id": "...", "goal_achieved": true, "turns": [...]}\n\n
data: {"type": "log", "msg": "L2 running: l2_fail_tone"}\n\n
data: {"type": "cancelled"}\n\n
data: {"type": "done"}\n\n
```

### Cancel Mechanism

`_cancel_event` is a module-level `threading.Event`.

- `POST /cancel` — sets the event
- Stream generator checks it between every case (`if _cancel_event.is_set(): yield cancelled; return`)
- `GeminiClient.chat()` checks it every 0.5s during all sleeps via `interruptible_sleep()`
- Event is cleared at the start of each new run

### Settings Persistence

`settings.json` at the project root stores Gemini and LangSmith credentials. Loaded on every `/run` call — keys only need to be entered once.

```json
{
  "gemini_api_key": "...",
  "gemini_model": "gemini-2.0-flash-lite",
  "langsmith_api_key": "...",
  "langsmith_project": "MyProject",
  "langsmith_endpoint": "https://eu.api.smith.langchain.com",
  "langsmith_tenant_id": "<uuid>",
  "langsmith_session_id": "<project-uuid>"
}
```

`langsmith_tenant_id` and `langsmith_session_id` are auto-resolved by the **Test LangSmith** button (queries `GET /sessions` on the LangSmith API and matches by project name).

### Run Persistence

Every SSE event emitted during a run is collected into a `run_events` list. After the stream completes, the list is written to `last_run.json` at the project root. On page load, the UI fetches `GET /last-run` and replays all events through the same card renderers — results survive a browser refresh without re-running the eval.

### History Ledger

After every run, an entry is appended to `reports/history_ledger.json`:

```json
{
  "timestamp": "2024-01-15T10:30:00+00:00",
  "prompt_file": "v2_strict.xml",
  "prompt_stem": "v2_strict",
  "model": "gemini-2.0-flash-lite",
  "provider": "gemini",
  "L1": {"passed": 4, "failed": 1, "total": 5},
  "L2": {"passed": 3, "failed": 0, "total": 3},
  "L3": {"passed": 2, "failed": 1, "total": 3}
}
```

Individual entries can be deleted via `DELETE /projects/{name}/ledger/{index}`.

---

## Design Decisions

**No hardcoded domain knowledge** — the engine has zero opinion about any specific domain. All phrases, patterns, thresholds, and rubrics live in the project config.

**Config-first** — adding a new rule requires only a YAML edit, not a code change.

**Provider-agnostic engines** — L2 and L3 engines call `client.chat()` without knowing whether they're talking to Ollama or Gemini.

**Fail-open parsing** — L2 verdict parsing never crashes. Badly formatted model output still produces a result rather than aborting the run.

**Cancel-safe sleeps** — all `time.sleep()` calls in the Gemini client use `interruptible_sleep()` which polls `cancel_event` every 0.5s. Cancel is never delayed more than 0.5s regardless of retry/rate-limit sleep duration.

**Single file UI** — `web.py` has no external templates or asset files. Trivial to copy, move, or share.

**Latency = API only** — `last_api_latency_ms` is captured immediately after the HTTP response, before the RPM sleep. This separates model inference time from infrastructure throttling, giving a true signal for prompt/model latency comparison.

**LangSmith non-blocking** — traces are posted in a `daemon=False` background thread. The eval run never waits for the trace to land; a failed trace logs a warning and is discarded. Setting `daemon=False` ensures the thread completes before the Python process exits.
