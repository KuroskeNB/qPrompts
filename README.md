# qPrompts — Agent Eval Pipeline

A **config-driven, three-level evaluation framework** for AI agent system prompts.
Powered by the **Gemini API** (free tier compatible). No local GPU or Ollama required. - Gemini 3.1 Flash Lite Recommendation

```
L1 → fast regex/heuristic checks       (no LLM needed)
L2 → LLM-as-a-judge rubric scoring     (Gemini)
L3 → full multi-turn simulation         (Gemini)
```

---

## Quick Start

```bash
git clone https://github.com/KuroskeNB/qPrompts.git
cd qPrompts
pip install -r requirements.txt

# Copy the settings template and add your Gemini API key
cp settings.example.json settings.json

# Launch the web UI
python web.py          # opens http://localhost:7860 automatically
```

Get a free Gemini API key at https://aistudio.google.com/apikey — paste it in the **Settings** tab.

---

## Evaluation Levels

| Level | Name | What it does | LLM needed? |
|-------|------|-------------|-------------|
| **L1** | Syntactic & Heuristic | Regex, word/sentence counts, phrase matching | No |
| **L2** | LLM-as-a-Judge | Gemini scores each static response against semantic rubrics | Yes |
| **L3** | Multi-turn Simulation | Gemini plays a user persona for N turns; goal verified at end | Yes |

All levels are independent — run only what you need. L1 is instant and zero-cost.

---

## Web UI

```bash
python web.py
# → http://localhost:7860
```

### What you can do in the UI

#### Run Tab
| Feature | Description |
|---------|-------------|
| Gemini API key | Enter and save your key — persisted to `settings.json` |
| Prompt version | Pick any `.xml` file from the project's `prompts/` folder |
| Level selector | Run **All Levels**, **L1 only**, **L2 only**, or **L3 only** |
| ▶ Run Evals | Stream results live via SSE — cards appear as each case finishes |
| ✕ Cancel | Abort a running eval mid-stream (cancel-aware, stops between cases) |
| Live log panel | Real-time timestamped log of every case start and PASS/FAIL result |
| Copy Results | Copy the full run output as plain text (L3 turns include per-turn latency) |
| 📊 History | Jump to the run ledger for the current project |
| Run persistence | Eval keeps running even after a page refresh — reconnects automatically |

#### Manage Tab — Config
| Feature | Description |
|---------|-------------|
| Project name & description | Rename the project |
| L1 rules | Add / edit / delete regex, sentence, word, and phrase rules |
| L2 rubrics | Add / edit / delete semantic rubrics and the judge model |
| L3 settings | Set default turns, turn assertions, goal verification prompt |
| Save Config | Persist all changes to `project_config.yaml` |

#### Manage Tab — Cases
| Feature | Description |
|---------|-------------|
| L1 cases | Add / edit / delete static response test cases with expected failure lists |
| L2 cases | Add / edit / delete cases with user message, agent response, expected failures |
| L3 cases | Add / edit / delete multi-turn cases with persona, goal state, turns, opening line |
| Import / Export | Paste JSON from clipboard to bulk-import cases; export to clipboard |

#### Manage Tab — Prompts
| Feature | Description |
|---------|-------------|
| Create prompt | Add a new `.xml` prompt version |
| Edit prompt | Modify any existing prompt file inline |
| Delete prompt | Remove a prompt version |

#### History Tab
| Feature | Description |
|---------|-------------|
| Run ledger | Table of all past runs: timestamp, prompt file, model, L1/L2/L3 pass/fail counts |
| Delete entry | Remove any individual history row with the ✕ button |

#### Sidebar
| Feature | Description |
|---------|-------------|
| + New Project | Create a new project with a starter config |
| Project list | Switch between projects |
| Delete project | Remove a project and all its files |

---

## Gemini API

1. Get a free key at https://aistudio.google.com/apikey
2. Paste it in the **Settings** tab and click **Save**

Rate limiting is handled automatically — the client enforces ~12 RPM with interruptible sleeps and retries on HTTP 429/503 (up to 4 retries, 10s each).

### LangSmith Tracing (optional)

> **Not required.** Leave the API key blank and evals run exactly the same — tracing is silently skipped.

Every Gemini call can be traced to a [LangSmith](https://smith.langchain.com) project for token usage monitoring and prompt debugging. Configure in **Settings**:

| Field | Description |
|-------|-------------|
| LangSmith API Key | Personal or Service key from LangSmith |
| LangSmith Project | Project name (auto-resolved to session UUID) |
| Endpoint | `https://api.smith.langchain.com` (US) or `https://eu.api.smith.langchain.com` (EU) |

Click **Test LangSmith** to auto-resolve your tenant ID and project UUID and save them. Once configured, every `chat()` call posts a trace in a background thread without blocking evaluation.

---

## Directory Structure

```
qPrompts/
├── engine/
│   ├── config_loader.py      # YAML loader + jsonschema validation + prompt versioning
│   ├── level1_engine.py      # Syntactic rule executor (pure Python, no LLM)
│   ├── level2_engine.py      # LLM-as-a-judge rubric executor
│   ├── level3_engine.py      # Multi-turn conversation simulator
│   └── gemini_client.py      # stdlib-only Gemini REST client (urllib only)
├── projects/
│   └── my_bot/
│       ├── project_config.yaml       # rules, rubrics, L3 settings
│       ├── prompts/
│       │   ├── default.xml           # baseline prompt (required)
│       │   └── v2_strict.xml         # versioned prompt variant
│       ├── test_cases/
│       │   ├── level1_cases.json
│       │   ├── level2_cases.json
│       │   └── level3_cases.json
│       └── reports/
│           └── history_ledger.json   # all-time run history
├── web.py                # localhost web UI (FastAPI + SSE streaming)
├── runner.py             # CLI entry point
├── settings.example.json # copy to settings.json and add your keys
└── requirements.txt
```

---

## Creating a Project

### Via the Web UI (recommended)
1. Click **+ New Project** in the sidebar
2. Enter a name → starter config created automatically
3. **Manage → Config** — add L1 rules and L2 rubrics
4. **Manage → Prompts** — paste or write your agent's system prompt
5. **Manage → Cases** — add test cases for each level
6. **Run** → ▶ Run Evals

### Via the file system
```bash
mkdir -p projects/my_bot/test_cases projects/my_bot/prompts
cp projects/dental_booking_bot/project_config.yaml projects/my_bot/
# edit project_config.yaml, add JSON test cases and prompts/default.xml
python runner.py --project my_bot
```

See [CONFIGURATION.md](CONFIGURATION.md) for the full YAML reference.

---

## CLI Reference

```bash
python runner.py --project my_bot                          # all levels
python runner.py --project my_bot --level 1                # L1 only
python runner.py --project my_bot --prompt v2_strict.xml   # specific prompt
python runner.py --project my_bot --prompt v2.xml --level 2
```

Exit codes: `0` = all tests passed · `1` = one or more failures

---

## Further Reading

- [ARCHITECTURE.md](ARCHITECTURE.md) — engine internals, data flow, Gemini client, SSE
- [CONFIGURATION.md](CONFIGURATION.md) — full YAML reference for all rule and rubric types
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to add new rule types or extend the pipeline
