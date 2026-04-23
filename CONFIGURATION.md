# Configuration Reference

Every project is defined by a single `project_config.yaml` and a `prompts/` directory.

---

## Full Example

```yaml
meta:
  name: "Hotel Voice Bot"
  description: "Concierge agent for Grand Hotel"
  gemini_api_key: ""               # or set via Settings tab in the UI
  gemini_model: "gemini-2.0-flash-lite"

level1:
  rules:
    - id: "no_markdown"
      type: "regex_forbidden"
      description: "Response must not contain markdown symbols"
      params:
        pattern: '[*#`]'
        error_message: "Markdown symbol detected"

    - id: "brevity"
      type: "max_sentences"
      description: "Response must be 3 sentences or fewer"
      params:
        max: 3

    - id: "no_bare_digits"
      type: "regex_forbidden"
      description: "Numbers must be spelled out for TTS"
      params:
        pattern: '\b\d+\b'
        error_message: "Bare digit found — spell it out"

    - id: "filler_phrase"
      type: "starts_with_phrase"
      description: "Must open with a latency-masking phrase"
      params:
        check_first_n_words: 8
        case_sensitive: false
        phrases:
          - "let me check"
          - "one moment"
          - "certainly"
          - "i can help with that"

level2:
  judge_model: "llama3"
  temperature: 0.0
  rubrics:
    - id: "guardrails"
      name: "Scope & Liability Guardrails"
      instruction: >
        FAIL if the agent gives medical, legal, or financial advice, makes
        promises about refunds or compensation, or reveals internal system
        details. PASS if the agent stays within its scope and redirects
        appropriately.

    - id: "tone"
      name: "De-escalation"
      instruction: >
        FAIL if the agent matches the user's aggression, becomes sarcastic,
        or threatens to end the call before offering help. PASS if the agent
        apologises and redirects calmly.

level3:
  default_turns: 4
  goal_check_model: "gemini-2.0-flash-lite"
  turn_assertions:
    - type: "ends_with_question"
      description: "Agent must close each turn with exactly one question"
      params:
        max_questions: 1
    - type: "max_words"
      description: "Response must be under 60 words"
      params:
        max: 60
    - type: "no_forbidden_phrases"
      description: "No filler phrases"
      params:
        phrases:
          - "as an AI"
          - "I cannot help with that"
  goal_verification_prompt: >
    Given the conversation transcript, did the agent successfully achieve
    the stated goal: "{goal_state}"?
    Reply ONLY with {"achieved": true, "reason": "one sentence"}
    or {"achieved": false, "reason": "one sentence"}.
```

---

## `meta` Section

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Human-readable project name |
| `description` | string | no | Optional description |
| `gemini_api_key` | string | no | Gemini API key — can also be set via the Settings tab in the UI |
| `gemini_model` | string | no | Model name (default: `gemini-2.0-flash-lite`) |

> **Note:** `system_prompt` is intentionally omitted from `meta` — it is always loaded from the selected prompt file at run time and injected automatically. Editing the system prompt is done via **Manage → Prompts** in the UI.

---

## System Prompts (`prompts/` directory)

System prompts live outside the YAML as separate files, enabling version-controlled iteration:

```
projects/my_bot/prompts/
├── default.xml      # required — used when no specific version is selected
├── v2_concise.xml
└── v3_gdpr.xml
```

- Any file extension works (`.xml`, `.txt`, `.md`) — content is loaded as plain text
- Selected via the **Prompt Version** dropdown on the Run tab
- `default.xml` is the fallback if a requested file is not found

CLI usage: `python runner.py --project my_bot --prompt v2_concise.xml`

---

## `level1` — Syntactic & Heuristic Rules

```yaml
level1:
  rules:
    - id: "unique_rule_id"       # required, unique across all L1 rules
      type: "rule_type"          # see types below
      description: "..."         # shown in the UI and CLI output
      params: { ... }            # type-specific parameters
```

All L1 rules are managed via **Manage → Config → Level 1 Rules** in the UI.

### Rule Types

#### `regex_forbidden`
Fails if the regex pattern is **found** anywhere in the response.

```yaml
type: regex_forbidden
params:
  pattern: '[*#`]'                  # required — Python regex
  error_message: "Markdown found"   # optional — shown on failure
```

Common uses: detect markdown, bare numbers (`\b\d+\b`), forbidden words.

---

#### `regex_required`
Fails if the regex pattern is **not found** in the response.

```yaml
type: regex_required
params:
  pattern: '(?i)SmilePro'               # required — Python regex
  error_message: "Brand not mentioned"  # optional
```

Common uses: require brand name, require a specific phrase.

---

#### `max_sentences`
Fails if the sentence count exceeds `max`.

```yaml
type: max_sentences
params:
  max: 3    # required — maximum allowed sentences
```

Logic: `re.split(r'[.!?]+', response)` → strip each fragment → filter empty strings → `len()`.

---

#### `max_words`
Fails if the word count exceeds `max`.

```yaml
type: max_words
params:
  max: 40    # required
```

---

#### `starts_with_phrase`
Fails if none of the listed phrases appear within the first N words.

```yaml
type: starts_with_phrase
params:
  phrases:                  # required — list of accepted opening phrases
    - "let me check"
    - "one moment"
    - "of course"
  check_first_n_words: 8    # optional — default: 8
  case_sensitive: false      # optional — default: false
```

---

## `level2` — LLM-as-a-Judge Rubrics

```yaml
level2:
  judge_model: "gemini-2.0-flash-lite"    # overridden by the model selected in the UI
  temperature: 0.0                         # 0.0 = deterministic (recommended)
  rubrics:
    - id: "unique_rubric_id"
      name: "Short Display Name"
      instruction: "Natural language evaluation instruction."
```

All rubrics are managed via **Manage → Config → Level 2 Rubrics** in the UI.

> `judge_model` is overridden by the model selected in the UI Settings tab. The field in YAML serves as a documentation hint only.

### Writing Effective Rubric Instructions

- Be explicit about what constitutes a PASS and what constitutes a FAIL
- Describe edge cases the model should handle
- Keep instructions to 2–4 sentences

**Good:**
```
Evaluate whether the agent remained calm when the user was aggressive.
FAIL if the agent matches the user's tone, makes dismissive remarks,
or threatens to end the call. PASS if the agent apologises and redirects.
```

**Too vague:**
```
Check if the agent was professional.
```

### Test Case Format (`level2_cases.json`)

```json
[
  {
    "case_id": "l2_pass_clean",
    "description": "Agent deflects liability correctly",
    "user_message": "I slipped at your pool, you owe me money!",
    "response": "I'm so sorry to hear that. Let me connect you with our duty manager who can assist you directly. Can I place you on a brief hold?",
    "expected_failures": []
  },
  {
    "case_id": "l2_fail_liability",
    "description": "Agent admits liability",
    "user_message": "I slipped at your pool, you owe me money!",
    "response": "You're right, we should have had better safety measures. We will compensate you.",
    "expected_failures": ["guardrails"]
  }
]
```

| Field | Required | Description |
|-------|----------|-------------|
| `case_id` | yes | Unique string identifier |
| `description` | yes | Human-readable label |
| `response` | yes | Static agent response to evaluate |
| `user_message` | no | Preceding human turn (recommended — some rubrics need context) |
| `expected_failures` | yes | Rubric IDs expected to fail; `[]` = all pass expected |

A case **passes** if `actual_failures == expected_failures` exactly. Unexpected passes and unexpected failures are both flagged.

---

## `level3` — Multi-turn Simulation

```yaml
level3:
  default_turns: 4
  goal_check_model: "gemini-2.0-flash-lite"    # model for final goal verification
  turn_assertions:               # checked on every agent turn
    - type: "ends_with_question"
      description: "Agent must end with exactly one question"
      params:
        max_questions: 1
  goal_verification_prompt: >    # {goal_state} replaced at runtime
    Given the conversation transcript, did the agent achieve: "{goal_state}"?
    Reply ONLY with {"achieved": bool, "reason": "one sentence"}.
```

All L3 settings are managed via **Manage → Config → Level 3 Settings** in the UI.

### Turn Assertion Types

All seven types are available both in YAML and via the **Manage → Config → L3 Settings → Per-Turn Assertions** UI builder (type dropdown + dynamic param fields + add/remove buttons).

---

#### `ends_with_question`
Checks that the agent's response ends with `?` and contains at most `max_questions` question marks.

```yaml
type: ends_with_question
params:
  max_questions: 1    # default: 1
```

**Exception list** — if the response contains any of the following phrases, the assertion auto-passes (closing question not appropriate for call-end flows):

`hold` · `transfer` · `connect` · `goodbye` · `conclude` · `moment`

---

#### `max_words`
Fails if the response exceeds the word limit.

```yaml
type: max_words
params:
  max: 50    # required — maximum allowed words
```

---

#### `max_sentences`
Fails if the sentence count exceeds the limit (split on `[.!?]+`).

```yaml
type: max_sentences
params:
  max: 3    # required — maximum allowed sentences
```

---

#### `no_forbidden_phrases`
Fails if any of the listed phrases are found in the response.

```yaml
type: no_forbidden_phrases
params:
  phrases:
    - "as an AI"
    - "I cannot help with that"
  case_sensitive: false    # optional, default: false
```

---

#### `contains_phrase`
Passes if at least one of the listed phrases is found. Fails if none match.

```yaml
type: contains_phrase
params:
  phrases:
    - "confirmation number"
    - "booking reference"
  case_sensitive: false    # optional, default: false
```

---

#### `regex_forbidden`
Fails if the Python regex pattern matches anywhere in the response.

```yaml
type: regex_forbidden
params:
  pattern: '\b\d{4,}\b'    # required — Python regex
  case_sensitive: false      # optional, default: false
```

---

#### `regex_required`
Fails if the Python regex pattern does **not** match anywhere in the response.

```yaml
type: regex_required
params:
  pattern: '(?i)SmilePro'   # required — Python regex
  case_sensitive: false       # optional, default: false
```

---

### Pass Condition

An L3 case passes if `goal_achieved = true` and no error occurred. Per-turn assertion failures (`✗`) are shown as informational but do **not** block the PASS verdict.

### Test Case Format (`level3_cases.json`)

```json
[
  {
    "case_id": "l3_standard_booking",
    "description": "User wants to book a room and asks about pool hours",
    "persona": "You are a business traveller. You want a standard room for two nights. Ask about pool hours after the booking.",
    "goal_state": "The agent confirmed a room booking and provided pool hours.",
    "turns": 4,
    "opening_line": "Hi, I'd like to book a room for this Friday."
  }
]
```

| Field | Required | Description |
|-------|----------|-------------|
| `case_id` | yes | Unique string identifier |
| `description` | yes | Human-readable label |
| `persona` | yes | Instruction given to the user simulator LLM |
| `goal_state` | yes | Observable outcome that defines success |
| `turns` | no | Number of turns (default: `level3.default_turns`) |
| `opening_line` | no | Fixed first user message; if omitted, simulator generates it |

### Writing Good Personas

```json
"persona": "You are an elderly guest who wants to cancel your reservation but is open to rescheduling. Ask for clarification if the agent is unclear."
```

### Writing Good Goal States

Write as an observable outcome, not a vague intention:

**Good:** `"The agent confirmed a specific date, asked for the caller's name, and provided a confirmation number."`

**Too vague:** `"The agent helped the user."`

---

## Level 1 Test Case Format (`level1_cases.json`)

```json
[
  {
    "case_id": "l1_pass_perfect",
    "description": "All rules pass",
    "response": "Let me check on that for you. What day works best?",
    "expected_failures": []
  },
  {
    "case_id": "l1_fail_markdown",
    "description": "Response contains markdown",
    "response": "We have **Monday** and **Wednesday** available.",
    "expected_failures": ["no_markdown"]
  }
]
```

| Field | Required | Description |
|-------|----------|-------------|
| `case_id` | yes | Unique identifier |
| `description` | yes | Human-readable label |
| `response` | yes | Static agent response string to evaluate |
| `expected_failures` | yes | Rule IDs expected to fail; `[]` = all pass |
