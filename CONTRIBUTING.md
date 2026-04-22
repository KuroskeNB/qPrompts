# Contributing

## Adding a New L1 Rule Type

L1 rule types are defined by handler methods on `Level1Engine` in `engine/level1_engine.py`.

### Step 1 — Register the type name

Add an entry to the `_HANDLERS` dispatch table:

```python
_HANDLERS: dict[str, str] = {
    "regex_forbidden":    "_check_regex_forbidden",
    "regex_required":     "_check_regex_required",
    "max_sentences":      "_check_max_sentences",
    "starts_with_phrase": "_check_starts_with_phrase",
    "max_words":          "_check_max_words",
    "my_new_type":        "_check_my_new_type",   # ← add here
}
```

### Step 2 — Implement the handler

```python
def _check_my_new_type(self, rule: dict, response: str) -> RuleResult:
    params = rule["params"]
    # ... your logic ...
    if failure_condition:
        return RuleResult(
            rule_id=rule["id"],
            description=rule["description"],
            passed=False,
            detail="Human-readable failure reason",
        )
    return _pass(rule)
```

### Step 3 — Add to the jsonschema (optional but recommended)

In `engine/config_loader.py`, add `"my_new_type"` to the `rule_type` enum so the validator catches typos in configs.

### Step 4 — Document it

Add a section to [CONFIGURATION.md](CONFIGURATION.md) under `level1 → Rule Types`.

---

## Adding a New L3 Turn Assertion

Turn assertions are evaluated in `_run_turn_assertions()` in `engine/level3_engine.py`.

```python
def _run_turn_assertions(self, agent_response: str) -> list[dict]:
    results = []
    for a in self._turn_assertions:
        atype = a.get("type", "")
        desc  = a.get("description", atype)
        params = a.get("params", {})

        if atype == "ends_with_question":
            passed, detail = _assert_ends_with_question(agent_response, params)
        elif atype == "my_new_assertion":                 # ← add branch
            passed, detail = _assert_my_new_assertion(agent_response, params)
        else:
            passed, detail = False, f"Unknown assertion type '{atype}'"

        results.append({"assertion": desc, "passed": passed, "detail": detail})
    return results
```

Add the implementation as a module-level function:

```python
def _assert_my_new_assertion(response: str, params: dict) -> tuple[bool, str]:
    # ... your logic ...
    return True, "Passed"   # or False, "Reason for failure"
```

---

## Adding a New LLM Provider

1. Create `engine/my_provider_client.py` with `MyProviderClient` implementing:
   ```python
   def chat(self, messages, system=None, temperature=0.0, cancel_event=None) -> str: ...
   def is_available(self) -> bool: ...
   ```
2. Add `MyProviderError` exception class.
3. Update `_make_client()` in both `level2_engine.py` and `level3_engine.py`:
   ```python
   if provider == "my_provider":
       return MyProviderClient(...)
   ```
4. Add `"my_provider"` as an option in the provider dropdown in `web.py`.
5. If your client uses sleeps (rate limiting, retries), use `interruptible_sleep()` from `gemini_client.py` as a reference so Cancel works.

---

## Project Structure Conventions

- All file I/O stays out of the engine layer — engines receive config dicts and return result dataclasses
- Test case JSON files are loaded and saved by `web.py` endpoints, not by the engines
- New API endpoints in `web.py` follow the pattern: `@app.get/post/put/delete("/projects/{name}/...")`
- All HTML/CSS/JS stays in `web.py` as embedded strings — no separate template or static files
- Settings that survive server restarts go in `settings.json` via `_load_settings()` / `_save_settings()`

---

## Running Tests (manual)

There is no automated test suite. Testing is done by running the eval pipeline against the sample projects:

```bash
python web.py
# → http://localhost:7860
# Select "dental_booking_bot" or "DiegoVoice" → Run Evals → All Levels
```

For L1 only (no LLM needed):
```bash
python runner.py --project dental_booking_bot --level 1
```
