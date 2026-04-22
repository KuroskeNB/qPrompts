"""
engine/level2_engine.py
───────────────────────
Level 2 — Semantic / LLM-as-a-Judge executor.

For each rubric defined in project_config.yaml → level2.rubrics the engine
sends a structured prompt to a local Ollama model and parses the JSON verdict.

Rubric verdict format (from config):
    {"pass": true/false, "reason": "one sentence"}

The engine is tolerant of imperfect model output:
  - Attempts strict JSON parse first.
  - Falls back to regex extraction of pass/reason fields.
  - Marks as FAIL with a parse-error reason if extraction fails.

Usage:
    from engine.config_loader import load_config
    from engine.level2_engine import Level2Engine

    config = load_config(Path("projects/dental_booking_bot"))
    engine = Level2Engine(config)

    results = engine.run(
        user_message="I need to see a dentist on Monday.",
        agent_response="Let me check our schedule. We have Monday at nine AM available.",
    )
    for r in results:
        print(r.rubric_id, "PASS" if r.passed else f"FAIL — {r.reason}")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from threading import Event

from engine.gemini_client import GeminiClient, GeminiError


def _make_client(config: dict, **_):
    meta = config["meta"]
    return GeminiClient(
        api_key=meta.get("gemini_api_key", ""),
        model=meta.get("gemini_model", "gemini-3.1-flash-lite-preview"),
        langsmith_api_key=meta.get("langsmith_api_key", ""),
        langsmith_project=meta.get("langsmith_project", "voice-agent-evals"),
        langsmith_endpoint=meta.get("langsmith_endpoint", "https://api.smith.langchain.com"),
        langsmith_tenant_id=meta.get("langsmith_tenant_id", ""),
        langsmith_session_id=meta.get("langsmith_session_id", ""),
    )


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class RubricResult:
    rubric_id:   str
    name:        str
    passed:      bool
    reason:      str = ""
    raw_verdict: str = ""   # full model reply, useful for debugging

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


# ── Engine ─────────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """\
You are a strict voice-agent quality evaluator.
You will receive:
  1. An evaluation rubric (what to check).
  2. The user's message (what the human said).
  3. The agent's response (what the AI said).

Your job: decide whether the agent's response PASSES or FAILS the rubric.

Output ONLY valid JSON on a single line — no markdown, no explanation outside the JSON:
{"pass": true, "reason": "one sentence explaining your verdict"}
"""

_JUDGE_USER_TEMPLATE = """\
=== RUBRIC ===
{rubric_name}: {rubric_instruction}

=== USER MESSAGE ===
{user_message}

=== AGENT RESPONSE ===
{agent_response}

Evaluate the AGENT RESPONSE against the RUBRIC. Reply with JSON only.
"""


class Level2Engine:

    def __init__(self, config: dict) -> None:
        l2 = config["level2"]
        self._rubrics: list[dict] = l2["rubrics"]
        # Temporarily inject judge_model so factory can pick it up
        config["meta"]["_l2_judge_model"] = l2["judge_model"]
        self._client = _make_client(config, model_key="_l2_judge_model")
        self._temperature: float = l2.get("temperature", 0.0)

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        agent_response: str,
        user_message: str = "",
        cancel_event: Event | None = None,
    ) -> list[RubricResult]:
        """
        Evaluate `agent_response` against every rubric in the config.

        Parameters
        ----------
        agent_response : The AI agent's reply text to evaluate.
        user_message   : The human turn that preceded the agent reply.
        cancel_event   : If set(), stops evaluation early.
        """
        results: list[RubricResult] = []
        for rubric in self._rubrics:
            if cancel_event and cancel_event.is_set():
                break
            results.append(self._evaluate_rubric(rubric, user_message, agent_response, cancel_event))
        return results

    def is_available(self) -> bool:
        """Return True if the Ollama backend is reachable."""
        return self._client.is_available()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _evaluate_rubric(
        self,
        rubric: dict,
        user_message: str,
        agent_response: str,
        cancel_event: Event | None = None,
    ) -> RubricResult:
        rubric_id   = rubric["id"]
        rubric_name = rubric["name"]
        instruction = rubric["instruction"].strip()

        prompt = _JUDGE_USER_TEMPLATE.format(
            rubric_name=rubric_name,
            rubric_instruction=instruction,
            user_message=user_message or "(not provided)",
            agent_response=agent_response,
        )

        kwargs: dict = dict(
            messages=[{"role": "user", "content": prompt}],
            system=_JUDGE_SYSTEM,
            temperature=self._temperature,
        )
        if isinstance(self._client, GeminiClient):
            kwargs["cancel_event"] = cancel_event

        try:
            raw = self._client.chat(**kwargs)
        except GeminiError as exc:
            return RubricResult(
                rubric_id=rubric_id,
                name=rubric_name,
                passed=False,
                reason=f"LLM error: {exc}",
            )

        passed, reason = _parse_verdict(raw)
        return RubricResult(
            rubric_id=rubric_id,
            name=rubric_name,
            passed=passed,
            reason=reason,
            raw_verdict=raw,
        )


# ── Verdict parsing ────────────────────────────────────────────────────────────

def _parse_verdict(raw: str) -> tuple[bool, str]:
    """
    Parse the model's reply into (passed, reason).

    Strategy:
    1. Try strict json.loads on the full string.
    2. Try extracting the first {...} block via regex, then json.loads.
    3. Try regex extraction of pass/reason fields directly.
    4. Fall back to FAIL with a parse-error notice.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

    # Strategy 1 — strict parse
    try:
        data = json.loads(cleaned)
        return _extract_fields(data)
    except json.JSONDecodeError:
        pass

    # Strategy 2 — extract first {...} block
    m = re.search(r"\{[^{}]+\}", cleaned, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            return _extract_fields(data)
        except json.JSONDecodeError:
            pass

    # Strategy 3 — regex field extraction
    pass_m   = re.search(r'"pass"\s*:\s*(true|false)', cleaned, re.IGNORECASE)
    reason_m = re.search(r'"reason"\s*:\s*"([^"]+)"', cleaned)

    if pass_m:
        passed = pass_m.group(1).lower() == "true"
        reason = reason_m.group(1) if reason_m else "(reason not parsed)"
        return passed, reason

    # Strategy 4 — give up
    return False, f"[parse error] model output: {raw[:120]}"


def _extract_fields(data: dict) -> tuple[bool, str]:
    passed = bool(data.get("pass", False))
    reason = str(data.get("reason", "")).strip() or "(no reason given)"
    return passed, reason
