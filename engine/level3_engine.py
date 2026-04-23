"""
engine/level3_engine.py
───────────────────────
Level 3 — Multi-turn Conversation Simulator.

Simulates a realistic conversation between a synthetic user persona and the
agent under test, then asserts structural rules on each turn and verifies
whether the stated goal was achieved at the end.

How it works
────────────
1. A "user simulator" LLM (same Ollama model) plays the caller role, driven
   by a persona and a goal from the test case.
2. The "agent" LLM (the same model, but with the project system prompt) plays
   the voice agent role.
3. After the configured number of turns, a "goal verifier" LLM checks whether
   the conversation achieved the stated goal.
4. Each agent turn is also checked against the turn_assertions from config
   (e.g. "ends_with_question").

Level 3 test case schema (level3_cases.json):
{
  "case_id": "l3_full_booking",
  "description": "User successfully books a Monday 9 AM appointment",
  "persona": "You are a busy professional. You want to book a dental cleaning on Monday morning. Be brief and direct.",
  "goal_state": "The agent confirmed a specific appointment time and asked for the user's name or contact info.",
  "system_prompt": "You are SmilePro, a friendly dental booking voice agent...",
  "turns": 4
}

Usage:
    engine = Level3Engine(config)
    result = engine.run(case)
    print(result.goal_achieved, result.transcript)
"""

from __future__ import annotations

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


_LLMError = (GeminiError,)


# ── Result dataclasses ─────────────────────────────────────────────────────────

@dataclass
class TurnResult:
    turn:             int
    user_message:     str
    agent_response:   str
    assertions:       list[dict]   # [{"assertion": str, "passed": bool, "detail": str}]
    latency_ms:       int = 0      # agent response time (excludes RPM sleep)

    @property
    def all_assertions_passed(self) -> bool:
        return all(a["passed"] for a in self.assertions)


@dataclass
class Level3Result:
    case_id:        str
    goal_achieved:  bool
    goal_reason:    str
    turns:          list[TurnResult] = field(default_factory=list)
    error:          str = ""

    @property
    def all_turns_passed(self) -> bool:
        return all(t.all_assertions_passed for t in self.turns)

    @property
    def passed(self) -> bool:
        return self.goal_achieved and not self.error

    def transcript_text(self) -> str:
        lines: list[str] = []
        for t in self.turns:
            lines.append(f"[Turn {t.turn}]")
            lines.append(f"  USER:  {t.user_message}")
            lines.append(f"  AGENT: {t.agent_response}")
        return "\n".join(lines)


# ── Engine ─────────────────────────────────────────────────────────────────────

_USER_SIM_SYSTEM = """\
You are playing the role of a caller in a phone conversation with a voice assistant.
Stay strictly in character — speak as the persona describes.
Keep your replies short (1-2 sentences). Do NOT play the agent's role.
Do NOT explain your actions. Just say what your character would say next.
"""

_GOAL_CHECK_SYSTEM = """\
You are an objective evaluator. Read the conversation transcript and decide
whether the stated goal was achieved. Reply ONLY with valid JSON:
{"achieved": true, "reason": "one sentence"}
or
{"achieved": false, "reason": "one sentence"}
No markdown, no explanation outside the JSON.
"""


class Level3Engine:

    def __init__(self, config: dict) -> None:
        l3 = config["level3"]
        meta = config["meta"]

        self._default_turns: int       = l3.get("default_turns", 4)
        self._turn_assertions: list    = l3.get("turn_assertions", [])
        self._goal_prompt_template: str = l3.get("goal_verification_prompt", "")
        self._goal_model: str          = l3.get("goal_check_model", meta["ollama_model"])

        self._global_system_prompt: str = meta.get("system_prompt", "").strip()

        self._agent_client = _make_client(config, model_key="ollama_model")
        config["meta"]["_goal_model"] = self._goal_model
        self._goal_client  = _make_client(config, model_key="_goal_model")

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, case: dict, cancel_event: Event | None = None) -> Level3Result:
        """
        Simulate a full multi-turn conversation for one test case.

        Parameters
        ----------
        case : dict loaded from level3_cases.json

        Returns
        -------
        Level3Result — full transcript, per-turn assertion results, goal verdict.
        """
        case_id      = case["case_id"]
        persona      = case["persona"]
        goal_state   = case["goal_state"]
        system_prompt = (case.get("system_prompt") or "").strip() or self._global_system_prompt or "You are a helpful voice assistant."
        num_turns    = case.get("turns", self._default_turns)
        opening_line = case.get("opening_line", "")

        history: list[dict[str, str]] = []  # OpenAI-style messages (agent's view)
        sim_history: list[dict[str, str]] = []  # user simulator's view

        turn_results: list[TurnResult] = []

        # ── Generate the first user message ────────────────────────────────────
        if opening_line:
            user_msg = opening_line
        else:
            try:
                user_msg = self._simulate_user(
                    persona=persona,
                    goal_state=goal_state,
                    sim_history=sim_history,
                    is_first=True,
                    cancel_event=cancel_event,
                )
            except _LLMError as exc:
                return Level3Result(case_id=case_id, goal_achieved=False,
                                    goal_reason="", error=str(exc))

        for turn_n in range(1, num_turns + 1):

            # ── Agent turn ─────────────────────────────────────────────────────
            history.append({"role": "user", "content": user_msg})
            sim_history.append({"role": "assistant", "content": user_msg})

            try:
                agent_reply = self._agent_client.chat(
                    messages=history,
                    system=system_prompt,
                    temperature=0.3,
                    **self._gkw(self._agent_client, cancel_event),
                )
            except _LLMError as exc:
                return Level3Result(case_id=case_id, goal_achieved=False,
                                    goal_reason="", error=str(exc))
            latency_ms = getattr(self._agent_client, "last_api_latency_ms", 0)

            history.append({"role": "assistant", "content": agent_reply})
            sim_history.append({"role": "user", "content": agent_reply})

            # ── Per-turn assertions ────────────────────────────────────────────
            assertions = _check_turn_assertions(agent_reply, self._turn_assertions)
            turn_results.append(TurnResult(
                turn=turn_n,
                user_message=user_msg,
                agent_response=agent_reply,
                assertions=assertions,
                latency_ms=latency_ms,
            ))

            if turn_n == num_turns:
                break

            # ── Next user message ──────────────────────────────────────────────
            try:
                user_msg = self._simulate_user(
                    persona=persona,
                    goal_state=goal_state,
                    sim_history=sim_history,
                    is_first=False,
                    cancel_event=cancel_event,
                )
            except _LLMError as exc:
                return Level3Result(case_id=case_id, goal_achieved=False,
                                    goal_reason="", error=str(exc),
                                    turns=turn_results)

        # ── Goal verification ──────────────────────────────────────────────────
        transcript = _format_transcript(turn_results)
        goal_achieved, goal_reason = self._verify_goal(transcript, goal_state, cancel_event)

        return Level3Result(
            case_id=case_id,
            goal_achieved=goal_achieved,
            goal_reason=goal_reason,
            turns=turn_results,
        )

    def is_available(self) -> bool:
        return self._agent_client.is_available()



    # ── Internal ───────────────────────────────────────────────────────────────

    def _gkw(self, client, cancel_event: Event | None) -> dict:
        """Extra kwargs to pass to chat() when client is Gemini."""
        return {"cancel_event": cancel_event} if isinstance(client, GeminiClient) else {}

    def _simulate_user(
        self,
        persona: str,
        goal_state: str,
        sim_history: list[dict],
        is_first: bool,
        cancel_event: Event | None = None,
    ) -> str:
        if is_first:
            prompt = (
                f"Your goal for this call: {goal_state}\n\n"
                "Start the conversation with your opening line."
            )
        else:
            prompt = (
                f"Your goal for this call: {goal_state}\n\n"
                "Continue the conversation. Respond to the agent's last message."
            )

        system = f"{_USER_SIM_SYSTEM}\n\nYour persona: {persona}"
        messages = sim_history + [{"role": "user", "content": prompt}]

        return self._agent_client.chat(
            messages=messages,
            system=system,
            temperature=0.7,
            **self._gkw(self._agent_client, cancel_event),
        )

    def _verify_goal(self, transcript: str, goal_state: str, cancel_event: Event | None = None) -> tuple[bool, str]:
        prompt_template = self._goal_prompt_template or (
            'Given the conversation transcript, did the agent successfully achieve '
            'the stated goal: "{goal_state}"? Reply ONLY with '
            '{"achieved": bool, "reason": "one sentence"}.'
        )
        prompt = prompt_template.replace("{goal_state}", goal_state)
        prompt += f"\n\n=== TRANSCRIPT ===\n{transcript}"

        try:
            raw = self._goal_client.chat(
                messages=[{"role": "user", "content": prompt}],
                system=_GOAL_CHECK_SYSTEM,
                temperature=0.0,
                **self._gkw(self._goal_client, cancel_event),
            )
        except _LLMError as exc:
            return False, f"Goal check failed: {exc}"

        import json, re as _re
        cleaned = _re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            data = json.loads(cleaned)
            return bool(data.get("achieved", False)), str(data.get("reason", ""))
        except json.JSONDecodeError:
            pass
        m = _re.search(r"\{[^{}]+\}", cleaned, _re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
                return bool(data.get("achieved", False)), str(data.get("reason", ""))
            except json.JSONDecodeError:
                pass
        return False, f"[parse error] {raw[:120]}"


# ── Per-turn assertion checkers ────────────────────────────────────────────────

def _check_turn_assertions(
    agent_response: str,
    assertion_configs: list[dict],
) -> list[dict]:
    results = []
    for cfg in assertion_configs:
        atype = cfg.get("type", "")
        params = cfg.get("params", {})
        desc   = cfg.get("description", atype)

        if atype == "ends_with_question":
            passed, detail = _assert_ends_with_question(agent_response, params)
        elif atype == "max_words":
            passed, detail = _assert_max_words(agent_response, params)
        elif atype == "max_sentences":
            passed, detail = _assert_max_sentences(agent_response, params)
        elif atype == "no_forbidden_phrases":
            passed, detail = _assert_no_forbidden_phrases(agent_response, params)
        elif atype == "contains_phrase":
            passed, detail = _assert_contains_phrase(agent_response, params)
        elif atype == "regex_forbidden":
            passed, detail = _assert_regex_forbidden(agent_response, params)
        elif atype == "regex_required":
            passed, detail = _assert_regex_required(agent_response, params)
        else:
            passed, detail = False, f"Unknown assertion type '{atype}'"

        results.append({"assertion": desc, "passed": passed, "detail": detail})
    return results


# ── Individual assertion implementations ───────────────────────────────────────

_TERMINAL_PHRASES = (
    "hold", "transfer", "connect", "goodbye", "conclude", "moment",
)

def _assert_ends_with_question(response: str, params: dict) -> tuple[bool, str]:
    lowered = response.lower()
    for phrase in _TERMINAL_PHRASES:
        if phrase in lowered:
            return True, f"Terminal phrase '{phrase}' detected — closing question not required"
    max_q = params.get("max_questions", 1)
    questions = re.findall(r"\?", response)
    count = len(questions)
    if count == 0:
        return False, "No question mark found — agent must close with a question"
    if count > max_q:
        return False, f"{count} question marks found — expected exactly {max_q}"
    if not response.strip().endswith("?"):
        return False, "Response does not end with a question"
    return True, f"Ends with exactly {count} question(s)"


def _assert_max_words(response: str, params: dict) -> tuple[bool, str]:
    limit = int(params.get("max", 50))
    count = len(response.split())
    if count > limit:
        return False, f"{count} words — limit is {limit}"
    return True, f"{count} words (limit {limit})"


def _assert_max_sentences(response: str, params: dict) -> tuple[bool, str]:
    limit = int(params.get("max", 3))
    sentences = [s for s in re.split(r"[.!?]+", response) if s.strip()]
    count = len(sentences)
    if count > limit:
        return False, f"{count} sentences — limit is {limit}"
    return True, f"{count} sentence(s) (limit {limit})"


def _assert_no_forbidden_phrases(response: str, params: dict) -> tuple[bool, str]:
    phrases = params.get("phrases", [])
    case_sensitive = params.get("case_sensitive", False)
    check = response if case_sensitive else response.lower()
    for phrase in phrases:
        needle = phrase if case_sensitive else phrase.lower()
        if needle in check:
            return False, f"Forbidden phrase detected: '{phrase}'"
    return True, f"None of {len(phrases)} forbidden phrase(s) found"


def _assert_contains_phrase(response: str, params: dict) -> tuple[bool, str]:
    phrases = params.get("phrases", [])
    case_sensitive = params.get("case_sensitive", False)
    check = response if case_sensitive else response.lower()
    for phrase in phrases:
        needle = phrase if case_sensitive else phrase.lower()
        if needle in check:
            return True, f"Required phrase found: '{phrase}'"
    return False, f"None of the required phrases found: {phrases}"


def _assert_regex_forbidden(response: str, params: dict) -> tuple[bool, str]:
    pattern = params.get("pattern", "")
    flags = 0 if params.get("case_sensitive", False) else re.IGNORECASE
    try:
        m = re.search(pattern, response, flags)
    except re.error as exc:
        return False, f"Invalid regex: {exc}"
    if m:
        return False, f"Forbidden pattern matched: '{m.group()[:60]}'"
    return True, "Forbidden pattern not found"


def _assert_regex_required(response: str, params: dict) -> tuple[bool, str]:
    pattern = params.get("pattern", "")
    flags = 0 if params.get("case_sensitive", False) else re.IGNORECASE
    try:
        m = re.search(pattern, response, flags)
    except re.error as exc:
        return False, f"Invalid regex: {exc}"
    if m:
        return True, f"Required pattern matched: '{m.group()[:60]}'"
    return False, "Required pattern not found"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _format_transcript(turns: list[TurnResult]) -> str:
    lines: list[str] = []
    for t in turns:
        lines.append(f"User:  {t.user_message}")
        lines.append(f"Agent: {t.agent_response}")
    return "\n".join(lines)
