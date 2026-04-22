"""
engine/level1_engine.py
───────────────────────
Generic Level 1 rule executor.

Reads rule definitions from project_config.yaml and applies them to
an agent response string. Zero hardcoded domain knowledge — all
phrases, patterns, and thresholds come from the config.

Supported rule types
────────────────────
  regex_forbidden    — fail if pattern is found in the response
  regex_required     — fail if pattern is NOT found in the response
  max_sentences      — fail if sentence count exceeds params.max
  starts_with_phrase — fail if response doesn't open with a listed phrase
  max_words          — fail if word count exceeds params.max

Usage:
    from engine.config_loader import load_config
    from engine.level1_engine import Level1Engine

    config  = load_config(Path("projects/dental_booking_bot"))
    engine  = Level1Engine(config)
    results = engine.run(response="Let me check on that for you. What time works best?")
    for r in results:
        print(r.rule_id, "PASS" if r.passed else f"FAIL — {r.excerpt}")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class RuleResult:
    rule_id:     str
    description: str
    passed:      bool
    excerpt:     str = ""   # snippet that caused the failure (empty on PASS)
    detail:      str = ""   # human-readable reason (populated on FAIL)

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


# ── Engine ─────────────────────────────────────────────────────────────────────

class Level1Engine:

    # Dispatch table: rule type → bound method
    _HANDLERS: dict[str, str] = {
        "regex_forbidden":    "_check_regex_forbidden",
        "regex_required":     "_check_regex_required",
        "max_sentences":      "_check_max_sentences",
        "starts_with_phrase": "_check_starts_with_phrase",
        "max_words":          "_check_max_words",
    }

    def __init__(self, config: dict) -> None:
        self._rules: list[dict] = config["level1"]["rules"]

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, response: str) -> list[RuleResult]:
        """
        Apply every rule in the config to `response`.
        Always runs all rules — failures do not short-circuit the rest.
        """
        results: list[RuleResult] = []
        for rule in self._rules:
            handler_name = self._HANDLERS.get(rule["type"])
            if handler_name is None:
                # Unknown type: skip with a warning result rather than crashing
                results.append(RuleResult(
                    rule_id=rule["id"],
                    description=rule.get("description", ""),
                    passed=False,
                    detail=f"Unknown rule type '{rule['type']}' — check config.",
                ))
                continue
            handler = getattr(self, handler_name)
            results.append(handler(rule, response))
        return results

    # ── Rule implementations ───────────────────────────────────────────────────

    def _check_regex_forbidden(self, rule: dict, response: str) -> RuleResult:
        """Fail if `pattern` is found anywhere in the response."""
        params = rule["params"]
        pattern = params["pattern"]
        match = re.search(pattern, response)

        if match:
            excerpt = _excerpt_around(response, match.start(), match.end())
            return RuleResult(
                rule_id=rule["id"],
                description=rule["description"],
                passed=False,
                excerpt=excerpt,
                detail=params.get("error_message", f"Pattern '{pattern}' found"),
            )

        return _pass(rule)

    def _check_regex_required(self, rule: dict, response: str) -> RuleResult:
        """Fail if `pattern` is NOT found anywhere in the response."""
        params = rule["params"]
        pattern = params["pattern"]
        match = re.search(pattern, response)

        if not match:
            return RuleResult(
                rule_id=rule["id"],
                description=rule["description"],
                passed=False,
                excerpt="(not found)",
                detail=params.get("error_message", f"Required pattern '{pattern}' missing"),
            )

        return _pass(rule)

    def _check_max_sentences(self, rule: dict, response: str) -> RuleResult:
        """Fail if the sentence count exceeds params.max."""
        params = rule["params"]
        max_allowed: int = params["max"]

        sentences = [
            s.strip()
            for s in re.split(r"[.!?]+", response)
            if s.strip()
        ]
        count = len(sentences)

        if count > max_allowed:
            return RuleResult(
                rule_id=rule["id"],
                description=rule["description"],
                passed=False,
                detail=f"Expected ≤ {max_allowed} sentences, got {count}",
            )

        return _pass(rule)

    def _check_starts_with_phrase(self, rule: dict, response: str) -> RuleResult:
        """
        Fail if none of the listed phrases appear within the first N words
        of the response.
        """
        params = rule["params"]
        n_words: int = params.get("check_first_n_words", 8)
        case_sensitive: bool = params.get("case_sensitive", False)
        phrases: list[str] = params.get("phrases", [])

        if not phrases:
            return RuleResult(
                rule_id=rule["id"],
                description=rule["description"],
                passed=False,
                detail="No phrases defined in config — nothing to match against.",
            )

        words = response.split()[:n_words]
        window = " ".join(words)

        compare_window = window if case_sensitive else window.lower()

        for phrase in phrases:
            compare_phrase = phrase if case_sensitive else phrase.lower()
            if compare_phrase in compare_window:
                return _pass(rule)

        return RuleResult(
            rule_id=rule["id"],
            description=rule["description"],
            passed=False,
            excerpt=f'First {n_words} words: "{window}"',
            detail=f"No filler phrase found. Expected one of: {phrases}",
        )

    def _check_max_words(self, rule: dict, response: str) -> RuleResult:
        """Fail if the word count exceeds params.max."""
        params = rule["params"]
        max_allowed: int = params["max"]

        words = response.split()
        count = len(words)

        if count > max_allowed:
            return RuleResult(
                rule_id=rule["id"],
                description=rule["description"],
                passed=False,
                excerpt=f"{count} words detected",
                detail=f"Expected ≤ {max_allowed} words, got {count}",
            )

        return _pass(rule)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _pass(rule: dict) -> RuleResult:
    return RuleResult(
        rule_id=rule["id"],
        description=rule["description"],
        passed=True,
    )


def _excerpt_around(text: str, start: int, end: int, context: int = 20) -> str:
    """Return a short excerpt centred on the match position."""
    lo = max(0, start - context)
    hi = min(len(text), end + context)
    snippet = text[lo:hi]
    if lo > 0:
        snippet = "…" + snippet
    if hi < len(text):
        snippet = snippet + "…"
    return f'"{snippet}"'
