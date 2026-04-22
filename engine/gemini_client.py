"""
engine/gemini_client.py
───────────────────────
Thin stdlib-only client for the Gemini REST API.

Implements the same interface as OllamaClient (chat / is_available)
so the two are interchangeable across the engine layer.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from threading import Event, Thread

log = logging.getLogger(__name__)


class GeminiError(Exception):
    """Raised when the Gemini API returns an error or is unreachable."""


class GeminiClient:
    """
    Synchronous HTTP client for the Gemini generateContent REST endpoint.
    Zero third-party dependencies — urllib only.

    Optional LangSmith integration: pass langsmith_api_key to enable
    per-call token tracking posted to api.smith.langchain.com.
    """

    _BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash-lite",
        timeout: int = 60,
        langsmith_api_key: str = "",
        langsmith_project: str = "voice-agent-evals",
        langsmith_endpoint: str = "https://api.smith.langchain.com",
        langsmith_tenant_id: str = "",
        langsmith_session_id: str = "",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._ls_key = langsmith_api_key.strip()
        self._ls_project = langsmith_project or "voice-agent-evals"
        self._ls_endpoint = (langsmith_endpoint or "https://api.smith.langchain.com").rstrip("/")
        self._ls_tenant = langsmith_tenant_id.strip()
        self._ls_session = langsmith_session_id.strip()  # project UUID — used as session_id

    # ── Public API (mirrors OllamaClient) ─────────────────────────────────────

    def chat(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        temperature: float = 0.0,
        cancel_event: Event | None = None,
        rpm_limit: int = 12,
    ) -> str:
        """
        Send a chat request to Gemini and return the model's reply text.

        Parameters
        ----------
        messages     : OpenAI-style message list [{"role": ..., "content": ...}]
        system       : Optional system instruction.
        temperature  : Sampling temperature (0.0 = deterministic).
        cancel_event : threading.Event — if set(), raises GeminiError("cancelled").
        rpm_limit    : Target max requests-per-minute (default 12, safely under free 15 RPM).
        """
        def interruptible_sleep(seconds: float) -> None:
            """Sleep in 0.5s ticks so cancel_event can abort early."""
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                if cancel_event and cancel_event.is_set():
                    raise GeminiError("cancelled")
                time.sleep(min(0.5, deadline - time.monotonic()))

        # Convert OpenAI-style roles to Gemini roles
        contents = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})

        payload: dict = {
            "contents": contents,
            "generationConfig": {"temperature": temperature},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        url = f"{self._BASE}/{self.model}:generateContent?key={self.api_key}"
        body = json.dumps(payload).encode("utf-8")

        def make_req() -> urllib.request.Request:
            return urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

        start_time = datetime.now(timezone.utc)
        t0 = time.monotonic()

        last_exc: Exception | None = None
        for attempt in range(5):  # 1 initial + 4 retries
            if cancel_event and cancel_event.is_set():
                raise GeminiError("cancelled")
            try:
                with urllib.request.urlopen(make_req(), timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break  # success
            except urllib.error.HTTPError as exc:
                err_body = exc.read().decode("utf-8", errors="replace")
                if exc.code in (429, 503):
                    last_exc = GeminiError(f"Gemini API HTTP {exc.code} — attempt {attempt+1}/5")
                    interruptible_sleep(10)
                    continue
                raise GeminiError(f"Gemini API HTTP {exc.code}: {err_body[:300]}") from exc
            except urllib.error.URLError as exc:
                raise GeminiError(f"Cannot reach Gemini API: {exc.reason}") from exc
        else:
            raise last_exc  # all 5 attempts hit 429

        end_time = datetime.now(timezone.utc)
        latency_ms = int((time.monotonic() - t0) * 1000)
        self.last_api_latency_ms: int = latency_ms  # pure API time, excludes RPM sleep

        # Rate-limit: enforce target RPM by sleeping between successful calls
        if rpm_limit > 0:
            interruptible_sleep(60.0 / rpm_limit)

        try:
            reply = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise GeminiError(f"Unexpected Gemini response: {str(data)[:200]}") from exc

        # Fire-and-forget LangSmith trace
        if self._ls_key:
            usage = data.get("usageMetadata", {})
            Thread(
                target=self._log_langsmith,
                args=(messages, system, reply, usage, start_time, end_time, latency_ms),
                daemon=False,
                name="langsmith-trace",
            ).start()

        return reply

    def is_available(self) -> bool:
        """Return True if an API key is configured."""
        return bool(self.api_key and self.api_key.strip())

    # ── LangSmith ─────────────────────────────────────────────────────────────

    def _log_langsmith(
        self,
        messages: list[dict],
        system: str | None,
        reply: str,
        usage: dict,
        start_time: datetime,
        end_time: datetime,
        latency_ms: int,
    ) -> None:
        prompt_tokens = usage.get("promptTokenCount", 0)
        completion_tokens = usage.get("candidatesTokenCount", 0)
        total_tokens = usage.get("totalTokenCount", 0)

        run_id = str(uuid.uuid4())
        # dotted_order: YYYYMMDDTHHMMSSffffffZ{uuid} — required by LangSmith API
        dotted_order = start_time.strftime("%Y%m%dT%H%M%S%f") + "Z" + run_id

        run = {
            "id": run_id,
            "name": self.model,
            "run_type": "llm",
            "inputs": {
                "messages": messages,
                **({"system": system} if system else {}),
            },
            "outputs": {"text": reply},
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "trace_id": run_id,
            "dotted_order": dotted_order,
            "extra": {
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
                "model": self.model,
                "latency_ms": latency_ms,
            },
            "tags": ["voice-agent-eval"],
            # session_id (project UUID) routes to the correct project; project_name is a fallback
            **({"session_id": self._ls_session} if self._ls_session else {"project_name": self._ls_project}),
        }
        try:
            body = json.dumps(run).encode("utf-8")
            hdrs = {"Content-Type": "application/json", "x-api-key": self._ls_key}
            if self._ls_tenant:
                hdrs["X-Tenant-ID"] = self._ls_tenant
            req = urllib.request.Request(
                f"{self._ls_endpoint}/runs",
                data=body,
                headers=hdrs,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = resp.status
            log.info("LangSmith trace posted (HTTP %s): %d prompt + %d completion tokens",
                     status, prompt_tokens, completion_tokens)
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")[:200]
            log.warning("LangSmith logging failed HTTP %s: %s", exc.code, err_body)
        except Exception as exc:
            log.warning("LangSmith logging failed: %s", exc)
