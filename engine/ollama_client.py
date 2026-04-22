"""
engine/ollama_client.py
───────────────────────
Thin async-compatible wrapper around the Ollama /api/chat REST endpoint.

Usage:
    client = OllamaClient(base_url="http://localhost:11434", model="llama3")
    reply  = client.chat(messages=[{"role": "user", "content": "Hello"}])
    # → "Hi there!"

    # With a system prompt:
    reply = client.chat(
        messages=[{"role": "user", "content": "Judge this response."}],
        system="You are a strict evaluator.",
        temperature=0.0,
    )
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class OllamaError(Exception):
    """Raised when the Ollama API returns an error or is unreachable."""


class OllamaClient:
    """
    Synchronous HTTP client for Ollama's /api/chat endpoint.

    Keeps zero third-party dependencies beyond the stdlib — urllib only.
    """

    def __init__(self, base_url: str, model: str, timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    # ── Public API ─────────────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        """
        Send a chat request to Ollama and return the assistant's reply text.

        Parameters
        ----------
        messages    : OpenAI-style message list [{"role": ..., "content": ...}]
        system      : Optional system prompt prepended to the conversation.
        temperature : Sampling temperature (0.0 = deterministic).

        Returns
        -------
        str — the model's reply, stripped of leading/trailing whitespace.

        Raises
        ------
        OllamaError — if the server is unreachable or returns an error body.
        """
        full_messages: list[dict[str, str]] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": full_messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        body = json.dumps(payload).encode("utf-8")
        url = f"{self.base_url}/api/chat"

        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self.base_url}: {exc.reason}"
            ) from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OllamaError(f"Malformed JSON from Ollama: {raw[:200]}") from exc

        if "error" in data:
            raise OllamaError(f"Ollama error: {data['error']}")

        try:
            return data["message"]["content"].strip()
        except (KeyError, TypeError) as exc:
            raise OllamaError(f"Unexpected Ollama response shape: {raw[:200]}") from exc

    def is_available(self) -> bool:
        """Return True if the Ollama server is reachable (GET /api/tags)."""
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/api/tags", timeout=5
            ):
                return True
        except Exception:
            return False
