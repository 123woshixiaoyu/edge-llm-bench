from __future__ import annotations

import time

import httpx

from .base import BackendResult
from ..schemas import Message


class LlamaCppBackend:
    def __init__(self, name: str, base_url: str, timeout_s: float = 60.0):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def available(self) -> bool:
        try:
            response = httpx.get(f"{self.base_url}/health", timeout=2.0)
            return response.status_code < 500
        except Exception:
            return False

    def generate(self, messages: list[Message], model: str, max_tokens: int | None = None) -> BackendResult:
        payload = {
            "model": model,
            "messages": [message.model_dump() for message in messages],
            "stream": False,
            "temperature": 0,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        start = time.perf_counter()
        try:
            response = httpx.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout_s,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            if response.status_code >= 400:
                return BackendResult(
                    ok=False,
                    text="",
                    latency_ms=latency_ms,
                    error=f"llama.cpp backend returned HTTP {response.status_code}: {response.text[:300]}",
                )
            data = response.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return BackendResult(ok=True, text=text, latency_ms=latency_ms)
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return BackendResult(ok=False, text="", latency_ms=latency_ms, error=str(exc))
