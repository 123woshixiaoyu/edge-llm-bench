from __future__ import annotations

import time

from .base import BackendResult
from ..schemas import Message


class MockBackend:
    def __init__(self, name: str):
        self.name = name

    def available(self) -> bool:
        return True

    def generate(self, messages: list[Message], model: str, max_tokens: int | None = None) -> BackendResult:
        start = time.perf_counter()
        prompt = "\n".join(message.content for message in messages)
        prompt_summary = prompt[:160].replace("\n", " ")
        text = (
            f"[mock:{self.name}] model={model}; max_tokens={max_tokens or 'default'}; "
            f"prompt_summary={prompt_summary}"
        )
        latency_ms = (time.perf_counter() - start) * 1000
        return BackendResult(ok=True, text=text, latency_ms=latency_ms)
