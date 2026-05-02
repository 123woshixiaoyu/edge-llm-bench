from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..schemas import Message


@dataclass
class BackendResult:
    ok: bool
    text: str
    latency_ms: float
    error: str | None = None


class Backend(Protocol):
    name: str

    def available(self) -> bool:
        ...

    def generate(self, messages: list[Message], model: str, max_tokens: int | None = None) -> BackendResult:
        ...
