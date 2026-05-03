from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass
class VLMResult:
    ok: bool
    model: str
    text: str
    latency_ms: float
    error: str = ""


class HttpVLMBackend:
    def __init__(self, base_url: str, *, timeout_s: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def available(self) -> bool:
        try:
            response = httpx.get(f"{self.base_url}/health", timeout=2.0)
            return response.status_code < 500
        except Exception:
            return False

    def complete_image(
        self,
        *,
        image_path: str | Path,
        prompt: str,
        max_tokens: int = 128,
    ) -> VLMResult:
        image_path = Path(image_path)
        if not image_path.exists():
            return VLMResult(False, "", "", 0.0, f"image not found: {image_path}")
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload = {
            "image_base64": image_b64,
            "image_mime_type": "image/jpeg",
            "prompt": prompt,
            "max_tokens": max_tokens,
        }
        start = time.perf_counter()
        try:
            response = httpx.post(
                f"{self.base_url}/v1/vision/completions",
                json=payload,
                timeout=self.timeout_s,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            if response.status_code >= 400:
                return VLMResult(
                    False,
                    "",
                    "",
                    latency_ms,
                    f"remote VLM returned HTTP {response.status_code}: {response.text[:300]}",
                )
            data = response.json()
            return VLMResult(
                ok=bool(data.get("ok")),
                model=str(data.get("model") or ""),
                text=str(data.get("text") or ""),
                latency_ms=float(data.get("latency_ms") or latency_ms),
                error=str(data.get("error") or ""),
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return VLMResult(False, "", "", latency_ms, str(exc))
