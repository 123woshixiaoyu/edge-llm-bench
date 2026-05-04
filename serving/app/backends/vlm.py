from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass
class VLMResult:
    ok: bool
    model: str
    text: str
    latency_ms: float
    error: str = ""
    timings: dict[str, Any] | None = None
    request_payload_bytes: int = 0
    client_read_ms: float = 0.0
    client_resize_ms: float = 0.0
    client_base64_encode_ms: float = 0.0
    network_roundtrip_ms: float = 0.0
    resized_width: int | None = None
    resized_height: int | None = None


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
        resize_width: int | None = None,
    ) -> VLMResult:
        image_path = Path(image_path)
        if not image_path.exists():
            return VLMResult(False, "", "", 0.0, f"image not found: {image_path}")
        start_read = time.perf_counter()
        image_bytes = image_path.read_bytes()
        client_read_ms = (time.perf_counter() - start_read) * 1000
        client_resize_ms = 0.0
        if resize_width and resize_width > 0:
            start_resize = time.perf_counter()
            from io import BytesIO

            from PIL import Image

            with Image.open(image_path) as image:
                width, height = image.size
                if width > resize_width:
                    new_height = max(1, int(round(height * (resize_width / width))))
                    resized = image.convert("RGB").resize((resize_width, new_height))
                    buffer = BytesIO()
                    resized.save(buffer, format="JPEG", quality=90)
                    image_bytes = buffer.getvalue()
                client_resize_ms = (time.perf_counter() - start_resize) * 1000
        start_b64 = time.perf_counter()
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        client_base64_encode_ms = (time.perf_counter() - start_b64) * 1000
        payload = {
            "image_base64": image_b64,
            "image_mime_type": "image/jpeg",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "resize_width": None,
            "mode": "subprocess_cli_mode",
        }
        request_payload_bytes = len(str(payload).encode("utf-8"))
        start = time.perf_counter()
        try:
            response = httpx.post(
                f"{self.base_url}/v1/vision/completions",
                json=payload,
                timeout=self.timeout_s,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            network_roundtrip_ms = latency_ms
            if response.status_code >= 400:
                return VLMResult(
                    False,
                    "",
                    "",
                    latency_ms,
                    f"remote VLM returned HTTP {response.status_code}: {response.text[:300]}",
                    request_payload_bytes=request_payload_bytes,
                    client_read_ms=client_read_ms,
                    client_resize_ms=client_resize_ms,
                    client_base64_encode_ms=client_base64_encode_ms,
                    network_roundtrip_ms=network_roundtrip_ms,
                )
            data = response.json()
            return VLMResult(
                ok=bool(data.get("ok")),
                model=str(data.get("model") or ""),
                text=str(data.get("text") or ""),
                latency_ms=float(data.get("latency_ms") or latency_ms),
                error=str(data.get("error") or ""),
                timings=dict(data.get("timings") or {}),
                request_payload_bytes=request_payload_bytes,
                client_read_ms=round(client_read_ms, 2),
                client_resize_ms=round(client_resize_ms, 2),
                client_base64_encode_ms=round(client_base64_encode_ms, 2),
                network_roundtrip_ms=round(network_roundtrip_ms, 2),
                resized_width=data.get("resized_width"),
                resized_height=data.get("resized_height"),
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return VLMResult(
                False,
                "",
                "",
                latency_ms,
                str(exc),
                request_payload_bytes=request_payload_bytes,
                client_read_ms=round(client_read_ms, 2),
                client_resize_ms=round(client_resize_ms, 2),
                client_base64_encode_ms=round(client_base64_encode_ms, 2),
                network_roundtrip_ms=round(latency_ms, 2),
            )
