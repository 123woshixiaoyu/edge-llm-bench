#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVING_ROOT = Path(__file__).resolve().parents[1]


TEXT_PROMPT = (
    "Write exactly eight numbered implementation notes for an edge AI gateway demo. "
    "Each note should be one concise sentence. "
    "End with DONE."
)
VISION_PROMPT = (
    "Write two concise sentences. Describe the scene and mention the main visible object. "
    "End with DONE."
)


def post_json(url: str, payload: dict[str, Any], timeout_s: float) -> tuple[int, dict[str, Any], float, str]:
    start = time.perf_counter()
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw), (time.perf_counter() - start) * 1000, ""
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"error": raw[:500]}
        return exc.code, data, (time.perf_counter() - start) * 1000, ""
    except Exception as exc:
        return 0, {}, (time.perf_counter() - start) * 1000, str(exc)


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def extract_text_response(data: dict[str, Any]) -> tuple[str, str, str]:
    decision = data.get("route_decision") or data.get("decision") or {}
    text = ""
    if data.get("choices"):
        text = data["choices"][0].get("message", {}).get("content", "")
    return decision.get("route", ""), decision.get("selected_model") or data.get("model", ""), text


def contains_reasoning_phrase(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "thinking process",
            "analyze the request",
            "analysis steps",
            "hidden reasoning",
            "chain-of-thought",
            "constraint",
        )
    )


def completion_marker_seen(text: str) -> bool:
    return bool(re.search(r"\bDONE\b\s*$", text.strip(), re.IGNORECASE))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare interactive demo output lengths at small vs larger max_tokens.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", type=Path, default=SERVING_ROOT / "results/raw/interactive_output_length_smoke.csv")
    parser.add_argument("--sample-image", type=Path, default=REPO_ROOT / "results/figures/camera_v05_positive_detection.jpg")
    parser.add_argument("--timeout-s", type=float, default=320.0)
    parser.add_argument("--image-source", choices=["upload", "camera"], default="upload")
    args = parser.parse_args()

    image_b64 = encode_image(args.sample_image) if args.image_source == "upload" else None
    base_url = args.url.rstrip("/")
    cases = [
        {
            "request_id": "text_local_128",
            "modality": "text",
            "max_tokens": 128,
            "expected_route": "local",
            "payload": {
                "messages": [{"role": "user", "content": TEXT_PROMPT}],
                "task_type": "qa",
                "privacy": "allow_remote",
                "quality": "low",
                "latency_budget_ms": 3000,
                "max_tokens": 128,
                "stream": False,
            },
        },
        {
            "request_id": "text_local_512",
            "modality": "text",
            "max_tokens": 512,
            "expected_route": "local",
            "payload": {
                "messages": [{"role": "user", "content": TEXT_PROMPT}],
                "task_type": "qa",
                "privacy": "allow_remote",
                "quality": "low",
                "latency_budget_ms": 3000,
                "max_tokens": 512,
                "stream": False,
            },
        },
        {
            "request_id": "vision_remote_128",
            "modality": "vision",
            "max_tokens": 128,
            "expected_route": "remote",
            "payload": {
                "task_type": "scene_description",
                "privacy": "allow_remote",
                "quality": "high",
                "latency_budget_ms": 30000,
                "prompt": VISION_PROMPT,
                "image_source": args.image_source,
                "image_base64": image_b64,
                "use_yolo_trt": True,
                "max_tokens": 128,
            },
        },
        {
            "request_id": "vision_remote_384",
            "modality": "vision",
            "max_tokens": 384,
            "expected_route": "remote",
            "payload": {
                "task_type": "scene_description",
                "privacy": "allow_remote",
                "quality": "high",
                "latency_budget_ms": 30000,
                "prompt": VISION_PROMPT,
                "image_source": args.image_source,
                "image_base64": image_b64,
                "use_yolo_trt": True,
                "max_tokens": 384,
            },
        },
    ]

    fieldnames = [
        "request_id",
        "modality",
        "max_tokens",
        "expected_route",
        "route",
        "match_expected",
        "selected_backend",
        "http_status",
        "output_chars",
        "completion_marker_seen",
        "contains_reasoning_phrase",
        "likely_truncated",
        "backend_latency_ms",
        "total_latency_ms",
        "response_preview",
        "response_tail",
        "error",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for case in cases:
        endpoint = "/v1/chat/completions" if case["modality"] == "text" else "/v1/vision/analyze"
        http_status, data, elapsed_ms, transport_error = post_json(f"{base_url}{endpoint}", case["payload"], args.timeout_s)
        if case["modality"] == "text":
            route, backend, text = extract_text_response(data)
            backend_latency_ms = data.get("backend_latency_ms", "")
            total_latency_ms = data.get("total_latency_ms", round(elapsed_ms, 2))
        else:
            route = data.get("route", "")
            backend = data.get("selected_backend") or ""
            text = data.get("remote_response_text") or data.get("error") or ""
            backend_latency_ms = data.get("remote_latency_ms", "")
            total_latency_ms = data.get("total_latency_ms", round(elapsed_ms, 2))
        marker = completion_marker_seen(text)
        reasoning = contains_reasoning_phrase(text)
        rows.append(
            {
                "request_id": case["request_id"],
                "modality": case["modality"],
                "max_tokens": case["max_tokens"],
                "expected_route": case["expected_route"],
                "route": route,
                "match_expected": route == case["expected_route"],
                "selected_backend": backend,
                "http_status": http_status,
                "output_chars": len(text),
                "completion_marker_seen": marker,
                "contains_reasoning_phrase": reasoning,
                "likely_truncated": not marker,
                "backend_latency_ms": backend_latency_ms,
                "total_latency_ms": total_latency_ms,
                "response_preview": text.replace("\n", " ")[:240],
                "response_tail": text.replace("\n", " ")[-180:],
                "error": transport_error or data.get("error", ""),
            }
        )

    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"out": str(args.out), "rows": rows}, indent=2, ensure_ascii=False))
    return 0 if all(row["match_expected"] and not row["error"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
