#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVING_ROOT = Path(__file__).resolve().parents[1]


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


def text_payload(
    prompt: str,
    task_type: str,
    privacy: str,
    quality: str,
    latency_budget_ms: int,
    *,
    max_tokens: int = 128,
) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": prompt}],
        "task_type": task_type,
        "privacy": privacy,
        "quality": quality,
        "latency_budget_ms": latency_budget_ms,
        "stream": False,
        "max_tokens": max_tokens,
    }


def normalize_text_response(data: dict[str, Any], http_status: int) -> dict[str, Any]:
    decision = data.get("route_decision") or data.get("decision") or {}
    message = ""
    if data.get("choices"):
        message = data["choices"][0].get("message", {}).get("content", "")
    return {
        "route": decision.get("route", "reject" if http_status >= 400 else ""),
        "selected_backend": decision.get("selected_model") or data.get("model", ""),
        "local_cv_backend": "",
        "remote_is_mock": "",
        "backend_latency_ms": data.get("backend_latency_ms", ""),
        "total_latency_ms": data.get("total_latency_ms", ""),
        "response_preview": (message or data.get("error", ""))[:240],
        "reasons": decision.get("reasons", []),
    }


def normalize_vision_response(data: dict[str, Any], http_status: int) -> dict[str, Any]:
    return {
        "route": data.get("route", "reject" if http_status >= 400 else ""),
        "selected_backend": data.get("selected_backend") or "",
        "local_cv_backend": data.get("local_cv_backend") or "",
        "remote_is_mock": data.get("remote_is_mock"),
        "backend_latency_ms": data.get("remote_latency_ms") or data.get("local_cv_inference_latency_ms") or "",
        "total_latency_ms": data.get("total_latency_ms", ""),
        "response_preview": (data.get("remote_response_text") or data.get("error") or "")[:240],
        "reasons": data.get("reasons", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="v0.9 smoke test for the interactive Jetson Gateway demo.")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Jetson Gateway base URL")
    parser.add_argument("--out", type=Path, default=SERVING_ROOT / "results/raw/interactive_gateway_smoke.csv")
    parser.add_argument("--sample-image", type=Path, default=REPO_ROOT / "results/figures/camera_v05_positive_detection.jpg")
    parser.add_argument("--timeout-s", type=float, default=260.0)
    parser.add_argument("--image-source", choices=["camera", "upload"], default="camera")
    parser.add_argument("--allow-remote-vlm-unavailable", action="store_true")
    args = parser.parse_args()

    image_b64 = encode_image(args.sample_image) if args.image_source == "upload" else None
    base_url = args.url.rstrip("/")
    cases = [
        {
            "request_id": "interactive_text_local",
            "modality": "text",
            "endpoint": "/v1/chat/completions",
            "expected_route": "local",
            "payload": text_payload(
                "Summarize why edge AI routers need privacy-aware local execution in one short sentence.",
                "summary",
                "allow_remote",
                "low",
                3000,
                max_tokens=48,
            ),
        },
        {
            "request_id": "interactive_text_remote",
            "modality": "text",
            "endpoint": "/v1/chat/completions",
            "expected_route": "remote",
            "payload": text_payload(
                "Write a short Python function that parses a CSV and computes average latency.",
                "code",
                "allow_remote",
                "high",
                10000,
                max_tokens=128,
            ),
        },
        {
            "request_id": "interactive_vision_local",
            "modality": "vision",
            "endpoint": "/v1/vision/analyze",
            "expected_route": "local",
            "payload": {
                "task_type": "detect",
                "privacy": "allow_remote",
                "quality": "low",
                "latency_budget_ms": 3000,
                "image_source": args.image_source,
                "image_base64": image_b64,
                "use_yolo_trt": True,
            },
        },
        {
            "request_id": "interactive_vision_remote",
            "modality": "vision",
            "endpoint": "/v1/vision/analyze",
            "expected_route": "remote",
            "payload": {
                "task_type": "scene_description",
                "privacy": "allow_remote",
                "quality": "high",
                "latency_budget_ms": 30000,
                "prompt": "Describe the image in one concise sentence. Do not explain your reasoning.",
                "image_source": args.image_source,
                "image_base64": image_b64,
                "use_yolo_trt": True,
                "max_tokens": 64,
            },
        },
        {
            "request_id": "interactive_vision_privacy_reject",
            "modality": "vision",
            "endpoint": "/v1/vision/analyze",
            "expected_route": "reject",
            "payload": {
                "task_type": "vqa",
                "privacy": "local_only",
                "quality": "high",
                "latency_budget_ms": 3000,
                "prompt": "What is happening in this image?",
                "image_source": args.image_source,
                "image_base64": image_b64,
                "use_yolo_trt": True,
            },
        },
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "request_id",
        "modality",
        "task_type",
        "privacy",
        "quality",
        "route",
        "selected_backend",
        "local_cv_backend",
        "remote_is_mock",
        "backend_latency_ms",
        "total_latency_ms",
        "http_status",
        "match_expected",
        "error",
        "reasons",
        "response_preview",
    ]
    rows = []
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            payload = dict(case["payload"])
            payload["request_id"] = case["request_id"]
            http_status, data, elapsed_ms, transport_error = post_json(
                f"{base_url}{case['endpoint']}",
                payload,
                args.timeout_s,
            )
            normalized = (
                normalize_text_response(data, http_status)
                if case["modality"] == "text"
                else normalize_vision_response(data, http_status)
            )
            match_expected = normalized["route"] == case["expected_route"]
            if (
                args.allow_remote_vlm_unavailable
                and case["request_id"] == "interactive_vision_remote"
                and normalized["route"] == "reject"
            ):
                match_expected = True
            row = {
                "request_id": case["request_id"],
                "modality": case["modality"],
                "task_type": payload.get("task_type", ""),
                "privacy": payload.get("privacy", ""),
                "quality": payload.get("quality", ""),
                "route": normalized["route"],
                "selected_backend": normalized["selected_backend"],
                "local_cv_backend": normalized["local_cv_backend"],
                "remote_is_mock": normalized["remote_is_mock"],
                "backend_latency_ms": normalized["backend_latency_ms"],
                "total_latency_ms": normalized["total_latency_ms"] or round(elapsed_ms, 2),
                "http_status": http_status,
                "match_expected": match_expected,
                "error": transport_error or data.get("error", ""),
                "reasons": json.dumps(normalized["reasons"], ensure_ascii=False),
                "response_preview": str(normalized["response_preview"]).replace("\n", " ")[:240],
            }
            writer.writerow(row)
            rows.append(row)

    print(json.dumps({"out": str(args.out), "rows": rows}, indent=2, ensure_ascii=False))
    return 0 if all(row["match_expected"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
