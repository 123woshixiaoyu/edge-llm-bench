#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def case(name: str, prompt: str, expected_route: str, **fields: Any) -> dict[str, Any]:
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "quality": fields.pop("quality", "medium"),
        "privacy": fields.pop("privacy", "allow_remote"),
        "latency_budget_ms": fields.pop("latency_budget_ms", 5000),
        "max_tokens": fields.pop("max_tokens", 48),
    }
    payload.update(fields)
    return {"case_name": name, "payload": payload, "expected_route": expected_route}


def build_cases() -> list[dict[str, Any]]:
    return [
        case(
            "remote_unavailable_complex",
            "Write Python code that appends a row to CSV.",
            "reject",
            task_type="code",
            state_override={"local_available": True, "remote_available": False, "local_queue_depth": 0},
        ),
        case(
            "local_unavailable_simple",
            "What is edge AI in one sentence?",
            "remote",
            task_type="qa",
            quality="low",
            state_override={"local_available": False, "remote_available": True, "local_queue_depth": 0},
        ),
        case(
            "local_queue_overloaded",
            "Summarize this short note: local queue is busy.",
            "remote",
            task_type="summary",
            quality="low",
            state_override={"local_available": True, "remote_available": True, "local_queue_depth": 2},
        ),
        case(
            "high_temperature",
            "Answer briefly: why route away from a hot Jetson?",
            "remote",
            task_type="qa",
            quality="low",
            state_override={"local_available": True, "remote_available": True, "jetson_temp_c": 80},
        ),
        case(
            "privacy_local_only_local_unavailable",
            "Private QA: summarize this private note.",
            "reject",
            task_type="qa",
            privacy="local_only",
            state_override={"local_available": False, "remote_available": True, "local_queue_depth": 0},
        ),
        case(
            "privacy_local_only_queue_overloaded",
            "Private summary: queue is overloaded.",
            "reject",
            task_type="summary",
            privacy="local_only",
            state_override={"local_available": True, "remote_available": True, "local_queue_depth": 2},
        ),
    ]


def post_json(url: str, payload: dict[str, Any], timeout_s: float) -> tuple[int, dict[str, Any], float]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            latency_ms = (time.perf_counter() - start) * 1000
            return response.status, json.loads(response.read().decode("utf-8")), latency_ms
    except urllib.error.HTTPError as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        raw = exc.read().decode("utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"error": raw}
        return exc.code, data, latency_ms
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        return 0, {"error": str(exc)}, latency_ms


def extract_text(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(
        message.get("content")
        or message.get("reasoning_content")
        or message.get("reasoning")
        or choices[0].get("text")
        or ""
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke failure-mode routing behavior.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", type=Path, default=Path("serving/results/raw/failure_modes_smoke.csv"))
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args()

    endpoint = args.url.rstrip("/") + "/v1/chat/completions"
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(build_cases()):
        payload = dict(item["payload"])
        payload["request_id"] = f"failure-smoke-{idx:03d}"
        status, data, measured_latency_ms = post_json(endpoint, payload, args.timeout_s)
        decision = data.get("route_decision") or data.get("decision") or {}
        route = str(decision.get("route") or "")
        text = extract_text(data)
        row = {
            "case_name": item["case_name"],
            "request_id": payload["request_id"],
            "route": route,
            "selected_model": decision.get("selected_model", ""),
            "expected_route": item["expected_route"],
            "match_expected": str(route == item["expected_route"]).lower(),
            "http_status": status,
            "backend_latency_ms": data.get("backend_latency_ms", ""),
            "total_latency_ms": data.get("total_latency_ms", round(measured_latency_ms, 2)),
            "is_mock_response": str("[mock:" in text).lower(),
            "reasons": "; ".join(decision.get("reasons", [])),
            "error": str(data.get("error") or ""),
        }
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    matches = sum(1 for row in rows if row["match_expected"] == "true")
    unexpected_errors = [row for row in rows if row["error"] and row["route"] != "reject"]
    print(f"wrote {len(rows)} rows to {args.out}")
    print(f"matches={matches}/{len(rows)} unexpected_errors={len(unexpected_errors)}")
    return 0 if matches == len(rows) and not unexpected_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
