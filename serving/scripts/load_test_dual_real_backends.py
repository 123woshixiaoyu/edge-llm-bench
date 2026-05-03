#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def case(prompt: str, expected_route: str, **fields: Any) -> dict[str, Any]:
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "quality": fields.pop("quality", "medium"),
        "privacy": fields.pop("privacy", "allow_remote"),
        "latency_budget_ms": fields.pop("latency_budget_ms", 5000),
        "max_tokens": fields.pop("max_tokens", 48),
    }
    payload.update(fields)
    return {"payload": payload, "expected_route": expected_route}


def build_templates() -> list[dict[str, Any]]:
    return [
        case("In one sentence, what is edge AI?", "local", task_type="qa", quality="low"),
        case("Summarize: quantization keeps Jetson inference fast.", "local", task_type="summary", quality="low"),
        case("请用一句中文解释端侧推理为什么适合隐私任务。", "local", task_type="qa", privacy="local_only"),
        case("Write Python code that appends a dictionary row to CSV.", "remote", task_type="code"),
        case("Reason about why Q4 improves useful throughput per watt.", "remote", task_type="reasoning"),
        case("Give a high quality answer about local versus remote routing.", "remote", task_type="qa", quality="high"),
        case("Complex code request with impossible latency.", "reject", task_type="code", latency_budget_ms=500),
    ]


def post_json(url: str, payload: dict[str, Any], timeout_s: float) -> tuple[int, dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"error": raw}
        return exc.code, data
    except Exception as exc:
        return 0, {"error": str(exc)}


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


def run_one(url: str, idx: int, template: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    payload = dict(template["payload"])
    payload["request_id"] = f"dual-load-{idx:04d}"
    start_time = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    status, data = post_json(url, payload, timeout_s)
    measured_latency_ms = (time.perf_counter() - start) * 1000
    end_time = datetime.now(timezone.utc).isoformat()
    decision = data.get("route_decision") or data.get("decision") or {}
    route = str(decision.get("route") or "")
    text = extract_text(data)
    error = str(data.get("error") or "")
    return {
        "request_id": payload["request_id"],
        "route": route,
        "selected_model": decision.get("selected_model", ""),
        "expected_route": template["expected_route"],
        "match_expected": str(route == template["expected_route"]).lower(),
        "http_status": status,
        "backend_latency_ms": data.get("backend_latency_ms", ""),
        "total_latency_ms": data.get("total_latency_ms", round(measured_latency_ms, 2)),
        "is_mock_response": str("[mock:" in text).lower(),
        "error": error,
        "start_time": start_time,
        "end_time": end_time,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Concurrent load test for dual real Edge LLM router backends.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--out", type=Path, default=Path("serving/results/raw/dual_real_backend_load_test.csv"))
    parser.add_argument("--timeout-s", type=float, default=180.0)
    args = parser.parse_args()

    endpoint = args.url.rstrip("/") + "/v1/chat/completions"
    templates = build_templates()
    jobs = [(idx, templates[idx % len(templates)]) for idx in range(args.requests)]
    rows: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(run_one, endpoint, idx, template, args.timeout_s) for idx, template in jobs]
        for future in as_completed(futures):
            rows.append(future.result())

    rows.sort(key=lambda row: row["request_id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    matches = sum(1 for row in rows if row["match_expected"] == "true")
    errors = sum(1 for row in rows if row["error"] and row["route"] != "reject")
    print(f"wrote {len(rows)} rows to {args.out}")
    print(f"matches={matches}/{len(rows)} backend_errors_or_timeouts={errors}")
    return 0 if matches == len(rows) and errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
