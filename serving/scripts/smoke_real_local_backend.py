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


def case(prompt: str, expected_route: str, **fields: Any) -> dict[str, Any]:
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "quality": fields.pop("quality", "medium"),
        "privacy": fields.pop("privacy", "allow_remote"),
        "latency_budget_ms": fields.pop("latency_budget_ms", 5000),
        "max_tokens": fields.pop("max_tokens", 64),
    }
    payload.update(fields)
    return {"payload": payload, "expected_route": expected_route}


def build_cases() -> list[dict[str, Any]]:
    return [
        case("In one sentence, what is edge AI?", "local", task_type="qa", quality="low"),
        case(
            "Summarize this note in one sentence: Q4 quantization lets a small LLM run faster on Jetson.",
            "local",
            task_type="summary",
            quality="low",
        ),
        case(
            "\u8bf7\u7528\u4e00\u53e5\u4e2d\u6587\u89e3\u91ca\u4e3a\u4ec0\u4e48\u7aef\u4fa7\u63a8\u7406\u91cd\u89c6\u9690\u79c1\u3002",
            "local",
            task_type="qa",
            privacy="local_only",
        ),
        case(
            "Summarize this private device log: battery normal, thermal state healthy, queue empty.",
            "local",
            task_type="summary",
            privacy="local_only",
        ),
        case("Write Python code that appends a dictionary row to a CSV file.", "remote", task_type="code"),
        case("Reason about why Q4 can improve useful throughput per watt.", "remote", task_type="reasoning"),
        case("Give a high quality answer about local versus remote LLM routing.", "remote", task_type="qa", quality="high"),
        case("Plan a careful debug strategy for an overloaded edge gateway.", "remote", task_type="reasoning"),
        case(
            "Complex code request with impossible latency budget.",
            "reject",
            task_type="code",
            latency_budget_ms=500,
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


def extract_response_text(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test real local llama.cpp backend through the router.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", type=Path, default=Path("serving/results/raw/real_local_backend_smoke.csv"))
    parser.add_argument("--timeout-s", type=float, default=90.0)
    args = parser.parse_args()

    url = args.url.rstrip("/") + "/v1/chat/completions"
    rows: list[dict[str, Any]] = []

    for idx, item in enumerate(build_cases()):
        payload = dict(item["payload"])
        payload["request_id"] = f"real-local-smoke-{idx:03d}"
        status, data, measured_latency_ms = post_json(url, payload, args.timeout_s)
        decision = data.get("route_decision") or data.get("decision") or {}
        route = str(decision.get("route") or "")
        text = extract_response_text(data)
        is_mock = "[mock:" in text
        backend_latency_ms = data.get("backend_latency_ms")
        total_latency_ms = data.get("total_latency_ms", round(measured_latency_ms, 2))
        row = {
            "request_id": payload["request_id"],
            "task_type": payload.get("task_type", ""),
            "quality": payload.get("quality", ""),
            "privacy": payload.get("privacy", ""),
            "route": route,
            "selected_model": decision.get("selected_model", ""),
            "is_mock_response": str(is_mock).lower(),
            "backend_latency_ms": backend_latency_ms if backend_latency_ms is not None else "",
            "total_latency_ms": total_latency_ms,
            "http_status": status,
            "short_response_preview": text[:160].replace("\n", " "),
            "reasons": "; ".join(decision.get("reasons", [])),
            "expected_route": item["expected_route"],
            "match_expected": str(route == item["expected_route"]).lower(),
            "error": str(data.get("error") or ""),
        }
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    local_rows = [row for row in rows if row["route"] == "local"]
    remote_rows = [row for row in rows if row["route"] == "remote"]
    reject_rows = [row for row in rows if row["route"] == "reject"]
    local_real_success = [
        row
        for row in local_rows
        if row["http_status"] == 200 and row["is_mock_response"] == "false" and not row["error"]
    ]
    route_matches = sum(1 for row in rows if row["match_expected"] == "true")

    print(f"wrote {len(rows)} rows to {args.out}")
    print(
        "routes="
        f"local:{len(local_rows)} remote:{len(remote_rows)} reject:{len(reject_rows)}; "
        f"local_real_success={len(local_real_success)}/{len(local_rows)}; "
        f"matches={route_matches}/{len(rows)}"
    )

    if route_matches != len(rows):
        return 1
    if len(local_real_success) != len(local_rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
