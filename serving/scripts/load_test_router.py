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


def request(prompt: str, expected_route: str, **fields: Any) -> dict[str, Any]:
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "quality": fields.pop("quality", "medium"),
        "privacy": fields.pop("privacy", "allow_remote"),
        "latency_budget_ms": fields.pop("latency_budget_ms", 5000),
    }
    payload.update(fields)
    return {"payload": payload, "expected_route": expected_route}


def build_cases() -> list[dict[str, Any]]:
    long_context = "Project note. " * 240
    too_long = "Very long input. " * 1400
    cases = [
        request("What is edge AI?", "local", task_type="qa", quality="low"),
        request("用中文解释端侧 LLM 的量化价值。", "local", task_type="qa", quality="medium"),
        request(
            "Summarize this short note: quantization reduces memory and can improve latency.",
            "local",
            task_type="summary",
            quality="low",
        ),
        request(
            "请总结：Jetson 上 Q4 比 F16 更适合作默认部署。",
            "local",
            task_type="summary",
            quality="medium",
        ),
        request("Answer a quick product FAQ about local inference.", "local", quality="low"),
        request(
            "local only private note: summarize my medical appointment text.",
            "local",
            task_type="summary",
            privacy="local_only",
        ),
        request("Private QA: what does this internal log mean?", "local", task_type="qa", privacy="local_only"),
        request("Summarize this paragraph in one sentence.", "local", task_type="summary", latency_budget_ms=3000),
        request("Write Python code that appends a row to CSV.", "remote", task_type="code"),
        request("Debug this Python traceback and propose a fix.", "remote", task_type="code"),
        request("Explain why quantization changes memory bandwidth and latency.", "remote", task_type="reasoning"),
        request("Plan the experiment order for Jetson and RTX benchmarks.", "remote", task_type="reasoning"),
        request(long_context, "remote", task_type="long_context"),
        request("Give a high quality answer about deployment trade-offs.", "remote", task_type="qa", quality="high"),
        request("High quality summary required.", "remote", task_type="summary", quality="high"),
        request("Analyze this architecture deeply and list risks.", "remote", quality="high"),
        request(
            "Write code, but data must stay local.",
            "local",
            task_type="code",
            privacy="local_only",
            latency_budget_ms=5000,
        ),
        request(
            "Reason about a private local-only plan.",
            "local",
            task_type="reasoning",
            privacy="local_only",
            latency_budget_ms=5000,
        ),
        request("Complex code request with impossible latency.", "reject", task_type="code", latency_budget_ms=500),
        request(
            "High quality reasoning with impossible latency.",
            "reject",
            task_type="reasoning",
            quality="high",
            latency_budget_ms=500,
        ),
        request(too_long, "reject", task_type="long_context"),
    ]
    # Add a few repeated everyday cases so the report has 30 requests.
    for idx in range(9):
        cases.append(
            request(
                f"Short QA case {idx}: explain local routing in one sentence.",
                "local",
                task_type="qa",
                quality="low",
            )
        )
    return cases


def post_json(url: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any], float]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            latency_ms = (time.perf_counter() - start) * 1000
            return response.status, json.loads(response.read().decode("utf-8")), latency_ms
    except urllib.error.HTTPError as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        data = json.loads(exc.read().decode("utf-8"))
        return exc.code, data, latency_ms


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a simple policy evaluation against the Edge LLM router.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--endpoint", choices=["route", "chat"], default="route")
    parser.add_argument("--out", type=Path, default=Path("serving/results/raw/router_policy_eval.csv"))
    args = parser.parse_args()

    endpoint_path = "/v1/route" if args.endpoint == "route" else "/v1/chat/completions"
    url = args.url.rstrip("/") + endpoint_path
    rows: list[dict[str, Any]] = []
    cases = build_cases()

    for idx, case in enumerate(cases):
        payload = dict(case["payload"])
        payload["request_id"] = f"loadtest-{idx:03d}"
        status, data, latency_ms = post_json(url, payload)
        decision = data.get("decision") or data.get("route_decision") or {}
        analysis = data.get("analysis") or {}
        usage = data.get("usage") or {}
        estimated_prompt_tokens = (
            analysis.get("estimated_prompt_tokens")
            or usage.get("estimated_prompt_tokens")
            or ""
        )
        route = decision.get("route", "")
        row = {
            "request_id": payload["request_id"],
            "task_type": payload.get("task_type", analysis.get("task_type", "")),
            "quality": payload.get("quality", ""),
            "privacy": payload.get("privacy", ""),
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "route": route,
            "selected_model": decision.get("selected_model", ""),
            "expected_route": case["expected_route"],
            "match_expected": str(route == case["expected_route"]).lower(),
            "reasons": "; ".join(decision.get("reasons", [])),
            "total_latency_ms": data.get("total_latency_ms", round(latency_ms, 2)),
            "http_status": status,
        }
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    matches = sum(1 for row in rows if row["match_expected"] == "true")
    print(f"wrote {len(rows)} rows to {args.out}")
    print(f"matches={matches}/{len(rows)}")
    return 0 if matches == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
