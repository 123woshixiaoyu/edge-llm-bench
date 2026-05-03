#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[idx]


def floats(rows: list[dict[str, str]], field: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = row.get(field, "")
        if value == "":
            continue
        try:
            out.append(float(value))
        except ValueError:
            continue
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Edge LLM router serving load test CSV.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, default=Path("serving/results/raw/dual_real_backend_load_test_summary.csv"))
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    local_rows = [row for row in rows if row.get("route") == "local"]
    remote_rows = [row for row in rows if row.get("route") == "remote"]
    reject_rows = [row for row in rows if row.get("route") == "reject"]
    timeout_rows = [
        row
        for row in rows
        if row.get("http_status") == "0" or "timed out" in row.get("error", "").lower()
    ]
    backend_error_rows = [
        row
        for row in rows
        if row.get("http_status") == "502" or (row.get("error") and row.get("route") != "reject")
    ]
    success_rows = [row for row in rows if row.get("http_status") == "200"]
    route_matches = [row for row in rows if row.get("match_expected") == "true"]

    summary = {
        "total_requests": len(rows),
        "success_count": len(success_rows),
        "reject_count": len(reject_rows),
        "backend_error_count": len(backend_error_rows),
        "timeout_count": len(timeout_rows),
        "route_match_rate": round(len(route_matches) / len(rows), 4) if rows else 0.0,
        "local_count": len(local_rows),
        "remote_count": len(remote_rows),
        "reject_route_count": len(reject_rows),
    }

    latency_groups = {
        "local": floats(local_rows, "total_latency_ms"),
        "remote": floats(remote_rows, "total_latency_ms"),
        "total": floats(rows, "total_latency_ms"),
    }
    for name, values in latency_groups.items():
        summary[f"{name}_p50_latency_ms"] = round(percentile(values, 0.50), 2)
        summary[f"{name}_p95_latency_ms"] = round(percentile(values, 0.95), 2)
        summary[f"{name}_p99_latency_ms"] = round(percentile(values, 0.99), 2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(f"wrote summary to {args.out}")
    for key, value in summary.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
