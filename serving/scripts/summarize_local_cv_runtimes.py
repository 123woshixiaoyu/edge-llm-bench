#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return ordered[low]
    fraction = index - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def as_float(value: str) -> float | None:
    if value == "" or value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row.get("runtime", ""), row.get("model_name", ""))].append(row)

    summary_rows = []
    for (runtime, model), group in sorted(groups.items()):
        successes = [row for row in group if str(row.get("ok", "")).lower() == "true"]
        latencies = [value for row in successes if (value := as_float(row.get("inference_latency_ms", ""))) is not None]
        total_latencies = [value for row in successes if (value := as_float(row.get("total_latency_ms", ""))) is not None]
        session_init_latencies = [
            value for row in group if (value := as_float(row.get("session_init_latency_ms", ""))) is not None
        ]
        labels = [row.get("detected_labels", "") for row in successes]
        label_counter = Counter(labels)
        mode_labels, mode_count = label_counter.most_common(1)[0] if label_counter else ("", 0)
        consistency = mode_count / len(successes) if successes else 0.0
        errors = sorted({row.get("error", "") for row in group if row.get("error", "")})
        summary_rows.append(
            {
                "runtime": runtime,
                "model": model,
                "runs": len(group),
                "success_count": len(successes),
                "detection_consistency": round(consistency, 4),
                "mode_detected_labels": mode_labels,
                "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else "",
                "p50_latency_ms": round(percentile(latencies, 0.50), 2) if latencies else "",
                "p95_latency_ms": round(percentile(latencies, 0.95), 2) if latencies else "",
                "p99_latency_ms": round(percentile(latencies, 0.99), 2) if latencies else "",
                "min_latency_ms": round(min(latencies), 2) if latencies else "",
                "max_latency_ms": round(max(latencies), 2) if latencies else "",
                "avg_total_latency_ms": round(statistics.mean(total_latencies), 2) if total_latencies else "",
                "avg_session_init_latency_ms": round(statistics.mean(session_init_latencies), 2)
                if session_init_latencies
                else "",
                "max_session_init_latency_ms": round(max(session_init_latencies), 2) if session_init_latencies else "",
                "errors": " | ".join(errors),
            }
        )
    return summary_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize local CV runtime benchmark CSV files.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict] = []
    for path in args.inputs:
        with path.open("r", newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "runtime",
        "model",
        "runs",
        "success_count",
        "detection_consistency",
        "mode_detected_labels",
        "avg_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "min_latency_ms",
        "max_latency_ms",
        "avg_total_latency_ms",
        "avg_session_init_latency_ms",
        "max_session_init_latency_ms",
        "errors",
    ]
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summarize(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
