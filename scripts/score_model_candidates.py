#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


INTEGRATION_SCORE = {
    "integrated_real_backend": 100.0,
    "benchmark_only": 70.0,
    "smoke_only": 60.0,
    "blocked": 0.0,
}

SCORE_FIELDS = [
    "performance_score",
    "quality_score",
    "memory_score",
    "power_score",
    "thermal_score",
    "size_score",
    "stability_score",
    "integration_score",
]


def load_profiles(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        return yaml.safe_load(text)
    except Exception:
        return json.loads(text)


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def matches_filter(row: dict[str, str], filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        if key == "hardware_contains":
            if str(expected).lower() not in row.get("hardware", "").lower():
                return False
            continue
        values = expected if isinstance(expected, list) else [expected]
        if row.get(key, "") not in values:
            return False
    return True


def metric_present(row: dict[str, str], metric: str) -> bool:
    return as_float(row.get(metric)) is not None


def hard_constraint_notes(row: dict[str, str], profile: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    constraints = profile.get("hard_constraints", {})
    failures = as_float(row.get("failures")) or 0.0

    if "failures_equal" in constraints and failures != float(constraints["failures_equal"]):
        notes.append(f"failures={failures:g} violates failures_equal={constraints['failures_equal']}")

    if "max_peak_memory_mb" in constraints:
        value = as_float(row.get("peak_memory_mb"))
        if value is None:
            notes.append("missing required peak_memory_mb")
        elif value > float(constraints["max_peak_memory_mb"]):
            notes.append(f"peak_memory_mb={value:g} exceeds {constraints['max_peak_memory_mb']}")

    if "min_decode_tok_s" in constraints:
        value = as_float(row.get("decode_tok_s"))
        if value is None:
            notes.append("missing required decode_tok_s")
        elif value < float(constraints["min_decode_tok_s"]):
            notes.append(f"decode_tok_s={value:g} below {constraints['min_decode_tok_s']}")

    if "min_quality_score_manual" in constraints:
        value = as_float(row.get("quality_score_manual"))
        if value is None:
            notes.append("missing required quality_score_manual")
        elif value < float(constraints["min_quality_score_manual"]):
            notes.append(f"quality_score_manual={value:g} below {constraints['min_quality_score_manual']}")

    if "privacy_capability_contains" in constraints:
        required = str(constraints["privacy_capability_contains"])
        if required not in row.get("privacy_capability", ""):
            notes.append(f"privacy_capability lacks {required}")

    if "remote_is_mock" in constraints:
        expected = bool(constraints["remote_is_mock"])
        actual = as_bool(row.get("remote_is_mock"))
        if actual is None:
            notes.append("missing remote_is_mock")
        elif actual != expected:
            notes.append(f"remote_is_mock={actual} violates expected {expected}")

    for metric in profile.get("required_metrics", []):
        if not metric_present(row, metric):
            notes.append(f"missing required {metric}")

    required_any = profile.get("required_any_metrics", [])
    if required_any and not any(metric_present(row, metric) for metric in required_any):
        notes.append(f"missing one of required metrics: {','.join(required_any)}")

    return notes


def normalize(value: float | None, values: list[float], *, higher_is_better: bool) -> float:
    if value is None:
        return 50.0
    if not values:
        return 50.0
    low = min(values)
    high = max(values)
    if high == low:
        return 100.0
    if higher_is_better:
        return (value - low) / (high - low) * 100.0
    return (high - value) / (high - low) * 100.0


def quality_score(row: dict[str, str]) -> float:
    value = as_float(row.get("quality_score_manual"))
    if value is None:
        return 50.0
    if value <= 5:
        return value * 20.0
    return min(value, 100.0)


def stability_score(row: dict[str, str]) -> float:
    failures = as_float(row.get("failures")) or 0.0
    return max(0.0, 100.0 - failures * 25.0)


def integration_score(row: dict[str, str]) -> float:
    return INTEGRATION_SCORE.get(row.get("integration_status", ""), 50.0)


def performance_metric(row: dict[str, str]) -> tuple[float | None, bool, str]:
    family = row.get("task_family", "")
    if family == "text":
        return as_float(row.get("decode_tok_s")), True, "decode_tok_s"
    if family == "cv":
        return as_float(row.get("cv_total_latency_ms")) or as_float(row.get("cv_inference_latency_ms")), False, "cv_latency_ms"
    if family == "vlm":
        return as_float(row.get("vlm_latency_ms")), False, "vlm_latency_ms"
    return None, True, "unknown"


def collect_metric_values(rows: list[dict[str, str]], getter) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = getter(row)
        if value is not None:
            values.append(value)
    return values


def score_profile(profile_name: str, profile: dict[str, Any], candidates: list[dict[str, str]]) -> list[dict[str, Any]]:
    applicable = [row for row in candidates if matches_filter(row, profile.get("filters", {}))]
    perf_values = collect_metric_values(applicable, lambda row: performance_metric(row)[0])
    memory_values = collect_metric_values(applicable, lambda row: as_float(row.get("peak_memory_mb")))
    power_values = collect_metric_values(
        applicable, lambda row: as_float(row.get("tok_per_w")) or as_float(row.get("power_w"))
    )
    thermal_values = collect_metric_values(applicable, lambda row: as_float(row.get("max_temp_c")))
    size_values = collect_metric_values(applicable, lambda row: as_float(row.get("model_size_gib")))

    rows: list[dict[str, Any]] = []
    for row in applicable:
        hard_notes = hard_constraint_notes(row, profile)
        hard_pass = not hard_notes
        metric_notes: list[str] = []

        perf_value, perf_higher, perf_name = performance_metric(row)
        if perf_value is None:
            metric_notes.append(f"missing_metric:{perf_name}")
        scores = {
            "performance_score": normalize(perf_value, perf_values, higher_is_better=perf_higher),
            "quality_score": quality_score(row),
            "memory_score": normalize(as_float(row.get("peak_memory_mb")), memory_values, higher_is_better=False),
            "power_score": normalize(
                as_float(row.get("tok_per_w")) or as_float(row.get("power_w")),
                power_values,
                higher_is_better=as_float(row.get("tok_per_w")) is not None,
            ),
            "thermal_score": normalize(as_float(row.get("max_temp_c")), thermal_values, higher_is_better=False),
            "size_score": normalize(as_float(row.get("model_size_gib")), size_values, higher_is_better=False),
            "stability_score": stability_score(row),
            "integration_score": integration_score(row),
        }

        for score_name, source_metric in (
            ("memory_score", "peak_memory_mb"),
            ("power_score", "tok_per_w/power_w"),
            ("thermal_score", "max_temp_c"),
            ("size_score", "model_size_gib"),
        ):
            if scores[score_name] == 50.0:
                metric = source_metric.split("/")[0]
                if as_float(row.get(metric)) is None and score_name in profile.get("weights", {}):
                    metric_notes.append(f"missing_metric:{source_metric}")

        weights = profile.get("weights", {})
        weighted_score = sum(scores[name] * float(weight) for name, weight in weights.items())
        output = {
            "profile": profile_name,
            "candidate_id": row.get("candidate_id", ""),
            "task_family": row.get("task_family", ""),
            "backend_type": row.get("backend_type", ""),
            "hardware": row.get("hardware", ""),
            "model": row.get("model", ""),
            "quant": row.get("quant", ""),
            "runtime": row.get("runtime", ""),
            "hard_constraint_pass": str(hard_pass).lower(),
            "weighted_score": round(weighted_score, 2) if hard_pass else "",
            "rank": "",
            **{name: round(scores[name], 2) for name in SCORE_FIELDS},
            "decision": "eligible" if hard_pass else "excluded",
            "score_notes": "; ".join(hard_notes + metric_notes),
        }
        rows.append(output)

    passing = sorted((row for row in rows if row["hard_constraint_pass"] == "true"), key=lambda r: r["weighted_score"], reverse=True)
    for rank, row in enumerate(passing, start=1):
        row["rank"] = rank
        row["decision"] = "recommend" if rank == 1 else "eligible"
    rows.sort(key=lambda r: (r["hard_constraint_pass"] != "true", r["rank"] or 9999, r["candidate_id"]))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Score model/backend candidates by constraint-aware profiles.")
    parser.add_argument("--candidates", type=Path, default=Path("results/raw/model_selection_candidates.csv"))
    parser.add_argument("--profiles", type=Path, default=Path("configs/model_selection_profiles.yaml"))
    parser.add_argument("--out", type=Path, default=Path("results/raw/model_selection_scores.csv"))
    args = parser.parse_args()

    with args.candidates.open("r", newline="", encoding="utf-8") as f:
        candidates = list(csv.DictReader(f))
    profile_config = load_profiles(args.profiles)

    output_rows: list[dict[str, Any]] = []
    for profile_name, profile in profile_config.get("profiles", {}).items():
        output_rows.extend(score_profile(profile_name, profile, candidates))

    fieldnames = [
        "profile",
        "candidate_id",
        "task_family",
        "backend_type",
        "hardware",
        "model",
        "quant",
        "runtime",
        "hard_constraint_pass",
        "weighted_score",
        "rank",
        *SCORE_FIELDS,
        "decision",
        "score_notes",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
