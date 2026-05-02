#!/usr/bin/env python3
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


FIELDS = [
    "model_key",
    "model_name",
    "quantization",
    "cases",
    "failures",
    "model_size_gib",
    "avg_prompt_tps",
    "avg_decode_tps",
    "max_peak_gpu_memory_mb",
    "avg_max_gpu_power_w",
    "max_gpu_temp_c",
    "avg_load_time_ms",
    "min_gpu_layers_offloaded",
    "max_gpu_layers_total",
]


def nums(rows: list[dict], name: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(name)
        if value in ("", None):
            continue
        try:
            values.append(float(value))
        except ValueError:
            continue
    return values


def fmt_mean(values: list[float], digits: int = 2) -> str:
    return f"{statistics.mean(values):.{digits}f}" if values else ""


def fmt_max(values: list[float], digits: int = 0) -> str:
    return f"{max(values):.{digits}f}" if values else ""


def fmt_min(values: list[float], digits: int = 0) -> str:
    return f"{min(values):.{digits}f}" if values else ""


def summarize(input_csv: Path) -> list[dict]:
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["model_key"]].append(row)

    summary = []
    for key, group in grouped.items():
        summary.append(
            {
                "model_key": key,
                "model_name": group[0].get("model_name", ""),
                "quantization": group[0].get("quantization", ""),
                "cases": len(group),
                "failures": sum(1 for row in group if str(row.get("return_code")) != "0"),
                "model_size_gib": group[0].get("model_size_gib", ""),
                "avg_prompt_tps": fmt_mean(nums(group, "prompt_tps")),
                "avg_decode_tps": fmt_mean(nums(group, "decode_tps")),
                "max_peak_gpu_memory_mb": fmt_max(nums(group, "peak_gpu_memory_mb")),
                "avg_max_gpu_power_w": fmt_mean(nums(group, "max_gpu_power_w")),
                "max_gpu_temp_c": fmt_max(nums(group, "max_gpu_temp_c")),
                "avg_load_time_ms": fmt_mean(nums(group, "load_time_ms")),
                "min_gpu_layers_offloaded": fmt_min(nums(group, "gpu_layers_offloaded")),
                "max_gpu_layers_total": fmt_max(nums(group, "gpu_layers_total")),
            }
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize benchmark CSV by model.")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = summarize(args.input_csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"summary_rows={len(rows)}")
    for row in rows:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
