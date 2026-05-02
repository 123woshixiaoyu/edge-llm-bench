#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def as_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def add_efficiency(row: dict[str, str]) -> dict[str, str]:
    power = as_float(row.get("avg_max_gpu_power_w", ""))
    prompt_tps = as_float(row.get("avg_prompt_tps", ""))
    decode_tps = as_float(row.get("avg_decode_tps", ""))
    out = dict(row)
    if power and prompt_tps is not None:
        out["prompt_tok_per_watt"] = f"{prompt_tps / power:.2f}"
    else:
        out["prompt_tok_per_watt"] = ""
    if power and decode_tps is not None:
        out["decode_tok_per_watt"] = f"{decode_tps / power:.2f}"
    else:
        out["decode_tok_per_watt"] = ""
    return out


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_markdown(rows: list[dict[str, str]]) -> None:
    headers = [
        "model_key",
        "quantization",
        "avg_prompt_tps",
        "avg_decode_tps",
        "avg_max_gpu_power_w",
        "prompt_tok_per_watt",
        "decode_tok_per_watt",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        print("| " + " | ".join(row.get(header, "") for header in headers) + " |")


def main() -> int:
    parser = argparse.ArgumentParser(description="Add tok/s/W efficiency metrics to a summary_by_model CSV.")
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    with args.summary_csv.open("r", encoding="utf-8", newline="") as f:
        rows = [add_efficiency(row) for row in csv.DictReader(f)]

    if args.out:
        write_csv(rows, args.out)
    if args.markdown or not args.out:
        print_markdown(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
