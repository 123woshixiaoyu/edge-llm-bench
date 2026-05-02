#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def read_by_model(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["model_key"]: row for row in csv.DictReader(f)}


def as_float(row: dict, key: str) -> float | None:
    value = row.get(key)
    if value in ("", None):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt(value: float | None, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if value is not None else ""


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def compare(base: dict[str, dict], edge: dict[str, dict]) -> list[dict]:
    rows = []
    for key in sorted(set(base) & set(edge)):
        b = base[key]
        e = edge[key]
        base_decode = as_float(b, "avg_decode_tps")
        edge_decode = as_float(e, "avg_decode_tps")
        base_prefill = as_float(b, "avg_prompt_tps")
        edge_prefill = as_float(e, "avg_prompt_tps")
        base_mem = as_float(b, "max_peak_gpu_memory_mb")
        edge_mem = as_float(e, "max_peak_gpu_memory_mb")
        rows.append(
            {
                "model_key": key,
                "model_name": e.get("model_name") or b.get("model_name", ""),
                "quantization": e.get("quantization") or b.get("quantization", ""),
                "base_decode_tps": base_decode,
                "edge_decode_tps": edge_decode,
                "decode_ratio": ratio(edge_decode, base_decode),
                "base_prefill_tps": base_prefill,
                "edge_prefill_tps": edge_prefill,
                "prefill_ratio": ratio(edge_prefill, base_prefill),
                "base_mem_mb": base_mem,
                "edge_mem_mb": edge_mem,
            }
        )
    return rows


def write_markdown(rows: list[dict], out: Path, base_name: str, edge_name: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {edge_name} vs {base_name}",
        "",
        "| Model | Quant | "
        f"{base_name} decode tok/s | {edge_name} decode tok/s | Edge/Base decode | "
        f"{base_name} prefill tok/s | {edge_name} prefill tok/s | Edge/Base prefill | "
        f"{base_name} peak MB | {edge_name} peak MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model_name"],
                    row["quantization"],
                    fmt(row["base_decode_tps"]),
                    fmt(row["edge_decode_tps"]),
                    fmt(row["decode_ratio"], 3),
                    fmt(row["base_prefill_tps"]),
                    fmt(row["edge_prefill_tps"]),
                    fmt(row["prefill_ratio"], 3),
                    fmt(row["base_mem_mb"], 0),
                    fmt(row["edge_mem_mb"], 0),
                ]
            )
            + " |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two benchmark summary CSV files by model_key.")
    parser.add_argument("--base", type=Path, required=True, help="Baseline summary CSV, e.g. RTX 5090.")
    parser.add_argument("--edge", type=Path, required=True, help="Edge-device summary CSV, e.g. Jetson.")
    parser.add_argument("--out", type=Path, required=True, help="Markdown comparison output.")
    parser.add_argument("--base-name", default="RTX 5090")
    parser.add_argument("--edge-name", default="Jetson")
    args = parser.parse_args()

    rows = compare(read_by_model(args.base), read_by_model(args.edge))
    write_markdown(rows, args.out, args.base_name, args.edge_name)
    print(f"comparison_rows={len(rows)}")
    print(f"wrote={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
