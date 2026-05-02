#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


DEFAULT_PROMPT_IDS = [
    "zh_qa_edge_ai",
    "en_qa_quant",
    "code_python_csv",
    "summary_en",
    "reasoning_ops",
]


MODEL_ORDER = {
    "qwen35_08b_ptq_f16": 0,
    "qwen35_08b_ptq_q8_0": 1,
    "qwen35_08b_ptq_q4_k_m": 2,
}


def load_prompts(path: Path) -> dict[str, dict]:
    prompts = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompt = json.loads(line)
                prompts[prompt["id"]] = prompt
    return prompts


def extract_output(text: str) -> str:
    marker = "generate: n_ctx"
    marker_idx = text.find(marker)
    if marker_idx >= 0:
        line_end = text.find("\n", marker_idx)
        if line_end >= 0:
            text = text[line_end + 1 :]

    perf_idx = text.find("common_perf_print:")
    if perf_idx >= 0:
        text = text[:perf_idx]

    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract model answer samples from llama.cpp benchmark logs.")
    parser.add_argument("--bench-csv", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt-ids", default=",".join(DEFAULT_PROMPT_IDS))
    parser.add_argument("--source-device", default="RTX-5090-Laptop-24GB-WSL2")
    args = parser.parse_args()

    prompt_ids = [item.strip() for item in args.prompt_ids.split(",") if item.strip()]
    prompt_id_set = set(prompt_ids)
    prompts = load_prompts(args.prompts)

    with args.bench_csv.open("r", encoding="utf-8", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row.get("prompt_id") in prompt_id_set]

    rows.sort(
        key=lambda row: (
            prompt_ids.index(row["prompt_id"]),
            MODEL_ORDER.get(row["model_key"], 99),
        )
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        for row in rows:
            prompt = prompts[row["prompt_id"]]
            log_path = Path(row["log_path"])
            if not log_path.is_absolute():
                log_path = args.bench_csv.resolve().parents[2] / log_path
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            sample = {
                "source_device": args.source_device,
                "model_key": row["model_key"],
                "model_name": row["model_name"],
                "quantization": row["quantization"],
                "prompt_id": row["prompt_id"],
                "prompt_category": prompt["category"],
                "prompt_language": prompt["language"],
                "prompt": prompt["prompt"],
                "output": extract_output(log_text),
                "decode_tps": row.get("decode_tps", ""),
                "log_path": row["log_path"],
            }
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} samples to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
