#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
DEFAULT_PROMPT = """Scene:
This is a fixed camera watching a store entrance.

Normal behavior:
- People may enter and leave calmly.
- Short stops near the doorway are normal.
- The doorway should remain visible and unblocked.

Review-worthy anomalies:
- running or panic movement
- physical conflict
- person falling
- doorway blocked
- unattended object
- after-hours entry
- large unexpected scene change

Candidate proposal:
The cheap monitor detected unusual motion or scene change near the doorway.

Question:
Does this evidence show a review-worthy abnormal event for this scene?

Answer exactly in this format:
FINAL_DECISION: REVIEW, NORMAL, or UNKNOWN
CATEGORY: one short category
SEVERITY: LOW, MEDIUM, or HIGH
REASON: one short reason
"""


DETAIL_FIELDS = [
    "run_id",
    "timestamp",
    "platform_label",
    "model_id",
    "image_path",
    "image_size",
    "max_new_tokens",
    "status",
    "device",
    "torch_version",
    "torch_cuda_available",
    "transformers_version",
    "load_latency_ms",
    "latency_ms",
    "generate_ms",
    "peak_memory_mb",
    "final_decision",
    "category",
    "severity",
    "reason",
    "parse_success",
    "output_text_chars",
    "output_text",
    "error",
    "device_info",
]


SUMMARY_FIELDS = [
    "platform_label",
    "model_id",
    "device",
    "runs",
    "success_count",
    "failure_count",
    "avg_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "parse_success_rate",
    "peak_memory_mb",
    "load_latency_ms",
    "status_set",
    "recommendation",
    "device_info",
    "notes",
]


@dataclass
class ParsedDecision:
    final_decision: str = ""
    category: str = ""
    severity: str = ""
    reason: str = ""
    parse_success: bool = False


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_text_command(command: list[str], timeout_s: float = 5.0) -> str:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout_s)
    except Exception as exc:
        return f"unavailable: {exc}"
    text = (completed.stdout or completed.stderr or "").strip()
    return text.replace("\n", " | ")[:500]


def safe_import_version(module_name: str) -> tuple[bool, str, str]:
    try:
        module = __import__(module_name)
    except Exception as exc:
        return False, "", str(exc)
    return True, str(getattr(module, "__version__", "")), ""


def collect_device_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "uname": run_text_command(["uname", "-a"]) if shutil.which("uname") else "",
        "l4t": "",
        "free_h": run_text_command(["free", "-h"]) if shutil.which("free") else "",
        "df_h": run_text_command(["df", "-h", "."]) if shutil.which("df") else "",
        "nvidia_smi": run_text_command(["nvidia-smi"]) if shutil.which("nvidia-smi") else "not_available",
        "tegrastats_available": bool(shutil.which("tegrastats")),
    }
    nv_tegra = Path("/etc/nv_tegra_release")
    if nv_tegra.exists():
        try:
            info["l4t"] = nv_tegra.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception as exc:
            info["l4t"] = f"unreadable: {exc}"
    torch_ok, torch_version, torch_error = safe_import_version("torch")
    transformers_ok, transformers_version, transformers_error = safe_import_version("transformers")
    info.update(
        {
            "torch_available": torch_ok,
            "torch_version": torch_version,
            "torch_error": torch_error,
            "transformers_available": transformers_ok,
            "transformers_version": transformers_version,
            "transformers_error": transformers_error,
        }
    )
    if torch_ok:
        try:
            import torch

            info["torch_cuda_available"] = bool(torch.cuda.is_available())
            info["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        except Exception as exc:
            info["torch_cuda_available"] = False
            info["cuda_device"] = f"unavailable: {exc}"
    else:
        info["torch_cuda_available"] = False
        info["cuda_device"] = ""
    return info


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resize_image(image: Any, image_size: int) -> tuple[Any, str]:
    image = image.convert("RGB")
    if image_size <= 0:
        return image, f"{image.width}x{image.height}"
    width, height = image.size
    scale = image_size / max(width, height)
    if scale >= 1.0:
        return image, f"{width}x{height}"
    resized = image.resize((max(1, round(width * scale)), max(1, round(height * scale))))
    return resized, f"{resized.width}x{resized.height}"


def parse_decision(text: str) -> ParsedDecision:
    result = ParsedDecision()
    lines = [line.strip() for line in text.replace("\r", "\n").split("\n") if line.strip()]
    for line in lines:
        upper = line.upper()
        if upper.startswith("FINAL_DECISION:"):
            value = line.split(":", 1)[1].strip().upper()
            if value in {"REVIEW", "NORMAL", "UNKNOWN"}:
                result.final_decision = value
        elif upper.startswith("FINAL_ANSWER:"):
            value = line.split(":", 1)[1].strip().upper()
            mapping = {"YES": "REVIEW", "NO": "NORMAL", "UNKNOWN": "UNKNOWN"}
            if value in mapping:
                result.final_decision = mapping[value]
        elif upper.startswith("CATEGORY:"):
            result.category = line.split(":", 1)[1].strip()
        elif upper.startswith("SEVERITY:"):
            value = line.split(":", 1)[1].strip().upper()
            if value in {"LOW", "MEDIUM", "HIGH"}:
                result.severity = value
        elif upper.startswith("REASON:"):
            result.reason = line.split(":", 1)[1].strip()

    if not result.final_decision and len(lines) == 1:
        value = lines[0].strip().upper()
        mapping = {"YES": "REVIEW", "NO": "NORMAL", "UNKNOWN": "UNKNOWN"}
        if value in mapping:
            result.final_decision = mapping[value]

    result.parse_success = result.final_decision in {"REVIEW", "NORMAL", "UNKNOWN"}
    return result


class SmolVLM2Runner:
    def __init__(self, model_id: str, *, local_files_only: bool) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        start = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
        kwargs = {"local_files_only": local_files_only}
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=dtype, **kwargs)
        except TypeError:
            self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype, **kwargs)
        self.model.to(self.device)
        self.model.eval()
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.load_latency_ms = round((time.perf_counter() - start) * 1000, 2)

    def run(self, image: Any, prompt: str, max_new_tokens: int) -> dict[str, Any]:
        torch = self.torch
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        if self.device == "cuda":
            torch.cuda.synchronize()
        generate_ms = round((time.perf_counter() - start) * 1000, 2)
        decoded = self.processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0].strip()
        peak_memory = ""
        if self.device == "cuda":
            peak_memory = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        return {"output_text": decoded, "generate_ms": generate_ms, "peak_memory_mb": peak_memory}


def make_blocker_rows(
    *,
    args: argparse.Namespace,
    device_info: dict[str, Any],
    status: str,
    error: str,
    image_resolution: str = "",
    load_latency_ms: str | float = "",
) -> list[dict[str, Any]]:
    return [
        {
            "run_id": 0,
            "timestamp": now_iso(),
            "platform_label": args.platform_label,
            "model_id": args.model_id,
            "image_path": str(args.image_path),
            "image_size": image_resolution or args.image_size,
            "max_new_tokens": args.max_new_tokens,
            "status": status,
            "device": device_info.get("cuda_device") or ("cuda" if device_info.get("torch_cuda_available") else "cpu_or_unavailable"),
            "torch_version": device_info.get("torch_version", ""),
            "torch_cuda_available": device_info.get("torch_cuda_available", False),
            "transformers_version": device_info.get("transformers_version", ""),
            "load_latency_ms": load_latency_ms,
            "latency_ms": "",
            "generate_ms": "",
            "peak_memory_mb": "",
            "final_decision": "",
            "category": "",
            "severity": "",
            "reason": "",
            "parse_success": False,
            "output_text_chars": "",
            "output_text": "",
            "error": error[:1000],
            "device_info": json.dumps(device_info, sort_keys=True),
        }
    ]


def percentile(values: list[float], pct: float) -> float | str:
    if not values:
        return ""
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((pct / 100) * (len(values) - 1))))
    return round(values[index], 2)


def summarize(rows: list[dict[str, Any]], *, args: argparse.Namespace, device_info: dict[str, Any]) -> list[dict[str, Any]]:
    successful = [row for row in rows if row["status"] == "success"]
    latencies = [float(row["latency_ms"]) for row in successful if row.get("latency_ms") not in {"", None}]
    parse_success = [row for row in successful if str(row.get("parse_success")).lower() == "true"]
    failures = [row for row in rows if row["status"] != "success"]
    peak_values = [float(row["peak_memory_mb"]) for row in rows if row.get("peak_memory_mb") not in {"", None, ""}]
    load_values = [float(row["load_latency_ms"]) for row in rows if row.get("load_latency_ms") not in {"", None, ""}]
    recommendation = "keep_rtx_smolvlm2_fallback"
    notes = ""
    if not successful:
        statuses = sorted({str(row["status"]) for row in rows})
        notes = "No successful Jetson inference; statuses=" + ",".join(statuses)
    else:
        p95 = float(percentile(latencies, 95))
        parse_rate = len(parse_success) / len(successful) if successful else 0.0
        if p95 <= 1500 and parse_rate >= 0.9:
            recommendation = "local_semantic_sentinel_candidate"
            notes = "Green: sub-1.5s P95, stable parsing, no observed OOM."
        elif p95 <= 5000 and parse_rate >= 0.8:
            recommendation = "local_async_low_frequency_candidate"
            notes = "Yellow: viable only for low-frequency async sentinel checks."
        else:
            recommendation = "keep_rtx_smolvlm2_fallback"
            notes = "Red: latency or parse stability is not strong enough for local sentinel use."

    return [
        {
            "platform_label": args.platform_label,
            "model_id": args.model_id,
            "device": device_info.get("cuda_device") or ("cuda" if device_info.get("torch_cuda_available") else "cpu_or_unavailable"),
            "runs": len(rows),
            "success_count": len(successful),
            "failure_count": len(failures),
            "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else "",
            "p50_latency_ms": percentile(latencies, 50),
            "p95_latency_ms": percentile(latencies, 95),
            "parse_success_rate": round(len(parse_success) / len(successful), 3) if successful else 0,
            "peak_memory_mb": round(max(peak_values), 2) if peak_values else "",
            "load_latency_ms": round(max(load_values), 2) if load_values else "",
            "status_set": ",".join(sorted({str(row["status"]) for row in rows})),
            "recommendation": recommendation,
            "device_info": json.dumps(device_info, sort_keys=True),
            "notes": notes,
        }
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SmolVLM2-256M as a Jetson local semantic sentinel.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output-csv", type=Path, default=REPO_ROOT / "serving/results/raw/jetson_smolvlm2_feasibility.csv")
    parser.add_argument("--summary-csv", type=Path, default=REPO_ROOT / "serving/results/raw/jetson_smolvlm2_feasibility_summary.csv")
    parser.add_argument("--platform-label", default="jetson_orin_nano")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device_info = collect_device_info()

    rows: list[dict[str, Any]]
    if not args.image_path.exists():
        rows = make_blocker_rows(args=args, device_info=device_info, status="failed", error=f"image not found: {args.image_path}")
        write_csv(args.output_csv, rows, DETAIL_FIELDS)
        write_csv(args.summary_csv, summarize(rows, args=args, device_info=device_info), SUMMARY_FIELDS)
        return 2

    try:
        from PIL import Image
    except Exception as exc:
        rows = make_blocker_rows(args=args, device_info=device_info, status="dependency_blocked", error=f"missing Pillow: {exc}")
        write_csv(args.output_csv, rows, DETAIL_FIELDS)
        write_csv(args.summary_csv, summarize(rows, args=args, device_info=device_info), SUMMARY_FIELDS)
        return 2

    missing: list[str] = []
    if not device_info.get("torch_available"):
        missing.append(f"torch ({device_info.get('torch_error')})")
    if not device_info.get("transformers_available"):
        missing.append(f"transformers ({device_info.get('transformers_error')})")
    if missing:
        rows = make_blocker_rows(
            args=args,
            device_info=device_info,
            status="dependency_blocked",
            error="missing Python dependencies: " + "; ".join(missing),
        )
        write_csv(args.output_csv, rows, DETAIL_FIELDS)
        write_csv(args.summary_csv, summarize(rows, args=args, device_info=device_info), SUMMARY_FIELDS)
        return 2

    with Image.open(args.image_path) as original:
        image, image_resolution = resize_image(original, args.image_size)

    try:
        runner = SmolVLM2Runner(args.model_id, local_files_only=args.local_files_only)
    except RuntimeError as exc:
        status = "oom" if "out of memory" in str(exc).lower() else "load_failed"
        rows = make_blocker_rows(args=args, device_info=device_info, status=status, error=f"model load failed: {exc}", image_resolution=image_resolution)
        write_csv(args.output_csv, rows, DETAIL_FIELDS)
        write_csv(args.summary_csv, summarize(rows, args=args, device_info=device_info), SUMMARY_FIELDS)
        return 2
    except Exception as exc:
        rows = make_blocker_rows(args=args, device_info=device_info, status="load_failed", error=f"model load failed: {exc}", image_resolution=image_resolution)
        write_csv(args.output_csv, rows, DETAIL_FIELDS)
        write_csv(args.summary_csv, summarize(rows, args=args, device_info=device_info), SUMMARY_FIELDS)
        return 2

    rows = []
    for run_id in range(1, max(args.runs, 1) + 1):
        total_start = time.perf_counter()
        try:
            result = runner.run(image, args.prompt, args.max_new_tokens)
            latency_ms = round((time.perf_counter() - total_start) * 1000, 2)
            parsed = parse_decision(result["output_text"])
            rows.append(
                {
                    "run_id": run_id,
                    "timestamp": now_iso(),
                    "platform_label": args.platform_label,
                    "model_id": args.model_id,
                    "image_path": str(args.image_path),
                    "image_size": image_resolution,
                    "max_new_tokens": args.max_new_tokens,
                    "status": "success" if parsed.parse_success else "parse_failed",
                    "device": runner.device,
                    "torch_version": device_info.get("torch_version", ""),
                    "torch_cuda_available": device_info.get("torch_cuda_available", False),
                    "transformers_version": device_info.get("transformers_version", ""),
                    "load_latency_ms": runner.load_latency_ms,
                    "latency_ms": latency_ms,
                    "generate_ms": result["generate_ms"],
                    "peak_memory_mb": result["peak_memory_mb"],
                    "final_decision": parsed.final_decision,
                    "category": parsed.category,
                    "severity": parsed.severity,
                    "reason": parsed.reason,
                    "parse_success": parsed.parse_success,
                    "output_text_chars": len(result["output_text"]),
                    "output_text": result["output_text"],
                    "error": "" if parsed.parse_success else "could not parse FINAL_DECISION/FINAL_ANSWER",
                    "device_info": json.dumps(device_info, sort_keys=True),
                }
            )
        except RuntimeError as exc:
            status = "oom" if "out of memory" in str(exc).lower() else "runtime_failed"
            rows.append(
                make_blocker_rows(
                    args=args,
                    device_info=device_info,
                    status=status,
                    error=str(exc),
                    image_resolution=image_resolution,
                    load_latency_ms=runner.load_latency_ms,
                )[0]
                | {"run_id": run_id}
            )
        except Exception as exc:
            rows.append(
                make_blocker_rows(
                    args=args,
                    device_info=device_info,
                    status="runtime_failed",
                    error=str(exc),
                    image_resolution=image_resolution,
                    load_latency_ms=runner.load_latency_ms,
                )[0]
                | {"run_id": run_id}
            )

    write_csv(args.output_csv, rows, DETAIL_FIELDS)
    summary = summarize(rows, args=args, device_info=device_info)
    write_csv(args.summary_csv, summary, SUMMARY_FIELDS)
    print(f"wrote {len(rows)} rows to {args.output_csv}")
    print(f"wrote summary to {args.summary_csv}")
    print(json.dumps(summary[0], indent=2))
    return 0 if any(row["status"] == "success" for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
