#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelCandidate:
    key: str
    model_id: str
    model_name: str
    model_size: str
    family: str
    runner: str


CANDIDATES: dict[str, ModelCandidate] = {
    "smolvlm2_500m": ModelCandidate(
        key="smolvlm2_500m",
        model_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        model_name="SmolVLM2-500M-Video-Instruct",
        model_size="500M",
        family="SmolVLM2",
        runner="transformers",
    ),
    "smolvlm2_256m": ModelCandidate(
        key="smolvlm2_256m",
        model_id="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        model_name="SmolVLM2-256M-Video-Instruct",
        model_size="256M",
        family="SmolVLM2",
        runner="transformers",
    ),
    "smolvlm2_2b": ModelCandidate(
        key="smolvlm2_2b",
        model_id="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        model_name="SmolVLM2-2.2B-Instruct",
        model_size="2.2B",
        family="SmolVLM2",
        runner="transformers",
    ),
    "fastvlm_05b": ModelCandidate(
        key="fastvlm_05b",
        model_id="apple/FastVLM-0.5B-fp16",
        model_name="FastVLM-0.5B-fp16",
        model_size="0.5B",
        family="FastVLM",
        runner="blocked_fastvlm_runtime",
    ),
}


PROMPTS: dict[str, str] = {
    "object_presence": (
        'Return JSON only:\n{"answer": "yes|no|unknown", "target": "lipstick", '
        '"confidence": 0.0, "reason": "short"}\n'
        "Question: Is there a lipstick in this image?"
    ),
    "person_fall": (
        'Return JSON only:\n{"answer": "yes|no|unknown", "event": '
        '"person_falling_or_lying_down", "confidence": 0.0, "reason": "short"}\n'
        "Question: Does this image show a person falling or lying on the ground?"
    ),
    "abnormal_scene": (
        'Return JSON only:\n{"answer": "yes|no|unknown", "event": '
        '"abnormal_or_attention_needed", "confidence": 0.0, "reason": "short"}\n'
        "Question: Is there anything in this monitoring scene that needs attention?"
    ),
    "general_description": (
        'Return JSON only:\n{"summary": "one sentence", "objects": ["..."], '
        '"attention_needed": true|false}\n'
        "Question: Describe this monitoring frame in one sentence."
    ),
}


FIELDS = [
    "model_name",
    "model_id",
    "model_size",
    "platform",
    "image_path",
    "prompt_type",
    "max_new_tokens",
    "input_resolution",
    "load_latency_ms",
    "latency_total_ms",
    "time_to_first_token_ms",
    "generate_ms",
    "peak_gpu_memory_mb",
    "output_text",
    "json_valid",
    "answer",
    "confidence",
    "error",
    "status",
]


SUMMARY_FIELDS = [
    "model_name",
    "model_id",
    "model_size",
    "platform",
    "requests",
    "success_count",
    "success_rate",
    "json_valid_count",
    "json_valid_rate",
    "avg_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "peak_gpu_memory_mb",
    "qualitative_notes",
    "recommendation",
]


def default_images() -> list[Path]:
    candidates = [
        REPO_ROOT / "results/figures/camera_v05_positive_detection.jpg",
        REPO_ROOT / "results/figures/camera_v05_real_vlm_sample.jpg",
        REPO_ROOT / "results/figures/camera_v05_sample.jpg",
    ]
    return [path for path in candidates if path.exists()]


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resize_image(image: Image.Image, max_width: int | None) -> tuple[Image.Image, str]:
    image = image.convert("RGB")
    if not max_width or image.width <= max_width:
        return image, f"{image.width}x{image.height}"
    height = max(1, round(image.height * (max_width / image.width)))
    return image.resize((max_width, height)), f"{max_width}x{height}"


def extract_json(text: str) -> tuple[bool, dict[str, Any] | None]:
    stripped = text.strip()
    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return True, parsed
    return False, None


def answer_fields(parsed: dict[str, Any] | None) -> tuple[str, str]:
    if not parsed:
        return "", ""
    answer = parsed.get("answer")
    if answer is None:
        answer = parsed.get("summary", "")
    confidence = parsed.get("confidence", "")
    return str(answer), str(confidence)


def dependency_blocked_row(
    candidate: ModelCandidate,
    *,
    platform: str,
    image_path: Path,
    prompt_type: str,
    max_new_tokens: int,
    input_resolution: str,
    error: str,
    status: str = "dependency_blocked",
) -> dict[str, Any]:
    return {
        "model_name": candidate.model_name,
        "model_id": candidate.model_id,
        "model_size": candidate.model_size,
        "platform": platform,
        "image_path": str(image_path),
        "prompt_type": prompt_type,
        "max_new_tokens": max_new_tokens,
        "input_resolution": input_resolution,
        "load_latency_ms": "",
        "latency_total_ms": "",
        "time_to_first_token_ms": "",
        "generate_ms": "",
        "peak_gpu_memory_mb": "",
        "output_text": "",
        "json_valid": False,
        "answer": "",
        "confidence": "",
        "error": error,
        "status": status,
    }


class TransformersImageTextRunner:
    def __init__(self, candidate: ModelCandidate, *, local_files_only: bool) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        load_start = time.perf_counter()
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(candidate.model_id, local_files_only=local_files_only)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        kwargs = {
            "local_files_only": local_files_only,
        }
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                candidate.model_id,
                torch_dtype=dtype,
                **kwargs,
            )
        except TypeError:
            self.model = AutoModelForImageTextToText.from_pretrained(
                candidate.model_id,
                dtype=dtype,
                **kwargs,
            )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.load_latency_ms = round((time.perf_counter() - load_start) * 1000, 2)

    def run(self, image: Image.Image, prompt: str, max_new_tokens: int) -> dict[str, Any]:
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
        decoded = self.processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0]
        peak_memory = ""
        if self.device == "cuda":
            peak_memory = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        return {
            "output_text": decoded.strip(),
            "generate_ms": generate_ms,
            "peak_gpu_memory_mb": peak_memory,
        }


def missing_dependency_message() -> str | None:
    missing: list[str] = []
    for module in ("torch", "transformers"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        return "missing Python dependencies: " + ", ".join(missing)
    return None


def run_candidate(
    candidate: ModelCandidate,
    *,
    images: list[Path],
    prompt_types: list[str],
    platform: str,
    max_new_tokens: int,
    input_width: int | None,
    local_files_only: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if candidate.runner == "blocked_fastvlm_runtime":
        for image_path in images:
            with Image.open(image_path) as image:
                _, input_resolution = resize_image(image, input_width)
            for prompt_type in prompt_types:
                rows.append(
                    dependency_blocked_row(
                        candidate,
                        platform=platform,
                        image_path=image_path,
                        prompt_type=prompt_type,
                        max_new_tokens=max_new_tokens,
                        input_resolution=input_resolution,
                        error=(
                            "FastVLM repo is ml-fastvlm/CoreML-oriented; no stable WSL/RTX "
                            "Transformers path is wired in this benchmark."
                        ),
                    )
                )
        return rows

    dependency_error = missing_dependency_message()
    if dependency_error:
        for image_path in images:
            with Image.open(image_path) as image:
                _, input_resolution = resize_image(image, input_width)
            for prompt_type in prompt_types:
                rows.append(
                    dependency_blocked_row(
                        candidate,
                        platform=platform,
                        image_path=image_path,
                        prompt_type=prompt_type,
                        max_new_tokens=max_new_tokens,
                        input_resolution=input_resolution,
                        error=dependency_error,
                    )
                )
        return rows

    try:
        runner = TransformersImageTextRunner(candidate, local_files_only=local_files_only)
    except Exception as exc:
        status = "dependency_blocked" if "local_files_only" in str(exc) or "not the path" in str(exc) else "failed"
        for image_path in images:
            with Image.open(image_path) as image:
                _, input_resolution = resize_image(image, input_width)
            for prompt_type in prompt_types:
                rows.append(
                    dependency_blocked_row(
                        candidate,
                        platform=platform,
                        image_path=image_path,
                        prompt_type=prompt_type,
                        max_new_tokens=max_new_tokens,
                        input_resolution=input_resolution,
                        error=f"model load failed: {exc}",
                        status=status,
                    )
                )
        return rows

    for image_path in images:
        with Image.open(image_path) as original:
            image, input_resolution = resize_image(original, input_width)
        for prompt_type in prompt_types:
            prompt = PROMPTS[prompt_type]
            total_start = time.perf_counter()
            try:
                result = runner.run(image, prompt, max_new_tokens)
                latency_total_ms = round((time.perf_counter() - total_start) * 1000, 2)
                json_valid, parsed = extract_json(result["output_text"])
                answer, confidence = answer_fields(parsed)
                rows.append(
                    {
                        "model_name": candidate.model_name,
                        "model_id": candidate.model_id,
                        "model_size": candidate.model_size,
                        "platform": platform,
                        "image_path": str(image_path),
                        "prompt_type": prompt_type,
                        "max_new_tokens": max_new_tokens,
                        "input_resolution": input_resolution,
                        "load_latency_ms": runner.load_latency_ms,
                        "latency_total_ms": latency_total_ms,
                        "time_to_first_token_ms": "",
                        "generate_ms": result["generate_ms"],
                        "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
                        "output_text": result["output_text"],
                        "json_valid": json_valid,
                        "answer": answer,
                        "confidence": confidence,
                        "error": "" if json_valid else "model output was not valid JSON",
                        "status": "success" if json_valid else "invalid_json",
                    }
                )
            except RuntimeError as exc:
                status = "oom" if "out of memory" in str(exc).lower() else "failed"
                rows.append(
                    dependency_blocked_row(
                        candidate,
                        platform=platform,
                        image_path=image_path,
                        prompt_type=prompt_type,
                        max_new_tokens=max_new_tokens,
                        input_resolution=input_resolution,
                        error=str(exc),
                        status=status,
                    )
                )
            except Exception as exc:
                rows.append(
                    dependency_blocked_row(
                        candidate,
                        platform=platform,
                        image_path=image_path,
                        prompt_type=prompt_type,
                        max_new_tokens=max_new_tokens,
                        input_resolution=input_resolution,
                        error=str(exc),
                        status="failed",
                    )
                )
    return rows


def percentile(values: list[float], pct: float) -> float | str:
    if not values:
        return ""
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((pct / 100) * (len(values) - 1))))
    return round(values[index], 2)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["model_name"], row["platform"]), []).append(row)

    summaries: list[dict[str, Any]] = []
    for (model_name, platform), group in groups.items():
        successes = [row for row in group if row["status"] == "success"]
        valid_json = [row for row in group if str(row["json_valid"]).lower() == "true"]
        completed = [row for row in group if row["status"] in {"success", "invalid_json"}]
        latencies = [
            float(row["latency_total_ms"])
            for row in completed
            if row["latency_total_ms"] not in {"", None}
        ]
        peak_values = [
            float(row["peak_gpu_memory_mb"])
            for row in group
            if row.get("peak_gpu_memory_mb") not in {"", None}
        ]
        recommendation = "blocker"
        notes = ""
        if not completed:
            statuses = sorted({str(row["status"]) for row in group})
            notes = "No completed inference requests; statuses=" + ",".join(statuses)
        else:
            avg_latency = statistics.mean(latencies)
            json_rate = len(valid_json) / len(group)
            if json_rate < 0.8:
                recommendation = "reject"
                notes = (
                    "Completed real inference but JSON compliance is below the structured verifier threshold; "
                    "latency is useful, output contract is not stable enough."
                )
            elif avg_latency < 1000:
                recommendation = "event_verifier_candidate"
            elif avg_latency < 3000:
                recommendation = "event_verifier_candidate"
            elif avg_latency < 5000:
                recommendation = "rtx_only_candidate"
            else:
                recommendation = "slow_fallback_only"
            if json_rate >= 0.8:
                notes = "Real-time path thresholds: <1s fast verifier, 1-3s semantic review, >5s slow fallback."

        model_id = group[0]["model_id"]
        model_size = group[0]["model_size"]
        summaries.append(
            {
                "model_name": model_name,
                "model_id": model_id,
                "model_size": model_size,
                "platform": platform,
                "requests": len(group),
                "success_count": len(successes),
                "success_rate": round(len(successes) / len(group), 3) if group else 0,
                "json_valid_count": len(valid_json),
                "json_valid_rate": round(len(valid_json) / len(group), 3) if group else 0,
                "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else "",
                "p50_latency_ms": percentile(latencies, 50),
                "p95_latency_ms": percentile(latencies, 95),
                "peak_gpu_memory_mb": round(max(peak_values), 2) if peak_values else "",
                "qualitative_notes": notes,
                "recommendation": recommendation,
            }
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark compact VLM candidates as event-level semantic verifiers.")
    parser.add_argument("--models", default="smolvlm2_500m")
    parser.add_argument("--platform-label", default="rtx_5090_wsl")
    parser.add_argument("--images", nargs="*", type=Path, default=default_images())
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "serving/results/raw/fast_vlm_verifier_benchmark.csv")
    parser.add_argument("--summary-out", type=Path, default=REPO_ROOT / "serving/results/raw/fast_vlm_verifier_summary.csv")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--timeout-s", type=float, default=120.0, help="Recorded for reproducibility; generation is not force-killed.")
    parser.add_argument("--input-width", type=int, default=448)
    parser.add_argument("--prompt-types", default="object_presence,person_fall,abnormal_scene,general_description")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit-images", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_keys = parse_csv_list(args.models)
    prompt_types = parse_csv_list(args.prompt_types)
    unknown_models = [key for key in model_keys if key not in CANDIDATES]
    if unknown_models:
        raise SystemExit(f"unknown model keys: {', '.join(unknown_models)}")
    unknown_prompts = [key for key in prompt_types if key not in PROMPTS]
    if unknown_prompts:
        raise SystemExit(f"unknown prompt types: {', '.join(unknown_prompts)}")

    images = list(args.images)
    if args.image_dir:
        images.extend(sorted(path for path in args.image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}))
    images = [path for path in images if path.exists()]
    if args.limit_images > 0:
        images = images[: args.limit_images]
    if not images:
        raise SystemExit("no input images found")

    rows: list[dict[str, Any]] = []
    for key in model_keys:
        rows.extend(
            run_candidate(
                CANDIDATES[key],
                images=images,
                prompt_types=prompt_types,
                platform=args.platform_label,
                max_new_tokens=args.max_new_tokens,
                input_width=args.input_width,
                local_files_only=args.local_files_only,
            )
        )

    write_csv(args.out, rows, FIELDS)
    write_csv(args.summary_out, summarize(rows), SUMMARY_FIELDS)
    print(f"wrote {len(rows)} rows to {args.out}")
    print(f"wrote summary to {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
