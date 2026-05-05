#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
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


@dataclass(frozen=True)
class TaskSpec:
    key: str
    question: str
    json_template: str
    fewshot_question: str
    fewshot_answer: str
    yes_no_question: str


CANDIDATES: dict[str, ModelCandidate] = {
    "smolvlm2_500m": ModelCandidate(
        key="smolvlm2_500m",
        model_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        model_name="SmolVLM2-500M-Video-Instruct",
        model_size="500M",
    ),
    "smolvlm2_256m": ModelCandidate(
        key="smolvlm2_256m",
        model_id="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        model_name="SmolVLM2-256M-Video-Instruct",
        model_size="256M",
    ),
    "smolvlm2_2b": ModelCandidate(
        key="smolvlm2_2b",
        model_id="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        model_name="SmolVLM2-2.2B-Instruct",
        model_size="2.2B",
    ),
}


TASKS: dict[str, TaskSpec] = {
    "lipstick_presence": TaskSpec(
        key="lipstick_presence",
        question="Is there a lipstick in this image?",
        json_template='{"answer":"yes|no|unknown","target":"lipstick","confidence":0.0,"reason":"short"}',
        fewshot_question="Is there a person?",
        fewshot_answer='{"answer":"yes","confidence":0.82,"reason":"person visible"}',
        yes_no_question="Is there a lipstick in this image?",
    ),
    "person_fall": TaskSpec(
        key="person_fall",
        question="Does this image show a person falling or lying on the ground?",
        json_template='{"answer":"yes|no|unknown","event":"person_falling_or_lying_down","confidence":0.0,"reason":"short"}',
        fewshot_question="Is there a person lying on the ground?",
        fewshot_answer='{"answer":"no","confidence":0.74,"reason":"no person is visible"}',
        yes_no_question="Does this image show a person falling or lying on the ground?",
    ),
    "abnormal_scene": TaskSpec(
        key="abnormal_scene",
        question="Is there anything in this monitoring scene that needs attention?",
        json_template='{"answer":"yes|no|unknown","event":"abnormal_or_attention_needed","confidence":0.0,"reason":"short"}',
        fewshot_question="Does this hallway need attention?",
        fewshot_answer='{"answer":"unknown","confidence":0.45,"reason":"scene is unclear"}',
        yes_no_question="Is there anything in this monitoring scene that needs attention?",
    ),
    "general_description": TaskSpec(
        key="general_description",
        question="Describe this monitoring frame in one sentence.",
        json_template='{"summary":"one sentence","objects":["..."],"attention_needed":false}',
        fewshot_question="Describe this monitoring frame in one sentence.",
        fewshot_answer='{"summary":"A chair is visible in a room.","objects":["chair"],"attention_needed":false}',
        yes_no_question="Does this monitoring frame need attention?",
    ),
}


PROMPT_VARIANTS = [
    "strict_json",
    "strict_json_fewshot",
    "yes_no_token",
    "final_line",
    "retry_strict",
]


FIELDS = [
    "model_name",
    "model_id",
    "model_size",
    "platform",
    "prompt_variant",
    "image_path",
    "task",
    "max_new_tokens",
    "input_resolution",
    "load_latency_ms",
    "latency_total_ms",
    "generate_ms",
    "peak_gpu_memory_mb",
    "output_text",
    "retry_output_text",
    "strict_json_valid",
    "relaxed_json_valid",
    "yes_no_parse_valid",
    "parsed_answer",
    "expected_answer",
    "semantic_match",
    "parse_method",
    "retry_used",
    "error",
    "status",
]


SUMMARY_FIELDS = [
    "model_name",
    "model_id",
    "model_size",
    "platform",
    "prompt_variant",
    "n",
    "avg_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "strict_json_valid_rate",
    "relaxed_json_valid_rate",
    "yes_no_parse_rate",
    "retry_success_rate",
    "overall_parse_success_rate",
    "semantic_match_rate",
    "failure_examples",
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


def make_prompt(task: TaskSpec, variant: str, previous_output: str | None = None) -> str:
    if variant == "strict_json":
        return f"Return JSON only:\n{task.json_template}\nQuestion: {task.question}"
    if variant == "strict_json_fewshot":
        return (
            f"Example:\nQuestion: {task.fewshot_question}\nAnswer:\n{task.fewshot_answer}\n\n"
            f"Now answer the next question. Return JSON only. No markdown.\n"
            f"Question: {task.question}\nAnswer:"
        )
    if variant == "yes_no_token":
        return (
            "Answer exactly one token from this set:\nYES\nNO\nUNKNOWN\n"
            f"Do not explain.\nQuestion: {task.yes_no_question}"
        )
    if variant == "final_line":
        return (
            "Answer in exactly this format on the final line:\n"
            "FINAL_ANSWER: YES|NO|UNKNOWN\n"
            f"No other final text.\nQuestion: {task.yes_no_question}"
        )
    if variant == "retry_strict":
        if previous_output is None:
            return make_prompt(task, "strict_json")
        return (
            f"Your previous answer was not machine-parseable:\n{previous_output}\n\n"
            "Return ONLY valid minified JSON with keys answer, confidence, reason. "
            "No markdown, no explanation."
        )
    raise ValueError(f"unknown prompt variant: {variant}")


def strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    normalized = str(value).strip().lower()
    if normalized in {"yes", "y", "true"}:
        return "yes"
    if normalized in {"no", "n", "false"}:
        return "no"
    if normalized in {"unknown", "unclear", "maybe"}:
        return "unknown"
    return normalized


def strict_json_parse(text: str) -> tuple[bool, dict[str, Any] | None]:
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        return False, None
    return (True, parsed) if isinstance(parsed, dict) else (False, None)


def relaxed_json_parse(text: str) -> tuple[bool, dict[str, Any] | None]:
    stripped = strip_markdown_fences(text)
    strict_ok, strict_parsed = strict_json_parse(stripped)
    if strict_ok:
        return True, strict_parsed
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        return False, None
    try:
        parsed = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return False, None
    return (True, parsed) if isinstance(parsed, dict) else (False, None)


def yes_no_parse(text: str) -> tuple[bool, str]:
    lines = [line.strip().upper() for line in text.strip().splitlines() if line.strip()]
    candidates: list[str] = []
    for line in lines[:1] + lines[-1:]:
        tokens = re.findall(r"\b(YES|NO|UNKNOWN)\b", line)
        if len(set(tokens)) == 1:
            candidates.append(tokens[0].lower())
    unique = sorted(set(candidates))
    if len(unique) == 1:
        return True, unique[0]
    return False, ""


def final_line_parse(text: str) -> tuple[bool, str]:
    matches = re.findall(r"FINAL_ANSWER\s*:\s*(YES|NO|UNKNOWN)\b", text, flags=re.IGNORECASE)
    unique = sorted({match.lower() for match in matches})
    if len(unique) == 1:
        return True, unique[0]
    return False, ""


def parse_output(text: str) -> dict[str, Any]:
    strict_ok, strict_parsed = strict_json_parse(text)
    relaxed_ok, relaxed_parsed = relaxed_json_parse(text)
    final_ok, final_answer = final_line_parse(text)
    yes_no_ok, yes_no_answer = yes_no_parse(text)
    parsed = strict_parsed or relaxed_parsed

    parsed_answer = ""
    parse_method = "failed"
    if strict_ok and parsed:
        parsed_answer = normalize_answer(parsed.get("answer") or parsed.get("summary"))
        parse_method = "strict_json"
    elif relaxed_ok and parsed:
        parsed_answer = normalize_answer(parsed.get("answer") or parsed.get("summary"))
        parse_method = "relaxed_json"
    elif final_ok:
        parsed_answer = final_answer
        parse_method = "final_line"
    elif yes_no_ok:
        parsed_answer = yes_no_answer
        parse_method = "yes_no"

    return {
        "strict_json_valid": strict_ok,
        "relaxed_json_valid": relaxed_ok,
        "yes_no_parse_valid": yes_no_ok or final_ok,
        "parsed_answer": parsed_answer,
        "parse_method": parse_method,
        "parse_success": parse_method != "failed",
    }


def expected_answer_for(task_key: str) -> str:
    if task_key in {"lipstick_presence", "person_fall"}:
        return "no"
    return ""


def semantic_match(parsed_answer: str, expected_answer: str) -> str:
    if not expected_answer:
        return ""
    if parsed_answer not in {"yes", "no", "unknown"}:
        return "False"
    if parsed_answer == expected_answer:
        return "True"
    return "False"


def missing_dependency_message() -> str | None:
    missing = []
    for module in ("torch", "transformers", "num2words"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        return "missing Python dependencies: " + ", ".join(missing)
    return None


class TransformersImageTextRunner:
    def __init__(self, candidate: ModelCandidate, *, local_files_only: bool) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        load_start = time.perf_counter()
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(candidate.model_id, local_files_only=local_files_only)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                candidate.model_id,
                torch_dtype=dtype,
                local_files_only=local_files_only,
            )
        except TypeError:
            self.model = AutoModelForImageTextToText.from_pretrained(
                candidate.model_id,
                dtype=dtype,
                local_files_only=local_files_only,
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
        decoded = self.processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0].strip()
        peak_memory = ""
        if self.device == "cuda":
            peak_memory = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        return {
            "output_text": decoded,
            "generate_ms": generate_ms,
            "peak_gpu_memory_mb": peak_memory,
        }


def error_row(
    candidate: ModelCandidate,
    *,
    platform: str,
    image_path: Path,
    task: str,
    prompt_variant: str,
    max_new_tokens: int,
    input_resolution: str,
    error: str,
    status: str,
) -> dict[str, Any]:
    return {
        "model_name": candidate.model_name,
        "model_id": candidate.model_id,
        "model_size": candidate.model_size,
        "platform": platform,
        "prompt_variant": prompt_variant,
        "image_path": str(image_path),
        "task": task,
        "max_new_tokens": max_new_tokens,
        "input_resolution": input_resolution,
        "load_latency_ms": "",
        "latency_total_ms": "",
        "generate_ms": "",
        "peak_gpu_memory_mb": "",
        "output_text": "",
        "retry_output_text": "",
        "strict_json_valid": False,
        "relaxed_json_valid": False,
        "yes_no_parse_valid": False,
        "parsed_answer": "",
        "expected_answer": expected_answer_for(task),
        "semantic_match": "",
        "parse_method": "failed",
        "retry_used": False,
        "error": error,
        "status": status,
    }


def run_one(
    *,
    candidate: ModelCandidate,
    runner: TransformersImageTextRunner,
    image: Image.Image,
    image_path: Path,
    input_resolution: str,
    task: TaskSpec,
    prompt_variant: str,
    platform: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    prompt = make_prompt(task, prompt_variant)
    start = time.perf_counter()
    retry_used = False
    retry_output = ""
    result = runner.run(image, prompt, max_new_tokens)
    output_text = result["output_text"]
    parse = parse_output(output_text)

    if prompt_variant == "retry_strict" and not parse["relaxed_json_valid"]:
        retry_used = True
        retry_prompt = make_prompt(task, "retry_strict", previous_output=output_text)
        retry_result = runner.run(image, retry_prompt, max_new_tokens)
        retry_output = retry_result["output_text"]
        retry_parse = parse_output(retry_output)
        if retry_parse["parse_success"]:
            parse = retry_parse
            result = retry_result
            parse["parse_method"] = "retry"

    latency_total_ms = round((time.perf_counter() - start) * 1000, 2)
    parse_success = bool(parse["parse_success"])
    expected_answer = expected_answer_for(task.key)
    semantic_ok = semantic_match(parse["parsed_answer"], expected_answer)
    return {
        "model_name": candidate.model_name,
        "model_id": candidate.model_id,
        "model_size": candidate.model_size,
        "platform": platform,
        "prompt_variant": prompt_variant,
        "image_path": str(image_path),
        "task": task.key,
        "max_new_tokens": max_new_tokens,
        "input_resolution": input_resolution,
        "load_latency_ms": runner.load_latency_ms,
        "latency_total_ms": latency_total_ms,
        "generate_ms": result["generate_ms"],
        "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
        "output_text": output_text,
        "retry_output_text": retry_output,
        "strict_json_valid": parse["strict_json_valid"],
        "relaxed_json_valid": parse["relaxed_json_valid"],
        "yes_no_parse_valid": parse["yes_no_parse_valid"],
        "parsed_answer": parse["parsed_answer"],
        "expected_answer": expected_answer,
        "semantic_match": semantic_ok,
        "parse_method": parse["parse_method"],
        "retry_used": retry_used,
        "error": "" if parse_success else "output did not match requested protocol",
        "status": "success" if parse_success else "parse_failed",
    }


def run_candidate(
    candidate: ModelCandidate,
    *,
    images: list[Path],
    tasks: list[str],
    prompt_variants: list[str],
    platform: str,
    max_new_tokens: int,
    input_width: int | None,
    local_files_only: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dependency_error = missing_dependency_message()
    if dependency_error:
        for image_path in images:
            with Image.open(image_path) as image:
                _, input_resolution = resize_image(image, input_width)
            for task_key in tasks:
                for prompt_variant in prompt_variants:
                    rows.append(
                        error_row(
                            candidate,
                            platform=platform,
                            image_path=image_path,
                            task=task_key,
                            prompt_variant=prompt_variant,
                            max_new_tokens=max_new_tokens,
                            input_resolution=input_resolution,
                            error=dependency_error,
                            status="dependency_blocked",
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
            for task_key in tasks:
                for prompt_variant in prompt_variants:
                    rows.append(
                        error_row(
                            candidate,
                            platform=platform,
                            image_path=image_path,
                            task=task_key,
                            prompt_variant=prompt_variant,
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
        for task_key in tasks:
            task = TASKS[task_key]
            for prompt_variant in prompt_variants:
                try:
                    rows.append(
                        run_one(
                            candidate=candidate,
                            runner=runner,
                            image=image,
                            image_path=image_path,
                            input_resolution=input_resolution,
                            task=task,
                            prompt_variant=prompt_variant,
                            platform=platform,
                            max_new_tokens=max_new_tokens,
                        )
                    )
                except RuntimeError as exc:
                    status = "oom" if "out of memory" in str(exc).lower() else "failed"
                    rows.append(
                        error_row(
                            candidate,
                            platform=platform,
                            image_path=image_path,
                            task=task_key,
                            prompt_variant=prompt_variant,
                            max_new_tokens=max_new_tokens,
                            input_resolution=input_resolution,
                            error=str(exc),
                            status=status,
                        )
                    )
                except Exception as exc:
                    rows.append(
                        error_row(
                            candidate,
                            platform=platform,
                            image_path=image_path,
                            task=task_key,
                            prompt_variant=prompt_variant,
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


def rate(count: int, total: int) -> float:
    return round(count / total, 3) if total else 0.0


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["model_name"], row["platform"], row["prompt_variant"]), []).append(row)

    summaries: list[dict[str, Any]] = []
    for (model_name, platform, prompt_variant), group in groups.items():
        latencies = [
            float(row["latency_total_ms"])
            for row in group
            if row.get("latency_total_ms") not in {"", None}
        ]
        strict_count = sum(str(row["strict_json_valid"]).lower() == "true" for row in group)
        relaxed_count = sum(str(row["relaxed_json_valid"]).lower() == "true" for row in group)
        yes_no_count = sum(str(row["yes_no_parse_valid"]).lower() == "true" for row in group)
        retry_rows = [row for row in group if str(row["retry_used"]).lower() == "true"]
        retry_success = sum(row["parse_method"] == "retry" for row in retry_rows)
        parse_success = sum(row["status"] == "success" for row in group)
        parse_rate = rate(parse_success, len(group))
        semantic_rows = [row for row in group if row.get("semantic_match") not in {"", None}]
        semantic_ok = sum(str(row.get("semantic_match")).lower() == "true" for row in semantic_rows)
        semantic_rate = rate(semantic_ok, len(semantic_rows)) if semantic_rows else ""
        avg_latency = round(statistics.mean(latencies), 2) if latencies else ""

        semantic_gate = semantic_rate == "" or semantic_rate >= 0.8
        if parse_rate >= 0.9 and avg_latency != "" and avg_latency < 1000 and semantic_gate:
            recommendation = "event_verifier_candidate"
        elif parse_rate >= 0.6 and (semantic_rate == "" or semantic_rate >= 0.6):
            recommendation = "usable_with_retry_or_human_review"
        else:
            recommendation = "not_suitable_for_automated_verifier"

        failures = [str(row["output_text"] or row["error"])[:80] for row in group if row["status"] != "success"]
        summaries.append(
            {
                "model_name": model_name,
                "model_id": group[0]["model_id"],
                "model_size": group[0]["model_size"],
                "platform": platform,
                "prompt_variant": prompt_variant,
                "n": len(group),
                "avg_latency_ms": avg_latency,
                "p50_latency_ms": percentile(latencies, 50),
                "p95_latency_ms": percentile(latencies, 95),
                "strict_json_valid_rate": rate(strict_count, len(group)),
                "relaxed_json_valid_rate": rate(relaxed_count, len(group)),
                "yes_no_parse_rate": rate(yes_no_count, len(group)),
                "retry_success_rate": rate(retry_success, len(retry_rows)) if retry_rows else "",
                "overall_parse_success_rate": parse_rate,
                "semantic_match_rate": semantic_rate,
                "failure_examples": " | ".join(failures[:3]),
                "recommendation": recommendation,
            }
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SmolVLM2 output-protocol robustness.")
    parser.add_argument("--models", default="smolvlm2_500m,smolvlm2_256m")
    parser.add_argument("--platform-label", default="rtx_5090_wsl")
    parser.add_argument("--images", nargs="*", type=Path, default=default_images())
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "serving/results/raw/vlm_output_robustness_benchmark.csv")
    parser.add_argument("--summary-out", type=Path, default=REPO_ROOT / "serving/results/raw/vlm_output_robustness_summary.csv")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--timeout-s", type=float, default=120.0, help="Recorded for reproducibility; generation is not force-killed.")
    parser.add_argument("--input-width", type=int, default=448)
    parser.add_argument("--tasks", default="lipstick_presence,person_fall,abnormal_scene,general_description")
    parser.add_argument("--prompt-variants", default="strict_json,strict_json_fewshot,yes_no_token,final_line,retry_strict")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit-images", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_keys = parse_csv_list(args.models)
    tasks = parse_csv_list(args.tasks)
    prompt_variants = parse_csv_list(args.prompt_variants)
    unknown_models = [key for key in model_keys if key not in CANDIDATES]
    if unknown_models:
        raise SystemExit(f"unknown model keys: {', '.join(unknown_models)}")
    unknown_tasks = [key for key in tasks if key not in TASKS]
    if unknown_tasks:
        raise SystemExit(f"unknown tasks: {', '.join(unknown_tasks)}")
    unknown_variants = [key for key in prompt_variants if key not in PROMPT_VARIANTS]
    if unknown_variants:
        raise SystemExit(f"unknown prompt variants: {', '.join(unknown_variants)}")

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
                tasks=tasks,
                prompt_variants=prompt_variants,
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
