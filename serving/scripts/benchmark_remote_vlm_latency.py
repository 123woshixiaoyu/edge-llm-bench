#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from serving.app.backends.vlm import HttpVLMBackend


CONCISE_SCENE = "Describe the image in one concise sentence. Do not explain your reasoning."
CONCISE_VQA = "Answer briefly. Is there a chair in the image?"
STRUCTURED = "Return JSON with keys: objects, scene, answer. Keep it short."
VERBOSE = "Describe the image for an edge AI routing demo. Include visible objects, scene context, and any notable details."


def image_info(path: Path, resize_width: int | None = None) -> dict[str, Any]:
    with Image.open(path) as image:
        width, height = image.size
    if resize_width and width > resize_width:
        resized_width = resize_width
        resized_height = max(1, int(round(height * (resize_width / width))))
    else:
        resized_width = width
        resized_height = height
    return {
        "image_width": width,
        "image_height": height,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "image_bytes": path.stat().st_size,
    }


def object_mentioned(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in ("chair", "bed", "table", "room", "furniture"))


def contains_reasoning_phrase(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in ("thinking process", "reasoning", "step-by-step", "i will analyze"))


def heuristic_quality(text: str) -> int:
    if not text:
        return 0
    score = 2
    if object_mentioned(text):
        score += 2
    if len(text) <= 320:
        score += 1
    if contains_reasoning_phrase(text):
        score -= 1
    return max(0, min(5, score))


def call_once(
    backend: HttpVLMBackend,
    *,
    image_path: Path,
    prompt: str,
    max_tokens: int,
    resize_width: int | None,
    request_id: str,
    prompt_template: str,
    resize_policy: str,
) -> dict[str, Any]:
    total_start = time.perf_counter()
    info = image_info(image_path, resize_width)
    result = backend.complete_image(
        image_path=image_path,
        prompt=prompt,
        max_tokens=max_tokens,
        resize_width=resize_width,
    )
    total_latency_ms = round((time.perf_counter() - total_start) * 1000, 2)
    timings = result.timings or {}
    text = result.text or ""
    return {
        "request_id": request_id,
        "image_path": str(image_path),
        "image_width": info["image_width"],
        "image_height": info["image_height"],
        "resize_policy": resize_policy,
        "resized_width": info["resized_width"],
        "resized_height": info["resized_height"],
        "image_bytes": info["image_bytes"],
        "prompt_template": prompt_template,
        "max_tokens": max_tokens,
        "client_read_ms": result.client_read_ms,
        "client_resize_ms": result.client_resize_ms,
        "client_base64_encode_ms": result.client_base64_encode_ms,
        "request_payload_bytes": result.request_payload_bytes,
        "network_roundtrip_ms": result.network_roundtrip_ms,
        "server_base64_decode_ms": timings.get("server_base64_decode_ms", ""),
        "server_image_write_ms": timings.get("server_image_write_ms", ""),
        "server_resize_ms": timings.get("server_resize_ms", ""),
        "server_subprocess_ms": timings.get("server_subprocess_ms", ""),
        "server_total_ms": timings.get("server_total_ms", ""),
        "server_endpoint_total_ms": timings.get("server_endpoint_total_ms", ""),
        "total_latency_ms": total_latency_ms,
        "output_chars": len(text),
        "response_preview": text[:240].replace("\n", " "),
        "contains_reasoning_phrase": contains_reasoning_phrase(text),
        "object_mentioned": object_mentioned(text),
        "answer_quality_manual": heuristic_quality(text),
        "ok": result.ok,
        "error": result.error,
        "mode": "subprocess_cli_mode",
    }


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_sample_image() -> Path:
    preferred = REPO_ROOT / "results/figures/camera_v05_real_vlm_sample.jpg"
    fallback = REPO_ROOT / "results/figures/camera_v05_positive_detection.jpg"
    return preferred if preferred.exists() else fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile remote VLM latency over the Jetson-to-RTX HTTP path.")
    parser.add_argument("--base-url", default="http://127.0.0.1:18091")
    parser.add_argument("--image", type=Path, default=default_sample_image())
    parser.add_argument("--mode", choices=["all", "breakdown", "resize-ablation", "prompt-ablation"], default="all")
    parser.add_argument("--breakdown-out", type=Path, default=REPO_ROOT / "serving/results/raw/remote_vlm_latency_breakdown.csv")
    parser.add_argument("--resize-out", type=Path, default=REPO_ROOT / "serving/results/raw/remote_vlm_resize_ablation.csv")
    parser.add_argument("--prompt-out", type=Path, default=REPO_ROOT / "serving/results/raw/remote_vlm_prompt_ablation.csv")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--repeat", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backend = HttpVLMBackend(args.base_url, timeout_s=args.timeout_s)
    image_digest = hashlib.sha256(args.image.read_bytes()).hexdigest()[:12]

    common_fields = [
        "request_id",
        "image_path",
        "image_width",
        "image_height",
        "resize_policy",
        "resized_width",
        "resized_height",
        "image_bytes",
        "prompt_template",
        "max_tokens",
        "client_read_ms",
        "client_resize_ms",
        "client_base64_encode_ms",
        "request_payload_bytes",
        "network_roundtrip_ms",
        "server_base64_decode_ms",
        "server_image_write_ms",
        "server_resize_ms",
        "server_subprocess_ms",
        "server_total_ms",
        "server_endpoint_total_ms",
        "total_latency_ms",
        "output_chars",
        "response_preview",
        "contains_reasoning_phrase",
        "object_mentioned",
        "answer_quality_manual",
        "ok",
        "error",
        "mode",
    ]

    if args.mode in {"all", "breakdown"}:
        rows = [
            call_once(
                backend,
                image_path=args.image,
                prompt=VERBOSE,
                max_tokens=128,
                resize_width=None,
                request_id=f"breakdown_{image_digest}_{idx+1:02d}",
                prompt_template="verbose_baseline",
                resize_policy="original",
            )
            for idx in range(args.repeat)
        ]
        write_rows(args.breakdown_out, rows, common_fields)

    if args.mode in {"all", "resize-ablation"}:
        rows = []
        for policy, width in [("original", None), ("672", 672), ("448", 448), ("336", 336)]:
            rows.append(
                call_once(
                    backend,
                    image_path=args.image,
                    prompt=CONCISE_SCENE,
                    max_tokens=64,
                    resize_width=width,
                    request_id=f"resize_{policy}_{image_digest}",
                    prompt_template="concise_scene",
                    resize_policy=policy,
                )
            )
        write_rows(args.resize_out, rows, common_fields)

    if args.mode in {"all", "prompt-ablation"}:
        prompts = [
            ("verbose_baseline", VERBOSE),
            ("concise_scene", CONCISE_SCENE),
            ("concise_vqa", CONCISE_VQA),
            ("structured_json", STRUCTURED),
        ]
        rows = []
        for name, prompt in prompts:
            for max_tokens in (32, 64, 128):
                rows.append(
                    call_once(
                        backend,
                        image_path=args.image,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        resize_width=448,
                        request_id=f"prompt_{name}_{max_tokens}_{image_digest}",
                        prompt_template=name,
                        resize_policy="448",
                    )
                )
        write_rows(args.prompt_out, rows, common_fields)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
