#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from serving.app.local_cv_yolo import YoloTensorRTDetector
from serving.app.local_cv_yolo_cpp import YoloTrtCppDetector


def detections_json(result) -> str:
    return json.dumps([detection.to_dict() for detection in result.detections], ensure_ascii=False)


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


def benchmark_detector(runtime: str, detector, image: Path, iterations: int, init_latency_ms: float | None) -> list[dict]:
    rows = []
    for iteration in range(1, iterations + 1):
        result = detector.detect(image)
        rows.append(
            {
                "runtime": runtime,
                "model_name": result.model,
                "image_path": str(image),
                "iteration": iteration,
                "detected_labels": result.detected_labels_json(),
                "detections": detections_json(result),
                "confidence": result.detections[0].confidence if result.detections else "",
                "inference_latency_ms": result.inference_latency_ms,
                "total_latency_ms": result.total_latency_ms,
                "adapter_init_latency_ms": init_latency_ms,
                "ok": result.ok,
                "error": result.error,
            }
        )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["runtime"]].append(row)
    summary = []
    for runtime, group in sorted(groups.items()):
        successes = [row for row in group if str(row.get("ok")).lower() == "true"]
        inf = [float(row["inference_latency_ms"]) for row in successes if row.get("inference_latency_ms") not in ("", None)]
        total = [float(row["total_latency_ms"]) for row in successes if row.get("total_latency_ms") not in ("", None)]
        labels = [row.get("detected_labels", "") for row in successes]
        mode_labels, mode_count = Counter(labels).most_common(1)[0] if labels else ("", 0)
        errors = sorted({row.get("error", "") for row in group if row.get("error", "")})
        summary.append(
            {
                "runtime": runtime,
                "runs": len(group),
                "success_count": len(successes),
                "detection_consistency": round(mode_count / len(successes), 4) if successes else 0,
                "mode_detected_labels": mode_labels,
                "avg_inference_latency_ms": round(statistics.mean(inf), 2) if inf else "",
                "p50_inference_latency_ms": round(percentile(inf, 0.50), 2) if inf else "",
                "p95_inference_latency_ms": round(percentile(inf, 0.95), 2) if inf else "",
                "p99_inference_latency_ms": round(percentile(inf, 0.99), 2) if inf else "",
                "avg_total_latency_ms": round(statistics.mean(total), 2) if total else "",
                "p50_total_latency_ms": round(percentile(total, 0.50), 2) if total else "",
                "p95_total_latency_ms": round(percentile(total, 0.95), 2) if total else "",
                "p99_total_latency_ms": round(percentile(total, 0.99), 2) if total else "",
                "errors": " | ".join(errors),
            }
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Python TensorRT vs C++ TensorRT YOLO adapters.")
    parser.add_argument("--image", type=Path, default=REPO_ROOT / "results/figures/camera_v05_positive_detection.jpg")
    parser.add_argument("--engine-path", type=Path, default=Path("/home/rainbow/models/vision/yolo_nano/yolov8n_fp16.engine"))
    parser.add_argument("--worker-path", type=Path, default=Path("/home/rainbow/edge-llm-bench/cpp/yolo_trt/build/yolo_trt_worker"))
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "serving/results/raw/local_cv_yolo_cpp_adapter.csv")
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=REPO_ROOT / "serving/results/raw/local_cv_yolo_cpp_adapter_summary.csv",
    )
    args = parser.parse_args()

    rows = []
    python_detector = YoloTensorRTDetector(engine_path=args.engine_path, confidence_threshold=0.2, warmup=args.warmup)
    rows.extend(
        benchmark_detector(
            "python_tensorrt_fp16",
            python_detector,
            args.image,
            args.iterations,
            python_detector.engine_init_latency_ms,
        )
    )
    python_detector.close()

    cpp_detector = YoloTrtCppDetector(worker_path=args.worker_path, engine_path=args.engine_path)
    for _ in range(max(0, args.warmup)):
        cpp_detector.detect(args.image)
    rows.extend(
        benchmark_detector(
            "cpp_tensorrt_worker_fp16",
            cpp_detector,
            args.image,
            args.iterations,
            cpp_detector.engine_init_latency_ms,
        )
    )
    cpp_detector.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "runtime",
        "model_name",
        "image_path",
        "iteration",
        "detected_labels",
        "detections",
        "confidence",
        "inference_latency_ms",
        "total_latency_ms",
        "adapter_init_latency_ms",
        "ok",
        "error",
    ]
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = summarize(rows)
    summary_fields = [
        "runtime",
        "runs",
        "success_count",
        "detection_consistency",
        "mode_detected_labels",
        "avg_inference_latency_ms",
        "p50_inference_latency_ms",
        "p95_inference_latency_ms",
        "p99_inference_latency_ms",
        "avg_total_latency_ms",
        "p50_total_latency_ms",
        "p95_total_latency_ms",
        "p99_total_latency_ms",
        "errors",
    ]
    with args.summary_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
