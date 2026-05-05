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


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_WEIGHT_DIR = REPO_ROOT / "runtime_data/yolo_world_weights"


@dataclass(frozen=True)
class ModelCandidate:
    key: str
    ultralytics_name: str
    display_name: str
    size: str


CANDIDATES: dict[str, ModelCandidate] = {
    "yolo_worldv2s": ModelCandidate(
        key="yolo_worldv2s",
        ultralytics_name="yolov8s-worldv2.pt",
        display_name="YOLO-World v2 small",
        size="small",
    ),
    "yolo_worldv2m": ModelCandidate(
        key="yolo_worldv2m",
        ultralytics_name="yolov8m-worldv2.pt",
        display_name="YOLO-World v2 medium",
        size="medium",
    ),
}


CLASS_SETS: dict[str, list[str]] = {
    "base_coco_like": ["person", "chair", "bowl", "bottle", "bed"],
    "custom_objects": ["lipstick", "package", "tool", "phone", "key"],
    "relation_proxy": ["object inside bowl", "item in bowl", "small object", "container with object"],
}


FIELDS = [
    "model_name",
    "model_key",
    "model_size",
    "platform",
    "image_path",
    "class_set_name",
    "class_prompt",
    "input_size",
    "latency_total_ms",
    "preprocess_ms",
    "inference_ms",
    "postprocess_ms",
    "peak_gpu_memory_mb",
    "detections_json",
    "detected_labels",
    "top_confidence",
    "success",
    "error",
    "status",
]


SUMMARY_FIELDS = [
    "model_name",
    "model_key",
    "model_size",
    "platform",
    "class_set_name",
    "input_size",
    "runs",
    "success_rate",
    "avg_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "avg_inference_ms",
    "peak_gpu_memory_mb",
    "detected_label_set",
    "qualitative_hit_miss_notes",
    "recommendation",
]


def default_images() -> list[Path]:
    candidates = [
        REPO_ROOT / "results/figures/camera_v05_positive_detection.jpg",
        REPO_ROOT / "results/figures/camera_v05_real_vlm_sample.jpg",
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


def resolve_model_path(candidate: ModelCandidate) -> str:
    runtime_path = RUNTIME_WEIGHT_DIR / candidate.ultralytics_name
    if runtime_path.exists():
        return str(runtime_path)
    repo_root_path = REPO_ROOT / candidate.ultralytics_name
    if repo_root_path.exists():
        return str(repo_root_path)
    return candidate.ultralytics_name


def dependency_error() -> str | None:
    missing = []
    for module in ("torch", "ultralytics"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        return "missing Python dependencies: " + ", ".join(missing)
    return None


def error_row(
    candidate: ModelCandidate,
    *,
    platform: str,
    image_path: Path,
    class_set_name: str,
    class_prompt: list[str],
    input_size: int,
    error: str,
    status: str,
) -> dict[str, Any]:
    return {
        "model_name": candidate.display_name,
        "model_key": candidate.key,
        "model_size": candidate.size,
        "platform": platform,
        "image_path": str(image_path),
        "class_set_name": class_set_name,
        "class_prompt": "|".join(class_prompt),
        "input_size": input_size,
        "latency_total_ms": "",
        "preprocess_ms": "",
        "inference_ms": "",
        "postprocess_ms": "",
        "peak_gpu_memory_mb": "",
        "detections_json": "[]",
        "detected_labels": "",
        "top_confidence": "",
        "success": False,
        "error": error,
        "status": status,
    }


def detections_from_result(result: Any, classes: list[str]) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return detections
    xyxy = boxes.xyxy.detach().cpu().tolist()
    confs = boxes.conf.detach().cpu().tolist() if boxes.conf is not None else []
    cls_ids = boxes.cls.detach().cpu().tolist() if boxes.cls is not None else []
    for index, box in enumerate(xyxy):
        cls_index = int(cls_ids[index]) if index < len(cls_ids) else -1
        label = classes[cls_index] if 0 <= cls_index < len(classes) else str(cls_index)
        confidence = float(confs[index]) if index < len(confs) else 0.0
        detections.append(
            {
                "label": label,
                "confidence": round(confidence, 4),
                "box": [round(float(value), 2) for value in box],
            }
        )
    return detections


def run_model(
    candidate: ModelCandidate,
    *,
    images: list[Path],
    class_sets: list[str],
    platform: str,
    imgsz: int,
    conf: float,
    device: str | None,
    warmup: int,
) -> list[dict[str, Any]]:
    dep_error = dependency_error()
    rows: list[dict[str, Any]] = []
    if dep_error:
        for image_path in images:
            for class_set_name in class_sets:
                classes = CLASS_SETS[class_set_name]
                rows.append(
                    error_row(
                        candidate,
                        platform=platform,
                        image_path=image_path,
                        class_set_name=class_set_name,
                        class_prompt=classes,
                        input_size=imgsz,
                        error=dep_error,
                        status="dependency_blocked",
                    )
                )
        return rows

    import torch
    from ultralytics import YOLO

    for class_set_name in class_sets:
        classes = CLASS_SETS[class_set_name]
        try:
            # Reinitialize per class set so offline vocabulary setup stays separate from timed inference.
            # This also avoids text-embedding device mismatches after a previous GPU prediction.
            model = YOLO(resolve_model_path(candidate))
            model.set_classes(classes)
        except Exception as exc:
            for image_path in images:
                rows.append(
                    error_row(
                        candidate,
                        platform=platform,
                        image_path=image_path,
                        class_set_name=class_set_name,
                        class_prompt=classes,
                        input_size=imgsz,
                        error=f"set_classes failed: {exc}",
                        status="failed",
                    )
                )
            continue

        for _ in range(max(0, warmup)):
            try:
                model.predict(
                    source=str(images[0]),
                    imgsz=imgsz,
                    conf=conf,
                    device=device,
                    verbose=False,
                )
            except Exception:
                break

        for image_path in images:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                results = model.predict(
                    source=str(image_path),
                    imgsz=imgsz,
                    conf=conf,
                    device=device,
                    verbose=False,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                latency_total_ms = round((time.perf_counter() - start) * 1000, 2)
                result = results[0]
                speed = getattr(result, "speed", {}) or {}
                detections = detections_from_result(result, classes)
                labels = sorted({det["label"] for det in detections})
                top_confidence = max((det["confidence"] for det in detections), default="")
                peak_memory = ""
                if torch.cuda.is_available():
                    peak_memory = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
                rows.append(
                    {
                        "model_name": candidate.display_name,
                        "model_key": candidate.key,
                        "model_size": candidate.size,
                        "platform": platform,
                        "image_path": str(image_path),
                        "class_set_name": class_set_name,
                        "class_prompt": "|".join(classes),
                        "input_size": imgsz,
                        "latency_total_ms": latency_total_ms,
                        "preprocess_ms": round(float(speed.get("preprocess", 0.0)), 2),
                        "inference_ms": round(float(speed.get("inference", 0.0)), 2),
                        "postprocess_ms": round(float(speed.get("postprocess", 0.0)), 2),
                        "peak_gpu_memory_mb": peak_memory,
                        "detections_json": json.dumps(detections, ensure_ascii=True),
                        "detected_labels": "|".join(labels),
                        "top_confidence": top_confidence,
                        "success": True,
                        "error": "",
                        "status": "success",
                    }
                )
            except RuntimeError as exc:
                status = "oom" if "out of memory" in str(exc).lower() else "failed"
                rows.append(
                    error_row(
                        candidate,
                        platform=platform,
                        image_path=image_path,
                        class_set_name=class_set_name,
                        class_prompt=classes,
                        input_size=imgsz,
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
                        class_set_name=class_set_name,
                        class_prompt=classes,
                        input_size=imgsz,
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


def qualitative_notes(class_set_name: str, labels: set[str], success_rate: float) -> str:
    if success_rate == 0:
        return "No successful inference."
    if class_set_name == "base_coco_like":
        if labels:
            return "Detects common class prompts: " + ", ".join(sorted(labels))
        return "Ran successfully but did not detect the expected common-object prompts."
    if class_set_name == "custom_objects":
        if labels:
            return "Detected custom prompts: " + ", ".join(sorted(labels))
        return "No custom-object hits on current public samples; this may reflect image content."
    if class_set_name == "relation_proxy":
        if labels:
            return "Some relation-proxy prompts fired, but phrases should be validated carefully."
        return "No relation-proxy hits; relation events likely still need ROI/change plus VLM verification."
    return ""


def recommendation_for(
    class_set_name: str,
    avg_latency: float | str,
    labels: set[str],
    success_rate: float,
    platform: str,
) -> str:
    if success_rate == 0 or avg_latency == "":
        return "blocked"
    latency = float(avg_latency)
    has_hits = bool(labels)
    is_jetson = "jetson" in platform.lower()
    if is_jetson and class_set_name == "base_coco_like" and latency < 100 and has_hits:
        return "replace_yolov8_candidate"
    if class_set_name == "custom_objects" and latency < 300 and has_hits:
        return "custom_trigger_candidate"
    if latency < 300:
        return "low_frequency_only"
    if latency >= 300:
        return "low_frequency_only"
    return "reject"


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["model_name"]),
            str(row["model_key"]),
            str(row["platform"]),
            str(row["class_set_name"]),
            str(row["input_size"]),
        )
        groups.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for (model_name, model_key, platform, class_set_name, input_size), group in groups.items():
        successes = [row for row in group if str(row["success"]).lower() == "true"]
        latencies = [float(row["latency_total_ms"]) for row in successes if row["latency_total_ms"] not in {"", None}]
        inferences = [float(row["inference_ms"]) for row in successes if row["inference_ms"] not in {"", None}]
        peak_values = [float(row["peak_gpu_memory_mb"]) for row in successes if row["peak_gpu_memory_mb"] not in {"", None}]
        label_set: set[str] = set()
        for row in successes:
            label_set.update(label for label in str(row["detected_labels"]).split("|") if label)
        success_rate = round(len(successes) / len(group), 3) if group else 0.0
        avg_latency = round(statistics.mean(latencies), 2) if latencies else ""
        notes = qualitative_notes(class_set_name, label_set, success_rate)
        summaries.append(
            {
                "model_name": model_name,
                "model_key": model_key,
                "model_size": group[0]["model_size"],
                "platform": platform,
                "class_set_name": class_set_name,
                "input_size": input_size,
                "runs": len(group),
                "success_rate": success_rate,
                "avg_latency_ms": avg_latency,
                "p50_latency_ms": percentile(latencies, 50),
                "p95_latency_ms": percentile(latencies, 95),
                "avg_inference_ms": round(statistics.mean(inferences), 2) if inferences else "",
                "peak_gpu_memory_mb": round(max(peak_values), 2) if peak_values else "",
                "detected_label_set": "|".join(sorted(label_set)),
                "qualitative_hit_miss_notes": notes,
                "recommendation": recommendation_for(class_set_name, avg_latency, label_set, success_rate, platform),
            }
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark YOLO-World as an open-vocabulary monitoring trigger detector.")
    parser.add_argument("--models", default="yolo_worldv2s")
    parser.add_argument("--platform-label", default="rtx_5090_wsl")
    parser.add_argument("--images", nargs="*", type=Path, default=default_images())
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--class-set", default="all", help="base_coco_like/custom_objects/relation_proxy/all or comma list")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "serving/results/raw/yolo_world_benchmark.csv")
    parser.add_argument("--summary-out", type=Path, default=REPO_ROOT / "serving/results/raw/yolo_world_summary.csv")
    parser.add_argument("--export-check", action="store_true", help="Reserved for explicit export feasibility runs; normal benchmark does not export.")
    parser.add_argument("--limit-images", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_keys = parse_csv_list(args.models)
    unknown_models = [key for key in model_keys if key not in CANDIDATES]
    if unknown_models:
        raise SystemExit(f"unknown model keys: {', '.join(unknown_models)}")
    if args.class_set == "all":
        class_sets = list(CLASS_SETS)
    else:
        class_sets = parse_csv_list(args.class_set)
    unknown_class_sets = [key for key in class_sets if key not in CLASS_SETS]
    if unknown_class_sets:
        raise SystemExit(f"unknown class sets: {', '.join(unknown_class_sets)}")
    if args.export_check:
        print("export feasibility is documented separately; this benchmark does not create ONNX or TensorRT artifacts")

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
            run_model(
                CANDIDATES[key],
                images=images,
                class_sets=class_sets,
                platform=args.platform_label,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                warmup=args.warmup,
            )
        )
    write_csv(args.out, rows, FIELDS)
    write_csv(args.summary_out, summarize(rows), SUMMARY_FIELDS)
    print(f"wrote {len(rows)} rows to {args.out}")
    print(f"wrote summary to {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
