#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SERVING_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from serving.app.backends.vlm import HttpVLMBackend
from serving.app.vision_router import VisionRequest, VisionRouter, detections_json


YOLO_TRT_CASES = [
    VisionRequest("detect", "allow_remote", "low", request_id="yolo_trt_detect_low", expected_route="local"),
    VisionRequest("classify", "allow_remote", "low", request_id="yolo_trt_classify_low", expected_route="local"),
    VisionRequest("detect", "local_only", "medium", request_id="yolo_trt_detect_private", expected_route="local"),
    VisionRequest("classify", "local_only", "medium", request_id="yolo_trt_classify_private", expected_route="local"),
    VisionRequest("scene_description", "allow_remote", "medium", request_id="yolo_trt_scene_remote", expected_route="remote"),
    VisionRequest("vqa", "allow_remote", "high", request_id="yolo_trt_vqa_high", expected_route="remote"),
    VisionRequest("scene_description", "allow_remote", "high", request_id="yolo_trt_scene_high", expected_route="remote"),
    VisionRequest("detect", "allow_remote", "high", request_id="yolo_trt_detect_high", expected_route="remote"),
    VisionRequest("vqa", "local_only", "medium", request_id="yolo_trt_vqa_private", expected_route="reject"),
    VisionRequest("scene_description", "local_only", "high", request_id="yolo_trt_scene_private_high", expected_route="reject"),
]


def write_local_cv_baseline(path: Path, capture, local_cv, local_cv_backend: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "camera_backend",
                "resolution",
                "capture_latency_ms",
                "local_cv_backend",
                "local_cv_model",
                "engine_path",
                "fallback_used",
                "model_dir",
                "detected_labels",
                "detections",
                "local_cv_inference_latency_ms",
                "local_cv_total_latency_ms",
                "ok",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_path": capture.image_path,
                "camera_backend": capture.backend,
                "resolution": capture.resolution,
                "capture_latency_ms": capture.capture_latency_ms,
                "local_cv_backend": local_cv_backend,
                "local_cv_model": local_cv.model,
                "engine_path": getattr(local_cv, "engine_path", ""),
                "fallback_used": getattr(local_cv, "fallback_used", False),
                "model_dir": local_cv.model_dir,
                "detected_labels": local_cv.detected_labels_json(),
                "detections": detections_json(local_cv),
                "local_cv_inference_latency_ms": local_cv.inference_latency_ms,
                "local_cv_total_latency_ms": local_cv.total_latency_ms,
                "ok": local_cv.ok,
                "error": local_cv.error,
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="v0.6 vision router smoke with YOLOv8n TensorRT FP16 local CV.")
    parser.add_argument("--sample-image", type=Path, default=REPO_ROOT / "results/figures/camera_v06_yolo_trt_sample.jpg")
    parser.add_argument("--out", type=Path, default=SERVING_ROOT / "results/raw/vision_router_yolo_trt_smoke.csv")
    parser.add_argument("--baseline-out", type=Path, default=SERVING_ROOT / "results/raw/local_cv_yolo_trt_router_baseline.csv")
    parser.add_argument("--model-dir", type=Path, default=Path("/home/rainbow/models/vision/mobilenet_ssd"))
    parser.add_argument("--engine-path", type=Path, default=Path("/home/rainbow/models/vision/yolo_nano/yolov8n_fp16.engine"))
    parser.add_argument("--remote-url", default=None)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sensor-id", type=int, default=0)
    parser.add_argument("--no-fallback", action="store_true")
    args = parser.parse_args()

    remote_backend = HttpVLMBackend(args.remote_url, timeout_s=args.timeout_s) if args.remote_url else None
    router = VisionRouter(
        model_dir=args.model_dir,
        local_cv_backend="yolo_tensorrt_fp16",
        yolo_engine_path=args.engine_path,
        fallback_to_mobilenet=not args.no_fallback,
        remote_backend=remote_backend,
    )
    capture, local_cv = router.capture_and_analyze(
        image_path=args.sample_image,
        device=args.device,
        sensor_id=args.sensor_id,
    )
    write_local_cv_baseline(args.baseline_out, capture, local_cv, "yolo_tensorrt_fp16")

    fieldnames = [
        "request_id",
        "image_source",
        "task_type",
        "privacy",
        "quality",
        "route",
        "selected_model",
        "remote_is_mock",
        "local_cv_backend",
        "local_cv_model",
        "engine_path",
        "fallback_used",
        "detected_labels",
        "detections",
        "capture_latency_ms",
        "local_cv_inference_latency_ms",
        "local_cv_total_latency_ms",
        "remote_latency_ms",
        "total_latency_ms",
        "expected_route",
        "match_expected",
        "error",
        "reasons",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case in YOLO_TRT_CASES:
            result = router.route_existing_frame(case, capture=capture, local_cv=local_cv)
            row = {
                "request_id": result.request_id,
                "image_source": case.image_source,
                "task_type": case.task_type,
                "privacy": case.privacy,
                "quality": case.quality,
                "route": result.decision.route,
                "selected_model": result.decision.selected_model,
                "remote_is_mock": result.decision.remote_is_mock,
                "local_cv_backend": "yolo_tensorrt_fp16",
                "local_cv_model": local_cv.model,
                "engine_path": getattr(local_cv, "engine_path", ""),
                "fallback_used": getattr(local_cv, "fallback_used", False),
                "detected_labels": local_cv.detected_labels_json(),
                "detections": detections_json(local_cv),
                "capture_latency_ms": capture.capture_latency_ms,
                "local_cv_inference_latency_ms": local_cv.inference_latency_ms,
                "local_cv_total_latency_ms": local_cv.total_latency_ms,
                "remote_latency_ms": result.remote_latency_ms,
                "total_latency_ms": round(result.total_latency_ms, 2),
                "expected_route": case.expected_route,
                "match_expected": result.match_expected,
                "error": result.error or capture.error or local_cv.error,
                "reasons": json.dumps(result.decision.reasons, ensure_ascii=False),
            }
            writer.writerow(row)
            rows.append(row)

    route_counts: dict[str, int] = {}
    for row in rows:
        route_counts[row["route"]] = route_counts.get(row["route"], 0) + 1
    matches = sum(1 for row in rows if row["match_expected"] is True)
    local_rows = [row for row in rows if row["route"] == "local"]
    local_uses_yolo = all(row["local_cv_model"] == "yolov8n_tensorrt_fp16" for row in local_rows)
    fallback_used = any(str(row["fallback_used"]).lower() == "true" for row in rows)
    engine_path_ok = all(row["engine_path"] == str(args.engine_path) for row in local_rows)
    remote_rows = [row for row in rows if row["route"] == "remote"]
    remote_rows_mock = all(str(row["remote_is_mock"]).lower() == str(remote_backend is None).lower() for row in remote_rows)
    summary = {
        "out": str(args.out),
        "baseline_out": str(args.baseline_out),
        "capture_ok": capture.ok,
        "local_cv_ok": local_cv.ok,
        "local_cv_model": local_cv.model,
        "detected_labels": local_cv.detected_labels,
        "local_cv_inference_latency_ms": local_cv.inference_latency_ms,
        "local_cv_total_latency_ms": local_cv.total_latency_ms,
        "route_counts": route_counts,
        "matches": matches,
        "total": len(rows),
        "local_uses_yolo": local_uses_yolo,
        "engine_path_ok": engine_path_ok,
        "fallback_used": fallback_used,
        "remote_rows_mock": remote_rows_mock,
        "remote_is_mock": remote_backend is None,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if capture.ok and local_cv.ok and matches == len(rows) and local_uses_yolo and engine_path_ok and not fallback_used and remote_rows_mock else 1


if __name__ == "__main__":
    raise SystemExit(main())
