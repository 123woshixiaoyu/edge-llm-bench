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

from serving.app.vision_router import VisionRequest, VisionRouter, detections_json


DEFAULT_CASES = [
    VisionRequest("detect", "allow_remote", "low", request_id="vision_detect_low", expected_route="local"),
    VisionRequest("classify", "allow_remote", "low", request_id="vision_classify_low", expected_route="local"),
    VisionRequest("detect", "local_only", "medium", request_id="vision_detect_private", expected_route="local"),
    VisionRequest("classify", "local_only", "medium", request_id="vision_classify_private", expected_route="local"),
    VisionRequest("scene_description", "allow_remote", "medium", request_id="vision_scene_remote", expected_route="remote"),
    VisionRequest("vqa", "allow_remote", "high", request_id="vision_vqa_high", expected_route="remote"),
    VisionRequest("scene_description", "allow_remote", "high", request_id="vision_scene_high", expected_route="remote"),
    VisionRequest("detect", "allow_remote", "high", request_id="vision_detect_high", expected_route="remote"),
    VisionRequest("vqa", "local_only", "medium", request_id="vision_vqa_private", expected_route="reject"),
    VisionRequest(
        "scene_description",
        "local_only",
        "high",
        request_id="vision_scene_private_high",
        expected_route="reject",
    ),
]


def write_local_cv_baseline(path: Path, capture, local_cv) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "camera_backend",
                "resolution",
                "capture_latency_ms",
                "local_cv_model",
                "model_dir",
                "detected_labels",
                "detections",
                "local_cv_latency_ms",
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
                "local_cv_model": local_cv.model,
                "model_dir": local_cv.model_dir,
                "detected_labels": local_cv.detected_labels_json(),
                "detections": detections_json(local_cv),
                "local_cv_latency_ms": local_cv.inference_latency_ms,
                "local_cv_total_latency_ms": local_cv.total_latency_ms,
                "ok": local_cv.ok,
                "error": local_cv.error,
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="v0.5a camera + local CV + vision routing smoke.")
    parser.add_argument("--out", type=Path, default=SERVING_ROOT / "results/raw/vision_router_smoke.csv")
    parser.add_argument("--baseline-out", type=Path, default=SERVING_ROOT / "results/raw/local_cv_baseline.csv")
    parser.add_argument("--sample-image", type=Path, default=REPO_ROOT / "results/figures/camera_v05_sample.jpg")
    parser.add_argument("--model-dir", type=Path, default=Path("/home/rainbow/models/vision/mobilenet_ssd"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--sensor-id", type=int, default=0)
    args = parser.parse_args()

    router = VisionRouter(model_dir=args.model_dir)
    capture, local_cv = router.capture_and_analyze(
        image_path=args.sample_image,
        device=args.device,
        sensor_id=args.sensor_id,
    )
    write_local_cv_baseline(args.baseline_out, capture, local_cv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "request_id",
        "image_source",
        "task_type",
        "privacy",
        "quality",
        "route",
        "selected_model",
        "remote_is_mock",
        "local_cv_model",
        "detected_labels",
        "detections",
        "capture_latency_ms",
        "local_cv_latency_ms",
        "remote_latency_ms",
        "total_latency_ms",
        "expected_route",
        "match_expected",
        "error",
        "reasons",
    ]

    results = []
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case in DEFAULT_CASES:
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
                "local_cv_model": local_cv.model,
                "detected_labels": local_cv.detected_labels_json(),
                "detections": detections_json(local_cv),
                "capture_latency_ms": capture.capture_latency_ms,
                "local_cv_latency_ms": local_cv.inference_latency_ms,
                "remote_latency_ms": result.remote_latency_ms,
                "total_latency_ms": round(result.total_latency_ms, 2),
                "expected_route": case.expected_route,
                "match_expected": result.match_expected,
                "error": result.error or capture.error or local_cv.error,
                "reasons": json.dumps(result.decision.reasons, ensure_ascii=False),
            }
            writer.writerow(row)
            results.append(row)

    route_counts: dict[str, int] = {}
    for row in results:
        route_counts[row["route"]] = route_counts.get(row["route"], 0) + 1
    matches = sum(1 for row in results if row["match_expected"] is True)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "baseline_out": str(args.baseline_out),
                "sample_image": str(args.sample_image),
                "capture_ok": capture.ok,
                "local_cv_ok": local_cv.ok,
                "route_counts": route_counts,
                "matches": matches,
                "total": len(results),
                "local_cv_latency_ms": local_cv.inference_latency_ms,
                "remote_is_mock": True,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if capture.ok and local_cv.ok and matches == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
