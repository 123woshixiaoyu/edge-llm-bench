#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from serving.app.local_cv import run_mobilenet_ssd
from serving.app.local_cv_onnx import OnnxLocalCVDetector, run_ssd_mobilenet_onnx
from serving.app.local_cv_tensorrt import run_tensorrt_fp16
from serving.app.local_cv_yolo import YoloOnnxDetector, YoloTensorRTDetector


def detections_json(result) -> str:
    return json.dumps([detection.to_dict() for detection in result.detections], ensure_ascii=False)


def run_once(runtime: str, image_path: Path, args, detector=None):
    if runtime == "opencv_dnn":
        return run_mobilenet_ssd(
            image_path,
            model_dir=args.opencv_model_dir,
            confidence_threshold=args.confidence_threshold,
        )
    if runtime == "onnxruntime":
        return run_ssd_mobilenet_onnx(
            image_path,
            model_path=args.onnx_model,
            confidence_threshold=args.confidence_threshold,
        )
    if runtime == "onnxruntime_cpu_reuse":
        return detector.detect(image_path)
    if runtime == "tensorrt_fp16":
        return run_tensorrt_fp16(image_path, engine_path=args.trt_engine)
    if runtime in ("yolo_onnxruntime_cpu_reuse", "yolo_tensorrt_fp16"):
        return detector.detect(image_path)
    raise ValueError(f"unknown runtime: {runtime}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark local CV runtimes on a fixed image.")
    parser.add_argument(
        "--runtime",
        choices=[
            "opencv_dnn",
            "onnxruntime",
            "onnxruntime_cpu_reuse",
            "tensorrt_fp16",
            "yolo_onnxruntime_cpu_reuse",
            "yolo_tensorrt_fp16",
        ],
        required=True,
    )
    parser.add_argument("--image", type=Path, default=REPO_ROOT / "results/figures/camera_v05_positive_detection.jpg")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--opencv-model-dir", type=Path, default=Path("/home/rainbow/models/vision/mobilenet_ssd"))
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=Path("/home/rainbow/models/vision/ssd_mobilenet_onnx/ssd_mobilenet_v1_10.onnx"),
    )
    parser.add_argument(
        "--trt-engine",
        type=Path,
        default=Path("/home/rainbow/models/vision/ssd_mobilenet_onnx/ssd_mobilenet_v1_10_fp16.engine"),
    )
    parser.add_argument(
        "--yolo-onnx-model",
        type=Path,
        default=Path("/home/rainbow/models/vision/yolo_nano/yolov8n.onnx"),
    )
    parser.add_argument(
        "--yolo-trt-engine",
        type=Path,
        default=Path("/home/rainbow/models/vision/yolo_nano/yolov8n_fp16.engine"),
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.2)
    parser.add_argument("--iou-threshold", type=float, default=0.45)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    detector = None
    session_init_latency_ms = ""
    if args.runtime == "onnxruntime_cpu_reuse":
        detector = OnnxLocalCVDetector(
            model_path=args.onnx_model,
            confidence_threshold=args.confidence_threshold,
            providers=["CPUExecutionProvider"],
            warmup=args.warmup,
        )
        session_init_latency_ms = detector.session_init_latency_ms
    elif args.runtime == "yolo_onnxruntime_cpu_reuse":
        detector = YoloOnnxDetector(
            model_path=args.yolo_onnx_model,
            confidence_threshold=args.confidence_threshold,
            iou_threshold=args.iou_threshold,
            providers=["CPUExecutionProvider"],
            warmup=args.warmup,
        )
        session_init_latency_ms = detector.session_init_latency_ms
    elif args.runtime == "yolo_tensorrt_fp16":
        detector = YoloTensorRTDetector(
            engine_path=args.yolo_trt_engine,
            confidence_threshold=args.confidence_threshold,
            iou_threshold=args.iou_threshold,
            warmup=args.warmup,
        )
        session_init_latency_ms = detector.engine_init_latency_ms

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
        "session_init_latency_ms",
        "ok",
        "error",
    ]
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for iteration in range(1, args.iterations + 1):
            result = run_once(args.runtime, args.image, args, detector)
            row_session_init_latency_ms = session_init_latency_ms
            if args.runtime == "onnxruntime":
                row_session_init_latency_ms = getattr(result, "session_init_latency_ms", "")
            top_confidence = result.detections[0].confidence if result.detections else ""
            writer.writerow(
                {
                    "runtime": args.runtime,
                    "model_name": result.model,
                    "image_path": args.image,
                    "iteration": iteration,
                    "detected_labels": result.detected_labels_json(),
                    "detections": detections_json(result),
                    "confidence": top_confidence,
                    "inference_latency_ms": result.inference_latency_ms,
                    "total_latency_ms": result.total_latency_ms,
                    "session_init_latency_ms": row_session_init_latency_ms,
                    "ok": result.ok,
                    "error": result.error,
                }
            )
            if args.runtime == "tensorrt_fp16" and not result.ok:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
