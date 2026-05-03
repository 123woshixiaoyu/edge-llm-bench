# Project 3 TensorRT Plan

## Goal

Project 3 will optimize the v0.5 local CV baseline for Jetson deployment. It should compare the same model across increasingly optimized runtimes:

1. PyTorch or original framework reference, if practical;
2. ONNX export or equivalent ONNX model;
3. ONNXRuntime benchmark;
4. TensorRT FP16 engine;
5. TensorRT INT8 engine with calibration.

This document is planning only. v0.5a does not implement TensorRT, INT8 calibration, QAT, or video streaming.

## Baseline Model

Chosen baseline for v0.5a:

```text
MobileNet-SSD, OpenCV DNN, PASCAL VOC labels
```

Why this model:

- it is lightweight enough for Jetson Orin Nano;
- it runs without PyTorch or TensorRT;
- it supports object detection rather than only image classification;
- it gives a simple local CV result that the vision router can use immediately;
- its small input size makes it a reasonable first TensorRT optimization target.

Current v0.5a runtime:

- input image: `1280x720` camera frame
- model input: `300x300`
- backend: OpenCV DNN
- local CV inference latency: about `76.6 ms`
- model files: `/home/rainbow/models/vision/mobilenet_ssd/`

## Metrics

Project 3 should measure:

- preprocessing latency;
- model inference latency;
- postprocessing latency;
- total per-frame latency;
- FPS under single-frame and short-burst runs;
- peak memory;
- GPU/CPU utilization when available;
- power and temperature through `tegrastats`;
- detection count and basic label consistency on a fixed image set.

## Phase 1 Update

Phase 1 keeps MobileNet-SSD as the v0.5 local CV baseline and uses an ONNX Model Zoo SSD-MobileNetV1 model for ONNXRuntime / TensorRT feasibility:

```text
/home/rainbow/models/vision/ssd_mobilenet_onnx/ssd_mobilenet_v1_10.onnx
```

This avoids brittle Caffe-to-ONNX conversion while staying in the same SSD-MobileNet detector family. YOLO-nano remains a backup option for a future phase, but it was not needed for Phase 1.

Phase 1 outputs:

- `serving/results/raw/local_cv_runtime_baseline.csv`
- `serving/results/raw/local_cv_onnx_baseline.csv`
- `serving/results/raw/local_cv_tensorrt_fp16.csv`
- `serving/results/raw/local_cv_runtime_summary.csv`
- `serving/docs/project3_tensorrt_report.md`

## ONNX Plan

Current path:

- use ONNX Model Zoo SSD-MobileNetV1 as the ONNX optimization target;
- store ONNX model files under `/home/rainbow/models/vision/`, not in git;
- run ONNXRuntime CPU first to prove model loading, preprocessing, output parsing, and deterministic detections;
- compare detection consistency against the v0.5 positive camera sample.

## TensorRT FP16 Plan

Steps:

1. Use the SSD-MobileNetV1 ONNX model from Phase 1.
2. Install TensorRT packages on Jetson if missing.
3. Build TensorRT FP16 engine on Jetson.
3. Benchmark the same image set used for ONNXRuntime.
4. Compare latency, memory, and power against OpenCV DNN and ONNXRuntime.

Expected benefit: lower inference latency and better throughput on Jetson GPU, with minimal accuracy change relative to FP32/ONNX.

Current Phase 1 blocker:

```text
trtexec not found
TensorRT apt candidate: tensorrt 10.3.0.30-1+cuda12.5
```

## TensorRT INT8 Plan

Steps:

1. Collect a small non-sensitive calibration set from camera frames and public sample images.
2. Build INT8 engine with calibration cache.
3. Verify label consistency against FP16 on the same images.
4. Benchmark latency, FPS, memory, and power.

Expected benefit: lower memory bandwidth and potentially better FPS. Risk is larger accuracy drift, especially on small or low-light objects.

## Risks

- Jetson TensorRT and CUDA versions may constrain ONNX opset support.
- Caffe MobileNet-SSD conversion to ONNX may be less clean than starting from a modern ONNX-native model.
- INT8 calibration needs representative, non-sensitive images.
- Camera exposure and low-light scenes can dominate detection quality, hiding runtime differences.
- TensorRT build time and engine portability can slow iteration.

## Boundary

Project 3 should start only after v0.5a is committed. It should not add VLM, multi-camera input, streaming, or QAT. The first milestone is a clean baseline table, not a large optimization stack.
