# Project 3 TensorRT Plan

## Goal

Project 3 optimizes the v0.5 local CV path for Jetson deployment. The plan is to compare practical detector runtimes rather than force one model through every toolchain:

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
- its small input size makes it a stable router baseline.

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

## Phase 1 / 1.5 Update

Phase 1 keeps MobileNet-SSD as the v0.5 local CV baseline and uses an ONNX Model Zoo SSD-MobileNetV1 model for ONNXRuntime / TensorRT feasibility:

```text
/home/rainbow/models/vision/ssd_mobilenet_onnx/ssd_mobilenet_v1_10.onnx
```

This avoids brittle Caffe-to-ONNX conversion while staying in the same SSD-MobileNet detector family. Phase 1.5 then showed that ONNXRuntime session reuse removes a large serving adapter overhead: total latency dropped from `4929.98 ms` to `49.04 ms`.

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

Phase 2 decision:

SSD-MobileNetV1 ONNX was attempted after TensorRT was installed, but TensorRT 10.3 failed to create an engine with an internal Myelin graph compatibility error:

```text
Device to shape host node should not be folded into myelin.
Engine could not be created from network.
```

The v0.5 router baseline therefore remains MobileNet-SSD, while the Project 3 TensorRT optimization object switches to YOLOv8n ONNX:

```text
/home/rainbow/models/vision/yolo_nano/yolov8n.onnx
```

YOLOv8n was chosen because it is small, detects common COCO objects, and follows a mature ONNX -> TensorRT deployment path on Jetson.

Steps:

1. Keep MobileNet-SSD/OpenCV DNN as the v0.5 router baseline.
2. Build YOLOv8n TensorRT FP16 engine on Jetson.
3. Benchmark YOLOv8n ONNXRuntime CPU reuse and YOLOv8n TensorRT FP16 on the same camera sample.
4. Compare latency and detection consistency within the YOLO model family.

Expected benefit: lower inference latency and better throughput on Jetson GPU, with minimal accuracy change relative to FP32/ONNX.

Phase 2 engine build command:

```bash
env LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/local/cuda/targets/aarch64-linux/lib \
  /usr/src/tensorrt/bin/trtexec \
  --onnx=/home/rainbow/models/vision/yolo_nano/yolov8n.onnx \
  --saveEngine=/home/rainbow/models/vision/yolo_nano/yolov8n_fp16.engine \
  --fp16 \
  --memPoolSize=workspace:1024 \
  --skipInference
```

Current Phase 2 result: YOLOv8n TensorRT FP16 runs successfully and lowers average inference latency from `91.70 ms` with YOLO ONNXRuntime CPU reuse to `14.54 ms`.

## TensorRT INT8 Plan

Steps:

1. Collect a small non-sensitive calibration set from camera frames and public sample images.
2. Build INT8 engine with calibration cache.
3. Verify label consistency against FP16 on the same images.
4. Benchmark latency, FPS, memory, and power.

Expected benefit: lower memory bandwidth and potentially better FPS. Risk is larger accuracy drift, especially on small or low-light objects. Phase 3 should use YOLOv8n as the INT8 target, because the FP16 engine is already working.

## Risks

- Jetson TensorRT and CUDA versions may constrain ONNX opset support.
- Older detector graphs such as SSD-MobileNetV1 can hit TensorRT graph compatibility blockers even after the toolchain is installed.
- INT8 calibration needs representative, non-sensitive images.
- Camera exposure and low-light scenes can dominate detection quality, hiding runtime differences.
- TensorRT build time and engine portability can slow iteration.

## Boundary

Project 3 should start only after v0.5a is committed. It should not add VLM, multi-camera input, streaming, or QAT. The first milestone is a clean baseline table, not a large optimization stack.
