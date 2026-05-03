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

## ONNX Plan

Preferred path:

- use an ONNX equivalent of the v0.5 local detector, or convert the detector through a reproducible script;
- keep input size fixed at `300x300` for the first pass;
- store ONNX model files under `/home/rainbow/models/vision/`, not in git;
- validate that ONNXRuntime produces similar labels on the same camera smoke image and a small fixed calibration image set.

If MobileNet-SSD Caffe-to-ONNX conversion becomes too brittle, choose a newer small detector with first-class ONNX export, such as YOLOv8n or YOLO11n. That model switch should be recorded as a Project 3 decision, not hidden inside v0.5.

## TensorRT FP16 Plan

Steps:

1. Export or obtain ONNX.
2. Build TensorRT FP16 engine on Jetson.
3. Benchmark the same image set used for ONNXRuntime.
4. Compare latency, memory, and power against OpenCV DNN and ONNXRuntime.

Expected benefit: lower inference latency and better throughput on Jetson GPU, with minimal accuracy change relative to FP32/ONNX.

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
