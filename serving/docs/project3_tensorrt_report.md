# Project 3 Phase 1: Local CV ONNX / TensorRT Optimization

## Engineering Goal

Project 3 starts from the v0.5 camera-aware router's local CV path. The question is not just whether a detector can run on Jetson, but which runtime path is suitable for a deployable local vision backend:

- OpenCV DNN baseline;
- ONNXRuntime;
- TensorRT FP16;
- later, TensorRT INT8 in Phase 2.

Phase 1 uses the same positive camera sample from v0.5a+:

```text
results/figures/camera_v05_positive_detection.jpg
```

## Model Decision

The v0.5 local CV baseline remains:

```text
MobileNet-SSD Caffe, OpenCV DNN, PASCAL VOC labels
```

For ONNXRuntime and the TensorRT build path, Phase 1 uses an ONNX Model Zoo SSD-MobileNetV1 model:

```text
/home/rainbow/models/vision/ssd_mobilenet_onnx/ssd_mobilenet_v1_10.onnx
```

This is not the exact same Caffe weight file used by v0.5, so detection labels are compared for consistency rather than treated as an accuracy benchmark. It is still the closest low-risk optimization target because it keeps the same detector family. A switch to YOLO-nano was not needed in Phase 1.

Source model record:

- ONNX Model Zoo SSD-MobileNetV1: `onnxmodelzoo/ssd_mobilenet_v1_10`
- source URL: <https://huggingface.co/onnxmodelzoo/ssd_mobilenet_v1_10>
- file size: about `28-29 MB`
- model file is stored under `/home/rainbow/models/vision/` and is not committed to git

## Jetson Environment

| Item | Value |
|---|---|
| Device | Jetson Orin Nano 8GB |
| L4T / JetPack line | `R36.4.3` |
| Python | `3.10.12` |
| CUDA | `12.6`, nvcc `V12.6.68` |
| OpenCV package | `opencv-python-headless 4.13.0.92` |
| ONNXRuntime | `1.23.2`, CPUExecutionProvider |
| TensorRT | not installed |
| TensorRT apt candidate | `10.3.0.30-1+cuda12.5` |

TensorRT was not available in the current Jetson environment: `trtexec` was missing, `python3 -m pip show tensorrt` was empty, and `dpkg-query` showed no installed `libnvinfer` / `tensorrt` package. The Phase 1 scripts therefore record a TensorRT blocker instead of fabricating FP16 numbers.

## Runtime Difference

OpenCV DNN is the current baseline. It reads the existing Caffe `deploy.prototxt` and `.caffemodel`, preprocesses to `300x300`, and runs through OpenCV's DNN backend.

ONNXRuntime validates the model-exchange path. It runs the SSD-MobileNet ONNX graph with `CPUExecutionProvider`. This proves the project can load an ONNX detector and produce structured detections on the same Jetson sample, but it is not yet GPU-accelerated.

TensorRT FP16 should be the Jetson deployment optimization path: build an FP16 engine from ONNX on-device, then run the engine for lower latency and better GPU utilization. In this Phase 1 run, TensorRT is blocked by missing packages, so the next step is environment installation and engine build, not model debugging.

## Results

CSV outputs:

- `serving/results/raw/local_cv_runtime_baseline.csv`
- `serving/results/raw/local_cv_onnx_baseline.csv`
- `serving/results/raw/local_cv_onnx_reuse_baseline.csv`
- `serving/results/raw/local_cv_tensorrt_fp16.csv`
- `serving/results/raw/local_cv_runtime_summary.csv`

Summary:

| Runtime | Model | Runs | Success | Detection consistency | Mode labels | Avg inference ms | P50 ms | P95 ms | P99 ms | Avg total ms |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| OpenCV DNN | `mobilenet_ssd_voc_opencv_dnn` | 30 | 30 | 1.0 | `["chair"]` | 90.71 | 91.79 | 95.87 | 97.12 | 127.83 |
| ONNXRuntime CPU | `ssd_mobilenet_v1_onnxruntime_cpu` | 30 | 30 | 1.0 | `["bed", "chair"]` | 53.23 | 52.12 | 60.92 | 71.91 | 4929.98 |
| ONNXRuntime CPU reuse | `ssd_mobilenet_v1_onnxruntime_cpu_reuse` | 30 | 30 | 1.0 | `["bed", "chair"]` | 42.02 | 40.96 | 41.67 | 61.82 | 49.04 |
| TensorRT FP16 | `ssd_mobilenet_v1_tensorrt_fp16` | 1 | 0 | 0.0 | n/a | n/a | n/a | n/a | n/a | n/a |

The ONNXRuntime inference step is faster than the OpenCV DNN forward pass on this image, but its total time is much larger because the current adapter creates a new ONNXRuntime session on each benchmark iteration. That is a useful engineering signal: production ONNX inference should keep the session resident, just like a production TensorRT path should keep the engine loaded.

## Phase 1.5: ONNXRuntime Session Reuse

Phase 1.5 isolates ONNXRuntime session creation from per-request inference. The code now provides `OnnxLocalCVDetector`, which creates `onnxruntime.InferenceSession` once in `__init__`, optionally runs warmup, and reuses the session in `detect(image_path)`.

This is a serving adapter optimization, not a model architecture optimization. The model file and detection logic are unchanged.

Phase 1.5 result:

| Metric | Old ONNX per-request session | Reused ONNX session |
|---|---:|---:|
| Session init latency | included in every request | `4907.15 ms` one-time startup |
| Avg inference latency | `53.23 ms` | `42.02 ms` |
| P95 inference latency | `60.92 ms` | `41.67 ms` |
| Avg total latency | `4929.98 ms` | `49.04 ms` |
| Detection consistency | `1.0` | `1.0` |

The old result made ONNXRuntime look unusable for serving because each request paid a roughly five-second session creation cost. With session reuse, total latency drops close to the actual inference latency. This demonstrates that the bottleneck was the adapter lifecycle, not the ONNX graph itself.

## Detection Consistency

OpenCV DNN reproduced the v0.5a+ positive result across all 30 runs:

```text
chair, confidence 0.9734
```

ONNXRuntime produced stable detections across all 30 runs:

```text
bed, confidence 0.8927
chair, confidence 0.3669
```

The label set is not identical because the ONNX model is an SSD-MobileNetV1 COCO model while the v0.5 baseline is a MobileNet-SSD VOC Caffe model. The important Phase 1 result is that both paths consistently detect a chair-like object on the same frame, and the ONNX path produces deterministic structured output.

ONNXRuntime session reuse kept the same stable label set:

```text
bed, confidence 0.8927
chair, confidence 0.3669
```

## TensorRT FP16 Blocker

The TensorRT build script is:

```bash
serving/scripts/build_local_cv_trt_engine.sh \
  /home/rainbow/models/vision/ssd_mobilenet_onnx/ssd_mobilenet_v1_10.onnx \
  /home/rainbow/models/vision/ssd_mobilenet_onnx/ssd_mobilenet_v1_10_fp16.engine
```

Current result:

```text
trtexec not found. Install TensorRT packages on Jetson before building the FP16 engine.
Candidate checked during Phase 1: tensorrt 10.3.0.30-1+cuda12.5 from the Jetson r36.4 apt repo.
```

Engine files must stay under `/home/rainbow/models/vision/...` and must not be committed.

## Engineering Conclusion

Phase 1 completed the baseline and ONNX feasibility parts:

- OpenCV DNN remains the stable v0.5 local CV baseline.
- ONNXRuntime CPU runs successfully and produces deterministic detections on the same camera sample.
- TensorRT FP16 is not yet blocked by model conversion; it is blocked earlier by missing TensorRT runtime/build tools on Jetson.

The next Project 3 action is to install Jetson TensorRT packages, build the FP16 engine from the ONNX model, and rerun `local_cv_tensorrt_fp16.csv`. INT8 calibration is intentionally deferred to Phase 2.
