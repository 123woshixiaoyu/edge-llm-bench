# Project 3: Local CV ONNX / TensorRT Optimization

## Engineering Goal

Project 3 starts from the v0.5 camera-aware router's local CV path. The goal is to compare Jetson-friendly detector runtimes on the same positive camera sample:

```text
results/figures/camera_v05_positive_detection.jpg
```

The comparison now covers:

- MobileNet-SSD Caffe through OpenCV DNN, the v0.5 router baseline;
- SSD-MobileNetV1 ONNX through ONNXRuntime, including session reuse;
- YOLOv8n ONNX through ONNXRuntime session reuse;
- YOLOv8n TensorRT FP16;
- YOLOv8n TensorRT INT8 with image calibration.

## Model Decision

The v0.5 router baseline remains:

```text
MobileNet-SSD Caffe, OpenCV DNN, PASCAL VOC labels
```

This stays in place because it is already integrated into the camera-aware router and has positive real-camera evidence:

```text
chair, confidence 0.9734
```

Phase 1 used SSD-MobileNetV1 ONNX as the closest ONNX-family comparison target:

```text
/home/rainbow/models/vision/ssd_mobilenet_onnx/ssd_mobilenet_v1_10.onnx
```

In Phase 2, the TensorRT optimization object switches to YOLO-nano because TensorRT 10.3 cannot build an engine from the SSD-MobileNetV1 ONNX graph under the tested settings. The router baseline is not changed; only the optimization benchmark target changes.

YOLO model record:

- model: `YOLOv8n ONNX`
- source: <https://huggingface.co/webml/yolov8n>
- downloaded file: `/home/rainbow/models/vision/yolo_nano/yolov8n.onnx`
- TensorRT engine: `/home/rainbow/models/vision/yolo_nano/yolov8n_fp16.engine`
- TensorRT INT8 engine: `/home/rainbow/models/vision/yolo_nano/yolov8n_int8.engine`
- input: `1x3x640x640`, float32 NCHW
- output: `1x84x8400`
- labels: COCO 80 classes
- ONNX file size: about `12.2 MiB`
- FP16 engine size: about `8.73 MiB`
- INT8 engine size: about `4.89 MiB`

The ONNX and engine files are runtime assets under `/home/rainbow/models/vision/...` and are not committed to git.

## Jetson Environment

| Item | Value |
|---|---|
| Device | Jetson Orin Nano 8GB |
| L4T / JetPack line | `R36.4.3` |
| Python | `3.10.12` |
| CUDA | `12.6`, nvcc `V12.6.68` |
| OpenCV package | `opencv-python-headless 4.13.0.92` |
| ONNXRuntime | `1.23.2`, CPUExecutionProvider |
| TensorRT | `10.3.0` |
| trtexec | `/usr/src/tensorrt/bin/trtexec` |

TensorRT Python import and `trtexec` require this runtime library path on the Jetson:

```bash
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/local/cuda/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}
```

The extra NVIDIA DLA compiler package was needed because TensorRT Python depends on:

```text
/usr/lib/aarch64-linux-gnu/nvidia/libnvdla_compiler.so
```

## SSD-MobileNet TensorRT Blocker

After TensorRT was installed, SSD-MobileNetV1 ONNX was tested with:

- FP16 build;
- FP32 build;
- `--builderOptimizationLevel=0`;
- explicit shape: `image_tensor*:1x300x300x3`;
- explicit input format: `uint8:hwc`.

All attempts failed with the same TensorRT internal Myelin error:

```text
myelinBuilderUtils.cpp::getMyelinSupportType
Device to shape host node should not be folded into myelin.
Engine could not be created from network
```

Engineering judgment: this is a TensorRT 10.3 compatibility blocker for this SSD-MobileNet ONNX graph, not a missing-toolchain problem anymore. Continuing to force this graph would spend time on model-graph surgery rather than the Phase 2 objective. Project 3 therefore keeps MobileNet-SSD as the v0.5 router baseline and uses YOLOv8n as the TensorRT optimization object.

## YOLO TensorRT Engine Build

Build command used on Jetson:

```bash
env LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/local/cuda/targets/aarch64-linux/lib \
  /usr/src/tensorrt/bin/trtexec \
  --onnx=/home/rainbow/models/vision/yolo_nano/yolov8n.onnx \
  --saveEngine=/home/rainbow/models/vision/yolo_nano/yolov8n_fp16.engine \
  --fp16 \
  --memPoolSize=workspace:1024 \
  --skipInference
```

Build summary:

- ONNX IR version: `0.0.8`
- opset: `17`
- producer: `pytorch 2.0.1`
- network tensors: `1` input, `3` output network tensors during build; runtime ONNX output is `output0 [1,84,8400]`
- engine generation time: about `452.45 s`
- engine size: about `8.73 MiB`
- deserialize time reported by `trtexec`: about `0.033 s`
- build result: `PASSED`

## YOLO TensorRT INT8 Calibration

Phase 3 extends the FP16 path with a TensorRT INT8 calibration benchmark. The calibration images are runtime-only assets under:

```text
/home/rainbow/models/vision/yolo_nano/calibration_images/
```

They are not committed to git. The committed manifest is:

```text
serving/results/raw/yolo_int8_calibration_manifest.csv
```

Calibration set summary:

- image count: `24`
- source: deterministic augmentations derived from existing non-sensitive project sample frames
- coverage note: limited diversity; enough for an engineering INT8 pipeline smoke/benchmark, not an accuracy-grade calibration set

The INT8 engine was built on Jetson with the project Python calibrator:

```bash
env LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/local/cuda/targets/aarch64-linux/lib \
  python3 serving/scripts/build_yolo_trt_int8_engine.py \
  --onnx /home/rainbow/models/vision/yolo_nano/yolov8n.onnx \
  --engine /home/rainbow/models/vision/yolo_nano/yolov8n_int8.engine \
  --calib-dir /home/rainbow/models/vision/yolo_nano/calibration_images \
  --calib-cache /home/rainbow/models/vision/yolo_nano/yolov8n_int8_calibration.cache \
  --fp16-fallback
```

Build summary:

- TensorRT: `10.3.0`
- calibration batches: `24`
- calibration time: about `52.7 s`
- engine generation time after calibration: about `685 s`
- INT8 engine size: `5,128,396 bytes` / about `4.89 MiB`
- FP16 engine size: `9,152,164 bytes` / about `8.73 MiB`
- build result: `PASSED`
- note: TensorRT reports entropy calibrator deprecation in TensorRT 10.x in favor of explicit quantization; this is acceptable for the Phase 3 PTQ benchmark but should be revisited for production.

## Results

CSV outputs:

- `serving/results/raw/local_cv_runtime_baseline.csv`
- `serving/results/raw/local_cv_onnx_baseline.csv`
- `serving/results/raw/local_cv_onnx_reuse_baseline.csv`
- `serving/results/raw/local_cv_yolo_onnx_reuse.csv`
- `serving/results/raw/local_cv_yolo_tensorrt_fp16.csv`
- `serving/results/raw/local_cv_yolo_tensorrt_int8.csv`
- `serving/results/raw/local_cv_runtime_summary.csv`

Summary:

| Runtime | Model | Runs | Success | Detection consistency | Mode labels | Avg inference ms | P50 ms | P95 ms | P99 ms | Avg total ms |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| OpenCV DNN | `mobilenet_ssd_voc_opencv_dnn` | 30 | 30 | 1.0 | `["chair"]` | 90.71 | 91.79 | 95.87 | 97.12 | 127.83 |
| ONNXRuntime CPU | `ssd_mobilenet_v1_onnxruntime_cpu` | 30 | 30 | 1.0 | `["bed", "chair"]` | 53.23 | 52.12 | 60.92 | 71.91 | 4929.98 |
| ONNXRuntime CPU reuse | `ssd_mobilenet_v1_onnxruntime_cpu_reuse` | 30 | 30 | 1.0 | `["bed", "chair"]` | 42.02 | 40.96 | 41.67 | 61.82 | 49.04 |
| YOLO ONNXRuntime CPU reuse | `yolov8n_onnxruntime_cpu_reuse` | 30 | 30 | 1.0 | `["bed"]` | 91.70 | 88.55 | 107.10 | 109.08 | 108.64 |
| YOLO TensorRT FP16 | `yolov8n_tensorrt_fp16` | 30 | 30 | 1.0 | `["bed"]` | 14.54 | 14.54 | 14.71 | 14.72 | 28.98 |
| YOLO TensorRT INT8 | `yolov8n_tensorrt_int8` | 30 | 30 | 1.0 | `["no_detection"]` | 11.60 | 11.57 | 11.93 | 11.98 | 27.64 |

YOLO TensorRT FP16 reduces YOLO inference latency from `91.70 ms` to `14.54 ms`, about a `6.3x` improvement. Total latency drops from `108.64 ms` to `28.98 ms`, about a `3.7x` improvement. Detection consistency remains `1.0` for both YOLO runtimes.

YOLO TensorRT INT8 further reduces inference latency from `14.54 ms` to `11.60 ms`, about a `1.25x` FP16-to-INT8 speedup, and reduces engine size from about `8.73 MiB` to about `4.89 MiB`. However, the current INT8 calibration produces a clear detection drift on the positive sample: FP16 consistently detects `["bed"]` with top confidence `0.7388`, while INT8 consistently returns `["no_detection"]` at the same confidence threshold. This means INT8 is technically functional but is not promoted to the router default under the current calibration set.

## ONNXRuntime Session Reuse

Phase 1.5 showed that ONNXRuntime total latency can be dominated by serving adapter lifecycle rather than inference:

| Metric | Old ONNX per-request session | Reused ONNX session |
|---|---:|---:|
| Session init latency | included in every request | `4907.15 ms` one-time startup |
| Avg inference latency | `53.23 ms` | `42.02 ms` |
| P95 inference latency | `60.92 ms` | `41.67 ms` |
| Avg total latency | `4929.98 ms` | `49.04 ms` |
| Detection consistency | `1.0` | `1.0` |

This is a system optimization: model weights and graph are unchanged, but the runtime object is kept resident just like a production TensorRT engine should be.

## Detection Consistency

The detector families are not compared as an accuracy benchmark because their weights and label sets differ:

- MobileNet-SSD Caffe / VOC: stable `["chair"]`;
- SSD-MobileNetV1 ONNX / COCO: stable `["bed", "chair"]`;
- YOLOv8n ONNX and YOLOv8n TensorRT / COCO: stable `["bed"]`.
- YOLOv8n TensorRT INT8 / COCO: stable `["no_detection"]` under the current calibration set and threshold.

The important Phase 2 result is within-model consistency: YOLO ONNXRuntime and YOLO TensorRT FP16 produce the same mode label on every run, with very close confidence and bounding box values in sanity checks. Phase 3 shows that INT8 latency and engine size improve, but calibration quality matters: a fast INT8 engine is not automatically deployable if detection behavior drifts.

## v0.6 Router Integration

The TensorRT result is now connected back to Project 2 under an explicit vision router configuration. The default vision router keeps MobileNet-SSD/OpenCV DNN as the baseline path, while `serving/scripts/smoke_vision_router_yolo_trt.py` initializes the router with:

```text
local_cv_backend = yolo_tensorrt_fp16
engine_path = /home/rainbow/models/vision/yolo_nano/yolov8n_fp16.engine
```

The v0.6 Jetson smoke uses a real CSI frame and the real YOLOv8n TensorRT FP16 engine. It produced `10/10` expected route matches with local `4`, remote `4`, reject `2`; all local rows used `yolo_tensorrt_fp16`, `fallback_used=false`, and local CV inference latency was about `14.4 ms`. Remote rows were intentionally mock in this smoke because v0.6 validates the local fast path; the real RTX VLM route was validated in v0.5b.

## Engineering Conclusion

Phase 2 successfully enabled TensorRT and produced a working FP16 engine benchmark on Jetson:

- MobileNet-SSD remains the v0.5 router baseline because it is already deployed and validated.
- SSD-MobileNetV1 ONNX is a TensorRT 10.3 compatibility blocker under the tested build settings.
- YOLOv8n is a better TensorRT benchmark target because its ONNX graph follows a mature deployment path.
- TensorRT FP16 brings a clear latency win for YOLOv8n on Jetson: roughly `6.3x` lower model inference latency and `3.7x` lower total latency than YOLO ONNXRuntime CPU reuse.
- v0.6 proves that the optimized YOLO TensorRT backend is not just a benchmark result; it can serve the vision router's local detect/classify path under an explicit optimized configuration.

Phase 3 successfully builds and benchmarks an INT8 engine, but it does not replace FP16 as the recommended router backend:

- INT8 inference is faster: `11.60 ms` vs FP16 `14.54 ms`.
- INT8 total latency is only slightly lower: `27.64 ms` vs FP16 `28.98 ms`, because preprocessing/postprocessing overhead becomes a larger share.
- INT8 engine size is smaller: about `4.89 MiB` vs FP16 about `8.73 MiB`.
- INT8 detection drifts to `["no_detection"]` on the positive sample, while FP16 stays stable at `["bed"]`.

Engineering judgment: keep YOLO TensorRT FP16 as the v0.6 local CV router default. Treat INT8 as a promising but not-yet-deployable optimization until calibration coverage and threshold tuning are improved.
