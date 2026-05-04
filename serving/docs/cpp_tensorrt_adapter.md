# C++ TensorRT Local CV Adapter

## Goal

Stage 5 explores a mixed runtime for the local CV hot path:

- Python stays responsible for routing, orchestration, benchmark scripts, and future API integration.
- C++ owns only the YOLO TensorRT detector hot path: engine load, execution context, CUDA buffers, preprocess, inference, postprocess, and latency measurement.

This avoids rewriting the whole gateway in C++ while still showing how the latency-critical edge inference path can move closer to a production Jetson runtime.

## Implementation

The implementation uses a persistent C++ worker rather than `pybind11` because the Jetson environment did not have `pybind11` installed. This is one of the two planned implementation options for Phase 4.

Files:

- `cpp/yolo_trt/CMakeLists.txt`
- `cpp/yolo_trt/yolo_trt_detector.cpp`
- `cpp/yolo_trt/README.md`
- `serving/app/local_cv_yolo_cpp.py`
- `serving/scripts/benchmark_yolo_cpp_adapter.py`

The worker starts once, loads the TensorRT FP16 engine once, creates the execution context once, preallocates CUDA buffers, then reads image paths from stdin and returns one JSON result per line. The Python wrapper keeps that process alive across detections.

Runtime engine:

```text
/home/rainbow/models/vision/yolo_nano/yolov8n_fp16.engine
```

## Build

Build command on Jetson:

```bash
cd /home/rainbow/edge-llm-bench/cpp/yolo_trt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

The compiled worker is:

```text
/home/rainbow/edge-llm-bench/cpp/yolo_trt/build/yolo_trt_worker
```

The build directory and binary are runtime artifacts and are not committed to git.

## Benchmark

Benchmark command:

```bash
env LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/local/cuda/targets/aarch64-linux/lib \
  python3 serving/scripts/benchmark_yolo_cpp_adapter.py \
  --image results/figures/camera_v05_positive_detection.jpg \
  --engine-path /home/rainbow/models/vision/yolo_nano/yolov8n_fp16.engine \
  --worker-path /home/rainbow/edge-llm-bench/cpp/yolo_trt/build/yolo_trt_worker \
  --iterations 30 \
  --warmup 3 \
  --out serving/results/raw/local_cv_yolo_cpp_adapter.csv \
  --summary-out serving/results/raw/local_cv_yolo_cpp_adapter_summary.csv
```

Outputs:

- `serving/results/raw/local_cv_yolo_cpp_adapter.csv`
- `serving/results/raw/local_cv_yolo_cpp_adapter_summary.csv`

## Results

| Runtime | Runs | Success | Consistency | Mode labels | Avg inference ms | P95 inference ms | P99 inference ms | Avg total ms | P95 total ms | P99 total ms |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Python TensorRT FP16 | 30 | 30 | 1.0 | `["bed"]` | 14.62 | 14.75 | 14.79 | 29.00 | 29.35 | 50.23 |
| C++ worker TensorRT FP16 | 30 | 30 | 1.0 | `["bed"]` | 14.07 | 14.18 | 14.20 | 29.29 | 31.38 | 32.21 |

Detection consistency is unchanged. The C++ worker has slightly lower inference latency and much tighter inference tail latency. Total latency is similar because the benchmark still crosses the Python/C++ process boundary and includes image read/preprocess/postprocess work.

## Engineering Interpretation

The C++ worker is not a dramatic average-latency win for YOLOv8n because the FP16 TensorRT Python adapter is already efficient and the model is small. The useful result is narrower tail latency on the inference segment and a cleaner separation of the production hot path:

- Python router: policy, request orchestration, logging, benchmark glue.
- C++ worker: resident TensorRT engine, CUDA buffers, inference and postprocess.

This is closer to an edge deployment architecture than rebuilding the TensorRT engine or context inside Python request code.

## Router Integration Status

The C++ worker is not connected back to the router in this phase. It remains a benchmarked optional backend. The v0.6 router default stays:

```text
yolo_tensorrt_fp16
```

This avoids replacing a working router path with an IPC-based worker before it has been tested under reliability and failure-mode scenarios.

## Limits

- Batch size is fixed at 1.
- Input shape is fixed at `1x3x640x640`.
- No video stream.
- No shared-memory IPC; stdin/stdout JSON is simple but not the lowest-overhead path.
- No production memory pool beyond persistent TensorRT/CUDA buffers.
- Build depends on Jetson TensorRT, CUDA, and OpenCV C++ paths.
