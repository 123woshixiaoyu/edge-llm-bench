# YOLO TensorRT C++ Worker

This directory contains the Stage 5 C++ TensorRT local CV worker.

The project intentionally keeps the FastAPI/router layer in Python. Only the hot local CV inference path is moved into C++:

- TensorRT engine load
- execution context creation
- CUDA buffers
- preprocess / inference / postprocess
- latency measurement

`yolo_trt_worker` is a persistent stdin/stdout worker. The Python wrapper starts it once, sends one image path per line, and receives one JSON result per line. This avoids a `pybind11` dependency on Jetson while still keeping the engine/context/buffers resident across requests.

## Build on Jetson

```bash
cd /home/rainbow/edge-llm-bench/cpp/yolo_trt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

The compiled binary lives under `cpp/yolo_trt/build/` and is not committed to git.

## Run Manually

```bash
env LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/local/cuda/targets/aarch64-linux/lib \
  ./build/yolo_trt_worker /home/rainbow/models/vision/yolo_nano/yolov8n_fp16.engine
```

Then type an image path and press Enter. Type `QUIT` to exit.
