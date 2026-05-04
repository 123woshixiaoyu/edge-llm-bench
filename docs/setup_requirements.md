# Setup / Requirements

This repo intentionally does not include model weights, GGUF files, ONNX files, TensorRT engines, calibration caches, or large runtime logs. Those assets live under local runtime directories such as:

```text
/mnt/d/AI/Models
/home/rainbow/models
```

## Python Router / Remote VLM Server

Install the lightweight serving dependencies:

```bash
python3 -m pip install -r serving/requirements.txt
```

This covers:

- FastAPI / Uvicorn for gateway and remote VLM HTTP wrappers;
- Pydantic / PyYAML for schemas and configs;
- HTTPX for backend calls;
- Pillow for remote VLM image resize profiling.

## Demo Dashboard

The reviewer dashboard is optional and sample-mode first:

```bash
python3 -m pip install -r demo/requirements.txt
streamlit run demo/app.py
```

Sample mode reads committed CSV/image evidence and does not require Jetson, RTX, model files, or TensorRT engines.

## Jetson Local CV Runtime

The Jetson CV path depends on JetPack-provided CUDA/TensorRT libraries plus Python packages already used in the benchmark environment:

- OpenCV / `cv2`
- NumPy
- ONNXRuntime CPU for ONNX feasibility runs
- TensorRT Python bindings from JetPack / NVIDIA apt packages

TensorRT Python import may require:

```bash
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/local/cuda/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}
```

The C++ worker build additionally needs:

- `cmake`
- `g++`
- OpenCV C++ headers/libraries
- TensorRT and CUDA headers/libraries from JetPack

Build example:

```bash
cd cpp/yolo_trt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

The resulting binary under `cpp/yolo_trt/build/` is a runtime artifact and should not be committed.

## Remote VLM Runtime

The remote VLM path expects an existing llama.cpp multimodal CLI build and local Gemma VLM assets:

```text
llama.cpp/build/bin/llama-mtmd-cli
gemma-4-E2B-it-Q4_K_M.gguf
mmproj-F16.gguf
```

These are not included in git. The v0.8 profiling remains in `subprocess_cli_mode`; a persistent VLM server is future work.
