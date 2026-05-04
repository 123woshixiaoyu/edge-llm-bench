# Edge LLM Quantized Deployment Benchmark

This project studies quantized GGUF LLM deployment on a laptop GPU and an embedded Jetson device. The goal is to make engineering decisions under edge constraints: model size, quantization format, prefill speed, decode speed, memory use, power efficiency, temperature, and rough output quality.

## Current Scope

- Backends: `llama.cpp` CUDA via `llama-completion`
- Quantization pipeline: Hugging Face source model -> F16 GGUF -> `Q8_0` / `Q4_K_M` via `llama-quantize`
- Models:
  - Qwen3.5 0.8B PTQ: `F16`, `Q8_0`, `Q4_K_M`
  - Qwen3.5 4B PTQ: `Q8_0`, `Q4_K_M`
  - Gemma 4 E2B IT: `Q4_K_M`, `Q8_0`
  - Gemma 4 E4B IT: `Q4_K_M`
  - Qwen3.5 4B: `Q4_K_M`, `Q8_0`
  - Qwen3.5 0.8B: `Q4_K_M`
- Hardware:
  - RTX 5090 Laptop 24GB via WSL2 Ubuntu 22.04
  - Jetson Orin Nano 8GB representative benchmark and quantization decision study
- Out of scope for this phase: fine-tuning, QAT, serving platform, TensorRT, VLM camera demo

## Repository Layout

```text
docs/                 project report and experiment notes
prompts/              benchmark prompt set
results/raw/*.csv     reproducible benchmark result tables
scripts/              benchmark and summary scripts
serving/              Edge LLM Task Router MVP
```

`llama.cpp/`, model weights, and raw per-case logs are intentionally excluded from git.

## Setup

Clone and build `llama.cpp` separately inside the project root:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j"$(nproc)"
```

Put GGUF models under a model root. On the WSL machine used for the RTX 5090 baseline:

```text
/mnt/d/AI/Models/
```

On Jetson, a good default is:

```text
~/models/
```

The benchmark script reads these environment variables:

- `EDGE_LLM_BENCH_ROOT`: project root, defaults to the parent of `scripts/`
- `LLAMA_CPP_DIR`: llama.cpp checkout, defaults to `<project>/llama.cpp`
- `LLAMA_COMPLETION`: explicit `llama-completion` binary path
- `MODEL_ROOT`: model directory, defaults to `/mnt/d/AI/Models`

## RTX 5090 Baseline

```bash
cd /mnt/d/AI/edge-llm-bench
python3 scripts/run_benchmark.py \
  --out results/raw/5090_baseline.csv \
  --log-dir results/raw/logs \
  --hardware "RTX 5090 Laptop 24GB / WSL2 Ubuntu-22.04" \
  --monitor nvidia-smi

python3 scripts/summarize_results.py \
  results/raw/5090_baseline.csv \
  --out results/raw/5090_baseline_summary_by_model.csv
```

## Jetson Smoke Test

Copy only the smallest model first:

```bash
mkdir -p ~/models/qwen3.5
# From WSL on the laptop, replace <jetson> with user@ip:
# scp /mnt/d/AI/Models/qwen3.5/Qwen3.5-0.8B-Q4_K_M.gguf <jetson>:~/models/qwen3.5/
```

Then on Jetson:

```bash
cd ~/edge-llm-bench
MODEL_ROOT=~/models python3 scripts/run_benchmark.py \
  --models qwen35_08b_q4_k_m \
  --prompt-ids en_qa_method \
  --out results/raw/jetson_smoke.csv \
  --log-dir results/raw/jetson_logs \
  --hardware "Jetson Orin Nano 8GB" \
  --monitor tegrastats \
  --timeout-s 300
```

After smoke test passes, run the first real Jetson batch:

```bash
MODEL_ROOT=~/models python3 scripts/run_benchmark.py \
  --models qwen35_08b_q4_k_m,gemma4_e2b_q4_k_m \
  --out results/raw/jetson_representative.csv \
  --log-dir results/raw/jetson_logs \
  --hardware "Jetson Orin Nano 8GB" \
  --monitor tegrastats \
  --timeout-s 600

python3 scripts/summarize_results.py \
  results/raw/jetson_representative.csv \
  --out results/raw/jetson_representative_summary_by_model.csv
```

Compare Jetson results with the RTX 5090 baseline:

```bash
python3 scripts/compare_hardware.py \
  --base results/raw/5090_baseline_summary_by_model.csv \
  --edge results/raw/jetson_representative_summary_by_model.csv \
  --out results/figures/jetson_vs_5090.md \
  --base-name "RTX 5090" \
  --edge-name "Jetson Orin Nano"
```

## Current Results

See [docs/report.md](docs/report.md) for the deployment benchmark report, [docs/quantization_pipeline.md](docs/quantization_pipeline.md) for the PTQ reproduction path, and [docs/quantization_decision_study.md](docs/quantization_decision_study.md) for the Jetson quantization decision study.

Jetson Qwen3.5 0.8B quantization decision:

| Quant | Cases | Failures | Size GiB | Avg decode tok/s | Peak memory MB | Avg max power W | Decode tok/s/W |
|---|---:|---:|---:|---:|---:|---:|---:|
| F16 | 10 | 0 | 1.413 | 24.98 | 3677 | 18.93 | 1.32 |
| Q8_0 | 10 | 0 | 0.756 | 47.10 | 2953 | 19.69 | 2.39 |
| Q4_K_M | 10 | 0 | 0.493 | 56.54 | 2703 | 20.20 | 2.80 |

Default Jetson recommendation: `Qwen3.5 0.8B Q4_K_M`. It is the smallest, fastest, lowest-memory 0.8B option in this study, with the best decode throughput per watt. `Q8_0` remains the conservative fallback when output stability matters more than latency and memory.

Representative Jetson run:

| Model | Quant | Cases | Failures | Avg prompt tok/s | Avg decode tok/s | Peak memory MB | Avg max power W | Max temp C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4 E2B it | Q4_K_M | 10 | 0 | 146.26 | 32.68 | 3664 | 21.90 | 68 |
| Gemma 4 E2B it | Q8_0 | 10 | 0 | 98.62 | 20.21 | 4485 | 20.58 | 67 |
| Qwen3.5 0.8B | Q4_K_M | 10 | 0 | 396.76 | 57.56 | 2616 | 20.10 | 68 |

Gemma 4 E2B Q8_0 is feasible on Jetson, but Q4_K_M is the better Gemma-side deployment candidate: it is smaller, faster, and uses less peak memory while preserving full GPU offload.

Cross-hardware comparison: [results/figures/jetson_vs_5090.md](results/figures/jetson_vs_5090.md).

## Model / Backend Selection Scorecard

The project now includes a constraint-aware scorecard that turns benchmark results into explicit backend choices. It uses hard constraints plus profile-specific weighted scoring across text LLM, local CV, and remote VLM candidates.

Scorecard files:

- [docs/model_selection_scorecard.md](docs/model_selection_scorecard.md)
- [configs/model_selection_profiles.yaml](configs/model_selection_profiles.yaml)
- [results/raw/model_selection_candidates.csv](results/raw/model_selection_candidates.csv)
- [results/raw/model_selection_scores.csv](results/raw/model_selection_scores.csv)

Run:

```bash
python3 scripts/score_model_candidates.py \
  --candidates results/raw/model_selection_candidates.csv \
  --profiles configs/model_selection_profiles.yaml \
  --out results/raw/model_selection_scores.csv
```

Current profile recommendations:

| Profile | Recommended backend |
|---|---|
| `text_local_default` | Qwen3.5 0.8B `Q4_K_M` on Jetson |
| `text_quality_fallback` | Qwen3.5 4B `Q4_K_M` on RTX, with 4B `Q8_0` as the heavier conservative alternative |
| `vision_local_fast_path` | YOLOv8n TensorRT FP16 on Jetson |
| `vision_remote_semantic_backend` | Gemma 4 E2B-it `Q4_K_M` + `mmproj-F16` remote VLM |

## Project 2: Edge LLM Task Router

Project 2 turns the benchmark decisions into a Jetson-first routing service. The gateway analyzes each request and decides whether it should run locally on Jetson, go to a remote RTX backend, or be rejected/degraded. The MVP uses a mock backend by default and includes an optional `llama.cpp` backend adapter.

Key docs:

- [serving/docs/serving_design.md](serving/docs/serving_design.md)
- [serving/docs/routing_policy.md](serving/docs/routing_policy.md)

Run the mock gateway:

```bash
cd /mnt/d/AI/edge-llm-bench
.venv/bin/python -m uvicorn serving.app.main:app --host 127.0.0.1 --port 8000
```

Run policy and chat-path load tests:

```bash
.venv/bin/python serving/scripts/load_test_router.py --url http://127.0.0.1:8000
.venv/bin/python serving/scripts/load_test_router.py --url http://127.0.0.1:8000 --endpoint chat --out serving/results/raw/router_chat_eval.csv
```

Current MVP result: both route-only and mock chat-path load tests match the expected route for 30/30 requests.

v0.2 connects the Jetson local route to a real `llama-server` running Qwen3.5 0.8B Q4_K_M while keeping the remote route mocked as a placeholder. The real-local smoke test writes [serving/results/raw/real_local_backend_smoke.csv](serving/results/raw/real_local_backend_smoke.csv): 9/9 requests matched the expected route, and 4/4 local requests returned non-mock model output with backend latency around 1.3-1.5 seconds.

Run the v0.2 hybrid gateway on Jetson after starting local `llama-server`:

```bash
EDGE_ROUTER_CONFIG_DIR=/home/rainbow/edge-llm-bench/serving/configs_llamacpp_local \
python3 -m uvicorn serving.app.main:app --host 127.0.0.1 --port 8000

python3 serving/scripts/smoke_real_local_backend.py --url http://127.0.0.1:8000
```

v0.3 connects the remote route to a real RTX 5090 `llama-server` running Qwen3.5 4B Q4_K_M. The Jetson gateway now supports the full heterogeneous loop: short/private QA and summary tasks run on Jetson 0.8B Q4, while code/reasoning/high-quality tasks route to RTX 4B Q4. In the current WSL networking setup, Windows exposes the WSL server on `127.0.0.1:8081` but not directly on the laptop LAN IP, so the Jetson smoke run used an SSH reverse tunnel:

```text
Jetson 127.0.0.1:18081 -> RTX/WSL 127.0.0.1:8081
```

Run the v0.3 gateway on Jetson after starting local `llama-server`, remote RTX `llama-server`, and the tunnel:

```bash
EDGE_ROUTER_CONFIG_DIR=/home/rainbow/edge-llm-bench/serving/configs_dual_llamacpp \
python3 -m uvicorn serving.app.main:app --host 127.0.0.1 --port 8000

python3 serving/scripts/smoke_dual_real_backends.py --url http://127.0.0.1:8000
```

The v0.3 smoke test writes [serving/results/raw/dual_real_backend_smoke.csv](serving/results/raw/dual_real_backend_smoke.csv): 11/11 requests matched the expected route, 4/4 local requests returned non-mock Jetson model output, and 6/6 remote requests returned non-mock RTX model output.

v0.4 focuses on serving reliability: real in-process queue depth, load-aware routing, explicit fallback/reject behavior, simulated telemetry hooks, concurrent load testing, and failure-mode smoke tests. It does not change the model lineup or introduce TensorRT/VLM.

Run v0.4 reliability checks after starting the same dual real backends:

```bash
python3 serving/scripts/smoke_dual_real_backends.py \
  --url http://127.0.0.1:8000

python3 serving/scripts/load_test_dual_real_backends.py \
  --url http://127.0.0.1:8000 \
  --concurrency 4 \
  --requests 40 \
  --out serving/results/raw/dual_real_backend_load_test.csv

python3 serving/scripts/summarize_serving_load_test.py \
  serving/results/raw/dual_real_backend_load_test.csv \
  --out serving/results/raw/dual_real_backend_load_test_summary.csv

python3 serving/scripts/smoke_failure_modes.py \
  --url http://127.0.0.1:8000
```

v0.5a adds camera-aware edge routing while keeping the text router intact. Camera capture is real, local CV is real, and remote VLM remains a marked placeholder for v0.5b.

```bash
python3 scripts/camera_smoke.py
python3 serving/scripts/smoke_vision_router.py
```

Current v0.4 reliability result: `40` concurrent-load requests at concurrency `4`, `40/40` expected routes matched, `0` backend errors, `0` timeouts.

Current v0.5a camera/CV result:

- camera: IMX219 on CAM1, configured as `Camera IMX219-C` on CSI Header 2
- capture backend: `GStreamer Argus`
- sample: [results/figures/camera_v05_sample.jpg](results/figures/camera_v05_sample.jpg)
- local CV: MobileNet-SSD through OpenCV DNN, model files under `/home/rainbow/models/vision/mobilenet_ssd/`
- local CV inference latency: about `76.6 ms`
- vision smoke: [serving/results/raw/vision_router_smoke.csv](serving/results/raw/vision_router_smoke.csv)
- route distribution: local `4`, remote `4`, reject `2`, expected routes `10/10`
- remote VLM: mock placeholder, explicitly marked with `remote_is_mock=true`

v0.5a+ positive detection evidence:

- positive sample: [results/figures/camera_v05_positive_detection.jpg](results/figures/camera_v05_positive_detection.jpg)
- local CV CSV: [serving/results/raw/local_cv_positive_detection.csv](serving/results/raw/local_cv_positive_detection.csv)
- vision router CSV: [serving/results/raw/vision_router_positive_detection.csv](serving/results/raw/vision_router_positive_detection.csv)
- detected label: `chair`, confidence `0.9734`, box `[65, 48, 1043, 706]`
- capture latency: about `1198.55 ms`
- local CV inference latency: about `77.96 ms`
- route distribution remains local `4`, remote `4`, reject `2`, expected routes `10/10`

The earlier `no_detection` result came from the content of the first sample frame, not a failed camera or local CV pipeline. The positive detection run uses a real CSI camera frame and confirms the real camera + real local CV detection path.

Vision design docs:

- [serving/docs/vision_routing_design.md](serving/docs/vision_routing_design.md)
- [serving/docs/real_vlm_backend.md](serving/docs/real_vlm_backend.md)
- [serving/docs/project3_tensorrt_plan.md](serving/docs/project3_tensorrt_plan.md)
- [serving/docs/project3_tensorrt_report.md](serving/docs/project3_tensorrt_report.md)

v0.5b replaces the remote VLM placeholder with a real RTX/WSL VLM backend while keeping the same camera-aware policy. Local detect/classify tasks still run on Jetson MobileNet-SSD; scene description, VQA, and high-quality visual tasks route to the RTX VLM when privacy allows; `privacy=local_only` semantic vision tasks are rejected instead of sending images off-device.

The v0.5b model choice is the existing local Gemma 4 E2B-it multimodal GGUF pair:

- text/model GGUF: `/mnt/d/AI/Models/gemma4/E2B-it/gemma-4-E2B-it-Q4_K_M.gguf`
- projector GGUF: `/mnt/d/AI/Models/gemma4/E2B-it/mmproj-F16.gguf`
- runtime: `llama.cpp/build/bin/llama-mtmd-cli`, wrapped by a small FastAPI server
- no new VLM weights were downloaded

Run the remote VLM server on the RTX/WSL side:

```bash
cd /mnt/d/AI/edge-llm-bench
PORT=8091 .venv/bin/python serving/scripts/run_remote_vlm_server_5090.py
```

In the current WSL networking setup, the Jetson accesses that server through an SSH reverse tunnel:

```text
Jetson 127.0.0.1:18091 -> RTX/WSL 127.0.0.1:8091
```

Then run the real VLM smoke test on Jetson:

```bash
python3 serving/scripts/smoke_vision_router_real_vlm.py \
  --remote-url http://127.0.0.1:18091 \
  --out serving/results/raw/vision_router_real_vlm_smoke.csv \
  --baseline-out serving/results/raw/local_cv_real_vlm_baseline.csv \
  --sample-image results/figures/camera_v05_real_vlm_sample.jpg
```

Current v0.5b result:

- smoke CSV: [serving/results/raw/vision_router_real_vlm_smoke.csv](serving/results/raw/vision_router_real_vlm_smoke.csv)
- local CV baseline: [serving/results/raw/local_cv_real_vlm_baseline.csv](serving/results/raw/local_cv_real_vlm_baseline.csv)
- sample frame: [results/figures/camera_v05_real_vlm_sample.jpg](results/figures/camera_v05_real_vlm_sample.jpg)
- route distribution: local `4`, remote `4`, reject `2`
- expected routes: `10/10`
- all remote rows have `remote_is_mock=false`
- remote model: `gemma4_e2b_it_q4_mmproj`
- remote VLM latency range: about `19.2-20.0 s` per remote call

Images are sent to the RTX backend as base64 in the HTTP request, not as local file paths, because Jetson paths are not readable from WSL.

v0.6 integrates the scorecard-selected local CV fast path back into the vision router. The default vision router still keeps MobileNet-SSD/OpenCV DNN as the baseline/fallback path, while the explicit YOLO TensorRT configuration makes local `detect` / `classify` requests use the Jetson TensorRT FP16 engine:

```bash
env LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/local/cuda/targets/aarch64-linux/lib \
  python3 serving/scripts/smoke_vision_router_yolo_trt.py
```

Current v0.6 result:

- smoke CSV: [serving/results/raw/vision_router_yolo_trt_smoke.csv](serving/results/raw/vision_router_yolo_trt_smoke.csv)
- local CV baseline CSV: [serving/results/raw/local_cv_yolo_trt_router_baseline.csv](serving/results/raw/local_cv_yolo_trt_router_baseline.csv)
- local CV backend: `yolo_tensorrt_fp16`, model `yolov8n_tensorrt_fp16`
- TensorRT engine: `/home/rainbow/models/vision/yolo_nano/yolov8n_fp16.engine`
- route distribution: local `4`, remote `4`, reject `2`
- expected routes: `10/10`
- local CV inference latency: about `14.4 ms`
- capture latency: about `1298.76 ms`
- `fallback_used=false` for all rows
- remote rows are intentionally mock in this v0.6 smoke; the real remote VLM path was already validated in v0.5b

## Project 3: Local CV ONNX / TensorRT Optimization

Project 3 starts from the v0.5 local CV path and measures runtime choices for Jetson deployment. MobileNet-SSD Caffe through OpenCV DNN remains the v0.5 router baseline. Phase 1 used SSD-MobileNetV1 ONNX for ONNXRuntime feasibility, Phase 1.5 fixed ONNXRuntime session lifecycle overhead, and Phase 2 switches the TensorRT optimization object to YOLOv8n because SSD-MobileNetV1 ONNX hits a TensorRT 10.3 graph compatibility blocker.

Run the fixed-image runtime benchmarks on Jetson:

```bash
python3 serving/scripts/benchmark_local_cv_runtimes.py \
  --runtime opencv_dnn \
  --image results/figures/camera_v05_positive_detection.jpg \
  --iterations 30 \
  --out serving/results/raw/local_cv_runtime_baseline.csv

python3 serving/scripts/benchmark_local_cv_runtimes.py \
  --runtime onnxruntime \
  --image results/figures/camera_v05_positive_detection.jpg \
  --iterations 30 \
  --out serving/results/raw/local_cv_onnx_baseline.csv

python3 serving/scripts/benchmark_local_cv_runtimes.py \
  --runtime onnxruntime_cpu_reuse \
  --image results/figures/camera_v05_positive_detection.jpg \
  --iterations 30 \
  --warmup 3 \
  --out serving/results/raw/local_cv_onnx_reuse_baseline.csv

python3 serving/scripts/benchmark_local_cv_runtimes.py \
  --runtime yolo_onnxruntime_cpu_reuse \
  --image results/figures/camera_v05_positive_detection.jpg \
  --iterations 30 \
  --warmup 3 \
  --out serving/results/raw/local_cv_yolo_onnx_reuse.csv

env LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/local/cuda/targets/aarch64-linux/lib \
  python3 serving/scripts/benchmark_local_cv_runtimes.py \
  --runtime yolo_tensorrt_fp16 \
  --image results/figures/camera_v05_positive_detection.jpg \
  --iterations 30 \
  --warmup 3 \
  --out serving/results/raw/local_cv_yolo_tensorrt_fp16.csv

python3 serving/scripts/summarize_local_cv_runtimes.py \
  serving/results/raw/local_cv_runtime_baseline.csv \
  serving/results/raw/local_cv_onnx_baseline.csv \
  serving/results/raw/local_cv_onnx_reuse_baseline.csv \
  serving/results/raw/local_cv_yolo_onnx_reuse.csv \
  serving/results/raw/local_cv_yolo_tensorrt_fp16.csv \
  --out serving/results/raw/local_cv_runtime_summary.csv
```

Current Project 3 summary:

| Runtime | Runs | Success | Detection consistency | Mode labels | Avg inference ms | P95 ms | Avg total ms |
|---|---:|---:|---:|---|---:|---:|---:|
| OpenCV DNN | 30 | 30 | 1.0 | `["chair"]` | 90.71 | 95.87 | 127.83 |
| ONNXRuntime CPU | 30 | 30 | 1.0 | `["bed", "chair"]` | 53.23 | 60.92 | 4929.98 |
| ONNXRuntime CPU reuse | 30 | 30 | 1.0 | `["bed", "chair"]` | 42.02 | 41.67 | 49.04 |
| YOLO ONNXRuntime CPU reuse | 30 | 30 | 1.0 | `["bed"]` | 91.70 | 107.10 | 108.64 |
| YOLO TensorRT FP16 | 30 | 30 | 1.0 | `["bed"]` | 14.54 | 14.71 | 28.98 |

Phase 1.5 shows that the earlier ONNXRuntime total latency problem was session lifecycle overhead: reusable session initialization costs about `4907.15 ms` once, then per-request total latency drops from `4929.98 ms` to `49.04 ms`.

TensorRT is now enabled on Jetson with `trtexec` at `/usr/src/tensorrt/bin/trtexec` and TensorRT `10.3.0`. Runtime commands need:

```bash
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/local/cuda/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}
```

SSD-MobileNetV1 ONNX was tested with FP16, FP32, low optimization level, explicit shape, and explicit `uint8:hwc` input format, but all builds failed with `Device to shape host node should not be folded into myelin`. The engineering decision is to keep MobileNet-SSD as the router baseline and use YOLOv8n for the TensorRT benchmark. INT8 calibration is deferred to Project 3 Phase 3.
