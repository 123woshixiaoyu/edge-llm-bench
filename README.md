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

v0.5 prep is limited to camera smoke. It only checks whether Jetson can open a camera and capture one non-sensitive frame:

```bash
python3 scripts/camera_smoke.py
```

Current v0.4 reliability result: `40` concurrent-load requests at concurrency `4`, `40/40` expected routes matched, `0` backend errors, `0` timeouts. Current v0.5 prep camera smoke did not capture a frame yet: Jetson exposes `/dev/media0`, but no `/dev/video*` device and no `nvarguscamerasrc` GStreamer element were available.
