# Edge LLM Quantized Deployment Benchmark

This project benchmarks quantized GGUF LLMs on a laptop GPU and an embedded Jetson device. The goal is to compare deployment trade-offs under edge constraints: model size, quantization format, prefill speed, decode speed, memory use, power, and temperature.

## Current Scope

- Backends: `llama.cpp` CUDA via `llama-completion`
- Models:
  - Gemma 4 E2B IT: `Q4_K_M`, `Q8_0`
  - Gemma 4 E4B IT: `Q4_K_M`
  - Qwen3.5 4B: `Q4_K_M`, `Q8_0`
  - Qwen3.5 0.8B: `Q4_K_M`
- Hardware:
  - RTX 5090 Laptop 24GB via WSL2 Ubuntu 22.04
  - Jetson Orin Nano 8GB, in progress
- Out of scope for this phase: fine-tuning, QAT, serving platform, new model downloads

## Repository Layout

```text
docs/                 project report and experiment notes
prompts/              benchmark prompt set
results/raw/*.csv     reproducible benchmark result tables
scripts/              benchmark and summary scripts
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
  --models qwen35_08b_q4_k_m,gemma4_e2b_q4_k_m,qwen35_4b_q4_k_m \
  --out results/raw/jetson_representative.csv \
  --log-dir results/raw/jetson_logs \
  --hardware "Jetson Orin Nano 8GB" \
  --monitor tegrastats \
  --timeout-s 600

python3 scripts/summarize_results.py \
  results/raw/jetson_representative.csv \
  --out results/raw/jetson_representative_summary_by_model.csv
```

## Current Results

See [docs/report.md](docs/report.md) for the latest RTX 5090 baseline and Jetson plan.
