# Edge LLM Quantized Deployment Benchmark Report

Date: 2026-05-02

Stage: RTX 5090 Laptop baseline and Jetson Orin Nano representative benchmark completed.

## 1. Goal

This project studies how quantized open LLMs behave when deployed on an edge-class device. The main question is not only "can the model run?", but also what latency, memory, power, and temperature trade-offs appear when the same GGUF model is moved from a high-end laptop GPU to a Jetson Orin Nano.

The project is intentionally scoped as an inference and deployment benchmark. It does not include fine-tuning, QAT, a serving layer, or new model training in this phase.

## 2. Experimental Scope

Backend:

- `llama.cpp` with CUDA backend
- Runtime binary: `llama-completion`
- Benchmark mode: completion mode, deterministic decoding, no prompt display, no warmup

Prompt set:

- 10 prompts covering Chinese QA, English QA, code generation, debugging, summarization, long-context mixed text, and operational reasoning.

Models on the RTX 5090 baseline:

- Gemma 4 E2B IT `Q4_K_M`
- Gemma 4 E2B IT `Q8_0`
- Gemma 4 E4B IT `Q4_K_M`
- Qwen3.5 4B `Q4_K_M`
- Qwen3.5 4B `Q8_0`
- Qwen3.5 0.8B `Q4_K_M`

Models on the Jetson representative run:

- Gemma 4 E2B IT `Q4_K_M`
- Qwen3.5 0.8B `Q4_K_M`

## 3. Environments

| Item | RTX 5090 Laptop baseline | Jetson Orin Nano run |
|---|---|---|
| OS | Windows + WSL2 Ubuntu 22.04 | Jetson Linux R36.4.3 |
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU | NVIDIA Jetson Orin Nano integrated GPU |
| Memory | 24 GB VRAM | 8 GB shared memory |
| CUDA | CUDA 13.2 toolkit in WSL | CUDA 12.6 toolkit |
| llama.cpp commit | `b97ebdc` | `0754b7b6fe6109909786bdaa763111167b7410c8` |
| Build | `-DGGML_CUDA=ON` | `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87` |
| Monitor | `nvidia-smi` | `tegrastats` |

Note: the two devices used different `llama.cpp` commits because the Jetson build was created later. The comparison is still useful for a deployment project, but a stricter research-style benchmark should pin the same commit on both devices.

## 4. RTX 5090 Baseline Results

| Model | Quant | Cases | Failures | Size GiB | Avg prompt tok/s | Avg decode tok/s | Peak GPU MB | Avg max power W | Max temp C | GPU layers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4 E2B it | Q4_K_M | 10 | 0 | 2.893 | 155.45 | 169.32 | 5192 | 94.93 | 56 | 36/36 |
| Gemma 4 E2B it | Q8_0 | 10 | 0 | 4.702 | 165.46 | 128.22 | 6189 | 107.43 | 56 | 36/36 |
| Gemma 4 E4B it | Q4_K_M | 10 | 0 | 4.635 | 152.09 | 99.62 | 6581 | 100.22 | 55 | 43/43 |
| Qwen3.5 4B | Q4_K_M | 10 | 0 | 2.553 | 406.04 | 123.50 | 6222 | 108.91 | 56 | 33/33 |
| Qwen3.5 4B | Q8_0 | 10 | 0 | 4.175 | 649.71 | 97.82 | 7825 | 109.91 | 60 | 33/33 |
| Qwen3.5 0.8B | Q4_K_M | 10 | 0 | 0.496 | 450.08 | 389.38 | 3865 | 103.62 | 59 | 25/25 |

## 5. Jetson Orin Nano Results

| Model | Quant | Cases | Failures | Size GiB | Avg prompt tok/s | Avg decode tok/s | Peak memory MB | Avg max power W | Max temp C | GPU layers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4 E2B it | Q4_K_M | 10 | 0 | 2.893 | 146.26 | 32.68 | 3664 | 21.90 | 68 | 36/36 |
| Qwen3.5 0.8B | Q4_K_M | 10 | 0 | 0.496 | 396.76 | 57.56 | 2616 | 20.10 | 68 | 25/25 |

Both Jetson models completed all 10 prompts with zero failures and full GPU layer offload. The larger Gemma 4 E2B Q4 model stayed within the 8 GB Jetson memory budget, peaking at about 3.6 GB during this benchmark.

## 6. Cross-Hardware Comparison

| Model | Quant | RTX 5090 decode tok/s | Jetson decode tok/s | Jetson/RTX decode | RTX 5090 prefill tok/s | Jetson prefill tok/s | Jetson/RTX prefill |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4 E2B it | Q4_K_M | 169.32 | 32.68 | 0.193 | 155.45 | 146.26 | 0.941 |
| Qwen3.5 0.8B | Q4_K_M | 389.38 | 57.56 | 0.148 | 450.08 | 396.76 | 0.882 |

Key observations:

- Jetson decode throughput is much lower than the RTX 5090 baseline, around 19.3% for Gemma 4 E2B Q4 and 14.8% for Qwen3.5 0.8B Q4.
- Jetson prefill throughput is relatively strong in this setup, reaching 94.1% of the RTX baseline for Gemma 4 E2B Q4 and 88.2% for Qwen3.5 0.8B Q4.
- The Jetson run is power efficient for edge deployment: average observed peak power was about 20-22 W, compared with roughly 95-104 W on the laptop GPU for the overlapping models.
- Qwen3.5 0.8B Q4 is the better smoke-test and fast-response model. Gemma 4 E2B Q4 is the better representative "real edge LLM" target because it is larger, still deployable, and exposes clearer memory and latency constraints.

## 7. Project Value

This project demonstrates a complete edge AI deployment workflow:

- model selection under device constraints
- GGUF quantized model deployment
- CUDA-enabled `llama.cpp` builds on x86 WSL and ARM Jetson
- reproducible prompt-based benchmarking
- runtime metric collection for throughput, memory, power, and temperature
- cross-device comparison and deployment trade-off analysis

For internship applications, the strongest story is:

> Built a reproducible edge LLM benchmark for quantized GGUF models across RTX 5090 and Jetson Orin Nano, measuring decode/prefill throughput, memory, power, and thermal behavior with `llama.cpp`, CUDA, `nvidia-smi`, and `tegrastats`.

## 8. Limitations and Next Steps

Limitations:

- The RTX 5090 and Jetson runs used different `llama.cpp` commits.
- Load time is affected by mmap and OS file cache, so it should not be treated as strict cold-start latency.
- The current Jetson representative run covers two deployable models, not the full six-model RTX sweep.
- The benchmark measures raw local inference, not an HTTP/gRPC serving stack.

Recommended next steps:

1. Pin the same `llama.cpp` commit on both devices and rerun the two overlapping models.
2. Add one Jetson stress run for Gemma 4 E4B Q4 if memory and time allow.
3. Add simple charts for decode throughput, memory, and power.
4. Create a short project README section with setup commands and one-line reproduction commands.
5. Optionally build a small local serving demo after the benchmark story is stable.
