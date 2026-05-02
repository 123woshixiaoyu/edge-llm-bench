# Quantization Reproduction Pipeline

Date: 2026-05-02

This document records the project-side PTQ reproduction path from original Hugging Face weights to GGUF conversion, llama.cpp post-training quantization, and RTX 5090 benchmark comparison. Large model artifacts are stored only under `/mnt/d/AI/Models`; this repository keeps commands, CSV results, and documentation only.

## Environment

- Host: Windows + WSL2 Ubuntu 22.04
- GPU: NVIDIA GeForce RTX 5090 Laptop GPU, 24 GB VRAM
- Project path: `/mnt/d/AI/edge-llm-bench`
- Model root: `/mnt/d/AI/Models`
- llama.cpp path: `/mnt/d/AI/edge-llm-bench/llama.cpp`
- llama.cpp commit: `b97ebdc`
- llama.cpp build: CUDA enabled with `GGML_CUDA=ON`
- Python environment: project venv at `/mnt/d/AI/edge-llm-bench/.venv`

## Source Model Confirmation

The original source repos were not present in the earlier project report. I therefore checked local records first, then inspected metadata embedded in the already downloaded GGUF files, and finally cross-checked the repos on Hugging Face without using or saving any token.

| Local prequantized GGUF | Embedded base repo | Status |
|---|---:|---|
| `/mnt/d/AI/Models/qwen3.5/Qwen3.5-0.8B-Q4_K_M.gguf` | `https://huggingface.co/Qwen/Qwen3.5-0.8B` | Confirmed |
| `/mnt/d/AI/Models/qwen3.5/Qwen3.5-4B-Q4_K_M.gguf` | `https://huggingface.co/Qwen/Qwen3.5-4B` | Confirmed |
| `/mnt/d/AI/Models/qwen3.5/Qwen3.5-4B-Q8_0.gguf` | `https://huggingface.co/Qwen/Qwen3.5-4B` | Confirmed |

The existing prequantized GGUF files also identify their publisher as `https://huggingface.co/unsloth`, so the PTQ artifacts below are independent reproductions from the official Qwen source repos rather than a re-labeling of the existing GGUF files.

## Qwen3.5 0.8B PTQ

Source repo: `Qwen/Qwen3.5-0.8B`

Local source path:

```bash
/mnt/d/AI/Models/source/qwen35_08b/
```

Converted and quantized outputs:

| Artifact | Path | Bytes | Approx size |
|---|---|---:|---:|
| F16 GGUF | `/mnt/d/AI/Models/quantized/qwen35_08b/qwen35_08b-f16.gguf` | 1,516,744,544 | 1.5G |
| Q8_0 GGUF | `/mnt/d/AI/Models/quantized/qwen35_08b/qwen35_08b-Q8_0.gguf` | 811,843,424 | 775M |
| Q4_K_M GGUF | `/mnt/d/AI/Models/quantized/qwen35_08b/qwen35_08b-Q4_K_M.gguf` | 529,297,248 | 505M |

Conversion command:

```bash
cd /mnt/d/AI/edge-llm-bench/llama.cpp
/mnt/d/AI/edge-llm-bench/.venv/bin/python convert_hf_to_gguf.py \
  /mnt/d/AI/Models/source/qwen35_08b \
  --outfile /mnt/d/AI/Models/quantized/qwen35_08b/qwen35_08b-f16.gguf \
  --outtype f16
```

Quantization commands:

```bash
cd /mnt/d/AI/edge-llm-bench/llama.cpp
./build/bin/llama-quantize \
  /mnt/d/AI/Models/quantized/qwen35_08b/qwen35_08b-f16.gguf \
  /mnt/d/AI/Models/quantized/qwen35_08b/qwen35_08b-Q8_0.gguf \
  Q8_0

./build/bin/llama-quantize \
  /mnt/d/AI/Models/quantized/qwen35_08b/qwen35_08b-f16.gguf \
  /mnt/d/AI/Models/quantized/qwen35_08b/qwen35_08b-Q4_K_M.gguf \
  Q4_K_M
```

Benchmark command:

```bash
cd /mnt/d/AI/edge-llm-bench
python3 scripts/run_benchmark.py \
  --models qwen35_08b_ptq_q8_0,qwen35_08b_ptq_q4_k_m \
  --out results/raw/qwen35_08b_ptq_5090.csv \
  --log-dir results/raw/logs/qwen35_08b_ptq_5090 \
  --timeout-s 240

python3 scripts/summarize_results.py \
  results/raw/qwen35_08b_ptq_5090.csv \
  --out results/raw/qwen35_08b_ptq_5090_summary_by_model.csv
```

RTX 5090 summary:

| Model | Quant | Cases | Failures | Model size GiB | Avg prompt tok/s | Avg decode tok/s | Peak GPU MB | Avg load ms | GPU layers |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5 0.8B PTQ | Q8_0 | 10 | 0 | 0.756 | 608.33 | 297.40 | 4592 | 92.71 | 25/25 |
| Qwen3.5 0.8B PTQ | Q4_K_M | 10 | 0 | 0.493 | 563.49 | 350.34 | 4253 | 101.97 | 25/25 |

Comparison with existing prequantized GGUF:

| Model | Quant | Size GiB | Avg prompt tok/s | Avg decode tok/s | Peak GPU MB |
|---|---|---:|---:|---:|---:|
| Existing prequantized Qwen3.5 0.8B | Q4_K_M | 0.496 | 450.08 | 389.38 | 3865 |
| Reproduced PTQ Qwen3.5 0.8B | Q4_K_M | 0.493 | 563.49 | 350.34 | 4253 |

The reproduced Q4_K_M file size is very close to the existing prequantized GGUF. Runtime is in the same broad range, but not identical: the reproduced PTQ has higher prompt throughput and lower decode throughput in this run. A likely reason is that the existing file is an Unsloth prequantized artifact and may have used a different quantization recipe, calibration/imatrix setting, or metadata packing path. The project result is still a valid PTQ reproduction because it starts from the official HF weights and uses llama.cpp `llama-quantize`.

## Qwen3.5 4B PTQ

Source repo: `Qwen/Qwen3.5-4B`

Local source path:

```bash
/mnt/d/AI/Models/source/qwen35_4b/
```

Converted and quantized outputs:

| Artifact | Path | Bytes | Approx size |
|---|---|---:|---:|
| F16 GGUF | `/mnt/d/AI/Models/quantized/qwen35_4b/qwen35_4b-f16.gguf` | 8,424,393,472 | 7.9G |
| Q8_0 GGUF | `/mnt/d/AI/Models/quantized/qwen35_4b/qwen35_4b-Q8_0.gguf` | 4,482,403,072 | 4.2G |
| Q4_K_M GGUF | `/mnt/d/AI/Models/quantized/qwen35_4b/qwen35_4b-Q4_K_M.gguf` | 2,708,804,352 | 2.6G |

Conversion command:

```bash
cd /mnt/d/AI/edge-llm-bench/llama.cpp
/mnt/d/AI/edge-llm-bench/.venv/bin/python convert_hf_to_gguf.py \
  /mnt/d/AI/Models/source/qwen35_4b \
  --outfile /mnt/d/AI/Models/quantized/qwen35_4b/qwen35_4b-f16.gguf \
  --outtype f16
```

Quantization commands:

```bash
cd /mnt/d/AI/edge-llm-bench/llama.cpp
./build/bin/llama-quantize \
  /mnt/d/AI/Models/quantized/qwen35_4b/qwen35_4b-f16.gguf \
  /mnt/d/AI/Models/quantized/qwen35_4b/qwen35_4b-Q8_0.gguf \
  Q8_0

./build/bin/llama-quantize \
  /mnt/d/AI/Models/quantized/qwen35_4b/qwen35_4b-f16.gguf \
  /mnt/d/AI/Models/quantized/qwen35_4b/qwen35_4b-Q4_K_M.gguf \
  Q4_K_M
```

Benchmark command:

```bash
cd /mnt/d/AI/edge-llm-bench
python3 scripts/run_benchmark.py \
  --models qwen35_4b_ptq_q8_0,qwen35_4b_ptq_q4_k_m \
  --out results/raw/qwen35_4b_ptq_5090.csv \
  --log-dir results/raw/logs/qwen35_4b_ptq_5090 \
  --timeout-s 300

python3 scripts/summarize_results.py \
  results/raw/qwen35_4b_ptq_5090.csv \
  --out results/raw/qwen35_4b_ptq_5090_summary_by_model.csv
```

RTX 5090 summary:

| Model | Quant | Cases | Failures | Model size GiB | Avg prompt tok/s | Avg decode tok/s | Peak GPU MB | Avg load ms | GPU layers |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5 4B PTQ | Q8_0 | 10 | 0 | 4.175 | 617.06 | 87.90 | 7897 | 87.51 | 33/33 |
| Qwen3.5 4B PTQ | Q4_K_M | 10 | 0 | 2.523 | 556.64 | 116.70 | 6203 | 100.37 | 33/33 |

Comparison with existing prequantized GGUF:

| Model | Quant | Size GiB | Avg prompt tok/s | Avg decode tok/s | Peak GPU MB |
|---|---|---:|---:|---:|---:|
| Existing prequantized Qwen3.5 4B | Q8_0 | 4.175 | 649.71 | 97.82 | 7825 |
| Reproduced PTQ Qwen3.5 4B | Q8_0 | 4.175 | 617.06 | 87.90 | 7897 |
| Existing prequantized Qwen3.5 4B | Q4_K_M | 2.553 | 406.04 | 123.50 | 6222 |
| Reproduced PTQ Qwen3.5 4B | Q4_K_M | 2.523 | 556.64 | 116.70 | 6203 |

The 4B reproduced Q8_0 file is byte-level almost identical in size to the existing prequantized Q8_0 file. The Q4_K_M reproduced file is also very close in size to the existing prequantized Q4_K_M. Performance is close enough for the benchmark comparison purpose, while still showing run-to-run and quantization-recipe differences. As with 0.8B, the reproduced PTQ path is the official-source HF model converted to F16 GGUF, then quantized locally with llama.cpp.

## Q8 vs Q4 Observations

- For both Qwen3.5 0.8B and 4B, Q4_K_M is much smaller than Q8_0 and uses less peak GPU memory.
- On these short prompt tests, Q8_0 has better prompt evaluation throughput, while Q4_K_M has better decode throughput.
- Q4_K_M is the more edge-oriented artifact for deployment size and memory pressure; Q8_0 is useful as a higher-precision reference point.

## Gemma 4 Strategy

Current Gemma 4 files:

| File | Role | Approx size |
|---|---|---:|
| `/mnt/d/AI/Models/gemma4/E2B-it/gemma-4-E2B-it-Q4_K_M.gguf` | text model GGUF | 2.9G |
| `/mnt/d/AI/Models/gemma4/E2B-it/gemma-4-E2B-it-Q8_0.gguf` | text model GGUF | 4.8G |
| `/mnt/d/AI/Models/gemma4/E2B-it/mmproj-F16.gguf` | multimodal projector GGUF | 940M |
| `/mnt/d/AI/Models/gemma4/E4B-it/gemma-4-E4B-it-Q4_K_M.gguf` | text model GGUF | 4.7G |

The Gemma 4 E2B metadata points to `https://huggingface.co/google/gemma-4-E2B-it` as the base model and `https://huggingface.co/unsloth` as the prequantized GGUF publisher. The local E2B layout includes a separate `mmproj-F16.gguf`, and the projector metadata contains vision-projector fields such as `clip.has_vision_encoder`, `clip.vision.projector_type`, and `gemma4v`. The current llama.cpp build includes multimodal-facing binaries such as `llama-mtmd-cli` and `llama-server`.

Project decision:

- Qwen3.5 is the main PTQ reproduction line for this project because it now has a complete source HF -> F16 GGUF -> Q8_0/Q4_K_M -> RTX 5090 benchmark loop.
- Gemma 4 remains a deployment benchmark line using prequantized GGUF files.
- Full Gemma 4 multimodal quantization should be treated as future work because it involves validating both text GGUF and `mmproj` GGUF, plus image or multimodal benchmark prompts. Running only text-side quantization would not prove the full multimodal deployment path.
- A text-side Gemma conversion may be feasible if the exact HF structure is accepted by the current `convert_hf_to_gguf.py`, but it should be handled as an optional extension after the Qwen PTQ deliverables, not as the main reproduction result.

## Issues And Blockers

- The earlier project docs did not record the source repo IDs for Qwen3.5. This was resolved by inspecting embedded GGUF metadata and checking the corresponding official Hugging Face repos.
- No Hugging Face token was used or saved. Downloads were unauthenticated, so future reruns may hit public rate limits.
- The Qwen3.5 HF repos are tagged as image-text-to-text, but the current llama.cpp converter supports the Qwen3.5 architecture used here and produced valid text GGUF files.
- WSL showed a harmless systemd user-session warning during commands. It did not affect conversion, quantization, or benchmark completion.
- Existing prequantized files are from an external GGUF publisher, so exact throughput should not be expected to match local plain llama.cpp PTQ bit-for-bit.

## Output Files

- `results/raw/qwen35_08b_ptq_5090.csv`
- `results/raw/qwen35_08b_ptq_5090_summary_by_model.csv`
- `results/raw/qwen35_4b_ptq_5090.csv`
- `results/raw/qwen35_4b_ptq_5090_summary_by_model.csv`
