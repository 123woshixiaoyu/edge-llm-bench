# Jetson SmolVLM2 Feasibility

## Purpose

This benchmark asks whether EdgeLog can run `HuggingFaceTB/SmolVLM2-256M-Video-Instruct` directly on the Jetson Orin Nano 8GB as a local semantic sentinel.

The question is architectural: if a compact VLM can run locally with stable latency and parseable short decisions, EdgeLog can add a privacy-preserving Jetson-side semantic check before offloading to RTX.

The updated result is Yellow: Jetson Orin Nano 8GB can run SmolVLM2-256M as a low-frequency local semantic sentinel, but it is not suitable for per-frame real-time VLM inference.

```text
Jetson cheap trigger / storage / privacy gate
  -> optional Jetson low-frequency SmolVLM2 sentinel
  -> RTX SmolVLM2 fallback / higher-throughput verifier
  -> RTX Gemma async describer
```

## Scope

This is only a feasibility benchmark.

- It does not change the EdgeLog UI.
- It does not replace the RTX SmolVLM2 verifier.
- It does not alter the gateway or routing policy.
- It uses one still keyframe, not video.
- It uses a short final-decision protocol, not free-form VQA.

## Model And Prompt

- Model: `HuggingFaceTB/SmolVLM2-256M-Video-Instruct`
- Input: one public sample keyframe resized to `384` or `448` on the long edge.
- Generation: `max_new_tokens=16` or `32`.
- Output protocol:

```text
FINAL_DECISION: REVIEW, NORMAL, or UNKNOWN
CATEGORY: one short category
SEVERITY: LOW, MEDIUM, or HIGH
REASON: one short reason
```

The parser also accepts the older verifier protocol:

- `FINAL_ANSWER: YES` -> `REVIEW`
- `FINAL_ANSWER: NO` -> `NORMAL`
- `FINAL_ANSWER: UNKNOWN` -> `UNKNOWN`

## Jetson Environment

The benchmark script records environment details in both CSV artifacts, including Python version, L4T line, free disk/memory summary, PyTorch / Transformers versions, CUDA availability, and whether `tegrastats` is available.

No system package changes are required by the script. If PyTorch / Transformers are missing, it records `dependency_blocked` rather than modifying the Jetson runtime environment.

Observed environment:

| Item | Value |
| --- | --- |
| Host | `rainbow-desktop` |
| Platform | Jetson Orin Nano / aarch64 |
| Python | `3.10.12` |
| L4T / JetPack line | `R36.4.3` |
| Kernel | `5.15.148-tegra` |
| CUDA line | `12.6` as reported by NVIDIA tooling |
| Memory at final test time | 7.4 GiB total, about 4.8 GiB available |
| Disk at final test time | 233 GiB total, about 188 GiB available |
| `tegrastats` | available |
| Runtime environment | isolated venv at `~/venvs/smolvlm2-jetson` |
| PyTorch | `2.8.0`, CUDA visible |
| Transformers | installed in the isolated venv |
| TorchVision | `0.23.0` |

The initial Red result was an environment blocker, not a model capability failure. The system Python lacked `torch` and `transformers`, so the first run correctly emitted `dependency_blocked`. The final run used an isolated environment and did not pollute system Python.

Important setup findings:

1. `torch 2.11.0` could see CUDA on Jetson, but CUDA matmul failed with `CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(handle)`.
2. Downgrading to `torch 2.8.0` with the system CUBLAS 12.6 stack and venv-local cuDNN/cuDSS allowed CUDA matmul and SmolVLM2 generation to run.
3. Missing `torchvision` prevented the SmolVLM image processor from loading; installing `torchvision 0.23.0` fixed it.
4. Jetson Hugging Face network / DNS was unstable, so the model cache was synced from the existing WSL / RTX Hugging Face cache instead of relying on Jetson downloads.
5. The longer `FINAL_DECISION` prompt caused format repetition and parse failures.
6. A shorter `YES` / `NO` / `UNKNOWN` style final-line prompt produced 10/10 parse success.

## Results

Artifacts:

- `serving/results/raw/jetson_smolvlm2_feasibility.csv`
- `serving/results/raw/jetson_smolvlm2_feasibility_summary.csv`

Result status:

| Metric | Result |
| --- | --- |
| Jetson run | Executed on `rainbow-desktop` |
| Model load | Success |
| Runs | 10 |
| Success | 10/10 |
| Average latency | `946.24 ms` |
| P50 / P95 | `860.65 ms` / `1675.01 ms` |
| Peak CUDA memory | `855.06 MB` |
| Load latency | `4817.17 ms` |
| Parse success | `1.0` |
| Status | `success` |
| Recommendation | `local_async_low_frequency_candidate` |

This is a Yellow result. Jetson no longer needs to be treated as only a cheap trigger device. It can host a low-frequency local semantic sentinel for privacy-sensitive or offline checks. However, the measured P95 around `1.68s` means it should not be called on every frame. Cheap anomaly triggers still gate VLM calls, and RTX remains useful for higher-throughput verification, Gemma descriptions, and text summaries.

## Decision Bands

| Band | Criteria | EdgeLog interpretation |
| --- | --- | --- |
| Green | P95 <= 1.5s, parse success >= 0.9, no OOM | `local_semantic_sentinel_candidate` |
| Yellow | P95 1.5s-5s, parse success >= 0.8, no OOM | `local_async_low_frequency_candidate` |
| Red | OOM, unstable dependencies, crashes, P95 > 5s, or weak parsing | `keep_rtx_smolvlm2_fallback` |

The current Jetson result lands in Yellow: viable for local async / heartbeat-style semantic checks, not per-frame inference.

## How To Reproduce

Run from the Jetson repo root after confirming the isolated environment is active:

```bash
~/venvs/smolvlm2-jetson/bin/python serving/scripts/benchmark_jetson_smolvlm2.py \
  --image-path results/figures/camera_v05_positive_detection.jpg \
  --runs 10 \
  --image-size 384 \
  --max-new-tokens 16 \
  --output-csv serving/results/raw/jetson_smolvlm2_feasibility.csv \
  --summary-csv serving/results/raw/jetson_smolvlm2_feasibility_summary.csv
```

Use `--local-files-only` when validating a pre-warmed Hugging Face cache without allowing downloads.

## Next Optimizations If Feasible

- Lock the known-good Jetson PyTorch / TorchVision dependency set in a separate setup note.
- Smaller image sizes.
- Shorter final-line prompts and `max_new_tokens`.
- Persistent sentinel service.
- ONNX / TensorRT export investigation.
- Quantized or smaller VLM candidate search.

These are useful next steps only after keeping the sentinel explicitly low-frequency and async.
