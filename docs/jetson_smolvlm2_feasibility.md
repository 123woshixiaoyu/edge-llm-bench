# Jetson SmolVLM2 Feasibility

## Purpose

This benchmark asks whether EdgeLog can run `HuggingFaceTB/SmolVLM2-256M-Video-Instruct` directly on the Jetson Orin Nano 8GB as a local semantic sentinel.

The question is architectural: if a compact VLM can run locally with stable latency and parseable short decisions, EdgeLog can add a privacy-preserving Jetson-side semantic check before offloading to RTX. If it cannot, the measured blocker justifies the current heterogeneous design:

```text
Jetson cheap trigger / storage / privacy gate -> RTX SmolVLM2 fast verifier -> RTX Gemma async describer
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
| Memory at test time | 7.4 GiB total, about 2.6 GiB available |
| Disk at test time | 233 GiB total, about 192 GiB available |
| `tegrastats` | available |
| System PyTorch | not installed |
| System Transformers | not installed |
| Existing `.venv-yolo-world` | present, but also lacks `torch`, `transformers`, and `ultralytics` |

Because Jetson PyTorch wheels are CUDA / JetPack-specific and large, this benchmark did not install a new PyTorch stack into the existing runtime. That keeps the current TensorRT / YOLO / gateway environment intact and records the dependency blocker explicitly.

## Results

Artifacts:

- `serving/results/raw/jetson_smolvlm2_feasibility.csv`
- `serving/results/raw/jetson_smolvlm2_feasibility_summary.csv`

Result status:

| Metric | Result |
| --- | --- |
| Jetson run | Executed on `rainbow-desktop` |
| Model load | Not attempted because dependencies are missing |
| Inference latency | Not available |
| P50 / P95 | Not available |
| Peak memory | Not available |
| Parse success | `0` because no inference request completed |
| Status | `dependency_blocked` |
| Recommendation | `keep_rtx_smolvlm2_fallback` |

The generated blocker row records:

```text
missing Python dependencies: torch; transformers
```

This is a Red result for immediate Jetson local semantic sentinel use. It is not a model-quality failure; it is an environment feasibility blocker. Jetson remains the local capture / cheap proposal / storage / privacy device, while RTX SmolVLM2 remains the measured fast semantic verifier.

## Decision Bands

| Band | Criteria | EdgeLog interpretation |
| --- | --- | --- |
| Green | P95 <= 1.5s, parse success >= 0.9, no OOM | `local_semantic_sentinel_candidate` |
| Yellow | P95 1.5s-5s, parse success >= 0.8, no OOM | `local_async_low_frequency_candidate` |
| Red | OOM, unstable dependencies, crashes, P95 > 5s, or weak parsing | `keep_rtx_smolvlm2_fallback` |

Red is still a useful result. It means Jetson remains responsible for local capture, cheap anomaly proposals, storage, and privacy enforcement, while SmolVLM2 remains an RTX fast verifier.

## How To Reproduce

Run from the Jetson repo root after confirming the environment is safe:

```bash
python3 serving/scripts/benchmark_jetson_smolvlm2.py \
  --image-path results/figures/camera_v05_positive_detection.jpg \
  --runs 10 \
  --image-size 384 \
  --max-new-tokens 16 \
  --output-csv serving/results/raw/jetson_smolvlm2_feasibility.csv \
  --summary-csv serving/results/raw/jetson_smolvlm2_feasibility_summary.csv
```

Use `--local-files-only` when validating a pre-warmed Hugging Face cache without allowing downloads.

## Next Optimizations If Feasible

- Jetson-specific PyTorch wheel check.
- Smaller image sizes.
- Lower `max_new_tokens`.
- Persistent sentinel service.
- ONNX / TensorRT export investigation.
- Quantized or smaller VLM candidate search.

These are not P0 until the measured feasibility result is at least Yellow.
