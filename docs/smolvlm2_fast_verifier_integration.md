# SmolVLM2 Fast Verifier Integration

## Purpose

EdgeLog now has a real RTX-side fast verifier service for the second layer of the semantic event pipeline:

```text
cheap trigger proposal -> SmolVLM2 FINAL_ANSWER verifier -> YES / NO / UNKNOWN -> event promotion
```

SmolVLM2 is not the realtime detector, not the free-form describer, and not a strict JSON backend. YOLO / ROI / motion remain cheap proposal triggers. Gemma remains the slower async semantic describer for confirmed or high-value events.

## Runtime

Default verifier:

- Model: `HuggingFaceTB/SmolVLM2-256M-Video-Instruct`
- Backend name: `smolvlm2_256m_final_line`
- Platform: RTX 5090 laptop / WSL
- Port: `127.0.0.1:8092`
- Protocol: `FINAL_ANSWER` final-line, with single-token `YES` / `NO` / `UNKNOWN` accepted as a robust fallback.

Start command:

```bash
.venv-vlm/bin/python serving/scripts/run_fast_vlm_verifier_5090.py --host 127.0.0.1 --port 8092
```

The runner uses local Hugging Face cache by default. Pass `--allow-download` only when intentionally preparing the runtime environment. Model cache, `.venv-vlm`, and downloaded weights are not tracked by git.

## API

Health:

```http
GET /health
```

Verifier:

```http
POST /v1/verify_event
```

Request fields:

- `rule`: natural-language event rule.
- `question`: verifier question.
- `image_base64` or `image_path`: candidate keyframe.
- `roi_name`: optional ROI label.
- `proposal_reason`: cheap trigger evidence.
- `objects`: cheap trigger labels.
- `max_new_tokens`: default 32.

Response fields:

- `backend`
- `final_answer`: `YES`, `NO`, or `UNKNOWN`
- `reason`
- `raw_text`
- `parse_success`
- `latency_ms`
- `error`

## Validation

Artifact:

- `serving/results/raw/smolvlm2_fast_verifier_validation.csv`

Validation covered:

| Case | Status |
| --- | --- |
| health endpoint ready | PASS |
| YES parse and event promotion | PASS |
| NO parse and event rejection | PASS |
| UNKNOWN parse and needs-review event | PASS |
| invalid image graceful error | PASS |
| service unavailable graceful fallback | PASS |
| latency recorded | PASS |

Observed validation latency:

- YES case: about `289.24 ms`.
- NO case: about `249.92 ms`.
- UNKNOWN case: about `314.18 ms`.
- Three-call average: about `284.45 ms`.

These calls align with the earlier sub-second verifier benchmark. Model load is measured separately by `/health` and was about `29538.72 ms` in this validation run.

In the full EdgeLog real-mode validation, the workflow-level SmolVLM2 verifier case passed with `267.15 ms` latency and parse success. That case is recorded in `serving/results/raw/edgelog_real_mode_validation.csv`.

## Jetson Local Feasibility

Artifact:

- `serving/results/raw/jetson_smolvlm2_feasibility_summary.csv`

SmolVLM2-256M has also been tested directly on Jetson Orin Nano 8GB in an isolated environment at `~/venvs/smolvlm2-jetson`.

Result:

| Metric | Value |
| --- | --- |
| Runs | 10 |
| Success | 10/10 |
| Average latency | `946.24 ms` |
| P50 / P95 | `860.65 ms` / `1675.01 ms` |
| Parse success | `1.0` |
| Peak CUDA memory | `855.06 MB` |
| Model load latency | `4817.17 ms` |
| Recommendation | `local_async_low_frequency_candidate` |

Important setup notes:

- The initial Jetson blocker was dependency-related, not a model capability failure.
- `torch 2.11.0` could see CUDA but failed CUBLAS matmul with `CUBLAS_STATUS_ALLOC_FAILED`.
- `torch 2.8.0` with system CUBLAS 12.6 and venv-local cuDNN/cuDSS ran CUDA matmul successfully.
- `torchvision 0.23.0` was required for the SmolVLM image processor.
- The model cache was synced from the WSL / RTX Hugging Face cache because Jetson Hugging Face network / DNS was unstable.
- Short YES / NO / UNKNOWN style prompts worked; longer `FINAL_DECISION` prompts caused format repetition.

Architecture impact: Jetson SmolVLM2 is now a local async / low-frequency semantic sentinel candidate. It should not be described as per-frame real-time VLM. RTX SmolVLM2 remains useful as a faster fallback or higher-throughput verifier, and Gemma remains the slower semantic describer.

## EdgeLog Behavior

When `verifier_backend=smolvlm2_fast`:

- `YES` promotes a proposal to a verified `semantic_event`.
- `NO` marks the proposal `rejected`.
- `UNKNOWN` stores the event as `unknown` / needs review.
- timeout or service unavailable records failure without blocking live monitoring.

When the service is not available, the UI can still use `mock_final_line` for offline workflow validation, but it must not label mock results as real SmolVLM2.

## Current Limits

- RTX/WSL remains the default fast verifier path for sub-second, higher-throughput checks.
- Jetson SmolVLM2 is feasible as a low-frequency async local sentinel, not as per-frame VLM inference.
- It is a short-answer semantic verifier only.
- It does not replace YOLO cheap triggers.
- It does not replace Gemma for longer descriptions.
- The protocol is not strict JSON.
- More diverse real-world event rules should be tested before using it for unattended alerts.
