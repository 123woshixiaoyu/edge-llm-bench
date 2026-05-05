# Fast VLM Semantic Verifier Benchmark

## Goal

The current Monitoring Workbench uses Gemma 4 E2B-it Q4 + mmproj as the real remote VLM path. It works, but the subprocess CLI route is slow: earlier v0.8 profiling measured roughly `19 s` baseline latency and about `16.7-17.2 s` after resize/prompt tuning.

This benchmark asks whether compact VLMs such as SmolVLM2 or FastVLM could become a faster event-level semantic verifier. It does not replace YOLO, does not change routing policy, and is not connected to the UI.

## Candidates

| Priority | Model | Runtime expectation | Notes |
| --- | --- | --- | --- |
| P0 | [HuggingFaceTB/SmolVLM2-500M-Video-Instruct](https://hf.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) | Transformers `AutoModelForImageTextToText` | Primary fast verifier candidate. |
| P1 | [HuggingFaceTB/SmolVLM2-256M-Video-Instruct](https://hf.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct) | Transformers `AutoModelForImageTextToText` | Smallest Jetson smoke candidate if RTX path succeeds. |
| P2 | [HuggingFaceTB/SmolVLM2-2.2B-Instruct](https://hf.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) | Transformers `AutoModelForImageTextToText` | More quality-oriented; not a Jetson-first candidate by default. |
| P3 | [apple/FastVLM-0.5B-fp16](https://hf.co/apple/FastVLM-0.5B-fp16) | `ml-fastvlm` / Apple-oriented runtime | Not wired into this WSL/RTX Transformers benchmark. |

Hugging Face model metadata reports the SmolVLM2 candidates as `image-text-to-text` models using the `transformers` library and `AutoModelForImageTextToText`. FastVLM is tagged with the `ml-fastvlm` library and `coreml`, so it is treated as a runtime blocker unless a stable WSL/RTX path is added separately.

## Task Set

Each model is evaluated with monitoring-oriented JSON-only prompts:

1. lipstick/object presence
2. person falling or lying down
3. abnormal or attention-needed scene
4. one-sentence scene description

Default full run:

```bash
python3 serving/scripts/benchmark_fast_vlm_verifiers.py \
  --models smolvlm2_500m,smolvlm2_256m,smolvlm2_2b,fastvlm_05b \
  --platform-label rtx_5090_wsl \
  --out serving/results/raw/fast_vlm_verifier_benchmark.csv \
  --summary-out serving/results/raw/fast_vlm_verifier_summary.csv
```

The script supports `--local-files-only` for dependency/cache smoke checks that do not download model weights.

## Environment

The benchmark was rerun in an isolated evaluation environment so the main project environment stays untouched:

```bash
python3 -m venv .venv-vlm
source .venv-vlm/bin/activate
python -m pip install -U pip wheel setuptools
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install "transformers>=4.57.0" accelerate pillow safetensors sentencepiece protobuf num2words
```

CUDA verification:

```text
torch 2.10.0+cu130
cuda True
device NVIDIA GeForce RTX 5090 Laptop GPU
```

The `.venv-vlm/` directory is ignored by git. Model weights were downloaded into the normal Hugging Face cache outside the repository and are not committed.

## Current Run

The official RTX/WSL run used three committed public sample images and four monitoring prompts per SmolVLM2 model. FastVLM was not run because its published runtime path is `ml-fastvlm`/CoreML-oriented rather than a stable WSL/RTX Transformers path.

```bash
source .venv-vlm/bin/activate
python serving/scripts/benchmark_fast_vlm_verifiers.py \
  --models smolvlm2_500m,smolvlm2_256m,smolvlm2_2b,fastvlm_05b \
  --platform-label rtx_5090_wsl \
  --max-new-tokens 96 \
  --timeout-s 240 \
  --out serving/results/raw/fast_vlm_verifier_benchmark.csv \
  --summary-out serving/results/raw/fast_vlm_verifier_summary.csv
```

Results:

| Model | Platform | Requests | Avg latency | P50 | P95 | Peak GPU memory | JSON valid rate | Recommendation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SmolVLM2-500M-Video-Instruct | RTX 5090 / WSL | 12 | 398.94 ms | 265.80 ms | 517.55 ms | 1347.35 MB | 0.0 | reject as structured verifier |
| SmolVLM2-256M-Video-Instruct | RTX 5090 / WSL | 12 | 353.20 ms | 382.42 ms | 423.36 ms | 855.57 MB | 0.0 | reject as structured verifier |
| SmolVLM2-2.2B-Instruct | RTX 5090 / WSL | 12 | 424.87 ms | 385.40 ms | 551.89 ms | 4635.18 MB | 0.0 | reject as structured verifier |
| FastVLM-0.5B-fp16 | RTX 5090 / WSL | 12 | n/a | n/a | n/a | n/a | 0.0 | blocker |

Artifacts:

- `serving/results/raw/fast_vlm_verifier_benchmark.csv`
- `serving/results/raw/fast_vlm_verifier_summary.csv`

No model weights, Hugging Face cache, or private images were added to git.

## Interpretation

SmolVLM2 is dramatically faster than the current Gemma remote VLM path. The current Gemma path is roughly `16.7-20 s` per event-level review, while the SmolVLM2 candidates complete most requests in well under one second after the model is loaded.

However, the structured output contract failed in this prompt set. All SmolVLM2 outputs were non-JSON despite JSON-only prompts. Typical outputs were terse answers such as `yes`, `No`, or one natural-language sentence. That makes the models interesting as fast semantic hints, but not drop-in replacements for the current remote VLM verifier without a stricter decoding/output wrapper, retry strategy, or a downstream parser.

Answers to the evaluation questions:

1. **Can SmolVLM2-500M beat Gemma VLM latency?** Yes. It is roughly sub-second on RTX/WSL after load, versus Gemma's tens-of-seconds subprocess path.
2. **Is SmolVLM2-256M a Jetson candidate?** Maybe for a future smoke because its peak RTX memory was under 1 GB, but the JSON contract failure means it is not worth perturbing the Jetson demo stack yet.
3. **Is JSON/final-answer output stable?** No. JSON valid rate was 0/36 across the SmolVLM2 runs.
4. **Can it serve as an event-level semantic verifier?** Not as-is. It may be useful behind an output parser or one-sentence verifier mode, but not as a structured policy decision source today.
5. **Can it replace YOLO?** No. YOLO TensorRT remains the local monitoring fast path. SmolVLM2 is still an event-level semantic model, not a 10+ FPS detector.
6. **Should we keep exploring?** Yes, but the next useful experiments are output-control work, YOLO-pose/open-vocabulary detection for specific event types, and possibly a persistent compact-VLM service rather than direct integration now.

Jetson smoke was intentionally skipped in this pass. The RTX run proved the compact models are fast but not structured-output reliable; installing a new VLM stack on Jetson is not justified until the verifier contract is made reliable off-device.

## Real-Time Thresholds

Use these thresholds when rerunning with dependencies:

| Latency tier | Meaning |
| --- | --- |
| `<1 s` | usable fast event verifier |
| `1-3 s` | acceptable semantic review |
| `>5 s` | slow fallback only |
| `10+ FPS` | can begin to challenge YOLO fast path |

Even if a compact VLM is fast, YOLO should remain the local monitoring fast path. VLM is for event-level semantic confirmation, open-vocabulary checks, and review questions, not every-frame monitoring.

## Next Steps

1. Create a separate local environment with `torch`, `transformers`, `accelerate`, and `pillow`.
2. Run P0 SmolVLM2-500M on RTX with the three committed sample images.
3. If P0 succeeds, run P1 SmolVLM2-256M and compare JSON validity/latency.
4. Only if P0/P1 are promising, run Jetson smoke with 1 image x 2 prompts.
5. Keep Gemma as quality fallback until a compact VLM demonstrates lower latency and acceptable JSON/final-answer stability.
