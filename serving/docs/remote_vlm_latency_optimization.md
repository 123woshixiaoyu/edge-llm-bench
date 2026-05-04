# Remote VLM Latency Profiling

## Goal

Project 2 v0.5b connected a real remote VLM backend:

```text
Gemma 4 E2B-it Q4_K_M + mmproj-F16
```

The route is real, not mock, but the original smoke latency was about `19-20 s`. Stage 6 profiles that latency and tests low-risk changes without changing the model or serving architecture.

## Setup

Runtime path:

```text
Jetson benchmark client -> Jetson 127.0.0.1:18091 -> SSH reverse tunnel -> WSL/RTX 127.0.0.1:8091 -> FastAPI VLM server -> llama-mtmd-cli subprocess
```

The server still runs in:

```text
subprocess_cli_mode
```

This means every request launches `llama-mtmd-cli`. The model is not resident across requests.

CSV outputs:

- `serving/results/raw/remote_vlm_latency_breakdown.csv`
- `serving/results/raw/remote_vlm_resize_ablation.csv`
- `serving/results/raw/remote_vlm_prompt_ablation.csv`

## Latency Breakdown

Breakdown run:

| Metric | Value |
|---|---:|
| Image size | `1280x720` |
| Image bytes | `63,972` |
| Request payload bytes | `85,547` |
| Client base64 encode | `0.21 ms` |
| Server base64 decode | `0.28 ms` |
| Server image write | `0.31 ms` |
| Server subprocess | `18,775.38 ms` |
| Server endpoint total | `18,776.08 ms` |
| Network roundtrip | `19,000.58 ms` |
| Client total | `19,017.62 ms` |

Interpretation: transport overhead is tiny relative to `llama-mtmd-cli` subprocess execution. The main bottleneck is not base64 or HTTP; it is model/CLI startup and generation inside the subprocess.

## Resize Ablation

Prompt:

```text
Describe the image in one concise sentence. Do not explain your reasoning.
```

`max_tokens=64`.

| Resize policy | Payload bytes | Latency ms | Object mentioned | Heuristic quality | Notes |
|---|---:|---:|---|---:|---|
| original | 85,508 | 18,227.81 | false | 2 | Long reasoning-style output |
| 672 wide | 38,804 | 18,081.37 | false | 2 | Smaller payload, little latency gain |
| 448 wide | 20,636 | 17,191.79 | true | 4 | Best quality/latency balance |
| 336 wide | 13,624 | 17,053.03 | false | 2 | Slightly faster, weaker semantic signal |

Recommendation: use `resize_width=448` as the default remote VLM image size. It materially reduces payload size and keeps the scene/object signal usable on this sample. `336` is slightly faster but loses the object cue in the heuristic check.

## Prompt / max_tokens Ablation

All prompt ablations use `resize_width=448`.

| Prompt | max_tokens | Latency ms | Output chars | Object mentioned | Heuristic quality |
|---|---:|---:|---:|---|---:|
| verbose baseline | 32 | 19,249.87 | 129 | false | 2 |
| verbose baseline | 64 | 16,187.88 | 288 | true | 4 |
| verbose baseline | 128 | 16,273.38 | 523 | true | 3 |
| concise scene | 32 | 16,691.21 | 138 | false | 2 |
| concise scene | 64 | 16,691.29 | 264 | true | 4 |
| concise scene | 128 | 17,714.86 | 509 | true | 3 |
| concise VQA | 32 | 17,614.51 | 114 | false | 2 |
| concise VQA | 64 | 18,129.30 | 251 | true | 4 |
| concise VQA | 128 | 18,808.63 | 495 | true | 3 |
| structured JSON | 32 | 16,869.25 | 102 | false | 2 |
| structured JSON | 64 | 18,560.88 | 237 | false | 2 |
| structured JSON | 128 | 18,872.30 | 469 | true | 3 |

The VLM still tends to emit reasoning-style text despite the prompt. Reducing `max_tokens` helps bound output length, but `32` tokens is often too short to preserve object mentions. `64` is the best default tradeoff in this run.

## Recommended Defaults

For v0.8-style remote semantic vision tasks:

- image resize: `448` px wide
- scene prompt: `Describe the image in one concise sentence. Do not explain your reasoning.`
- VQA prompt style: `Answer briefly. <question>`
- max tokens: `64`
- mode: `subprocess_cli_mode`

Observed latency improvement:

```text
Original verbose baseline: about 19.0 s
Recommended 448 + concise scene + 64 tokens: about 16.7-17.2 s
```

This is a modest improvement because the subprocess/model-load path dominates. It is still useful because it reduces payload size and removes unnecessary image resolution without changing the model.

## What Was Not Changed

- No new VLM was downloaded.
- No Qwen3-VL comparison was added.
- No TensorRT/VLM acceleration was attempted.
- No persistent multimodal server was implemented.
- No batching, streaming, or async job mode was added.
- No cache was implemented in this phase.

## Next Steps

The next meaningful latency improvement is likely architectural:

- keep the VLM model resident in a persistent server process;
- evaluate llama.cpp multimodal server support instead of per-request CLI subprocesses;
- add async job mode for high-latency visual reasoning;
- add response cache keyed by image hash, prompt, max tokens, resize policy, and model id;
- compare Qwen3-VL or another compact VLM only as a separate model-selection experiment.
