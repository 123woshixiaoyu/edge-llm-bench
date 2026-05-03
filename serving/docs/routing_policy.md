# Routing Policy

## Inputs

The router uses:

- messages / prompt text
- task type, either user-specified or inferred
- quality: `low`, `medium`, `high`
- privacy: `local_only`, `allow_remote`
- latency budget
- estimated prompt tokens, using `chars / 4`
- backend availability
- mocked Jetson queue depth and temperature

## Task Analysis Rules

If `task_type` is not provided, the analyzer uses simple rules:

- prompt includes `code`, `python`, `debug`, `function`, or `traceback`: `code`
- prompt includes `summarize`, `summary`, `总结`, or `概括`: `summary`
- prompt includes `reason`, `plan`, `why`, `分析`, `推理`, or `规划`: `reasoning`
- estimated prompt tokens > 512 or chars > 2048: `long_context`
- otherwise: `qa`

These rules are intentionally transparent. They are easy to inspect and easy to replace later.

## Local Route Rules

A request should run locally when:

- privacy is `local_only` and local backend is available;
- or task type is `qa` / `summary`;
- and estimated prompt tokens are at or below the local threshold;
- and quality is not `high`;
- and Jetson state is healthy;
- and local backend is available.

Engineering reason: Project 1 showed Qwen3.5 0.8B Q4_K_M is fast and efficient on Jetson. It is the right default for short, private, or latency-sensitive simple tasks.

Default local model:

```text
qwen35_08b_q4
```

In v0.2 this model can be backed by a real Jetson `llama-server` endpoint using the `hybrid` backend mode. In that configuration, local routes call Qwen3.5 0.8B Q4_K_M through llama.cpp instead of returning mock text.

Quality-oriented local candidate:

```text
qwen35_4b_q4
```

Gemma medium candidate kept in config:

```text
gemma4_e2b_q4
```

## Remote Route Rules

A request should go remote when privacy allows remote execution and one or more of these are true:

- task type is `code`, `reasoning`, or `long_context`;
- quality is `high`;
- estimated prompt tokens exceed the local threshold;
- Jetson queue depth is at or above the queue limit;
- Jetson temperature is at or above the thermal limit;
- local backend is unavailable.

Engineering reason: these tasks are either more quality-sensitive or more likely to stress Jetson. The RTX backend should absorb tasks that need more compute, longer context, or higher answer quality.

Default remote model:

```text
qwen35_4b_q4
```

High-quality remote placeholder:

```text
remote_large_model
```

The v0.2 remote route remains a mock placeholder. It proves that the router can keep sending complex or high-quality tasks away from Jetson, but the real RTX backend is deliberately left for a later stage.

In v0.3 the remote route can be backed by a real RTX 5090 `llama-server` running Qwen3.5 4B Q4_K_M. The router still uses the same policy triggers, but the action changes from placeholder behavior to actual heterogeneous inference:

- simple QA, short summary, and `privacy=local_only` stay on Jetson;
- code, reasoning, long-context, and `quality=high` tasks go to RTX when privacy allows;
- impossible latency budgets and unsafe fallback cases are still rejected.

Engineering reason: Jetson should keep low-latency and privacy-sensitive work near the edge, while RTX absorbs tasks that benefit from the larger 4B model and higher compute budget.

## v0.4 Reliability Rules

Queue-aware routing:

- local route increments `local_queue_depth` while the backend call is inflight and decrements it in `finally`;
- remote route does the same for `remote_queue_depth`;
- when `local_queue_depth >= local_queue_limit`, `allow_remote` requests are routed remote;
- when `privacy=local_only` and local queue is overloaded, the request is rejected because remote fallback would violate privacy.

Fallback behavior:

- if local is unavailable and privacy allows remote, simple tasks can fall back to remote;
- if local is unavailable and `privacy=local_only`, the request is rejected;
- if remote is unavailable for code, reasoning, long-context, or `quality=high`, the request is rejected instead of being forced onto Jetson;
- both-backends-unavailable is always rejected.

Telemetry-aware routing:

- `jetson_temp_c >= max_jetson_temp_c` pushes `allow_remote` tasks to RTX;
- `privacy=local_only` under high temperature rejects, because the only safe backend is currently unsafe;
- v0.4 exposes `/state` and `/state/reset` for simulation-based smoke tests. A real `tegrastats` sidecar remains future work.

## v0.5a Vision Routing Rules

The vision path is separate from the text router. It is implemented by `serving/app/camera.py`, `serving/app/local_cv.py`, `serving/app/vision_analyzer.py`, `serving/app/vision_policy.py`, `serving/app/vision_router.py`, and `serving/scripts/smoke_vision_router.py`.

Vision request fields:

- `task_type`: `detect`, `classify`, `vqa`, `scene_description`
- `privacy`: `local_only`, `allow_remote`
- `quality`: `low`, `medium`, `high`
- `image_source`: `camera`, `file`
- `latency_budget_ms`

Local vision route:

- `task_type=detect` or `classify`;
- local CV backend is available;
- `quality` is not `high`.

Engineering reason: simple object detection/classification can be handled on Jetson without sending images off-device. v0.5a uses MobileNet-SSD through OpenCV DNN as a small, real local CV baseline.

Remote vision route:

- `privacy=allow_remote`;
- `task_type=vqa` or `scene_description`;
- or `quality=high`.

Engineering reason: high-level visual explanation needs a VLM, which is outside the current Jetson local baseline. v0.5a marks this route as `remote_is_mock=true`; v0.5b attaches a real RTX/WSL VLM backend after model selection.

Vision reject route:

- `privacy=local_only` and the task requires semantic visual reasoning;
- unknown vision task type;
- semantic task with an impossible latency budget;
- required backend unavailable.

Engineering reason: a private image should not be sent to a remote VLM just because the local detector cannot explain the whole scene. Rejecting is the safer and more honest behavior.

## v0.5b Real VLM Routing Rules

v0.5b keeps the same vision policy and changes the remote action from a placeholder to a real backend call:

- local `detect` / `classify`: Jetson MobileNet-SSD through OpenCV DNN;
- remote `vqa` / `scene_description`: RTX/WSL Gemma 4 E2B-it multimodal backend when privacy allows;
- remote high-quality visual task: RTX/WSL VLM when privacy allows;
- reject `privacy=local_only` semantic vision tasks: no remote image transfer.

The selected remote backend is:

```text
gemma4_e2b_it_q4_mmproj
```

It is implemented as a small HTTP wrapper around `llama-mtmd-cli`, using the existing local model files:

```text
/mnt/d/AI/Models/gemma4/E2B-it/gemma-4-E2B-it-Q4_K_M.gguf
/mnt/d/AI/Models/gemma4/E2B-it/mmproj-F16.gguf
```

Images are transmitted as base64 in the request body. The router must not send a Jetson filesystem path as the remote input because that path is not readable from WSL and would make the backend depend on shared storage.

The current smoke run uses an SSH reverse tunnel:

```text
Jetson 127.0.0.1:18091 -> RTX/WSL 127.0.0.1:8091
```

Policy evidence in `serving/results/raw/vision_router_real_vlm_smoke.csv`:

- `10/10` expected routes matched;
- route distribution: local `4`, remote `4`, reject `2`;
- all remote rows have `remote_is_mock=false`;
- local-only semantic requests are rejected and are not sent to the RTX VLM.

## Reject Rules

A request should be rejected when:

- privacy is `local_only` but local backend is unavailable;
- estimated prompt tokens exceed the total prompt limit;
- latency budget is too low for a complex task;
- both local and remote backends are unavailable;
- remote would be required but remote is unavailable and local fallback is unsafe.

Engineering reason: rejection is better than pretending the system can safely satisfy a request it cannot handle. This is especially important for privacy and latency guarantees.

## Current Thresholds

From `serving/configs/policy.yaml`:

- `max_local_prompt_tokens`: 512
- `max_total_prompt_tokens`: 4096
- `local_queue_limit`: 2
- `max_jetson_temp_c`: 75
- `low_latency_budget_ms`: 5000
- `too_low_complex_latency_ms`: 1500

## Why The Policy Uses Project 1 Data

Project 1 established that:

- Qwen3.5 0.8B Q4_K_M is the best low-latency Jetson default.
- Qwen3.5 4B Q4_K_M is feasible but much slower, so it belongs in a quality-oriented lane.
- Q8/F16 are useful for reference and conservative checks but not for default Jetson serving.

This lets the router policy be grounded in benchmark evidence instead of arbitrary model preferences.

## Extension Path

The MVP can grow in stages:

- replace mocked queue depth with real in-process queue tracking;
- parse `tegrastats` or a telemetry sidecar for thermal-aware routing;
- add streaming support to `/v1/chat/completions`;
- replace the SSH tunnel with a production network path or service discovery entry;
- add per-task quality evaluation logs;
- turn the v0.5a script-driven camera path into HTTP endpoints once the model and privacy boundaries are stable;
- improve the v0.5b remote VLM server by keeping the model resident instead of launching `llama-mtmd-cli` once per request;
- add concurrency tests and overload behavior once the single-request policy is stable.
