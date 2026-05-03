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
- add camera/VLM routing later, with Gemma multimodal policy separated from text-only routing;
- add concurrency tests and overload behavior once the single-request policy is stable.
