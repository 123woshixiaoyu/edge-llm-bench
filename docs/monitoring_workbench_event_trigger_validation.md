# Monitoring Workbench Event-Triggered Validation

Date: 2026-05-05

Scope: second-stage Monitoring Workbench validation for output completeness, local snapshot monitoring, event triggers, privacy blocking, and event-level VLM review.

Evidence CSV: [monitoring_workbench_event_trigger_validation.csv](../serving/results/raw/monitoring_workbench_event_trigger_validation.csv)

## Results

| Check | Result | Notes |
| --- | --- | --- |
| System status | PASS | Gateway, local LLM, remote LLM, and real remote VLM were ready; `vision_remote_is_mock=false`. |
| Live Monitor camera detection | PASS | Jetson camera snapshot routed local through YOLO TensorRT; labels observed: `bottle`, `bowl`; local CV latency `15.29 ms`. |
| Local alert trigger | PASS | Lightweight rule matched watched label `bottle` and produced a local candidate alert. |
| Local alert history event | PASS | Event persisted in JSONL runtime history. |
| Privacy blocked event | PASS | `local_only` plus required semantic review produced a privacy-blocked reject event. |
| VLM structured review | PASS | Event-level VLM review routed remote, `remote_is_mock=false`, structured answer parsed successfully; remote latency `25121.52 ms`. |
| Thinking-only validation | PASS | A thinking-only output was classified as `incomplete_generation` instead of being displayed as final answer. |
| Monitoring Assistant output | PASS | Text backend used `1024` output-token budget and returned a complete monitoring answer. |
| Incomplete-generation history event | PASS | Runtime history can represent an `incomplete_generation` event state. |

## Implementation Notes

- `max_tokens` defaults to `1024` for text and vision generation controls.
- `latency_budget_ms` remains a routing constraint, not an output-length control.
- UI request timeout remains a client wait limit.
- VLM review is event-triggered and queued through a single background worker, so it does not block the Live Monitor snapshot loop.
- VLM responses are accepted as final answers only when they contain a final marker, valid structured JSON, or a non-thinking direct answer.
- If generation contains thinking/analysis without a final answer, the workbench marks the event `incomplete_generation` and keeps raw output in a debug view.

## Limits

- Auto refresh is a Streamlit snapshot loop, not WebRTC or true video streaming.
- VLM review still uses the existing remote VLM backend and can take 20+ seconds.
- Event history is JSONL runtime state, not a production database.
