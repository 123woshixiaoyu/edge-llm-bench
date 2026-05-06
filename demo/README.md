# EdgeLog Demo

EdgeLog is a **local-first semantic event memory system for video**.

The demo is not a YOLO object log and not a router debugger. YOLO/motion/ROI changes are cheap triggers that propose candidate events. A semantic verifier decides `YES`, `NO`, or `UNKNOWN` for the user's event rule. Slow VLM/LLM backends add descriptions and daily summaries later.

## Pages

- **Live Event Stream**: capture once or run a page-local snapshot loop. Cheap triggers propose candidate events, and the verifier promotes/rejects them.
- **Event Rules**: edit the semantic event rule, ROI name, privacy mode, verifier backend, trigger interval, and risk level.
- **Event Search**: search semantic events by rule, object evidence, ROI, risk, verifier answer, and description.
- **Daily Summary**: summarize verified/high-risk/unknown events from structured records, not from raw video.
- **System Status**: Jetson Gateway, local LLM/CV, RTX LLM, and RTX VLM readiness.
- **Model / Routing Policy**: explains YOLO as trigger, SmolVLM2 as fast verifier, Gemma as slow describer, and Qwen as summary assistant.

## Start Real Mode

From the repository root:

```bash
python3 demo/run_interactive_stack.py
python3 demo/check_interactive_stack.py
streamlit run demo/app.py
```

The stack launcher also starts the optional RTX fast verifier when `.venv-vlm` is available:

```text
SmolVLM2 verifier: http://127.0.0.1:8092
```

The Streamlit page usually opens at `http://127.0.0.1:8501`. Real mode should point to the Jetson Gateway, defaulting to:

```text
http://192.168.1.102:8000
```

Override it if the Jetson IP changes:

```bash
EDGE_GATEWAY_URL=http://custom-jetson:8000 streamlit run demo/app.py
```

## Semantic Event Rule

The verifier prompt uses a short final-line protocol:

```text
FINAL_ANSWER: YES
REASON: short reason
```

Allowed answers are `YES`, `NO`, and `UNKNOWN`. Strict JSON is intentionally not the default for fast verifier candidates such as SmolVLM2.

Current implementation:

- `mock_final_line` validates the workflow without adding a new service.
- `smolvlm2_fast` calls the RTX SmolVLM2 verifier service at `EDGELOG_FAST_VLM_URL` or `http://127.0.0.1:8092`.
- Gemma VLM remains the slower async describer for confirmed/high-value events.

## Event Memory And Retention

EdgeLog does not save every auto-refresh frame. Ordinary frames overwrite the latest snapshot only. Long-term history stores proposals/events that are verified, unknown, failed, rejected, reviewed, summarized, or explicitly saved.

Runtime data is ignored by git:

```text
runtime_data/events/events.jsonl
runtime_data/events/images/
runtime_data/events/latest_snapshot.jpg
```

Default retention:

- `MONITORING_MAX_EVENTS=500`
- `MONITORING_MAX_IMAGES_MB=512`
- `MONITORING_RETENTION_DAYS=7`

See [../docs/storage_retention_policy.md](../docs/storage_retention_policy.md).

## Output Budgets

- `max_tokens` controls model output length.
- `latency_budget_ms` controls routing constraints.
- `Request timeout seconds` controls how long the UI waits.

Daily summaries read event metadata and completed semantic descriptions. They do not read video frames.

## Limits

- Single camera, fixed scene.
- Snapshot loop, not WebRTC/video streaming.
- Fast verifier real mode requires the RTX SmolVLM2 service; mock mode remains available for offline demos.
- Keyframe retention is stable; `clip_path` is reserved for a future ring buffer.
- Search is JSONL keyword/filter search, not SQLite FTS5 or embeddings yet.
- No login, cloud deployment, face recognition, or production audit database.
