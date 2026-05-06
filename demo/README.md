# EdgeLog Demo

EdgeLog is a **local-first video event memory box for semantic search and daily summaries**.

The Streamlit app is a product workbench, not a router debugger. Jetson handles the fast path: camera snapshot, YOLO TensorRT, simple event rules, keyframe retention. RTX backends handle async semantic descriptions and optional daily narrative summaries when privacy allows.

## Pages

- **Live Event Stream**: capture a snapshot or start a page-local snapshot loop. Jetson YOLO/rules create `person_enter_exit`, `roi_intrusion`, `object_change`, and `loitering` events.
- **Event Search**: keyword/filter search over local JSONL events by type, objects, ROI, risk, and semantic status.
- **Daily Summary**: deterministic counts, high-risk events, pending semantic reviews, completed descriptions, and timeline. Optional LLM narrative summary reads only event metadata.
- **System Status**: Jetson Gateway, local LLM/CV, RTX LLM, and RTX VLM readiness.
- **Model / Routing Policy**: read-only explanation of Jetson fast path, RTX async semantic path, and privacy reject behavior.

## Start Real Mode

From the repository root:

```bash
python3 demo/run_interactive_stack.py
python3 demo/check_interactive_stack.py
streamlit run demo/app.py
```

The Streamlit page usually opens at `http://127.0.0.1:8501`. Real mode should point to the Jetson Gateway, defaulting to:

```text
http://192.168.1.102:8000
```

Override it if the Jetson IP changes:

```bash
EDGE_GATEWAY_URL=http://custom-jetson:8000 streamlit run demo/app.py
```

Stop the launched stack with:

```bash
python3 demo/stop_interactive_stack.py
```

## Sample Mode

Sample mode reads committed CSV/image evidence and does not require Jetson, RTX, model files, or TensorRT engines. It is evidence playback. Real backend mode calls the Jetson Gateway.

## Event Memory And Retention

EdgeLog does not save every auto-refresh frame. Ordinary frames overwrite the latest snapshot only. Long-term history stores:

- triggered events
- local alerts
- async VLM review events
- privacy rejects
- backend errors/fallbacks
- assistant summaries
- user-saved snapshots

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

## Async VLM Review

VLM is not in the real-time loop. The flow is:

```text
event keyframe -> queue semantic review -> RTX VLM -> semantic_status completed/failed
```

The UI shows `semantic_status` as `not_required`, `pending`, `completed`, or `failed`. A slow remote VLM request is acceptable because the event has already been created locally.

## Output Budgets

- `max_tokens` controls model output length.
- `latency_budget_ms` controls routing constraints.
- `Request timeout seconds` controls how long the UI waits.

Daily summaries do not send video to an LLM. They summarize structured event metadata and completed semantic descriptions.

## Limits

- Single camera, fixed scene.
- Snapshot loop, not WebRTC/video streaming.
- Keyframe retention is stable; `clip_path` is reserved for a future ring buffer.
- Search is JSONL keyword/filter search, not SQLite FTS5 or embeddings yet.
- No login, cloud deployment, face recognition, or production audit database.
