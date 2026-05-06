# EdgeLog Interactive Demo

The former v0.9 gateway demo is now framed as **EdgeLog**:

> Local-first video event memory box for semantic search and daily summaries.

The interactive stack still uses the Jetson Gateway, local YOLO TensorRT, remote text LLM, and remote VLM. The product story is no longer "generic AI gateway." The gateway is infrastructure for EdgeLog's event memory loop.

## Product Flow

```text
camera frame
  -> Jetson YOLO TensorRT + simple rules
  -> event state machine
  -> keyframe + JSONL event memory
  -> async RTX VLM semantic description
  -> search and daily summary
```

The realtime path must stay local and fast. The remote VLM is asynchronous and never part of every-frame monitoring.

## Runtime Configuration

Recommended WSL entry point:

```bash
python3 demo/run_interactive_stack.py
python3 demo/check_interactive_stack.py
streamlit run demo/app.py
```

Expected services:

- Jetson local LLM on `127.0.0.1:8080`
- Jetson Gateway on `192.168.1.102:8000`
- RTX remote LLM through Jetson tunnel `127.0.0.1:18081`
- RTX remote VLM through Jetson tunnel `127.0.0.1:18091`
- YOLO TensorRT engine at `/home/rainbow/models/vision/yolo_nano/yolov8n_fp16.engine`

The Streamlit UI runs locally, usually at `http://127.0.0.1:8501`, but real backend mode should point to the Jetson Gateway. Override with:

```bash
EDGE_GATEWAY_URL=http://custom-jetson:8000 streamlit run demo/app.py
```

The orchestrator is demo stack orchestration, not production deployment. There is no Docker, systemd, Kubernetes, or autoscaling.

## EdgeLog Pages

- **Live Event Stream**: capture once or start a snapshot loop. YOLO TensorRT and simple rules create events.
- **Event Search**: local search over retained event metadata, objects, ROI names, risk, and semantic descriptions.
- **Daily Summary**: deterministic event counts/timeline plus optional LLM narrative summary from structured metadata.
- **System Status**: product-language health for Jetson Gateway, local LLM/CV, RTX LLM, and RTX VLM.
- **Model / Routing Policy**: why Jetson is the fast path and RTX is the async semantic path.

## Event Types

EdgeLog v1 supports four fixed-scene event types:

| Event type | Trigger |
| --- | --- |
| `person_enter_exit` | YOLO person appears or disappears. |
| `roi_intrusion` | Person center enters a configured ROI. |
| `object_change` | ROI object signature changes across frames. |
| `loitering` | Person remains inside ROI beyond the configured threshold. |

Each event records:

- `event_id`
- `event_type`
- `start_time`, `end_time`, `duration_s`
- `status`: `active`, `ended`, or `cooldown`
- `objects`, `roi_name`, `confidence`, `risk_level`
- `semantic_status`: `not_required`, `pending`, `completed`, or `failed`
- `semantic_description`
- `keyframe_path`
- `clip_path` reserved for a future ring buffer
- route/backend/latency metadata

## Async Semantic Review

When a local event needs semantic review and privacy allows it:

1. EdgeLog saves the event and keyframe immediately.
2. UI shows the event with `semantic_status=pending`.
3. A background worker sends the keyframe to the RTX VLM.
4. The same event is updated to `completed` or `failed`.

If privacy is `local_only`, the event is recorded as a privacy reject and the image is not sent to the remote workstation.

Remote VLM latency of 15-30 seconds is acceptable here because the trigger is already local. VLM is for event annotation, not realtime detection.

## Search And Daily Summary

Search MVP uses local JSONL keyword/filter matching:

- event type
- objects
- ROI name
- risk level
- semantic status
- semantic description

This can later be replaced with SQLite FTS5, embeddings, or FAISS.

Daily Summary reads structured event records only:

- total events
- counts by type
- high-risk events
- pending semantic reviews
- completed semantic descriptions
- timeline

The optional LLM summary receives this table, not video frames.

## Retention

Ordinary auto-refresh frames are not retained as long-term history. They overwrite the latest snapshot. Saved records are limited to events, alerts, reviews, rejects, backend errors/fallbacks, assistant summaries, and user-saved snapshots.

Runtime paths:

```text
runtime_data/events/events.jsonl
runtime_data/events/images/
runtime_data/events/latest_snapshot.jpg
```

Defaults:

- `MONITORING_MAX_EVENTS=500`
- `MONITORING_MAX_IMAGES_MB=512`
- `MONITORING_RETENTION_DAYS=7`

See [../../docs/storage_retention_policy.md](../../docs/storage_retention_policy.md).

## API Surface

The UI still calls:

- `POST /v1/vision/analyze` for snapshot detection and semantic review
- `POST /v1/chat/completions` for optional daily narrative summaries
- `GET /health` for system status

The UI does not call RTX services directly.

## Limits

- Single camera and fixed scene.
- Snapshot loop, not video streaming.
- Keyframes are retained; event clips are schema-ready but not implemented as a ring buffer in v1.
- No face recognition, identity, multi-camera, cloud upload, login, or production audit database.
