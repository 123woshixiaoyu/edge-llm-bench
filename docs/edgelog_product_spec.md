# EdgeLog Product Spec

## Positioning

**EdgeLog: Local-first video event memory box for semantic search and daily summaries.**

EdgeLog turns a fixed camera stream into searchable, summarizable event memory. Jetson handles realtime detection and event slicing. RTX backends asynchronously add semantic understanding and daily summaries. The user searches history instead of manually reviewing long video.

The gateway remains important infrastructure, but it is no longer the product headline.

## Why Event Memory

All-day recording creates a review problem: the user still has to watch video. EdgeLog keeps the useful parts:

- structured event records
- keyframes
- event type and objects
- ROI/risk metadata
- optional async semantic descriptions
- daily summaries

Ordinary frames are overwritten as the latest snapshot and are not retained forever.

## Jetson / RTX Split

| Device | Responsibility |
| --- | --- |
| Jetson | Camera capture, YOLO TensorRT inference, simple event rules, keyframe/event retention, privacy-first reject path. |
| RTX workstation | Async VLM event review, higher-quality text summaries, heavier semantic reasoning. |

VLM is explicitly asynchronous. A 15-30 second remote VLM request is acceptable for post-event annotation but not for realtime triggering.

## EdgeLog v1 Event Types

| Event type | Trigger | v1 implementation |
| --- | --- | --- |
| `person_enter_exit` | Person enters/leaves scene | YOLO `person` state transition. |
| `roi_intrusion` | Person enters ROI | Person box center inside configured ROI. |
| `object_change` | Object appears/disappears/moves in ROI | Keyframe object signature from YOLO labels and coarse centers. |
| `loitering` | Person remains in ROI too long | Time threshold, default 10 seconds. |

Out of scope for v1:

- face recognition
- "who is this person"
- multi-camera
- complex behavior recognition
- realtime VLM
- cloud upload
- production video management

## Event Schema

Each retained event normalizes to:

```json
{
  "event_id": "...",
  "event_type": "person_enter_exit | roi_intrusion | object_change | loitering | assistant_summary | privacy_reject | backend_error",
  "start_time": "...",
  "end_time": "...",
  "duration_s": 0.0,
  "status": "active | ended | cooldown",
  "objects": [],
  "roi_name": null,
  "confidence": null,
  "risk_level": "low | medium | high",
  "semantic_status": "not_required | pending | completed | failed",
  "semantic_description": "",
  "keyframe_path": null,
  "clip_path": null,
  "route": "local | remote | reject",
  "backend": "",
  "latency_ms": null,
  "created_at": "..."
}
```

`clip_path` is reserved for a future ring buffer. v1 keeps keyframes as the stable event artifact.

## Search MVP

The current search is local JSONL keyword/filter search across:

- event type
- objects
- ROI name
- risk level
- semantic status
- semantic description

This is intentionally simple and stable. The next backend can be SQLite FTS5, then optional embedding/FAISS retrieval.

## Daily Summary MVP

Daily summary is generated from structured event records, not from video:

- total event count
- counts by event type
- high-risk events
- pending semantic reviews
- completed semantic descriptions
- timeline

An optional LLM narrative summary can read the event table. It never receives raw video.

## Current Completed Capabilities

- Jetson CSI camera smoke and real capture path.
- YOLOv8n TensorRT FP16 local CV backend integrated into the vision router.
- Remote LLM and remote VLM backends through SSH reverse tunnels.
- Streamlit EdgeLog UI with Live Event Stream, Event Search, Daily Summary, System Status, and Model / Routing Policy.
- Event JSONL store with retention policy.
- Async VLM review queue for triggered events.
- Validation CSV for EdgeLog v1 state/search/summary/retention behavior.

## Roadmap

1. Clip ring buffer with pre-event/post-event seconds.
2. SQLite FTS5 event index.
3. Optional embedding search for semantic descriptions.
4. Persistent remote VLM server to reduce subprocess latency.
5. YOLO-World or open-vocabulary trigger backend after controlled Jetson validation.
6. Recorded demo and submission packaging.
