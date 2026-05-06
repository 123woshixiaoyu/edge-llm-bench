# EdgeLog Product Spec

## Positioning

**EdgeLog: Local-first semantic event memory for video.**

EdgeLog does not store all-day video and does not create a log of raw object detections. It converts a fixed camera feed into semantic event memory:

```text
cheap trigger proposes candidate -> fast VLM verifier decides YES/NO/UNKNOWN -> event memory stores semantic event -> slow describer/search/daily summary
```

## Why Not Object Detection Logs

Users do not want a timeline of "chair detected" or "bottle detected." They want answers to event questions:

- Did someone enter a restricted area?
- Did someone approach a dangerous zone?
- Was equipment removed?
- Was a door or cabinet left open?
- Did someone linger too long?
- What events today deserve review?

YOLO TensorRT is useful because it is fast and local, but it is only a cheap trigger. VLM-style verification is the semantic decision layer.

## Three-Layer Event Understanding

### 1. Cheap Candidate Trigger

Purpose: quickly notice that something may have happened.

Possible signals:

- motion or frame difference
- person detector
- ROI overlap
- object change
- YOLO label
- scene change

Output is an `event_proposal`, not a final event:

```json
{
  "proposal_id": "...",
  "trigger_type": "motion | person | roi_overlap | object_change | yolo_label | scene_change",
  "timestamp": "...",
  "roi_name": "...",
  "objects": [],
  "keyframe_path": "...",
  "confidence": 0.0,
  "proposal_reason": "motion detected in desk ROI"
}
```

### 2. Fast VLM Semantic Verifier

Purpose: decide whether the candidate matches the user-defined event rule.

Protocol:

```text
FINAL_ANSWER: YES
REASON: short reason
```

Allowed answers are `YES`, `NO`, and `UNKNOWN`. Strict JSON is not the default because SmolVLM2 benchmark evidence showed that final-line protocols are more robust than strict JSON for compact VLMs.

Current implementation:

- `mock_final_line` verifies the workflow.
- SmolVLM2 is a fast verifier candidate based on benchmark evidence.
- A live SmolVLM2 service is future work.

### 3. Slow Semantic Describer / Summarizer

Purpose: describe confirmed or high-value events after they are already stored.

Backends:

- Gemma VLM for event descriptions.
- Qwen text LLM for daily summaries/search assistance.

Slow VLM is not in the real-time loop.

## Event Schema

Events normalize to proposal + verification + description + storage:

```json
{
  "event_id": "...",
  "event_rule": "someone approaches the chemical cabinet",
  "event_type": "semantic_event | roi_intrusion | object_change | loitering | person_activity | unknown",
  "status": "proposed | verified | rejected | unknown | described | failed",
  "start_time": "...",
  "end_time": "...",
  "duration_s": 0.0,
  "proposal": {
    "proposal_id": "...",
    "trigger_type": "motion | person | roi_overlap | object_change | yolo_label | scene_change",
    "proposal_reason": "...",
    "objects": [],
    "roi_name": null,
    "confidence": null,
    "keyframe_path": null,
    "clip_path": null
  },
  "verification": {
    "verifier_backend": "smolvlm2_fast | gemma_vlm | mock",
    "semantic_status": "not_required | pending | yes | no | unknown | failed",
    "final_answer": "YES | NO | UNKNOWN",
    "reason": "",
    "latency_ms": null
  },
  "description": {
    "describer_backend": "gemma_vlm | qwen_text | deterministic",
    "semantic_description": "",
    "risk_level": "low | medium | high",
    "latency_ms": null
  },
  "storage": {
    "keyframe_path": null,
    "clip_path": null,
    "stored_image": false,
    "sent_to_remote": false,
    "retention_expires_at": null
  },
  "route": "local | remote | reject",
  "backend": "",
  "created_at": "..."
}
```

Flat legacy fields remain for compatibility with older JSONL rows.

## Search MVP

Search currently uses local JSONL keyword/filter matching over:

- event rule
- event type
- proposal reason
- objects
- ROI name
- risk level
- verification result
- semantic description

Future work: SQLite FTS5, then optional embedding/FAISS retrieval.

## Daily Summary MVP

Daily summary reads structured event records, not video frames:

- total verified/unknown events
- counts by event type
- high-risk events
- pending semantic review
- completed descriptions
- timeline

Rejected candidates are not counted as meaningful events.

## Current Completed Capabilities

- Jetson CSI camera and YOLO TensorRT local fast path.
- Gateway local/remote/reject infrastructure.
- Event proposal generation from YOLO/ROI/object-change signals.
- Mock FINAL_ANSWER verifier workflow.
- JSONL event memory with retention.
- Search and deterministic daily summaries.
- Async slow VLM description path remains available for event review.

## Roadmap

1. Connect live SmolVLM2 final-line verifier service on RTX.
2. Add clip ring buffer with pre/post event seconds.
3. Add SQLite FTS5 event index.
4. Add embedding search for semantic descriptions.
5. Evaluate YOLO-World as a custom cheap trigger after controlled Jetson runtime validation.
