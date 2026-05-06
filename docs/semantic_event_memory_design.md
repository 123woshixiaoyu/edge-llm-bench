# Semantic Event Memory Design

## Core Change

EdgeLog is not an object detection log. The core product question is not "what objects did YOLO detect?" The question is "did an event the user cares about happen?"

The pipeline is:

```text
Cheap Candidate Trigger
  -> event_proposal
  -> Fast VLM Semantic Verifier
  -> semantic event
  -> Slow Semantic Describer / Search / Daily Summary
```

## Layer 1: Cheap Candidate Trigger

Cheap triggers are fast, local, and intentionally shallow. They wake up the semantic layer but do not decide final meaning.

Examples:

- motion / frame difference
- person detector
- ROI overlap
- object change
- YOLO label
- scene change

Output:

```json
{
  "proposal_id": "...",
  "trigger_type": "motion | person | roi_overlap | object_change | yolo_label | scene_change",
  "timestamp": "...",
  "roi_name": "desk_roi",
  "objects": ["person"],
  "keyframe_path": "...",
  "confidence": 0.91,
  "proposal_reason": "person entered desk_roi"
}
```

## Layer 2: Fast VLM Semantic Verifier

The verifier checks the proposal against a user-defined event rule:

```text
User rule: "Alert me when someone approaches the chemical cabinet."
Question: Does this candidate frame show that event?
Answer with exactly one FINAL_ANSWER line: YES, NO, or UNKNOWN, then one short REASON line.
```

Expected response:

```text
FINAL_ANSWER: YES
REASON: person is near the configured cabinet area
```

Why final-line instead of strict JSON:

- SmolVLM2 benchmark showed sub-second RTX latency but strict JSON validity was poor.
- Output robustness tests showed simpler parse protocols are more reliable.
- The verifier should not write long scene descriptions.

Current status:

- Mock verifier validates workflow in the demo.
- SmolVLM2 is an RTX fast-verifier candidate.
- Jetson SmolVLM2 deployment is future work.

## Layer 3: Slow Semantic Describer / Summarizer

The slow describer runs only after a proposal is verified or needs review.

Roles:

- Gemma VLM: event-level semantic description.
- Qwen text LLM: daily summary and search assistant.

This layer does not run on every frame and does not participate in realtime triggering.

## Event Promotion

| Verifier result | Event action |
| --- | --- |
| `YES` | Promote proposal to `verified` semantic event. |
| `NO` | Mark proposal `rejected`; do not count in daily meaningful event totals. |
| `UNKNOWN` | Save as `unknown` / needs review. |
| failure | Save as `failed`; live monitoring continues. |

## Why This Matters

YOLO can detect fixed classes quickly, but it cannot answer "is this a safety event?" or "was the important device removed?" by itself. VLMs can reason about semantic rules, but they are too slow and expensive to run on every frame. The three-layer design combines both:

- local cheap triggers for speed;
- fast VLM verifier for semantic yes/no decisions;
- slow describer for rich event memory.

## Current Limitations

- Fast verifier is mock-integrated, not a live SmolVLM2 service.
- Object change is rule-based and coarse.
- Search is JSONL keyword/filter search.
- Clip storage is schema-ready but not implemented as a ring buffer.
