# EdgeLog v1 Validation

## Scope

This validation checks the EdgeLog v1 product loop:

```text
Jetson local detection -> event state -> retained event -> search -> daily summary -> async semantic status update
```

The validation is intentionally lightweight. It does not start new models and does not write runtime camera images into git.

## Hardware / Runtime Context

- Jetson Orin Nano 8GB is the target local device.
- RTX 5090 laptop / WSL is the target remote semantic backend.
- Existing gateway, YOLO TensorRT, remote LLM, and remote VLM work remain unchanged.
- Validation script: `scripts/validate_edgelog_v1.py`
- Result CSV: `serving/results/raw/edgelog_v1_validation.csv`

Current health check during this validation:

- Jetson Gateway reachable: yes.
- Local backend available: yes.
- Remote LLM and remote VLM services reachable from WSL: yes.
- Jetson Gateway reported remote backends unavailable: yes, likely stale or inactive reverse tunnel from Jetson to WSL.

Because the live remote path was not fully ready at validation time, the EdgeLog v1 checks below use simulated state-machine events plus the existing proven gateway/VLM evidence from earlier stages. Re-run `demo/run_interactive_stack.py` and `demo/check_interactive_stack.py` before recording a live demo.

## Validation Results

| Check | Status | Notes |
| --- | --- | --- |
| person enter / local event created | PASS | Simulated YOLO `person` state transition. |
| ROI intrusion event created | PASS | Simulated person center inside `front_desk_roi`. |
| object_change event created | PASS | Rule-based smoke uses synthetic detections and coarse ROI object signature. |
| loitering event created | PASS | Simulated time threshold crossing. |
| cooldown prevents duplicate spam | PASS | Repeated identical frame does not create duplicate enter/intrusion events. |
| event search returns expected event | PASS | JSONL keyword/filter search finds stored ROI event. |
| daily summary contains correct counts | PASS | Deterministic summary count matches retained event table. |
| async VLM review updates semantic_status | PASS | Simulated update changes event from pending to completed. |
| retention prevents ordinary frame spam | PASS | Ordinary frame updates latest snapshot only, not long-term history. |
| system status page can read backend readiness | PASS / NEEDS_USER_CHECK | Real stack should be checked with `demo/check_interactive_stack.py`. |

## Real vs Simulated Coverage

Real camera and backend capabilities were validated in earlier project stages. This v1 validation focuses on the new EdgeLog state/schema/search/summary behavior.

- Real camera tested earlier: CSI IMX219 + GStreamer Argus.
- Real local detector tested earlier: YOLOv8n TensorRT FP16 through the vision router.
- Real remote VLM tested earlier: Gemma VLM through the Jetson gateway/tunnel path.
- Simulated here: event state transitions for the four EdgeLog event types.

This is enough for product workflow validation, but not a production video analytics benchmark.

## Known Limits

- `object_change` is a simple rule-based smoke, not robust object identity tracking.
- `loitering` validation uses simulated time.
- `clip_path` is schema-only in v1; keyframes are the stable artifact.
- Search is JSONL keyword/filter search, not SQLite FTS5 or embeddings.
- System status readiness should be rechecked before a live demo with:

```bash
python3 demo/run_interactive_stack.py
python3 demo/check_interactive_stack.py
```

## Recommendation

EdgeLog v1 is ready as a product prototype narrative:

- local-first event trigger path on Jetson
- async semantic annotation on RTX
- event memory with retention
- local search and daily summaries

The next engineering step is a clip ring buffer, not realtime VLM.
