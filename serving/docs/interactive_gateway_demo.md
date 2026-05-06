# EdgeLog Interactive Demo

The interactive demo is now **EdgeLog: local-first semantic event memory for video**.

The UI is no longer a generic gateway dashboard. It demonstrates a product workflow:

```text
cheap visual trigger -> candidate proposal -> semantic verifier -> event memory -> search/daily summary
```

The Jetson Gateway still provides local/remote/reject inference infrastructure. The UI does not call RTX directly.

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

The Streamlit UI runs locally, usually at `http://127.0.0.1:8501`, but real backend mode should point to the Jetson Gateway.

## EdgeLog Pages

- **Live Event Stream**: cheap triggers create proposals; verifier marks YES / NO / UNKNOWN.
- **Event Rules**: define the natural-language event rule, ROI, privacy mode, verifier backend, trigger interval, and risk level.
- **Event Search**: query verified, rejected, unknown, and described events.
- **Daily Summary**: summarize structured event records and completed descriptions.
- **System Status**: Jetson Gateway, local LLM/CV, RTX LLM, and RTX VLM readiness.
- **Model / Routing Policy**: explains model roles.

## Model Roles

| Model / signal | Role |
| --- | --- |
| YOLOv8n TensorRT / motion / ROI / scene change | Cheap candidate trigger. Not final event understanding. |
| SmolVLM2 | Fast semantic verifier candidate using FINAL_ANSWER yes/no protocol. Not default live service yet. |
| Gemma VLM | Slow async semantic describer for verified or high-value events. |
| Qwen text LLM | Daily summary and search assistant over event metadata. |

## Event Schema

EdgeLog events are proposal-centric:

- `proposal`: cheap trigger evidence.
- `verification`: verifier backend, YES/NO/UNKNOWN answer, reason, latency.
- `description`: slow semantic description and risk.
- `storage`: keyframe/clip/retention/remote-send metadata.

Flat legacy fields are still present for compatibility with earlier JSONL history.

## VLM Timing

VLM is not run on every frame.

- Fast verifier: intended to be short-answer and event-level.
- Slow describer: async, can take 15-30 seconds with current Gemma CLI path.

This design accepts slow semantic review because the candidate proposal and keyframe are already saved.

## Retention

Ordinary frames overwrite the latest snapshot. Long-term history stores proposals/events that are verified, unknown, failed, rejected, reviewed, summarized, or explicitly saved.

Runtime paths:

```text
runtime_data/events/events.jsonl
runtime_data/events/images/
runtime_data/events/latest_snapshot.jpg
```

These are ignored by git.

## Limits

- Single camera and fixed scene.
- Snapshot loop, not video streaming.
- Mock verifier workflow until SmolVLM2 service integration.
- Keyframes are retained; `clip_path` is reserved for future ring buffer.
- Search is JSONL keyword/filter search, not SQLite FTS5 or embeddings.
