# Monitoring Workbench Demo

This Streamlit demo presents the project as a **Jetson Local-First Monitoring Gateway**:

> A local-first edge monitoring workbench that uses Jetson for low-latency local detection and routes event review / summaries to RTX backends only when privacy and system constraints allow.

The UI is intentionally thin. It calls the existing Jetson Gateway in real mode and reads committed evidence in sample mode. It does not duplicate router policy, download models, start a database, or bypass Jetson to call RTX directly.

## Workbench Tabs

- **Live Monitor**: capture or load a single snapshot and run the Jetson YOLO TensorRT local monitoring fast path.
- **Event Review**: ask semantic questions about the current/recent snapshot. Privacy-safe mode rejects remote VLM review instead of sending the image out.
- **Monitoring Assistant**: use the text router for event summaries, routing explanations, and backend status questions.
- **Event History**: local JSONL event log for detections, reviews, rejects, and assistant summaries.
- **System Status**: product-language health view for Jetson Gateway, local LLM/CV, RTX LLM, and RTX VLM.
- **Model Policy**: read-only explanation of why each backend is used.

## Recommended Real-Mode Start

From the repository root:

```bash
python3 demo/run_interactive_stack.py
python3 demo/check_interactive_stack.py
```

This starts the RTX-side remote LLM/VLM services, SSH reverse tunnels to Jetson, the Jetson local LLM, and the Jetson Gateway. Runtime logs and PID state are written under `runtime_logs/interactive_stack/`, which is intentionally ignored by git.

Open the workbench separately:

```bash
streamlit run demo/app.py
```

The Streamlit page is usually served at `http://127.0.0.1:8501`. Streamlit itself runs locally, but real backend mode should point to the Jetson Gateway.

Or start it with the stack:

```bash
python3 demo/run_interactive_stack.py --with-ui
```

When finished:

```bash
python3 demo/stop_interactive_stack.py
python3 demo/check_interactive_stack.py
```

In the browser, **Router API base URL** defaults to `http://192.168.1.102:8000` for real backend mode. If the Jetson IP changes, override it before starting Streamlit:

```bash
EDGE_GATEWAY_URL=http://custom-host:8000 streamlit run demo/app.py
```

The old manual multi-terminal flow still works as a fallback: start the RTX remote llama-server, RTX VLM server, SSH reverse tunnels, Jetson local llama-server, Jetson Gateway, and Streamlit UI separately.

## Modes

- **Sample mode**: default. Reads existing result files and the sample camera frame. No Jetson, RTX, model files, or TensorRT engine is required. Some sample rows only store response previews.
- **Real backend mode**: optional. Live Monitor and Event Review call `/v1/vision/analyze`; Monitoring Assistant calls `/v1/chat/completions`. If the backend is unavailable, the UI shows a friendly error or sample fallback instead of crashing.

`max_tokens` controls how many output tokens the selected model may generate. It is separate from `latency_budget_ms`, which is used by the router as a routing constraint. If an answer looks cut off, increase max output tokens; this can also increase latency.

`Request timeout seconds` is the UI client's wait limit for text responses. It is separate from `latency_budget_ms`: if the timeout is too small, the UI may show committed sample fallback even though the backend would have completed with more time.

## Product Flow

- YOLO TensorRT is the local monitoring fast path for snapshot detection/classification.
- VLM is an event review tool, not the real-time monitoring engine.
- LLM is the Monitoring Assistant for summaries, policy explanations, and backend status questions.
- Router decisions remain explicit: local, remote, or reject.
- History closes the loop: detections, semantic reviews, rejects, and assistant summaries are recorded locally.

## Event History

The demo stores local runtime events in:

```text
runtime_data/events/events.jsonl
runtime_data/events/images/
```

This directory is intentionally ignored by git. Each event records source, task type, privacy, route, backend, local CV labels, latency, final answer text, reasons, errors, and mode. It is a lightweight product loop, not a production database.

## Real Mode Services

Use `serving/configs_interactive_demo` for the Jetson Gateway. Expected runtime services:

- Jetson local llama-server: `127.0.0.1:8080`
- RTX remote llama-server tunnel: `127.0.0.1:18081`
- Optional RTX remote VLM tunnel: `127.0.0.1:18091`
- Jetson YOLO TensorRT engine: `/home/rainbow/models/vision/yolo_nano/yolov8n_fp16.engine`

## Current Limits

- This is a workbench demo, not production serving.
- Sample mode is evidence playback, not live inference.
- Real vision mode is single-frame interaction, not video streaming.
- Remote VLM routes are real when the RTX VLM service/tunnel is running, but they can take 15-20 seconds.
- No login, database, cloud deploy, video stream, model download, or new benchmark is included.
