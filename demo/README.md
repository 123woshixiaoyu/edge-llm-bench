# Lightweight Demo Dashboard

This is a reviewer-facing dashboard for the Jetson-First Edge AI Inference Gateway.

The demo is intentionally thin: it does not duplicate the router implementation or require Jetson/RTX services in its default path. Sample mode reads committed CSV/image evidence from the project and shows how the gateway routes text and vision tasks across local, remote, and reject paths. Real mode calls the Jetson Gateway API.

## Recommended Real-Mode Start

From the repository root:

```bash
python3 demo/run_interactive_stack.py
python3 demo/check_interactive_stack.py
```

This starts the RTX-side remote LLM/VLM services, SSH reverse tunnels to Jetson, the Jetson local LLM, and the Jetson Gateway. Runtime logs and PID state are written under `runtime_logs/interactive_stack/`, which is intentionally ignored by git.

Open the dashboard separately:

```bash
streamlit run demo/app.py
```

Or start it with the stack:

```bash
python3 demo/run_interactive_stack.py --with-ui
```

When finished:

```bash
python3 demo/stop_interactive_stack.py
python3 demo/check_interactive_stack.py
```

In the browser, set **Router API base URL** to `http://192.168.1.102:8000` for real backend mode.

The old manual multi-terminal flow still works as a fallback: start the RTX remote llama-server, RTX VLM server, SSH reverse tunnels, Jetson local llama-server, Jetson Gateway, and Streamlit UI separately.

## Modes

- **Sample mode**: default. Reads existing result files and the sample camera frame. No Jetson, RTX, model files, or TensorRT engine is required. Some sample rows only store response previews.
- **Real backend mode**: optional. Text requests call `/v1/chat/completions`; vision requests call `/v1/vision/analyze`. If the backend is unavailable, the UI shows a friendly error or sample fallback instead of crashing. Real mode shows full model responses when the API returns them.

`max_tokens` controls how many output tokens the model may generate. It is separate from `latency_budget_ms`, which is used by the router as a routing constraint. If an answer looks cut off, increase max output tokens; this can also increase latency.

Real text requests ask llama.cpp to disable template-level thinking when supported. Real remote VLM requests use a final-answer marker so the demo can show the final semantic answer instead of the model's intermediate reasoning text.

`Request timeout seconds` is the UI client's wait limit for text responses. It is separate from `latency_budget_ms`: if the timeout is too small, the UI may show a committed sample fallback even though the model/backend is still working. For remote text routes, check the Jetson Gateway, RTX llama-server, and SSH tunnel before treating a timeout as a model failure.

## What It Shows

- Text task routing across Jetson local LLM, RTX remote LLM, and reject paths.
- Vision task routing across YOLOv8n TensorRT local CV, real remote VLM when available, and privacy rejects.
- A clear split between **Local CV Precheck**, **Routing Decision**, and **Final Routed Answer**.
- Backend status for the selected local/remote LLM/CV/VLM roles.
- Key result snapshots for quantization, ONNXRuntime session reuse, YOLO TensorRT FP16, and v0.7 reliability.

## Vision Answer Sources

Vision requests can run a cheap Jetson local CV precheck even when the final route is remote.

- Local YOLO boxes and labels come from Jetson local YOLO TensorRT.
- For `detect` / `classify` local routes, those labels/boxes are the final answer.
- For `scene_description` / `vqa` remote routes, the final semantic answer comes from the RTX remote VLM.
- For privacy rejects, no model answer is produced; the UI shows policy reasons.

This separates cheap local perception from expensive semantic reasoning.

## Current Limits

- This is a dashboard, not a production serving layer.
- Sample mode is evidence playback, not live inference.
- Sample mode may only show stored previews; use real backend mode for full responses.
- Real vision mode is single-frame interaction, not video streaming.
- Remote VLM routes are real when the RTX VLM service/tunnel is running, but they can take 15-20 seconds.
- Remote VLM requests default to a larger output budget than text preview rows because small budgets can be consumed by unwanted reasoning-style text.
- No login, database, cloud deploy, video stream, model download, or new benchmark is included.

## Real Mode Services

Use `serving/configs_interactive_demo` for the Jetson Gateway:

```bash
EDGE_ROUTER_CONFIG_DIR=serving/configs_interactive_demo \
uvicorn serving.app.main:app --host 127.0.0.1 --port 8000
```

Expected runtime services:

- Jetson local llama-server: `127.0.0.1:8080`
- RTX remote llama-server tunnel: `127.0.0.1:18081`
- Optional RTX remote VLM tunnel: `127.0.0.1:18091`
- Jetson YOLO TensorRT engine: `/home/rainbow/models/vision/yolo_nano/yolov8n_fp16.engine`
