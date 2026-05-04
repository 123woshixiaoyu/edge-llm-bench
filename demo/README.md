# Lightweight Demo Dashboard

This is a reviewer-facing dashboard for the Jetson-First Edge AI Inference Gateway.

The demo is intentionally thin: it does not duplicate the router implementation or require Jetson/RTX services in its default path. Sample mode reads committed CSV/image evidence from the project and shows how the gateway routes text and vision tasks across local, remote, and reject paths.

## Start

From the repository root:

```bash
streamlit run demo/app.py
```

## Modes

- **Sample mode**: default. Reads existing result files and the sample camera frame. No Jetson, RTX, model files, or TensorRT engine is required.
- **Real backend mode**: optional. The text panel tries to call a running router API at `http://127.0.0.1:8000` or the URL entered in the dashboard. If the backend is unavailable, the UI shows a friendly sample fallback instead of crashing.

## What It Shows

- Text task routing across Jetson local LLM, RTX remote LLM, and reject paths.
- Vision task routing across YOLOv8n TensorRT local CV, remote VLM placeholder/sample rows, and privacy rejects.
- Backend status for the selected local/remote LLM/CV/VLM roles.
- Key result snapshots for quantization, ONNXRuntime session reuse, YOLO TensorRT FP16, and v0.7 reliability.

## Current Limits

- This is a dashboard, not a production serving layer.
- Sample mode is evidence playback, not live inference.
- Real vision backend calls are intentionally not wired in this lightweight UI.
- No login, database, cloud deploy, video stream, model download, or new benchmark is included.
