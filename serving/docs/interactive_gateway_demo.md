# v0.9 Interactive Gateway Demo

v0.9 turns the reviewer dashboard from evidence playback into an optional interactive client for the real Jetson Gateway.

The main rule is that real requests go to the Jetson Gateway first. The UI does not call RTX services directly and does not duplicate routing policy.

## What Real Mode Exercises

| Request type | Expected route | Backend |
| --- | --- | --- |
| short `qa` / `summary` text | local | Jetson local LLM, Qwen3.5 0.8B Q4 |
| `code` / `reasoning` / high-quality text | remote | RTX remote LLM, Qwen3.5 4B Q4 |
| `detect` / `classify` vision | local | Jetson camera + YOLOv8n TensorRT FP16 |
| `scene_description` / `vqa` vision | remote | Jetson camera + RTX remote VLM |
| `local_only` semantic vision | reject | privacy-preserving reject |

## API

The new endpoint is:

```text
POST /v1/vision/analyze
```

Request fields:

- `task_type`: `detect`, `classify`, `scene_description`, or `vqa`
- `privacy`: `allow_remote` or `local_only`
- `quality`: `low`, `medium`, or `high`
- `latency_budget_ms`
- `prompt`, optional
- `image_source`: `camera`, `upload`, or `sample`
- `image_base64`, required for `upload`
- `use_yolo_trt`, default true
- `max_tokens`

Response fields include:

- route decision and reasons
- selected backend/model
- capture metadata
- local CV labels/boxes
- remote VLM text, when routed remote
- capture, local CV, remote, and total latency
- `remote_is_mock`
- error/status fields

The UI intentionally separates local CV precheck evidence from the final routed answer:

- YOLO boxes and labels are produced by Jetson local CV.
- For local `detect` / `classify`, local CV is the final answer.
- For remote `scene_description` / `vqa`, local CV remains pre-analysis/logging, and the final answer comes from the RTX remote VLM.
- For reject routes, the final answer source is the routing policy.

## Runtime Configuration

Use the v0.9 config directory on Jetson:

```bash
EDGE_ROUTER_CONFIG_DIR=serving/configs_interactive_demo \
uvicorn serving.app.main:app --host 127.0.0.1 --port 8000
```

The config expects:

- Jetson local llama-server on `127.0.0.1:8080`
- RTX remote llama-server through tunnel `127.0.0.1:18081`
- RTX remote VLM through tunnel `127.0.0.1:18091`
- YOLO TensorRT engine at `/home/rainbow/models/vision/yolo_nano/yolov8n_fp16.engine`

Remote VLM is real but slow. A semantic vision request can take about `15-20 s`, depending on prompt/image size and whether the remote CLI path has to load the model.

## Dashboard

Start the dashboard from the repo root:

```bash
streamlit run demo/app.py
```

Modes:

- Sample mode: reads committed CSV/image evidence. It does not require Jetson, RTX, model files, or TensorRT engines.
- Real backend mode: calls the configured Jetson Gateway. Backend failures are shown as friendly errors or sample fallback in the UI.

Sample mode may only show stored previews because CSV evidence intentionally avoids storing long generations. Real backend mode displays full model responses when the API returns them.

Vision real mode is single-frame interaction. It is not a video stream and does not attempt continuous camera/VLM processing.

## Smoke Test

Run:

```bash
python3 serving/scripts/smoke_interactive_gateway.py \
  --url http://127.0.0.1:8000 \
  --out serving/results/raw/interactive_gateway_smoke.csv
```

For non-Jetson development, use `--image-source upload` so the script sends the committed sample image as base64. On Jetson, the default `camera` source captures a real CSI frame.

If the remote VLM tunnel is intentionally down, add `--allow-remote-vlm-unavailable` to keep local/reject path validation while recording the remote failure.

## Limits

- This is still a demo client, not a production UI.
- No login, database, cloud deployment, or video stream is included.
- The UI does not bypass the router to call RTX.
- Remote VLM serving remains subprocess/CLI-based unless a separate persistent VLM server is started.
