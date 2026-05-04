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

`max_tokens` is the output budget passed to the selected model backend. It is not the same as `latency_budget_ms`: the latency budget helps the router choose local/remote/reject, while `max_tokens` controls how long the generated answer may be. Larger values reduce cutoff risk but can increase latency.

For text llama.cpp backends, the gateway sends `chat_template_kwargs={"enable_thinking": false}` so the demo output budget is spent on the user-visible answer instead of template-level thinking text.

The dashboard also exposes a text `Request timeout seconds` control. This is the UI client's network wait limit, not a routing policy. If it is too small, the UI can fall back to committed sample evidence while the real backend would have completed with more time. For remote text routes, verify the Jetson Gateway, RTX llama-server, and SSH tunnel before classifying a timeout as a model failure.

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

The recommended demo entry point is the lightweight orchestrator from the WSL repo root:

```bash
python3 demo/run_interactive_stack.py
python3 demo/check_interactive_stack.py
```

It starts the RTX local services, SSH reverse tunnels, Jetson local LLM, and Jetson Gateway, then writes logs and PID state to `runtime_logs/interactive_stack/`. It is demo stack orchestration, not production deployment: there is no Docker, systemd, autoscaling, or process supervisor beyond start/check/stop scripts.

Use `--with-ui` to start Streamlit as part of the stack, or run it separately:

```bash
streamlit run demo/app.py
```

The Streamlit page is usually `http://127.0.0.1:8501`. Streamlit itself runs locally, but real backend mode should point to the Jetson Gateway. The dashboard defaults the Router API base URL to `http://192.168.1.102:8000`; override it with `EDGE_GATEWAY_URL=http://custom-host:8000 streamlit run demo/app.py` if the Jetson IP changes.

Stop only the processes launched by the orchestrator:

```bash
python3 demo/stop_interactive_stack.py
```

The manual fallback is to use the v0.9 config directory on Jetson:

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

The Router API base URL field defaults to the Jetson Gateway at `http://192.168.1.102:8000`, while the Streamlit UI itself typically opens at `http://127.0.0.1:8501`. Set `EDGE_GATEWAY_URL` before launch to use a different Jetson address.

Modes:

- Sample mode: reads committed CSV/image evidence. It does not require Jetson, RTX, model files, or TensorRT engines.
- Real backend mode: calls the configured Jetson Gateway. Backend failures are shown as friendly errors or sample fallback in the UI.

Sample mode may only show stored previews because CSV evidence intentionally avoids storing long generations. Real backend mode displays full model responses when the API returns them.

For remote VLM calls, the demo uses a final-answer-only prompt style and appends a `FINAL_ANSWER:` marker before generation:

```text
Return only the final answer. Do not include reasoning, thinking process, analysis steps, constraints, or hidden chain-of-thought.
<user prompt>
FINAL_ANSWER:
```

The server extracts text after `FINAL_ANSWER:` or llama.cpp's explicit final-channel marker when present. It does not perform broad string deletion; this keeps the output budget focused on the user-visible answer without hiding unmarked model behavior.

Vision real mode is single-frame interaction. It is not a video stream and does not attempt continuous camera/VLM processing.

## Smoke Test

Run:

```bash
python3 serving/scripts/smoke_interactive_gateway.py \
  --url http://127.0.0.1:8000 \
  --out serving/results/raw/interactive_gateway_smoke.csv
```

To validate output length behavior:

```bash
python3 serving/scripts/smoke_interactive_output_lengths.py \
  --url http://127.0.0.1:8000 \
  --out serving/results/raw/interactive_output_length_smoke.csv
```

This compares text `128` vs `512` output tokens and vision VLM `128` vs `384` output tokens.

For non-Jetson development, use `--image-source upload` so the script sends the committed sample image as base64. On Jetson, the default `camera` source captures a real CSI frame.

If the remote VLM tunnel is intentionally down, add `--allow-remote-vlm-unavailable` to keep local/reject path validation while recording the remote failure.

## Limits

- This is still a demo client, not a production UI.
- No login, database, cloud deployment, or video stream is included.
- The UI does not bypass the router to call RTX.
- Remote VLM serving remains subprocess/CLI-based unless a separate persistent VLM server is started.
