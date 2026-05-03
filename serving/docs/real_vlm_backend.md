# Real Remote VLM Backend

## Purpose

v0.5b turns the vision router's remote branch from a mock placeholder into a real RTX/WSL visual-language backend. The goal is to validate the heterogeneous loop:

```text
Jetson camera -> local CV -> routing policy -> RTX VLM -> decision CSV
```

This is still an MVP. It is designed for single-image smoke tests, not high-throughput serving.

## Model Choice

The selected model is the locally available Gemma 4 E2B-it multimodal GGUF pair:

```text
/mnt/d/AI/Models/gemma4/E2B-it/gemma-4-E2B-it-Q4_K_M.gguf
/mnt/d/AI/Models/gemma4/E2B-it/mmproj-F16.gguf
```

The backend runs through:

```text
/mnt/d/AI/edge-llm-bench/llama.cpp/build/bin/llama-mtmd-cli
```

This choice was made because:

- the model was already present locally, so v0.5b did not need a new VLM download;
- it ran successfully on the RTX 5090 with the existing llama.cpp build;
- the GGUF + mmproj path matches the project's existing llama.cpp deployment style;
- it is simple enough to wrap with a small HTTP server for the MVP.

Qwen3-VL 4B/8B remains a reasonable future option, but it would require a new Hugging Face download and a transformers runtime setup. That is outside the v0.5b scope.

## Server

The remote server is `serving/app/remote_vlm_server.py`. It exposes:

- `GET /health`
- `POST /v1/vision/completions`

Start it on RTX/WSL:

```bash
cd /mnt/d/AI/edge-llm-bench
PORT=8091 .venv/bin/python serving/scripts/run_remote_vlm_server_5090.py
```

Default model paths can be overridden with environment variables:

- `VLM_MODEL_PATH`
- `VLM_MMPROJ_PATH`
- `VLM_CLI_PATH`
- `VLM_MODEL_NAME`
- `VLM_CTX_SIZE`
- `VLM_GPU_LAYERS`

## Image Transfer

The Jetson sends image bytes as base64 in the HTTP request body. It does not send a Jetson image path as the remote input.

Reason: Jetson paths such as `/home/rainbow/edge-llm-bench/results/figures/...` are not readable from WSL. Base64 transfer keeps the remote backend stateless with respect to Jetson storage and makes the privacy boundary explicit.

## Network Path

The current validation used an SSH reverse tunnel:

```text
Jetson 127.0.0.1:18091 -> RTX/WSL 127.0.0.1:8091
```

The Jetson smoke script calls:

```text
http://127.0.0.1:18091/v1/vision/completions
```

This mirrors the v0.3 text-router workaround: WSL localhost services were reliable from Windows, but not directly exposed on the laptop LAN IP.

## Smoke Test

Run on Jetson:

```bash
python3 serving/scripts/smoke_vision_router_real_vlm.py \
  --remote-url http://127.0.0.1:18091 \
  --out serving/results/raw/vision_router_real_vlm_smoke.csv \
  --baseline-out serving/results/raw/local_cv_real_vlm_baseline.csv \
  --sample-image results/figures/camera_v05_real_vlm_sample.jpg
```

Current result:

- cases: `10`
- expected routes: `10/10`
- route distribution: local `4`, remote `4`, reject `2`
- remote model: `gemma4_e2b_it_q4_mmproj`
- all remote rows: `remote_is_mock=false`
- remote response preview: non-empty real VLM text
- remote latency range: about `19.2-20.0 s`

## Limitations

- The server launches `llama-mtmd-cli` per request, so latency includes process start and model load.
- The current prompt can produce reasoning-style prose before the final image description.
- Only single-image requests are supported.
- There is no streaming, batching, TensorRT, video pipeline, or production concurrency control.
- Privacy policy is enforced by routing: `privacy=local_only` semantic vision tasks are rejected instead of being sent remote.
