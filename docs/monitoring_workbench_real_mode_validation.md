# Monitoring Workbench Real-Mode Validation

Date: 2026-05-04

Baseline commit tested: `9c11cbd Productize demo as monitoring workbench`

Validation target: **Jetson Local-First Monitoring Gateway** real mode.

## Environment

| Component | Value |
| --- | --- |
| Jetson | `rainbow@192.168.1.102` |
| Jetson Gateway | `http://192.168.1.102:8000` |
| Jetson local LLM | Qwen3.5 0.8B Q4 via llama.cpp |
| Jetson local CV | YOLOv8n TensorRT FP16 |
| RTX remote LLM | Qwen3.5 4B Q4 via llama.cpp tunnel |
| RTX remote VLM | Gemma 4 E2B-it Q4 + `mmproj-F16` tunnel |
| UI mode | Real backend mode, sample mode disabled |

Startup path:

```bash
python3 demo/run_interactive_stack.py
python3 demo/check_interactive_stack.py
streamlit run demo/app.py
```

The workbench was validated by exercising the same helper paths used by the Streamlit UI against the real Jetson Gateway. Runtime event history was written under `runtime_data/events/`, which is intentionally not committed.

## Workflow Results

Evidence CSV: [monitoring_workbench_real_mode_validation.csv](../serving/results/raw/monitoring_workbench_real_mode_validation.csv)

| Workflow | Expected | Actual | Status | Evidence / Notes |
| --- | --- | --- | --- | --- |
| System Status | Gateway, local backend, remote backend, and real remote VLM ready | Gateway health returned `local_backend_available=true`, `remote_backend_available=true`, `vision_remote_vlm_available=true`, `vision_remote_is_mock=false` | PASS | Health latency `459.46 ms` |
| Live Monitor | Jetson camera local detection, route `local`, YOLO TensorRT, history event | Route `local`; backend `yolov8n_tensorrt_fp16`; local CV backend `yolo_tensorrt_fp16`; labels `["toilet"]` from current camera view | PASS | Local CV `9.7 ms`; total `1277.98 ms`; event `db49b3443641` |
| Event Review remote VLM | Semantic review routes remote, VLM is real, final answer exists, history event | Route `remote`; backend `gemma4_e2b_it_q4_mmproj`; `remote_is_mock=false`; answer length `80` chars | PASS | Local CV `14.4 ms`; remote VLM `20385.82 ms`; total `20412.82 ms`; event `bb2f113ffb88` |
| Event Review privacy-safe reject | `local_only` semantic vision is rejected with privacy explanation and history event | Route `reject`; reasons include privacy/local-only remote VLM block | PASS | Local CV `14.43 ms`; total `26.19 ms`; event `1844c1067292` |
| Monitoring Assistant | Real text backend answers monitoring summary request and records history | Route `local`; backend `qwen35_08b_q4`; answer length `1831` chars | PASS | Total `9834.46 ms`; event `3d2e27aa7c14` |
| Event History | Events persist and route filters work | Recent store contained `all=8`, `local=4`, `remote=2`, `reject=2`; image-backed events were present | PASS | Runtime path `runtime_data/events/events.jsonl` |

## Issues Fixed During Validation

### Stale SSH reverse tunnel detection

Initial stack startup found existing SSH tunnel processes, but Jetson could not actually reach the forwarded `18081` / `18091` endpoints. This made Gateway health show:

- `remote_backend_available=false`
- `vision_remote_vlm_available=false`

Fix:

- `demo/run_interactive_stack.py` now reuses an existing tunnel only if Jetson can reach the forwarded health endpoints.
- New managed tunnels are launched with `ExitOnForwardFailure=yes`.
- `demo/check_interactive_stack.py` now treats a reachable Gateway with unavailable required backends as not fully ready.

After restarting the tunnel, Gateway health showed all required backends ready and the six workbench workflows passed.

## Remaining Limits

- This is still a single-frame workbench validation, not video streaming.
- Remote VLM is real but still slow because the current implementation uses the subprocess CLI path.
- Event history is a lightweight JSONL product loop, not a production database or audit store.
- Runtime images and history are intentionally excluded from git.
- The camera scene can affect detected labels; a no-detection or unexpected label is acceptable as long as local detection succeeds and history records the event.
