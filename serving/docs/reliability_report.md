# Serving Reliability Report

## Engineering Goal

Project 2 v0.4 upgrades the router from a working heterogeneous demo into a more reliable serving prototype. The goal is not production concurrency. The goal is clear behavior when the Jetson local backend is busy, hot, unavailable, or when the RTX remote backend is unavailable.

## What Changed

- The gateway now tracks real in-process local and remote inflight requests.
- `/health` returns `local_queue_depth` and `remote_queue_depth`.
- The routing policy uses local queue depth and Jetson temperature to reduce local routing pressure.
- Privacy-sensitive requests reject instead of leaving Jetson when local execution is unavailable, overloaded, or thermally unsafe.
- Remote-required tasks reject when RTX is unavailable, instead of silently degrading to the smaller Jetson model.
- `/state` and `/state/reset` allow failure-mode smoke tests to simulate backend availability, queue pressure, and temperature.

## Why This Matters

Without queue and failure behavior, a router can look correct in a single-request smoke test while failing badly under real serving conditions. Edge devices are especially sensitive to overload because local memory, thermals, and power headroom are limited. v0.4 makes the policy explain when it routes remote, when it keeps work local, and when it rejects.

## Expected Validation

Run these checks after starting:

- Jetson local `llama-server` on `127.0.0.1:8080`
- RTX remote `llama-server` on WSL `127.0.0.1:8081`
- SSH reverse tunnel from Jetson `127.0.0.1:18081` to RTX/WSL `127.0.0.1:8081`
- Jetson gateway using `serving/configs_dual_llamacpp`

Commands:

```bash
python3 serving/scripts/smoke_dual_real_backends.py --url http://127.0.0.1:8000
python3 serving/scripts/load_test_dual_real_backends.py --url http://127.0.0.1:8000 --concurrency 4 --requests 40
python3 serving/scripts/summarize_serving_load_test.py serving/results/raw/dual_real_backend_load_test.csv
python3 serving/scripts/smoke_failure_modes.py --url http://127.0.0.1:8000
```

Expected output files:

- `serving/results/raw/dual_real_backend_load_test.csv`
- `serving/results/raw/dual_real_backend_load_test_summary.csv`
- `serving/results/raw/failure_modes_smoke.csv`

## Results

v0.4 was validated on the same dual-backend setup as v0.3:

- Jetson local Qwen3.5 0.8B Q4_K_M `llama-server`
- RTX 5090 / WSL remote Qwen3.5 4B Q4_K_M `llama-server`
- SSH reverse tunnel from Jetson `127.0.0.1:18081` to RTX/WSL `127.0.0.1:8081`
- Jetson gateway using `serving/configs_dual_llamacpp`

Regression smoke:

- output: `serving/results/raw/dual_real_backend_smoke_v04_regression.csv`
- requests: 11
- route match: 11/11
- local real success: 4/4
- remote real success: 6/6
- reject: 1/1

Failure-mode smoke:

- output: `serving/results/raw/failure_modes_smoke.csv`
- cases: 6
- route match: 6/6
- unexpected errors: 0
- covered: remote unavailable, local unavailable, local queue overloaded, high temperature, privacy local-only with local unavailable, and privacy local-only with local queue overload

Concurrent load test:

- output: `serving/results/raw/dual_real_backend_load_test.csv`
- summary: `serving/results/raw/dual_real_backend_load_test_summary.csv`
- requests: 40
- concurrency: 4
- success count: 35
- reject count: 5
- backend errors: 0
- timeouts: 0
- route match rate: 1.0
- route distribution: local 18, remote 17, reject 5
- local latency: P50 2613.36 ms, P95 2852.85 ms, P99 2853.24 ms
- remote latency: P50 705.55 ms, P95 1233.71 ms, P99 1497.67 ms
- total latency: P50 1233.71 ms, P95 2793.71 ms, P99 2853.24 ms

## Current Limitations

- Queue depth is per gateway process, not shared across multiple processes.
- Temperature is simulated through `/state`; a real `tegrastats` parser is still future work.
- There is no explicit degrade-to-local policy for remote-required tasks.
- No streaming cancellation or retry budget yet.
- The SSH reverse tunnel is acceptable for the lab setup, but production would use a stable network path or service discovery.
