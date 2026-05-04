# Serving Reliability Report

## Scope

v0.7-minimal is a prototype-level reliability benchmark for the vision router. It does not change the text router, does not start a real remote VLM, and does not introduce new models. The goal is to check whether the Jetson-first gateway has explainable behavior under concurrency, local queue pressure, backend unavailability, privacy constraints, and timeout-like failures.

## What Is Covered

- local detect/classify through the v0.6 YOLOv8n TensorRT FP16 backend
- remote scene description / VQA through a mock remote VLM path
- privacy `local_only` semantic vision rejection
- local queue overload fallback or rejection
- local backend unavailable
- remote backend unavailable
- impossible latency budget rejection
- timeout case using a simulated slow remote backend

## How To Run

Run on Jetson from the project root:

```bash
env LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/local/cuda/targets/aarch64-linux/lib \
  python3 serving/scripts/reliability_benchmark.py \
  --mode vision \
  --concurrency 1,2,4,8 \
  --requests 20 \
  --sample-image results/figures/camera_v05_positive_detection.jpg \
  --engine-path /home/rainbow/models/vision/yolo_nano/yolov8n_fp16.engine
```

Outputs:

- `serving/results/raw/reliability_benchmark.csv`
- `serving/results/raw/reliability_failure_modes.csv`
- `serving/results/raw/reliability_summary.csv`

## Results

The benchmark runs 20 requests at each concurrency level: `1`, `2`, `4`, and `8`. The workload mixes local CV, remote mock VLM, privacy rejection, and impossible-latency rejection. Failure modes are recorded separately.

Current summary:

| Concurrency | Requests | Success | Reject | Backend error | Timeout | Fallback | Local | Remote | P50 ms | P95 ms | P99 ms | Pass rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 14 | 6 | 0 | 0 | 0 | 8 | 6 | 1.10 | 42.77 | 269.55 | 1.00 |
| 2 | 20 | 14 | 6 | 0 | 0 | 0 | 8 | 6 | 1.19 | 76.51 | 97.99 | 1.00 |
| 4 | 20 | 14 | 6 | 0 | 0 | 6 | 2 | 12 | 1.07 | 29.68 | 52.29 | 1.00 |
| 8 | 20 | 14 | 6 | 0 | 0 | 6 | 2 | 12 | 1.07 | 28.00 | 47.96 | 1.00 |

Failure-mode summary:

| Scenario | Expected behavior | Result |
|---|---|---|
| normal local detect | local success | pass |
| normal remote scene | remote success | pass |
| privacy local-only VQA | reject | pass |
| local backend unavailable | reject for local-only | pass |
| remote backend unavailable | reject | pass |
| local queue overloaded, allow_remote | remote fallback | pass |
| local queue overloaded, local_only | reject | pass |
| impossible latency budget | reject | pass |
| timeout | timeout/backend error | pass |

The full CSVs are:

- `serving/results/raw/reliability_benchmark.csv`
- `serving/results/raw/reliability_failure_modes.csv`
- `serving/results/raw/reliability_summary.csv`

## Engineering Interpretation

This benchmark is intentionally small. It proves routing reliability behavior rather than production throughput. Queue depth is in-process state inside the benchmark harness, not a distributed queue. At concurrency `4` and `8`, local queue pressure redirects six allow-remote local-CV requests to the remote mock path; privacy-sensitive requests still reject instead of leaving Jetson. The remote backend is a mock because v0.7 focuses on policy behavior; v0.5b already validated the real RTX VLM path.

## Limitations

- in-process queue only
- no distributed queue or Kubernetes/autoscaling
- no Prometheus or production observability stack
- remote VLM in this benchmark is mock
- timeout is simulated rather than caused by the real remote VLM server
- no streaming cancellation or retry budget
