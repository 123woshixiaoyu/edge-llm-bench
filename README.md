# Jetson-First Edge AI Inference Gateway

**Constraint-aware routing across local LLM/CV and remote LLM/VLM backends.**

This project is a Jetson-first edge AI inference gateway, not a single-model demo. It turns benchmark results into routing policy: given task type, privacy, quality, latency budget, backend health, and queue/device state, the gateway chooses whether work should run locally on Jetson, route to an RTX backend, or be rejected/degraded. Jetson handles low-latency private text and local sensing; RTX handles heavier text reasoning and semantic vision.

## What This Project Is

- **Project 1:** quantized LLM deployment decision study for Jetson and RTX.
- **Project 2:** explainable task router across Jetson local LLM and RTX remote LLM.
- **Project 2 v0.5:** camera-aware router with CSI camera, local CV, and remote VLM.
- **Project 3:** local CV runtime optimization with ONNXRuntime and TensorRT.
- **v0.6/v0.7:** optimized YOLO TensorRT backend integrated into the vision router, then checked with a small reliability benchmark.

Model responsibilities are intentionally separated:

- **Qwen text LLM:** local/remote text backend.
- **MobileNet-SSD:** v0.5 camera/CV system integration baseline.
- **YOLOv8n TensorRT:** optimized local CV backend.
- **Gemma 4 VLM:** remote semantic vision backend.

## Architecture

```mermaid
flowchart TD
    A["User / app request"] --> B["Jetson Gateway"]
    B --> C["Task analyzer"]
    C --> D["Policy engine"]
    D -->|short/private text| E["Local LLM\nQwen3.5 0.8B Q4 on Jetson"]
    D -->|local detect/classify| F["Local CV\nYOLOv8n TensorRT FP16\nMobileNet-SSD baseline/fallback"]
    D -->|complex text| G["Remote LLM\nQwen3.5 4B Q4 on RTX"]
    D -->|semantic vision| H["Remote VLM\nGemma 4 E2B-it Q4 + mmproj on RTX"]
    D -->|privacy conflict / overload / impossible budget| I["Reject / degrade"]
    E --> J["Decision logs + benchmark CSVs"]
    F --> J
    G --> J
    H --> J
    I --> J
```

## Key Results

### 1. LLM Quantization Decision

Jetson Orin Nano 8GB results for the same Qwen3.5 0.8B PTQ model:

| Quant | Decode tok/s | Peak memory MB | Decode tok/s/W | Decision |
|---|---:|---:|---:|---|
| F16 | 24.98 | 3677 | 1.32 | Reference only; too slow/heavy for default |
| Q8_0 | 47.10 | 2953 | 2.39 | Conservative fallback when quality margin matters |
| Q4_K_M | 56.54 | 2703 | 2.80 | Default Jetson local text backend |

Source: [docs/quantization_decision_study.md](docs/quantization_decision_study.md)

### 2. Model / Backend Scorecard Recommendations

| Profile | Recommended backend | Why |
|---|---|---|
| `text_local_default` | Qwen3.5 0.8B `Q4_K_M` on Jetson | Best local text latency, memory, and tok/s/W balance |
| `text_quality_fallback` | Qwen3.5 4B `Q4_K_M` on RTX | Stronger text model for quality-oriented tasks |
| `vision_local_fast_path` | YOLOv8n TensorRT FP16 on Jetson | Fastest measured local CV runtime |
| `vision_remote_semantic_backend` | Gemma 4 E2B-it `Q4_K_M` + `mmproj-F16` on RTX | Real remote VLM for VQA / scene description |

Source: [docs/model_selection_scorecard.md](docs/model_selection_scorecard.md)

### 3. Local CV Optimization

| Runtime | Model | Avg inference ms | P95 ms | Avg total ms | Note |
|---|---|---:|---:|---:|---|
| OpenCV DNN | MobileNet-SSD | 90.71 | 95.87 | 127.83 | v0.5 integration baseline |
| ONNXRuntime old | SSD-MobileNetV1 | 53.23 | 60.92 | 4929.98 | Session recreated per request |
| ONNXRuntime reuse | SSD-MobileNetV1 | 42.02 | 41.67 | 49.04 | Session reuse removes adapter overhead |
| YOLO ONNXRuntime reuse | YOLOv8n | 91.70 | 107.10 | 108.64 | TensorRT comparison baseline |
| YOLO TensorRT FP16 | YOLOv8n | 14.54 | 14.71 | 28.98 | Optimized local CV fast path |

Source: [serving/docs/project3_tensorrt_report.md](serving/docs/project3_tensorrt_report.md)

### 4. Routing / Reliability Demos

| Demo | What it proves | Result |
|---|---|---|
| v0.3 text LLM local/remote | Jetson local 0.8B Q4 + RTX remote 4B Q4 both real | 11/11 expected routes |
| v0.5a camera + local CV | CSI camera + MobileNet-SSD local detection path | 10/10 expected routes |
| v0.5b real remote VLM | Gemma 4 multimodal backend is real, not mock | 10/10 routes, `remote_is_mock=false` |
| v0.6 YOLO TensorRT router | Optimized local CV backend is used by router | 10/10 routes, `fallback_used=false` |
| v0.7 reliability benchmark | Queue/fallback/reject/timeout behavior is explainable | 80 benchmark requests, pass rate 1.0 at concurrency 1/2/4/8; 9/9 failure modes |

Source: [serving/docs/reliability_report.md](serving/docs/reliability_report.md)

## Model / Backend Selection Scorecard

The scorecard makes deployment preferences explicit instead of relying on subjective model choice. It combines hard constraints with profile-specific weights for performance, quality, memory, power, stability, integration status, and size.

Run:

```bash
python3 scripts/score_model_candidates.py \
  --candidates results/raw/model_selection_candidates.csv \
  --profiles configs/model_selection_profiles.yaml \
  --out results/raw/model_selection_scores.csv
```

Artifacts:

- [configs/model_selection_profiles.yaml](configs/model_selection_profiles.yaml)
- [results/raw/model_selection_candidates.csv](results/raw/model_selection_candidates.csv)
- [results/raw/model_selection_scores.csv](results/raw/model_selection_scores.csv)
- [docs/model_selection_scorecard.md](docs/model_selection_scorecard.md)

## Project Evolution

1. **Quantization decision study:** Qwen3.5 0.8B Q4 becomes the Jetson default because Q4 gives similar power draw but much higher useful throughput per watt than F16.
2. **Text router:** local text tasks run on Jetson; code/reasoning/high-quality text can route to RTX.
3. **Camera-aware router:** Jetson captures CSI frames, handles simple local CV, and routes semantic vision to a remote VLM when privacy allows.
4. **Local CV optimization:** MobileNet-SSD proves the integration path; YOLOv8n TensorRT becomes the optimized local CV backend.
5. **Reliability benchmark:** v0.7 checks queue pressure, fallback, reject, backend unavailable, privacy, and timeout behavior.

## How To Reproduce Main Demos

Main entry points:

- LLM quantization decision: [docs/quantization_decision_study.md](docs/quantization_decision_study.md)
- Scorecard: `python3 scripts/score_model_candidates.py ...`
- Text router smoke: `python3 serving/scripts/smoke_dual_real_backends.py --url http://127.0.0.1:8000`
- Vision router smoke: `python3 serving/scripts/smoke_vision_router.py`
- Real remote VLM smoke: `python3 serving/scripts/smoke_vision_router_real_vlm.py --remote-url http://127.0.0.1:18091`
- YOLO TensorRT router smoke: `python3 serving/scripts/smoke_vision_router_yolo_trt.py`
- Reliability benchmark: `python3 serving/scripts/reliability_benchmark.py --mode vision --concurrency 1,2,4,8 --requests 20`

Runtime assets are intentionally outside git:

- `llama.cpp/`
- GGUF models
- ONNX models
- TensorRT engines
- calibration images/caches

## Documentation Map

- [docs/model_selection_scorecard.md](docs/model_selection_scorecard.md)
- [docs/quantization_decision_study.md](docs/quantization_decision_study.md)
- [serving/docs/vision_routing_design.md](serving/docs/vision_routing_design.md)
- [serving/docs/project3_tensorrt_report.md](serving/docs/project3_tensorrt_report.md)
- [serving/docs/reliability_report.md](serving/docs/reliability_report.md)
- [serving/docs/serving_design.md](serving/docs/serving_design.md)
- [serving/docs/routing_policy.md](serving/docs/routing_policy.md)

## Limitations

- Prototype, not production serving.
- v0.7 reliability uses in-process queue state, not a distributed queue.
- Remote VLM latency is still high because the current wrapper uses subprocess CLI serving.
- Model weights and TensorRT engines are not included in git.
- Quality scores are partly manual / heuristic.
- No Kubernetes, autoscaling, or production observability stack.
- No INT8 calibration yet.
- No C++ hot-path adapter yet.

## Next Steps

- TensorRT INT8 calibration for YOLOv8n.
- Python + C++ TensorRT hot-path adapter.
- Remote VLM latency profiling and low-risk optimization.
- Optional lightweight demo dashboard.
