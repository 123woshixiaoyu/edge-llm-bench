# YOLO-World Trigger Detector Benchmark

## Goal

The Monitoring Workbench currently uses YOLOv8n TensorRT FP16 as the local monitoring fast path. That path is fast on Jetson, but it has a fixed class set. YOLO-World is an open-vocabulary detector, so it may be useful for user-defined triggers such as `lipstick`, `helmet`, `package`, or `tool`.

This benchmark evaluates YOLO-World as a possible **custom-object trigger detector**. It does not connect YOLO-World to the router or UI, and it does not replace the current YOLOv8n TensorRT backend.

## Setup

RTX runtime:

- Platform tested: RTX 5090 Laptop / WSL
- Evaluation environment: existing isolated `.venv-vlm`
- Package: `ultralytics 8.4.46`
- Model: `yolov8s-worldv2.pt`
- Mode: offline vocabulary via `model.set_classes(...)`
- Input size: `640`
- Images:
  - `results/figures/camera_v05_positive_detection.jpg`
  - `results/figures/camera_v05_real_vlm_sample.jpg`

Command:

```bash
source .venv-vlm/bin/activate
python serving/scripts/benchmark_yolo_world.py \
  --models yolo_worldv2s \
  --platform-label rtx_5090_wsl \
  --class-set all \
  --imgsz 640 \
  --device 0 \
  --warmup 1 \
  --out serving/results/raw/yolo_world_benchmark.csv \
  --summary-out serving/results/raw/yolo_world_summary.csv
```

Artifacts:

- `serving/results/raw/yolo_world_benchmark.csv`
- `serving/results/raw/yolo_world_summary.csv`

No `.pt`, `.onnx`, `.engine`, cache, or runtime logs are committed.

Jetson follow-up:

- Platform checked: Jetson Orin Nano 8GB
- Project path: `/home/rainbow/edge-llm-bench`
- Model evaluated: `yolov8s-worldv2.pt`
- Result status: `dependency_blocked`

The Jetson check synced only the benchmark script, committed public sample images, and the already-downloaded YOLO-World small weight into Jetson runtime storage. It did not modify the Gateway, YOLOv8n TensorRT engine, router policy, or active services.

## Class Sets

| Set | Prompts |
| --- | --- |
| `base_coco_like` | `person`, `chair`, `bowl`, `bottle`, `bed` |
| `custom_objects` | `lipstick`, `package`, `tool`, `phone`, `key` |
| `relation_proxy` | `object inside bowl`, `item in bowl`, `small object`, `container with object` |

## RTX Results

| Class set | Runs | Avg total latency | Avg inference | Detected labels | Notes |
| --- | ---: | ---: | ---: | --- | --- |
| `base_coco_like` | 2 | 13.59 ms | 5.99 ms | `bed`, `bowl` | Detects common prompts on the committed samples. |
| `custom_objects` | 2 | 13.77 ms | 5.80 ms | none | No custom-object hits on the current public samples; this is not a full custom-object failure because the sample images may not contain those objects. |
| `relation_proxy` | 2 | 16.59 ms | 7.83 ms | `object inside bowl` | Relation-style phrases can fire, but should be validated carefully because YOLO-World is still a detector, not a spatial-reasoning engine. |

The measured RTX latency is very low after warmup. The first cold run is intentionally excluded from the summary because it includes model/vocabulary setup and dependency initialization.

## Export Feasibility

ONNX export was tested for `yolov8s-worldv2.pt`:

```text
ONNX export success
Output: runtime_data/yolo_world_weights/yolov8s-worldv2.onnx
Size: 49.01 MB
```

The ONNX file is under `runtime_data/` and is not committed.

TensorRT export was **not** attempted in this pass. The main reason is scope control: TensorRT engine generation should be tested on Jetson with the target TensorRT version, and `.engine` artifacts must remain outside git.

## Jetson Smoke

Manual SSH to the Jetson was confirmed and the Jetson-only smoke was attempted. The runtime did not have the required Python dependencies:

```text
python 3.10.12 /usr/bin/python3
torch_error ModuleNotFoundError No module named 'torch'
ultralytics_error ModuleNotFoundError No module named 'ultralytics'
cv2 4.13.0
```

Creating an isolated Jetson virtual environment with `python3 -m venv .venv-yolo-world` also failed because the Jetson image does not currently include `ensurepip` / `python3.10-venv`. I did not install `python3.10-venv` with apt because this phase is benchmark-only and should not alter the working Gateway/TensorRT environment.

I checked a user-level pip dry run for `ultralytics`. It would pull a large new stack including `torch 2.11.0`, `torchvision 0.26.0`, CUDA 13 toolkit packages, cuDNN, cuBLAS, cuFFT, cuSOLVER, cuSPARSE, NCCL, NVSHMEM, and Triton. That is too invasive for a Jetson smoke whose explicit goal is not to disturb the existing YOLOv8n TensorRT and Gateway setup.

The Jetson rows were therefore generated as explicit `dependency_blocked` records in:

- `serving/results/raw/yolo_world_benchmark.csv`
- `serving/results/raw/yolo_world_summary.csv`

This is an environment blocker, not evidence that YOLO-World cannot run on Jetson.

## Engineering Interpretation

1. **Can YOLO-World detect common classes?**  
   Yes on RTX/WSL. It detected `bed` and `bowl` using the base prompt set.

2. **Can it detect custom classes?**  
   Not proven from the current public samples. There were no `lipstick`, `package`, `tool`, `phone`, or `key` detections. A proper custom-object test needs public sample images that actually contain those objects.

3. **Can it handle relation triggers such as `object inside bowl`?**  
   It produced a hit for `object inside bowl`, but this should be treated as a weak detector cue, not reliable relation reasoning. Relationship triggers still need ROI/change logic or a VLM verifier.

4. **Can it replace YOLOv8n TensorRT as the default local detector?**  
   No. The current YOLOv8n TensorRT path is proven on Jetson at roughly `10-30 ms` inference and is already integrated. YOLO-World is blocked on Jetson dependency/runtime setup in this pass and still needs Jetson latency plus TensorRT validation before it can be considered as a replacement.

5. **Is it useful as a custom-object trigger backend?**  
   Potentially yes, but currently as an RTX-proven candidate only. The RTX result shows very low latency and successful open-vocabulary operation. Its best role is likely a lower-frequency custom-object trigger path, not the every-frame Jetson default.

6. **Can it reduce VLM calls?**  
   Yes, if validated on a target object set. YOLO-World can provide boxes/confidence for custom prompts, so it is a better trigger detector than asking a VLM every frame. VLM should remain the semantic verifier for event review and relation-heavy questions.

## Recommended Architecture

Do not replace the current local monitoring stack yet. The recommended future direction is:

```text
YOLOv8n TensorRT
  = default continuous fixed-class local detector

YOLO-World
  = optional lower-frequency custom-object trigger detector

SmolVLM2 final-line verifier
  = optional fast semantic yes/no verifier for candidate events

Gemma remote VLM
  = quality fallback for richer scene review
```

Before integration, run:

1. Prepare a controlled Jetson YOLO-World runtime, preferably a separate container or vetted Jetson-compatible PyTorch/Ultralytics environment.
2. Re-run Jetson smoke with `yolov8s-worldv2.pt`.
3. Test custom-object images with non-private public examples containing lipstick / tool / package.
4. Run Jetson ONNXRuntime or TensorRT export benchmark from the already-successful ONNX export path.
5. Check false positives for relation prompts.
