# YOLO-World Trigger Detector Benchmark

## Goal

The Monitoring Workbench currently uses YOLOv8n TensorRT FP16 as the local monitoring fast path. That path is fast on Jetson, but it has a fixed class set. YOLO-World is an open-vocabulary detector, so it may be useful for user-defined triggers such as `lipstick`, `helmet`, `package`, or `tool`.

This benchmark evaluates YOLO-World as a possible **custom-object trigger detector**. It does not connect YOLO-World to the router or UI, and it does not replace the current YOLOv8n TensorRT backend.

## Setup

Runtime:

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

Jetson smoke was not run in this pass. SSH BatchMode access failed with:

```text
Permission denied (publickey,password)
```

I did not install Ultralytics or modify the Jetson runtime environment. This keeps the existing Gateway / YOLO TensorRT stack untouched.

## Engineering Interpretation

1. **Can YOLO-World detect common classes?**  
   Yes on RTX/WSL. It detected `bed` and `bowl` using the base prompt set.

2. **Can it detect custom classes?**  
   Not proven from the current public samples. There were no `lipstick`, `package`, `tool`, `phone`, or `key` detections. A proper custom-object test needs public sample images that actually contain those objects.

3. **Can it handle relation triggers such as `object inside bowl`?**  
   It produced a hit for `object inside bowl`, but this should be treated as a weak detector cue, not reliable relation reasoning. Relationship triggers still need ROI/change logic or a VLM verifier.

4. **Can it replace YOLOv8n TensorRT as the default local detector?**  
   Not yet. The current YOLOv8n TensorRT path is proven on Jetson at roughly `10-30 ms` inference and is already integrated. YOLO-World needs Jetson latency and TensorRT validation before it can be considered as a replacement.

5. **Is it useful as a custom-object trigger backend?**  
   Potentially yes. The RTX result shows very low latency and successful open-vocabulary operation. Its best role is likely a lower-frequency custom-object trigger path, not the every-frame default.

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

1. Jetson smoke with `yolov8s-worldv2.pt`.
2. Custom-object image test with non-private public examples containing lipstick / tool / package.
3. Jetson ONNXRuntime or TensorRT export benchmark.
4. False-positive check for relation prompts.

