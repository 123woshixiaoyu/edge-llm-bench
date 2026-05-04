#!/usr/bin/env bash
set -euo pipefail

ONNX_MODEL="${1:-/home/rainbow/models/vision/yolo_nano/yolov8n.onnx}"
ENGINE_PATH="${2:-/home/rainbow/models/vision/yolo_nano/yolov8n_fp16.engine}"

if command -v trtexec >/dev/null 2>&1; then
  TRTEXEC="$(command -v trtexec)"
elif [ -x /usr/src/tensorrt/bin/trtexec ]; then
  TRTEXEC="/usr/src/tensorrt/bin/trtexec"
else
  echo "trtexec not found. Install TensorRT packages on Jetson before building the FP16 engine." >&2
  echo "Candidate checked during Phase 1: tensorrt 10.3.0.30-1+cuda12.5 from the Jetson r36.4 apt repo." >&2
  exit 2
fi

mkdir -p "$(dirname "$ENGINE_PATH")"

export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/nvidia:/usr/local/cuda/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}"

"$TRTEXEC" \
  --onnx="$ONNX_MODEL" \
  --saveEngine="$ENGINE_PATH" \
  --fp16 \
  --memPoolSize=workspace:1024 \
  --skipInference
