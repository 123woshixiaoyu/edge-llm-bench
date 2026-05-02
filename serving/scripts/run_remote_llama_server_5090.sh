#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/d/AI/edge-llm-bench}"
MODEL_PATH="${MODEL_PATH:-/mnt/d/AI/Models/quantized/qwen35_4b/qwen35_4b-Q4_K_M.gguf}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8081}"

cd "$PROJECT_ROOT/llama.cpp"

exec ./build/bin/llama-server \
  -m "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  -ngl 99 \
  -c 4096
