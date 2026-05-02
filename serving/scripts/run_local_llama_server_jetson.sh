#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/rainbow/edge-llm-bench}"
MODEL_PATH="${MODEL_PATH:-/home/rainbow/models/quantized/qwen35_08b/qwen35_08b-Q4_K_M.gguf}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"

cd "$PROJECT_ROOT/llama.cpp"

exec ./build/bin/llama-server \
  -m "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  -ngl 99 \
  -c 4096
