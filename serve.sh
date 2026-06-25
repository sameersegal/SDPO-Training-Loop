#!/usr/bin/env bash
# Launch vLLM OpenAI-compatible server for gemma-4-E2B-it on the GB10.
set -euo pipefail
cd /home/sameersegal/Code/SparkyCoder
source .venv/bin/activate

# Text-only eval: cap context to keep KV cache small; GB10 has unified memory.
exec vllm serve google/gemma-4-E2B-it \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 64 \
  "$@"
