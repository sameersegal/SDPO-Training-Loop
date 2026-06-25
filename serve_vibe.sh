#!/usr/bin/env bash
# Launch vLLM OpenAI-compatible server for VibeThinker-3B (Qwen2.5-3B reasoning).
set -euo pipefail
cd /home/sameersegal/Code/SparkyCoder
source .venv/bin/activate

# Reasoning model: needs a large generation budget. Cap context at 24k for GSM8K
# (model supports 64k) to bound the KV cache on the GB10.
exec vllm serve WeiboAI/VibeThinker-3B \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 24576 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 64 \
  "$@"
