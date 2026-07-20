#!/usr/bin/env bash
# Launch vLLM OpenAI-compatible server for the base model (Qwen3-8B) on the GB10.
set -euo pipefail
cd /home/sameersegal/Code/SparkyCoder
source .venv/bin/activate

# Thinking-ON eval needs a big generation budget (an 8k cap produces NO_CODE):
# 40960 ctx matches the cloud eval defaults. GB10 note: keep client concurrency
# modest and n=1 per request (the GB10 hangs on high-concurrency multi-sample
# inference — run pass@k on Modal).
exec vllm serve Qwen/Qwen3-8B \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 32 \
  "$@"
