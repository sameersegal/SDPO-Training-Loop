#!/usr/bin/env bash
cd /home/sameersegal/Code/SparkyCoder
source .venv/bin/activate
set -a; source .env; set +a
vllm serve google/gemma-4-E2B-it --enable-lora --lora-modules sdpo=adapter100 \
  --max-lora-rank 32 --port 8000 --dtype bfloat16 --max-model-len 4096 \
  --gpu-memory-utilization 0.85 --max-num-seqs 64 > vllm_adapter_gsm8k.log 2>&1 &
SRV=$!
for i in $(seq 1 90); do curl -s -m 2 http://localhost:8000/v1/models 2>/dev/null | grep -q '"sdpo"' && break; sleep 10; done
python eval_runner.py --dataset gsm8k --sample-frac 1.0 --model sdpo --out results_gsm8k_sdpo100.json
kill $SRV 2>/dev/null
echo "ADAPTER_GSM8K_DONE"
