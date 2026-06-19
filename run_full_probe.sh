#!/usr/bin/env bash
set -euo pipefail

LIMIT="${LIMIT:-20}"
MAX_K="${MAX_K:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OUT_ROOT="${OUT_ROOT:-outputs/full_probe}"

for model in llada-8b dream-7b; do
  for dataset in gsm8k humaneval; do
    python3 run_probe.py \
      --model "$model" \
      --dataset "$dataset" \
      --split test \
      --limit "$LIMIT" \
      --probe all \
      --max-k "$MAX_K" \
      --batch-size "$BATCH_SIZE" \
      --out-dir "$OUT_ROOT/${model}_${dataset}"
  done
done
