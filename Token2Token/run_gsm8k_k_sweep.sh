#!/usr/bin/env bash
set -euo pipefail

ROOT="outputs/token2token/eval_k_sweep"
ANCHOR_ADAPTER="outputs/token2token/ig_anchor_v1_gsm8k_full_run1/adapter-final"
STANDARD_ADAPTER="outputs/token2token/standard_lora_gsm8k_full_run1/adapter-final"
K_VALUES="${K_VALUES:-5,4,3,2,1}"
BATCH_SIZE="${BATCH_SIZE:-4}"

mkdir -p "$ROOT"

python3 -m Token2Token.eval_gsm8k \
  --model-label anchor_lora \
  --adapter-path "$ANCHOR_ADAPTER" \
  --k-values "$K_VALUES" \
  --batch-size "$BATCH_SIZE" \
  --resume \
  --output-dir "$ROOT/anchor_lora"

python3 -m Token2Token.eval_gsm8k \
  --model-label standard_lora \
  --adapter-path "$STANDARD_ADAPTER" \
  --k-values "$K_VALUES" \
  --batch-size "$BATCH_SIZE" \
  --resume \
  --output-dir "$ROOT/standard_lora"

python3 -m Token2Token.eval_gsm8k \
  --model-label base \
  --k-values "$K_VALUES" \
  --batch-size "$BATCH_SIZE" \
  --resume \
  --output-dir "$ROOT/base"

python3 -m Token2Token.summarize_gsm8k_sweep \
  --root "$ROOT" \
  --k-values "$K_VALUES"
