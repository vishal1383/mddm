#!/usr/bin/env bash
set -euo pipefail

TARGETS_FILE="${TARGETS_FILE:-outputs/token2token/threshold_unlock/gsm8k_train_t095_gain_text_q07_max512.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/token2token/anchor_transition_v3/train500_kl5}"

mkdir -p "$OUTPUT_DIR"

python3 -m Token2Token.main.train_anchor_transition \
  --targets-file "$TARGETS_FILE" \
  --record-limit 500 \
  --max-steps 500 \
  --transitions-per-example 3 \
  --max-unlock-tokens 2 \
  --standard-loss-weight 1.0 \
  --anchor-loss-weight 0.25 \
  --unlock-loss-weight 0.5 \
  --base-kl-weight 5.0 \
  --learning-rate 5e-6 \
  --lora-rank 4 \
  --lora-alpha 8 \
  --lora-targets v_proj,attn_out \
  --save-every 500 \
  --output-dir "$OUTPUT_DIR"
