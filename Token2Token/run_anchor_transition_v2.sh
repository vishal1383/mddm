#!/usr/bin/env bash
set -euo pipefail

TARGETS_FILE="${TARGETS_FILE:-outputs/token2token/threshold_unlock/gsm8k_train_t095_gain_text_q07_max512.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/token2token/anchor_transition_v2/full_lr1e5_top2}"
MAX_STEPS="${MAX_STEPS:-7473}"
RECORD_LIMIT="${RECORD_LIMIT:-7473}"

mkdir -p "$OUTPUT_DIR"

python3 -m Token2Token.train_anchor_transition \
  --targets-file "$TARGETS_FILE" \
  --record-limit "$RECORD_LIMIT" \
  --max-steps "$MAX_STEPS" \
  --transitions-per-example 3 \
  --max-unlock-tokens 2 \
  --standard-loss-weight 1.0 \
  --anchor-loss-weight 0.5 \
  --unlock-loss-weight 1.0 \
  --learning-rate 1e-5 \
  --save-every 500 \
  --output-dir "$OUTPUT_DIR"
