#!/usr/bin/env bash
set -euo pipefail

TARGETS_FILE="${TARGETS_FILE:-outputs/token2token/threshold_unlock/gsm8k_train_t095_gain_text_q07_max512.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/token2token/threshold_unlock/llada8b_gsm8k_t095_text_q07_anchor_only}"
MAX_STEPS="${MAX_STEPS:-7473}"
STAGES_PER_EXAMPLE="${STAGES_PER_EXAMPLE:-8}"
STANDARD_LOSS_WEIGHT="${STANDARD_LOSS_WEIGHT:-0.0}"

mkdir -p "$OUTPUT_DIR"

python3 -m Token2Token.main.train_threshold_unlock \
  --targets-file "$TARGETS_FILE" \
  --max-steps "$MAX_STEPS" \
  --stages-per-example "$STAGES_PER_EXAMPLE" \
  --standard-loss-weight "$STANDARD_LOSS_WEIGHT" \
  --output-dir "$OUTPUT_DIR"
