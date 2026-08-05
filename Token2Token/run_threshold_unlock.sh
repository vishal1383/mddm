#!/usr/bin/env bash
set -euo pipefail

TARGETS_FILE="${TARGETS_FILE:-outputs/token2token/threshold_unlock/gsm8k_train_t095_gain.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/token2token/threshold_unlock/llada8b_gsm8k_t095}"
MAX_STEPS="${MAX_STEPS:-7473}"
STAGES_PER_EXAMPLE="${STAGES_PER_EXAMPLE:-8}"

mkdir -p "$OUTPUT_DIR"

python3 -m Token2Token.train_threshold_unlock \
  --targets-file "$TARGETS_FILE" \
  --max-steps "$MAX_STEPS" \
  --stages-per-example "$STAGES_PER_EXAMPLE" \
  --output-dir "$OUTPUT_DIR"
