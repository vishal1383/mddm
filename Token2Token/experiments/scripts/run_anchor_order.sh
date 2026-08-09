#!/usr/bin/env bash
set -euo pipefail

TARGETS_FILE="${TARGETS_FILE:-outputs/token2token/anchor_targets/gsm8k_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/token2token/anchor_order_lora_gsm8k}"
MAX_STEPS="${MAX_STEPS:-7473}"

python3 -m Token2Token.experiments.train_anchor_order \
  --targets-file "$TARGETS_FILE" \
  --max-steps "$MAX_STEPS" \
  --anchors 5 \
  --output-dir "$OUTPUT_DIR"
