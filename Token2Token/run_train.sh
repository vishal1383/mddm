#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-gsm8k}"
MAX_STEPS="${MAX_STEPS:-100}"
IG_BATCH_SIZE="${IG_BATCH_SIZE:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/token2token/llada8b_${DATASET}}"

python3 -m Token2Token.train \
  --dataset "$DATASET" \
  --max-steps "$MAX_STEPS" \
  --anchors 5 \
  --ig-batch-size "$IG_BATCH_SIZE" \
  --target-right-fraction 0.75 \
  --output-dir "$OUTPUT_DIR"
