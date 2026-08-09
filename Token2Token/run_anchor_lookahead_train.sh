#!/usr/bin/env bash
set -euo pipefail

TARGETS_FILE="${TARGETS_FILE:-outputs/token2token/anchor_lookahead/cache/gsm8k_train_t095_gain_text_q07_max512.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/token2token/anchor_lookahead/train_step6000}"
RECORD_LIMIT="${RECORD_LIMIT:-7473}"
MAX_STEPS="${MAX_STEPS:-6000}"

if [[ ! -f "$TARGETS_FILE" ]]; then
  echo "Missing cache: $TARGETS_FILE" >&2
  echo "Run Token2Token/run_anchor_lookahead_cache.sh first." >&2
  exit 1
fi
if [[ -e "$OUTPUT_DIR/train.jsonl" ]]; then
  echo "Refusing to overwrite an existing run: $OUTPUT_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

PYTHONUNBUFFERED=1 python3 -m Token2Token.train_online_lookahead \
  --targets-file "$TARGETS_FILE" \
  --record-limit "$RECORD_LIMIT" \
  --max-steps "$MAX_STEPS" \
  --completion-length 128 \
  --block-length 128 \
  --lookahead 2 \
  --teacher-policy threshold-catalyst \
  --confidence-threshold .90 \
  --numeric-threshold .99 \
  --states-per-example 4 \
  --max-unlock-tokens 2 \
  --transition-loss-weight 1 \
  --selection-loss-weight 1 \
  --selection-margin .1 \
  --preserve-kl-weight 5 \
  --learning-rate 3e-5 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout .05 \
  --lora-targets q_proj,k_proj,v_proj,attn_out \
  --save-every 1000 \
  --bf16 \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$OUTPUT_DIR/runner.log"
