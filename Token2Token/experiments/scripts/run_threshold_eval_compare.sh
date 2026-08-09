#!/usr/bin/env bash
set -euo pipefail

ADAPTER_PATH="${ADAPTER_PATH:-outputs/token2token/threshold_unlock/llada8b_gsm8k_t095_text_q07_anchor_ltr_1epoch/adapter-final}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/token2token/threshold_unlock/eval_gsm8k_t095_text_q07_anchor_ltr_1epoch}"
THRESHOLD="${THRESHOLD:-0.95}"
COMPLETION_LENGTH="${COMPLETION_LENGTH:-128}"
BATCH_SIZE="${BATCH_SIZE:-8}"

mkdir -p "$OUTPUT_ROOT"

python3 -m Token2Token.main.eval_threshold_gsm8k \
  --model-label base_llada8b \
  --thresholds "$THRESHOLD" \
  --completion-length "$COMPLETION_LENGTH" \
  --batch-size "$BATCH_SIZE" \
  --output-dir "$OUTPUT_ROOT/base" \
  2>&1 | tee "$OUTPUT_ROOT/base.log"

python3 -m Token2Token.main.eval_threshold_gsm8k \
  --adapter-path "$ADAPTER_PATH" \
  --model-label anchor_ltr_lora \
  --thresholds "$THRESHOLD" \
  --completion-length "$COMPLETION_LENGTH" \
  --batch-size "$BATCH_SIZE" \
  --output-dir "$OUTPUT_ROOT/trained" \
  2>&1 | tee "$OUTPUT_ROOT/trained.log"

python3 -m Token2Token.experiments.summarize_threshold_comparison \
  --baseline-summary "$OUTPUT_ROOT/base/summary.json" \
  --trained-summary "$OUTPUT_ROOT/trained/summary.json" \
  --threshold "$THRESHOLD" \
  --output-dir "$OUTPUT_ROOT"
