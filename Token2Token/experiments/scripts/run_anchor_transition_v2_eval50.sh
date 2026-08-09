#!/usr/bin/env bash
set -euo pipefail

ADAPTER_PATH="${ADAPTER_PATH:-outputs/token2token/anchor_transition_v2/full_lr1e5_top2/adapter-final}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/token2token/anchor_transition_v2/eval50_lr1e5_top2}"

mkdir -p "$OUTPUT_ROOT"

python3 -m Token2Token.main.eval_threshold_gsm8k \
  --model-label base_llada8b \
  --thresholds 0.95 \
  --max-threshold-tokens 2 \
  --completion-length 128 \
  --batch-size 8 \
  --limit 50 \
  --output-dir "$OUTPUT_ROOT/base" \
  2>&1 | tee "$OUTPUT_ROOT/base.log"

python3 -m Token2Token.main.eval_threshold_gsm8k \
  --adapter-path "$ADAPTER_PATH" \
  --model-label anchor_transition_v2 \
  --thresholds 0.95 \
  --max-threshold-tokens 2 \
  --completion-length 128 \
  --batch-size 8 \
  --limit 50 \
  --output-dir "$OUTPUT_ROOT/trained" \
  2>&1 | tee "$OUTPUT_ROOT/trained.log"

python3 -m Token2Token.experiments.summarize_threshold_comparison \
  --baseline-summary "$OUTPUT_ROOT/base/summary.json" \
  --trained-summary "$OUTPUT_ROOT/trained/summary.json" \
  --threshold 0.95 \
  --output-dir "$OUTPUT_ROOT"
