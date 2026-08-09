#!/usr/bin/env bash
set -euo pipefail

cd /workspace/DhruveshProject
export PYTHONUNBUFFERED=1

ROOT="${ROOT:-outputs/token2token/adaptive_validation}"
LIMIT="${LIMIT:-500}"
BATCH="${BATCH:-8}"
HELDOUT_START="${HELDOUT_START:-50}"
BASE_DIR="$ROOT/base_k1"
ADAPTIVE_DIR="$ROOT/adaptive_t099"

mkdir -p "$BASE_DIR" "$ADAPTIVE_DIR"

python3 -m Token2Token.main.eval_threshold_gsm8k \
  --model-label base_llada8b_block32_k1 \
  --decoder topk \
  --tokens-per-step 1 \
  --block-length 32 \
  --completion-length 128 \
  --batch-size "$BATCH" \
  --limit "$LIMIT" \
  --resume \
  --output-dir "$BASE_DIR" \
  2>&1 | tee -a "$ROOT/base_k1.log"

python3 -m Token2Token.main.eval_threshold_gsm8k \
  --model-label base_llada8b_adaptive_block32_t099 \
  --thresholds 0.99 \
  --completion-length 128 \
  --batch-size "$BATCH" \
  --limit "$LIMIT" \
  --resume \
  --commit-threshold-on-first-forward \
  --no-unlock-forward \
  --block-length 32 \
  --catalyst-filter any \
  --output-dir "$ADAPTIVE_DIR" \
  2>&1 | tee -a "$ROOT/adaptive_t099.log"

COMMON=(
  --baseline-predictions "$BASE_DIR/predictions_t0p95.jsonl"
  --trained-predictions "$ADAPTIVE_DIR/predictions_t0p99.jsonl"
  --baseline-label standard_k1
  --trained-label adaptive_t099
)

python3 -m Token2Token.main.paired_comparison \
  "${COMMON[@]}" \
  --output "$ROOT/paired_all.md"

python3 -m Token2Token.main.paired_comparison \
  "${COMMON[@]}" \
  --min-example-id "$HELDOUT_START" \
  --output "$ROOT/paired_heldout.md"
