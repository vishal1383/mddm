#!/usr/bin/env bash
set -euo pipefail

ADAPTER_PATH="${ADAPTER_PATH:-outputs/token2token/anchor_lookahead/train_step6000/checkpoint-006000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/token2token/anchor_lookahead/full_test_1319}"
LIMIT="${LIMIT:-1319}"
BATCH_SIZE="${BATCH_SIZE:-16}"

if [[ ! -d "$ADAPTER_PATH" ]]; then
  echo "Missing adapter: $ADAPTER_PATH" >&2
  exit 1
fi

decoder_args=(
  --thresholds .90
  --numeric-threshold .99
  --dataset-split test
  --completion-length 128
  --batch-size "$BATCH_SIZE"
  --commit-threshold-on-first-forward
  --no-unlock-forward
  --catalyst-filter text-below
  --catalyst-min-length 0
  --force-catalyst always
  --catalyst-tokens-per-forward 2
  --catalyst-additional-min-confidence .60
  --catalyst-additional-min-ratio .85
  --limit "$LIMIT"
  --resume
)

run_eval() {
  local label="$1"
  local output="$2"
  shift 2
  mkdir -p "$output"
  PYTHONUNBUFFERED=1 python3 -m Token2Token.main.eval_threshold_gsm8k \
    --model-label "$label" \
    --output-dir "$output" \
    "${decoder_args[@]}" \
    "$@" \
    2>&1 | tee "$output/runner.log"
}

run_eval frozen_base_anchor_lookahead_decoder "$OUTPUT_ROOT/base"
run_eval anchor_lookahead_lora_step6000 "$OUTPUT_ROOT/trained" \
  --adapter-path "$ADAPTER_PATH" \
  --merge-adapter

python3 -m Token2Token.main.paired_comparison \
  --baseline-predictions "$OUTPUT_ROOT/base/predictions_t0p9.jsonl" \
  --trained-predictions "$OUTPUT_ROOT/trained/predictions_t0p9.jsonl" \
  --baseline-label frozen_base \
  --trained-label anchor_lookahead_step6000 \
  --output "$OUTPUT_ROOT/paired_comparison.md"
