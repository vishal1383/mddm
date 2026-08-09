#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-outputs/token2token/threshold_lookahead_v7/full_train7473_t090_num099}"
TARGETS_FILE="${TARGETS_FILE:-outputs/token2token/threshold_unlock/gsm8k_train_t095_gain_text_q07_max512.jsonl}"
TRAIN_DIR="$ROOT/train"
GATE_DIR="$ROOT/gate_test_s128_n64"
FULL_DIR="$ROOT/full_test_1319"
mkdir -p "$TRAIN_DIR" "$GATE_DIR" "$FULL_DIR"

if [[ ! -d "$TRAIN_DIR/adapter-final" ]]; then
  PYTHONUNBUFFERED=1 python3 -m Token2Token.train_online_lookahead \
    --targets-file "$TARGETS_FILE" \
    --record-limit 7473 \
    --max-steps 7473 \
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
    --output-dir "$TRAIN_DIR" \
    2>&1 | tee "$TRAIN_DIR/runner.log"
fi

decoder_args=(
  --thresholds .90
  --numeric-threshold .99
  --dataset-split test
  --completion-length 128
  --batch-size 16
  --commit-threshold-on-first-forward
  --no-unlock-forward
  --catalyst-filter text-below
  --catalyst-min-length 0
  --force-catalyst always
  --catalyst-tokens-per-forward 2
  --catalyst-additional-min-confidence .60
  --catalyst-additional-min-ratio .85
  --resume
)

run_eval() {
  local label="$1"
  local adapter="$2"
  local output="$3"
  local start="$4"
  local limit="$5"
  local adapter_args=()
  if [[ -n "$adapter" ]]; then
    adapter_args=(--adapter-path "$adapter" --merge-adapter)
  fi
  mkdir -p "$output"
  PYTHONUNBUFFERED=1 python3 -m Token2Token.eval_threshold_gsm8k \
    --model-label "$label" \
    --output-dir "$output" \
    --start-index "$start" \
    --limit "$limit" \
    "${decoder_args[@]}" \
    "${adapter_args[@]}" \
    2>&1 | tee "$output/runner.log"
}

run_eval base_adaptive_medium "" "$GATE_DIR/base" 128 64

candidate_args=()
for checkpoint in 002000 004000 006000; do
  adapter="$TRAIN_DIR/checkpoint-$checkpoint"
  output="$GATE_DIR/checkpoint-$checkpoint"
  run_eval "threshold_lookahead_$checkpoint" "$adapter" "$output" 128 64
  candidate_args+=(
    --candidate "$checkpoint" "$output/predictions_t0p9.jsonl" "$adapter"
  )
done
run_eval threshold_lookahead_final "$TRAIN_DIR/adapter-final" "$GATE_DIR/final" 128 64
candidate_args+=(
  --candidate final "$GATE_DIR/final/predictions_t0p9.jsonl" "$TRAIN_DIR/adapter-final"
)

python3 -m Token2Token.select_paired_candidate \
  --baseline-predictions "$GATE_DIR/base/predictions_t0p9.jsonl" \
  "${candidate_args[@]}" \
  --expected-threshold .90 \
  --expected-numeric-threshold .99 \
  --min-example-id 128 \
  --max-example-id 192 \
  --minimum-tokens-per-forward 4 \
  --minimum-tokens-per-forward-gain .1 \
  --max-correct-loss 1 \
  --output "$GATE_DIR/selection.json" \
  2>&1 | tee "$GATE_DIR/selection.log"

best_adapter="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_adapter_path"])' "$GATE_DIR/selection.json")"
best_label="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_label"])' "$GATE_DIR/selection.json")"
gate_passed="$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["any_candidate_passed"]).lower())' "$GATE_DIR/selection.json")"

if [[ "$gate_passed" != "true" ]]; then
  touch "$ROOT/GATE_FAILED"
  exit 0
fi

run_eval base_adaptive_medium_full "" "$FULL_DIR/base" 0 1319
run_eval "threshold_lookahead_${best_label}_full" "$best_adapter" "$FULL_DIR/trained" 0 1319

python3 -m Token2Token.paired_comparison \
  --baseline-predictions "$FULL_DIR/base/predictions_t0p9.jsonl" \
  --trained-predictions "$FULL_DIR/trained/predictions_t0p9.jsonl" \
  --baseline-label base_adaptive_medium \
  --trained-label "threshold_lookahead_$best_label" \
  --output "$FULL_DIR/paired_comparison.md"

python3 -m Token2Token.paired_comparison \
  --baseline-predictions "$FULL_DIR/base/predictions_t0p9.jsonl" \
  --trained-predictions "$FULL_DIR/trained/predictions_t0p9.jsonl" \
  --baseline-label base_adaptive_medium \
  --trained-label "threshold_lookahead_$best_label" \
  --min-example-id 192 \
  --max-example-id 1319 \
  --output "$FULL_DIR/paired_comparison_untouched_192_1318.md"

python3 -m Token2Token.micro_trial_gate \
  --baseline-predictions "$FULL_DIR/base/predictions_t0p9.jsonl" \
  --trained-predictions "$FULL_DIR/trained/predictions_t0p9.jsonl" \
  --expected-threshold .90 \
  --expected-numeric-threshold .99 \
  --minimum-tokens-per-forward 4 \
  --minimum-tokens-per-forward-gain .1 \
  --max-correct-loss 26 \
  --output "$FULL_DIR/paired_metrics.json"

python3 -m Token2Token.micro_trial_gate \
  --baseline-predictions "$FULL_DIR/base/predictions_t0p9.jsonl" \
  --trained-predictions "$FULL_DIR/trained/predictions_t0p9.jsonl" \
  --expected-threshold .90 \
  --expected-numeric-threshold .99 \
  --min-example-id 192 \
  --max-example-id 1319 \
  --minimum-tokens-per-forward 4 \
  --minimum-tokens-per-forward-gain .1 \
  --max-correct-loss 23 \
  --output "$FULL_DIR/paired_metrics_untouched_192_1318.json"

touch "$ROOT/COMPLETE"
