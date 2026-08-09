#!/usr/bin/env bash
# Train on each cached catalyst plus its complete U_after target set, then
# evaluate with the matching adaptive one-forward threshold decoder.
set -euo pipefail

cd /workspace/DhruveshProject
export PYTHONUNBUFFERED=1

ROOT="${ROOT:-outputs/token2token/all_unlocked_v1}"
NAME="${NAME:-t095_full}"
TARGETS_FILE="${TARGETS_FILE:-outputs/token2token/threshold_unlock/gsm8k_train_t095_gain_text_q07_max512.jsonl}"
RECORD_LIMIT="${RECORD_LIMIT:-7473}"
# One of the 7,473 cached records has no catalyst in the 128-token canvas.
# A full one-pass run therefore contains 7,472 optimizer updates.
MAX_STEPS="${MAX_STEPS:-7472}"
STATES_PER_EXAMPLE="${STATES_PER_EXAMPLE:-3}"
TARGET_LOSS_WEIGHT="${TARGET_LOSS_WEIGHT:-1.0}"
SELECTION_LOSS_WEIGHT="${SELECTION_LOSS_WEIGHT:-1.0}"
PRESERVE_KL_WEIGHT="${PRESERVE_KL_WEIGHT:-5.0}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
SAVE_EVERY="${SAVE_EVERY:-500}"
LIMIT="${LIMIT:-1319}"
BATCH="${BATCH:-16}"
EVAL_ALL_CHECKPOINTS="${EVAL_ALL_CHECKPOINTS:-0}"

TRAIN_DIR="$ROOT/$NAME/train"
EVAL_DIR="$ROOT/$NAME/eval1319"
BASE_DIR="$ROOT/base_global_t095_1319"
mkdir -p "$TRAIN_DIR" "$EVAL_DIR" "$BASE_DIR"

if [[ ! -f "$TRAIN_DIR/adapter-final/adapter_config.json" ]]; then
  python3 -m Token2Token.train_all_unlocked \
    --targets-file "$TARGETS_FILE" \
    --record-limit "$RECORD_LIMIT" \
    --max-steps "$MAX_STEPS" \
    --completion-length 128 \
    --states-per-example "$STATES_PER_EXAMPLE" \
    --commit-threshold 0.95 \
    --target-loss-weight "$TARGET_LOSS_WEIGHT" \
    --selection-loss-weight "$SELECTION_LOSS_WEIGHT" \
    --selection-margin 0.1 \
    --preserve-kl-weight "$PRESERVE_KL_WEIGHT" \
    --learning-rate "$LEARNING_RATE" \
    --save-every "$SAVE_EVERY" \
    --output-dir "$TRAIN_DIR" \
    2>&1 | tee "$ROOT/$NAME/train.log"
fi

DECODER_ARGS=(
  --thresholds 0.95
  --commit-threshold-on-first-forward
  --no-unlock-forward
  --catalyst-filter text
  --force-catalyst always
)

if [[ ! -f "$BASE_DIR/summary.json" ]]; then
  python3 -m Token2Token.eval_threshold_gsm8k \
    --model-label base_llada8b_global_adaptive_t095 \
    --completion-length 128 \
    --batch-size "$BATCH" \
    --limit "$LIMIT" \
    --resume \
    --output-dir "$BASE_DIR" \
    "${DECODER_ARGS[@]}" \
    2>&1 | tee "$BASE_DIR/eval.log"
fi

evaluate_adapter() {
  local checkpoint="$1"
  local tag="$2"
  local output="$EVAL_DIR/$tag"
  [[ -f "$output/summary.json" ]] && return
  mkdir -p "$output"
  python3 -m Token2Token.eval_threshold_gsm8k \
    --adapter-path "$checkpoint" \
    --model-label "all_unlocked_v1_${tag}_global_adaptive_t095" \
    --completion-length 128 \
    --batch-size "$BATCH" \
    --limit "$LIMIT" \
    --resume \
    --output-dir "$output" \
    "${DECODER_ARGS[@]}" \
    2>&1 | tee "$output/eval.log"
}

# Produce the requested base-versus-final result before any optional sweep.
evaluate_adapter "$TRAIN_DIR/adapter-final" adapter-final

COMPARISON_DIR="$ROOT/$NAME/comparison"
python3 -m Token2Token.summarize_threshold_comparison \
  --baseline-summary "$BASE_DIR/summary.json" \
  --trained-summary "$EVAL_DIR/adapter-final/summary.json" \
  --threshold 0.95 \
  --output-dir "$COMPARISON_DIR"
python3 -m Token2Token.paired_comparison \
  --baseline-predictions "$BASE_DIR/predictions_t0p95.jsonl" \
  --trained-predictions "$EVAL_DIR/adapter-final/predictions_t0p95.jsonl" \
  --baseline-label base_llada8b_t095 \
  --trained-label all_unlocked_v1_t095 \
  --output "$COMPARISON_DIR/paired_comparison.md"

if [[ "$EVAL_ALL_CHECKPOINTS" == "1" ]]; then
  for checkpoint in "$TRAIN_DIR"/checkpoint-*; do
  [[ -d "$checkpoint" ]] || continue
  tag="$(basename "$checkpoint")"
    evaluate_adapter "$checkpoint" "$tag"
  done
fi

python3 -m Token2Token.summarize_decoder_sweep --sweep-dir "$EVAL_DIR" || true
