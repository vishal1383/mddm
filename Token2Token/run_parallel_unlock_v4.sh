#!/usr/bin/env bash
# Train one parallel-decodability config and evaluate it against base under an
# identical decoder. Every knob is an environment variable so the iteration
# loop can sweep configs without editing this file.
set -euo pipefail

cd /workspace/DhruveshProject
export PYTHONUNBUFFERED=1

NAME="${NAME:?set NAME}"
ROOT="${ROOT:-outputs/token2token/parallel_unlock_v4}"
TARGETS_FILE="${TARGETS_FILE:-outputs/token2token/threshold_unlock/gsm8k_train_t095_gain_text_q07_max512.jsonl}"
RECORD_LIMIT="${RECORD_LIMIT:-500}"
MAX_STEPS="${MAX_STEPS:-500}"
PROMOTE_WEIGHT="${PROMOTE_WEIGHT:-1.0}"
REPAIR_WEIGHT="${REPAIR_WEIGHT:-0.0}"
REPAIR_MAX_GOLD_RANK="${REPAIR_MAX_GOLD_RANK:-5}"
PRESERVE_KL_WEIGHT="${PRESERVE_KL_WEIGHT:-20.0}"
PROMOTE_MIN_CONFIDENCE="${PROMOTE_MIN_CONFIDENCE:-0.5}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
CANVASES="${CANVASES:-4}"
SAVE_EVERY="${SAVE_EVERY:-250}"
LIMIT="${LIMIT:-50}"
BATCH="${BATCH:-8}"
# Decoder used for evaluation; must match whatever base was measured under.
DECODER_ARGS="${DECODER_ARGS:---thresholds 0.95 --commit-threshold-on-first-forward}"

TRAIN_DIR="$ROOT/$NAME/train"
EVAL_DIR="$ROOT/$NAME/eval50"
mkdir -p "$TRAIN_DIR" "$EVAL_DIR"

if [[ ! -f "$TRAIN_DIR/adapter-final/adapter_config.json" ]]; then
  python3 -m Token2Token.train_parallel_unlock \
    --targets-file "$TARGETS_FILE" \
    --record-limit "$RECORD_LIMIT" \
    --max-steps "$MAX_STEPS" \
    --canvases-per-example "$CANVASES" \
    --commit-threshold 0.95 \
    --promote-min-confidence "$PROMOTE_MIN_CONFIDENCE" \
    --promote-loss-weight "$PROMOTE_WEIGHT" \
    --repair-loss-weight "$REPAIR_WEIGHT" \
    --repair-max-gold-rank "$REPAIR_MAX_GOLD_RANK" \
    --preserve-kl-weight "$PRESERVE_KL_WEIGHT" \
    --learning-rate "$LEARNING_RATE" \
    --lora-rank "$LORA_RANK" \
    --lora-alpha "$LORA_ALPHA" \
    --save-every "$SAVE_EVERY" \
    --output-dir "$TRAIN_DIR" \
    2>&1 | tee "$ROOT/$NAME/train.log"
fi

for checkpoint in "$TRAIN_DIR"/checkpoint-* "$TRAIN_DIR/adapter-final"; do
  [[ -d "$checkpoint" ]] || continue
  tag="$(basename "$checkpoint")"
  if [[ -f "$EVAL_DIR/$tag/summary.json" ]]; then
    echo "== skip eval $tag =="
    continue
  fi
  echo "== eval $NAME/$tag =="
  python3 -m Token2Token.eval_threshold_gsm8k \
    --adapter-path "$checkpoint" \
    --model-label "${NAME}_${tag}" \
    --completion-length 128 \
    --batch-size "$BATCH" \
    --limit "$LIMIT" \
    --output-dir "$EVAL_DIR/$tag" \
    $DECODER_ARGS \
    2>&1 | tee "$EVAL_DIR/$tag.log"
done

python3 -m Token2Token.summarize_decoder_sweep --sweep-dir "$EVAL_DIR"
