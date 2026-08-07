#!/usr/bin/env bash
set -euo pipefail

cd /workspace/DhruveshProject
export PYTHONUNBUFFERED=1

ROOT="${ROOT:-outputs/token2token/online_lookahead_v6}"
NAME="${NAME:-k2_train100_step250}"
RECORD_LIMIT="${RECORD_LIMIT:-100}"
MAX_STEPS="${MAX_STEPS:-250}"
SAVE_EVERY="${SAVE_EVERY:-125}"
TARGETS_FILE="${TARGETS_FILE:-outputs/token2token/threshold_unlock/gsm8k_train_t095_gain_text_q07_max512.jsonl}"
TRAIN_DIR="$ROOT/$NAME/train"
EVAL_DIR="$ROOT/$NAME/eval50"
BASE_DIR="$ROOT/base50/block32_k2"

mkdir -p "$ROOT/base50" "$TRAIN_DIR" "$EVAL_DIR"

if [[ ! -f "$TRAIN_DIR/adapter-final/adapter_config.json" ]]; then
  python3 -m Token2Token.train_online_lookahead \
    --targets-file "$TARGETS_FILE" \
    --record-limit "$RECORD_LIMIT" \
    --max-steps "$MAX_STEPS" \
    --completion-length 128 \
    --block-length 32 \
    --lookahead 2 \
    --states-per-example 4 \
    --transition-loss-weight 1.0 \
    --preserve-kl-weight 5.0 \
    --learning-rate 3e-5 \
    --save-every "$SAVE_EVERY" \
    --output-dir "$TRAIN_DIR" \
    2>&1 | tee "$ROOT/$NAME/train.log"
fi

if [[ ! -f "$BASE_DIR/summary.json" ]]; then
  python3 -m Token2Token.eval_threshold_gsm8k \
    --model-label base_block32_k2 \
    --decoder topk --tokens-per-step 2 --block-length 32 \
    --completion-length 128 --batch-size 8 --limit 50 \
    --output-dir "$BASE_DIR" \
    2>&1 | tee "$BASE_DIR.log"
fi

for checkpoint in "$TRAIN_DIR"/checkpoint-*; do
  [[ -d "$checkpoint" ]] || continue
  tag="$(basename "$checkpoint")"
  [[ -f "$EVAL_DIR/$tag/summary.json" ]] && continue
  python3 -m Token2Token.eval_threshold_gsm8k \
    --adapter-path "$checkpoint" \
    --model-label "${NAME}_${tag}_block32_k2" \
    --decoder topk --tokens-per-step 2 --block-length 32 \
    --completion-length 128 --batch-size 8 --limit 50 \
    --output-dir "$EVAL_DIR/$tag" \
    2>&1 | tee "$EVAL_DIR/$tag.log"
done

python3 -m Token2Token.summarize_decoder_sweep --sweep-dir "$EVAL_DIR" || true
