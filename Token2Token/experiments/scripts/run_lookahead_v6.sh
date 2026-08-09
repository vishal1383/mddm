#!/usr/bin/env bash
# Distil two frozen block-k1 teacher steps into one student forward, then gate
# the result against base block-k1 and base block-k2 on the same 50 examples.
set -euo pipefail

cd /workspace/DhruveshProject
export PYTHONUNBUFFERED=1

ROOT="${ROOT:-outputs/token2token/lookahead_v6}"
NAME="${NAME:-k2_train100}"
TRAIN_EXAMPLES="${TRAIN_EXAMPLES:-100}"
MAX_STEPS="${MAX_STEPS:-500}"
LOOKAHEAD="${LOOKAHEAD:-2}"
SAVE_EVERY="${SAVE_EVERY:-250}"
ROLLOUTS="$ROOT/base_block32_k1_train${TRAIN_EXAMPLES}.jsonl"
TRAIN_DIR="$ROOT/$NAME/train"
EVAL_DIR="$ROOT/$NAME/eval50"

mkdir -p "$ROOT" "$ROOT/base50" "$TRAIN_DIR" "$EVAL_DIR"

resume=()
[[ -f "$ROLLOUTS" ]] && resume=(--resume)
rows=0
[[ -f "$ROLLOUTS" ]] && rows="$(grep -cve '^$' "$ROLLOUTS")"
if (( rows < TRAIN_EXAMPLES )); then
  python3 -m Token2Token.main.precompute_teacher_rollouts \
    --examples "$TRAIN_EXAMPLES" \
    --completion-length 128 \
    --block-length 32 \
    --batch-size 8 \
    --output "$ROLLOUTS" \
    "${resume[@]}" \
    2>&1 | tee "$ROOT/precompute_train${TRAIN_EXAMPLES}.log"
fi

if [[ ! -f "$TRAIN_DIR/adapter-final/adapter_config.json" ]]; then
  python3 -m Token2Token.main.train_lookahead_distillation \
    --targets-file "$ROLLOUTS" \
    --record-limit "$TRAIN_EXAMPLES" \
    --max-steps "$MAX_STEPS" \
    --lookahead "$LOOKAHEAD" \
    --states-per-example 4 \
    --transition-loss-weight 1.0 \
    --preserve-kl-weight 5.0 \
    --learning-rate 3e-5 \
    --lora-rank 8 \
    --lora-alpha 16 \
    --save-every "$SAVE_EVERY" \
    --output-dir "$TRAIN_DIR" \
    2>&1 | tee "$ROOT/$NAME/train.log"
fi

BASE_DIR="$ROOT/base50/block32_k${LOOKAHEAD}"
if [[ ! -f "$BASE_DIR/summary.json" ]]; then
  python3 -m Token2Token.main.eval_threshold_gsm8k \
    --model-label "base_block32_k${LOOKAHEAD}" \
    --decoder topk \
    --tokens-per-step "$LOOKAHEAD" \
    --block-length 32 \
    --completion-length 128 \
    --batch-size 8 \
    --limit 50 \
    --output-dir "$BASE_DIR" \
    2>&1 | tee "$BASE_DIR.log"
fi

for checkpoint in "$TRAIN_DIR"/checkpoint-*; do
  [[ -d "$checkpoint" ]] || continue
  tag="$(basename "$checkpoint")"
  [[ -f "$EVAL_DIR/$tag/summary.json" ]] && continue
  python3 -m Token2Token.main.eval_threshold_gsm8k \
    --adapter-path "$checkpoint" \
    --model-label "${NAME}_${tag}_block32_k${LOOKAHEAD}" \
    --decoder topk \
    --tokens-per-step "$LOOKAHEAD" \
    --block-length 32 \
    --completion-length 128 \
    --batch-size 8 \
    --limit 50 \
    --output-dir "$EVAL_DIR/$tag" \
    2>&1 | tee "$EVAL_DIR/$tag.log"
done

python3 -m Token2Token.experiments.summarize_decoder_sweep --sweep-dir "$EVAL_DIR" || true
