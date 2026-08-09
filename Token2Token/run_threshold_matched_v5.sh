#!/usr/bin/env bash
# V5: co-design the training target and the decode threshold.
#
# V4a trained positions toward gold probability 0.95 and moved them from 0.735
# to 0.785, so nothing crossed and nothing changed. But 0.95 was a free
# parameter, not a fact. At a decode threshold of 0.85 those same positions are
# already committable, and the sweep shows base LLaDA loses 4 points at 0.90
# and 10 at 0.80 -- losses that are wrong commits in the 0.80-0.95 confidence
# band. That band is precisely what training can repair.
#
# Target: decoding at THRESHOLD as accurate as base at 0.95, at THRESHOLD's
# throughput. Same quality, better throughput.
#
# Train and decode at the same threshold, always.
set -uo pipefail

cd /workspace/DhruveshProject
export PYTHONUNBUFFERED=1

NAME="${NAME:?set NAME}"
THRESHOLD="${THRESHOLD:?set THRESHOLD}"
ROOT="${ROOT:-outputs/token2token/threshold_matched_v5}"
TARGETS_FILE="${TARGETS_FILE:-outputs/token2token/threshold_unlock/gsm8k_train_t095_gain_text_q07_max512.jsonl}"
RECORD_LIMIT="${RECORD_LIMIT:-2000}"
MAX_STEPS="${MAX_STEPS:-2000}"
# Aim slightly above the commit threshold so a promoted position clears it with
# margin rather than sitting exactly on the boundary.
PROMOTE_TARGET="${PROMOTE_TARGET:-$(python3 -c "print(min(0.99, $THRESHOLD + 0.03))")}"
# Only promote positions already within striking distance of the threshold.
# v5_t85 ran with a floor of 0.35 and its committable fraction plateaued near
# 0.14 while drift climbed from 0.028 to 0.044: the band was full of positions
# at 0.4 that cannot be pushed to 0.88 without large distortion, so they
# diluted the gradient and moved the model without ever crossing. A floor a
# quarter below the threshold keeps the positions that can actually convert.
PROMOTE_MIN="${PROMOTE_MIN:-$(python3 -c "print(round(max(0.3, $THRESHOLD - 0.25), 3))")}"
PROMOTE_WEIGHT="${PROMOTE_WEIGHT:-1.0}"
# Repair matters far more at a low threshold than at 0.95. A position base
# commits at 0.85-0.95 confidence with the wrong token is a genuine error, not
# the phrasing disagreement that made repair dangerous at 0.95.
REPAIR_WEIGHT="${REPAIR_WEIGHT:-0.5}"
REPAIR_MAX_GOLD_RANK="${REPAIR_MAX_GOLD_RANK:-5}"
PRESERVE_KL_WEIGHT="${PRESERVE_KL_WEIGHT:-5.0}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
# Three canvases plus the random-denoising one. A second agent shares this
# box and a third 8B process gets OOM-killed; v5_t85 died at step 1001 of
# 2000 that way. Fewer canvases per step is the cheapest way to shrink the
# activation footprint without changing the objective.
CANVASES="${CANVASES:-3}"
# Save often: a killed run should still leave a usable checkpoint.
SAVE_EVERY="${SAVE_EVERY:-400}"
LIMIT="${LIMIT:-50}"
BATCH="${BATCH:-8}"
SINGLE="--commit-threshold-on-first-forward --no-unlock-forward"

TRAIN_DIR="$ROOT/$NAME/train"
EVAL_DIR="$ROOT/$NAME/eval50"
mkdir -p "$TRAIN_DIR" "$EVAL_DIR"

echo "=== $NAME: train and decode at threshold $THRESHOLD (promote target $PROMOTE_TARGET) ==="

if [[ ! -f "$TRAIN_DIR/adapter-final/adapter_config.json" ]]; then
  python3 -m Token2Token.train_parallel_unlock \
    --targets-file "$TARGETS_FILE" \
    --record-limit "$RECORD_LIMIT" \
    --max-steps "$MAX_STEPS" \
    --canvases-per-example "$CANVASES" \
    --commit-threshold "$THRESHOLD" \
    --promote-loss hinge \
    --promote-target-confidence "$PROMOTE_TARGET" \
    --promote-min-confidence "$PROMOTE_MIN" \
    --promote-loss-weight "$PROMOTE_WEIGHT" \
    --repair-loss-weight "$REPAIR_WEIGHT" \
    --repair-max-gold-rank "$REPAIR_MAX_GOLD_RANK" \
    --preserve-kl-weight "$PRESERVE_KL_WEIGHT" \
    --protect-numeric-positions \
    --learning-rate "$LEARNING_RATE" \
    --lora-rank "$LORA_RANK" \
    --lora-alpha "$LORA_ALPHA" \
    --save-every "$SAVE_EVERY" \
    --output-dir "$TRAIN_DIR" \
    2>&1 | tee "$ROOT/$NAME/train.log" || echo "$NAME training failed"
fi

for checkpoint in "$TRAIN_DIR"/checkpoint-* "$TRAIN_DIR/adapter-final"; do
  [[ -d "$checkpoint" ]] || continue
  tag="$(basename "$checkpoint")"
  [[ -f "$EVAL_DIR/$tag/summary.json" ]] && continue
  python3 -m Token2Token.eval_threshold_gsm8k \
    --adapter-path "$checkpoint" \
    --model-label "${NAME}_${tag}" \
    --thresholds "$THRESHOLD" \
    --completion-length 128 \
    --batch-size "$BATCH" \
    --limit "$LIMIT" \
    --output-dir "$EVAL_DIR/$tag" \
    $SINGLE \
    2>&1 | tee "$EVAL_DIR/$tag.log" || echo "$NAME eval $tag failed"
done

python3 -m Token2Token.summarize_decoder_sweep --sweep-dir "$EVAL_DIR" || true
echo "=== $NAME complete ==="
