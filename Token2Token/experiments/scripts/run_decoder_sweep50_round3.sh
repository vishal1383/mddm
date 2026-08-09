#!/usr/bin/env bash
# Round 3: extend the threshold curve downward.
#
# Round 2 showed 0.99 losing on both axes against 0.95, and 0.90 running ahead
# of 0.95 on both. That is not a speed/quality trade being tuned; it says the
# threshold was set too high, so too much of the completion came from the
# forced commit rather than from threshold commits. Push down until accuracy
# actually breaks, so the reported operating point is chosen from a curve
# rather than inherited from the original 0.95.
set -euo pipefail

cd /workspace/DhruveshProject
export PYTHONUNBUFFERED=1

ROOT="${ROOT:-outputs/token2token/decoder_sweep/base50}"
LIMIT="${LIMIT:-50}"
BATCH="${BATCH:-8}"
THRESHOLDS="${THRESHOLDS:-0.70,0.60,0.50}"
SINGLE="--commit-threshold-on-first-forward --no-unlock-forward"

mkdir -p "$ROOT"

if [[ -f "$ROOT/single_forward_low_thresholds/summary.json" ]]; then
  echo "== already done =="
else
  python3 -m Token2Token.main.eval_threshold_gsm8k \
    --model-label base_llada8b_single_forward_low_thresholds \
    --thresholds "$THRESHOLDS" \
    --completion-length 128 \
    --batch-size "$BATCH" \
    --limit "$LIMIT" \
    --output-dir "$ROOT/single_forward_low_thresholds" \
    $SINGLE \
    2>&1 | tee "$ROOT/single_forward_low_thresholds.log"
fi

python3 -m Token2Token.experiments.summarize_decoder_sweep --sweep-dir "$ROOT"
