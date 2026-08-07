#!/usr/bin/env bash
# Round 4: is a better anchor worth more than a cheaper one?
#
# Round 2 established that anchor choice, not anchor harvesting, is what makes
# the decoder fast: forcing a content word unlocks 2.089 positions per forward,
# forcing the globally most confident token unlocks 0.850. The alphabetic
# filter still admits "the", "is", "a" -- high-confidence, low-information
# words that fail for the same reason punctuation does. Requiring a longer
# token is the cheapest available proxy for information content.
set -euo pipefail

cd /workspace/DhruveshProject
export PYTHONUNBUFFERED=1

ROOT="${ROOT:-outputs/token2token/decoder_sweep/base50}"
LIMIT="${LIMIT:-50}"
BATCH="${BATCH:-8}"
SINGLE="--commit-threshold-on-first-forward --no-unlock-forward"

mkdir -p "$ROOT"

run() {
  local name="$1"
  shift
  if [[ -f "$ROOT/$name/summary.json" ]]; then
    echo "== skip $name (already done) =="
    return
  fi
  echo "== $name =="
  python3 -m Token2Token.eval_threshold_gsm8k \
    --model-label "base_llada8b_${name}" \
    --thresholds 0.95 \
    --completion-length 128 \
    --batch-size "$BATCH" \
    --limit "$LIMIT" \
    --output-dir "$ROOT/$name" \
    "$@" \
    2>&1 | tee "$ROOT/$name.log"
}

run single_forward_len3 $SINGLE --catalyst-min-length 3
run single_forward_len5 $SINGLE --catalyst-min-length 5

echo "== round 4 complete =="
python3 -m Token2Token.summarize_decoder_sweep --sweep-dir "$ROOT"
