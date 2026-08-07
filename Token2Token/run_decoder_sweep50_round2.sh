#!/usr/bin/env bash
# Round 2 of the base decoder frontier. Round 1 showed the second forward is
# not earning its cost, so this explores around single-forward decoding:
# the threshold trade-off curve, whether the alphabetic catalyst rule still
# contributes, and whether committing several catalysts at once helps.
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
    --completion-length 128 \
    --batch-size "$BATCH" \
    --limit "$LIMIT" \
    --output-dir "$ROOT/$name" \
    "$@" \
    2>&1 | tee "$ROOT/$name.log"
}

# Threshold trade-off curve for single-forward decoding. One process, one
# model load, three thresholds. 0.95 is already measured in round 1.
run single_forward_thresholds --thresholds 0.99,0.90,0.80 $SINGLE

# Control: same decoder with no alphabetic catalyst restriction. If this
# matches single_forward, the text-anchor rule is contributing nothing and
# the decoder is ordinary confidence-plus-threshold decoding.
run single_forward_any --thresholds 0.95 $SINGLE --catalyst-filter any

# Commit two catalysts per forward instead of one.
run single_forward_cat2 --thresholds 0.95 $SINGLE \
  --catalyst-tokens-per-forward 2

# Skip the forced commit whenever the threshold already selects something.
# The forced token exists only to guarantee progress; when it is not needed it
# commits the most confident position the model still judged not confident
# enough, which is the least reliable commit in the cycle.
run single_forward_noforce --thresholds 0.95 $SINGLE \
  --force-catalyst when-empty

# Latency-matched control. single_forward spends 41.4 forwards/example; plain
# top-k with k=3 spends 128/3 = 42.7. If top-k matches its accuracy at the
# same budget, the threshold rule is not what is buying the speedup.
run topk_k3 --decoder topk --tokens-per-step 3 --thresholds 0.95

echo "== round 2 complete =="
python3 -m Token2Token.summarize_decoder_sweep --sweep-dir "$ROOT"
