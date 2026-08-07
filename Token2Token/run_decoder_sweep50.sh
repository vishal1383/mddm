#!/usr/bin/env bash
# Base-model decoder frontier on the matched 50 GSM8K test examples.
# No training involved: this measures how much of the latency gap is decoder
# scheduling rather than model quality.
set -euo pipefail

cd /workspace/DhruveshProject
export PYTHONUNBUFFERED=1

ROOT="${ROOT:-outputs/token2token/decoder_sweep/base50}"
ADAPTER_ARGS=()
if [[ -n "${ADAPTER_PATH:-}" ]]; then
  ADAPTER_ARGS=(--adapter-path "$ADAPTER_PATH")
fi
LABEL="${LABEL:-base_llada8b}"
LIMIT="${LIMIT:-50}"
BATCH="${BATCH:-8}"

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
    "${ADAPTER_ARGS[@]}" \
    --model-label "${LABEL}_${name}" \
    --completion-length 128 \
    --batch-size "$BATCH" \
    --limit "$LIMIT" \
    --output-dir "$ROOT/$name" \
    "$@" \
    2>&1 | tee "$ROOT/$name.log"
}

# 1. Current two-forward catalyst decoder, uncapped burst.
run catalyst_uncapped --thresholds 0.95

# 2. Same, but the catalyst/cleanup forward also commits its own >=t tokens.
#    Fixes the cleanup path, which otherwise commits one token per forward.
run catalyst_first_commit --thresholds 0.95 --commit-threshold-on-first-forward

# 3. Drop the dedicated unlock forward: one forward per cycle.
run single_forward --thresholds 0.95 \
  --commit-threshold-on-first-forward --no-unlock-forward

# 4/5. Ordinary fixed-k confidence decoding, the baseline never yet measured.
run topk_k2 --decoder topk --tokens-per-step 2 --thresholds 0.95
run topk_k1 --decoder topk --tokens-per-step 1 --thresholds 0.95

echo "== sweep complete =="
python3 -m Token2Token.summarize_decoder_sweep --sweep-dir "$ROOT"
