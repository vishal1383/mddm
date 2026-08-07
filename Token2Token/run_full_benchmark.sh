#!/usr/bin/env bash
# Full GSM8K benchmark of the decoder change, and of a trained adapter on top
# of it if ADAPTER_PATH is set. Every arm resumes, so this can be interrupted
# and restarted without losing completed examples.
#
# The two-forward base arm is not re-run here: it already exists at
# outputs/token2token/threshold_unlock/eval_gsm8k_t095_text_q07_anchor_ltr_1epoch/base
# with 951/1319 = 72.10% at 57.35 forwards/example.
set -euo pipefail

cd /workspace/DhruveshProject
export PYTHONUNBUFFERED=1

ROOT="${ROOT:-outputs/token2token/full_benchmark}"
LIMIT="${LIMIT:-1319}"
BATCH="${BATCH:-8}"
THRESHOLD="${THRESHOLD:-0.95}"
SINGLE="--commit-threshold-on-first-forward --no-unlock-forward"

mkdir -p "$ROOT"

evaluate() {
  local name="$1"
  shift
  echo "== $name =="
  python3 -m Token2Token.eval_threshold_gsm8k \
    --model-label "$name" \
    --thresholds "$THRESHOLD" \
    --completion-length 128 \
    --batch-size "$BATCH" \
    --limit "$LIMIT" \
    --resume \
    --output-dir "$ROOT/$name" \
    "$@" \
    2>&1 | tee -a "$ROOT/$name.log"
}

evaluate base_single_forward $SINGLE

if [[ -n "${ADAPTER_PATH:-}" ]]; then
  evaluate trained_single_forward $SINGLE --adapter-path "$ADAPTER_PATH"
  python3 -m Token2Token.paired_comparison \
    --baseline-predictions "$ROOT/base_single_forward/predictions_t0p95.jsonl" \
    --trained-predictions "$ROOT/trained_single_forward/predictions_t0p95.jsonl" \
    --baseline-label base --trained-label trained \
    --output "$ROOT/paired_trained_vs_base.md"
fi

python3 -m Token2Token.summarize_decoder_sweep --sweep-dir "$ROOT"
