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
# Two thresholds in one process, so the model loads once. On 50 examples the
# threshold moved throughput a lot (2.54 -> 3.09 -> 3.59 tokens/forward for
# 0.99/0.95/0.90) but accuracy only between 34 and 36 correct, which 50
# examples cannot resolve. Deciding the operating point needs the full set.
# 0.95 must stay in the list: the existing full two-forward run is at 0.95, and
# that pairing is the headline comparison.
THRESHOLD="${THRESHOLD:-0.95,0.90}"
SINGLE="--commit-threshold-on-first-forward --no-unlock-forward"

mkdir -p "$ROOT"

BLOCK_LIMIT="${BLOCK_LIMIT:-500}"

evaluate() {
  local name="$1"
  local limit="$2"
  local thresholds="$3"
  shift 3
  echo "== $name (limit $limit, thresholds $thresholds) =="
  python3 -m Token2Token.main.eval_threshold_gsm8k \
    --model-label "$name" \
    --thresholds "$thresholds" \
    --completion-length 128 \
    --batch-size "$BATCH" \
    --limit "$limit" \
    --resume \
    --output-dir "$ROOT/$name" \
    "$@" \
    2>&1 | tee -a "$ROOT/$name.log"
}

evaluate base_single_forward "$LIMIT" "$THRESHOLD" $SINGLE

# Semi-autoregressive block decoding is how LLaDA is normally generated, so it
# is the baseline a decoding claim has to clear. k=3 spends 42.7
# forwards/example against single-forward's 41.4, which makes it a latency
# match rather than a quality-versus-speed trade. Run on a subset: 500 paired
# examples resolve a difference of about 4 pp, and the arm costs as much per
# example as the main one.
# Block decoding ignores the threshold, so run it at one value only;
# passing the list would decode the same thing twice.
evaluate base_block32_k3 "$BLOCK_LIMIT" 0.95 \
  --decoder topk --tokens-per-step 3 --block-length 32

python3 -m Token2Token.main.paired_comparison \
  --baseline-predictions "$ROOT/base_block32_k3/predictions_t0p95.jsonl" \
  --trained-predictions "$ROOT/base_single_forward/predictions_t0p95.jsonl" \
  --baseline-label block32_k3 --trained-label single_forward \
  --output "$ROOT/paired_single_vs_block32_k3.md"

# The two-forward arm already exists at full scale under an identical config
# (threshold 0.95, 128 tokens, uncapped burst, no adapter), so pair against it
# rather than spending another 1319 decodes.
TWO_FORWARD="outputs/token2token/threshold_unlock/eval_gsm8k_t095_text_q07_anchor_ltr_1epoch/base/predictions_t0p95.jsonl"
if [[ -f "$TWO_FORWARD" ]]; then
  python3 -m Token2Token.main.paired_comparison \
    --baseline-predictions "$TWO_FORWARD" \
    --trained-predictions "$ROOT/base_single_forward/predictions_t0p95.jsonl" \
    --baseline-label two_forward --trained-label single_forward \
    --output "$ROOT/paired_single_vs_two_full.md"
fi

if [[ -n "${ADAPTER_PATH:-}" ]]; then
  evaluate trained_single_forward "$LIMIT" "$THRESHOLD" $SINGLE \
    --adapter-path "$ADAPTER_PATH"
  python3 -m Token2Token.main.paired_comparison \
    --baseline-predictions "$ROOT/base_single_forward/predictions_t0p95.jsonl" \
    --trained-predictions "$ROOT/trained_single_forward/predictions_t0p95.jsonl" \
    --baseline-label base --trained-label trained \
    --output "$ROOT/paired_trained_vs_base.md"
fi

python3 -m Token2Token.experiments.summarize_decoder_sweep --sweep-dir "$ROOT"
