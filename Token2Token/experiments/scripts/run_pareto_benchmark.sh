#!/usr/bin/env bash
# Accuracy/latency Pareto frontier for base and a trained adapter under one
# decoder. Fine-tuning moves confidence calibration, so a single fixed
# threshold can flatter either model; the honest comparison is whether the
# trained frontier sits above the base frontier across thresholds.
set -euo pipefail

cd /workspace/DhruveshProject
export PYTHONUNBUFFERED=1

ADAPTER_PATH="${ADAPTER_PATH:?set ADAPTER_PATH}"
NAME="${NAME:-trained}"
ROOT="${ROOT:-outputs/token2token/pareto}"
THRESHOLDS="${THRESHOLDS:-0.99,0.95,0.90,0.80}"
LIMIT="${LIMIT:-500}"
BATCH="${BATCH:-16}"
DECODER_ARGS="${DECODER_ARGS:---commit-threshold-on-first-forward}"

mkdir -p "$ROOT"

evaluate() {
  local name="$1"
  shift
  echo "== $name =="
  python3 -m Token2Token.main.eval_threshold_gsm8k \
    --model-label "$name" \
    --thresholds "$THRESHOLDS" \
    --completion-length 128 \
    --batch-size "$BATCH" \
    --limit "$LIMIT" \
    --resume \
    --output-dir "$ROOT/$name" \
    $DECODER_ARGS \
    "$@" \
    2>&1 | tee -a "$ROOT/$name.log"
}

evaluate base
evaluate "$NAME" --adapter-path "$ADAPTER_PATH"

python3 -m Token2Token.experiments.summarize_decoder_sweep --sweep-dir "$ROOT"
