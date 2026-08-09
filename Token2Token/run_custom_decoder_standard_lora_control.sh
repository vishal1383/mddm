#!/usr/bin/env bash
set -euo pipefail

cd /workspace/DhruveshProject
export PYTHONUNBUFFERED=1

OUTPUT_DIR="${OUTPUT_DIR:-outputs/token2token/threshold_lookahead_v7/full_train7473_t090_num099/full_test_1319/standard_lora_merged}"

if [[ -f "$OUTPUT_DIR/summary.json" ]]; then
  echo "already complete: $OUTPUT_DIR"
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
python3 -m Token2Token.eval_threshold_gsm8k \
  --model-label standard_lora_merged_adaptive_full \
  --adapter-path outputs/token2token/standard_lora_gsm8k_full_run1/adapter-final \
  --merge-adapter \
  --output-dir "$OUTPUT_DIR" \
  --dataset-split test \
  --completion-length 128 \
  --batch-size 16 \
  --limit 1319 \
  --resume \
  --thresholds 0.90 \
  --numeric-threshold 0.99 \
  --commit-threshold-on-first-forward \
  --no-unlock-forward \
  --catalyst-filter text-below \
  --catalyst-min-length 0 \
  --force-catalyst always \
  --catalyst-tokens-per-forward 2 \
  --catalyst-additional-min-confidence 0.60 \
  --catalyst-additional-min-ratio 0.85 \
  2>&1 | tee -a "$OUTPUT_DIR/runner.log"
