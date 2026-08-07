#!/usr/bin/env bash
# V5 sweep: train and decode at a matched threshold, across the space V4
# never touched.
#
# V4 varied the objective while holding the decode threshold at 0.95. That was
# the wrong axis to hold fixed: 0.95 was chosen arbitrarily at the start of the
# project, and its only justification is that base LLaDA happens to peak there.
# A trained model has no reason to peak in the same place. Lowering the
# threshold is free throughput; the accuracy it costs is wrong commits in the
# 0.80-0.95 confidence band, which is exactly what the promote and repair
# buckets are shaped to fix.
#
# Reference points on base, single-forward, one anchor:
#   0.95 -> 72% at 41.4 forwards/example
#   0.90 -> 68% at 35.7
#   0.80 -> 62% at 28.1
# A trained model reaching 72% at threshold 0.85 or below is the win condition.
set -uo pipefail

cd /workspace/DhruveshProject
export PYTHONUNBUFFERED=1

for pid in "$@"; do
  echo "waiting for pid $pid"
  while kill -0 "$pid" 2>/dev/null; do sleep 30; done
done

ROOT=outputs/token2token/decoder_sweep/base50
SINGLE="--commit-threshold-on-first-forward --no-unlock-forward"

# Base reference at the thresholds the trained models will decode at, so every
# trained arm has a same-threshold control rather than only a 0.95 comparison.
if [[ ! -f "$ROOT/single_forward_mid_thresholds/summary.json" ]]; then
  echo "=== base reference at 0.85 and 0.75 ==="
  python3 -m Token2Token.eval_threshold_gsm8k \
    --model-label base_llada8b_single_forward_mid_thresholds \
    --thresholds 0.85,0.75 \
    --completion-length 128 --batch-size 8 --limit 50 \
    --output-dir "$ROOT/single_forward_mid_thresholds" \
    $SINGLE 2>&1 | tee "$ROOT/single_forward_mid_thresholds.log" \
    || echo "base reference failed"
fi

run_config() {
  local name="$1"
  local threshold="$2"
  shift 2
  echo "=== $name (threshold $threshold) ==="
  env NAME="$name" THRESHOLD="$threshold" MAX_STEPS=1200 RECORD_LIMIT=1200 \
    SAVE_EVERY=600 "$@" \
    bash Token2Token/run_threshold_matched_v5.sh \
    || echo "$name failed"
}

# Threshold axis, holding the objective fixed.
run_config v5_t90 0.90
run_config v5_t80 0.80

# Aggressiveness axis at the most promising threshold. V4a's throttles were
# preserve KL 20 and learning rate 1e-5; loosen both.
run_config v5_t85_aggr 0.85 PRESERVE_KL_WEIGHT=1.0 LEARNING_RATE=1e-4 \
  REPAIR_WEIGHT=1.0

# Isolate the two buckets: does repair carry the gain, or promote?
run_config v5_t85_norepair 0.85 REPAIR_WEIGHT=0.0
run_config v5_t85_repaironly 0.85 PROMOTE_WEIGHT=0.0 REPAIR_WEIGHT=1.0

# More capacity, in case rank 8 is the binding constraint rather than the
# learning rate.
run_config v5_t85_rank32 0.85 LORA_RANK=32 LORA_ALPHA=64 LEARNING_RATE=1e-4

echo "=== v5 sweep complete ==="
