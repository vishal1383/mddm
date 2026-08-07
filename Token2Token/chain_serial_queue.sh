#!/usr/bin/env bash
# One sequential queue, deliberately single-job.
#
# A second agent is working this repo and running its own GPU jobs. The box
# fits two 8B processes; a third gets OOM-killed, which is how an earlier base
# control and four of my parallel chains died mid-run. So this runs exactly one
# job at a time and leaves the other slot free.
#
# Priority order reflects the stated goal: a training regime that exploits
# anchors for the same quality at better throughput. The V5 threshold-matched
# sweep comes first; the full benchmark of the decoder result comes after.
set -uo pipefail

cd /workspace/DhruveshProject
export PYTHONUNBUFFERED=1

for pid in "$@"; do
  echo "waiting for pid $pid"
  while kill -0 "$pid" 2>/dev/null; do sleep 30; done
done

ROOT=outputs/token2token/decoder_sweep/base50
SINGLE="--commit-threshold-on-first-forward --no-unlock-forward"

step() {
  local label="$1"
  shift
  echo "=== $label ==="
  "$@" || echo "$label FAILED"
}

# Same-threshold controls. Every trained arm needs base measured at the
# threshold it decodes at, not only at 0.95. The 0.90-0.95 gap is where the
# other agent's V4a experiments are operating and where V5 lands.
if [[ ! -f "$ROOT/single_forward_mid_thresholds/summary.json" ]]; then
  step "base control 0.85/0.75" \
    python3 -m Token2Token.eval_threshold_gsm8k \
      --model-label base_llada8b_single_forward_mid_thresholds \
      --thresholds 0.85,0.75 --completion-length 128 --batch-size 8 --limit 50 \
      --output-dir "$ROOT/single_forward_mid_thresholds" $SINGLE
fi

if [[ ! -f "$ROOT/single_forward_hi_thresholds/summary.json" ]]; then
  step "base control 0.948/0.94/0.92" \
    python3 -m Token2Token.eval_threshold_gsm8k \
      --model-label base_llada8b_single_forward_hi_thresholds \
      --thresholds 0.948,0.94,0.92 --completion-length 128 --batch-size 8 \
      --limit 50 --output-dir "$ROOT/single_forward_hi_thresholds" $SINGLE
fi

# V5 sweep: the axes V4 never varied.
run_v5() {
  local name="$1"
  local threshold="$2"
  shift 2
  [[ -f "outputs/token2token/threshold_matched_v5/$name/eval50/adapter-final/summary.json" ]] && return
  step "$name" env NAME="$name" THRESHOLD="$threshold" MAX_STEPS=1200 \
    RECORD_LIMIT=1200 SAVE_EVERY=600 "$@" \
    bash Token2Token/run_threshold_matched_v5.sh
}

run_v5 v5_t90 0.90
run_v5 v5_t85_aggr 0.85 PRESERVE_KL_WEIGHT=1.0 LEARNING_RATE=1e-4 REPAIR_WEIGHT=1.0
run_v5 v5_t80 0.80
run_v5 v5_t85_norepair 0.85 REPAIR_WEIGHT=0.0
run_v5 v5_t85_rank32 0.85 LORA_RANK=32 LORA_ALPHA=64 LEARNING_RATE=1e-4

# Finish the interrupted round-4 decoder arms.
step "round 4 remainder" bash Token2Token/run_decoder_sweep50_round4.sh

# The full benchmark of the decoder result, which never ran.
step "full benchmark" env BATCH=8 LIMIT=1319 bash Token2Token/run_full_benchmark.sh

echo "=== serial queue complete ==="
