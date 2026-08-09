#!/usr/bin/env bash
# Second queue, run after chain_remaining_work.sh finishes.
#
# 1. Mechanism experiment. The claim that the post-anchor unlock forward
#    commits dependent tokens prematurely rests on two anecdotes. Re-decode
#    with --record-commit-phase so every position carries the rule that placed
#    it, then contrast the commit mix of right and wrong answers.
# 2. V4b. V4a's promote hinge flattened with its positions near gold
#    probability 0.785, short of the 0.95 needed to change any commit, so
#    loosen the two throttles: preserve KL 20 -> 5 and learning rate
#    1e-5 -> 3e-5, over more steps. Numeric protection and the hinge stay.
set -uo pipefail

cd /workspace/DhruveshProject
export PYTHONUNBUFFERED=1

for pid in "$@"; do
  echo "waiting for pid $pid"
  while kill -0 "$pid" 2>/dev/null; do sleep 30; done
  echo "pid $pid finished"
done

MECHANISM="outputs/token2token/mechanism/two_forward_phase200"
echo "=== mechanism: two-forward decode with commit-phase recording ==="
mkdir -p "$MECHANISM"
python3 -m Token2Token.main.eval_threshold_gsm8k \
  --model-label two_forward_phase \
  --thresholds 0.95 \
  --completion-length 128 \
  --batch-size 8 \
  --limit 200 \
  --resume \
  --record-commit-phase \
  --output-dir "$MECHANISM" \
  2>&1 | tee -a "$MECHANISM.log" || echo "mechanism run failed"

python3 -m Token2Token.experiments.commit_phase_analysis \
  --predictions "$MECHANISM/predictions_t0p95.jsonl" \
  --output "$MECHANISM/commit_phase_report.md" || echo "phase analysis failed"

echo "=== v4b: looser preservation, higher learning rate ==="
NAME=v4b \
RECORD_LIMIT=2000 \
MAX_STEPS=2000 \
PRESERVE_KL_WEIGHT=5.0 \
LEARNING_RATE=3e-5 \
SAVE_EVERY=500 \
  bash Token2Token/experiments/scripts/run_parallel_unlock_v4.sh || echo "v4b failed"

echo "=== followup chain complete ==="
